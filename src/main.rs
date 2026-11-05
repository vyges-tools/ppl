// SPDX-License-Identifier: Apache-2.0
//! `vyges-ppl` CLI — IO pin placement over a `.odb`.
//!
//! Today it exposes the first half of the engine: `slots`, which reports every legal pin position
//! on the die boundary. That is deliberately shippable on its own — assignment is a choice *among*
//! slots, so a slot list that disagrees with upstream's makes every later comparison meaningless,
//! and one that agrees makes the rest checkable a stage at a time.
//!
//! Exit status: 0 slots found, 1 the design offers none, 2 usage/read error.

use std::process::ExitCode;
use vyges_opendb::Db;

/// A diagnostic as a structured event (`PPL-ERROR`, `vyges-events/1.0`) instead of a bare stderr line;
/// text after a blank line (the usage) stays plain help. The `VYGPPL` probe lines stay plain.
fn diag_line(msg: String) {
    let (head, help) = match msg.split_once("\n\n") {
        Some((h, rest)) => (h.to_string(), Some(rest.to_string())),
        None => (msg, None),
    };
    let text = head.strip_prefix("vyges-ppl: ").unwrap_or(&head).to_string();
    vyges_events::emit(&vyges_events::Event::new("vyges-ppl", vyges_events::Severity::Error, text).with_code("PPL-ERROR"));
    if let Some(h) = help {
        eprintln!("{h}");
    }
}

macro_rules! diag {
    ($($t:tt)*) => { diag_line(format!($($t)*)) };
}

/// The run's message lines (`[INFO PPL-nnnn] text`), written through the suite logger: each line
/// on stderr as it is logged, and an event under its code.
struct RunLog(vyges_events::logger::Logger);

impl RunLog {
    fn new() -> Self {
        RunLog(vyges_events::logger::Logger::new().events("vyges-ppl"))
    }

    fn info(&mut self, (id, text): (u16, String)) {
        self.0.info("PPL", id, text);
        self.flush();
    }

    fn warn(&mut self, id: u16, text: String) {
        self.0.warn("PPL", id, text);
        self.flush();
    }

    /// An uncoded line the command's Tcl prints (`utl::report`).
    fn report(&mut self, text: String) {
        self.0.report(text);
        self.flush();
    }

    fn flush(&mut self) {
        for l in self.0.take_lines() {
            eprintln!("{l}");
        }
    }
}

use vyges_ppl::{annealing, matching, Boundary, Edge, Group, Params, Pin, Slot, SLOTS_PER_SECTION};

// ⚠️ The prefix here is the CLI GROUP this engine belongs to, and vyges-cli's MODULES
// registry is what actually decides it (`group: "physical"`). It read `vyges loom ppl`
// for a release after the construction engines were split out of the loom suite, because
// nothing ties this string to that registry -- `vyges loom ppl` is now REFUSED by the CLI,
// so the help was telling users a command that no longer runs. If the group ever moves,
// this string moves with it. Running the binary directly as `vyges-ppl` always works and
// is group-independent.
const USAGE: &str = "\
vyges physical ppl — IO pin placement: pins on the die boundary, where the wiring is cheapest

USAGE:
  vyges physical ppl slots      <design.odb> --hor-layers L[,L…] --ver-layers L[,L…] [options]
  vyges physical ppl place-pins <design.odb> --hor-layers L[,L…] --ver-layers L[,L…] [options]
  vyges physical ppl --describe
  vyges physical ppl --help

OPTIONS:
  --hor-layers L,…       layers carrying pins on the LEFT and RIGHT edges (else set_place_config's)
  --ver-layers L,…       layers carrying pins on the BOTTOM and TOP edges (else set_place_config's)
  --min-distance D       minimum spacing between pins, in MICRONS
                         (omitted: candidates every 2 tracks — not 'no spacing')
  --min-distance-in-tracks   read --min-distance as a count of candidate slots instead
  --corner-avoidance D   keep pins this far from each corner, in MICRONS
                         (omitted: 2 tracks, capped at 1um)
  --hor-multiplier M     widen pins on the left/right edges by this factor
  --ver-multiplier M     widen pins on the bottom/top edges by this factor
  --hor-length L         `set_pin_length -hor_length`, microns: the depth of pins on the
                         left/right edges, which also deepens what keeps them off nearby shapes
  --ver-length L         the same for pins on the bottom/top edges
  --hor-extension L      `set_pin_length_extension -hor_extension`, microns: how far pins on the
                         left/right edges reach OUT past the die (written pins only)
  --ver-extension L      the same for pins on the bottom/top edges
  --out-odb FILE         write the placed pins into a copy of the database: each port's pin
                         replaced by one rectangle on its layer, PLACED
  --slots-per-section N  slots per matching section (default 200)
  --annealing            place by simulated annealing instead of optimal matching
  --temperature T        annealing start temperature (default 1.0)
  --max-iterations N     annealing temperature steps (default 2000)
  --perturb-per-iter N   perturbations per step (default: scaled to the pin count)
  --alpha A              annealing cooling rate (default 0.985)
  --random-seed N        annealing seed (default 42)
  --evaluate FILE        also score a reference placement under the same cost model.
                         FILE is JSON mapping each pin name to an [x, y] pair in DBU.
                         Reports reference_hpwl, so a placement that merely DIFFERS
                         can be told from one that is WORSE.
  -o FILE                write the report to FILE instead of stdout
  --json                 emit JSON (the default)
  --describe             print a machine-readable JSON description of the command

EXIT STATUS:
  0  ok        slots were generated / every pin was placed
  1  refused   no legal pin position on the layers given, or not enough room for the pins
  2  error     usage error, unreadable database, or no DBU scale
";


/// The pin, inherited from the crate every engine already depends on.
const CRATE_PIN: &str = vyges_opendb::OPENROAD_PIN;

/// The pin this binary was built against, injected into the descriptor at print time.
///
/// 🔑 **One definition for the whole programme, inherited rather than typed.** The SHA lives in
/// `openroad-pin.yaml` in `vyges-opendb-lib` and reaches here through `vyges-opendb`, which this
/// engine already depends on. Before this, every engine spelled the pin out in its own
/// `--describe` prose, and four of them were still quoting the previous one a day after it moved.
///
/// ⚠️ **It reports what this BINARY was built against — not that the binary is current.** A stale
/// build reports its stale pin quite happily. That is the point: a harness compares this against
/// the oracle image it is about to launch and refuses on a mismatch, which is the check that was
/// missing when two engines ran a whole gate against the previous pin's oracle.
const PIN_TOKEN: &str = "@OPENROAD_PIN@";

fn describe() -> String {
    DESCRIBE.replace(PIN_TOKEN, CRATE_PIN)
}

// ⚠️ When a limitation below stops being true, REPLACE IT -- never append the new truth beside
// it. Two entries here outlived their behaviour: one said polygon dies were not handled and one
// said annealing was "DEFERRED, not built", both while the accurate entries describing the
// shipped polygon path and the bit-exact annealing stream sat higher in the same array. A reader
// cannot tell which of two contradicting claims is current, and this array is PUBLIC -- it is
// what `--describe` emits and what an agent reads to decide what this engine can do. A
// contradiction here is worse than silence: it makes the honest entries unreliable too.
//
// NOTE FOR EDITORS: everything between the r#" and "# is JSON, not Rust. A `//` line in there is
// literal text and breaks the parse -- which is exactly what happened while writing this comment.
const DESCRIBE: &str = r#"{
  "schema": "vyges-tool-descriptor/1.1",
  "openroad_pin": "@OPENROAD_PIN@",
  "name": "ppl",
  "summary": "IO pin placement: pins on the die boundary, positioned to minimise the wire needed to reach them",
  "maturity": "structured",
  "provenance_limitations": [
      "input_hash covers the argument vector, not the content of the .odb it names.",
      "SCOPE: this build implements slot generation, EXCLUDED REGIONS (`exclude_io_pin_region`), REGION CONSTRAINTS (`set_io_pin_constraint -region edge:lo-hi`, by pin name or by direction), PIN GROUPS (`-group`/`-order`, including fallback placement for groups too large for a section), MIRRORED PIN PAIRS (`-mirrored_pins`), ports already fixed by `place_pin`, and the deterministic assignment of the remaining pins -- sections plus optimal (Hungarian) matching within each section. TOP-LAYER placement (`define_pin_shape_pattern` + `-region up:`), and the deterministic assignment of the rest. POLYGON (rectilinear) dies, SIMULATED ANNEALING for plain pins, and the deterministic assignment of the rest.",
      "ANNEALING (`--annealing`) reproduces the reference EXACTLY, including its random stream: the reference draws from Boost, whose algorithms are specified and portable, so the engine, both distributions and the shuffle are reimplemented bit-for-bit. Verified by comparing all 86000 perturbations of a run against the reference own debug trace -- every cost and delta identical. Scope: the whole reference annealer -- region constraints (unsorted, as an annealing run reads them), pin groups with all four group moves, and mirrored pairs; verified on every one of upstream 31 annealing cases, perturbation by perturbation, against an instrumented reference. A pin sent to the top layer is the one case still declined: the reference anneals boundary slots only.",
      "⚠️ The committed annealing goldens in the reference test suite are STALE: a live run of the pinned build disagrees with `annealing1.defok` on 49 of 54 pins. Compare annealing against a live run, never against those files.",
      "A POLYGON die has no named edges, so its boundary is handled as a list of segments: an edge is a segment, its direction comes from the order of its endpoints, and sections are cut per segment. Five points is a RECTANGLE (the ring repeats its first point); more than five takes the polygon path. LIMITATION: edge-named region constraints (`-region bottom:...`) are REPORTED AND IGNORED on a polygon die rather than reinterpreted against the bounding box, which would satisfy a constraint the design did not ask for.",
      "TOP-LAYER pins are placed on a 2-D lattice INSIDE the die rather than on its boundary, so almost none of the edge rules apply to them: there is no direction to order a group along and no opposite side to mirror to. A lattice position is legal only if a pin of the declared size FITS there -- inside the die, and clear of routing blockages, the power grid and fixed ports on that layer by at least the keepout. Non-rectangular grid regions are not handled.",
      "A MIRRORED pair is one decision, not two: only one half competes for a position and it is costed for both, its partner taking the reflection of whatever it gets. Mirrored pins are placed before free ones -- they need two positions open at once, so they have the least room to manoeuvre. If a reflection is unavailable, BOTH halves are reported unplaced; half a pair is a broken symmetry, not a partial success.",
      "MEASURED: all 62 comparable reference cases match the reference total wirelength or beat it -- 25 of them position-for-position, the rest by a cost-equal tie. No case is worse, none violates a constraint, and none leaves a pin unplaced.",
      "A slot is unusable for two independent reasons, both read from the block: it falls inside an EXCLUDED region, or it is covered by the metal of a port already placed FIXED. An excluded region is strict at both ends -- a slot exactly on the boundary is still usable, which is the reference's own convention.",
      "A pin GROUP occupies a contiguous run of slots and is placed before any individual pin, because a single pin dropped into the only long enough run destroys it irrecoverably. `-order` fixes the sequence, and only changes the result on the top and left edges, whose slot lists run in the opposite direction.",
      "A group larger than one section takes a FALLBACK path: the first contiguous run long enough, searched over slot indices rather than sections, so the run may cross an edge. On the top and left edges that path reverses the group unconditionally, where the matched path reverses only when `-order` is given -- a difference inherited from the reference, not a rule with a stated reason.",
      "Constraints are read back from the DATABASE, where `set_io_pin_constraint` stores them on the ports -- they are not command-line arguments here. A constrained pin is placed BEFORE any free pin, into sections cut from its own region, and the slots it takes are withdrawn; the reverse order would let a free pin occupy a region a constrained pin has no alternative to.",
      "Where two constraint regions OVERLAP, the one with more room per pin is served first, since whoever is served first takes the shared slots. Non-overlapping constraints keep the design's own order.",
      "A constrained pin that does not fit its region is reported UNPLACED, not relocated: the design asked for a region, and somewhere else is not a smaller version of that answer.",
      "It reports the chosen pin positions, and with `--out-odb` WRITES them: each placed port's pins replaced by one rectangle on its layer, PLACED -- the depth from the layer's minimum width and area (or `--hor-length`/`--ver-length`), widened by the thickness multiplier, extended out of the die by `--hor-extension`/`--ver-extension`, oriented from the position (from probes either side of the segment on a polygon die). Measured 2026-10-14: every pin both this and the reference place at the same point carries an identical DEF record, 4853 of 4853, rectangle and polygon dies alike. A placement that leaves a pin unplaced writes nothing.",
      "Assignment is optimal WITHIN a section and greedy BETWEEN sections: pins are routed to the cheapest section with room, and only then matched optimally inside it. This is the reference decomposition, not an approximation introduced here.",
      "Cost is half-perimeter wirelength over the net bounding box. A net whose driver or loads are unplaced uses the die centre for them, as the reference does.",
      "The optimal assignment COST is unique but the optimal PAIRING is not: where two pairings cost the same, this and the reference may place two pins in swapped slots and both be correct. Compare total cost before treating a difference as a defect -- `--evaluate` scores a reference placement under the same cost model for exactly this.",
      "Slots are generated on the DIE boundary from each layer's routing track patterns, minus corner avoidance, minus half the pin width at each end, minus the requested minimum distance.",
      "PARTIAL: slot availability accounts for excluded regions and for fixed ports' metal, but NOT yet for macros or routing obstructions. Where a macro abuts the boundary, availability remains optimistic.",
      "The default corner avoidance is resolved once from a layer's FIRST track pattern and reused for the rest, which is the reference's behavior and is observable on layers carrying mixed-pitch patterns. Reproduced deliberately.",
      "`-min_distance_in_tracks` with a distance of 0 is a division by zero in the reference; here it keeps every candidate. A deliberate divergence, on an input that has no defined meaning.",
      "Written against the reference ppl sources at pin @OPENROAD_PIN@. The algorithm is reimplemented from the published behavior, not transliterated."
  ],
  "invocation": {
    "args_template": ["place-pins", "{odb}", "--hor-layers", "{hor_layers}", "--ver-layers", "{ver_layers}"],
    "optional": [
      { "arg": "min_distance", "flag": "--min-distance" },
      { "arg": "min_distance_in_tracks", "flag": "--min-distance-in-tracks" },
      { "arg": "corner_avoidance", "flag": "--corner-avoidance" },
      { "arg": "hor_length", "flag": "--hor-length" },
      { "arg": "ver_length", "flag": "--ver-length" },
      { "arg": "hor_extension", "flag": "--hor-extension" },
      { "arg": "ver_extension", "flag": "--ver-extension" },
      { "arg": "annealing", "flag": "--annealing" },
      { "arg": "random_seed", "flag": "--random-seed" },
      { "arg": "out", "flag": "-o" },
      { "arg": "out_odb", "flag": "--out-odb" }
    ],
    "emits_json": true
  },
  "inputs": {
    "type": "object",
    "required": ["odb", "hor_layers", "ver_layers"],
    "properties": {
      "odb": { "type": "string", "description": "path to the design database (.odb)" },
      "hor_layers": { "type": "string", "description": "comma-separated layers for the left/right edges" },
      "ver_layers": { "type": "string", "description": "comma-separated layers for the bottom/top edges" },
      "min_distance": { "type": "number", "description": "minimum pin spacing in microns (omitted: candidates every 2 tracks)" },
      "min_distance_in_tracks": { "type": "boolean", "description": "read min_distance as a count of candidate slots instead" },
      "corner_avoidance": { "type": "number", "description": "clearance from each corner in microns (omitted: 2 tracks, capped at 1um)" },
      "hor_length": { "type": "number", "description": "depth of pins on the left/right edges, in microns" },
      "ver_length": { "type": "number", "description": "depth of pins on the bottom/top edges, in microns" },
      "hor_extension": { "type": "number", "description": "how far pins on the left/right edges reach past the die, in microns" },
      "ver_extension": { "type": "number", "description": "the same for pins on the bottom/top edges" },
      "annealing": { "type": "boolean", "description": "place by simulated annealing instead of optimal matching" },
      "random_seed": { "type": "integer", "description": "annealing seed (default 42)" },
      "out": { "type": "string", "description": "write the report to FILE instead of stdout" },
      "out_odb": { "type": "string", "description": "write the database with the placed pins to FILE" }
    }
  },
  "consumes": ["odb"],
  "produces": ["odb"],
  "artifacts": [ { "role": "slot_report", "field": "report_path" } ],
  "assertion": {
    "id": "pins-placed",
    "field": "status",
    "pass_when": { "eq": "ok" }
  }
}
"#;

#[derive(Debug, Default)]
struct Opts {
    odb: String,
    keys: Vec<(String, String)>,
    in_tracks: bool,
    annealing: bool,
}

impl Opts {
    fn get(&self, k: &str) -> Option<&str> {
        self.keys.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts::default();
    let mut odb = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--json" => {}
            "--min-distance-in-tracks" => o.in_tracks = true,
            "--annealing" => o.annealing = true,
            a if a.starts_with("--") || a == "-o" => {
                i += 1;
                let v = args.get(i).cloned().ok_or_else(|| format!("{a} needs a value"))?;
                o.keys.push((a.trim_start_matches('-').to_string(), v));
            }
            a if a.starts_with('-') => return Err(format!("unknown option `{a}`")),
            a => odb = Some(a.to_string()),
        }
        i += 1;
    }
    o.odb = odb.ok_or("a path to a .odb is required")?;
    Ok(o)
}

/// A length given in microns, as DBU.
fn microns(opts: &Opts, key: &str, dbu: i32, default: i32) -> Result<i32, String> {
    match opts.get(key) {
        None => Ok(default),
        Some(v) => v
            .parse::<f64>()
            .map(|m| (m * dbu as f64).round() as i32)
            .map_err(|_| format!("--{key} wants a number, got `{v}`")),
    }
}

fn multiplier(opts: &Opts, key: &str) -> Result<f64, String> {
    match opts.get(key) {
        None => Ok(1.0),
        Some(v) => v.parse::<f64>().map_err(|_| format!("--{key} wants a number, got `{v}`")),
    }
}

fn layer_list(opts: &Opts, key: &str) -> Vec<String> {
    opts.get(key)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn edge_name(e: Edge) -> &'static str {
    match e {
        Edge::Bottom => "bottom",
        Edge::Right => "right",
        Edge::Top => "top",
        Edge::Left => "left",
        Edge::Invalid => "top-layer",
    }
}

use vyges_ppl::session::Prepared;

/// The run's pin layers: each direction from its switch, else from `set_place_config`'s stored
/// list (`applySettings`: `hor_layers_`/`ver_layers_` filled only when empty).
fn run_layers(opts: &Opts, db: &Db) -> (Vec<String>, Vec<String>) {
    let (hor, ver) = (layer_list(opts, "hor-layers"), layer_list(opts, "ver-layers"));
    if !hor.is_empty() && !ver.is_empty() {
        return (hor, ver);
    }
    let stored = vyges_ppl::session::place_config_args(db);
    (if hor.is_empty() { stored.hor } else { hor }, if ver.is_empty() { stored.ver } else { ver })
}

/// Rule (`IOPlacer::applySettings`): a corner avoidance left unset (negative) takes the stored
/// one, and a minimum distance left at 0 takes the stored distance AND its in-tracks flag; a
/// value the run gave stands. (The layers fall back likewise, each direction on its own.)
///
/// ⏭ Pending upstream #12094 (per-layer IO pin minimum distance, ppl and gpl) would change what
/// the stored minimum distance is; if it lands, this function and its test are the repin edit.
fn apply_settings(mut given: Params, stored: &Params) -> Params {
    if given.corner_avoidance < 0 {
        given.corner_avoidance = stored.corner_avoidance;
    }
    if given.min_distance == 0 {
        given.min_distance = stored.min_distance;
        given.min_distance_in_tracks = stored.min_distance_in_tracks;
    }
    given
}

fn prepare(opts: &Opts) -> Result<Prepared, String> {
    let db = Db::open(&opts.odb).map_err(|e| format!("cannot read {}: {e}", opts.odb))?;
    let dbu = db.dbu_per_micron();
    if dbu <= 0 {
        return Err("no DBU scale".into());
    }
    // `applySettings`: what `set_place_config` stored on the block stands in for each switch
    // this run did not give — see [`apply_settings`].
    let stored = vyges_ppl::session::place_config_args(&db);
    let (hor, ver) = run_layers(opts, &db);
    if hor.is_empty() {
        return Err("[ERROR PPL-0017] -hor_layers or set_place_config -io_pin_hor_layers is required.".into());
    }
    if ver.is_empty() {
        return Err("[ERROR PPL-0018] -ver_layers or set_place_config -io_pin_ver_layers is required.".into());
    }
    let given = Params {
        // In tracks it is a raw count; otherwise a length in microns. Not converting the
        // in-tracks form is the whole point of the flag.
        min_distance: if opts.in_tracks {
            opts.get("min-distance")
                .unwrap_or("0")
                .parse::<i32>()
                .map_err(|_| "--min-distance in tracks wants a whole number".to_string())?
        } else {
            microns(opts, "min-distance", dbu, 0)?
        },
        min_distance_in_tracks: opts.in_tracks,
        corner_avoidance: microns(opts, "corner-avoidance", dbu, -1)?,
        thickness_multiplier_h: multiplier(opts, "hor-multiplier")?,
        thickness_multiplier_v: multiplier(opts, "ver-multiplier")?,
    };
    let params = apply_settings(given, &stored.params);

    // `set_pin_length -hor_length/-ver_length` (microns, to database units as `microns_to_dbu`
    // does): it replaces the pin's depth, so it also deepens every keep-out the pin is measured
    // against. Not stored in the database, so it arrives as a flag.
    let user_length = |key: &str| -> Result<Option<i32>, String> { Ok(Some(microns(opts, key, dbu, -1)?).filter(|v| *v != -1)) };
    let (hor_length, ver_length) = (user_length("hor-length")?, user_length("ver-length")?);
    vyges_ppl::session::prepare_slots(db, &vyges_ppl::session::SlotArgs { hor, ver, params, hor_length, ver_length })
}

fn write_report(opts: &Opts, report: &str) -> Result<(), String> {
    match opts.get("o") {
        Some(path) => std::fs::write(path, format!("{report}\n"))
            .map_err(|e| format!("cannot write {path}: {e}")),
        None => {
            println!("{report}");
            Ok(())
        }
    }
}

fn slots(args: &[String]) -> ExitCode {
    let opts = match parse_opts(args) {
        Ok(o) => o,
        Err(e) => {
            diag!("vyges-ppl: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let p = match prepare(&opts) {
        Ok(p) => p,
        Err(e) => {
            diag!("vyges-ppl: {e}");
            return ExitCode::from(2);
        }
    };

    emit_slot_events(&p.slots, &p.trackless);
    let report = report_json(&p.slots, p.boundary, p.dbu, &p.trackless);
    if let Err(e) = write_report(&opts, &report) {
        diag!("vyges-ppl: {e}");
        return ExitCode::from(2);
    }
    if p.slots.is_empty() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn place_pins(args: &[String]) -> ExitCode {
    let opts = match parse_opts(args) {
        Ok(o) => o,
        Err(e) => {
            diag!("vyges-ppl: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut p = match prepare(&opts) {
        Ok(p) => p,
        Err(e) => {
            diag!("vyges-ppl: {e}");
            return ExitCode::from(2);
        }
    };
    // `place_pins`' Tcl, before it hands over: every BLOCK instance is a blockage when placed
    // (PPL-15 when not), and the count is reported; with no -min_distance, the 2-track default.
    let mut log = RunLog::new();
    let mut macros = 0;
    for inst in p.db.inst_names() {
        if p.db.master_get_type(&p.db.inst_master(&inst)).unwrap_or_default().starts_with("BLOCK") {
            if p.db.inst_is_placed(&inst) {
                macros += 1;
            } else {
                log.warn(15, format!("Macro {inst} is not placed."));
            }
        }
    }
    log.report(format!("Found {macros} macro blocks."));
    // Only when set_place_config stored no distance either (`get_default_min_distance` is 0).
    if opts.get("min-distance").is_none() && vyges_ppl::session::place_config_args(&p.db).params.min_distance == 0 {
        log.report("Using 2 tracks default min distance between IO pins.".into());
    }
    let per_section = match opts.get("slots-per-section") {
        None => SLOTS_PER_SECTION,
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                diag!("vyges-ppl: --slots-per-section wants a positive whole number");
                return ExitCode::from(2);
            }
        },
    };

    let die = (p.boundary.x0, p.boundary.y0, p.boundary.x1, p.boundary.y1);
    let net = matching::init_netlist(&p.db, die, &p.slots, per_section, false);

    // ── Top layer ────────────────────────────────────────────────────────────────────────────
    // Pins sent to the top layer are placed on their own lattice, not on the boundary, so they are
    // taken out of the edge problem entirely rather than competing in it.
    let (top_slots, top_pins) = matching::read_top_layer(&p.db, die, &net.pins);
    // `printConfig`: what the run was given, before any pin is assigned.
    let config = matching::config(&p.db, &p.slots, &top_slots, &net, (!opts.annealing).then_some(per_section), false);
    for line in vyges_ppl::report::config(&config) {
        log.info(line);
    }
    let top_set: std::collections::BTreeSet<usize> = top_pins.iter().map(|&(i, _)| i).collect();
    let (top_placed, mut unplaced) = matching::place_top_layer(&top_slots, &net.pins, &top_pins, per_section);
    let (pins, groups, constraints) = (&net.pins, &net.groups, &net.constraints);

    // ── Annealing ────────────────────────────────────────────────────────────────────────────
    // A wholly different optimiser, so it REPLACES the boundary placement rather than adjusting
    // it. It refuses anything needing move types it does not implement: a partial implementation
    // would desynchronise its random stream and produce a plausible wrong answer, which is worse
    // than declining.
    let annealed = if opts.annealing {
        let cfg = match anneal_config(&opts) {
            Ok(c) => c,
            Err(e) => {
                diag!("vyges-ppl: {e}");
                return ExitCode::from(2);
            }
        };
        // `runAnnealing`: the constraints as `initConstraints(true)` leaves them — in the order
        // read, unsorted — and the whole netlist annealed over the boundary slots. A pin sent to
        // the top layer is the one thing the reference's annealer does not place.
        let outcome = if top_set.is_empty() {
            let regions = annealing::regions(&p.slots, &net.unsorted_constraints, groups, per_section);
            annealing::run_with(&p.slots, pins, groups, &regions, die, &cfg)
        } else {
            Err(annealing::Unsupported::NeedsExtraMoves)
        };
        match outcome {
            Ok(v) => Some(v),
            Err(why) => {
                diag!(
                    "vyges-ppl: cannot anneal this design ({why:?}); \
                     re-run without --annealing to place it by matching"
                );
                return ExitCode::from(1);
            }
        }
    } else {
        None
    };

    let (edge_placed, edge_unplaced) = match &annealed {
        Some(v) => (v.clone(), Vec::new()),
        None => matching::place_boundary(&p.slots, p.on_segment.as_deref(), &net, &top_set, per_section, die),
    };
    unplaced.extend(edge_unplaced.into_iter().filter(|i| !top_set.contains(i)));
    let (placed, unplaced) = matching::resolve((&top_placed, &top_slots), &edge_placed, &p.slots, &top_set, unplaced);
    if annealed.is_none() {
        log.info(vyges_ppl::report::sections_assigned(pins.len() - unplaced.len(), pins.len()));
    }
    if unplaced.is_empty() {
        log.info(vyges_ppl::report::io_hpwl(pins, placed.iter().map(|x| (x.pin, (x.x, x.y))), p.dbu));
    }

    emit_slot_events(&p.slots, &p.trackless);
    if p.exclusions > 0 {
        let n = p.slots.iter().filter(|s| s.blocked).count();
        vyges_events::emit(
            &vyges_events::Event::new(
                "vyges-ppl",
                vyges_events::Severity::Info,
                format!("{} excluded region(s) block {n} slot(s)", p.exclusions),
            )
            .with_code("PPL-EXCLUDED"),
        );
    }
    emit_constraint_events(constraints, groups, pins);
    emit_placement_events(pins, &placed, &unplaced);

    let reference = match opts.get("evaluate") {
        None => None,
        Some(path) => match score_reference(path, pins) {
            Ok(v) => Some(v),
            Err(e) => {
                diag!("vyges-ppl: {e}");
                return ExitCode::from(2);
            }
        },
    };

    let outline = p.db.die_area_polygon().unwrap_or_default();
    let report =
        placement_json(pins, groups, &placed, &unplaced, p.boundary, &outline, p.dbu, reference);
    if let Err(e) = write_report(&opts, &report) {
        diag!("vyges-ppl: {e}");
        return ExitCode::from(2);
    }
    if let Some(out) = opts.get("out-odb") {
        if let Err((code, e)) = commit_pins(&opts, &mut p, pins, &placed, &unplaced, out) {
            diag!("vyges-ppl: {e}");
            return ExitCode::from(code);
        }
    }
    if unplaced.is_empty() && !pins.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// `commitIOPlacementToDB` ([`matching::commit`]) with this command line's pin sizing, and the
/// database written to `out`. Exit 1 for an unplaced pin (PPL-0039), 2 for anything else.
fn commit_pins(opts: &Opts, p: &mut Prepared, pins: &[Pin], placed: &[matching::Placed], unplaced: &[usize], out: &str) -> Result<(), (u8, String)> {
    let length = |key: &str| -> Result<Option<i32>, (u8, String)> { Ok(Some(microns(opts, key, p.dbu, -1).map_err(|e| (2, e))?).filter(|v| *v != -1)) };
    let shape = matching::PinShape {
        hor_length: length("hor-length")?,
        ver_length: length("ver-length")?,
        hor_extension: length("hor-extension")?.unwrap_or(0),
        ver_extension: length("ver-extension")?.unwrap_or(0),
        hor_multiplier: multiplier(opts, "hor-multiplier").map_err(|e| (2, e))?,
        ver_multiplier: multiplier(opts, "ver-multiplier").map_err(|e| (2, e))?,
    };
    let (hor, ver) = run_layers(opts, &p.db);
    let surface = matching::Surface { boundary: p.boundary, slots: &p.slots, on_segment: p.on_segment.as_deref(), hor: &hor, ver: &ver };
    matching::commit(&mut p.db, &surface, &shape, pins, placed, unplaced).map_err(|f| (if matches!(f, matching::Fail::Error(39, _)) { 1 } else { 2 }, f.line()))?;
    p.db.write(out).map_err(|e| (2, format!("cannot write {out}: {e}")))
}

/// The placement, with the total cost alongside it.
///
/// ⚠️ The cost is not decoration. The optimal *pairing* is not unique where two pairings tie, so a
/// position-by-position diff against another correct implementation can differ legitimately. The
/// total is the invariant, and a comparison should check it before reporting a mismatch.
/// The annealing knobs, from the command line, falling back to the reference's defaults.
fn anneal_config(opts: &Opts) -> Result<annealing::Anneal, String> {
    let num = |key: &str, default: f32| -> Result<f32, String> {
        match opts.get(key) {
            None => Ok(default),
            Some(v) => v.parse::<f32>().map_err(|_| format!("--{key} wants a number, got `{v}`")),
        }
    };
    let int = |key: &str, default: i32| -> Result<i32, String> {
        match opts.get(key) {
            None => Ok(default),
            Some(v) => {
                v.parse::<i32>().map_err(|_| format!("--{key} wants a whole number, got `{v}`"))
            }
        }
    };
    let d = annealing::Anneal::default();
    Ok(annealing::Anneal {
        init_temperature: num("temperature", d.init_temperature)?,
        max_iterations: int("max-iterations", d.max_iterations)?,
        perturb_per_iter: int("perturb-per-iter", d.perturb_per_iter)?,
        alpha: num("alpha", d.alpha)?,
        seed: int("random-seed", d.seed as i32)? as u32,
    })
}

/// A JSON string literal for `s`, escaped.
///
/// ⛔ **DEF escapes a bracket in a name as `\\[`, and `\\[` is not a legal JSON escape.** This
/// report is assembled with `format!`, so a name carrying a backslash produced a file that no JSON
/// parser would read: `ppl-place-check` died with *"Invalid \\escape"* on the whole run, not on
/// the one case. Found 2026-09-03 when the `7d490b8` re-pin brought upstream's new
/// `annealing_pdn_boundary` case, whose pins are `req_msg\\[0\\]`.
///
/// 🔑 **The escaping belongs to the SERIALIZER, not the caller.** `serde_json` is already a
/// dependency and knows every case (backslash, quote, control characters); spelling out a
/// `replace` chain here would be the same bug waiting for a different character.
fn json_str(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

#[allow(clippy::too_many_arguments)]
fn placement_json(
    pins: &[Pin],
    groups: &[Group],
    placed: &[matching::Placed],
    unplaced: &[usize],
    boundary: Boundary,
    outline: &[(i32, i32)],
    dbu: i32,
    reference: Option<i64>,
) -> String {
    let total: i64 = placed.iter().map(|p| vyges_ppl::net_hpwl(&pins[p.pin].sinks, (p.x, p.y))).sum();
    let list = placed
        .iter()
        .map(|p| {
            format!(
                "    {{\"pin\": {}, \"slot\": {}, \"x\": {}, \"y\": {}, \"layer\": \"{}\", \
                 \"edge\": \"{}\", \"hpwl\": {}}}",
                json_str(&pins[p.pin].name),
                p.slot,
                p.x,
                p.y,
                p.layer,
                edge_name(p.edge),
                vyges_ppl::net_hpwl(&pins[p.pin].sinks, (p.x, p.y))
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let missed = unplaced
        .iter()
        .map(|&i| json_str(&pins[i].name))
        .collect::<Vec<_>>()
        .join(", ");
    // The groups as READ, so a checker can verify contiguity without re-deriving them from a Tcl
    // script it may not be able to evaluate. This reports the input, not the outcome — whether
    // these pins ended up adjacent is still for the checker to decide.
    let group_list = groups
        .iter()
        .map(|g| {
            let names: Vec<String> =
                g.pins.iter().map(|&i| json_str(&pins[i].name)).collect();
            format!("    {{\"ordered\": {}, \"pins\": [{}]}}", g.ordered, names.join(", "))
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let group_list = if group_list.is_empty() {
        String::from("\n  \"groups\": [],")
    } else {
        format!("\n  \"groups\": [\n{group_list}\n  ],")
    };

    // The die OUTLINE, not just its bounding box. A checker undoing a pin's length offset has to
    // know which boundary the pin sits on, and on a rectilinear die the bounding box does not say.
    let outline_json = outline
        .iter()
        .map(|(x, y)| format!("[{x}, {y}]"))
        .collect::<Vec<_>>()
        .join(", ");

    let reference = match reference {
        Some(v) => format!("\n  \"reference_hpwl\": {v},"),
        None => String::new(),
    };
    format!(
        "{{\n  \"tool\": \"vyges-ppl\",\n  \"status\": \"{}\",\n  \"dbu_per_micron\": {dbu},\n  \
         \"die_area\": [{}, {}, {}, {}],\n  \"die_polygon\": [{outline_json}],\n  \"pins_total\": {},\n  \
         \"pins_placed\": {},\n  \"total_hpwl\": {total},{reference}\n  \"unplaced\": [{missed}],\
         {group_list}\n  \"placements\": [\n{list}\n  ]\n}}",
        // ⚠️ Not just "nothing was reported unplaced": every pin must actually have a position.
        // A pin that is neither placed nor reported is the worst kind of failure, because the
        // status looks clean. Counting is the only way to catch it.
        if unplaced.is_empty() && placed.len() == pins.len() && !pins.is_empty() {
            "ok"
        } else {
            "refused"
        },
        boundary.x0,
        boundary.y0,
        boundary.x1,
        boundary.y1,
        pins.len(),
        placed.len(),
    )
}

fn report_json(found: &[Slot], boundary: Boundary, dbu: i32, trackless: &[String]) -> String {
    let mut per_edge: std::collections::BTreeMap<&str, usize> = Default::default();
    for s in found {
        *per_edge.entry(edge_name(s.edge)).or_default() += 1;
    }
    let counts = per_edge
        .iter()
        .map(|(e, n)| format!("\"{e}\": {n}"))
        .collect::<Vec<_>>()
        .join(", ");
    // Emitted in generation order, because the order IS the contract: later stages assign into
    // contiguous runs of this list.
    let list = found
        .iter()
        .map(|s| {
            format!(
                "    {{\"x\": {}, \"y\": {}, \"layer\": \"{}\", \"edge\": \"{}\", \"blocked\": {}}}",
                s.x,
                s.y,
                s.layer,
                edge_name(s.edge),
                s.blocked
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let skipped =
        trackless.iter().map(|t| format!("\"{t}\"")).collect::<Vec<_>>().join(", ");
    format!(
        "{{\n  \"tool\": \"vyges-ppl\",\n  \"status\": \"{}\",\n  \"dbu_per_micron\": {dbu},\n  \
         \"die_area\": [{}, {}, {}, {}],\n  \"slots_total\": {},\n  \"slots_by_edge\": {{{counts}}},\n  \
         \"layers_without_tracks\": [{skipped}],\n  \"slots\": [\n{list}\n  ]\n}}",
        if found.is_empty() { "refused" } else { "ok" },
        boundary.x0,
        boundary.y0,
        boundary.x1,
        boundary.y1,
        found.len(),
    )
}

fn emit_slot_events(found: &[Slot], trackless: &[String]) {
    use vyges_events::{Event, Severity};

    for layer in trackless {
        // A layer named on the command line that carries no track grid contributes nothing. That
        // is almost always a wrong layer name, and it is silent otherwise.
        vyges_events::emit(
            &Event::new(
                "vyges-ppl",
                Severity::Warn,
                format!("layer {layer} has no routing tracks; it offers no pin positions"),
            )
            .with_code("PPL-LAYER-NO-TRACKS")
            .with_objects(vec![format!("layer:{layer}")]),
        );
    }

    let mut per_edge: std::collections::BTreeMap<&str, usize> = Default::default();
    for s in found {
        *per_edge.entry(edge_name(s.edge)).or_default() += 1;
    }
    for (edge, n) in &per_edge {
        vyges_events::emit(
            &Event::new("vyges-ppl", Severity::Info, format!("{n} slot(s) on the {edge} edge"))
                .with_code("PPL-EDGE-SLOTS")
                .with_objects(vec![format!("edge:{edge}")]),
        );
    }
    vyges_events::emit(
        &Event::new(
            "vyges-ppl",
            if found.is_empty() { Severity::Warn } else { Severity::Info },
            format!("{} legal pin position(s) over {} edge(s)", found.len(), per_edge.len()),
        )
        .with_code("PPL-SLOTS"),
    );
}

/// What the design asked for, in the causal trail.
///
/// Worth emitting even when everything succeeds: a constraint that quietly matched nothing — a
/// misspelled port, a region on the wrong edge — is otherwise indistinguishable from no constraint.
fn emit_constraint_events(
    constraints: &[vyges_ppl::Constraint],
    groups: &[Group],
    pins: &[Pin],
) {
    use vyges_events::{Event, Severity};
    for g in groups {
        let names: Vec<&str> = g.pins.iter().map(|&i| pins[i].name.as_str()).collect();
        vyges_events::emit(
            &Event::new(
                "vyges-ppl",
                Severity::Info,
                format!(
                    "pin group of {}{}: [ {} ]",
                    g.pins.len(),
                    if g.ordered { ", ordered" } else { "" },
                    names.join(" ")
                ),
            )
            .with_code("PPL-PIN-GROUP")
            .with_objects(names.iter().map(|n| format!("pin:{n}")).collect()),
        );
    }
    for c in constraints {
        let names: Vec<&str> = c.pins.iter().map(|&i| pins[i].name.as_str()).collect();
        vyges_events::emit(
            &Event::new(
                "vyges-ppl",
                Severity::Info,
                format!(
                    "restrict {} pin(s) to {} {}-{} [ {} ]",
                    c.pins.len(),
                    edge_name(c.interval.edge),
                    c.interval.begin,
                    c.interval.end,
                    names.join(" ")
                ),
            )
            .with_code("PPL-CONSTRAINT")
            .with_objects(names.iter().map(|n| format!("pin:{n}")).collect()),
        );
    }
}

/// Score somebody else's placement under our own cost model.
///
/// The point is to separate two very different outcomes that look identical in a position diff: a
/// placement that differs because it is **worse**, and one that differs because the optimum is
/// **tied** and the tie broke the other way. Only the first is a defect.
///
/// Pins the reference does not mention are skipped rather than guessed at, and the count of those
/// is left to the caller — a partial reference scores lower purely for being partial.
fn score_reference(path: &str, pins: &[Pin]) -> Result<i64, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let map: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let obj = map.as_object().ok_or_else(|| format!("{path}: expected an object of pin -> [x, y]"))?;

    let mut total = 0i64;
    for pin in pins {
        let Some(v) = obj.get(&pin.name) else { continue };
        let at = v
            .as_array()
            .and_then(|a| Some((a.first()?.as_i64()? as i32, a.get(1)?.as_i64()? as i32)))
            .ok_or_else(|| format!("{path}: {} is not a [x, y] pair", pin.name))?;
        total += vyges_ppl::net_hpwl(&pin.sinks, at);
    }
    Ok(total)
}

/// What the placement did, in the causal trail.
fn emit_placement_events(pins: &[Pin], placed: &[matching::Placed], unplaced: &[usize]) {
    use vyges_events::{Event, Severity};

    for &i in unplaced {
        // A pin with nowhere to go is the one failure a caller must not miss: the design has more
        // IO than the boundary can carry at the spacing asked for.
        vyges_events::emit(
            &Event::new(
                "vyges-ppl",
                Severity::Error,
                format!("no free slot for pin {}", pins[i].name),
            )
            .with_code("PPL-PIN-UNPLACED")
            .with_objects(vec![format!("pin:{}", pins[i].name)]),
        );
    }
    vyges_events::emit(
        &Event::new(
            "vyges-ppl",
            if unplaced.is_empty() { Severity::Info } else { Severity::Error },
            format!("placed {} of {} pin(s)", placed.len(), pins.len()),
        )
        .with_code("PPL-DONE"),
    );
}

pub fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // ⛔ Before any database exists: libodb then logs to the events trail (stderr) only, and
    // stdout carries nothing but the report a caller parses.
    vyges_opendb::init_events_logging();
    match args.first().map(String::as_str) {
        // 🔑 **The commit, not just the version.** Two binaries can share a version and differ by a
        // fix, so a bug report needs the build. build.rs prefers GITHUB_SHA on CI, which is what stops
        // a release being stamped -dirty by the untracked files a release run leaves behind.
        //
        // ⚠️ Answered before --describe, --help and any argument parsing: asking a binary what it is
        // must not depend on the rest of the command line being valid.
        Some("--version") | Some("-V") => {
            println!("vyges-ppl {} ({})", vyges_ppl::VERSION, env!("VYGES_GIT_SHA"));
            println!("{}", vyges_ppl::COPYRIGHT);
            ExitCode::SUCCESS
        }
        Some("--describe") => {
            println!("{}", describe());
            ExitCode::SUCCESS
        }
        Some("--help") | Some("-h") | None => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("slots") => slots(&args[1..]),
        Some("place-pins") => place_pins(&args[1..]),
        Some(other) => {
            diag!("vyges-ppl: unknown command `{other}`\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_setting_the_run_did_not_give_comes_from_set_place_config() {
        // Rule (IOPlacer::applySettings): corner avoidance < 0 takes the stored value; a minimum
        // distance of 0 takes the stored distance AND its in-tracks flag; given values stand.
        let stored = Params { corner_avoidance: 700, min_distance: 3, min_distance_in_tracks: true, ..Params::default() };
        let got = apply_settings(Params::default(), &stored);
        assert_eq!((got.corner_avoidance, got.min_distance, got.min_distance_in_tracks), (700, 3, true));
        let given = Params { corner_avoidance: 0, min_distance: 140, ..Params::default() };
        let got = apply_settings(given, &stored);
        assert_eq!((got.corner_avoidance, got.min_distance, got.min_distance_in_tracks), (0, 140, false));
    }

    /// ⛔ **DEF escapes a bracket as `\[`, and `\[` is not a legal JSON escape.**
    ///
    /// The report is assembled with `format!`, so a pin named `req_msg\[0\]` produced a file no
    /// parser would read and `ppl-place-check` died on the WHOLE run — 47 already-compared cases
    /// lost to one unescaped character. Surfaced by upstream's new `annealing_pdn_boundary` case
    /// at the `7d490b8` re-pin; nothing in the corpus before it had a backslash in a name.
    #[test]
    fn a_name_carrying_a_def_escape_still_serialises_to_readable_json() {
        let raw = r"req_msg\[0\]";
        let out = json_str(raw);
        assert_eq!(out, r#""req_msg\\[0\\]""#, "the backslash must itself be escaped");
        let back: serde_json::Value =
            serde_json::from_str(&out).expect("a JSON parser must be able to read it back");
        assert_eq!(back.as_str().unwrap(), raw, "and it must round-trip to the original name");
    }

    /// The other characters a hand-rolled `format!` would also have got wrong.
    #[test]
    fn json_str_escapes_quotes_and_control_characters_too() {
        for raw in [r#"a"b"#, "a\tb", "a\nb", r"a\b"] {
            let out = json_str(raw);
            let back: serde_json::Value =
                serde_json::from_str(&out).expect("must parse");
            assert_eq!(back.as_str().unwrap(), raw);
        }
    }

    #[test]
    fn the_layer_lists_are_required_and_options_are_checked() {
        assert!(parse_opts(&[]).is_err(), "a .odb is required");
        let ok = ["d.odb", "--hor-layers", "met3,met5"].map(String::from);
        assert_eq!(parse_opts(&ok).unwrap().get("hor-layers"), Some("met3,met5"));
        let dangling = ["d.odb", "--hor-layers"].map(String::from);
        assert!(parse_opts(&dangling).unwrap_err().contains("--hor-layers"));
        let bad: Vec<String> = vec!["-x".to_string()];
        assert!(parse_opts(&bad).unwrap_err().contains("unknown"));
    }

    #[test]
    fn a_layer_list_tolerates_spacing_and_a_trailing_comma() {
        let o = parse_opts(&["d.odb", "--ver-layers", "met2, met4,"].map(String::from)).unwrap();
        assert_eq!(layer_list(&o, "ver-layers"), vec!["met2", "met4"]);
        assert!(layer_list(&o, "hor-layers").is_empty(), "an absent list is empty, not [\"\"]");
    }

    #[test]
    fn distances_are_microns_except_when_they_are_track_counts() {
        // The one unit trap in the command: --min-distance changes meaning with the flag beside
        // it, so converting both the same way would be wrong by a factor of the DBU scale.
        let o = parse_opts(&["d.odb", "--min-distance", "0.5"].map(String::from)).unwrap();
        assert_eq!(microns(&o, "min-distance", 1000, 0).unwrap(), 500);
        assert_eq!(microns(&o, "corner-avoidance", 1000, -1).unwrap(), -1, "absent stays unset");
        assert!(microns(&o, "min-distance", 1000, 0).unwrap() > 0);

        let tracks = parse_opts(
            &["d.odb", "--min-distance", "3", "--min-distance-in-tracks"].map(String::from),
        )
        .unwrap();
        assert!(tracks.in_tracks);
        assert_eq!(tracks.get("min-distance"), Some("3"), "kept as a count, not scaled");
    }

    #[test]
    fn a_non_numeric_distance_is_an_error_rather_than_a_zero() {
        let o = parse_opts(&["d.odb", "--min-distance", "wide"].map(String::from)).unwrap();
        assert!(microns(&o, "min-distance", 1000, 0).unwrap_err().contains("min-distance"));
        let m = parse_opts(&["d.odb", "--hor-multiplier", "x2"].map(String::from)).unwrap();
        assert!(multiplier(&m, "hor-multiplier").is_err());
    }

    #[test]
    fn the_report_keeps_the_slots_in_generation_order() {
        // The order is the contract, so the report must not sort or group them.
        let mk = |x, edge| Slot { x, y: 0, layer: "met1".into(), edge, blocked: false };
        let found = vec![mk(10, Edge::Bottom), mk(90, Edge::Top), mk(20, Edge::Bottom)];
        let json = report_json(&found, Boundary { x0: 0, y0: 0, x1: 100, y1: 100 }, 1000, &[]);
        let at = |needle: &str| json.find(needle).unwrap();
        assert!(at("\"x\": 10") < at("\"x\": 90"));
        assert!(at("\"x\": 90") < at("\"x\": 20"), "not reordered by edge or position");
        assert!(json.contains("\"slots_total\": 3"));
        assert!(json.contains("\"status\": \"ok\""));
    }

    #[test]
    fn a_design_offering_no_slots_reports_refused_rather_than_an_empty_success() {
        let json = report_json(&[], Boundary { x0: 0, y0: 0, x1: 100, y1: 100 }, 1000, &[]);
        assert!(json.contains("\"status\": \"refused\""));
        assert!(json.contains("\"slots_total\": 0"));
    }
}

#[cfg(test)]
mod pin_tests {
    use super::{describe, PIN_TOKEN};

    #[test]
    fn the_descriptor_reports_the_pin_this_binary_was_built_against() {
        let d = describe();
        assert!(
            !d.contains(PIN_TOKEN),
            "the pin placeholder survived into the output -- the substitution did not run"
        );
        let v: serde_json::Value =
            serde_json::from_str(&d).expect("the descriptor is still valid JSON once filled in");
        assert_eq!(
            v["openroad_pin"], super::CRATE_PIN,
            "the descriptor must report the pin this binary was actually built against"
        );
        assert_eq!(super::CRATE_PIN.len(), 40, "a full commit SHA, not an abbreviation");
    }

    /// ⛔ The whole point of inheriting the pin is that no engine carries one of its own.
    #[test]
    fn no_sha_is_hardcoded_anywhere_in_the_descriptor() {
        let raw = super::DESCRIBE;
        for tok in raw.split(|c: char| !c.is_ascii_hexdigit()) {
            assert!(
                tok.len() < 40,
                "{tok} looks like a hardcoded commit -- use the {PIN_TOKEN} placeholder"
            );
        }
    }
}

#[cfg(test)]
mod maturity_guard {
    //! ⛔ **`maturity` is a CLOSED ENUM of three** — `discovered`, `structured`,
    //! `workflow-validated` — and an unrecognised word is not a modest claim, it is a DISCARDED
    //! RESULT. `Maturity::parse` returns `None`, the consumer treats the engine as `discovered`,
    //! `can_assert()` is false, and the verdict is suppressed to `unknown` however well-formed
    //! the assertion is. The JSON schema's `enum` rejects it too.
    //!
    //! ⚠️ **Four engines shipped an invalid one at once** — `ppl`, `pad` and `dpl` said `partial`,
    //! `pdn` said `correlated` — each chosen to sound honest about incompleteness, each silently
    //! throwing its own verdict away. None of the four had a test on it.
    //!
    //! 🔑 **The rung is about the shape of the EVIDENCE, not feature completeness.** What is
    //! unbuilt belongs in `provenance_limitations`, which is required and can carry nuance a
    //! one-word rung cannot. `workflow-validated` additionally needs a pinned design IN THIS
    //! REPO that the suite runs end to end and asserts against.
    use super::DESCRIBE;

    #[test]
    fn maturity_is_one_of_the_three_legal_rungs() {
        let v: serde_json::Value =
            serde_json::from_str(DESCRIBE).expect("the descriptor is valid JSON");
        let m = v["maturity"].as_str().unwrap_or_default().to_string();
        assert!(["discovered", "structured", "workflow-validated"].contains(&m.as_str()),
                "`{m}` is not a legal maturity; an unrecognised one suppresses the verdict");
        assert!(!v["provenance_limitations"].as_array().expect("required").is_empty(),
                "provenance_limitations is required and states what the hash does not cover");
    }
}

#[cfg(test)]
mod descriptor_args_tests {
    use super::DESCRIBE;
    /// Every advertised input reaches the command line, and every command-line token is
    /// advertised -- for the main operation and each `tools[]` entry. `vyges mcp` builds the
    /// argv from these and refuses an argument the template would drop; 0.1.51 shipped inputs
    /// that never arrived (ifp's die area, ppl's layers, fin's rules).
    #[test]
    fn every_advertised_input_reaches_the_command_line() {
        use std::collections::BTreeSet;
        let d: serde_json::Value = serde_json::from_str(DESCRIBE).expect("descriptor is valid JSON");
        let mut ops = vec![d.clone()];
        ops.extend(d["tools"].as_array().cloned().unwrap_or_default());
        for op in &ops {
            let inv = &op["invocation"];
            let mut used: BTreeSet<String> = inv["args_template"]
                .as_array()
                .expect("an args_template")
                .iter()
                .filter_map(|t| t.as_str())
                .filter(|t| t.starts_with('{') && t.ends_with('}'))
                .map(|t| t[1..t.len() - 1].to_string())
                .collect();
            if let Some(o) = inv.get("optional") {
                let o = o.as_array().expect("invocation.optional is an array of {arg, flag}");
                used.extend(o.iter().map(|x| x["arg"].as_str().expect("an arg").to_string()));
            }
            let props: BTreeSet<String> = op["inputs"]["properties"]
                .as_object()
                .expect("inputs.properties")
                .keys()
                .cloned()
                .collect();
            assert_eq!(used, props, "{}: command line vs advertised inputs", op["name"]);
        }
    }
}
