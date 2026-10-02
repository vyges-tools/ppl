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
use vyges_ppl::{collect_constraints, define_slots, interval_from_rect, is_blocked,
    annealing, place_with_constraints, polygon, run_round, sort_constraints, toplayer, Boundary,
    Edge, Group, LayerTracks, Params, Pin, Placement, Slot, TrackPattern, SLOTS_PER_SECTION};

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
  --hor-layers L,…       layers carrying pins on the LEFT and RIGHT edges (required)
  --ver-layers L,…       layers carrying pins on the BOTTOM and TOP edges (required)
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
      "ANNEALING (`--annealing`) reproduces the reference EXACTLY, including its random stream: the reference draws from Boost, whose algorithms are specified and portable, so the engine, both distributions and the shuffle are reimplemented bit-for-bit. Verified by comparing all 86000 perturbations of a run against the reference own debug trace -- every cost and delta identical. Scope: plain pins only. A design with groups, constraints or mirrored pairs is REFUSED rather than annealed, because those add move types whose draws would desynchronise the stream and yield a plausible wrong answer.",
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
      "It reports the chosen pin positions; it does not yet write them to the database, because the pin RECTANGLE depends on pin length and extension handling that is not built.",
      "Assignment is optimal WITHIN a section and greedy BETWEEN sections: pins are routed to the cheapest section with room, and only then matched optimally inside it. This is the reference decomposition, not an approximation introduced here.",
      "Cost is half-perimeter wirelength over the net bounding box. A net whose driver or loads are unplaced uses the die centre for them, as the reference does.",
      "The optimal assignment COST is unique but the optimal PAIRING is not: where two pairings cost the same, this and the reference may place two pins in swapped slots and both be correct. Compare total cost before treating a difference as a defect -- `--evaluate` scores a reference placement under the same cost model for exactly this.",
      "Slots are generated on the DIE boundary from each layer's routing track patterns, minus corner avoidance, minus half the pin width at each end, minus the requested minimum distance.",
      "PARTIAL: slot availability accounts for excluded regions and for fixed ports' metal, but NOT yet for macros or routing obstructions. Where a macro abuts the boundary, availability remains optimistic.",
      "The default corner avoidance is resolved once from a layer's FIRST track pattern and reused for the rest, which is upstream's behavior and is observable on layers carrying mixed-pitch patterns. Reproduced deliberately.",
      "`-min_distance_in_tracks` with a distance of 0 is a division by zero upstream; here it keeps every candidate. A deliberate divergence, on an input that has no defined meaning.",
      "Written against the upstream ppl sources at pin @OPENROAD_PIN@. The algorithm is reimplemented from the published behavior, not transliterated."
  ],
  "invocation": {
    "args_template": ["place-pins", "{odb}"],
    "optional": [ { "arg": "out", "flag": "-o" } ],
    "emits_json": true
  },
  "inputs": {
    "type": "object",
    "required": ["odb", "hor_layers", "ver_layers"],
    "properties": {
      "odb": { "type": "string", "description": "path to the design database (.odb)" },
      "hor_layers": { "type": "string", "description": "comma-separated layers for the left/right edges" },
      "ver_layers": { "type": "string", "description": "comma-separated layers for the bottom/top edges" },
      "min_distance": { "type": "string", "description": "minimum pin spacing in microns" },
      "corner_avoidance": { "type": "string", "description": "clearance from each corner in microns" },
      "out": { "type": "string", "description": "write the report to FILE instead of stdout" }
    }
  },
  "consumes": ["odb"],
  "produces": [],
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

/// Read one layer's contribution to a slot list.
///
/// ⚠️ Two accessors on the layer are a letter apart and both bridged: upstream takes
/// `getWidth()`, **not** `getMinWidth()`. Picking the other one compiles and quietly misplaces the
/// outermost slot on every edge, so it is worth naming here rather than in a commit message.
///
/// The axis follows the edge, not the layer's own routing direction: pins on the bottom and top
/// run down the **x** track patterns, pins on the left and right down the **y** ones.
fn read_layer(db: &Db, layer: &str, vertical_pins: bool) -> Result<LayerTracks, String> {
    let (x, y) = db
        .track_patterns(layer)
        .map_err(|e| format!("layer `{layer}`: cannot read track patterns: {e}"))?;
    let chosen = if vertical_pins { x } else { y };
    Ok(LayerTracks {
        layer: layer.to_string(),
        patterns: chosen
            .into_iter()
            .map(|(origin, count, step)| TrackPattern { origin, count, step })
            .collect(),
        min_width: db.layer_get_width(layer) as i32,
    })
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

/// Everything both verbs need out of the database and the command line.
struct Prepared {
    db: Db,
    boundary: Boundary,
    dbu: i32,
    slots: Vec<Slot>,
    trackless: Vec<String>,
    exclusions: usize,
    /// For a **polygon** die: which boundary segment each slot came from. `None` for the ordinary
    /// four-edge case. Sections are cut per segment, so this is not optional detail — it is how a
    /// polygon's slot list is divided at all.
    on_segment: Option<Vec<usize>>,
}

fn prepare(opts: &Opts) -> Result<Prepared, String> {
    let hor = layer_list(opts, "hor-layers");
    let ver = layer_list(opts, "ver-layers");
    if hor.is_empty() && ver.is_empty() {
        return Err("--hor-layers and/or --ver-layers is required".into());
    }

    let db = Db::open(&opts.odb).map_err(|e| format!("cannot read {}: {e}", opts.odb))?;
    let dbu = db.dbu_per_micron();
    if dbu <= 0 {
        return Err("no DBU scale".into());
    }

    // Upstream reaches this through a type named `Core`, but it is built from the DIE area — see
    // `Boundary`. Reading the core area here would inset every pin by the core-to-die margin.
    let boundary = Boundary {
        x0: db.block_get_die_area_x_min(),
        y0: db.block_get_die_area_y_min(),
        x1: db.block_get_die_area_x_max(),
        y1: db.block_get_die_area_y_max(),
    };

    let params = Params {
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

    let read = |names: &[String], vertical: bool| -> Result<Vec<LayerTracks>, String> {
        names.iter().map(|l| read_layer(&db, l, vertical)).collect()
    };
    let ver_tracks = read(&ver, true)?;
    let hor_tracks = read(&hor, false)?;

    // What the design has already ruled out. Both are read from the block, like constraints:
    // regions the design excluded outright, and the metal of ports already fixed in place.
    let die = (boundary.x0, boundary.y0, boundary.x1, boundary.y1);
    let exclusions: Vec<vyges_ppl::Interval> = db
        .blocked_regions_for_pins()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|r| interval_from_rect(r, die))
        .collect();
    // `IOPlacer::getBlockedRegions`: the placed macros (every layer), then each routing obstruction
    // and each power-grid shape (its own layer) — see `vyges_ppl::blocked`.
    let mut exclusions = exclusions;
    for inst in db.block_get_insts() {
        let master = db.inst_get_master(&inst);
        if !(db.master_is_block(&master) && db.inst_is_placed(&inst)) {
            continue;
        }
        if let Ok(b) = db.inst_bbox(&inst) {
            if let [x0, y0, x1, y1] = b[..] {
                let clipped = (x0.max(die.0), y0.max(die.1), x1.min(die.2), y1.min(die.3));
                exclusions.extend(vyges_ppl::blocked::find_blocked_intervals(die, clipped));
            }
        }
    }
    let mfg = db.tech_get_manufacturing_grid();
    // `set_pin_length -hor_length/-ver_length` (microns, to database units as `microns_to_dbu`
    // does): it replaces the pin's depth, so it also deepens every keep-out the pin is measured
    // against. Not stored in the database, so it arrives as a flag.
    let user_length = |key: &str| -> Result<Option<i32>, String> { Ok(Some(microns(opts, key, dbu, -1)?).filter(|v| *v != -1)) };
    let (hor_length, ver_length) = (user_length("hor-length")?, user_length("ver-length")?);
    let directions: std::collections::HashMap<String, String> = db.layers_with_direction().unwrap_or_default().into_iter().collect();
    // `IOPlacer::computePinKeepout`: a shape grown by the pin a slot on its layer would hold, plus
    // the spacing the layer asks between the two (`computeShapeSpacing`).
    let compute_pin_keepout = |layer: &str, b: (i32, i32, i32, i32)| {
        let vertical = directions.get(layer).is_some_and(|d| d == "VERTICAL");
        let multiplier = if vertical { params.thickness_multiplier_v } else { params.thickness_multiplier_h };
        let user_length = if vertical { ver_length } else { hor_length };
        let (half_width, height) = vyges_ppl::blocked::pin_size(db.layer_get_width(layer) as i32, db.layer_get_area(layer).unwrap_or(0), multiplier, user_length, mfg);
        let shape_width = (b.2 - b.0).min(b.3 - b.1).max(2 * half_width);
        let mut spacing = db.layer_get_spacing_width_length(layer, shape_width, height);
        if spacing == 0 {
            spacing = db.layer_get_width(layer) as i32;
        }
        vyges_ppl::blocked::pin_keepout(b, vertical, half_width, height, spacing)
    };
    let mut layered: Vec<(vyges_ppl::Interval, String)> = Vec::new();
    // `IOPlacer::excludeBoundaryShape`.
    let mut exclude_boundary_shape = |layer_no: i64, b: (i32, i32, i32, i32)| {
        let layer = db.layer_name_by_number(layer_no);
        if db.layer_get_routing_level(&layer) == 0 {
            return;
        }
        let vertical = directions.get(&layer).is_some_and(|d| d == "VERTICAL");
        if !(ver.is_empty() && hor.is_empty()) && !(if vertical { &ver } else { &hor }).contains(&layer) {
            return;
        }
        let keepout = compute_pin_keepout(&layer, b);
        for i in vyges_ppl::blocked::boundary_shape_intervals(die, keepout, vertical) {
            layered.push((i, layer.clone()));
        }
    };
    for (l, x0, y0, x1, y1) in db.pin_obstruction_boxes().unwrap_or_default() {
        exclude_boundary_shape(l, (x0, y0, x1, y1));
    }
    for (l, x0, y0, x1, y1) in db.swire_boxes().unwrap_or_default() {
        exclude_boundary_shape(l, (x0, y0, x1, y1));
    }
    // `initNetlist`: a fixed pin's shapes are kept PADDED by `computePinKeepout`, "so the pins
    // created near them keep the min spacing" — every layer, no layer filter.
    let fixed_shapes: Vec<(String, i32, i32, i32, i32)> = db
        .fixed_bterm_shapes()
        .unwrap_or_default()
        .into_iter()
        .map(|(l, x0, y0, x1, y1)| {
            let layer = db.layer_name_by_number(l);
            let k = compute_pin_keepout(&layer, (x0, y0, x1, y1));
            (layer, k.0, k.1, k.2, k.3)
        })
        .collect();

    // A rectilinear die takes a different path entirely: its boundary is a list of segments, not
    // four named edges. The point count is the reference's own branch condition.
    let outline = db.die_area_polygon().unwrap_or_default();
    let (mut slots, on_segment) = if polygon::is_polygon(&outline) {
        let (s, seg) =
            polygon::define_slots(&outline, &ver_tracks, &hor_tracks, &params, dbu, &|_, _, _| false);
        (s, Some(seg))
    } else {
        // A slot's edge is not passed to the callback, so blocking is resolved per edge afterwards
        // — `define_slots` knows the edge, and it is what decides which axis an exclusion compares.
        (define_slots(&ver_tracks, &hor_tracks, boundary, &params, dbu, &|_, _, _| false), None)
    };
    for s in slots.iter_mut() {
        s.blocked = is_blocked(s.x, s.y, &s.layer, s.edge, &exclusions, &fixed_shapes, &layered);
    }
    let trackless = ver_tracks
        .iter()
        .chain(hor_tracks.iter())
        .filter(|t| t.patterns.is_empty())
        .map(|t| t.layer.clone())
        .collect();

    Ok(Prepared { exclusions: exclusions.len(), on_segment, db, boundary, dbu, slots, trackless })
}

/// The IO pins to place, and where each one's net actually pulls it.
///
/// Follows the database's own port order, which is what makes the assignment reproducible: pins
/// are considered in this order when competing for a section, so a different order is a different
/// placement.
///
/// Two exclusions, both of which change the answer if missed. A port whose placement is already
/// **fixed** is not ours to move. A port with **no net** has nothing to be near.
fn read_pins(db: &Db, center: (i32, i32)) -> Vec<Pin> {
    let mut out = Vec::new();
    for name in db.bterm_names() {
        if is_fixed(&db.bterm_get_first_pin_placement_status(&name)) {
            continue;
        }
        let net = db.bterm_net(&name);
        if net.is_empty() {
            continue;
        }
        let mut sinks = Vec::new();
        for iterm in db.net_iterms(&net) {
            // "instance/terminal", and an instance name may itself contain a separator, so the
            // split has to come from the right.
            let Some((inst, mterm)) = iterm.rsplit_once('/') else { continue };
            // An unplaced instance has no position to be near, so it is treated as sitting at the
            // middle of the die — which leaves the pin free to go wherever else its net wants.
            if is_unplaced(&db.inst_get_placement_status(inst)) {
                sinks.push(center);
            } else if let Some(p) = db.iterm_avg_xy(inst, mterm) {
                sinks.push(p);
            }
        }
        out.push(Pin { name, sinks, mirror: None });
    }
    out
}

/// odb reports placement status as a name. `isFixed` is true for exactly these three.
fn is_fixed(status: &str) -> bool {
    matches!(status, "LOCKED" | "FIRM" | "COVER")
}

/// ...and these two mean "has no meaningful position yet".
fn is_unplaced(status: &str) -> bool {
    matches!(status, "NONE" | "UNPLACED")
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
            eprintln!("vyges-ppl: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let p = match prepare(&opts) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("vyges-ppl: {e}");
            return ExitCode::from(2);
        }
    };

    emit_slot_events(&p.slots, &p.trackless);
    let report = report_json(&p.slots, p.boundary, p.dbu, &p.trackless);
    if let Err(e) = write_report(&opts, &report) {
        eprintln!("vyges-ppl: {e}");
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
            eprintln!("vyges-ppl: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let p = match prepare(&opts) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("vyges-ppl: {e}");
            return ExitCode::from(2);
        }
    };
    let per_section = match opts.get("slots-per-section") {
        None => SLOTS_PER_SECTION,
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!("vyges-ppl: --slots-per-section wants a positive whole number");
                return ExitCode::from(2);
            }
        },
    };

    let center = ((p.boundary.x0 + p.boundary.x1) / 2, (p.boundary.y0 + p.boundary.y1) / 2);
    let mut pins = read_pins(&p.db, center);

    // Mirrored pairs. Only ONE of a pair competes for a slot — the one the database records as
    // having a mirrored partner; the other is placed at the reflected position of whatever the
    // first gets. Giving both a say would let the two halves disagree about where the pair goes.
    for i in 0..pins.len() {
        if !p.db.bterm_has_mirrored_b_term(&pins[i].name) {
            continue;
        }
        let partner = p.db.bterm_get_mirrored_b_term(&pins[i].name);
        pins[i].mirror = pins.iter().position(|q| q.name == partner);
    }
    let pins = pins;

    // Constraints are read back from the DATABASE, not from this command line: the design writes
    // them onto its ports, so a placer told them separately could disagree with what it reads.
    let die = (p.boundary.x0, p.boundary.y0, p.boundary.x1, p.boundary.y1);
    let regions: Vec<Option<(i32, i32, i32, i32)>> = pins
        .iter()
        .map(|pin| p.db.bterm_constraint_region(&pin.name).ok().flatten())
        .collect();
    let mirrored: Vec<bool> = pins.iter().map(|pin| p.db.bterm_is_mirrored(&pin.name)).collect();
    // Pin groups, likewise stored on the block rather than passed in. Named ports are resolved to
    // pin indices here; a group naming a port that is fixed or netless simply loses that member.
    let index_of = |name: &str| pins.iter().position(|p| p.name == name);
    let groups: Vec<Group> = p
        .db
        .bterm_groups()
        .unwrap_or_default()
        .into_iter()
        .map(|(names, ordered)| Group {
            pins: names.iter().filter_map(|n| index_of(n)).collect(),
            ordered,
        })
        .filter(|g| !g.pins.is_empty())
        .collect();

    let mut constraints = collect_constraints(&pins, die, &|i| regions[i], &|i| mirrored[i]);
    // Where two regions share slots, whoever is served first takes them, so the order is part of
    // the answer rather than a detail of iteration.
    sort_constraints(&mut constraints, &p.slots, per_section);

    // ── Top layer ────────────────────────────────────────────────────────────────────────────
    // Pins sent to the top layer are placed on their own lattice, not on the boundary, so they are
    // taken out of the edge problem entirely rather than competing in it.
    let (top_slots, top_pins) = read_top_layer(&p.db, die, &pins);
    let top_set: std::collections::BTreeSet<usize> = top_pins.iter().map(|&(i, _)| i).collect();
    let (top_placed, mut unplaced) = place_top_layer(&top_slots, &pins, &top_pins, per_section);

    // ── The boundary ─────────────────────────────────────────────────────────────────────────
    let boundary_pins: Vec<Pin> = pins
        .iter()
        .enumerate()
        .map(|(i, pin)| {
            // A top-layer pin is removed from the edge problem by giving it no name to match and
            // no constraint; it is simpler and safer to filter it where the two lists are joined.
            let _ = i;
            pin.clone()
        })
        .collect();
    let edge_constraints: Vec<vyges_ppl::Constraint> = constraints
        .iter()
        .map(|c| vyges_ppl::Constraint {
            interval: c.interval,
            pins: c.pins.iter().copied().filter(|i| !top_set.contains(i)).collect(),
        })
        .filter(|c| !c.pins.is_empty())
        .collect();
    let edge_groups: Vec<Group> = groups
        .iter()
        .filter(|g| !g.pins.iter().any(|i| top_set.contains(i)))
        .cloned()
        .collect();

    // ── Annealing ────────────────────────────────────────────────────────────────────────────
    // A wholly different optimiser, so it REPLACES the boundary placement rather than adjusting
    // it. It refuses anything needing move types it does not implement: a partial implementation
    // would desynchronise its random stream and produce a plausible wrong answer, which is worse
    // than declining.
    let annealed = if opts.annealing {
        let cfg = match anneal_config(&opts) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("vyges-ppl: {e}");
                return ExitCode::from(2);
            }
        };
        let outcome = if constraints.is_empty() && groups.is_empty() {
            annealing::run(&p.slots, &pins, die, &cfg)
        } else {
            Err(annealing::Unsupported::NeedsExtraMoves)
        };
        match outcome {
            Ok(v) => Some(v),
            Err(why) => {
                eprintln!(
                    "vyges-ppl: cannot anneal this design ({why:?}); \
                     re-run without --annealing to place it by matching"
                );
                return ExitCode::from(1);
            }
        }
    } else {
        None
    };

    let (edge_placed, edge_unplaced) = match (&annealed, &p.on_segment) {
        (Some(v), _) => (v.clone(), Vec::new()),
        // A polygon die's sections come from its segments. Edge-named region constraints have no
        // meaning against an arbitrary outline, so they are reported and not applied — quietly
        // reinterpreting them against the bounding box would satisfy a constraint the design did
        // not ask for.
        (None, Some(on_segment)) => {
            if !edge_constraints.is_empty() {
                vyges_events::emit(
                    &vyges_events::Event::new(
                        "vyges-ppl",
                        vyges_events::Severity::Warn,
                        format!(
                            "{} edge region constraint(s) ignored: this die is a polygon and has \
                             no named edges",
                            edge_constraints.len()
                        ),
                    )
                    .with_code("PPL-POLYGON-NO-CONSTRAINTS"),
                );
            }
            let mut slots: Vec<Slot> = p.slots.clone();
            let mut sections = polygon::create_sections(&slots, on_segment, per_section);
            let which_pins: Vec<usize> =
                (0..pins.len()).filter(|i| !top_set.contains(i)).collect();
            let which_groups: Vec<usize> = (0..edge_groups.len()).collect();
            run_round(
                &mut slots,
                &mut sections,
                &boundary_pins,
                &edge_groups,
                &which_groups,
                &which_pins,
                per_section,
                die,
            )
        }
        (None, None) => {
            // Only the pins this boundary owns. A top-layer pin left in the pool would take a
            // boundary slot and then be discarded, quietly pushing a real boundary pin onto a
            // worse one.
            let eligible: std::collections::BTreeSet<usize> =
                (0..pins.len()).filter(|i| !top_set.contains(i)).collect();
            place_with_constraints(
                &p.slots,
                &boundary_pins,
                &edge_groups,
                &edge_constraints,
                &eligible,
                per_section,
                die,
            )
        }
    };
    unplaced.extend(edge_unplaced.into_iter().filter(|i| !top_set.contains(i)));

    // Resolve both lists to coordinates. Slot indices mean different things in each, so they are
    // never mixed — only the resolved positions are.
    let mut placed: Vec<Placed> = Vec::new();
    placed.extend(top_placed.iter().map(|x| Placed::of(x, &top_slots, p.slots.len())));
    placed.extend(
        edge_placed
            .iter()
            .filter(|x| !top_set.contains(&x.pin))
            .map(|x| Placed::of(x, &p.slots, 0)),
    );
    placed.sort_by_key(|x| x.pin);
    unplaced.sort_unstable();
    unplaced.dedup();

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
    emit_constraint_events(&constraints, &groups, &pins);
    emit_placement_events(&pins, &placed, &unplaced);

    let reference = match opts.get("evaluate") {
        None => None,
        Some(path) => match score_reference(path, &pins) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("vyges-ppl: {e}");
                return ExitCode::from(2);
            }
        },
    };

    let outline = p.db.die_area_polygon().unwrap_or_default();
    let report =
        placement_json(&pins, &groups, &placed, &unplaced, p.boundary, &outline, p.dbu, reference);
    if let Err(e) = write_report(&opts, &report) {
        eprintln!("vyges-ppl: {e}");
        return ExitCode::from(2);
    }
    if unplaced.is_empty() && !pins.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
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

/// Read the top-layer grid and build its lattice, if the design declares one.
///
/// Obstructions are gathered from the three sources that matter on that layer and no others:
/// routing blockages, the power grid, and ports already fixed. All are filtered to the grid's own
/// layer — metal on a different layer is not in the way.
fn read_top_layer(
    db: &Db,
    die: (i32, i32, i32, i32),
    pins: &[Pin],
) -> (Vec<Slot>, Vec<(usize, (i32, i32, i32, i32))>) {
    let Ok(Some(g)) = db.bterm_top_layer_grid() else { return (Vec::new(), Vec::new()) };
    if !g.region_is_rect || g.pin_width <= 0 {
        // A non-rectangular grid is not handled by the reference either.
        return (Vec::new(), Vec::new());
    }
    let grid = toplayer::Grid {
        layer: g.layer.clone(),
        x_step: g.x_step,
        y_step: g.y_step,
        pin_width: g.pin_width,
        pin_height: g.pin_height,
        keepout: g.keepout,
        region: g.region,
    };

    let on_layer = |v: Vec<(i64, i32, i32, i32, i32)>| -> Vec<(i32, i32, i32, i32)> {
        v.into_iter()
            .filter(|&(l, ..)| db.layer_name_by_number(l) == g.layer)
            .map(|(_, x0, y0, x1, y1)| (x0, y0, x1, y1))
            .collect()
    };
    let mut obstructions = on_layer(db.obstruction_boxes().unwrap_or_default());
    obstructions.extend(on_layer(db.swire_boxes().unwrap_or_default()));
    obstructions.extend(on_layer(db.fixed_bterm_shapes().unwrap_or_default()));

    let slots = toplayer::slots(&grid, die, &obstructions);
    let claimed = toplayer::constrained_pins(pins, &|i| {
        db.bterm_constraint_region(&pins[i].name).ok().flatten()
    });
    (slots, claimed)
}

/// Place the top-layer pins on their lattice: sections per region, then optimal matching.
fn place_top_layer(
    slots: &[Slot],
    pins: &[Pin],
    claimed: &[(usize, (i32, i32, i32, i32))],
    per_section: usize,
) -> (Vec<Placement>, Vec<usize>) {
    if claimed.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if slots.is_empty() {
        // Pins were sent to a top layer the design never defined a grid for.
        return (Vec::new(), claimed.iter().map(|&(i, _)| i).collect());
    }
    let mut slots = slots.to_vec();
    let mut placed = Vec::new();
    let mut unplaced = Vec::new();
    // One round per distinct region, so pins constrained to different boxes do not compete.
    let mut by_region: std::collections::BTreeMap<(i32, i32, i32, i32), Vec<usize>> =
        Default::default();
    for &(i, r) in claimed {
        by_region.entry(r).or_default().push(i);
    }
    for (region, members) in by_region {
        let mut sections = toplayer::sections_for(&slots, region, per_section);
        unplaced.extend(vyges_ppl::assign_pins_to_sections(
            pins,
            &members,
            &mut sections,
            (0, 0, 0, 0),
        ));
        let (p, u) = vyges_ppl::solve_sections(
            &mut slots,
            &sections,
            pins,
            &[],
            &[],
            // A lattice has no opposite edge, so no mirroring can apply here.
            (0, 0, 0, 0),
        );
        placed.extend(p);
        unplaced.extend(u);
    }
    (placed, unplaced)
}

/// One pin's chosen position, with its slot already resolved.
///
/// The two placement surfaces index their own slot lists, so a raw slot number means nothing on
/// its own. Resolving here — and offsetting the top-layer numbers past the boundary's — keeps them
/// distinguishable without ever mixing the two index spaces.
struct Placed {
    pin: usize,
    x: i32,
    y: i32,
    layer: String,
    edge: Edge,
    slot: usize,
}

impl Placed {
    fn of(p: &Placement, slots: &[Slot], offset: usize) -> Placed {
        let s = &slots[p.slot];
        Placed {
            pin: p.pin,
            x: s.x,
            y: s.y,
            layer: s.layer.clone(),
            edge: s.edge,
            slot: p.slot + offset,
        }
    }
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

fn placement_json(
    pins: &[Pin],
    groups: &[Group],
    placed: &[Placed],
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
fn emit_placement_events(pins: &[Pin], placed: &[Placed], unplaced: &[usize]) {
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
        .with_code("PPL-PLACED"),
    );
}

fn main() -> ExitCode {
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
            eprintln!("vyges-ppl: unknown command `{other}`\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
