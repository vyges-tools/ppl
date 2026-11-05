// SPDX-License-Identifier: Apache-2.0
//! `IOPlacer::runHungarianMatching`: the pins read from the block, the run's configuration
//! reported, the top layer and then the boundary placed by matching, and the result committed to
//! the database. Each stage is its own function, in the reference's order; the command line
//! composes them with its own options (and its annealing alternative), the global placer's
//! `-place_ios` calls [`run_hungarian_matching`].

use crate::{place_with_constraints, polygon, run_round, toplayer, Boundary, Constraint, Edge, Group, Pin, Placement, Slot};
use vyges_opendb::Db;

/// The IO pins to place, and where each one's net actually pulls it.
///
/// Follows the database's own port order, which is what makes the assignment reproducible: pins
/// are considered in this order when competing for a section, so a different order is a different
/// placement.
///
/// Two exclusions, both of which change the answer if missed. A port whose placement is already
/// **fixed** is not ours to move. A port with **no net** has nothing to be near.
///
/// Rule (`IOPlacer::initNetlist`, `IOPin`'s position): with `minimize_displacement` each pin starts
/// where `getFirstPinLocation` puts it — the centre of its first placed pin box, the origin when
/// it has none — and that is what its cost is measured from. Without it no pin carries one.
pub fn read_pins(db: &Db, center: (i32, i32), minimize_displacement: bool) -> Vec<Pin> {
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
        let initial = initial_position(minimize_displacement, db.bterm_first_pin_location(&name));
        out.push(Pin { name, sinks, mirror: None, initial });
    }
    out
}

/// Rule (`IOPin`'s initial position, from `dbBTerm::getFirstPinLocation`): the centre of the
/// port's first placed pin box, and the ORIGIN when it has none — not "no position". Only a
/// displacement-minimising run carries it; otherwise the pin has none.
pub fn initial_position(minimize_displacement: bool, first_pin: Option<(i32, i32)>) -> Option<(i32, i32)> {
    minimize_displacement.then(|| first_pin.unwrap_or((0, 0)))
}

/// odb reports placement status as a name. `isFixed` is true for exactly these three.
pub fn is_fixed(status: &str) -> bool {
    matches!(status, "LOCKED" | "FIRM" | "COVER")
}

/// ...and these two mean "has no meaningful position yet".
pub fn is_unplaced(status: &str) -> bool {
    matches!(status, "NONE" | "UNPLACED")
}


/// Read the top-layer grid and build its lattice, if the design declares one.
///
/// Obstructions are gathered from the three sources that matter on that layer and no others:
/// routing blockages, the power grid, and ports already fixed. All are filtered to the grid's own
/// layer — metal on a different layer is not in the way.
#[allow(clippy::type_complexity)]
pub fn read_top_layer(
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

    // `filterObstructedSlotsForTopLayer`, in its order: routing obstructions (system-reserved
    // ones too — "the only filter for slots outside of polygon dies" — but NOT `+ FILLS` /
    // `+ SLOTS`, which "still allow routing metal over them"), with their DEF rules; the power
    // grid (`{rect}`, no rules); then every bpin whose OWN status is fixed, with its rules.
    let mut shapes: Vec<vyges_opendb::RuledShape> = db
        .obstruction_shapes_with_rules()
        .unwrap_or_default()
        .into_iter()
        .filter(|o| !(o.fill || o.slot))
        .map(|o| o.shape)
        .collect();
    shapes.extend(db.swire_boxes().unwrap_or_default().into_iter().map(|(l, x0, y0, x1, y1)| {
        vyges_opendb::RuledShape { layer: l, rect: (x0, y0, x1, y1), min_spacing: -1, effective_width: -1 }
    }));
    shapes.extend(db.bpin_shapes_with_rules().unwrap_or_default().into_iter().map(|b| b.shape));
    // Each keeps out by the grid's keepout FLOORED at the spacing it asks (`computeShapeSpacing`,
    // the pin's smaller dimension as its width and its larger as the run length).
    // `VYGES_PPL_TOPLAYER_TRACE`: the reference's `VYGPPL topobs` / `topslot` lines
    // (`instruments/ppl/ppl-toplayer-patch.py`), so the two diff directly.
    let trace = std::env::var_os("VYGES_PPL_TOPLAYER_TRACE").is_some();
    let (pin_min, pin_max) = (g.pin_width.min(g.pin_height), g.pin_width.max(g.pin_height));
    let layer_width = db.layer_get_width(&g.layer) as i32;
    let obstructions: Vec<toplayer::Obstruction> = shapes
        .iter()
        .filter(|s| db.layer_name_by_number(s.layer) == g.layer)
        .map(|s| {
            let b = s.rect;
            let spacing = crate::blocked::shape_spacing(b, s.min_spacing, s.effective_width, pin_min, pin_max, layer_width, |w, l| {
                db.layer_get_spacing_width_length(&g.layer, w, l)
            });
            let keepout = grid.keepout.max(spacing);
            if trace {
                let k = toplayer::keepout_rect(&grid, b, keepout);
                eprintln!(
                    "VYGPPL topobs|{} {} {} {}|{}|{}|{spacing}|{keepout}|{} {} {} {}",
                    b.0, b.1, b.2, b.3, s.min_spacing, s.effective_width, k.0, k.1, k.2, k.3
                );
            }
            (b, keepout)
        })
        .collect();

    let slots = toplayer::slots(&grid, die, &obstructions);
    if trace {
        for (i, s) in slots.iter().enumerate() {
            eprintln!("VYGPPL topslot|{i}|{} {}|{}", s.x, s.y, u8::from(s.blocked));
        }
    }
    let claimed = toplayer::constrained_pins(pins, &|i| {
        db.bterm_constraint_region(&pins[i].name).ok().flatten()
    });
    (slots, claimed)
}

/// Place the top-layer pins on their lattice: sections per region, then optimal matching.
#[allow(clippy::type_complexity)]
pub fn place_top_layer(
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
        unplaced.extend(crate::assign_pins_to_sections(
            pins,
            &members,
            &mut sections,
            &slots,
            (0, 0, 0, 0),
        ));
        let (p, u) = crate::solve_sections(
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
#[derive(Debug, Clone, PartialEq)]
pub struct Placed {
    pub pin: usize,
    pub x: i32,
    pub y: i32,
    pub layer: String,
    pub edge: Edge,
    /// Into the boundary's slot list; a top-layer pin's is offset past its end.
    pub slot: usize,
}

impl Placed {
    pub fn of(p: &Placement, slots: &[Slot], offset: usize) -> Placed {
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


/// The pins, their groups and their region constraints, as `initNetlistAndCore`,
/// `initMirroredPins` and `initConstraints` leave them.
pub struct Netlist {
    pub pins: Vec<Pin>,
    pub groups: Vec<Group>,
    /// In the order they are served ([`crate::sort_constraints`]).
    pub constraints: Vec<Constraint>,
    /// The same constraints in the order they are READ (`getConstraintsFromDB`), before that
    /// sort: what an annealing run uses, since `initConstraints(true)` does not sort.
    pub unsorted_constraints: Vec<Constraint>,
}

/// `initNetlistAndCore` + `initMirroredPins` + `initConstraints`: the pins in port order, each
/// mirrored pair resolved, then the groups and region constraints the design stored on the block.
pub fn init_netlist(db: &Db, die: (i32, i32, i32, i32), slots: &[Slot], per_section: usize, minimize_displacement: bool) -> Netlist {
    let center = ((die.0 + die.2) / 2, (die.1 + die.3) / 2);
    let mut pins = read_pins(db, center, minimize_displacement);

    // Mirrored pairs. Only ONE of a pair competes for a slot — the one the database records as
    // having a mirrored partner; the other is placed at the reflected position of whatever the
    // first gets. Giving both a say would let the two halves disagree about where the pair goes.
    for i in 0..pins.len() {
        if !db.bterm_has_mirrored_b_term(&pins[i].name) {
            continue;
        }
        let partner = db.bterm_get_mirrored_b_term(&pins[i].name);
        pins[i].mirror = pins.iter().position(|q| q.name == partner);
    }

    // Constraints are read back from the DATABASE, not from a command line: the design writes
    // them onto its ports, so a placer told them separately could disagree with what it reads.
    let regions: Vec<Option<(i32, i32, i32, i32)>> = pins.iter().map(|pin| db.bterm_constraint_region(&pin.name).ok().flatten()).collect();
    let mirrored: Vec<bool> = pins.iter().map(|pin| db.bterm_is_mirrored(&pin.name)).collect();
    // Pin groups, likewise stored on the block rather than passed in. Named ports are resolved to
    // pin indices here; a group naming a port that is fixed or netless simply loses that member.
    let index_of = |name: &str| pins.iter().position(|p| p.name == name);
    let groups: Vec<Group> = db
        .bterm_groups()
        .unwrap_or_default()
        .into_iter()
        .map(|(names, ordered)| Group { pins: names.iter().filter_map(|n| index_of(n)).collect(), ordered })
        .filter(|g| !g.pins.is_empty())
        .collect();

    let mut constraints = crate::collect_constraints(&pins, die, &|i| regions[i], &|i| mirrored[i]);
    let unsorted_constraints = constraints.clone();
    // Where two regions share slots, whoever is served first takes them, so the order is part of
    // the answer rather than a detail of iteration.
    crate::sort_constraints(&mut constraints, slots, per_section);
    Netlist { pins, groups, constraints, unsorted_constraints }
}

/// `printConfig`'s inputs: what the run was given, before any pin is assigned.
/// `slots_per_section` is `None` for an annealing run, which has no sections.
pub fn config(db: &Db, slots: &[Slot], top_slots: &[Slot], net: &Netlist, slots_per_section: Option<usize>, minimize_displacement: bool) -> crate::report::Config {
    crate::report::Config {
        available_slots: slots.iter().filter(|s| !s.blocked).count(),
        top_layer_slots: top_slots.len(),
        ios: net.pins.len(),
        ios_without_sink: net.pins.iter().filter(|pin| db.net_iterms(&db.bterm_net(&pin.name)).is_empty()).count(),
        groups: net.groups.len(),
        slots_per_section,
        minimize_displacement,
    }
}

/// The boundary's half of the placement by matching: every pin not sent to the top layer, by
/// region constraint, group and section. Returns the placements (indices into `slots`) and the
/// pins left without a slot.
#[allow(clippy::too_many_arguments)]
pub fn place_boundary(
    slots: &[Slot],
    on_segment: Option<&[usize]>,
    net: &Netlist,
    top_set: &std::collections::BTreeSet<usize>,
    per_section: usize,
    die: (i32, i32, i32, i32),
) -> (Vec<Placement>, Vec<usize>) {
    let pins = &net.pins;
    let edge_constraints: Vec<Constraint> = net
        .constraints
        .iter()
        .map(|c| Constraint { interval: c.interval, pins: c.pins.iter().copied().filter(|i| !top_set.contains(i)).collect() })
        .filter(|c| !c.pins.is_empty())
        .collect();
    let edge_groups: Vec<Group> = net.groups.iter().filter(|g| !g.pins.iter().any(|i| top_set.contains(i))).cloned().collect();
    match on_segment {
        // A polygon die's sections come from its segments. Edge-named region constraints have no
        // meaning against an arbitrary outline, so they are reported and not applied — quietly
        // reinterpreting them against the bounding box would satisfy a constraint the design did
        // not ask for.
        Some(on_segment) => {
            if !edge_constraints.is_empty() {
                vyges_events::emit(
                    &vyges_events::Event::new(
                        "vyges-ppl",
                        vyges_events::Severity::Warn,
                        format!("{} edge region constraint(s) ignored: this die is a polygon and has no named edges", edge_constraints.len()),
                    )
                    .with_code("PPL-POLYGON-NO-CONSTRAINTS"),
                );
            }
            let mut slots: Vec<Slot> = slots.to_vec();
            let mut sections = polygon::create_sections(&slots, on_segment, per_section);
            let which_pins: Vec<usize> = (0..pins.len()).filter(|i| !top_set.contains(i)).collect();
            let which_groups: Vec<usize> = (0..edge_groups.len()).collect();
            run_round(&mut slots, &mut sections, pins, &edge_groups, &which_groups, &which_pins, per_section, die)
        }
        None => {
            // Only the pins this boundary owns. A top-layer pin left in the pool would take a
            // boundary slot and then be discarded, quietly pushing a real boundary pin onto a
            // worse one.
            let eligible: std::collections::BTreeSet<usize> = (0..pins.len()).filter(|i| !top_set.contains(i)).collect();
            place_with_constraints(slots, pins, &edge_groups, &edge_constraints, &eligible, per_section, die)
        }
    }
}

/// Both surfaces' placements resolved to coordinates, in pin order, with the unplaced pins sorted
/// and each named once. Slot indices mean different things in each list, so they are never mixed
/// — only the resolved positions are.
pub fn resolve(
    top: (&[Placement], &[Slot]),
    edge: &[Placement],
    slots: &[Slot],
    top_set: &std::collections::BTreeSet<usize>,
    mut unplaced: Vec<usize>,
) -> (Vec<Placed>, Vec<usize>) {
    let mut placed: Vec<Placed> = Vec::new();
    placed.extend(top.0.iter().map(|x| Placed::of(x, top.1, slots.len())));
    placed.extend(edge.iter().filter(|x| !top_set.contains(&x.pin)).map(|x| Placed::of(x, slots, 0)));
    placed.sort_by_key(|x| x.pin);
    unplaced.sort_unstable();
    unplaced.dedup();
    (placed, unplaced)
}

/// How a committed pin is sized, in database units: `set_pin_length`, `set_pin_length_extension`
/// and the thickness multipliers. The default is a run that set none of them.
#[derive(Debug, Clone, PartialEq)]
pub struct PinShape {
    pub hor_length: Option<i32>,
    pub ver_length: Option<i32>,
    pub hor_extension: i32,
    pub ver_extension: i32,
    pub hor_multiplier: f64,
    pub ver_multiplier: f64,
}

impl Default for PinShape {
    fn default() -> Self {
        PinShape { hor_length: None, ver_length: None, hor_extension: 0, ver_extension: 0, hor_multiplier: 1.0, ver_multiplier: 1.0 }
    }
}

/// Why a placement stopped: the reference's own error (`PPL-nnnn` and its text), or something
/// this engine does not model.
#[derive(Debug, Clone, PartialEq)]
pub enum Fail {
    Error(u16, String),
    Refused(String),
}

impl Fail {
    /// The line as the command line reports it.
    pub fn line(&self) -> String {
        match self {
            Fail::Error(id, t) => format!("PPL-{id:04} {t}"),
            Fail::Refused(t) => t.clone(),
        }
    }
}

/// Where the slots came from, for the commit: the die, its slot list and, on a polygon die, each
/// slot's segment.
pub struct Surface<'a> {
    pub boundary: Boundary,
    pub slots: &'a [Slot],
    pub on_segment: Option<&'a [usize]>,
    /// The pin layers (`PPL-0077` names a placed pin on any other).
    pub hor: &'a [String],
    pub ver: &'a [String],
}

/// `IOPlacer::commitIOPlacementToDB`: every placed pin's port pins replaced by one pin on its
/// layer — the rectangle `updatePinArea` gives it — PLACED. The database is not written here.
///
/// A pin's orientation comes from its position on a rectangle (`updateOrientation`), from probes
/// either side of its segment on a polygon (`updateOrientationPolygon`). Refused: a placement that
/// left a pin unplaced (upstream stops with PPL-0039 before committing anything).
/// Errors as upstream's: no manufacturing grid (PPL-0020), a pin smaller than its layer's minimum
/// area (PPL-0079).
pub fn commit(db: &mut Db, s: &Surface, shape: &PinShape, pins: &[Pin], placed: &[Placed], unplaced: &[usize]) -> Result<(), Fail> {
    use crate::commit::{area, orientation, pin_rect, polygon_orientation, top_layer_rect, Orientation};
    let ring = db.die_area_polygon().unwrap_or_default();
    let segments = polygon::segments(&ring);
    if !unplaced.is_empty() {
        // The reference's error, word for word: it stops here too, nothing written.
        return Err(Fail::Error(39, format!("Assigned {} pins out of {} IO pins.", placed.len(), placed.len() + unplaced.len())));
    }
    let mfg = db.tech_get_manufacturing_grid();
    let dbu_um = f64::from(db.tech_get_db_units_per_micron());
    if mfg == 0 {
        return Err(Fail::Error(20, "Manufacturing grid is not defined.".into()));
    }
    let directions: std::collections::HashMap<String, String> = db.layers_with_direction().unwrap_or_default().into_iter().collect();
    let top = db.bterm_top_layer_grid().ok().flatten();
    let b = s.boundary;
    let die = (b.x0, b.y0, b.x1, b.y1);
    let mut rects = Vec::with_capacity(placed.len());
    for x in placed {
        let name = &pins[x.pin].name;
        let r = if x.slot >= s.slots.len() {
            let g = top.as_ref().ok_or_else(|| Fail::Refused(format!("{name}: a top-layer pin with no top-layer grid")))?;
            top_layer_rect((x.x, x.y), g.pin_width, g.pin_height)
        } else {
            if !s.hor.contains(&x.layer) && !s.ver.contains(&x.layer) {
                return Err(Fail::Error(77, format!("Layer {} of Pin {name} not found.", x.layer)));
            }
            let o = match s.on_segment {
                // The slot's segment says whether to probe across x or across y.
                Some(on) => polygon_orientation((x.x, x.y), segments[on[x.slot]].a.0 == segments[on[x.slot]].b.0, &ring).ok_or_else(|| {
                    Fail::Error(9999, format!("Could not determine orientation for pin {name} at ({:.2}um, {:.2}um)", f64::from(x.x) / dbu_um, f64::from(x.y) / dbu_um))
                })?,
                None => orientation((x.x, x.y), die).ok_or_else(|| Fail::Refused(format!("{name}: placed off the die boundary")))?,
            };
            // `computePinSize` reads the LAYER's direction, not the edge's.
            let vertical = directions.get(&x.layer).is_some_and(|d| d == "VERTICAL");
            let (mul, user) = if vertical { (shape.ver_multiplier, shape.ver_length) } else { (shape.hor_multiplier, shape.hor_length) };
            let min_area = db.layer_get_area(&x.layer).unwrap_or(0);
            let (half_width, height) = crate::blocked::pin_size(db.layer_get_width(&x.layer) as i32, min_area, mul, user, mfg);
            let ext = if matches!(o, Orientation::North | Orientation::South) { shape.ver_extension } else { shape.hor_extension };
            let r = pin_rect((x.x, x.y), o, half_width, height, ext);
            if area(r) < min_area {
                let um2 = |a: i64| a as f64 / (dbu_um * dbu_um);
                return Err(Fail::Error(79, format!("Pin {name} area {:2.4}um^2 is lesser than the minimum required area {:2.4}um^2.", um2(area(r)), um2(min_area))));
            }
            r
        };
        rects.push((name.clone(), x.layer.clone(), r));
    }
    for (name, layer, r) in &rects {
        db.bterm_clear_unfixed_bpins(name).map_err(|e| Fail::Refused(format!("{name}: {e}")))?;
        db.place_bterm(name, layer, r.0, r.1, r.2, r.3).map_err(|e| Fail::Refused(format!("{name}: {e}")))?;
    }
    Ok(())
}

/// The debug group `PPL displacement 1`'s line: each placed pin's move from where it stood.
fn displacement_line(net: &Netlist, placed: &[Placed], dbu: i32) -> String {
    let moves: Vec<crate::report::Move> = placed.iter().filter_map(|x| net.pins[x.pin].initial.map(|from| (from, (x.x, x.y)))).collect();
    crate::report::displacement(&moves, dbu)
}

/// `IOPlacer::runHungarianMatching(minimize_displacement)` on `db`, with the settings
/// `set_place_config` stored in the block (`applySettings`) and the default section size — the
/// call the global placer's `-place_ios` makes once Nesterov placement is done. Nothing but the
/// reference's call sequence; each stage is a function above.
pub fn run_hungarian_matching(db: &mut Db, minimize_displacement: bool, log: &mut vyges_events::logger::Logger) -> Result<(), Fail> {
    let args = crate::session::place_config_args(db);
    let grid = crate::session::slot_grid(db, &args).map_err(Fail::Refused)?;
    let per_section = crate::SLOTS_PER_SECTION;
    let b = grid.boundary;
    let die = (b.x0, b.y0, b.x1, b.y1);
    let net = init_netlist(db, die, &grid.slots, per_section, minimize_displacement);
    let (top_slots, top_pins) = read_top_layer(db, die, &net.pins);
    for (id, text) in crate::report::config(&config(db, &grid.slots, &top_slots, &net, Some(per_section), minimize_displacement)) {
        log.info("PPL", id, text);
    }
    let top_set: std::collections::BTreeSet<usize> = top_pins.iter().map(|&(i, _)| i).collect();
    let (top_placed, mut unplaced) = place_top_layer(&top_slots, &net.pins, &top_pins, per_section);
    let (edge_placed, edge_unplaced) = place_boundary(&grid.slots, grid.on_segment.as_deref(), &net, &top_set, per_section, die);
    unplaced.extend(edge_unplaced.into_iter().filter(|i| !top_set.contains(i)));
    let (placed, unplaced) = resolve((&top_placed, &top_slots), &edge_placed, &grid.slots, &top_set, unplaced);
    let (id, text) = crate::report::sections_assigned(net.pins.len() - unplaced.len(), net.pins.len());
    log.info("PPL", id, text);
    let fail = |log: &mut vyges_events::logger::Logger, f: Fail| -> Result<(), Fail> {
        if let Fail::Error(id, t) = &f {
            let _ = log.error::<()>("PPL", *id, t);
        }
        Err(f)
    };
    if !unplaced.is_empty() {
        return fail(log, Fail::Error(39, format!("Assigned {} pins out of {} IO pins.", placed.len(), net.pins.len())));
    }
    if minimize_displacement {
        log.debug("PPL", "displacement", 1, || displacement_line(&net, &placed, grid.dbu));
    }
    let (id, text) = crate::report::io_hpwl(&net.pins, placed.iter().map(|x| (x.pin, (x.x, x.y))), grid.dbu);
    log.info("PPL", id, text);
    let surface = Surface { boundary: grid.boundary, slots: &grid.slots, on_segment: grid.on_segment.as_deref(), hor: &args.hor, ver: &args.ver };
    match commit(db, &surface, &PinShape::default(), &net.pins, &placed, &unplaced) {
        Err(f) => fail(log, f),
        ok => ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream rule (`initNetlist` → `IOPin(bterm, Point(x_pos, y_pos))`, `getFirstPinLocation`
    /// zeroing both on a port with no placed box): a displacement-minimising run measures every pin
    /// from its first placed box's centre, or from the origin; any other run from nowhere.
    #[test]
    fn a_pin_with_no_placed_box_is_displaced_from_the_origin() {
        assert_eq!(initial_position(true, Some((120, 40))), Some((120, 40)));
        assert_eq!(initial_position(true, None), Some((0, 0)), "no placed box: the origin, not no position");
        assert_eq!(initial_position(false, Some((120, 40))), None, "only a displacement-minimising run has one");
    }

    fn placed(pin: usize, x: i32, y: i32) -> Placed {
        Placed { pin, x, y, layer: "m2".into(), edge: Edge::Bottom, slot: 0 }
    }

    /// Upstream rule (`runHungarianMatching`'s `PPL displacement 1` line): over the ASSIGNMENT —
    /// every placed pin — each pin's Manhattan move from its initial position to where it went.
    #[test]
    fn the_displacement_line_measures_each_placed_pin_from_where_it_stood() {
        let pin = |name: &str, initial| Pin { name: name.into(), sinks: vec![], mirror: None, initial: Some(initial) };
        let net = Netlist { pins: vec![pin("a", (0, 0)), pin("b", (500, 0)), pin("c", (900, 0))], groups: vec![], constraints: vec![], unsorted_constraints: vec![] };
        // a stays; b moves 300 + 100; c is not placed and is not counted.
        let line = displacement_line(&net, &[placed(0, 0, 0), placed(1, 200, 100)], 100);
        assert_eq!(line, "Moved 1 of 2 pins, mean 4.00 um, max 4.00 um.");
    }
}
