// SPDX-License-Identifier: Apache-2.0
//! The pin placer's session on a database: the slot list and what blocks it, read from the block
//! as `IOPlacer` does before any assignment. Shared by the command line and by the global
//! placer's concurrent IO placement (`buildSlotGrid`).

use crate::{define_slots, interval_from_rect, is_blocked, polygon, Boundary, LayerTracks, Params, Slot, TrackPattern};
use vyges_opendb::Db;

/// What the slot list is built from: the pin layers and the slot parameters (the command line's
/// flags, or `set_place_config`'s block properties).
#[derive(Debug, Clone)]
pub struct SlotArgs {
    /// Layers for the left and right edges (horizontal pins).
    pub hor: Vec<String>,
    /// Layers for the bottom and top edges (vertical pins).
    pub ver: Vec<String>,
    /// Spacing, corner avoidance and pin thickness.
    pub params: Params,
    /// `set_pin_length -hor_length` in database units.
    pub hor_length: Option<i32>,
    /// `set_pin_length -ver_length` in database units.
    pub ver_length: Option<i32>,
}

/// Everything both verbs need out of the database and the command line.
pub struct Prepared {
    pub db: Db,
    pub boundary: Boundary,
    pub dbu: i32,
    pub slots: Vec<Slot>,
    pub trackless: Vec<String>,
    pub exclusions: usize,
    /// For a **polygon** die: which boundary segment each slot came from. `None` for the ordinary
    /// four-edge case. Sections are cut per segment, so this is not optional detail — it is how a
    /// polygon's slot list is divided at all.
    pub on_segment: Option<Vec<usize>>,
    /// The run's pin layers and settings, after `applySettings` (what `reportMinDistances` reads).
    pub hor: Vec<String>,
    pub ver: Vec<String>,
    pub params: Params,
}

/// Read one layer's contribution to a slot list.
///
/// ⚠️ Two accessors on the layer are a letter apart and both bridged: upstream takes
/// `getWidth()`, **not** `getMinWidth()`. Picking the other one compiles and quietly misplaces the
/// outermost slot on every edge, so it is worth naming here rather than in a commit message.
///
/// The axis follows the edge, not the layer's own routing direction: pins on the bottom and top
/// run down the **x** track patterns, pins on the left and right down the **y** ones.
pub fn read_layer(db: &Db, layer: &str, vertical_pins: bool) -> Result<LayerTracks, String> {
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

/// The slot list on `db` for `a`: the die boundary, every edge's slots on its layers, each marked
/// blocked by the excluded regions, the placed macros, the obstructions and power grid, and the
/// fixed pins' keep-outs.
pub fn prepare_slots(db: Db, a: &SlotArgs) -> Result<Prepared, String> {
    let g = slot_grid(&db, a)?;
    Ok(Prepared {
        exclusions: g.exclusions,
        on_segment: g.on_segment,
        db,
        boundary: g.boundary,
        dbu: g.dbu,
        slots: g.slots,
        trackless: g.trackless,
        hor: a.hor.clone(),
        ver: a.ver.clone(),
        params: a.params.clone(),
    })
}

/// The slot list [`prepare_slots`] builds, without taking the database.
pub struct SlotGrid {
    /// The die rectangle the slots are on.
    pub boundary: Boundary,
    /// Database units per micron.
    pub dbu: i32,
    /// Every slot, in the reference's order, blocked ones marked.
    pub slots: Vec<Slot>,
    /// Pin layers with no track pattern.
    pub trackless: Vec<String>,
    /// How many excluded regions the block names.
    pub exclusions: usize,
    /// For a polygon die, the boundary segment of each slot.
    pub on_segment: Option<Vec<usize>>,
}

/// `IOPlacer::buildSlotGrid`'s slots on `db` for `a` (see [`prepare_slots`]).
pub fn slot_grid(db: &Db, a: &SlotArgs) -> Result<SlotGrid, String> {
    let (hor, ver, params) = (&a.hor, &a.ver, a.params.clone());
    let (hor_length, ver_length) = (a.hor_length, a.ver_length);
    let dbu = db.dbu_per_micron();
    // Upstream reaches this through a type named `Core`, but it is built from the DIE area — see
    // `Boundary`. Reading the core area here would inset every pin by the core-to-die margin.
    let boundary = Boundary {
        x0: db.block_get_die_area_x_min(),
        y0: db.block_get_die_area_y_min(),
        x1: db.block_get_die_area_x_max(),
        y1: db.block_get_die_area_y_max(),
    };

    let read = |names: &[String], vertical: bool| -> Result<Vec<LayerTracks>, String> {
        names.iter().map(|l| read_layer(db, l, vertical)).collect()
    };
    let ver_tracks = read(ver, true)?;
    let hor_tracks = read(hor, false)?;

    // What the design has already ruled out. Both are read from the block, like constraints:
    // regions the design excluded outright, and the metal of ports already fixed in place.
    let die = (boundary.x0, boundary.y0, boundary.x1, boundary.y1);
    let exclusions: Vec<crate::Interval> = db
        .blocked_regions_for_pins()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|r| interval_from_rect(r, die))
        .collect();
    // `IOPlacer::getBlockedRegions`: the placed macros (every layer), then each routing obstruction
    // and each power-grid shape (its own layer) — see `crate::blocked`.
    let mut exclusions = exclusions;
    for inst in db.block_get_insts() {
        let master = db.inst_get_master(&inst);
        if !(db.master_is_block(&master) && db.inst_is_placed(&inst)) {
            continue;
        }
        if let Ok(b) = db.inst_bbox(&inst) {
            if let [x0, y0, x1, y1] = b[..] {
                let clipped = (x0.max(die.0), y0.max(die.1), x1.min(die.2), y1.min(die.3));
                exclusions.extend(crate::blocked::find_blocked_intervals(die, clipped));
            }
        }
    }
    let mfg = db.tech_get_manufacturing_grid();
    let directions: std::collections::HashMap<String, String> = db.layers_with_direction().unwrap_or_default().into_iter().collect();
    // `IOPlacer::computePinKeepout`: a shape grown by the pin a slot on its layer would hold, plus
    // the spacing the layer asks between the two (`computeShapeSpacing`).
    // `VYGES_PPL_BOUNDARY_TRACE`: the reference's own call sequence, line for line as
    // `instruments/ppl/ppl-boundary-patch.py` prints it, so the two diff directly.
    let trace = std::env::var_os("VYGES_PPL_BOUNDARY_TRACE").is_some();
    let compute_pin_keepout = |layer: &str, s: &vyges_opendb::RuledShape| {
        let b = s.rect;
        let vertical = directions.get(layer).is_some_and(|d| d == "VERTICAL");
        let multiplier = if vertical { params.thickness_multiplier_v } else { params.thickness_multiplier_h };
        let user_length = if vertical { ver_length } else { hor_length };
        let layer_width = db.layer_get_width(layer) as i32;
        let (half_width, height) = crate::blocked::pin_size(layer_width, db.layer_get_area(layer).unwrap_or(0), multiplier, user_length, mfg);
        let spacing = crate::blocked::shape_spacing(b, s.min_spacing, s.effective_width, 2 * half_width, height, layer_width, |w, l| {
            db.layer_get_spacing_width_length(layer, w, l)
        });
        let k = crate::blocked::pin_keepout(b, vertical, half_width, height, spacing);
        if trace {
            eprintln!(
                "VYGPPL keepout|{layer}|{} {} {} {}|{}|{}|{half_width}|{height}|{spacing}|{} {} {} {}",
                b.0, b.1, b.2, b.3, s.min_spacing, s.effective_width, k.0, k.1, k.2, k.3
            );
        }
        k
    };
    let mut layered: Vec<(crate::Interval, String)> = Vec::new();
    // `IOPlacer::excludeBoundaryShape`.
    let mut exclude_boundary_shape = |s: &vyges_opendb::RuledShape| {
        let layer = db.layer_name_by_number(s.layer);
        if db.layer_get_routing_level(&layer) == 0 {
            return;
        }
        let vertical = directions.get(&layer).is_some_and(|d| d == "VERTICAL");
        if !(ver.is_empty() && hor.is_empty()) && !(if vertical { ver } else { hor }).contains(&layer) {
            return;
        }
        let keepout = compute_pin_keepout(&layer, s);
        for i in crate::blocked::boundary_shape_intervals(die, keepout, vertical) {
            if trace {
                // The reference's `Edge` numbering: top, bottom, left, right.
                let e = match i.edge {
                    crate::Edge::Top => 0,
                    crate::Edge::Bottom => 1,
                    crate::Edge::Left => 2,
                    crate::Edge::Right => 3,
                    crate::Edge::Invalid => 4,
                };
                eprintln!("VYGPPL interval|{layer}|{e}|{}|{}", i.begin, i.end);
            }
            layered.push((i, layer.clone()));
        }
    };
    // `getBlockedRegionsFromDbObstructions`: every obstruction but the system-reserved ones (the
    // outside of a polygon die) and the fill / slot ones (routing metal may run over those), each
    // with the DEF spacing rules it carries.
    for o in db.obstruction_shapes_with_rules().unwrap_or_default() {
        if o.system_reserved || o.fill || o.slot {
            continue;
        }
        if trace {
            let (s, b) = (&o.shape, o.shape.rect);
            eprintln!("VYGPPL shape|obs|{}|{} {} {} {}|{}|{}", db.layer_name_by_number(s.layer), b.0, b.1, b.2, b.3, s.min_spacing, s.effective_width);
        }
        exclude_boundary_shape(&o.shape);
    }
    // `getBlockedRegionsFromPDN`: a power-grid shape carries no DEF rule (`{rect}`, both -1).
    for (l, x0, y0, x1, y1) in db.swire_boxes().unwrap_or_default() {
        let s = vyges_opendb::RuledShape { layer: l, rect: (x0, y0, x1, y1), min_spacing: -1, effective_width: -1 };
        exclude_boundary_shape(&s);
    }
    // `initNetlist` → `addFixedPinKeepouts`: for a term whose FIRST pin is fixed, each bpin that is
    // itself fixed (a multi-bpin term can mix statuses), each box on a routing layer, kept PADDED
    // by `computePinKeepout` with the bpin's own DEF rules, "so the pins created near them keep
    // the min spacing". No direction-layer filter here.
    let fixed_shapes: Vec<(String, i32, i32, i32, i32)> = db
        .bpin_shapes_with_rules()
        .unwrap_or_default()
        .into_iter()
        .filter(|p| p.term_fixed)
        .filter_map(|p| {
            let layer = db.layer_name_by_number(p.shape.layer);
            if db.layer_get_routing_level(&layer) == 0 {
                return None;
            }
            let k = compute_pin_keepout(&layer, &p.shape);
            Some((layer, k.0, k.1, k.2, k.3))
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

    Ok(SlotGrid { exclusions: exclusions.len(), on_segment, boundary, dbu, slots, trackless })
}



/// Upstream rule (`IOPlacer::getSettings`, #12094): `values >> layer_name >> min_dist` over the
/// stored string, so reading stops at the first pair whose distance is not an integer; a layer
/// with routing level 0 (unknown, or not a routing layer) is dropped; the pairs land in a map by
/// routing level, so they come back in level order and a repeated layer keeps its last value.
pub fn layer_min_distance_pairs(text: &str, in_tracks: bool, level: impl Fn(&str) -> i32) -> Vec<(String, i32, bool)> {
    let mut by_level = std::collections::BTreeMap::new();
    let mut words = text.split_whitespace();
    while let (Some(name), Some(dist)) = (words.next(), words.next()) {
        let Ok(d) = dist.parse::<i32>() else { break };
        let l = level(name);
        if l != 0 {
            by_level.insert(l, (name.to_string(), d, in_tracks));
        }
    }
    by_level.into_values().collect()
}

/// `IOPlacer::getSettings` + `applySettings` for a caller that set nothing of its own (the global
/// placer's `buildSlotGrid`): the pin layers, corner avoidance and minimum distance that
/// `set_place_config` stored in the block's properties. Each layer list is a set of routing
/// levels, so it comes back in level order with a repeated name once; a name the technology does
/// not have is dropped. The values are already in database units (or tracks).
pub fn place_config_args(db: &Db) -> SlotArgs {
    let known: std::collections::HashSet<String> = db.layers_with_direction().unwrap_or_default().into_iter().map(|(l, _)| l).collect();
    let layers = |prop: &str| -> Vec<String> {
        let names = db.block_string_property(prop).ok().flatten().unwrap_or_default();
        let mut by_level = std::collections::BTreeMap::new();
        for n in names.split_whitespace().filter(|n| known.contains(*n)) {
            by_level.entry(db.layer_get_routing_level(n)).or_insert_with(|| n.to_string());
        }
        by_level.into_values().collect()
    };
    let mut params = Params::default();
    if let Some(v) = db.block_int_property("place_config_io_pin_corner_avoidance").ok().flatten() {
        params.corner_avoidance = v;
    }
    if let Some(v) = db.block_int_property("place_config_io_pin_min_distance").ok().flatten() {
        params.min_distance = v;
        params.min_distance_in_tracks = db.block_bool_property("place_config_io_pin_min_distance_in_tracks").ok().flatten().unwrap_or(false);
    }
    // `getSettings` (#12094): the stored "layer distance ..." pairs, a layer the technology does
    // not have as a routing layer dropped, keyed by routing level (a std::map, so level order and
    // a repeated layer's last value), with their own in-tracks flag.
    let text = db.block_string_property("place_config_io_pin_layer_min_distance").ok().flatten().unwrap_or_default();
    let in_tracks = db.block_bool_property("place_config_io_pin_layer_min_distance_in_tracks").ok().flatten().unwrap_or(false);
    params.layer_min_distance = layer_min_distance_pairs(&text, in_tracks, |n| {
        if known.contains(n) { db.layer_get_routing_level(n) } else { 0 }
    });
    SlotArgs { hor: layers("place_config_io_pin_hor_layers"), ver: layers("place_config_io_pin_ver_layers"), params, hor_length: None, ver_length: None }
}

/// One line of [`min_distance_lines`], in the logger's three kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MinDistanceLine {
    /// An uncoded line (`logger_->report`).
    Report(String),
    Warn(u16, String),
    Info(u16, String),
}

/// Upstream rule (`IOPlacer::reportMinDistances`, OpenROAD #12094, in the f11b4276 pin; called by
/// both runs after `initNetlistAndCore`, so `place_pins` AND `global_placement -place_ios` print
/// it): with no per-layer list, the 2-track default line when the run's min distance (after
/// `applySettings`) is unset, and nothing otherwise; with a list, PPL-0125 for each listed layer
/// (level order) that is not a pin layer, then PPL-0126 with every pin layer (hor and ver
/// together, level order) and its tracks between slots. A layer's pitch is its largest track
/// pitch along its edges (`getTechMinDistance`: the X patterns for a vertical-pin layer).
pub fn min_distance_lines(db: &Db, hor: &[String], ver: &[String], params: &Params) -> Vec<MinDistanceLine> {
    min_distance_lines_with(hor, ver, params, |l| db.layer_get_routing_level(l), |l, vertical| {
        read_layer(db, l, vertical).ok().and_then(|t| t.patterns.iter().map(|p| p.step).max()).unwrap_or(0)
    })
}

/// [`min_distance_lines`] on a routing-level map and a pitch reader (`pitch(layer, vertical)`).
pub fn min_distance_lines_with(
    hor: &[String],
    ver: &[String],
    params: &Params,
    level: impl Fn(&str) -> i32,
    pitch: impl Fn(&str, bool) -> i32,
) -> Vec<MinDistanceLine> {
    let mut out = Vec::new();
    if params.layer_min_distance.is_empty() {
        if params.min_distance == 0 {
            out.push(MinDistanceLine::Report(format!("Using {} tracks default min distance between IO pins.", crate::slots::DEFAULT_MIN_DIST)));
        }
        return out;
    }
    let is_pin_layer = |l: &str| hor.iter().chain(ver).any(|x| x == l);
    for (name, _, _) in &params.layer_min_distance {
        if !is_pin_layer(name) {
            out.push(MinDistanceLine::Warn(125, format!("The min distance of layer {name} is unused because it is not an IO pin layer.")));
        }
    }
    let mut layers: std::collections::BTreeMap<i32, &str> = std::collections::BTreeMap::new();
    for l in hor.iter().chain(ver) {
        layers.entry(level(l)).or_insert(l);
    }
    let per_layer: Vec<String> = layers
        .values()
        .map(|l| {
            let vertical = ver.iter().any(|x| x == l);
            format!("{l} {}", params.min_distance_in_tracks(l, pitch(l, vertical)))
        })
        .collect();
    out.push(MinDistanceLine::Info(126, format!("Min distance in tracks    {}", per_layer.join(", "))));
    out
}

/// Upstream rule (`ppl::parse_min_distance`, place_pins' and set_place_config's Tcl, #12094):
/// in tracks, a decimal whole number above zero (`^0*([1-9][0-9]*)$`, leading zeros dropped, so
/// `010` is ten and `0x4` is refused), else PPL-0124; in microns, digits with an optional point
/// (`^([0-9]+\.?[0-9]*|\.[0-9]+)$`) that read as non-zero, else PPL-0130, then
/// `ord::microns_to_dbu` (rounded) and PPL-0128 when that is zero. Returns tracks or DBU.
pub fn parse_min_distance(option: &str, dist: &str, in_tracks: bool, dbu_per_micron: i32) -> Result<i32, String> {
    if in_tracks {
        let digits = dist.trim_start_matches('0');
        let ok = !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
        return match digits.parse::<i32>() {
            Ok(n) if ok => Ok(n),
            _ => Err(format!("[ERROR PPL-0124] {option}: {dist} is not a whole number of tracks greater than zero.")),
        };
    }
    let (int, frac) = dist.split_once('.').unwrap_or((dist, ""));
    let digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    let shape = digits(int) && digits(frac) && !(int.is_empty() && frac.is_empty()) && !(int.is_empty() && !dist.contains('.'));
    let microns = if shape { format!("0{dist}").parse::<f64>().unwrap_or(0.0) } else { 0.0 };
    if !shape || microns == 0.0 {
        return Err(format!("[ERROR PPL-0130] {option}: {dist} is not a number of microns greater than zero."));
    }
    let dbu = (microns * dbu_per_micron as f64).round() as i32;
    if dbu == 0 {
        return Err(format!("[ERROR PPL-0128] {option}: {dist} um is smaller than one database unit."));
    }
    Ok(dbu)
}

/// Upstream rule (`ppl::parse_layer_min_distance`, #12094): an even, non-empty list of layer and
/// distance pairs (PPL-0123); each layer known (`parse_layer_name`, PPL-0051), a routing layer
/// (PPL-0129) and named once (PPL-0127, by the technology's name); each distance through
/// [`parse_min_distance`] with the option and layer in its message. Pairs in the order given.
/// `level` is the routing level of a known layer, `None` for an unknown name.
pub fn parse_layer_min_distance(
    option: &str,
    text: &str,
    in_tracks: bool,
    dbu_per_micron: i32,
    level: impl Fn(&str) -> Option<i32>,
) -> Result<Vec<(String, i32)>, String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || !words.len().is_multiple_of(2) {
        return Err(format!("[ERROR PPL-0123] {option} must be a list of layer and distance pairs."));
    }
    let mut out: Vec<(String, i32)> = Vec::new();
    for pair in words.chunks(2) {
        let (name, dist) = (pair[0], pair[1]);
        match level(name) {
            None => return Err(format!("[ERROR PPL-0051] Layer {name} not found.")),
            Some(0) => return Err(format!("[ERROR PPL-0129] {option} layer {name} is not a routing layer.")),
            Some(_) => {}
        }
        if out.iter().any(|(n, _)| n == name) {
            return Err(format!("[ERROR PPL-0127] {option} gives layer {name} more than once."));
        }
        out.push((name.to_string(), parse_min_distance(&format!("{option} {name}"), dist, in_tracks, dbu_per_micron)?));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Nangate45 as upstream's place_config test reads it: DBU 2000, metal2 (vertical pins) track
    // pitch 0.19 um = 380, metal3 (horizontal pins) 0.14 um = 280; metal4 a routing layer not used
    // for pins; via1 a cut layer.
    fn level(l: &str) -> i32 {
        match l {
            "metal2" => 2,
            "metal3" => 3,
            "metal4" => 4,
            _ => 0,
        }
    }
    fn pitch(l: &str, _vertical: bool) -> i32 {
        match l {
            "metal2" => 380,
            "metal3" => 280,
            _ => 0,
        }
    }
    fn lines(p: &Params) -> Vec<MinDistanceLine> {
        min_distance_lines_with(&["metal3".into()], &["metal2".into()], p, level, pitch)
    }

    /// Upstream rule (`IOPlacer::reportMinDistances` + `getMinDistanceInTracks`, OpenROAD #12094,
    /// in the f11b4276 pin), every line its test `ppl/test/place_config.ok` prints for a run:
    /// no list and no distance gives the default line; a list gives PPL-0126 over every pin layer
    /// in level order (a listed layer its own value, the rest the run's distance or the 2-track
    /// default; microns over the pitch rounded up); a listed layer that is not a pin layer warns
    /// PPL-0125 first; a distance with no list prints nothing.
    #[test]
    fn the_run_reports_each_pin_layers_min_distance_in_tracks() {
        let info = |s: &str| MinDistanceLine::Info(126, format!("Min distance in tracks    {s}"));
        assert_eq!(lines(&Params::default()), vec![MinDistanceLine::Report("Using 2 tracks default min distance between IO pins.".into())]);
        assert!(lines(&Params { min_distance: 2000, ..Params::default() }).is_empty());
        let stored = vec![("metal2".to_string(), 3, true), ("metal3".to_string(), 4, true)];
        assert_eq!(lines(&Params { layer_min_distance: stored, ..Params::default() }), vec![info("metal2 3, metal3 4")]);
        let one = vec![("metal3".to_string(), 1, true)];
        assert_eq!(lines(&Params { layer_min_distance: one, ..Params::default() }), vec![info("metal2 2, metal3 1")]);
        // set_place_config -io_pin_min_distance 0.38 -io_pin_layer_min_distance {metal3 0.28}
        let um = vec![("metal3".to_string(), 560, false)];
        assert_eq!(lines(&Params { min_distance: 760, layer_min_distance: um, ..Params::default() }), vec![info("metal2 2, metal3 2")]);
        // ... -io_pin_layer_min_distance {metal4 1} -io_pin_min_distance_in_tracks: 0.38 um is
        // 2 metal2 tracks and ceil(760 / 280) = 3 metal3 tracks.
        let off = vec![("metal4".to_string(), 1, true)];
        assert_eq!(
            lines(&Params { min_distance: 760, layer_min_distance: off, ..Params::default() }),
            vec![
                MinDistanceLine::Warn(125, "The min distance of layer metal4 is unused because it is not an IO pin layer.".into()),
                info("metal2 2, metal3 3"),
            ]
        );
    }

    /// Upstream rule (`Parameters::getMinDistance(layer)`, #12094): a listed layer takes its own
    /// distance and in-tracks flag, every other layer the run's; the slot step and the spacing
    /// filter read it per layer.
    #[test]
    fn a_listed_layer_takes_its_own_min_distance_and_unit() {
        let p = Params { min_distance: 760, layer_min_distance: vec![("metal3".into(), 4, true)], ..Params::default() };
        let m3 = p.for_layer("metal3");
        assert_eq!((m3.min_distance, m3.min_distance_in_tracks), (4, true));
        let m2 = p.for_layer("metal2");
        assert_eq!((m2.min_distance, m2.min_distance_in_tracks), (760, false));
        // An explicit per-layer distance puts candidates on every track of that layer only.
        assert_eq!(crate::slot_step(&m3, 280), 280);
        assert_eq!(crate::slot_step(&Params::default().for_layer("metal2"), 380), 760);
    }

    /// Upstream rule (`getSettings`, #12094): the stored pairs by routing level, a non-routing or
    /// unknown layer dropped, the in-tracks flag their own.
    #[test]
    fn the_stored_list_is_read_by_routing_level() {
        assert_eq!(
            layer_min_distance_pairs("metal3 4 metal2 3 via1 2", true, level),
            vec![("metal2".to_string(), 3, true), ("metal3".to_string(), 4, true)]
        );
        assert!(layer_min_distance_pairs("", false, level).is_empty());
    }

    /// Upstream rule (`ppl::parse_min_distance` / `parse_layer_min_distance`, #12094), each error
    /// `place_config.ok` prints, and the values it accepts (DBU 2000).
    #[test]
    fn min_distances_are_checked_as_the_reference_checks_them() {
        let lv = |l: &str| if l == "nope" { None } else { Some(level(l)) };
        let o = "-io_pin_layer_min_distance";
        let err = |t: &str, tr: bool| parse_layer_min_distance(o, t, tr, 2000, lv).unwrap_err();
        assert_eq!(err("metal3", false), "[ERROR PPL-0123] -io_pin_layer_min_distance must be a list of layer and distance pairs.");
        assert_eq!(err("metal3 0", false), "[ERROR PPL-0130] -io_pin_layer_min_distance metal3: 0 is not a number of microns greater than zero.");
        assert_eq!(err("metal3 1 metal3 2", false), "[ERROR PPL-0127] -io_pin_layer_min_distance gives layer metal3 more than once.");
        assert_eq!(err("metal3 0.0001", false), "[ERROR PPL-0128] -io_pin_layer_min_distance metal3: 0.0001 um is smaller than one database unit.");
        assert_eq!(err("metal3 0x4", true), "[ERROR PPL-0124] -io_pin_layer_min_distance metal3: 0x4 is not a whole number of tracks greater than zero.");
        assert_eq!(err("via1 2", false), "[ERROR PPL-0129] -io_pin_layer_min_distance layer via1 is not a routing layer.");
        assert_eq!(err("nope 2", false), "[ERROR PPL-0051] Layer nope not found.");
        let one = |d: &str, tr: bool| parse_min_distance("-min_distance", d, tr, 2000);
        assert_eq!(one("-1", false).unwrap_err(), "[ERROR PPL-0130] -min_distance: -1 is not a number of microns greater than zero.");
        assert_eq!(one("0x4", false).unwrap_err(), "[ERROR PPL-0130] -min_distance: 0x4 is not a number of microns greater than zero.");
        assert_eq!(one("2.5", true).unwrap_err(), "[ERROR PPL-0124] -min_distance: 2.5 is not a whole number of tracks greater than zero.");
        assert_eq!(one("0", true).unwrap_err(), "[ERROR PPL-0124] -min_distance: 0 is not a whole number of tracks greater than zero.");
        assert!(one(".", false).is_err() && one("", false).is_err() && one("1.2.3", false).is_err());
        assert_eq!((one("010", true), one("0.38", false), one("1.", false), one(".5", false)), (Ok(10), Ok(760), Ok(2000), Ok(1000)));
        assert_eq!(parse_layer_min_distance(o, "metal3 4 metal2 3", true, 2000, lv), Ok(vec![("metal3".into(), 4), ("metal2".into(), 3)]));
    }
}
