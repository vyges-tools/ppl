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
    Ok(Prepared { exclusions: g.exclusions, on_segment: g.on_segment, db, boundary: g.boundary, dbu: g.dbu, slots: g.slots, trackless: g.trackless })
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
    SlotArgs { hor: layers("place_config_io_pin_hor_layers"), ver: layers("place_config_io_pin_ver_layers"), params, hor_length: None, ver_length: None }
}
