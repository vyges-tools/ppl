// SPDX-License-Identifier: Apache-2.0
//! Top-layer pins — placed on a lattice inside the die, not on its boundary.
//!
//! `define_pin_shape_pattern` declares a grid on the topmost routing layer, and
//! `set_io_pin_constraint -region up:*` sends pins to it. Those pins are reached from above — by a
//! package bump or a die-to-die connection — so they do not belong on an edge at all.
//!
//! Almost nothing from the boundary path carries over. There is no edge, so there is no direction
//! along which to order or mirror; positions come from a two-dimensional step rather than a track
//! pattern; and a position is legal only if a pin of the declared size **fits** there, which makes
//! blocking a question about rectangles rather than about intervals.
//!
//! What *is* shared is the part that matters: sections and optimal matching. A top-layer section
//! is a run of the same lattice, and pins are matched into it exactly as they are on an edge.
//!
//! Nothing here touches a database.

use crate::sections::{Pin, Section};
use crate::slots::{Edge, Slot};

/// The lattice top-layer pins are placed on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    pub layer: String,
    pub x_step: i32,
    pub y_step: i32,
    /// The size of a pin placed here. A position is legal only if this rectangle, centred on it,
    /// fits inside the die and clears every obstruction.
    pub pin_width: i32,
    pub pin_height: i32,
    /// Extra clearance a pin must keep from an obstruction, beyond its own size.
    pub keepout: i32,
    pub region: (i32, i32, i32, i32),
}

/// **T1** — every position on the lattice, in the order the engine generates them.
///
/// **Column-major: x outer, y inner.** Sections are cut from contiguous runs of this list, so a
/// top-layer section is a *column* of the grid. Generating it row-major would still produce the
/// same positions and a completely different set of sections.
///
/// The region's upper bounds are exclusive, so a grid whose region is an exact multiple of the
/// step does not get a final row or column outside it.
pub fn positions(grid: &Grid) -> Vec<(i32, i32)> {
    let (x0, y0, x1, y1) = grid.region;
    let mut out = Vec::new();
    if grid.x_step <= 0 || grid.y_step <= 0 {
        return out;
    }
    let mut x = x0;
    while x < x1 {
        let mut y = y0;
        while y < y1 {
            out.push((x, y));
            y += grid.y_step;
        }
        x += grid.x_step;
    }
    out
}

/// **T2** — is a position unusable?
///
/// Two reasons, and both are about the **pin's rectangle**, not the point:
///
/// - The pin would extend **beyond the die**. A lattice position near the region's edge is a legal
///   point and an illegal pin.
/// - The pin, grown by the keepout, would **touch an obstruction** — a routing blockage, a
///   power-grid wire, or a port already fixed, on the grid's own layer.
///
/// Checking the point instead of the rectangle passes positions where the pin does not fit, which
/// is the whole failure mode this guards against.
pub fn blocked(grid: &Grid, at: (i32, i32), die: (i32, i32, i32, i32), obstructions: &[(i32, i32, i32, i32)]) -> bool {
    let (hw, hh) = (grid.pin_width / 2, grid.pin_height / 2);
    let (dx0, dy0, dx1, dy1) = die;
    if at.0 - hw < dx0 || at.1 - hh < dy0 || at.0 + hw > dx1 || at.1 + hh > dy1 {
        return true;
    }
    let k = grid.keepout;
    let (px0, py0, px1, py1) = (at.0 - hw - k, at.1 - hh - k, at.0 + hw + k, at.1 + hh + k);
    obstructions
        .iter()
        .any(|&(ox0, oy0, ox1, oy1)| px0 < ox1 && ox0 < px1 && py0 < oy1 && oy0 < py1)
}

/// **T1, T2** — the top-layer slot list.
///
/// Slots carry [`Edge::Invalid`] because they are on no edge; every rule that branches on edge
/// direction — ordering a group, reflecting a mirrored pin — is meaningless here and must not fire.
pub fn slots(
    grid: &Grid,
    die: (i32, i32, i32, i32),
    obstructions: &[(i32, i32, i32, i32)],
) -> Vec<Slot> {
    positions(grid)
        .into_iter()
        .map(|(x, y)| Slot {
            x,
            y,
            layer: grid.layer.clone(),
            edge: Edge::Invalid,
            blocked: blocked(grid, (x, y), die, obstructions),
        })
        .collect()
}

/// **T3** — sections for a top-layer constraint region, one run per grid **column**.
///
/// A column is the natural unit because the slot list is column-major: a run of consecutive
/// indices is a run of the same x. Sections are then cut from each column exactly as they are
/// along an edge.
pub fn sections_for(
    slots: &[Slot],
    region: (i32, i32, i32, i32),
    slots_per_section: usize,
) -> Vec<Section> {
    let (x0, y0, x1, y1) = region;
    let mut out = Vec::new();
    let mut i = 0;
    while i < slots.len() {
        let x = slots[i].x;
        let mut j = i;
        while j < slots.len() && slots[j].x == x {
            j += 1;
        }
        if x >= x0 && x <= x1 {
            let inside: Vec<usize> =
                (i..j).filter(|&k| slots[k].y >= y0 && slots[k].y < y1).collect();
            if let (Some(&first), Some(&last)) = (inside.first(), inside.last()) {
                out.extend(crate::sections::find_sections(
                    slots,
                    first,
                    last,
                    Edge::Invalid,
                    slots_per_section,
                ));
            }
        }
        i = j;
    }
    out
}

/// Which pins the design sent to the top layer: those whose constraint region has real **area**.
///
/// 🔑 The same command declares both kinds of constraint, and the *shape of the rectangle* is what
/// distinguishes them — a degenerate one is an edge interval, one with area is a top-layer region.
pub fn constrained_pins(
    pins: &[Pin],
    region_of: &dyn Fn(usize) -> Option<(i32, i32, i32, i32)>,
) -> Vec<(usize, (i32, i32, i32, i32))> {
    (0..pins.len())
        .filter_map(|i| {
            let r = region_of(i)?;
            (r.0 != r.2 && r.1 != r.3).then_some((i, r))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid {
            layer: "met9".into(),
            x_step: 100,
            y_step: 100,
            pin_width: 40,
            pin_height: 40,
            keepout: 0,
            region: (0, 0, 300, 300),
        }
    }

    const DIE: (i32, i32, i32, i32) = (-1000, -1000, 1000, 1000);

    #[test]
    fn the_lattice_is_generated_column_by_column() {
        // The order is a contract: sections are cut from contiguous runs, so a top-layer section
        // is a COLUMN. Row-major would give the same points and different sections.
        let p = positions(&grid());
        assert_eq!(p.len(), 9);
        assert_eq!(&p[..3], &[(0, 0), (0, 100), (0, 200)], "x outer, y inner");
        assert_eq!(p[3], (100, 0));
    }

    #[test]
    fn the_regions_upper_bounds_are_exclusive() {
        // A region that is an exact multiple of the step gets no extra row or column outside it.
        let p = positions(&grid());
        assert!(p.iter().all(|&(x, y)| x < 300 && y < 300));
        let single = Grid { region: (0, 0, 1, 1), ..grid() };
        assert_eq!(positions(&single), vec![(0, 0)]);
    }

    #[test]
    fn a_degenerate_step_yields_nothing_rather_than_looping_forever() {
        for bad in [(0, 100), (100, 0), (-5, 100)] {
            let g = Grid { x_step: bad.0, y_step: bad.1, ..grid() };
            assert!(positions(&g).is_empty(), "{bad:?}");
        }
    }

    #[test]
    fn a_position_is_blocked_when_the_PIN_leaves_the_die_not_the_point() {
        // The point is inside; the 40-wide pin centred on it is not.
        let g = grid();
        let tight = (0, 0, 300, 300);
        assert!(!blocked(&g, (150, 150), tight, &[]), "well inside");
        assert!(blocked(&g, (10, 150), tight, &[]), "the pin overhangs the left edge");
        assert!(blocked(&g, (150, 290), tight, &[]), "and the top");
        assert!(!blocked(&g, (20, 20), tight, &[]), "exactly touching is still inside");
    }

    #[test]
    fn an_obstruction_blocks_a_position_whose_pin_would_touch_it() {
        let g = grid();
        let obs = [(200, 200, 260, 260)];
        assert!(blocked(&g, (230, 230), DIE, &obs), "straight through it");
        assert!(!blocked(&g, (100, 100), DIE, &obs), "well clear");
        // The pin is 40 wide, so its edge reaches 180..220 — it clips the obstruction at 200.
        assert!(blocked(&g, (200, 230), DIE, &obs), "the pin RECTANGLE clips it");
    }

    #[test]
    fn the_keepout_widens_what_counts_as_touching() {
        let obs = [(200, 200, 260, 260)];
        let bare = grid();
        // The pin is 40 wide, so at x = 100 its own edge reaches 120 — 80 short of the
        // obstruction. A keepout of 110 closes that gap and nothing smaller does.
        let guarded = Grid { keepout: 110, ..grid() };
        assert!(!blocked(&bare, (100, 230), DIE, &obs), "clear without a keepout");
        assert!(!blocked(&Grid { keepout: 60, ..grid() }, (100, 230), DIE, &obs), "60 is not enough");
        assert!(blocked(&guarded, (100, 230), DIE, &obs), "110 reaches it");
    }

    #[test]
    fn top_layer_slots_belong_to_no_edge() {
        // Every rule that branches on edge direction — group ordering, mirroring — must not fire
        // here, and this is what stops it.
        let s = slots(&grid(), DIE, &[]);
        assert_eq!(s.len(), 9);
        assert!(s.iter().all(|x| x.edge == Edge::Invalid));
        assert!(s.iter().all(|x| x.layer == "met9"));
        assert!(s.iter().all(|x| !x.blocked));
    }

    #[test]
    fn sections_follow_the_columns_of_the_grid() {
        let s = slots(&grid(), DIE, &[]);
        let secs = sections_for(&s, (0, 0, 300, 300), 200);
        assert_eq!(secs.len(), 3, "one per column");
        for sec in &secs {
            let xs: Vec<i32> = (sec.begin_slot..=sec.end_slot).map(|i| s[i].x).collect();
            assert!(xs.windows(2).all(|w| w[0] == w[1]), "a section spans one column: {xs:?}");
            assert_eq!(sec.edge, Edge::Invalid);
        }
    }

    #[test]
    fn a_constraint_region_narrows_which_positions_are_offered() {
        let s = slots(&grid(), DIE, &[]);
        let secs = sections_for(&s, (0, 0, 150, 150), 200);
        for sec in &secs {
            for i in sec.begin_slot..=sec.end_slot {
                assert!(s[i].x <= 150 && s[i].y < 150, "{:?} is outside the region", (s[i].x, s[i].y));
            }
        }
        assert!(!secs.is_empty());
    }

    #[test]
    fn a_region_the_grid_does_not_reach_yields_no_sections() {
        let s = slots(&grid(), DIE, &[]);
        assert!(sections_for(&s, (5000, 5000, 6000, 6000), 200).is_empty());
    }

    #[test]
    fn a_rectangle_with_area_is_a_top_layer_pin_and_a_flat_one_is_not() {
        // The same command declares both kinds; only the shape tells them apart.
        let pins: Vec<Pin> = (0..3)
            .map(|i| Pin { name: format!("p{i}"), sinks: vec![], mirror: None })
            .collect();
        let region = |i: usize| match i {
            0 => Some((10, 10, 90, 90)),   // area: top layer
            1 => Some((0, 10, 0, 90)),     // degenerate: an edge interval
            _ => None,
        };
        assert_eq!(constrained_pins(&pins, &region), vec![(0, (10, 10, 90, 90))]);
    }
}
