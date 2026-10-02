// SPDX-License-Identifier: Apache-2.0
//! **P10** — what the design already blocks on the die boundary, before any pin is placed: the
//! placed macros, the routing obstructions and the power grid.
//!
//! One function per stage of the rule, in its call order:
//! - [`find_blocked_intervals`] — where a box meets the die edge, one interval per edge it touches;
//! - [`pin_size`] — the pin a slot would hold on a layer (its half width and its depth);
//! - [`pin_keepout`] — a shape grown by that pin plus the spacing the layer asks between the two;
//! - [`boundary_shape_intervals`] — a shape's blocked intervals: grown, clipped to the die, kept
//!   only on the edges whose pins run on its layer's axis.
//!
//! A macro blocks every layer; an obstruction or a power-grid shape blocks only its own layer.

use crate::slots::{Edge, Interval};

/// A rectangle, `(x0, y0, x1, y1)`.
pub type Rect = (i32, i32, i32, i32);

/// Rule (`findBlockedIntervals`): a box meets an edge when its side lies ON the die's side — one
/// interval per edge it meets, along that edge, in the order bottom, top, left, right.
pub fn find_blocked_intervals(die: Rect, b: Rect) -> Vec<Interval> {
    let mut out = Vec::new();
    if die.1 == b.1 {
        out.push(Interval { edge: Edge::Bottom, begin: b.0, end: b.2 });
    }
    if die.3 == b.3 {
        out.push(Interval { edge: Edge::Top, begin: b.0, end: b.2 });
    }
    if die.0 == b.0 {
        out.push(Interval { edge: Edge::Left, begin: b.1, end: b.3 });
    }
    if die.2 == b.2 {
        out.push(Interval { edge: Edge::Right, begin: b.1, end: b.3 });
    }
    out
}

/// What a pin on one layer measures: `(half width, depth into the die)`.
///
/// Rule (`computePinSize`): half width = `int(ceil(min width / 2)) × multiplier`, truncated; depth
/// = `max(2 × half width, ceil(min area / (2 × half width)))`, a `set_pin_length` replacing it,
/// rounded UP to the manufacturing grid.
pub fn pin_size(min_width: i32, min_area: i64, multiplier: f64, user_length: Option<i32>, mfg_grid: i32) -> (i32, i32) {
    let half_width = ((f64::from(min_width) / 2.0).ceil() * multiplier) as i32;
    let mut height = (2.0 * f64::from(half_width)).max((min_area as f64 / (2.0 * f64::from(half_width))).ceil()) as i32;
    if let Some(l) = user_length {
        height = l;
    }
    if mfg_grid > 0 && height % mfg_grid != 0 {
        height = mfg_grid * (height as f32 / mfg_grid as f32).ceil() as i32;
    }
    (half_width, height)
}

/// Rule (`computePinKeepout`): the shape grown by the pin's half width plus the spacing ALONG the
/// edge, and by the pin's depth plus the spacing ACROSS it. A vertical layer's pins sit on the
/// horizontal edges, so it grows x by the half width and y by the depth; a horizontal one the
/// other way round. `spacing` is the layer's rule for the wider of the two at that run length.
pub fn pin_keepout(b: Rect, vertical_layer: bool, half_width: i32, height: i32, spacing: i32) -> Rect {
    let (along, across) = (half_width + spacing, height + spacing);
    if vertical_layer {
        (b.0 - along, b.1 - across, b.2 + along, b.3 + across)
    } else {
        (b.0 - across, b.1 - along, b.2 + across, b.3 + along)
    }
}

/// Rule (`excludeBoundaryShape`, after its layer filter): the shape's keepout, if it reaches the die
/// at all (touching counts), clipped to the die; its blocked intervals, kept only on the edges whose
/// pins run on the layer's axis (a vertical layer's on the bottom and top edges).
pub fn boundary_shape_intervals(die: Rect, keepout: Rect, vertical_layer: bool) -> Vec<Interval> {
    let meets = keepout.0 <= die.2 && die.0 <= keepout.2 && keepout.1 <= die.3 && die.1 <= keepout.3;
    if !meets {
        return Vec::new();
    }
    let clipped = (keepout.0.max(die.0), keepout.1.max(die.1), keepout.2.min(die.2), keepout.3.min(die.3));
    find_blocked_intervals(die, clipped).into_iter().filter(|i| i.edge.is_vertical_pin() == vertical_layer).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIE: Rect = (0, 0, 1000, 1000);

    // Rule (findBlockedIntervals): one interval per die edge the box lies on; a corner box two.
    #[test]
    fn a_box_blocks_each_edge_it_lies_on() {
        assert!(find_blocked_intervals(DIE, (100, 100, 200, 200)).is_empty(), "inside, touching nothing");
        let corner = find_blocked_intervals(DIE, (0, 0, 50, 80));
        assert_eq!(corner, vec![Interval { edge: Edge::Bottom, begin: 0, end: 50 }, Interval { edge: Edge::Left, begin: 0, end: 80 }]);
    }

    // Rule (computePinSize): the depth is at least twice the half width, else what the minimum
    // area needs, then rounded up to the manufacturing grid; a set_pin_length replaces it.
    #[test]
    fn the_pin_depth_meets_the_minimum_area() {
        assert_eq!(pin_size(70, 0, 1.0, None, 5), (35, 70));
        assert_eq!(pin_size(70, 14000, 1.0, None, 5), (35, 200));
        assert_eq!(pin_size(70, 14001, 1.0, None, 5), (35, 205), "rounded UP to the grid");
        assert_eq!(pin_size(70, 14000, 2.0, None, 5), (70, 140));
        assert_eq!(pin_size(70, 14000, 1.0, Some(333), 5), (35, 335));
    }

    // Rules (computePinKeepout, excludeBoundaryShape): a vertical-layer stripe ending near the
    // bottom edge blocks the bottom edge only, over its width grown by half a pin plus spacing.
    #[test]
    fn a_power_stripe_blocks_the_edge_its_pins_run_on() {
        let stripe = (400, 20, 420, 600);
        let k = pin_keepout(stripe, true, 35, 70, 10);
        assert_eq!(k, (355, -60, 465, 680));
        assert_eq!(boundary_shape_intervals(DIE, k, true), vec![Interval { edge: Edge::Bottom, begin: 355, end: 465 }]);
        assert!(boundary_shape_intervals(DIE, k, false).is_empty(), "the left/right edges' pins run the other way");
        assert!(boundary_shape_intervals(DIE, pin_keepout((400, 300, 420, 600), true, 35, 70, 10), true).is_empty(), "too far in");
    }
}
