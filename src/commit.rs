// SPDX-License-Identifier: Apache-2.0
//! What a placed pin becomes in the database: its rectangle (`IOPlacer::updatePinArea`) and the
//! port pin that carries it (`commitIOPinToDB`).

/// The way a boundary pin points: into the die from the edge it sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    North,
    South,
    East,
    West,
}

/// Rule (`IOPlacer::updateOrientation`): read from the position alone. On the left edge a pin
/// points east — except at the top-left corner, south; on the right edge west — except at the
/// bottom-right corner, north; otherwise bottom → north and top → south. `None` off the boundary.
pub fn orientation((x, y): (i32, i32), die: (i32, i32, i32, i32)) -> Option<Orientation> {
    let (x0, y0, x1, y1) = die;
    if x == x0 {
        return Some(if y == y1 { Orientation::South } else { Orientation::East });
    }
    if x == x1 {
        return Some(if y == y0 { Orientation::North } else { Orientation::West });
    }
    if y == y0 {
        return Some(Orientation::North);
    }
    if y == y1 {
        return Some(Orientation::South);
    }
    None
}

/// Rule (`IOPlacer::isPointInsidePolygon`): a ray cast to +x over the ring's edges (last vertex to
/// first included), in `int` arithmetic — the crossing x truncates toward zero, as C++ division
/// does. A vertex on the ray's line counts by the half-open `(yi > y) != (yj > y)` test.
pub fn point_inside_polygon((x, y): (i32, i32), ring: &[(i32, i32)]) -> bool {
    let mut inside = false;
    let Some(mut j) = ring.len().checked_sub(1) else { return false };
    for i in 0..ring.len() {
        let ((xi, yi), (xj, yj)) = (ring[i], ring[j]);
        if (yi > y) != (yj > y) && x < (xj - xi).wrapping_mul(y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Rule (`IOPlacer::updateOrientationPolygon`): a pin on a VERTICAL segment points east if the
/// point 50 DBU to its right is inside the die, else west if the one to its left is; on a
/// horizontal segment north if 50 DBU above is inside, else south if below is. `None` when neither
/// probe lands inside (upstream warns PPL-9999 and leaves the orientation unset).
pub fn polygon_orientation((x, y): (i32, i32), vertical_segment: bool, ring: &[(i32, i32)]) -> Option<Orientation> {
    const OFFSET: i32 = 50;
    let probes = if vertical_segment {
        [((x + OFFSET, y), Orientation::East), ((x - OFFSET, y), Orientation::West)]
    } else {
        [((x, y + OFFSET), Orientation::North), ((x, y - OFFSET), Orientation::South)]
    };
    probes.into_iter().find(|(p, _)| point_inside_polygon(*p, ring)).map(|(_, o)| o)
}

/// Rule (`IOPlacer::updatePinArea`, boundary pins): `half_width` across the pin, `height` into the
/// die from the slot, and the length extension `ext` OUT of it (`set_pin_length_extension`; 0 when
/// unset). North/south take the vertical extension, east/west the horizontal one. Returned as
/// (x_min, y_min, x_max, y_max) — `dbBox::create` orders the corners, so a south pin's
/// "lower bound" above its "upper bound" lands the same way.
pub fn pin_rect((x, y): (i32, i32), o: Orientation, half_width: i32, height: i32, ext: i32) -> (i32, i32, i32, i32) {
    let (a, b) = match o {
        Orientation::North => ((x - half_width, y - ext), (x + half_width, y + height)),
        Orientation::South => ((x - half_width, y + ext), (x + half_width, y - height)),
        Orientation::East => ((x - ext, y - half_width), (x + height, y + half_width)),
        Orientation::West => ((x - height, y - half_width), (x + ext, y + half_width)),
    };
    (a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1))
}

/// Rule (`updatePinArea`, top-layer pins): the pattern's pin centred on its lattice point, the
/// halves truncated (`pin_width / 2`).
pub fn top_layer_rect((x, y): (i32, i32), pin_width: i32, pin_height: i32) -> (i32, i32, i32, i32) {
    (x - pin_width / 2, y - pin_height / 2, x + pin_width / 2, y + pin_height / 2)
}

/// A rectangle's area in DBU², as `IOPin::getArea` computes it for the PPL-0079 minimum-area check.
pub fn area(r: (i32, i32, i32, i32)) -> i64 {
    i64::from(r.2 - r.0) * i64::from(r.3 - r.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIE: (i32, i32, i32, i32) = (0, 0, 1000, 800);

    /// Rule (updateOrientation): the two corners that are not decided by the x test.
    #[test]
    fn corners_point_the_way_the_reference_tests_them() {
        assert_eq!(orientation((0, 800), DIE), Some(Orientation::South), "top-left");
        assert_eq!(orientation((0, 0), DIE), Some(Orientation::East), "bottom-left");
        assert_eq!(orientation((1000, 0), DIE), Some(Orientation::North), "bottom-right");
        assert_eq!(orientation((1000, 800), DIE), Some(Orientation::West), "top-right");
        assert_eq!(orientation((500, 0), DIE), Some(Orientation::North));
        assert_eq!(orientation((500, 800), DIE), Some(Orientation::South));
        assert_eq!(orientation((500, 400), DIE), None);
    }

    /// Rules (updateOrientationPolygon, isPointInsidePolygon): an L-shaped die — the probe 50 DBU
    /// off the pin decides, so a pin on the inner step points into the die, not toward its centre.
    #[test]
    fn a_polygon_pin_points_to_the_side_that_is_inside() {
        // L: (0,0) (1000,0) (1000,400) (400,400) (400,1000) (0,1000), ring closed.
        let ring = [(0, 0), (1000, 0), (1000, 400), (400, 400), (400, 1000), (0, 1000), (0, 0)];
        assert!(point_inside_polygon((200, 200), &ring));
        assert!(!point_inside_polygon((700, 700), &ring), "the notch is outside");
        assert_eq!(polygon_orientation((400, 700), true, &ring), Some(Orientation::West), "inner vertical step");
        assert_eq!(polygon_orientation((700, 400), false, &ring), Some(Orientation::South), "inner horizontal step");
        assert_eq!(polygon_orientation((0, 500), true, &ring), Some(Orientation::East));
        assert_eq!(polygon_orientation((500, 0), false, &ring), Some(Orientation::North));
        assert_eq!(polygon_orientation((2000, 2000), false, &ring), None);
    }

    /// Rule (updatePinArea): the depth goes into the die, the extension out of it.
    #[test]
    fn a_pin_reaches_into_the_die_and_extends_out_of_it() {
        assert_eq!(pin_rect((500, 0), Orientation::North, 70, 280, 0), (430, 0, 570, 280));
        assert_eq!(pin_rect((500, 800), Orientation::South, 70, 280, 0), (430, 520, 570, 800));
        assert_eq!(pin_rect((0, 400), Orientation::East, 70, 280, 20), (-20, 330, 280, 470));
        assert_eq!(pin_rect((1000, 400), Orientation::West, 70, 280, 20), (720, 330, 1020, 470));
        assert_eq!(top_layer_rect((500, 400), 141, 300), (430, 250, 570, 550));
        assert_eq!(area((430, 0, 570, 280)), 140 * 280);
    }
}
