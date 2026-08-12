// SPDX-License-Identifier: Apache-2.0
//! Polygon dies — a boundary made of arbitrarily many rectilinear edges.
//!
//! A rectangular die has four edges and every rule can name them: *bottom*, *right*, *top*, *left*.
//! A notched or L-shaped die has eight, twelve, twenty — and no name for any of them. So the whole
//! boundary path is re-expressed over **line segments**: an edge is a segment, its direction comes
//! from the order of its two endpoints, and everything that was decided by an edge's *name* is
//! decided by its *geometry* instead.
//!
//! What survives unchanged is the important half. Slots are still tracks along an edge, still
//! inset by corner avoidance and half a pin width; sections are still runs of the slot list; and
//! the matching is the same matching. Only the enumeration of edges changes.
//!
//! Nothing here touches a database.

use crate::sections::Section;
use crate::slots::{layer_slots, Edge, LayerTracks, Params, Slot};

/// One boundary segment, from `a` to `b`. Axis-aligned: a rectilinear outline has no other kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub a: (i32, i32),
    pub b: (i32, i32),
}

impl Segment {
    /// Does this segment carry pins that point up or down — i.e. is it horizontal?
    ///
    /// The same question `Edge::is_vertical_pin` answers for a named edge, asked of geometry.
    pub fn is_vertical_pin(&self) -> bool {
        self.a.1 == self.b.1
    }

    /// Does the segment run backwards along its axis?
    ///
    /// This replaces the "top and left edges are reversed" rule. On a rectangle those are exactly
    /// the edges whose traversal runs down in coordinate, so the geometric test generalises the
    /// named one rather than approximating it.
    pub fn is_reversed(&self) -> bool {
        if self.is_vertical_pin() {
            self.a.0 > self.b.0
        } else {
            self.a.1 > self.b.1
        }
    }

    /// The span along the segment, low to high.
    pub fn span(&self) -> (i32, i32) {
        if self.is_vertical_pin() {
            (self.a.0.min(self.b.0), self.a.0.max(self.b.0))
        } else {
            (self.a.1.min(self.b.1), self.a.1.max(self.b.1))
        }
    }

    /// The coordinate the segment holds fixed — the one every slot on it shares.
    pub fn fixed(&self) -> i32 {
        if self.is_vertical_pin() {
            self.a.1
        } else {
            self.a.0
        }
    }

    /// Does a point lie on this segment?
    pub fn contains(&self, at: (i32, i32)) -> bool {
        let (lo, hi) = self.span();
        if self.is_vertical_pin() {
            at.1 == self.fixed() && (lo..=hi).contains(&at.0)
        } else {
            at.0 == self.fixed() && (lo..=hi).contains(&at.1)
        }
    }
}

/// **Y1** — is this outline a real polygon, or just a rectangle written out?
///
/// ⚠️ **Five points is a rectangle** — the ring repeats its first point to close. More than five is
/// a genuine outline. This is the reference's own branch condition, not a description, so the
/// count matters exactly.
pub fn is_polygon(points: &[(i32, i32)]) -> bool {
    points.len() > 5
}

/// **Y2** — the boundary segments, in the order the engine walks them.
///
/// ⚠️ **The ring is walked BACKWARDS**: segments are `(p[n], p[n-1])`, `(p[n-1], p[n-2])`, down to
/// `(p[1], p[0])`. Walking it forwards produces the same set of segments with every one of them
/// reversed, which flips the direction of the slot list on every edge — and the slot list's order
/// is a contract that later stages assign into.
pub fn segments(points: &[(i32, i32)]) -> Vec<Segment> {
    if points.len() < 2 {
        return Vec::new();
    }
    (1..points.len())
        .rev()
        .map(|i| Segment { a: points[i], b: points[i - 1] })
        .filter(|s| s.a != s.b)
        .collect()
}

/// **Y3** — every slot on a polygon boundary.
///
/// Each segment is offered the layers whose pins point out of it: the vertical-pin layers for a
/// horizontal segment, the horizontal-pin layers for a vertical one. Within a segment the
/// arithmetic is [`layer_slots`] unchanged — the span comes from the segment's endpoints instead of
/// the die's, and that is the only difference.
///
/// Slots carry [`Edge::Invalid`]: a polygon segment is not one of the four named edges, and the
/// rules that key off those names must not fire. `on_segment` records which segment each slot came
/// from, because sections are cut per segment.
pub fn define_slots(
    points: &[(i32, i32)],
    ver_layers: &[LayerTracks],
    hor_layers: &[LayerTracks],
    params: &Params,
    dbu_per_micron: i32,
    is_blocked: &dyn Fn(i32, i32, &str) -> bool,
) -> (Vec<Slot>, Vec<usize>) {
    let mut out = Vec::new();
    let mut on_segment = Vec::new();

    for (idx, seg) in segments(points).iter().enumerate() {
        let layers = if seg.is_vertical_pin() { ver_layers } else { hor_layers };
        let (lo, hi) = seg.span();
        // `layer_slots` reads the span from a rectangle, so the segment is handed to it as one:
        // a degenerate box along the segment's own axis.
        let bounds = if seg.is_vertical_pin() {
            crate::slots::Boundary { x0: lo, y0: seg.fixed(), x1: hi, y1: seg.fixed() }
        } else {
            crate::slots::Boundary { x0: seg.fixed(), y0: lo, x1: seg.fixed(), y1: hi }
        };
        let edge = if seg.is_vertical_pin() { Edge::Bottom } else { Edge::Left };

        for tracks in layers {
            let candidates = layer_slots(tracks, edge, bounds, params, dbu_per_micron);
            if candidates.is_empty() {
                continue;
            }
            // ⚠️ Spacing is enforced before the reversal here too, for the same reason as on a
            // rectangle: mirrored positions must exist on both of a pair of facing segments.
            let mut kept = crate::slots::enforce_spacing(&candidates, params);
            if seg.is_reversed() {
                kept.reverse();
            }
            for (x, y) in kept {
                out.push(Slot {
                    x,
                    y,
                    layer: tracks.layer.clone(),
                    edge: Edge::Invalid,
                    blocked: is_blocked(x, y, &tracks.layer),
                });
                on_segment.push(idx);
            }
        }
    }
    (out, on_segment)
}

/// **Y4** — sections, cut per segment and per layer.
///
/// A run of the slot list belongs to one segment on one layer, which is the polygon equivalent of
/// "one edge, one layer" — and it is why [`define_slots`] returns which segment each slot came
/// from. Two collinear segments meeting at a straight join are still *two* runs, because upstream
/// cuts by segment identity rather than by geometry.
pub fn create_sections(
    slots: &[Slot],
    on_segment: &[usize],
    slots_per_section: usize,
) -> Vec<Section> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < slots.len() {
        let (seg, layer) = (on_segment[i], slots[i].layer.clone());
        let mut j = i;
        while j < slots.len() && on_segment[j] == seg && slots[j].layer == layer {
            j += 1;
        }
        out.extend(crate::sections::find_sections(
            slots,
            i,
            j - 1,
            Edge::Invalid,
            slots_per_section,
        ));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slots::TrackPattern;

    /// An L-shaped die: a 1000x1000 square with the top-right quarter removed.
    fn l_shape() -> Vec<(i32, i32)> {
        vec![(0, 0), (1000, 0), (1000, 500), (500, 500), (500, 1000), (0, 1000), (0, 0)]
    }

    fn rect() -> Vec<(i32, i32)> {
        vec![(0, 0), (1000, 0), (1000, 1000), (0, 1000), (0, 0)]
    }

    fn tracks(step: i32) -> Vec<LayerTracks> {
        vec![LayerTracks {
            layer: "met1".into(),
            patterns: vec![TrackPattern { origin: 0, count: 201, step }],
            min_width: 0,
        }]
    }

    fn params() -> Params {
        Params { min_distance: 1, corner_avoidance: 0, ..Default::default() }
    }

    #[test]
    fn five_points_is_a_rectangle_not_a_polygon() {
        // The reference's own branch condition, and it turns on the count exactly.
        assert!(!is_polygon(&rect()), "a closed rectangle is five points");
        assert!(is_polygon(&l_shape()));
        assert!(!is_polygon(&[]));
    }

    #[test]
    fn the_ring_is_walked_backwards() {
        // Forwards gives the same segments with every one reversed, which flips the slot order on
        // every edge — and that order is a contract.
        // The ring closes, so the last point repeats the first: the walk starts at the closing
        // point and steps down, giving the left edge first and the bottom edge last.
        let s = segments(&rect());
        assert_eq!(s.len(), 4);
        assert_eq!(s[0], Segment { a: (0, 0), b: (0, 1000) }, "from the closing point, backwards");
        assert_eq!(s[1], Segment { a: (0, 1000), b: (1000, 1000) });
        assert_eq!(s[3], Segment { a: (1000, 0), b: (0, 0) });
        // Forwards would give every one of these reversed.
        let forwards: Vec<Segment> = rect()
            .windows(2)
            .map(|w| Segment { a: w[0], b: w[1] })
            .collect();
        assert!(s.iter().all(|x| forwards.contains(&Segment { a: x.b, b: x.a })));
    }

    #[test]
    fn a_degenerate_repeat_is_not_a_segment() {
        let with_dup = vec![(0, 0), (500, 0), (500, 0), (500, 500), (0, 0)];
        assert!(segments(&with_dup).iter().all(|s| s.a != s.b));
        assert!(segments(&[(1, 1)]).is_empty());
    }

    #[test]
    fn a_segments_orientation_and_direction_come_from_its_endpoints() {
        let horiz = Segment { a: (0, 500), b: (900, 500) };
        assert!(horiz.is_vertical_pin(), "a horizontal segment carries vertical pins");
        assert!(!horiz.is_reversed());
        assert_eq!(horiz.span(), (0, 900));
        assert_eq!(horiz.fixed(), 500);

        let back = Segment { a: (900, 500), b: (0, 500) };
        assert!(back.is_reversed(), "running backwards along its axis");
        assert_eq!(back.span(), (0, 900), "the span is still low to high");

        let vert = Segment { a: (500, 0), b: (500, 900) };
        assert!(!vert.is_vertical_pin());
        assert_eq!(vert.fixed(), 500);
    }

    #[test]
    fn containment_is_along_the_segment_only() {
        let s = Segment { a: (0, 500), b: (900, 500) };
        assert!(s.contains((450, 500)));
        assert!(s.contains((0, 500)) && s.contains((900, 500)), "endpoints count");
        assert!(!s.contains((450, 501)), "off the line");
        assert!(!s.contains((950, 500)), "past the end");
    }

    #[test]
    fn an_l_shaped_die_gets_slots_on_all_six_edges() {
        let (slots, on_seg) = define_slots(
            &l_shape(),
            &tracks(50),
            &tracks(50),
            &params(),
            1000,
            &|_, _, _| false,
        );
        assert!(!slots.is_empty());
        assert_eq!(slots.len(), on_seg.len());
        let used: std::collections::BTreeSet<usize> = on_seg.iter().copied().collect();
        assert_eq!(used.len(), 6, "every segment of the L carries slots");

        // The notch is real: no slot lies inside the removed quarter.
        assert!(
            !slots.iter().any(|s| s.x > 500 && s.y > 500),
            "a slot landed in the cut-out corner"
        );
    }

    #[test]
    fn every_slot_lies_on_the_segment_it_is_attributed_to() {
        // The attribution is what sections are cut by, so a slot on the wrong segment silently
        // joins the wrong run.
        let segs = segments(&l_shape());
        let (slots, on_seg) = define_slots(
            &l_shape(),
            &tracks(50),
            &tracks(50),
            &params(),
            1000,
            &|_, _, _| false,
        );
        for (s, &i) in slots.iter().zip(&on_seg) {
            assert!(segs[i].contains((s.x, s.y)), "{:?} is not on segment {:?}", (s.x, s.y), segs[i]);
        }
    }

    #[test]
    fn polygon_slots_belong_to_no_named_edge() {
        let (slots, _) =
            define_slots(&l_shape(), &tracks(50), &tracks(50), &params(), 1000, &|_, _, _| false);
        assert!(slots.iter().all(|s| s.edge == Edge::Invalid));
    }

    #[test]
    fn a_reversed_segment_runs_its_slots_backwards() {
        // The geometric generalisation of "top and left are reversed".
        let (slots, on_seg) =
            define_slots(&rect(), &tracks(50), &tracks(50), &params(), 1000, &|_, _, _| false);
        let segs = segments(&rect());
        for (i, seg) in segs.iter().enumerate() {
            let along: Vec<i32> = slots
                .iter()
                .zip(&on_seg)
                .filter(|(_, &s)| s == i)
                .map(|(s, _)| if seg.is_vertical_pin() { s.x } else { s.y })
                .collect();
            if along.len() < 2 {
                continue;
            }
            let ascending = along.windows(2).all(|w| w[1] > w[0]);
            assert_eq!(ascending, !seg.is_reversed(), "segment {i} {seg:?} runs the wrong way");
        }
    }

    #[test]
    fn sections_are_cut_per_segment() {
        let (slots, on_seg) =
            define_slots(&l_shape(), &tracks(50), &tracks(50), &params(), 1000, &|_, _, _| false);
        let secs = create_sections(&slots, &on_seg, 200);
        assert!(!secs.is_empty());
        for sec in &secs {
            let segs: std::collections::BTreeSet<usize> =
                (sec.begin_slot..=sec.end_slot).map(|i| on_seg[i]).collect();
            assert_eq!(segs.len(), 1, "a section spans two segments");
            let layers: std::collections::BTreeSet<&str> =
                (sec.begin_slot..=sec.end_slot).map(|i| slots[i].layer.as_str()).collect();
            assert_eq!(layers.len(), 1, "a section spans two layers");
        }
    }

    #[test]
    fn a_rectangle_expressed_as_a_polygon_still_produces_slots_on_four_edges() {
        // The polygon path must not be a special case that only works for odd shapes.
        let (slots, on_seg) =
            define_slots(&rect(), &tracks(50), &tracks(50), &params(), 1000, &|_, _, _| false);
        assert_eq!(on_seg.iter().collect::<std::collections::BTreeSet<_>>().len(), 4);
        assert!(slots.iter().all(|s| s.x == 0 || s.x == 1000 || s.y == 0 || s.y == 1000));
    }

    #[test]
    fn nothing_to_place_is_not_an_error() {
        let (slots, on_seg) =
            define_slots(&[], &tracks(50), &tracks(50), &params(), 1000, &|_, _, _| false);
        assert!(slots.is_empty() && on_seg.is_empty());
        assert!(create_sections(&[], &[], 200).is_empty());
    }
}
