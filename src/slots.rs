// SPDX-License-Identifier: Apache-2.0
//! Slots — the legal positions an IO pin may occupy on the die boundary.
//!
//! This is the foundation of pin placement: every later stage is an *assignment onto these slots*,
//! so a slot list that differs from upstream's makes every downstream comparison meaningless, and
//! one that matches makes the rest checkable a stage at a time.
//!
//! A slot is a routing track on a boundary edge, minus three things: the corners (a pin too close
//! to one cannot be routed out), the boundary itself (a wide pin must not overhang it), and any
//! spacing the caller asked for between pins.
//!
//! Nothing here touches a database.
//!
//! # Provenance
//!
//! Rules **P1**…**P8**, reimplemented from the behaviour of OpenROAD's `IOPlacer` slot
//! generation. Nothing is copied from it.

/// Which boundary edge a slot sits on.
///
/// The order is the order slots are generated in, and it matters: it is what makes the slot list
/// run counter-clockwise around the die, which later stages rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Edge {
    Bottom,
    Right,
    Top,
    Left,
    /// **No edge at all** — a top-layer position, on a lattice inside the die.
    ///
    /// Not a placeholder: it is what stops every direction-dependent rule from firing where
    /// direction is meaningless. A top-layer pin has no edge to run along, so it cannot be ordered
    /// within a group and cannot be reflected to an opposite side.
    Invalid,
}

impl Edge {
    /// Does a pin on this edge run vertically? True for the horizontal edges, whose pins point up
    /// or down out of the die — and so the edges fed from `-ver_layers`.
    pub fn is_vertical_pin(self) -> bool {
        matches!(self, Edge::Bottom | Edge::Top)
    }

    /// Is this a position on the die boundary at all?
    pub fn is_boundary(self) -> bool {
        !matches!(self, Edge::Invalid)
    }

    /// Slots on these edges are generated ascending and then reversed, which is what carries the
    /// list counter-clockwise.
    pub fn is_reversed(self) -> bool {
        matches!(self, Edge::Top | Edge::Left)
    }
}

/// One track pattern on a layer: tracks at `origin + k * step`, for `k` in `0..count`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackPattern {
    pub origin: i32,
    pub count: i32,
    pub step: i32,
}

/// What one layer contributes to slot generation on one *orientation* of edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerTracks {
    pub layer: String,
    /// The patterns running along the edge, in the order the technology declares them.
    pub patterns: Vec<TrackPattern>,
    /// The layer's minimum pin width **in the direction that matters for this list**: the caller
    /// supplies the X width for the vertical-pin list and the Y width for the horizontal-pin one.
    /// It decides how far a pin's edge sticks out from its centre.
    pub min_width: i32,
}

/// The knobs `place_pins` exposes that reach slot generation.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// `-min_distance`. **Zero means "not set"**, which is not the same as "no spacing": unset
    /// puts candidate slots every `DEFAULT_MIN_DIST` tracks, while an explicit value puts one on
    /// every track and filters afterwards. The two produce genuinely different slot sets.
    pub min_distance: i32,
    /// `-min_distance_in_tracks` — read `min_distance` as a count of candidates, not a length.
    pub min_distance_in_tracks: bool,
    /// `-corner_avoidance`. **Negative means "not set"**, which defaults to `NUM_TRACKS_OFFSET`
    /// tracks, capped at 1 µm.
    pub corner_avoidance: i32,
    /// `set_pin_thick_multiplier`, which widens a pin and so pushes the outermost slots inward.
    pub thickness_multiplier_h: f64,
    pub thickness_multiplier_v: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            min_distance: 0,
            min_distance_in_tracks: false,
            corner_avoidance: -1,
            thickness_multiplier_h: 1.0,
            thickness_multiplier_v: 1.0,
        }
    }
}

/// Candidate slots are placed every this many tracks when `-min_distance` is not given.
pub const DEFAULT_MIN_DIST: i32 = 2;
/// Tracks kept clear of each corner when `-corner_avoidance` is not given.
///
/// ⚠️ **Fifteen, not two.** On any ordinary pitch this is far past the 1 µm cap, so the default
/// avoidance is simply *1 µm* and the track count never shows. It only matters on a very coarse
/// layer — which is exactly why guessing a smaller number went unnoticed: every case that pinned
/// `-corner_avoidance` explicitly agreed, and the ones that did not were only ever checked by a
/// necessary condition that a too-permissive slot list satisfies.
pub const NUM_TRACKS_OFFSET: i32 = 15;

/// One legal pin position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub x: i32,
    pub y: i32,
    pub layer: String,
    pub edge: Edge,
    /// Something in the design already occupies this position, so no pin may take it. It stays in
    /// the list regardless — see [`define_slots`].
    pub blocked: bool,
}

/// The rectangle slots are generated on: the **die** area.
///
/// ⚠️ Upstream reaches this through a type named `Core`, and `Core::getBoundary()` reads like the
/// core area — but `initCore` builds it from `getDieArea()`. IO pins belong on the die edge, so
/// the value is right and only the name misleads. Taking the core area instead would place every
/// pin inside the block by the core-to-die margin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Boundary {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

/// **P1** — the corner avoidance actually used, resolving "not set".
///
/// Unset means [`NUM_TRACKS_OFFSET`] tracks' worth, **capped at 1 µm**. In practice the cap almost
/// always wins, so "the default corner avoidance" is 1 µm for any normal pitch; the track count
/// only takes over on a layer coarse enough that fifteen tracks is less than a micron.
pub fn corner_avoidance(params: &Params, track_step: i32, dbu_per_micron: i32) -> i32 {
    if params.corner_avoidance >= 0 {
        return params.corner_avoidance;
    }
    (NUM_TRACKS_OFFSET * track_step).min(dbu_per_micron)
}

/// **P2** — the spacing between *candidate* slots on one track pattern.
///
/// Unset (`0`) does **not** mean "every track": it means every `DEFAULT_MIN_DIST` tracks. An
/// explicit value means every track here, with the requested spacing enforced afterwards in
/// [`enforce_spacing`].
pub fn slot_step(params: &Params, track_step: i32) -> i32 {
    if params.min_distance == 0 {
        DEFAULT_MIN_DIST * track_step
    } else {
        track_step
    }
}

/// **P3, P4, P5** — the candidate slots one layer contributes to one edge, sorted along it.
///
/// The first and last slots are pulled inward by **half the pin's width** as well as by the corner
/// avoidance. A wide pin centred on the outermost track would hang over the boundary, which is an
/// off-grid violation rather than a pin.
///
/// ⚠️ **P5 — the corner avoidance is resolved once, from the first pattern.** A layer with several
/// track patterns of different pitch computes its default avoidance from the *first* pattern's step
/// and reuses that figure for the rest, so patterns 2..n do not each get their own default. That is
/// a consequence of upstream caching the resolved value in a member; it is observable whenever a
/// layer carries mixed-pitch patterns, so we reproduce it rather than "fix" it.
pub fn layer_slots(
    tracks: &LayerTracks,
    edge: Edge,
    boundary: Boundary,
    params: &Params,
    dbu_per_micron: i32,
) -> Vec<(i32, i32)> {
    let vertical = edge.is_vertical_pin();
    let (min, max) = if vertical {
        (boundary.x0, boundary.x1)
    } else {
        (boundary.y0, boundary.y1)
    };
    let multiplier = if vertical {
        params.thickness_multiplier_v
    } else {
        params.thickness_multiplier_h
    };
    // Half a pin's width, rounded UP before the multiplier applies: half of an odd width still has
    // to clear the boundary. The multiplier then truncates, which is upstream's arithmetic.
    let half_width = ((tracks.min_width as f64 / 2.0).ceil() * multiplier) as i32;

    let fixed = match edge {
        Edge::Bottom => boundary.y0,
        Edge::Top => boundary.y1,
        Edge::Left => boundary.x0,
        Edge::Right => boundary.x1,
        // The boundary generator is never asked for a non-edge; a top-layer lattice comes from
        // `toplayer::slots` instead.
        Edge::Invalid => return Vec::new(),
    };

    // Resolved once, from the first usable pattern — see P5 above.
    let mut avoidance: Option<i32> = if params.corner_avoidance >= 0 {
        Some(params.corner_avoidance)
    } else {
        None
    };

    let mut out = Vec::new();
    for p in &tracks.patterns {
        if p.step <= 0 || p.count <= 0 {
            continue;
        }
        let step = slot_step(params, p.step);
        let avoid =
            *avoidance.get_or_insert_with(|| corner_avoidance(params, p.step, dbu_per_micron));
        let tracks_offset = (avoid as f64 / step as f64).ceil() as i32;

        let start = 0.0_f64.max(((min + half_width - p.origin) as f64 / step as f64).ceil()) as i32
            + tracks_offset;
        let end = (p.count - 1).min((max - half_width - p.origin) / step) - tracks_offset;

        for i in start..=end {
            let along = p.origin + i * step;
            out.push(if vertical { (along, fixed) } else { (fixed, along) });
        }
    }
    // Sorted along the edge, so patterns of different pitch interleave by position rather than
    // coming out grouped by which pattern produced them.
    out.sort_by_key(|&(x, y)| if vertical { x } else { y });
    out
}

/// **P6** — drop candidates that sit closer together than the caller asked.
///
/// ⚠️ **This runs before the list is reversed**, which is upstream's own note and the reason it is
/// a separate step: filtering after a reversal would keep a different subset on the top and left
/// edges than on the bottom and right, and a mirrored pin would then have no partner to mirror to.
///
/// Two readings of `-min_distance`, and they are genuinely different. As a **length**, a candidate
/// survives when it is far enough from the last one *kept*. As a **count of tracks**, every Nth
/// candidate survives regardless of distance.
pub fn enforce_spacing(slots: &[(i32, i32)], params: &Params) -> Vec<(i32, i32)> {
    if slots.is_empty() {
        return Vec::new();
    }
    // `-min_distance_in_tracks` with no distance is a modulo by zero upstream. Keeping everything
    // is the only reading of "every 0th candidate" that is not a trap.
    if params.min_distance_in_tracks && params.min_distance == 0 {
        return slots.to_vec();
    }

    let mut out = Vec::new();
    let mut last = slots[0];
    // Counts every candidate examined, kept or not — which is what makes the in-tracks reading a
    // stride over the candidates rather than over the survivors.
    let mut seen = 0;
    for &pos in slots {
        let keep = if params.min_distance_in_tracks {
            pos == last || seen % params.min_distance == 0
        } else {
            pos == last
                || (last.0 - pos.0).abs() >= params.min_distance
                || (last.1 - pos.1).abs() >= params.min_distance
        };
        if keep {
            last = pos;
            out.push(pos);
        }
        seen += 1;
    }
    out
}

/// A stretch of one die edge, in DBU along that edge.
///
/// Used for both halves of "where a pin may go": a constraint region says *only here*, an excluded
/// region says *not here*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub edge: Edge,
    pub begin: i32,
    pub end: i32,
}

/// **P9** — is this position unusable?
///
/// Two independent reasons, and both have to be checked or a pin lands somewhere it cannot go.
///
/// **An excluded region** — `exclude_io_pin_region` — blocks positions along one edge.
/// ⚠️ The comparison is **strictly** inside the interval at both ends, so a slot sitting exactly on
/// a boundary is *not* excluded. That is upstream's shape, and off-by-one here silently changes
/// how many slots a region gives up.
///
/// **A fixed port's own metal** blocks whatever it covers, on its own layer — passed in already
/// grown by the pin keep-out (half a pin plus spacing along the edge, the pin depth plus spacing
/// across it), so a new pin keeps its distance as well as not landing on it; the test on the
/// grown box is inclusive. A port placed by `place_pin` is not ours to move.
///
/// **A layer's own blockage** — an obstruction or a power-grid shape near the edge
/// ([`crate::blocked`]) — blocks only positions on that layer, by the same strict test.
pub fn is_blocked(
    x: i32,
    y: i32,
    layer: &str,
    edge: Edge,
    exclusions: &[Interval],
    fixed_shapes: &[(String, i32, i32, i32, i32)],
    layered: &[(Interval, String)],
) -> bool {
    for (l, x0, y0, x1, y1) in fixed_shapes {
        if l == layer && (*x0..=*x1).contains(&x) && (*y0..=*y1).contains(&y) {
            return true;
        }
    }
    let along = if edge.is_vertical_pin() { x } else { y };
    let inside = |e: &Interval| e.edge == edge && along > e.begin.min(e.end) && along < e.begin.max(e.end);
    exclusions.iter().any(inside) || layered.iter().any(|(e, l)| l == layer && inside(e))
}

/// **P7, P8** — every slot on the die boundary, in the order the engine generates them.
///
/// The edge order (bottom, right, top, left) with the top and left reversed carries the list
/// counter-clockwise around the die. Later stages assign into *contiguous runs* of this list, so
/// the order is part of the contract and not a presentation detail.
///
/// `is_blocked` reports whether a position is already occupied. A blocked slot stays in the list
/// and is only marked: removing it would shift every slot after it and change the assignment.
pub fn define_slots(
    ver_layers: &[LayerTracks],
    hor_layers: &[LayerTracks],
    boundary: Boundary,
    params: &Params,
    dbu_per_micron: i32,
    is_blocked: &dyn Fn(i32, i32, &str) -> bool,
) -> Vec<Slot> {
    let mut out = Vec::new();
    for (edge, layers) in [
        (Edge::Bottom, ver_layers),
        (Edge::Right, hor_layers),
        (Edge::Top, ver_layers),
        (Edge::Left, hor_layers),
    ] {
        for tracks in layers {
            let candidates = layer_slots(tracks, edge, boundary, params, dbu_per_micron);
            if candidates.is_empty() {
                continue;
            }
            let mut kept = enforce_spacing(&candidates, params);
            if edge.is_reversed() {
                kept.reverse();
            }
            out.extend(kept.into_iter().map(|(x, y)| Slot {
                x,
                y,
                layer: tracks.layer.clone(),
                edge,
                blocked: is_blocked(x, y, &tracks.layer),
            }));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracks(step: i32, count: i32, origin: i32) -> LayerTracks {
        LayerTracks {
            layer: "met1".into(),
            patterns: vec![TrackPattern { origin, count, step }],
            min_width: 0,
        }
    }

    fn die() -> Boundary {
        Boundary { x0: 0, y0: 0, x1: 10_000, y1: 10_000 }
    }

    fn never_blocked() -> impl Fn(i32, i32, &str) -> bool {
        |_, _, _| false
    }

    #[test]
    fn unset_min_distance_means_every_other_track_not_every_track() {
        // The trap: 0 does not mean "no spacing".
        let p = Params { corner_avoidance: 0, ..Default::default() };
        assert_eq!(slot_step(&p, 100), 200, "unset spaces candidates two tracks apart");
        let explicit = Params { min_distance: 500, corner_avoidance: 0, ..Default::default() };
        assert_eq!(slot_step(&explicit, 100), 100, "an explicit value uses every track");
    }

    #[test]
    fn corner_avoidance_defaults_to_fifteen_tracks_but_is_capped_at_one_micron() {
        let unset = Params { corner_avoidance: -1, ..Default::default() };
        // ⚠️ On any ordinary pitch the CAP wins, which is why an understated track count hides:
        // 15 x 560 is 8400, and the answer is 2000 either way at 2000 DBU/um.
        assert_eq!(corner_avoidance(&unset, 560, 2000), 2000, "the cap wins on a normal pitch");
        // Only a very fine pitch lets the track count show.
        assert_eq!(corner_avoidance(&unset, 100, 2000), 1500, "15 tracks, under the cap");
        let set = Params { corner_avoidance: 42, ..Default::default() };
        assert_eq!(corner_avoidance(&set, 900, 1000), 42, "an explicit value wins, uncapped");
    }

    #[test]
    fn slots_sit_on_tracks_along_the_edge_at_the_edges_own_coordinate() {
        let p = Params { corner_avoidance: 0, ..Default::default() };
        let t = tracks(100, 101, 0); // 0..10000
        let bottom = layer_slots(&t, Edge::Bottom, die(), &p, 1000);
        assert!(!bottom.is_empty());
        assert!(bottom.iter().all(|&(_, y)| y == 0), "bottom slots sit on y = y0");
        assert!(bottom.iter().all(|&(x, _)| x % 200 == 0), "and on every other track");
        assert!(bottom.windows(2).all(|w| w[1].0 > w[0].0), "ascending along the edge");

        let left = layer_slots(&t, Edge::Left, die(), &p, 1000);
        assert!(left.iter().all(|&(x, _)| x == 0), "left slots sit on x = x0");
        let right = layer_slots(&t, Edge::Right, die(), &p, 1000);
        assert!(right.iter().all(|&(x, _)| x == 10_000), "right slots sit on x = x1");
    }

    #[test]
    fn a_wide_pin_gives_up_the_outermost_slots() {
        // Half the pin width must clear the boundary, or the pin hangs over it.
        let p = Params { corner_avoidance: 0, ..Default::default() };
        let thin = tracks(100, 101, 0);
        let wide = LayerTracks { min_width: 600, ..thin.clone() };

        let a = layer_slots(&thin, Edge::Bottom, die(), &p, 1000);
        let b = layer_slots(&wide, Edge::Bottom, die(), &p, 1000);
        assert!(b.len() < a.len(), "a wider pin has fewer legal slots");
        assert!(b[0].0 >= 300, "the first slot clears half the width: {:?}", b[0]);
        assert!(b.last().unwrap().0 <= 9_700, "and so does the last");
    }

    #[test]
    fn the_thickness_multiplier_widens_the_pin_and_narrows_the_run() {
        let base = Params { corner_avoidance: 0, ..Default::default() };
        let thick = Params { thickness_multiplier_v: 4.0, ..base.clone() };
        let t = LayerTracks { min_width: 200, ..tracks(100, 101, 0) };
        let a = layer_slots(&t, Edge::Bottom, die(), &base, 1000);
        let b = layer_slots(&t, Edge::Bottom, die(), &thick, 1000);
        assert!(b.len() < a.len(), "a thicker pin fits in fewer places");
    }

    #[test]
    fn the_multiplier_applies_to_the_already_rounded_half_width() {
        // An odd width rounds UP to a half-width first, and only then scales. Rounding after
        // scaling would give a smaller figure and a different first slot.
        let p = Params {
            corner_avoidance: 0,
            min_distance: 1,
            thickness_multiplier_v: 2.0,
            ..Default::default()
        };
        let t = LayerTracks { min_width: 101, ..tracks(1, 20_001, 0) };
        // ceil(101/2) = 51, then x2 = 102 — not ceil(101/2 * 2) = 101.
        let s = layer_slots(&t, Edge::Bottom, die(), &p, 1000);
        assert_eq!(s[0].0, 102, "first slot clears the scaled, pre-rounded half width");
    }

    #[test]
    fn corner_avoidance_removes_slots_from_both_ends() {
        let none = Params { corner_avoidance: 0, ..Default::default() };
        let avoid = Params { corner_avoidance: 1000, ..Default::default() };
        let t = tracks(100, 101, 0);
        let a = layer_slots(&t, Edge::Bottom, die(), &none, 1000);
        let b = layer_slots(&t, Edge::Bottom, die(), &avoid, 1000);
        assert!(b.len() + 8 <= a.len(), "1000 DBU at each end, at 200 per slot, removes ~10");
        assert!(b[0].0 > a[0].0 && b.last().unwrap().0 < a.last().unwrap().0);
    }

    #[test]
    fn the_default_corner_avoidance_is_resolved_once_from_the_first_pattern() {
        // P5, and a genuine quirk rather than a tidy rule: a second pattern of a different pitch
        // inherits the FIRST pattern's default avoidance. Reproduced deliberately — it is
        // observable, so "fixing" it would be a difference from upstream, not an improvement.
        let p = Params::default(); // corner_avoidance unset
        let mixed = LayerTracks {
            layer: "met1".into(),
            patterns: vec![
                TrackPattern { origin: 0, count: 200, step: 50 }, // default avoidance = 100
                TrackPattern { origin: 0, count: 20, step: 500 }, // would be 1000 on its own
            ],
            min_width: 0,
        };
        let got = layer_slots(&mixed, Edge::Bottom, die(), &p, 100_000);

        // The second pattern's own default (2 x 500 = 1000) would start it at 2000; inheriting
        // 100 from the first starts it at 1000.
        assert!(got.contains(&(1000, 0)), "second pattern inherited the first pattern's avoidance");

        // Swap the order and the inherited figure changes — which is what makes it observable.
        let swapped =
            LayerTracks { patterns: mixed.patterns.iter().rev().copied().collect(), ..mixed.clone() };
        let other = layer_slots(&swapped, Edge::Bottom, die(), &p, 100_000);
        assert_ne!(got, other, "pattern order changes the result, because avoidance is cached");
    }

    #[test]
    fn spacing_as_a_length_measures_from_the_last_slot_kept() {
        // Candidates every 100; asking for 250 keeps every third (0, 300, 600 …) because the
        // distance is measured from the last slot KEPT, not from the previous candidate.
        let p = Params { min_distance: 250, ..Default::default() };
        let cands: Vec<(i32, i32)> = (0..10).map(|i| (i * 100, 0)).collect();
        assert_eq!(enforce_spacing(&cands, &p), vec![(0, 0), (300, 0), (600, 0), (900, 0)]);
    }

    #[test]
    fn spacing_in_tracks_is_a_stride_over_candidates_not_a_distance() {
        // The other reading: every Nth candidate, whatever the distance between them.
        let p = Params { min_distance: 3, min_distance_in_tracks: true, ..Default::default() };
        let cands: Vec<(i32, i32)> = (0..10).map(|i| (i * 100, 0)).collect();
        assert_eq!(enforce_spacing(&cands, &p), vec![(0, 0), (300, 0), (600, 0), (900, 0)]);

        // With uneven candidates the two readings genuinely diverge.
        let uneven = vec![(0, 0), (10, 0), (20, 0), (1000, 0)];
        assert_eq!(enforce_spacing(&uneven, &p), vec![(0, 0), (1000, 0)]);
        let by_length = Params { min_distance: 3, ..Default::default() };
        assert_eq!(enforce_spacing(&uneven, &by_length).len(), 4, "all are 3 apart or more");
    }

    #[test]
    fn in_tracks_with_no_distance_keeps_everything_instead_of_dividing_by_zero() {
        // Upstream would take `% 0` here. Keeping every candidate is the only reading of
        // "every 0th" that is not a trap, and it is a deliberate divergence.
        let p = Params { min_distance: 0, min_distance_in_tracks: true, ..Default::default() };
        let cands: Vec<(i32, i32)> = (0..5).map(|i| (i * 100, 0)).collect();
        assert_eq!(enforce_spacing(&cands, &p), cands);
    }

    #[test]
    fn the_slot_list_travels_counter_clockwise_around_the_die() {
        // Later stages assign into contiguous runs of this list, so the order is a contract.
        let p = Params { corner_avoidance: 0, ..Default::default() };
        let t = vec![tracks(1000, 11, 0)];
        let slots = define_slots(&t, &t, die(), &p, 1000, &never_blocked());

        let edges: Vec<Edge> = slots.iter().map(|s| s.edge).collect();
        let first_of = |e: Edge| edges.iter().position(|x| *x == e).unwrap();
        assert!(first_of(Edge::Bottom) < first_of(Edge::Right));
        assert!(first_of(Edge::Right) < first_of(Edge::Top));
        assert!(first_of(Edge::Top) < first_of(Edge::Left));

        // Bottom runs left-to-right and top runs right-to-left, which is what counter-clockwise
        // means for the list as a whole.
        let along = |e: Edge, f: fn(&Slot) -> i32| {
            slots.iter().filter(|s| s.edge == e).map(f).collect::<Vec<_>>()
        };
        assert!(along(Edge::Bottom, |s| s.x).windows(2).all(|w| w[1] > w[0]), "bottom ascends");
        assert!(along(Edge::Right, |s| s.y).windows(2).all(|w| w[1] > w[0]), "right ascends");
        assert!(along(Edge::Top, |s| s.x).windows(2).all(|w| w[1] < w[0]), "top descends");
        assert!(along(Edge::Left, |s| s.y).windows(2).all(|w| w[1] < w[0]), "left descends");
    }

    #[test]
    fn spacing_is_applied_before_reversal_so_the_two_sides_agree() {
        // Upstream's own note, and the reason the filter is its own step: filter after reversing
        // and the top edge keeps a different subset than the bottom, leaving a mirrored pin with
        // no partner. The two edges must hold the same positions.
        let p = Params { min_distance: 250, corner_avoidance: 0, ..Default::default() };
        let t = vec![tracks(100, 101, 0)];
        let slots = define_slots(&t, &t, die(), &p, 1000, &never_blocked());

        let sorted = |e: Edge, f: fn(&Slot) -> i32| {
            let mut v: Vec<i32> = slots.iter().filter(|s| s.edge == e).map(f).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(
            sorted(Edge::Bottom, |s| s.x),
            sorted(Edge::Top, |s| s.x),
            "every bottom slot has a partner directly above it"
        );
        assert_eq!(sorted(Edge::Left, |s| s.y), sorted(Edge::Right, |s| s.y));
    }

    #[test]
    fn blocked_positions_are_kept_but_marked() {
        // A blocked slot still occupies its place in the list — dropping it would shift every
        // later slot and change the assignment.
        let p = Params { corner_avoidance: 0, ..Default::default() };
        let t = vec![tracks(1000, 11, 0)];
        let all = define_slots(&t, &t, die(), &p, 1000, &never_blocked());
        let some = define_slots(&t, &t, die(), &p, 1000, &|x, _, _| x == 2000);

        assert_eq!(all.len(), some.len(), "the list is the same length");
        assert!(some.iter().any(|s| s.blocked), "and the blocked ones are marked");
        assert!(all.iter().all(|s| !s.blocked));
    }

    #[test]
    fn several_patterns_on_a_layer_interleave_in_position_order() {
        // A layer may carry patterns of different pitch; the slots must come out sorted along the
        // edge, not grouped by which pattern produced them.
        let p = Params { min_distance: 1, corner_avoidance: 0, ..Default::default() };
        let t = LayerTracks {
            layer: "met1".into(),
            patterns: vec![
                TrackPattern { origin: 0, count: 3, step: 1000 },
                TrackPattern { origin: 500, count: 3, step: 1000 },
            ],
            min_width: 0,
        };
        let xs: Vec<i32> =
            layer_slots(&t, Edge::Bottom, die(), &p, 1000).iter().map(|s| s.0).collect();
        assert_eq!(xs, vec![0, 500, 1000, 1500, 2000, 2500]);
    }

    #[test]
    fn several_layers_each_contribute_their_own_slots_at_the_same_positions() {
        // Two layers on the same edge is normal, and their slots coexist: a position is not
        // consumed by the first layer that offers it.
        let p = Params { corner_avoidance: 0, ..Default::default() };
        let t = vec![
            LayerTracks { layer: "met2".into(), ..tracks(1000, 11, 0) },
            LayerTracks { layer: "met4".into(), ..tracks(1000, 11, 0) },
        ];
        let slots = define_slots(&t, &[], die(), &p, 1000, &never_blocked());
        let m2: Vec<i32> = slots.iter().filter(|s| s.layer == "met2").map(|s| s.x).collect();
        let m4: Vec<i32> = slots.iter().filter(|s| s.layer == "met4").map(|s| s.x).collect();
        assert!(!m2.is_empty() && m2 == m4, "same positions offered on both layers");
    }

    #[test]
    fn a_track_pattern_reaching_past_the_boundary_is_clipped_not_wrapped() {
        // count says 40 tracks but the die ends at 10000, so the run stops there.
        let p = Params { min_distance: 1, corner_avoidance: 0, ..Default::default() };
        let t = tracks(1000, 40, 0);
        let xs: Vec<i32> =
            layer_slots(&t, Edge::Bottom, die(), &p, 1000).iter().map(|s| s.0).collect();
        assert_eq!(*xs.last().unwrap(), 10_000);
        assert_eq!(xs.len(), 11);
    }

    #[test]
    fn a_pattern_starting_before_the_die_begins_inside_it() {
        // A pattern may start outside the die (a shifted origin), so the first legal index is not 0.
        let p = Params { min_distance: 1, corner_avoidance: 0, ..Default::default() };
        let t = tracks(1000, 40, -5_000);
        let xs: Vec<i32> =
            layer_slots(&t, Edge::Bottom, die(), &p, 1000).iter().map(|s| s.0).collect();
        assert_eq!(xs[0], 0, "the first slot is the first track inside the die");
        assert!(xs.iter().all(|&x| (0..=10_000).contains(&x)));
    }

    #[test]
    fn an_edge_too_short_for_any_slot_yields_none_rather_than_a_negative_range() {
        // Avoidance eating the whole edge must produce an empty list, not a panic or a reversed
        // range that wraps.
        let p = Params { corner_avoidance: 9_000, ..Default::default() };
        let t = tracks(1000, 11, 0);
        assert!(layer_slots(&t, Edge::Bottom, die(), &p, 1000).is_empty());
    }

    #[test]
    fn an_excluded_region_blocks_positions_strictly_inside_it() {
        // ⚠️ Strict at BOTH ends: a slot exactly on the boundary is still usable. Off by one here
        // and a region silently gives up two more slots than it should.
        let ex = [Interval { edge: Edge::Bottom, begin: 200, end: 500 }];
        let b = |x| is_blocked(x, 0, "met1", Edge::Bottom, &ex, &[], &[]);
        assert!(!b(200), "the lower boundary is not excluded");
        assert!(b(201) && b(499));
        assert!(!b(500), "nor the upper one");
        assert!(!b(600));
    }

    #[test]
    fn an_exclusion_only_applies_to_its_own_edge_and_its_own_axis() {
        let ex = [Interval { edge: Edge::Bottom, begin: 200, end: 500 }];
        assert!(!is_blocked(300, 0, "met1", Edge::Top, &ex, &[], &[]), "a different edge is untouched");
        // On a vertical edge the coordinate compared is y, not x.
        let left = [Interval { edge: Edge::Left, begin: 200, end: 500 }];
        assert!(is_blocked(0, 300, "met1", Edge::Left, &left, &[], &[]));
        assert!(!is_blocked(300, 0, "met1", Edge::Left, &left, &[], &[]));
    }

    #[test]
    fn a_reversed_exclusion_interval_still_blocks_the_same_stretch() {
        let ex = [Interval { edge: Edge::Bottom, begin: 500, end: 200 }];
        assert!(is_blocked(300, 0, "met1", Edge::Bottom, &ex, &[], &[]));
    }

    #[test]
    fn a_fixed_ports_metal_blocks_its_own_layer_only() {
        // place_pin puts a port somewhere explicit; a later pin on the same slot shorts to it.
        let fixed = [("met1".to_string(), 100, -50, 300, 50)];
        assert!(is_blocked(200, 0, "met1", Edge::Bottom, &[], &fixed, &[]));
        assert!(!is_blocked(200, 0, "met2", Edge::Bottom, &[], &fixed, &[]), "a different layer is free");
        assert!(!is_blocked(400, 0, "met1", Edge::Bottom, &[], &fixed, &[]), "and so is a clear slot");
    }

    #[test]
    fn nothing_declared_blocks_nothing() {
        assert!(!is_blocked(200, 0, "met1", Edge::Bottom, &[], &[], &[]));
    }

    #[test]
    fn a_degenerate_pattern_contributes_nothing_rather_than_panicking() {
        let p = Params::default();
        for bad in [
            TrackPattern { origin: 0, count: 0, step: 100 },
            TrackPattern { origin: 0, count: 10, step: 0 },
            TrackPattern { origin: 0, count: 10, step: -5 },
        ] {
            let t = LayerTracks { layer: "m".into(), patterns: vec![bad], min_width: 0 };
            assert!(layer_slots(&t, Edge::Bottom, die(), &p, 1000).is_empty(), "{bad:?}");
        }
        assert!(enforce_spacing(&[], &p).is_empty());
        assert!(define_slots(&[], &[], die(), &p, 1000, &never_blocked()).is_empty());
    }
}
