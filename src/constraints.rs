// SPDX-License-Identifier: Apache-2.0
//! Constraints — pins the design insists go somewhere particular.
//!
//! `set_io_pin_constraint -region left:10-40` says a named set of pins must land on the left edge
//! between 10 µm and 40 µm. That is a hard requirement, not a preference, and it changes the shape
//! of the problem: constrained pins are placed **first**, into sections cut from their own region
//! only, and the slots they take are then withdrawn before anything else is placed.
//!
//! Doing it the other way round — free pins first — lets a free pin sit in a region a constrained
//! pin has no alternative to, and the constraint then cannot be met at all.
//!
//! Nothing here touches a database.

use crate::groups::Group;
use crate::sections::{assign_pins_to_sections, find_sections, Pin, Placement, Section};
use crate::slots::{Edge, Interval, Slot};

/// One region constraint: a set of pins, and where they are allowed to go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub interval: Interval,
    /// Indices into the pin list, in database port order.
    pub pins: Vec<usize>,
}

/// **C1** — read a constraint rectangle as an interval on an edge.
///
/// 🔑 **The rectangle's own shape says which kind of constraint it is.** A **degenerate**
/// rectangle — zero width or zero height — is an edge interval, and which edge is decided by
/// whether it sits on the die's low or high side. A rectangle with real area is a *top-layer*
/// region, an entirely different placement path, and returns `None` here rather than being
/// mistaken for an edge.
pub fn interval_from_rect(rect: (i32, i32, i32, i32), die: (i32, i32, i32, i32)) -> Option<Interval> {
    let (x0, y0, x1, y1) = rect;
    let (dx0, dy0, _, _) = die;
    if x0 == x1 {
        Some(Interval {
            edge: if dx0 == x0 { Edge::Left } else { Edge::Right },
            begin: y0,
            end: y1,
        })
    } else if y0 == y1 {
        Some(Interval {
            edge: if dy0 == y0 { Edge::Bottom } else { Edge::Top },
            begin: x0,
            end: x1,
        })
    } else {
        None
    }
}

/// The order constraints are considered in.
///
/// ⚠️ **Not the order slots are generated in.** Slots run bottom, right, top, left; constraints
/// run **top, bottom, left, right**, because they are keyed by an interval whose ordering follows
/// a differently-declared edge enum. Two orders in one engine is a trap worth naming: the two are
/// unrelated and neither can be substituted for the other.
fn edge_rank(e: Edge) -> u8 {
    match e {
        Edge::Top => 0,
        Edge::Bottom => 1,
        Edge::Left => 2,
        Edge::Right => 3,
        Edge::Invalid => 4,
    }
}

/// **C2** — group pins by the region they are constrained to.
///
/// Pins sharing an interval form one constraint, because they compete for the same slots and must
/// be matched together. `region_of` reports a pin's constraint rectangle, if it has one.
///
/// Mirrored pins are excluded: their constraint is derived from their partner rather than declared
/// by the design, and treating a derived constraint as a declared one places the pair twice.
pub fn collect(
    pins: &[Pin],
    die: (i32, i32, i32, i32),
    region_of: &dyn Fn(usize) -> Option<(i32, i32, i32, i32)>,
    is_mirrored: &dyn Fn(usize) -> bool,
) -> Vec<Constraint> {
    let mut out: Vec<Constraint> = Vec::new();
    for i in 0..pins.len() {
        if is_mirrored(i) {
            continue;
        }
        let Some(rect) = region_of(i) else { continue };
        let Some(interval) = interval_from_rect(rect, die) else { continue };
        match out.iter_mut().find(|c| c.interval == interval) {
            Some(c) => c.pins.push(i),
            None => out.push(Constraint { interval, pins: vec![i] }),
        }
    }
    out.sort_by_key(|c| (edge_rank(c.interval.edge), c.interval.begin, c.interval.end));
    out
}

/// Do two constraints compete for the same stretch of the same edge?
pub fn overlapping(a: &Constraint, b: &Constraint) -> bool {
    a.interval.edge == b.interval.edge
        && a.interval.begin.max(b.interval.begin) <= a.interval.end.min(b.interval.end)
}

/// **C5** — reorder OVERLAPPING constraints by how tightly packed they are; leave the rest alone.
///
/// Two regions that share slots are decided by whoever goes first, so the order is part of the
/// answer. Where they do not overlap, the design's own order is kept — a constraint on the left
/// edge has no bearing on one at the bottom, and reordering them would be noise.
///
/// The ranking is *pins per available slot*, ascending: the constraint with the most room to spare
/// is served first.
///
/// ⚠️ **This predicate is not a total order** — it compares by pressure only when the two overlap,
/// so "equal" is not transitive across a chain of partially-overlapping regions. That is upstream's
/// shape, and it means three or more mutually-overlapping constraints can settle differently under
/// a different sort algorithm. Two is the case the suite exercises, and two is unambiguous.
pub fn sort_constraints(constraints: &mut [Constraint], slots: &[Slot], slots_per_section: usize) {
    let pressure: Vec<f64> = constraints
        .iter()
        .map(|c| {
            let available: usize = sections_for(slots, c.interval, slots_per_section)
                .iter()
                .map(|s| s.num_slots)
                .sum();
            if available == 0 {
                f64::INFINITY
            } else {
                c.pins.len() as f64 / available as f64
            }
        })
        .collect();

    // Insertion sort: stable, and it moves an element only past those it genuinely compares less
    // than, which is what keeps a non-overlapping neighbour in place.
    let mut order: Vec<usize> = (0..constraints.len()).collect();
    for i in 1..order.len() {
        let mut j = i;
        while j > 0 {
            let (a, b) = (order[j], order[j - 1]);
            let less = pressure[a] < pressure[b] && overlapping(&constraints[a], &constraints[b]);
            if !less {
                break;
            }
            order.swap(j, j - 1);
            j -= 1;
        }
    }
    let reordered: Vec<Constraint> = order.iter().map(|&i| constraints[i].clone()).collect();
    constraints.clone_from_slice(&reordered);
}

/// **C3** — the slots a constraint may use: its edge, its layers, inside its interval.
///
/// Half-open at the far end, which is upstream's shape and not an accident: the run stops at the
/// first slot that reaches the interval's end, so a slot sitting exactly on the boundary belongs
/// to the next region rather than this one.
pub fn sections_for(
    slots: &[Slot],
    interval: Interval,
    slots_per_section: usize,
) -> Vec<Section> {
    let along = |s: &Slot| if s.edge.is_vertical_pin() { s.x } else { s.y };
    let (lo, hi) = (interval.begin.min(interval.end), interval.begin.max(interval.end));

    let mut out = Vec::new();
    let mut i = 0;
    while i < slots.len() {
        let (edge, layer) = (slots[i].edge, slots[i].layer.clone());
        let mut j = i;
        while j < slots.len() && slots[j].edge == edge && slots[j].layer == layer {
            j += 1;
        }
        if edge == interval.edge {
            // Within this layer's run, the stretch that falls inside the interval. The run may be
            // ascending or descending along the edge, so it is filtered by value, not by index.
            let inside: Vec<usize> =
                (i..j).filter(|&k| along(&slots[k]) >= lo && along(&slots[k]) < hi).collect();
            if let (Some(&first), Some(&last)) = (inside.first(), inside.last()) {
                out.extend(find_sections(slots, first, last, edge, slots_per_section));
            }
        }
        i = j;
    }
    out
}

/// One placement round over a fixed set of sections: groups first, then the individual pins.
///
/// Public because a polygon die builds its own sections — from boundary segments rather than named
/// edges — and then needs exactly this, unchanged.
///
/// Both phases of [`place_with_constraints`] are the same round with a different set of sections —
/// a constraint's own region, or the whole boundary — so the sequencing lives in one place.
pub fn run_round(
    slots: &mut Vec<Slot>,
    sections: &mut Vec<Section>,
    pins: &[Pin],
    groups: &[Group],
    which_groups: &[usize],
    which_pins: &[usize],
    slots_per_section: usize,
    die: (i32, i32, i32, i32),
) -> (Vec<Placement>, Vec<usize>) {
    let (members, mut unplaced) = crate::sections::assign_groups_to_sections(
        groups,
        which_groups,
        sections,
        pins,
        slots,
        slots_per_section,
        die,
    );

    // A pin already spoken for by a group in this round does not also compete on its own, and
    // neither does a mirrored partner — it takes the reflection of whatever its pair receives.
    let grouped: std::collections::BTreeSet<usize> =
        which_groups.iter().flat_map(|&g| groups[g].pins.iter().copied()).collect();
    let partners: std::collections::BTreeSet<usize> = pins.iter().filter_map(|p| p.mirror).collect();
    let singles: Vec<usize> = which_pins
        .iter()
        .copied()
        .filter(|p| !grouped.contains(p) && !partners.contains(p))
        .collect();
    // 🔑 Real pin indices throughout — never a compacted subset. A `Pin` carries the index of its
    // mirrored partner, so a renumbered copy of the pin list makes that index point at the wrong
    // pin, or off the end of a shorter one.
    unplaced.extend(assign_pins_to_sections(pins, &singles, sections, die));

    let (placed, missed) =
        crate::sections::solve_sections(slots, sections, pins, groups, &members, die);
    unplaced.extend(missed);
    (placed, unplaced)
}

/// **C4** — place the constrained pins, then everything else.
///
/// The order is the whole point. Each constraint is satisfied inside its own region and the slots
/// it consumes are marked used, so a later constraint — and every unconstrained pin — sees a
/// boundary that has already given up what it owed.
///
/// A group counts as constrained when **every** one of its pins is: a group split across two
/// regions has no region of its own, and placing it into either would break the other constraint.
///
/// ⚠️ **`eligible` is which pins this boundary is responsible for**, and it is not optional. A pin
/// destined for the top layer must not compete here: it would take a boundary slot, be discarded
/// from the result, and leave that slot looking free while a real boundary pin was pushed onto a
/// worse one. The symptom is a placement that is *slightly* too expensive for no visible reason.
///
/// Returns the placements and the pins that could not be placed. A constrained pin that does not
/// fit its region is reported unplaced rather than quietly placed somewhere legal-looking: the
/// design asked for a region, and somewhere else is not a smaller version of that answer.
pub fn place_with_constraints(
    slots: &[Slot],
    pins: &[Pin],
    groups: &[Group],
    constraints: &[Constraint],
    eligible: &std::collections::BTreeSet<usize>,
    slots_per_section: usize,
    die: (i32, i32, i32, i32),
) -> (Vec<Placement>, Vec<usize>) {
    // A local copy, so consuming a slot is visible to every later stage.
    let mut slots: Vec<Slot> = slots.to_vec();
    let mut placed: Vec<Placement> = Vec::new();
    let mut unplaced: Vec<usize> = Vec::new();
    let mut done = vec![false; pins.len()];
    let mut group_done = vec![false; groups.len()];

    // **Fallback first.** A group larger than a whole section can never be assigned to one, so it
    // is placed directly, before anything else competes for the boundary. Doing it later would
    // mean searching for a long contiguous run through a boundary already peppered with pins.
    for (gi, g) in groups.iter().enumerate() {
        if g.pins.len() <= slots_per_section || !g.pins.iter().all(|p| eligible.contains(p)) {
            continue;
        }
        group_done[gi] = true;
        // A constrained oversized group is confined to its own region; upstream tries the middle
        // of that region first so the group sits centred rather than jammed against one end.
        let window = constraints
            .iter()
            .find(|c| g.pins.iter().all(|p| c.pins.contains(p)))
            .and_then(|c| {
                let secs = sections_for(&slots, c.interval, slots_per_section);
                Some((secs.first()?.begin_slot, secs.last()?.end_slot))
            });
        let start = match window {
            Some((first, last)) => {
                let mid = first + (last - first) / 2 - g.pins.len() / 2;
                crate::groups::first_run(&slots, mid, last, g.pins.len())
                    .or_else(|| crate::groups::first_run(&slots, first, last, g.pins.len()))
            }
            None => crate::groups::first_run(&slots, 0, slots.len() - 1, g.pins.len()),
        };
        match start {
            Some(at) => {
                let p = crate::groups::place_fallback(&mut slots, g, at);
                // Reflections are claimed here, beside the placement that caused them — the one
                // place they can be claimed exactly once.
                let (mirrors, failed) = crate::sections::expand_mirrors(&mut slots, pins, &p, die);
                for x in p.iter().chain(mirrors.iter()) {
                    done[x.pin] = true;
                }
                placed.extend(p);
                placed.extend(mirrors);
                unplaced.extend(failed);
            }
            None => unplaced.extend(g.pins.iter().copied()),
        }
    }

    // ⚠️ **Two passes over every constraint: mirrored pins first, then the rest.** A mirrored pin
    // needs two positions to be free at once — its own and the reflection — so it has far less
    // room to manoeuvre than a free pin. Running one constraint to completion before starting the
    // next lets ordinary pins in the first region consume slots a mirrored pair in a later region
    // has no alternative to.
    for mirrored_only in [true, false] {
        for c in constraints {
            // Only pins still waiting: a pin named by two constraints belongs to the first that
            // can take it, and upstream warns rather than placing it twice.
            let waiting: Vec<usize> = c
                .pins
                .iter()
                .copied()
                .filter(|&p| {
                    eligible.contains(&p)
                        && !done[p]
                        && pins[p].mirror.is_some() == mirrored_only
                })
                .collect();
            let mine: Vec<usize> = (0..groups.len())
                .filter(|&g| {
                    !group_done[g]
                        && groups[g].pins.iter().all(|p| c.pins.contains(p))
                        // A group counts as mirrored when a member CARRIES the mirror pointer —
                        // the driver of a pair, not merely either half of one.
                        && groups[g].pins.iter().any(|&p| pins[p].mirror.is_some())
                            == mirrored_only
                })
                .collect();
            if waiting.is_empty() && mine.is_empty() {
                continue;
            }
            for &g in &mine {
                group_done[g] = true;
            }

            let mut sections = sections_for(&slots, c.interval, slots_per_section);
            let (p, u) = run_round(
                &mut slots,
                &mut sections,
                pins,
                groups,
                &mine,
                &waiting,
                slots_per_section,
                die,
            );
            for x in &p {
                done[x.pin] = true;
            }
            placed.extend(p);
            unplaced.extend(u);
        }
    }

    // Everything left, over what the constraints did not take.
    let rest_groups: Vec<usize> = (0..groups.len()).filter(|&g| !group_done[g]).collect();
    let partners: std::collections::BTreeSet<usize> = pins.iter().filter_map(|p| p.mirror).collect();
    let rest_pins: Vec<usize> = (0..pins.len())
        .filter(|&p| {
            eligible.contains(&p) && !done[p] && !unplaced.contains(&p) && !partners.contains(&p)
        })
        .collect();
    if !rest_groups.is_empty() || !rest_pins.is_empty() {
        let mut sections = crate::sections::create_sections(&slots, slots_per_section);
        let (p, u) = run_round(
            &mut slots,
            &mut sections,
            pins,
            groups,
            &rest_groups,
            &rest_pins,
            slots_per_section,
            die,
        );
        placed.extend(p);
        unplaced.extend(u);
    }

    placed.sort_by_key(|p| p.pin);
    unplaced.sort_unstable();
    unplaced.dedup();
    (placed, unplaced)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIE: (i32, i32, i32, i32) = (0, 0, 1000, 1000);

    fn slot(x: i32, y: i32, edge: Edge) -> Slot {
        Slot { x, y, layer: "met1".into(), edge, blocked: false }
    }

    /// Slots all the way round a 1000x1000 die, one every 100 units.
    fn ring() -> Vec<Slot> {
        let mut v: Vec<Slot> = (0..11).map(|i| slot(i * 100, 0, Edge::Bottom)).collect();
        v.extend((0..11).map(|i| slot(1000, i * 100, Edge::Right)));
        v.extend((0..11).rev().map(|i| slot(i * 100, 1000, Edge::Top)));
        v.extend((0..11).rev().map(|i| slot(0, i * 100, Edge::Left)));
        v
    }

    /// Every pin is this boundary's responsibility — the ordinary case for a test.
    fn all(pins: &[Pin]) -> std::collections::BTreeSet<usize> {
        (0..pins.len()).collect()
    }

    fn pin(name: &str, sinks: &[(i32, i32)]) -> Pin {
        Pin { name: name.into(), sinks: sinks.to_vec(), mirror: None }
    }

    #[test]
    fn a_degenerate_rectangle_is_an_edge_interval_and_which_edge_is_its_position() {
        assert_eq!(
            interval_from_rect((0, 100, 0, 400), DIE),
            Some(Interval { edge: Edge::Left, begin: 100, end: 400 })
        );
        assert_eq!(
            interval_from_rect((1000, 100, 1000, 400), DIE),
            Some(Interval { edge: Edge::Right, begin: 100, end: 400 })
        );
        assert_eq!(
            interval_from_rect((100, 0, 400, 0), DIE),
            Some(Interval { edge: Edge::Bottom, begin: 100, end: 400 })
        );
        assert_eq!(
            interval_from_rect((100, 1000, 400, 1000), DIE),
            Some(Interval { edge: Edge::Top, begin: 100, end: 400 })
        );
    }

    #[test]
    fn a_rectangle_with_area_is_a_top_layer_region_not_an_edge() {
        // Reading it as an edge would silently pin a top-layer constraint to a boundary.
        assert_eq!(interval_from_rect((100, 100, 400, 400), DIE), None);
    }

    #[test]
    fn pins_sharing_a_region_become_one_constraint() {
        let pins = vec![pin("a", &[]), pin("b", &[]), pin("c", &[])];
        let region = |i: usize| match i {
            0 | 2 => Some((0, 100, 0, 400)),
            _ => None,
        };
        let c = collect(&pins, DIE, &region, &|_| false);
        assert_eq!(c.len(), 1, "two pins, one region, one constraint");
        assert_eq!(c[0].pins, vec![0, 2], "in database port order");
        assert_eq!(c[0].interval.edge, Edge::Left);
    }

    #[test]
    fn a_mirrored_pin_is_not_treated_as_carrying_a_declared_constraint() {
        // Its region is derived from its partner. Counting it here would place the pair twice.
        let pins = vec![pin("a", &[]), pin("b", &[])];
        let region = |_: usize| Some((0, 100, 0, 400));
        let c = collect(&pins, DIE, &region, &|i| i == 1);
        assert_eq!(c[0].pins, vec![0]);
    }

    #[test]
    fn constraints_are_ordered_by_edge_in_the_constraint_order_not_the_slot_order() {
        // The trap: slots run bottom, right, top, left. Constraints run top, bottom, left, right.
        let pins: Vec<Pin> = (0..4).map(|i| pin(&format!("p{i}"), &[])).collect();
        let region = |i: usize| {
            Some(match i {
                0 => (100, 0, 400, 0),       // bottom
                1 => (1000, 100, 1000, 400), // right
                2 => (100, 1000, 400, 1000), // top
                _ => (0, 100, 0, 400),       // left
            })
        };
        let c = collect(&pins, DIE, &region, &|_| false);
        let edges: Vec<Edge> = c.iter().map(|x| x.interval.edge).collect();
        assert_eq!(edges, vec![Edge::Top, Edge::Bottom, Edge::Left, Edge::Right]);
    }

    #[test]
    fn a_constraints_sections_cover_only_its_own_region() {
        let slots = ring();
        let interval = Interval { edge: Edge::Left, begin: 200, end: 600 };
        let sections = sections_for(&slots, interval, 200);
        assert!(!sections.is_empty());
        for s in &sections {
            for i in s.begin_slot..=s.end_slot {
                assert_eq!(slots[i].edge, Edge::Left, "wrong edge in a constraint section");
                assert!((200..600).contains(&slots[i].y), "slot at y={} is outside", slots[i].y);
            }
        }
    }

    #[test]
    fn the_far_end_of_a_region_belongs_to_the_next_one() {
        // Half-open, so two abutting regions do not both claim the slot between them.
        let slots = ring();
        let lower = sections_for(&slots, Interval { edge: Edge::Left, begin: 0, end: 500 }, 200);
        let upper = sections_for(&slots, Interval { edge: Edge::Left, begin: 500, end: 1000 }, 200);
        let ys = |v: &[Section]| -> Vec<i32> {
            v.iter().flat_map(|s| (s.begin_slot..=s.end_slot).map(|i| slots[i].y)).collect()
        };
        let (a, b) = (ys(&lower), ys(&upper));
        assert!(a.contains(&400) && !a.contains(&500));
        assert!(b.contains(&500));
        assert!(a.iter().all(|y| !b.contains(y)), "the two regions overlap");
    }

    #[test]
    fn a_region_no_slot_falls_into_produces_no_sections() {
        let slots = ring();
        assert!(sections_for(&slots, Interval { edge: Edge::Left, begin: 10, end: 20 }, 200).is_empty());
    }

    #[test]
    fn a_constrained_pin_lands_in_its_region_even_when_its_net_pulls_elsewhere() {
        // The whole point of a constraint: it beats the cost function.
        let slots = ring();
        let pins = vec![pin("stubborn", &[(900, 900)])];
        let c = vec![Constraint {
            interval: Interval { edge: Edge::Left, begin: 200, end: 600 },
            pins: vec![0],
        }];
        let (placed, unplaced) = place_with_constraints(&slots, &pins, &[], &c, &all(&pins), 200, DIE);
        assert!(unplaced.is_empty());
        assert_eq!(placed.len(), 1);
        let s = &slots[placed[0].slot];
        assert_eq!(s.edge, Edge::Left, "the constraint won");
        assert!((200..600).contains(&s.y));
    }

    #[test]
    fn constrained_pins_take_their_slots_before_free_pins_can() {
        // Order matters: a free pin that wants the same region must be pushed out of it, or the
        // constraint cannot be met at all.
        let slots = ring();
        // Four slots in the region (y = 200, 300, 400, 500); five pins all want that corner.
        let want = [(0, 350)];
        let mut pins: Vec<Pin> = (0..4).map(|i| pin(&format!("c{i}"), &want)).collect();
        pins.push(pin("free", &want));
        let c = vec![Constraint {
            interval: Interval { edge: Edge::Left, begin: 200, end: 600 },
            pins: vec![0, 1, 2, 3],
        }];
        let (placed, unplaced) = place_with_constraints(&slots, &pins, &[], &c, &all(&pins), 200, DIE);
        assert!(unplaced.is_empty(), "everything fits somewhere");

        let at = |p: usize| &slots[placed.iter().find(|x| x.pin == p).unwrap().slot];
        for p in 0..4 {
            assert_eq!(at(p).edge, Edge::Left);
            assert!((200..600).contains(&at(p).y), "constrained pin {p} left its region");
        }
        let free = at(4);
        assert!(
            free.edge != Edge::Left || !(200..600).contains(&free.y),
            "the free pin took a slot the constraint needed"
        );
    }

    #[test]
    fn no_two_pins_share_a_slot_across_the_two_phases() {
        // The consumed-slot bookkeeping is what prevents this, and it is invisible until it fails.
        let slots = ring();
        let pins: Vec<Pin> = (0..12).map(|i| pin(&format!("p{i}"), &[(i as i32 * 80, 500)])).collect();
        let c = vec![
            Constraint { interval: Interval { edge: Edge::Left, begin: 0, end: 400 }, pins: vec![0, 1] },
            Constraint { interval: Interval { edge: Edge::Bottom, begin: 0, end: 400 }, pins: vec![2, 3] },
        ];
        let (placed, unplaced) = place_with_constraints(&slots, &pins, &[], &c, &all(&pins), 200, DIE);
        assert!(unplaced.is_empty());
        assert_eq!(placed.len(), 12);
        let mut used: Vec<usize> = placed.iter().map(|p| p.slot).collect();
        used.sort_unstable();
        used.dedup();
        assert_eq!(used.len(), 12, "a slot was handed out twice");
    }

    #[test]
    fn a_pin_that_cannot_fit_its_region_is_reported_rather_than_placed_elsewhere() {
        // The design asked for a region. Somewhere else is not a smaller version of that answer.
        let slots = ring();
        let pins: Vec<Pin> = (0..5).map(|i| pin(&format!("p{i}"), &[])).collect();
        // Only two slots at y = 200, 300.
        let c = vec![Constraint {
            interval: Interval { edge: Edge::Left, begin: 200, end: 400 },
            pins: vec![0, 1, 2, 3, 4],
        }];
        let (placed, unplaced) = place_with_constraints(&slots, &pins, &[], &c, &all(&pins), 200, DIE);
        assert_eq!(placed.len(), 2, "only what the region holds");
        assert_eq!(unplaced.len(), 3, "the rest are reported, not relocated");
        for p in &placed {
            assert_eq!(slots[p.slot].edge, Edge::Left);
        }
    }

    #[test]
    fn a_pin_named_by_two_constraints_is_placed_once() {
        let slots = ring();
        let pins = vec![pin("a", &[])];
        let c = vec![
            Constraint { interval: Interval { edge: Edge::Top, begin: 0, end: 400 }, pins: vec![0] },
            Constraint { interval: Interval { edge: Edge::Left, begin: 0, end: 400 }, pins: vec![0] },
        ];
        let (placed, unplaced) = place_with_constraints(&slots, &pins, &[], &c, &all(&pins), 200, DIE);
        assert_eq!(placed.len(), 1, "placed once");
        assert!(unplaced.is_empty());
        assert_eq!(slots[placed[0].slot].edge, Edge::Top, "the first constraint took it");
    }

    #[test]
    fn overlap_is_about_sharing_a_stretch_of_the_same_edge() {
        let c = |edge, begin, end| Constraint { interval: Interval { edge, begin, end }, pins: vec![] };
        assert!(overlapping(&c(Edge::Bottom, 0, 500), &c(Edge::Bottom, 400, 900)));
        assert!(!overlapping(&c(Edge::Bottom, 0, 300), &c(Edge::Bottom, 400, 900)));
        assert!(!overlapping(&c(Edge::Bottom, 0, 500), &c(Edge::Left, 0, 500)), "different edges");
        assert!(overlapping(&c(Edge::Bottom, 0, 400), &c(Edge::Bottom, 400, 900)), "touching counts");
    }

    #[test]
    fn overlapping_constraints_are_reordered_by_pressure_roomiest_first() {
        // Who goes first decides who gets the shared slots, so the order is part of the answer.
        let slots = ring();
        let crowded = Constraint {
            interval: Interval { edge: Edge::Bottom, begin: 0, end: 500 },
            pins: vec![0, 1, 2, 3],
        };
        let roomy = Constraint {
            interval: Interval { edge: Edge::Bottom, begin: 300, end: 1000 },
            pins: vec![4],
        };
        let mut cs = vec![crowded.clone(), roomy.clone()];
        sort_constraints(&mut cs, &slots, 200);
        assert_eq!(cs[0], roomy, "the one with room to spare is served first");
        assert_eq!(cs[1], crowded);
    }

    #[test]
    fn constraints_that_do_not_overlap_keep_the_designs_own_order() {
        // A left-edge constraint has no bearing on a bottom-edge one; reordering them is noise.
        let slots = ring();
        let a = Constraint {
            interval: Interval { edge: Edge::Bottom, begin: 0, end: 400 },
            pins: vec![0, 1, 2, 3],
        };
        let b = Constraint { interval: Interval { edge: Edge::Left, begin: 0, end: 900 }, pins: vec![4] };
        let mut cs = vec![a.clone(), b.clone()];
        sort_constraints(&mut cs, &slots, 200);
        assert_eq!(cs, vec![a, b], "untouched despite very different pressure");
    }

    #[test]
    fn a_constraint_with_no_slots_sorts_last_rather_than_dividing_by_zero() {
        let slots = ring();
        let impossible = Constraint {
            interval: Interval { edge: Edge::Bottom, begin: 10, end: 20 },
            pins: vec![0],
        };
        let fine = Constraint {
            interval: Interval { edge: Edge::Bottom, begin: 0, end: 1000 },
            pins: vec![1],
        };
        let mut cs = vec![impossible.clone(), fine.clone()];
        sort_constraints(&mut cs, &slots, 200);
        assert_eq!(cs[0], fine);
    }

    #[test]
    fn with_no_constraints_it_is_the_plain_placement() {
        let slots = ring();
        let pins: Vec<Pin> = (0..6).map(|i| pin(&format!("p{i}"), &[(i as i32 * 150, 500)])).collect();
        let a = place_with_constraints(&slots, &pins, &[], &[], &all(&pins), 200, DIE);
        let b = crate::sections::place(&slots, &pins, &[], 200, DIE);
        assert_eq!(a, b, "no constraints must change nothing");
    }
}
