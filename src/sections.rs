// SPDX-License-Identifier: Apache-2.0
//! Sections — dividing the boundary so the assignment problem stays tractable.
//!
//! Matching every pin against every slot at once is the right answer and the wrong computation: a
//! die with 2,000 slots and 500 pins is a million-cell matrix, and the algorithm is cubic. So the
//! slot list is cut into fixed-size runs, each pin is sent to whichever run is cheapest **and has
//! room**, and the optimal matching is then solved inside each run independently.
//!
//! That is an approximation, and worth being clear about: the result is optimal *within* each
//! section and only greedy *between* them. It is upstream's decomposition, so reproducing it is
//! part of reproducing the placement.
//!
//! Nothing here touches a database.

use crate::groups::Group;
use crate::slots::{Edge, Slot};

/// How many slots one section spans. Upstream's default, and the only value the command exposes.
pub const SLOTS_PER_SECTION: usize = 200;

/// A run of consecutive slots, and the pins routed to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// The position of the run's middle slot — what a pin's cost to this section is measured
    /// against. A section is represented by one point, not by its extent.
    pub pos: (i32, i32),
    /// How many slots here are actually usable, blocked ones excluded.
    pub num_slots: usize,
    pub used_slots: usize,
    /// Index range into the slot list, **inclusive** at both ends.
    pub begin_slot: usize,
    pub end_slot: usize,
    pub edge: Edge,
    /// Indices into the pin list, in the order they were assigned here.
    pub pins: Vec<usize>,
}

/// One IO pin and everything its cost depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub name: String,
    /// Where the net's instance pins sit. An unconnected pin has none, and then its cost is the
    /// same everywhere — it goes wherever there is room.
    pub sinks: Vec<(i32, i32)>,
    /// The partner this pin is mirrored with, if any.
    ///
    /// Set on **one** of the pair only — the one that competes for a slot. Its partner does not
    /// compete at all: it is placed at the mirrored position of whatever this pin gets, so giving
    /// both a say would let them disagree.
    pub mirror: Option<usize>,
}

/// **M1** — reflect a position to the opposite edge.
///
/// A pin on the left edge mirrors to the right at the same height; one on the bottom mirrors to
/// the top at the same offset. Which axis flips is decided by **which boundary the position is
/// already on**, so this only makes sense for a position that is actually on the die edge.
pub fn mirrored_position(at: (i32, i32), die: (i32, i32, i32, i32)) -> (i32, i32) {
    let (x0, y0, x1, y1) = die;
    let (x, y) = at;
    if x == x0 {
        (x1, y)
    } else if x == x1 {
        (x0, y)
    } else if y == y0 {
        (x, y1)
    } else {
        (x, y0)
    }
}

/// **M1** — the edge a mirrored pin ends up on.
pub fn mirrored_edge(edge: Edge) -> Edge {
    match edge {
        Edge::Left => Edge::Right,
        Edge::Right => Edge::Left,
        Edge::Bottom => Edge::Top,
        Edge::Top => Edge::Bottom,
        // A lattice position has no opposite side to reflect to.
        Edge::Invalid => Edge::Invalid,
    }
}

/// **M2** — what it costs to put a pin somewhere, counting its mirror.
///
/// A mirrored pair is placed as one decision, so it is costed as one: the pin's own wirelength
/// plus its partner's at the reflected position. Costing only the competing pin would optimise
/// half a pair and let the other half land wherever the reflection happened to fall.
pub fn pin_cost(pins: &[Pin], idx: usize, at: (i32, i32), die: (i32, i32, i32, i32)) -> i64 {
    let own = net_hpwl(&pins[idx].sinks, at);
    match pins[idx].mirror {
        Some(m) => own + net_hpwl(&pins[m].sinks, mirrored_position(at, die)),
        None => own,
    }
}

/// **M3** — how *unbalanced* a mirrored pair would be at this position: the larger of the two
/// halves.
///
/// Used only to break ties. Two positions with the same total can split it evenly or lopsidedly,
/// and the even split is the better placement.
pub fn pin_balance(pins: &[Pin], idx: usize, at: (i32, i32), die: (i32, i32, i32, i32)) -> i64 {
    let own = net_hpwl(&pins[idx].sinks, at);
    match pins[idx].mirror {
        Some(m) => own.max(net_hpwl(&pins[m].sinks, mirrored_position(at, die))),
        None => own,
    }
}

/// **H1** — the cost of putting a pin at a position: the half-perimeter of the box enclosing the
/// position and every instance pin on its net.
///
/// Half-perimeter is the standard estimate of the wire needed to connect a net, and it is what
/// makes the whole placement a wirelength minimisation rather than a packing exercise.
pub fn net_hpwl(sinks: &[(i32, i32)], at: (i32, i32)) -> i64 {
    let (mut min_x, mut max_x) = (at.0, at.0);
    let (mut min_y, mut max_y) = (at.1, at.1);
    for &(x, y) in sinks {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (max_x - min_x) as i64 + (max_y - min_y) as i64
}

/// **H2** — cut one contiguous run of slots into sections.
///
/// `end` is inclusive. A section's `num_slots` counts only the usable slots, but its index range
/// still spans the blocked ones — the run is cut by position, and blocked slots occupy position.
///
/// ⚠️ Faithful to an upstream edge case: the loop tests the *end* index against `end` starting
/// from zero, so a single-slot run at index 0 produces no section at all. Reproduced rather than
/// tidied, because a section list that differs shifts every later assignment.
pub fn find_sections(
    slots: &[Slot],
    begin: usize,
    end: usize,
    edge: Edge,
    slots_per_section: usize,
) -> Vec<Section> {
    let mut out = Vec::new();
    if begin > end || end >= slots.len() || slots_per_section == 0 {
        return out;
    }
    let mut begin = begin;
    let mut end_slot = 0usize;
    while end_slot < end {
        end_slot = (begin + slots_per_section - 1).min(end);
        let blocked = slots[begin..=end_slot].iter().filter(|s| s.blocked).count();
        let middle = begin + (end_slot - begin) / 2;
        out.push(Section {
            pos: (slots[middle].x, slots[middle].y),
            num_slots: end_slot - begin + 1 - blocked,
            used_slots: 0,
            begin_slot: begin,
            end_slot,
            edge,
            pins: Vec::new(),
        });
        end_slot += 1;
        begin = end_slot;
    }
    out
}

/// **H3** — every section on the boundary, following the slot list's own order.
///
/// One run per (edge, layer) pair, taken as the *contiguous* block that
/// [`crate::slots::define_slots`] emits. The section list therefore inherits the slot list's
/// counter-clockwise order, which is what lets a pin's choice of section be a choice of
/// neighbourhood.
pub fn create_sections(slots: &[Slot], slots_per_section: usize) -> Vec<Section> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < slots.len() {
        let (edge, layer) = (slots[i].edge, slots[i].layer.clone());
        let mut j = i;
        while j < slots.len() && slots[j].edge == edge && slots[j].layer == layer {
            j += 1;
        }
        out.extend(find_sections(slots, i, j - 1, edge, slots_per_section));
        i = j;
    }
    out
}

/// **H4** — route each pin to the cheapest section that still has room.
///
/// Pins are considered in the order given, and each takes the best section still open to it. This
/// is the greedy half of the decomposition: it decides *neighbourhood*, and the matching inside a
/// section then decides *position* optimally.
///
/// Returns the indices of any pins that found no room at all — the caller must not treat those as
/// placed.
pub fn assign_pins_to_sections(
    pins: &[Pin],
    which: &[usize],
    sections: &mut [Section],
    die: (i32, i32, i32, i32),
) -> Vec<usize> {
    let mut unplaced = Vec::new();
    // ⚠️ **Mirrored pins first.** A mirrored pin constrains two positions at once, so it has far
    // less freedom than a free pin; letting free pins fill up the sections first would leave a
    // pair with no position whose reflection is also open.
    let mirrored_first = {
        let mut v: Vec<usize> = which.to_vec();
        v.sort_by_key(|&i| pins[i].mirror.is_none());
        v
    };
    for idx in mirrored_first {
        let mut order: Vec<usize> = (0..sections.len()).collect();
        let costs: Vec<i64> =
            sections.iter().map(|s| pin_cost(pins, idx, s.pos, die)).collect();
        // Stable, so equal-cost sections stay in boundary order rather than in an arbitrary one.
        order.sort_by_key(|&i| costs[i]);
        match order.into_iter().find(|&i| sections[i].used_slots < sections[i].num_slots) {
            Some(i) => {
                sections[i].pins.push(idx);
                sections[i].used_slots += 1;
            }
            None => unplaced.push(idx),
        }
    }
    unplaced
}

/// **M3** — rank positions by a secondary measure, 1-based, in the low byte of the cost.
///
/// ⚠️ The rank is a **byte**, so a section with more than 255 positions wraps around and the
/// ranking restarts. Upstream's type, kept: at 200 slots per section it is unreachable by default,
/// and changing it would change assignments.
pub fn tie_break_rank(measure: &[i64]) -> Vec<u8> {
    let mut order: Vec<usize> = (0..measure.len()).collect();
    order.sort_by_key(|&i| measure[i]);
    let mut rank = vec![0u8; measure.len()];
    for (n, &i) in order.iter().enumerate() {
        rank[i] = (n as u8).wrapping_add(1);
    }
    rank
}

/// The slot at a given position on a given layer, if there is one.
pub fn slot_at(slots: &[Slot], at: (i32, i32), layer: &str) -> Option<usize> {
    slots.iter().position(|s| s.x == at.0 && s.y == at.1 && s.layer == layer)
}

/// **M4** — place every mirrored partner opposite the pin that was placed for it.
///
/// The partner never competed for a position, so this is where it gets one: the reflection of its
/// pair's slot, on the same layer. If that slot does not exist or is already taken, the pair
/// cannot be honoured and **both** are reported unplaced — half a mirrored pair is not a partial
/// success, it is a broken symmetry.
pub fn expand_mirrors(
    slots: &mut [Slot],
    pins: &[Pin],
    placed: &[Placement],
    die: (i32, i32, i32, i32),
) -> (Vec<Placement>, Vec<usize>) {
    let mut added = Vec::new();
    let mut failed = Vec::new();
    for p in placed {
        let Some(m) = pins[p.pin].mirror else { continue };
        let at = mirrored_position((slots[p.slot].x, slots[p.slot].y), die);
        match slot_at(slots, at, &slots[p.slot].layer) {
            Some(i) if !slots[i].blocked || i == p.slot => {
                slots[i].blocked = true;
                added.push(Placement { pin: m, slot: i });
            }
            _ => {
                failed.push(p.pin);
                failed.push(m);
            }
        }
    }
    (added, failed)
}

/// Where one pin ended up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub pin: usize,
    pub slot: usize,
}

/// **H5** — solve one section: the optimal pairing of its pins to its usable slots.
///
/// Blocked slots are skipped when the matrix is built, so a row index is a position among the
/// *usable* slots and has to be mapped back to a real slot index afterwards. Getting that mapping
/// wrong places pins on slots that are already occupied, silently.
///
/// Returns `None` if the section holds more pins than usable slots, which the caller's capacity
/// accounting should already have prevented.
pub fn match_section(
    slots: &[Slot],
    section: &Section,
    pins: &[Pin],
    die: (i32, i32, i32, i32),
) -> Option<Vec<Placement>> {
    if section.pins.is_empty() {
        return Some(Vec::new());
    }
    let usable: Vec<usize> = (section.begin_slot..=section.end_slot)
        .filter(|&i| !slots[i].blocked)
        .collect();
    if usable.len() < section.pins.len() {
        return None;
    }

    // Pins are rows and slots are columns: there are never fewer slots than pins in a section, and
    // the solver needs the short side first.
    //
    // ⚠️ **A mirrored pin's row is scaled by 256 with a tie-break rank in the low byte.** The rank
    // orders that pin's positions by how evenly the pair's cost splits, so among equal totals the
    // balanced position wins. The scaling is upstream's and it is not neutral: a mirrored row's
    // costs dominate an unmirrored row's in the same matrix. Reproduced rather than normalised,
    // because the resulting assignment is what the reference produces.
    let cost: Vec<Vec<i64>> = section
        .pins
        .iter()
        .map(|&p| {
            let row: Vec<i64> = usable
                .iter()
                .map(|&s| pin_cost(pins, p, (slots[s].x, slots[s].y), die))
                .collect();
            if pins[p].mirror.is_none() {
                return row;
            }
            let balance: Vec<i64> = usable
                .iter()
                .map(|&s| pin_balance(pins, p, (slots[s].x, slots[s].y), die))
                .collect();
            let rank = tie_break_rank(&balance);
            row.iter().zip(&rank).map(|(c, r)| (c << 8) | *r as i64).collect()
        })
        .collect();

    let assignment = crate::hungarian::solve(&cost)?;
    Some(
        assignment
            .into_iter()
            .enumerate()
            .map(|(row, col)| Placement { pin: section.pins[row], slot: usable[col] })
            .collect(),
    )
}

/// **H6** — solve a set of sections that already have their pins and groups assigned.
///
/// The order inside a section is the load-bearing part: **groups are matched first**, their slots
/// are consumed, and only then are the single pins matched over what remains. A single pin dropped
/// into the middle of the only long enough run destroys it, and nothing later can undo that.
///
/// `slots` is taken by value and returned consumed, so a caller running several rounds — one per
/// constraint, then the free pins — sees each round's slots already withdrawn.
pub fn solve_sections(
    slots: &mut [Slot],
    sections: &[Section],
    pins: &[Pin],
    groups: &[Group],
    members: &[Vec<usize>],
    die: (i32, i32, i32, i32),
) -> (Vec<Placement>, Vec<usize>) {
    let mut placed = Vec::new();
    let mut unplaced = Vec::new();

    let trace = std::env::var_os("VYGES_PPL_SECTION_TRACE").is_some();
    for (i, section) in sections.iter().enumerate() {
        if trace {
            eprintln!(
                "section {i}: slots {}..{} num={} pins={:?}",
                section.begin_slot, section.end_slot, section.num_slots, section.pins
            );
        }
        let mine: &[usize] = members.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
        match crate::groups::match_in_section(slots, section, pins, groups, mine, die) {
            Some(ps) => {
                for p in &ps {
                    slots[p.slot].blocked = true;
                }
                // A grouped pin can be mirrored too, and its partner needs the reflection just as
                // much. Missing this dropped those partners silently — placed nowhere, reported
                // nowhere.
                let (mirrors, failed) = expand_mirrors(slots, pins, &ps, die);
                placed.extend(ps);
                placed.extend(mirrors);
                unplaced.extend(failed);
            }
            None => unplaced.extend(mine.iter().flat_map(|&g| groups[g].pins.iter().copied())),
        }

        // Recount: the groups just placed took slots this section thought it had.
        let available =
            (section.begin_slot..=section.end_slot).filter(|&k| !slots[k].blocked).count();
        let refreshed = Section { num_slots: available, used_slots: 0, ..section.clone() };
        match match_section(slots, &refreshed, pins, die) {
            Some(ps) => {
                for p in &ps {
                    slots[p.slot].blocked = true;
                }
                // ⚠️ **Claim the reflections now, not after every section.** A mirrored partner's
                // slot is not reserved by any section's accounting, so leaving the claim until the
                // end lets a later section hand that slot to somebody else and the pair fails for
                // a reason that has nothing to do with it.
                let (mirrors, failed) = expand_mirrors(slots, pins, &ps, die);
                placed.extend(ps);
                placed.extend(mirrors);
                unplaced.extend(failed);
            }
            None => unplaced.extend(refreshed.pins.iter().copied()),
        }
    }
    (placed, unplaced)
}

/// Route groups to sections, reporting the ones that fit nowhere.
///
/// Returns, per section, the groups it took.
pub fn assign_groups_to_sections(
    groups: &[Group],
    which: &[usize],
    sections: &mut [Section],
    pins: &[Pin],
    slots: &[Slot],
    slots_per_section: usize,
    die: (i32, i32, i32, i32),
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); sections.len()];
    let mut unplaced = Vec::new();
    for &g in which {
        let sizes: Vec<Vec<usize>> = members
            .iter()
            .map(|m| m.iter().map(|&x| groups[x].pins.len()).collect())
            .collect();
        match crate::groups::assign_to_section(
            &groups[g],
            sections,
            &sizes,
            pins,
            slots,
            slots_per_section,
            die,
        ) {
            Some(i) => members[i].push(g),
            None => unplaced.extend(groups[g].pins.iter().copied()),
        }
    }
    (members, unplaced)
}

/// The whole deterministic placement for pins under no constraint: sections, groups, then singles.
///
/// Returns the placements and the pins that could not be placed.
pub fn place(
    slots: &[Slot],
    pins: &[Pin],
    groups: &[Group],
    slots_per_section: usize,
    die: (i32, i32, i32, i32),
) -> (Vec<Placement>, Vec<usize>) {
    let mut slots = slots.to_vec();
    let mut sections = create_sections(&slots, slots_per_section);
    let which: Vec<usize> = (0..groups.len()).collect();
    let (members, mut unplaced) = assign_groups_to_sections(
        groups,
        &which,
        &mut sections,
        pins,
        &slots,
        slots_per_section,
        die,
    );

    // Only pins that are nobody's group member compete individually — and neither does a mirrored
    // partner, which takes whatever position its pair's reflection lands on.
    let grouped: std::collections::BTreeSet<usize> =
        groups.iter().flat_map(|g| g.pins.iter().copied()).collect();
    let partners: std::collections::BTreeSet<usize> = pins.iter().filter_map(|p| p.mirror).collect();
    let singles: Vec<usize> =
        (0..pins.len()).filter(|p| !grouped.contains(p) && !partners.contains(p)).collect();
    unplaced.extend(assign_pins_to_sections(pins, &singles, &mut sections, die));

    let (placed, missed) = solve_sections(&mut slots, &sections, pins, groups, &members, die);
    unplaced.extend(missed);

    let mut placed = placed;
    placed.sort_by_key(|p| p.pin);
    unplaced.sort_unstable();
    unplaced.dedup();
    (placed, unplaced)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(x: i32, y: i32, edge: Edge, layer: &str) -> Slot {
        Slot { x, y, layer: layer.into(), edge, blocked: false }
    }

    /// A row of `n` slots along the bottom edge, one every 100 units.
    fn bottom_row(n: usize) -> Vec<Slot> {
        (0..n).map(|i| slot(i as i32 * 100, 0, Edge::Bottom, "met1")).collect()
    }

    /// Every pin index — the common case for a test that places the whole list.
    fn all(pins: &[Pin]) -> Vec<usize> {
        (0..pins.len()).collect()
    }

    fn pin(name: &str, sinks: &[(i32, i32)]) -> Pin {
        Pin { name: name.into(), sinks: sinks.to_vec(), mirror: None }
    }

    #[test]
    fn hpwl_is_the_half_perimeter_of_the_box_around_the_net() {
        assert_eq!(net_hpwl(&[(100, 0)], (0, 0)), 100, "a two-point net is its own span");
        assert_eq!(net_hpwl(&[(100, 50)], (0, 0)), 150, "x span plus y span");
        assert_eq!(net_hpwl(&[(10, 10), (90, 30)], (50, 20)), 100, "the box covers everything");
        assert_eq!(net_hpwl(&[], (7, 9)), 0, "a pin with no sinks costs the same everywhere");
        // A sink on the far side of the position still counts: the box is not one-sided.
        assert_eq!(net_hpwl(&[(-100, 0)], (100, 0)), 200);
    }

    #[test]
    fn a_run_is_cut_into_sections_of_the_given_size() {
        let slots = bottom_row(450);
        let s = find_sections(&slots, 0, 449, Edge::Bottom, 200);
        assert_eq!(s.len(), 3, "200 + 200 + 50");
        assert_eq!((s[0].begin_slot, s[0].end_slot), (0, 199));
        assert_eq!((s[1].begin_slot, s[1].end_slot), (200, 399));
        assert_eq!((s[2].begin_slot, s[2].end_slot), (400, 449));
        assert_eq!(s[2].num_slots, 50, "the last section is short, not padded");
        assert!(s.iter().all(|x| x.used_slots == 0 && x.pins.is_empty()));
    }

    #[test]
    fn a_section_is_represented_by_its_middle_slot() {
        // The whole greedy stage compares pins against this one point, so which point it is
        // decides which section a pin picks.
        let slots = bottom_row(11);
        let s = find_sections(&slots, 0, 10, Edge::Bottom, 200);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].pos, (500, 0), "slot 5 of 0..10");
    }

    #[test]
    fn blocked_slots_lower_the_capacity_but_still_occupy_their_place() {
        // Capacity and extent are different things: a blocked slot cannot hold a pin, but the run
        // is still cut by position, so removing it from the range would shift every later section.
        let mut slots = bottom_row(10);
        slots[3].blocked = true;
        slots[7].blocked = true;
        let s = find_sections(&slots, 0, 9, Edge::Bottom, 200);
        assert_eq!(s[0].num_slots, 8, "two of ten are unusable");
        assert_eq!((s[0].begin_slot, s[0].end_slot), (0, 9), "the range is unchanged");
    }

    #[test]
    fn sections_follow_the_slot_lists_own_edge_and_layer_runs() {
        let mut slots = bottom_row(3);
        slots.extend((0..3).map(|i| slot(1000, i * 100, Edge::Right, "met1")));
        slots.extend((0..3).map(|i| slot(i * 100, 0, Edge::Bottom, "met2")));
        let s = create_sections(&slots, 200);
        assert_eq!(s.len(), 3, "one section per (edge, layer) run");
        assert_eq!(s[0].edge, Edge::Bottom);
        assert_eq!(s[1].edge, Edge::Right);
        assert_eq!(s[2].edge, Edge::Bottom, "a second bottom run, on the other layer");
        assert_eq!((s[2].begin_slot, s[2].end_slot), (6, 8));
    }

    #[test]
    fn a_pin_goes_to_the_section_nearest_its_net() {
        // Two sections far apart; each pin's net sits under one of them.
        let mut slots = bottom_row(400);
        for s in slots.iter_mut() {
            s.y = 0;
        }
        let mut sections = create_sections(&slots, 200);
        assert_eq!(sections.len(), 2);
        let (left_mid, right_mid) = (sections[0].pos.0, sections[1].pos.0);

        let pins = vec![pin("a", &[(right_mid, 500)]), pin("b", &[(left_mid, 500)])];
        assert!(assign_pins_to_sections(&pins, &all(&pins), &mut sections, (0, 0, 10_000, 10_000)).is_empty());
        assert_eq!(sections[1].pins, vec![0], "pin a went right, where its net is");
        assert_eq!(sections[0].pins, vec![1], "pin b went left");
    }

    #[test]
    fn a_full_section_pushes_the_next_pin_to_its_second_choice() {
        // Capacity is what makes this greedy rather than a plain nearest-section map.
        let slots = bottom_row(4);
        let mut sections = find_sections(&slots, 0, 3, Edge::Bottom, 2);
        assert_eq!(sections.len(), 2);
        // Every pin wants section 0, which holds two.
        let want = sections[0].pos;
        let pins: Vec<Pin> = (0..3).map(|i| pin(&format!("p{i}"), &[want])).collect();
        let unplaced = assign_pins_to_sections(&pins, &all(&pins), &mut sections, (0, 0, 10_000, 10_000));
        assert!(unplaced.is_empty());
        assert_eq!(sections[0].pins, vec![0, 1], "first two take it");
        assert_eq!(sections[1].pins, vec![2], "the third is pushed on");
    }

    #[test]
    fn pins_beyond_the_total_capacity_are_reported_not_dropped() {
        let slots = bottom_row(2);
        let mut sections = find_sections(&slots, 0, 1, Edge::Bottom, 200);
        let pins: Vec<Pin> = (0..5).map(|i| pin(&format!("p{i}"), &[])).collect();
        let unplaced = assign_pins_to_sections(&pins, &all(&pins), &mut sections, (0, 0, 10_000, 10_000));
        assert_eq!(unplaced, vec![2, 3, 4], "three pins had nowhere to go");
    }

    #[test]
    fn matching_within_a_section_beats_taking_each_pins_own_favourite() {
        // The reason a matching is solved at all. Both pins prefer slot 0; the optimum gives it to
        // the one that loses more by moving.
        let slots = bottom_row(2); // x = 0 and 100
        let mut sections = find_sections(&slots, 0, 1, Edge::Bottom, 200);
        sections[0].pins = vec![0, 1];
        sections[0].used_slots = 2;
        let pins = vec![pin("far", &[(0, 1000)]), pin("near", &[(40, 1000)])];

        let placed = match_section(&slots, &sections[0], &pins, (0, 0, 10_000, 10_000)).unwrap();
        let at = |p: usize| placed.iter().find(|x| x.pin == p).unwrap().slot;
        assert_ne!(at(0), at(1), "two pins never share a slot");
        // Total cost is what the algorithm optimises, so that is what the test asserts.
        let cost: i64 = placed
            .iter()
            .map(|p| net_hpwl(&pins[p.pin].sinks, (slots[p.slot].x, slots[p.slot].y)))
            .sum();
        let swapped: i64 = net_hpwl(&pins[0].sinks, (slots[at(1)].x, slots[at(1)].y))
            + net_hpwl(&pins[1].sinks, (slots[at(0)].x, slots[at(0)].y));
        assert!(cost <= swapped, "the chosen pairing is no worse than the other one");
    }

    #[test]
    fn a_blocked_slot_is_never_assigned_to() {
        // The row-to-slot mapping is the trap: rows index USABLE slots, not all slots, so an
        // off-by-one here puts a pin on top of something.
        let mut slots = bottom_row(6);
        slots[0].blocked = true;
        slots[1].blocked = true;
        slots[2].blocked = true;
        let mut sections = find_sections(&slots, 0, 5, Edge::Bottom, 200);
        sections[0].pins = vec![0, 1, 2];
        let placed = match_section(&slots, &sections[0], &vec![pin("a", &[]); 3], (0, 0, 10_000, 10_000)).unwrap();
        assert_eq!(placed.len(), 3);
        assert!(
            placed.iter().all(|p| !slots[p.slot].blocked),
            "a pin landed on a blocked slot: {placed:?}"
        );
        let mut used: Vec<usize> = placed.iter().map(|p| p.slot).collect();
        used.sort_unstable();
        assert_eq!(used, vec![3, 4, 5]);
    }

    #[test]
    fn a_section_with_more_pins_than_usable_slots_refuses_rather_than_overfills() {
        let mut slots = bottom_row(3);
        slots[0].blocked = true;
        let mut sections = find_sections(&slots, 0, 2, Edge::Bottom, 200);
        sections[0].pins = vec![0, 1, 2];
        assert!(match_section(&slots, &sections[0], &vec![pin("a", &[]); 3], (0, 0, 10_000, 10_000)).is_none());
    }

    #[test]
    fn place_puts_every_pin_somewhere_distinct() {
        let slots = bottom_row(50);
        let pins: Vec<Pin> =
            (0..20).map(|i| pin(&format!("p{i}"), &[(i as i32 * 200, 700)])).collect();
        let (placed, unplaced) = place(&slots, &pins, &[], 8, (0, 0, 10_000, 10_000));

        assert!(unplaced.is_empty(), "room for all of them");
        assert_eq!(placed.len(), 20);
        let mut used: Vec<usize> = placed.iter().map(|p| p.slot).collect();
        used.sort_unstable();
        used.dedup();
        assert_eq!(used.len(), 20, "no slot used twice");
        assert!(placed.windows(2).all(|w| w[0].pin < w[1].pin), "reported in pin order");

        // Pins whose nets are spread left-to-right should come out ordered left-to-right: the
        // decomposition is only useful if a pin lands near its net.
        let xs: Vec<i32> = placed.iter().map(|p| slots[p.slot].x).collect();
        assert!(xs.windows(2).all(|w| w[0] < w[1]), "placement follows the nets: {xs:?}");
    }

    #[test]
    fn a_position_reflects_to_the_opposite_edge_at_the_same_offset() {
        let die = (0, 0, 1000, 1000);
        assert_eq!(mirrored_position((0, 300), die), (1000, 300), "left to right");
        assert_eq!(mirrored_position((1000, 300), die), (0, 300), "right to left");
        assert_eq!(mirrored_position((400, 0), die), (400, 1000), "bottom to top");
        assert_eq!(mirrored_position((400, 1000), die), (400, 0), "top to bottom");
        assert_eq!(mirrored_edge(Edge::Left), Edge::Right);
        assert_eq!(mirrored_edge(Edge::Bottom), Edge::Top);
    }

    #[test]
    fn a_mirrored_pin_is_costed_as_a_pair() {
        // Costing only the competing half would optimise one pin and let the other land wherever
        // the reflection happened to fall.
        let die = (0, 0, 1000, 1000);
        let pins = vec![
            Pin { name: "a".into(), sinks: vec![(0, 500)], mirror: Some(1) },
            Pin { name: "b".into(), sinks: vec![(1000, 500)], mirror: None },
        ];
        // Put `a` on the left: it is on top of its own net, and `b` lands on the right, on top of
        // its net too. Both halves are cheap.
        let together = pin_cost(&pins, 0, (0, 500), die);
        // Put `a` on the right instead: both halves are now far from their nets.
        let apart = pin_cost(&pins, 1, (1000, 500), die) + net_hpwl(&pins[0].sinks, (1000, 500));
        assert_eq!(together, 0, "the pair is perfectly placed");
        assert!(apart > together);
        // The unmirrored pin costs only itself.
        assert_eq!(pin_cost(&pins, 1, (0, 500), die), net_hpwl(&pins[1].sinks, (0, 500)));
    }

    #[test]
    fn balance_is_the_larger_half_not_the_total() {
        // Two positions can share a total and split it very differently; the even split is better.
        let die = (0, 0, 1000, 1000);
        let pins = vec![
            Pin { name: "a".into(), sinks: vec![(0, 0)], mirror: Some(1) },
            Pin { name: "b".into(), sinks: vec![(1000, 0)], mirror: None },
        ];
        let at = (0, 500);
        assert_eq!(
            pin_balance(&pins, 0, at, die),
            net_hpwl(&pins[0].sinks, at).max(net_hpwl(&pins[1].sinks, mirrored_position(at, die)))
        );
    }

    #[test]
    fn the_tie_break_rank_is_one_based_and_follows_the_measure() {
        assert_eq!(tie_break_rank(&[30, 10, 20]), vec![3, 1, 2]);
        assert_eq!(tie_break_rank(&[]), Vec::<u8>::new());
        // ⚠️ A byte: past 255 entries the ranking wraps rather than saturating.
        let big: Vec<i64> = (0..258).map(|i| i as i64).collect();
        let r = tie_break_rank(&big);
        assert_eq!(r[0], 1);
        assert_eq!(r[255], 0, "wrapped");
        assert_eq!(r[256], 1);
    }

    #[test]
    fn a_mirrored_partner_lands_opposite_its_pair() {
        let die = (0, 0, 1000, 1000);
        let mut slots = vec![
            Slot { x: 0, y: 300, layer: "m1".into(), edge: Edge::Left, blocked: false },
            Slot { x: 1000, y: 300, layer: "m1".into(), edge: Edge::Right, blocked: false },
        ];
        let pins = vec![
            Pin { name: "a".into(), sinks: vec![], mirror: Some(1) },
            Pin { name: "b".into(), sinks: vec![], mirror: None },
        ];
        let (added, failed) =
            expand_mirrors(&mut slots, &pins, &[Placement { pin: 0, slot: 0 }], die);
        assert!(failed.is_empty());
        assert_eq!(added, vec![Placement { pin: 1, slot: 1 }]);
        assert!(slots[1].blocked, "the reflected slot is consumed");
    }

    #[test]
    fn a_pair_whose_reflection_is_unavailable_fails_as_a_pair() {
        // Half a mirrored pair is not a partial success.
        let die = (0, 0, 1000, 1000);
        let mut slots =
            vec![Slot { x: 0, y: 300, layer: "m1".into(), edge: Edge::Left, blocked: false }];
        let pins = vec![
            Pin { name: "a".into(), sinks: vec![], mirror: Some(1) },
            Pin { name: "b".into(), sinks: vec![], mirror: None },
        ];
        let (added, failed) =
            expand_mirrors(&mut slots, &pins, &[Placement { pin: 0, slot: 0 }], die);
        assert!(added.is_empty());
        assert_eq!(failed, vec![0, 1], "both halves are reported, not just the orphan");
    }

    #[test]
    fn mirrored_pins_are_offered_sections_before_free_pins() {
        // A mirrored pin constrains two positions at once, so it has the least freedom and must
        // choose first. One section, one slot: the mirrored pin should take it.
        let die = (0, 0, 10_000, 10_000);
        let slots = bottom_row(2);
        let mut sections = find_sections(&slots, 0, 1, Edge::Bottom, 200);
        sections[0].num_slots = 1;
        let pins = vec![
            Pin { name: "free".into(), sinks: vec![], mirror: None },
            Pin { name: "pair".into(), sinks: vec![], mirror: Some(2) },
            Pin { name: "partner".into(), sinks: vec![], mirror: None },
        ];
        let unplaced = assign_pins_to_sections(&pins, &all(&pins), &mut sections, die);
        assert_eq!(sections[0].pins, vec![1], "the mirrored pin chose first");
        assert!(unplaced.contains(&0));
    }

    #[test]
    fn nothing_to_place_is_not_an_error() {
        let slots = bottom_row(10);
        assert_eq!(place(&slots, &[], &[], 200, (0, 0, 10_000, 10_000)), (Vec::new(), Vec::new()));
        assert!(create_sections(&[], 200).is_empty());
        assert!(find_sections(&[], 0, 0, Edge::Bottom, 200).is_empty());
    }
}
