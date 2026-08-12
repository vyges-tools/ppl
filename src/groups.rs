// SPDX-License-Identifier: Apache-2.0
//! Pin groups — ports that must end up side by side.
//!
//! A bus is not 32 independent pins. Scattering `data[0..31]` around the boundary is legal and
//! useless: the pins have to arrive together to be routed together. `set_io_pin_constraint -group`
//! says so, and `-order` additionally fixes the sequence.
//!
//! That makes a group a **placement primitive rather than a preference**, and it changes the
//! problem in two ways:
//!
//! - A group needs a **contiguous run** of free slots, so a section with plenty of room spread
//!   across gaps may still have nowhere to put it.
//! - Groups are placed **before** individual pins, because a single pin dropped in the middle of
//!   the only long enough run destroys it, and there is no way to recover from that afterwards.
//!
//! Nothing here touches a database.

use crate::sections::{pin_cost, Pin, Placement, Section};
use crate::slots::{Edge, Slot};

/// Ports that must land on adjacent slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    /// Pin indices, in the order the design declared them.
    pub pins: Vec<usize>,
    /// Whether that declared order must be preserved along the edge.
    pub ordered: bool,
}

/// **G1** — the longest run of free slots inside a section.
///
/// The number that decides whether a group fits. It is *not* the count of free slots: ten free
/// slots in two runs of five will not hold a group of six.
pub fn max_contiguous(slots: &[Slot], begin: usize, end: usize) -> usize {
    let mut best = 0;
    let mut run = 0;
    for s in &slots[begin..=end.min(slots.len().saturating_sub(1))] {
        run = if s.blocked { 0 } else { run + 1 };
        best = best.max(run);
    }
    best
}

/// **G2** — the slot positions a group could start at, and the size each was found for.
///
/// Candidates are found per distinct group size, **largest first**, and after each accepted
/// position the scan jumps forward by that size — so the candidate set is sparse rather than every
/// possible offset. That keeps the matching small, and it is upstream's choice, not an optimisation
/// introduced here.
///
/// ⚠️ **The scan position carries across sizes.** Where several sizes are searched, each one
/// resumes from where the previous left off rather than restarting, so a smaller group is offered
/// only positions in the tail. Reproduced deliberately: it is observable whenever a section holds
/// groups of different sizes, and "fixing" it would be a difference from the reference rather than
/// an improvement.
pub fn starting_slots(
    slots: &[Slot],
    begin: usize,
    end: usize,
    sizes_desc: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let (mut starts, mut caps) = (Vec::new(), Vec::new());
    if end >= slots.len() || begin > end {
        return (starts, caps);
    }
    let mut resume = begin;
    for &size in sizes_desc {
        if size == 0 || size > end - begin + 1 {
            continue;
        }
        let last = end + 1 - size;
        let mut i = resume;
        while i <= last {
            match (0..size).find(|&k| slots[i + k].blocked) {
                Some(k) => {
                    // Step forward until the slot that blocked us is free, rather than one at a
                    // time: everything in between fails for the same reason.
                    loop {
                        i += 1;
                        if i > last || !slots[i + k].blocked {
                            break;
                        }
                    }
                }
                None => {
                    starts.push(i);
                    caps.push(size);
                    i += size;
                    resume = i;
                }
            }
        }
    }
    (starts, caps)
}

/// **G3** — send a group to the cheapest section that can actually hold it.
///
/// Cost is the sum of its pins' wirelength to the section, so a group goes where its *nets* are,
/// not where the first free run happens to be. Sections are then tried in that order and the first
/// that passes every test takes it.
///
/// Two tests beyond "does it fit", both upstream's, and both about keeping the later matching
/// solvable: a **large** group will not share a section with any other group, and a section that
/// already holds a large group takes no more. "Large" is more than half a section.
///
/// Returns the chosen section, or `None` — a group that fits nowhere is the caller's problem to
/// report, not something to quietly split.
pub fn assign_to_section(
    group: &Group,
    sections: &mut [Section],
    existing: &[Vec<usize>],
    pins: &[Pin],
    slots: &[Slot],
    slots_per_section: usize,
    die: (i32, i32, i32, i32),
) -> Option<usize> {
    let size = group.pins.len();
    let cost: Vec<i64> = sections
        .iter()
        .map(|s| group.pins.iter().map(|&p| pin_cost(pins, p, s.pos, die)).sum())
        .collect();

    let mut order: Vec<usize> = (0..sections.len()).collect();
    // Ties break toward the emptier section, which spreads groups instead of stacking them.
    order.sort_by_key(|&i| (cost[i], sections[i].used_slots));

    for i in order {
        let here = &existing[i];
        let biggest_here = here.iter().copied().max().unwrap_or(0);
        if (size > slots_per_section / 2 && !here.is_empty())
            || biggest_here > slots_per_section / 2
        {
            continue;
        }
        let room = sections[i].num_slots - sections[i].used_slots;
        if size <= max_contiguous(slots, sections[i].begin_slot, sections[i].end_slot)
            && size <= room
        {
            sections[i].used_slots += size;
            return Some(i);
        }
    }
    None
}

/// **G4** — place every group in one section, optimally, on contiguous runs.
///
/// Rows are groups and columns are candidate starting positions, so the matching decides *where
/// each group starts* rather than where each pin goes. A group's cost at a candidate is the sum of
/// its pins' wirelength **measured at the starting slot** — one point for the whole group, which is
/// what makes the matrix small enough to solve.
///
/// A candidate found for a smaller group cannot hold a larger one, and is forbidden rather than
/// merely expensive.
///
/// ⚠️ **`-order` only bites on the top and left edges.** Their slot lists run backwards relative to
/// the others, so keeping a bus in declaration order *in space* means counting the slot indices
/// down. On the bottom and right edges the declared order already matches, and the flag changes
/// nothing.
pub fn match_in_section(
    slots: &[Slot],
    section: &Section,
    pins: &[Pin],
    groups: &[Group],
    members: &[usize],
    die: (i32, i32, i32, i32),
) -> Option<Vec<Placement>> {
    if members.is_empty() {
        return Some(Vec::new());
    }
    let mut sizes: Vec<usize> = members.iter().map(|&g| groups[g].pins.len()).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let (starts, caps) = starting_slots(slots, section.begin_slot, section.end_slot, &sizes);
    if starts.len() < members.len() {
        return None;
    }

    const FORBIDDEN: i64 = crate::hungarian::FORBIDDEN;
    let cost: Vec<Vec<i64>> = members
        .iter()
        .map(|&g| {
            let group = &groups[g];
            let row: Vec<i64> = starts
                .iter()
                .zip(&caps)
                .map(|(&s, &cap)| {
                    if group.pins.len() > cap {
                        return FORBIDDEN;
                    }
                    let at = (slots[s].x, slots[s].y);
                    group.pins.iter().map(|&p| pin_cost(pins, p, at, die)).sum()
                })
                .collect();
            // A group containing a mirrored pin gets the same scaled cost and tie-break rank as a
            // mirrored single (**M3**): among equal totals, prefer the start where the mirrored
            // pairs split their cost most evenly. Applying this to singles but not to groups left
            // ordered groups with a mirrored member measurably worse than the reference.
            if !group.pins.iter().any(|&p| pins[p].mirror.is_some()) {
                return row;
            }
            let balance: Vec<i64> = starts
                .iter()
                .map(|&s| {
                    let at = (slots[s].x, slots[s].y);
                    group.pins.iter().map(|&p| crate::sections::pin_balance(pins, p, at, die)).sum()
                })
                .collect();
            let rank = crate::sections::tie_break_rank(&balance);
            row.iter()
                .zip(&rank)
                .map(|(c, r)| if *c >= FORBIDDEN { *c } else { (c << 8) | *r as i64 })
                .collect()
        })
        .collect();

    // See the crate docs: this is how a group that starts one step off gets localised.
    if std::env::var_os("VYGES_PPL_GROUP_TRACE").is_some() {
        for (row, &g) in members.iter().enumerate() {
            let mut ranked: Vec<(i64, usize)> =
                cost[row].iter().copied().zip(starts.iter().copied()).collect();
            ranked.sort();
            eprintln!(
                "group {g} size {} candidates {} best: {:?}",
                groups[g].pins.len(),
                starts.len(),
                ranked.iter().take(4).map(|(c, s)| (*c, *s, slots[*s].x)).collect::<Vec<_>>()
            );
        }
    }
    let assignment = crate::hungarian::solve(&cost)?;
    let mut out = Vec::new();
    for (row, col) in assignment.into_iter().enumerate() {
        if cost[row][col] >= FORBIDDEN {
            return None;
        }
        let group = &groups[members[row]];
        let start = starts[col];
        let reversed = group.ordered && matches!(section.edge, Edge::Top | Edge::Left);
        for (n, &p) in group.pins.iter().enumerate() {
            let offset = if reversed { group.pins.len() - 1 - n } else { n };
            out.push(Placement { pin: p, slot: start + offset });
        }
    }
    Some(out)
}

/// **G5** — the first run of free slots long enough to hold a group, searched directly.
///
/// The escape hatch for a group too large for any section. It scans slot indices rather than
/// sections, so a run is allowed to **cross an edge**: a 256-pin bus on a die whose bottom edge
/// holds 200 slots continues around the corner. That is upstream's behaviour and it is what makes
/// oversized groups placeable at all.
pub fn first_run(slots: &[Slot], first: usize, last: usize, size: usize) -> Option<usize> {
    if size == 0 || last >= slots.len() || first > last {
        return None;
    }
    let mut i = first;
    while i <= last {
        if slots[i].blocked {
            i += 1;
            continue;
        }
        let start = i;
        while i <= last && !slots[i].blocked {
            i += 1;
            if i - start >= size {
                return Some(start);
            }
        }
    }
    None
}

/// **G6** — put an oversized group on consecutive slots from `start`, consuming them.
///
/// ⚠️ **The reversal here is unconditional on the top and left edges**, unlike the matched path
/// where it depends on `-order`. Reproduced as-is: it is a difference in upstream between the two
/// paths, not a rule with a stated reason, and a group placed by fallback comes out mirrored
/// relative to one placed by matching.
pub fn place_fallback(slots: &mut [Slot], group: &Group, start: usize) -> Vec<Placement> {
    let reverse = matches!(slots[start].edge, Edge::Top | Edge::Left);
    let last = group.pins.len() - 1;
    let mut out = Vec::new();
    for i in 0..=last {
        let pin = group.pins[if reverse { last - i } else { i }];
        let slot = start + i;
        slots[slot].blocked = true;
        out.push(Placement { pin, slot });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(n: usize, edge: Edge) -> Vec<Slot> {
        (0..n)
            .map(|i| Slot { x: i as i32 * 100, y: 0, layer: "met1".into(), edge, blocked: false })
            .collect()
    }

    fn pin(name: &str, sinks: &[(i32, i32)]) -> Pin {
        Pin { name: name.into(), sinks: sinks.to_vec(), mirror: None }
    }

    fn section(begin: usize, end: usize, num_slots: usize, edge: Edge) -> Section {
        Section {
            pos: (0, 0),
            num_slots,
            used_slots: 0,
            begin_slot: begin,
            end_slot: end,
            edge,
            pins: Vec::new(),
        }
    }

    #[test]
    fn contiguity_is_not_the_same_as_free_space() {
        // The distinction the whole module turns on.
        let mut slots = row(10, Edge::Bottom);
        slots[5].blocked = true;
        assert_eq!(max_contiguous(&slots, 0, 9), 5, "two runs of five, not one of nine");
        slots[0].blocked = true;
        assert_eq!(max_contiguous(&slots, 0, 9), 4);
        for s in slots.iter_mut() {
            s.blocked = true;
        }
        assert_eq!(max_contiguous(&slots, 0, 9), 0);
    }

    #[test]
    fn candidate_starts_step_by_the_group_size_rather_than_by_one() {
        let slots = row(10, Edge::Bottom);
        let (starts, caps) = starting_slots(&slots, 0, 9, &[3]);
        assert_eq!(starts, vec![0, 3, 6], "sparse, not every offset");
        assert_eq!(caps, vec![3, 3, 3]);
    }

    #[test]
    fn a_blocked_slot_pushes_the_window_past_it() {
        let mut slots = row(10, Edge::Bottom);
        slots[1].blocked = true;
        let (starts, _) = starting_slots(&slots, 0, 9, &[3]);
        assert!(!starts.contains(&0), "a window covering the blocked slot is not offered");
        assert!(starts.iter().all(|&s| (s..s + 3).all(|k| !slots[k].blocked)));
    }

    #[test]
    fn the_scan_position_carries_across_group_sizes() {
        // G2's quirk, reproduced deliberately: the smaller size resumes where the larger stopped
        // instead of restarting, so it is only offered positions in the tail.
        let slots = row(12, Edge::Bottom);
        let (starts, caps) = starting_slots(&slots, 0, 11, &[5, 2]);
        assert_eq!(&starts[..2], &[0, 5], "the size-5 scan");
        assert!(
            starts[2..].iter().all(|&s| s >= 10),
            "the size-2 scan resumed near the end, not at 0: {starts:?}"
        );
        assert!(caps.iter().filter(|&&c| c == 2).count() >= 1);
    }

    #[test]
    fn a_group_larger_than_the_run_gets_no_candidate_at_all() {
        let slots = row(4, Edge::Bottom);
        assert!(starting_slots(&slots, 0, 3, &[5]).0.is_empty());
        assert!(starting_slots(&slots, 0, 3, &[0]).0.is_empty(), "an empty group is not a position");
    }

    #[test]
    fn a_group_takes_consecutive_slots() {
        let slots = row(12, Edge::Bottom);
        let sec = section(0, 11, 12, Edge::Bottom);
        let pins: Vec<Pin> = (0..4).map(|i| pin(&format!("d{i}"), &[(500, 900)])).collect();
        let groups = vec![Group { pins: vec![0, 1, 2, 3], ordered: false }];

        let placed = match_in_section(&slots, &sec, &pins, &groups, &[0], (0, 0, 1000, 1000)).unwrap();
        let mut used: Vec<usize> = placed.iter().map(|p| p.slot).collect();
        used.sort_unstable();
        assert_eq!(used.len(), 4);
        assert!(used.windows(2).all(|w| w[1] == w[0] + 1), "not contiguous: {used:?}");
    }

    #[test]
    fn an_ordered_group_runs_forward_on_the_bottom_edge_and_backward_on_the_top() {
        // The top edge's slot list runs right-to-left, so preserving the declared order IN SPACE
        // means counting the indices down. Without this a bus comes out mirrored.
        let pins: Vec<Pin> = (0..3).map(|i| pin(&format!("d{i}"), &[])).collect();
        let groups = vec![Group { pins: vec![0, 1, 2], ordered: true }];

        for (edge, expect_forward) in
            [(Edge::Bottom, true), (Edge::Right, true), (Edge::Top, false), (Edge::Left, false)]
        {
            let slots = row(6, edge);
            let sec = section(0, 5, 6, edge);
            let placed = match_in_section(&slots, &sec, &pins, &groups, &[0], (0, 0, 1000, 1000)).unwrap();
            let slot_of = |p: usize| placed.iter().find(|x| x.pin == p).unwrap().slot;
            if expect_forward {
                assert!(slot_of(0) < slot_of(1) && slot_of(1) < slot_of(2), "{edge:?}");
            } else {
                assert!(slot_of(0) > slot_of(1) && slot_of(1) > slot_of(2), "{edge:?}");
            }
        }
    }

    #[test]
    fn an_unordered_group_is_contiguous_but_its_direction_is_not_promised() {
        // -order is what fixes the sequence; without it only adjacency is guaranteed, and the flag
        // changes nothing at all on the bottom and right edges.
        let pins: Vec<Pin> = (0..3).map(|i| pin(&format!("d{i}"), &[])).collect();
        let slots = row(6, Edge::Top);
        let sec = section(0, 5, 6, Edge::Top);
        let unordered = vec![Group { pins: vec![0, 1, 2], ordered: false }];
        let placed = match_in_section(&slots, &sec, &pins, &unordered, &[0], (0, 0, 1000, 1000)).unwrap();
        let slot_of = |p: usize| placed.iter().find(|x| x.pin == p).unwrap().slot;
        assert!(slot_of(0) < slot_of(1), "unordered follows the index direction");
    }

    #[test]
    fn two_groups_go_to_the_starts_that_suit_their_own_nets() {
        let slots = row(12, Edge::Bottom);
        let sec = section(0, 11, 12, Edge::Bottom);
        // One group's net is at the far left, the other's at the far right.
        let mut pins: Vec<Pin> = (0..3).map(|i| pin(&format!("l{i}"), &[(0, 900)])).collect();
        pins.extend((0..3).map(|i| pin(&format!("r{i}"), &[(1100, 900)])));
        let groups = vec![
            Group { pins: vec![0, 1, 2], ordered: false },
            Group { pins: vec![3, 4, 5], ordered: false },
        ];
        let placed = match_in_section(&slots, &sec, &pins, &groups, &[0, 1], (0, 0, 1000, 1000)).unwrap();
        let x = |p: usize| slots[placed.iter().find(|q| q.pin == p).unwrap().slot].x;
        assert!(x(0) < x(3), "the left-net group is left of the right-net group");
        let mut used: Vec<usize> = placed.iter().map(|p| p.slot).collect();
        used.sort_unstable();
        used.dedup();
        assert_eq!(used.len(), 6, "the two groups overlap");
    }

    #[test]
    fn a_section_without_a_long_enough_run_is_passed_over() {
        // Room is not the same as a run: this section has four free slots and no run of three.
        let mut slots = row(8, Edge::Bottom);
        slots[2].blocked = true;
        slots[4].blocked = true;
        slots[6].blocked = true;
        let mut sections = vec![section(0, 7, 5, Edge::Bottom)];
        let pins: Vec<Pin> = (0..3).map(|i| pin(&format!("d{i}"), &[])).collect();
        let g = Group { pins: vec![0, 1, 2], ordered: false };
        assert_eq!(assign_to_section(&g, &mut sections, &[vec![], vec![]], &pins, &slots, 200, (0, 0, 1000, 1000)), None);
    }

    #[test]
    fn a_group_goes_to_the_section_nearest_its_nets() {
        let slots = row(20, Edge::Bottom);
        let mut sections = vec![section(0, 9, 10, Edge::Bottom), section(10, 19, 10, Edge::Bottom)];
        sections[0].pos = (0, 0);
        sections[1].pos = (1900, 0);
        let pins: Vec<Pin> = (0..3).map(|i| pin(&format!("d{i}"), &[(1900, 800)])).collect();
        let g = Group { pins: vec![0, 1, 2], ordered: false };
        assert_eq!(
            assign_to_section(&g, &mut sections, &[vec![], vec![]], &pins, &slots, 200, (0, 0, 1000, 1000)),
            Some(1)
        );
        assert_eq!(sections[1].used_slots, 3, "the whole group is accounted for at once");
    }

    #[test]
    fn a_large_group_will_not_share_a_section_with_another_group() {
        // Upstream's rule, and it is about keeping the later matching solvable rather than about
        // quality: two big groups in one section can leave the matrix with no legal assignment.
        let slots = row(40, Edge::Bottom);
        let mut sections = vec![section(0, 19, 20, Edge::Bottom), section(20, 39, 20, Edge::Bottom)];
        let pins: Vec<Pin> = (0..12).map(|i| pin(&format!("d{i}"), &[])).collect();
        let big = Group { pins: (0..11).collect(), ordered: false };
        // Section 0 already holds a group; 11 > 20/2, so it must go elsewhere.
        let chosen =
            assign_to_section(&big, &mut sections, &[vec![2], vec![]], &pins, &slots, 20, (0, 0, 1000, 1000));
        assert_eq!(chosen, Some(1));
    }

    #[test]
    fn a_section_already_holding_a_large_group_takes_no_more() {
        let slots = row(40, Edge::Bottom);
        let mut sections = vec![section(0, 19, 20, Edge::Bottom), section(20, 39, 20, Edge::Bottom)];
        let pins: Vec<Pin> = (0..2).map(|i| pin(&format!("d{i}"), &[])).collect();
        let small = Group { pins: vec![0, 1], ordered: false };
        let chosen = assign_to_section(&small, &mut sections, &[vec![15], vec![]], &pins, &slots, 20, (0, 0, 1000, 1000));
        assert_eq!(chosen, Some(1), "even a small group stays out");
    }

    #[test]
    fn a_group_that_fits_nowhere_is_refused_rather_than_split() {
        let slots = row(4, Edge::Bottom);
        let mut sections = vec![section(0, 3, 4, Edge::Bottom)];
        let pins: Vec<Pin> = (0..6).map(|i| pin(&format!("d{i}"), &[])).collect();
        let g = Group { pins: (0..6).collect(), ordered: false };
        assert_eq!(assign_to_section(&g, &mut sections, &[vec![], vec![]], &pins, &slots, 200, (0, 0, 1000, 1000)), None);
        assert_eq!(sections[0].used_slots, 0, "and nothing was consumed on the way");

        let sec = section(0, 3, 4, Edge::Bottom);
        assert!(match_in_section(&slots, &sec, &pins, &[g], &[0], (0, 0, 1000, 1000)).is_none());
    }

    #[test]
    fn a_fallback_run_is_the_first_one_long_enough() {
        let mut slots = row(20, Edge::Bottom);
        for k in 3..6 {
            slots[k].blocked = true;
        }
        assert_eq!(first_run(&slots, 0, 19, 3), Some(0), "the first run fits");
        assert_eq!(first_run(&slots, 0, 19, 4), Some(6), "the first run is too short");
        assert_eq!(first_run(&slots, 0, 19, 15), None, "nothing is long enough");
        assert_eq!(first_run(&slots, 0, 19, 0), None);
    }

    #[test]
    fn a_fallback_run_may_cross_an_edge() {
        // What makes an oversized group placeable at all: no single edge holds it.
        let mut slots = row(6, Edge::Bottom);
        slots.extend(row(6, Edge::Right));
        assert_eq!(first_run(&slots, 0, 11, 9), Some(0));
    }

    #[test]
    fn a_fallback_group_takes_consecutive_slots_and_consumes_them() {
        let mut slots = row(10, Edge::Bottom);
        let g = Group { pins: vec![0, 1, 2, 3], ordered: true };
        let placed = place_fallback(&mut slots, &g, 2);
        assert_eq!(
            placed,
            vec![
                Placement { pin: 0, slot: 2 },
                Placement { pin: 1, slot: 3 },
                Placement { pin: 2, slot: 4 },
                Placement { pin: 3, slot: 5 },
            ]
        );
        assert!(slots[2..6].iter().all(|s| s.blocked), "the slots are withdrawn");
        assert!(!slots[6].blocked);
    }

    #[test]
    fn fallback_reverses_on_the_top_edge_regardless_of_the_order_flag() {
        // G6's quirk: the matched path checks `ordered`, this one does not.
        let mut slots = row(10, Edge::Top);
        let g = Group { pins: vec![0, 1, 2], ordered: false };
        let placed = place_fallback(&mut slots, &g, 0);
        assert_eq!(placed[0], Placement { pin: 2, slot: 0 }, "declared order runs backwards");
        assert_eq!(placed[2], Placement { pin: 0, slot: 2 });
    }

    #[test]
    fn nothing_to_place_is_not_an_error() {
        let slots = row(6, Edge::Bottom);
        let sec = section(0, 5, 6, Edge::Bottom);
        assert_eq!(match_in_section(&slots, &sec, &[], &[], &[], (0, 0, 1000, 1000)), Some(Vec::new()));
    }
}
