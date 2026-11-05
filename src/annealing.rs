// SPDX-License-Identifier: Apache-2.0
//! Simulated annealing — the other optimiser.
//!
//! Hungarian matching finds the optimum *within a section* and is greedy between sections.
//! Annealing throws that decomposition away: it starts from a random assignment over the whole
//! boundary, perturbs it, and keeps changes that help — plus, with a probability that falls as the
//! run "cools", changes that hurt. Accepting a worse placement occasionally is the entire point;
//! it is what lets the search leave a local minimum that the greedy split cannot.
//!
//! # Reproducing it exactly
//!
//! Every decision here comes from [`crate::rng`], which reproduces the reference's Boost stream
//! bit for bit. That is what makes an annealed placement comparable at all: the algorithm is
//! deterministic given its stream, so matching the stream and matching the algorithm is enough.
//!
//! **The order and number of draws is therefore part of the algorithm**, not an implementation
//! detail. A rejected candidate that costs a draw must cost a draw here too; a branch that returns
//! early without drawing must not draw. One draw out of step and every later decision differs.
//!
//! # Scope
//!
//! The whole reference annealer: lone pins, region constraints, pin groups (ordered or not) and
//! mirrored pairs, with every move the reference makes — swap, move to a free slot, move a group
//! to free slots, shift a group, rearrange the groups sharing a constraint. Each is its own
//! function below, named after the reference function it transcribes and kept in its order.
//!
//! 🔑 **The reference's scratch containers are state, and they are kept as it keeps them.** A
//! shift that finds no room returns a failure WITHOUT clearing what it recorded or re-marking the
//! slots it freed; the lone-pin move that follows appends to the same lists, and a lone move that
//! then fails clears them — leaving the group's slots free for the rest of the run. That is the
//! reference's behaviour and the stream and the placement both depend on it, so it is reproduced,
//! not repaired.
//!
//! What still declines: a pin sent to the top layer (the reference anneals only the boundary),
//! and a design whose search the reference itself would never finish (a lone-pin move with no
//! lone pin, which spins forever there).
//!
//! `VYGES_PPL_ANNEAL_TRACE` prints the `VYGPPL sa_*` lines the instrumented reference prints
//! (`correlation/instruments/ppl/ppl-anneal-patch.py`), so the two diff line for line.
//!
//! Nothing here touches a database.

use std::collections::BTreeSet;

use crate::constraints::Constraint;
use crate::groups::Group;
use crate::rng::Mt19937;
use crate::sections::{Pin, Placement};
use crate::slots::{Edge, Slot};

/// The knobs `set_simulated_annealing` exposes, with the reference's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct Anneal {
    pub init_temperature: f32,
    pub max_iterations: i32,
    /// Perturbations per temperature step. Zero means "derive it from the design", which is the
    /// usual case — see [`perturbations_for`].
    pub perturb_per_iter: i32,
    /// How fast the temperature falls. Below 1, so late iterations accept almost nothing worse.
    pub alpha: f32,
    pub seed: u32,
}

impl Default for Anneal {
    fn default() -> Self {
        Anneal {
            init_temperature: 1.0,
            max_iterations: 2000,
            perturb_per_iter: 0,
            alpha: 0.985,
            seed: 42,
        }
    }
}

/// Below this draw a perturbation swaps two pins; at or above it, one pin moves to a free slot.
const SWAP_PINS: f32 = 0.5;
/// A constrained group's move: above this draw (and with company) rearrange, else shift.
const SHIFT_GROUP: f32 = 0.8;
/// A roomy constraint's group: at or below this draw move to free slots, else shift.
const GROUP_TO_FREE_SLOTS: f32 = 0.7;
/// A constraint at or below this many pins per slot counts as roomy.
const PINS_PER_SLOT_LIMIT: f32 = 0.5;
/// A move that could not be made. Distinct from a zero-cost move, which is a perfectly good one.
const MOVE_FAIL: i64 = -1;

/// How many perturbations one temperature step gets, when the caller does not say.
///
/// Rule (`SimulatedAnnealing`'s constructor): `lone_pins * 0.8 + groups * 10`, truncated, where a
/// lone pin is one not in a group — a mirrored partner of a grouped pin counting as grouped too.
pub fn perturbations_for(lone_pins: usize, groups: usize) -> i32 {
    (lone_pins as f64 * 0.8 + groups as f64 * 10.0) as i32
}

/// Why a design cannot be annealed by this implementation, or why the reference stops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// A pin sent to the top layer: the reference anneals the boundary slots only.
    NeedsExtraMoves,
    /// Nothing to place, or nowhere to put it.
    Empty,
    /// The reference's search would never return (a lone-pin move with no lone pin).
    Hangs,
    /// The reference stops with this error (`PPL-<code>`).
    Error(u32, String),
}

/// One region constraint as the annealer sees it (`Constraint` after `initConstraints(true)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Its pins, ascending (the database's own order).
    pub pins: Vec<usize>,
    /// The groups any of its pins belong to, ascending.
    pub groups: BTreeSet<usize>,
    /// The slot range it may use, both ends inclusive.
    pub first_slot: usize,
    pub last_slot: usize,
    /// Pins over usable slots, in `f32` as the reference divides.
    pub pins_per_slots: f32,
}

/// `createSectionsPerConstraint`'s slot range for one interval.
///
/// Rule: for each layer on the constraint's edge, in slot-list order, the run begins at the first
/// slot of that edge and layer inside the interval — `along >= begin` on bottom/right, `along <=
/// end` on top/left — and ends just before the first slot from there that reaches the far end
/// (`>= end`, resp. `<= begin`) or leaves the edge or layer. The range starts where the FIRST
/// layer's run begins and ends where the LAST layer's run ends.
pub fn constraint_slot_range(slots: &[Slot], c: &Constraint) -> Option<(usize, usize)> {
    let edge = c.interval.edge;
    let (begin, end) = (c.interval.begin, c.interval.end);
    let along = |s: &Slot| if s.edge.is_vertical_pin() { s.x } else { s.y };
    let ascending = matches!(edge, Edge::Bottom | Edge::Right);
    let mut layers: Vec<&str> = Vec::new();
    for s in slots.iter().filter(|s| s.edge == edge) {
        if !layers.contains(&s.layer.as_str()) {
            layers.push(&s.layer);
        }
    }
    let mut range: Option<(usize, usize)> = None;
    for layer in layers {
        let start = slots.iter().position(|s| {
            s.edge == edge && s.layer == layer && if ascending { along(s) >= begin } else { along(s) <= end }
        })?;
        let stop = slots[start..]
            .iter()
            .position(|s| {
                (if ascending { along(s) >= end } else { along(s) <= begin }) || s.edge != edge || s.layer != layer
            })
            .map_or(slots.len(), |k| start + k);
        let last = stop.checked_sub(1)?;
        range = Some((range.map_or(start, |r| r.0), last));
    }
    range
}

/// `initConstraints(true)`: each constraint's range and pressure, in the order given — an
/// annealing run does NOT sort them (`sortConstraints` is the matcher's alone).
pub fn regions(slots: &[Slot], constraints: &[Constraint], groups: &[Group], slots_per_section: usize) -> Vec<Region> {
    let group_of = |p: usize| groups.iter().position(|g| g.pins.contains(&p));
    constraints
        .iter()
        .map(|c| {
            let (first_slot, last_slot) = constraint_slot_range(slots, c).unwrap_or((0, 0));
            let usable: usize =
                crate::constraints::sections_for(slots, c.interval, slots_per_section).iter().map(|s| s.num_slots).sum();
            let mut pins = c.pins.clone();
            pins.sort_unstable();
            Region {
                groups: pins.iter().filter_map(|&p| group_of(p)).collect(),
                pins,
                first_slot,
                last_slot,
                pins_per_slots: c.pins.len() as f32 / usable as f32,
            }
        })
        .collect()
}

/// A plain run: no groups, no constraints.
pub fn run(slots: &[Slot], pins: &[Pin], die: (i32, i32, i32, i32), cfg: &Anneal) -> Result<Vec<Placement>, Unsupported> {
    run_with(slots, pins, &[], &[], die, cfg)
}

/// `SimulatedAnnealing::run` then `getAssignment`: one annealing run over the boundary.
pub fn run_with(
    slots: &[Slot],
    pins: &[Pin],
    groups: &[Group],
    regions: &[Region],
    die: (i32, i32, i32, i32),
    cfg: &Anneal,
) -> Result<Vec<Placement>, Unsupported> {
    let usable = slots.iter().filter(|s| !s.blocked).count();
    if pins.is_empty() || usable < pins.len() {
        return Err(Unsupported::Empty);
    }
    let mut state = State::new(slots, pins, groups, regions, die, cfg);
    state.run(cfg)?;
    Ok(state.get_assignment())
}

/// Every pin's standing, as the reference's `IOPin` carries it.
#[derive(Debug, Clone, Default)]
struct PinInfo {
    group: Option<usize>,
    constraint: Option<usize>,
    /// Both halves of a mirrored pair are mirrored (`initMirroredPins` marks the two).
    mirror: Option<usize>,
}

struct State<'a> {
    slots: &'a [Slot],
    pins: &'a [Pin],
    /// `Core`'s boundary: what a position is reflected across.
    die: (i32, i32, i32, i32),
    info: Vec<PinInfo>,
    /// The groups' pin orders. Mutable: `sortPinsFromGroup` reverses an ordered group IN PLACE
    /// every time it lands on the top or left edge, and later moves read the reversed order.
    groups: Vec<Group>,
    regions: &'a [Region],
    /// `getSlotIdxByPosition(getMirroredPosition(slot))` for every slot: the first slot at the
    /// reflected position on the same layer.
    mirror_slot: Vec<Option<usize>>,
    /// `pin_assignment_`: which slot each pin holds.
    assignment: Vec<usize>,
    /// `Slot::used`.
    used: Vec<bool>,
    /// `generator_`, the search's engine.
    gen: Mt19937,
    seed: u32,
    lone_pins: usize,
    swappable_pins: usize,
    unconstrained_swappable_pins: usize,
    perturb_per_iter: i32,
    /// `pins_`, `prev_slots_`, `new_slots_`: what one perturbation touched.
    moved: Vec<usize>,
    prev_slots: Vec<usize>,
    new_slots: Vec<usize>,
    trace: bool,
    /// One letter per move function entered this perturbation (the trace's `path`).
    path: String,
}

fn join(v: &[usize]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

impl<'a> State<'a> {
    /// The constructor: pins' standings, slot mirrors, `countLonePins`, the default effort.
    fn new(slots: &'a [Slot], pins: &'a [Pin], groups: &[Group], regions: &'a [Region], die: (i32, i32, i32, i32), cfg: &Anneal) -> State<'a> {
        let mut info = vec![PinInfo::default(); pins.len()];
        for (g, group) in groups.iter().enumerate() {
            for &p in &group.pins {
                info[p].group = Some(g);
            }
        }
        for (c, r) in regions.iter().enumerate() {
            for &p in &r.pins {
                info[p].constraint = Some(c);
            }
        }
        for (i, p) in pins.iter().enumerate() {
            if let Some(m) = p.mirror {
                info[i].mirror = Some(m);
                info[m].mirror = Some(i);
            }
        }
        let mirror_slot = slots
            .iter()
            .map(|s| {
                let (x, y) = crate::sections::mirrored_position((s.x, s.y), die);
                slots.iter().position(|t| t.x == x && t.y == y && t.layer == s.layer)
            })
            .collect();
        let mut st = State {
            slots,
            pins,
            die,
            info,
            groups: groups.to_vec(),
            regions,
            mirror_slot,
            assignment: vec![0; pins.len()],
            used: vec![false; slots.len()],
            gen: Mt19937::new(cfg.seed),
            seed: cfg.seed,
            lone_pins: 0,
            swappable_pins: 0,
            unconstrained_swappable_pins: 0,
            perturb_per_iter: 0,
            moved: Vec::new(),
            prev_slots: Vec::new(),
            new_slots: Vec::new(),
            trace: std::env::var_os("VYGES_PPL_ANNEAL_TRACE").is_some(),
            path: String::new(),
        };
        st.count_lone_pins();
        st.perturb_per_iter = perturbations_for(st.lone_pins, st.groups.len());
        st
    }

    fn mirrored(&self, p: usize) -> bool {
        self.info[p].mirror.is_some()
    }

    fn in_group(&self, p: usize) -> bool {
        self.info[p].group.is_some()
    }

    fn available(&self, slot: usize) -> bool {
        !self.used[slot] && !self.slots[slot].blocked
    }

    /// `run`: the opening assignment, then the cooling loop.
    fn run(&mut self, cfg: &Anneal) -> Result<(), Unsupported> {
        self.init(cfg);
        self.random_assignment()?;
        let mut pre_cost = self.get_assignment_cost();
        let mut temperature = cfg.init_temperature;
        if cfg.init_temperature == 0.0 {
            temperature = 1.0;
        }
        let alpha = if cfg.alpha != 0.0 { cfg.alpha } else { 0.985 };
        let max_iterations = if cfg.max_iterations != 0 { cfg.max_iterations } else { 2000 };
        self.trace_start();
        for iter in 0..max_iterations {
            for perturb in 0..self.perturb_per_iter {
                let prev_cost = self.perturb_assignment()?;
                let cost = pre_cost + self.get_delta_cost(prev_cost);
                let delta_cost = cost - pre_cost;
                let rand_float = self.gen.uniform_real();
                let accept_prob = (-(delta_cost as f32) / temperature).exp();
                let accept = delta_cost <= 0 || accept_prob > rand_float;
                if self.trace {
                    eprintln!(
                        "VYGPPL sa_p|{iter}|{perturb}|{}|{}|{}|{}|{prev_cost}|{delta_cost}|{}|{}",
                        self.path,
                        join(&self.moved),
                        join(&self.prev_slots),
                        join(&self.new_slots),
                        accept as u8,
                        self.gen.draws()
                    );
                }
                self.path.clear();
                if accept {
                    pre_cost = cost;
                    if !self.prev_slots.is_empty() && !self.new_slots.is_empty() {
                        for &s in &self.prev_slots {
                            self.used[s] = false;
                        }
                        for &s in &self.new_slots {
                            self.used[s] = true;
                        }
                    }
                } else {
                    for &s in &self.prev_slots {
                        self.used[s] = true;
                    }
                    self.restore_previous_assignment();
                }
                self.prev_slots.clear();
                self.new_slots.clear();
                self.moved.clear();
            }
            temperature *= alpha;
        }
        if self.trace {
            eprintln!("VYGPPL sa_final|{}|{}", self.gen.draws(), join(&self.assignment));
        }
        Ok(())
    }

    /// `getAssignment`: every pin at the slot it holds.
    fn get_assignment(&self) -> Vec<Placement> {
        (0..self.pins.len()).map(|p| Placement { pin: p, slot: self.assignment[p] }).collect()
    }

    /// `init`: an explicit per-iteration effort overrides the derived one; the engine is seeded.
    fn init(&mut self, cfg: &Anneal) {
        if cfg.perturb_per_iter != 0 {
            self.perturb_per_iter = cfg.perturb_per_iter;
        }
        self.gen = Mt19937::new(self.seed);
    }

    fn trace_start(&self) {
        if !self.trace {
            return;
        }
        eprintln!(
            "VYGPPL sa_init|{}|{}|{}|{}|{}|{}|{}",
            self.pins.len(),
            self.slots.len(),
            self.groups.len(),
            self.lone_pins,
            self.swappable_pins,
            self.unconstrained_swappable_pins,
            self.perturb_per_iter
        );
        for (c, r) in self.regions.iter().enumerate() {
            let g: Vec<usize> = r.groups.iter().copied().collect();
            eprintln!(
                "VYGPPL sa_con|{c}|{}|{}|{:08x}|{}|{}",
                r.first_slot,
                r.last_slot,
                r.pins_per_slots.to_bits(),
                join(&r.pins),
                join(&g)
            );
        }
        for (g, group) in self.groups.iter().enumerate() {
            eprintln!("VYGPPL sa_grp|{g}|{}|{}", group.ordered as u8, join(&group.pins));
        }
        let idx = |v: Option<usize>| v.map_or(-1, |x| x as i64);
        for (i, p) in self.pins.iter().enumerate() {
            let inf = &self.info[i];
            eprintln!(
                "VYGPPL sa_pin|{i}|{}|{}|{}|{}|{}",
                p.name,
                idx(inf.group),
                idx(inf.constraint),
                inf.mirror.is_some() as u8,
                idx(inf.mirror)
            );
        }
        eprintln!("VYGPPL sa_rand|{}|{}", self.gen.draws(), join(&self.assignment));
    }

    /// `randomAssignment`: groups first, then constrained pins by draws in their range, then the
    /// rest from a shuffled slot order.
    ///
    /// Rule: the shuffle runs on an engine of its OWN, seeded like the search's, while constrained
    /// pins and constrained groups draw from the SEARCH engine — so the opening already advances
    /// the stream the cooling loop continues. A placed pin's mirror partner is placed with it and
    /// skipped later; a pin that is not mirrored records partner "-1", which matches nothing.
    fn random_assignment(&mut self) -> Result<(), Unsupported> {
        let mut slot_indices: Vec<usize> = (0..self.slots.len()).collect();
        let mut shuffler = Mt19937::new(self.seed);
        shuffler.shuffle(&mut slot_indices);

        let mut placed_pins: BTreeSet<usize> = BTreeSet::new();
        let mut slot_idx = self.random_assignment_for_groups(&mut placed_pins, &slot_indices)?;

        let n_slots = self.slots.len();
        for only_constraints in [true, false] {
            for i in 0..self.pins.len() {
                if placed_pins.contains(&i) {
                    continue;
                }
                let constrained = self.info[i].constraint.is_some();
                if constrained && only_constraints {
                    let (first, last) = self.get_slots_range(i);
                    let mut slot = self.gen.uniform_int(first as i64, last as i64) as usize;
                    let mut mirrored_slot = 0;
                    let mut free = self.available(slot);
                    if self.mirrored(i) && free {
                        free = self.is_free_for_mirrored(slot, &mut mirrored_slot)?;
                    }
                    let mut attempts = 0;
                    while !free && attempts < n_slots * 10 {
                        slot = self.gen.uniform_int(first as i64, last as i64) as usize;
                        free = self.available(slot);
                        if self.mirrored(i) && free {
                            free = self.is_free_for_mirrored(slot, &mut mirrored_slot)?;
                        }
                        attempts += 1;
                    }
                    if !free {
                        return Err(Unsupported::Error(117, format!("No available slot found for constrained pin {}.", self.pins[i].name)));
                    }
                    self.place_pin(i, slot, mirrored_slot, &mut placed_pins);
                }
                if !constrained && !only_constraints {
                    let mut slot = *slot_indices.get(slot_idx).ok_or(Unsupported::Empty)?;
                    let mut mirrored_slot = 0;
                    let mut free = self.available(slot);
                    if self.mirrored(i) && free {
                        free = self.is_free_for_mirrored(slot, &mut mirrored_slot)?;
                    }
                    while !free && slot_idx < slot_indices.len() - 1 {
                        slot_idx += 1;
                        slot = slot_indices[slot_idx];
                        free = self.available(slot);
                        if self.mirrored(i) && free {
                            free = self.is_free_for_mirrored(slot, &mut mirrored_slot)?;
                        }
                    }
                    if !free {
                        return Err(Unsupported::Error(114, format!("No available slot found for unconstrained pin {}.", self.pins[i].name)));
                    }
                    self.place_pin(i, slot, mirrored_slot, &mut placed_pins);
                    slot_idx += 1;
                }
            }
        }
        Ok(())
    }

    /// The tail both branches of `randomAssignment` share: the pin and its partner take their
    /// slots, and the PARTNER (not the pin) is recorded as placed.
    fn place_pin(&mut self, i: usize, slot: usize, mirrored_slot: usize, placed_pins: &mut BTreeSet<usize>) {
        self.assignment[i] = slot;
        self.used[slot] = true;
        if let Some(m) = self.info[i].mirror {
            self.assignment[m] = mirrored_slot;
            self.used[mirrored_slot] = true;
            placed_pins.insert(m);
        }
    }

    /// `randomAssignmentForGroups`: each group at a contiguous free run.
    ///
    /// Rule: a constrained group draws its start in its range until `isFreeForGroup` (which walks
    /// the start FORWARD to a free run) succeeds, and takes where the walk stopped. An unconstrained
    /// group walks the shuffled order instead, and takes the shuffled entry at its index — NOT the
    /// position `isFreeForGroup` walked to. Every group, placed or not, advances that index.
    fn random_assignment_for_groups(&mut self, placed_pins: &mut BTreeSet<usize>, slot_indices: &[usize]) -> Result<usize, Unsupported> {
        let n_slots = self.slots.len();
        let mut slot_idx = 0;
        for g in 0..self.groups.len() {
            let size = self.groups[g].pins.len();
            let first_pin = self.groups[g].pins[0];
            let mut group_slot;
            if self.info[first_pin].constraint.is_some() {
                let (first, last) = self.get_slots_range(first_pin);
                let mut slot = self.gen.uniform_int(first as i64, last as i64) as usize;
                let mut attempts = 0;
                while !self.is_free_for_group(&mut slot, size, last)? && attempts < n_slots * 10 {
                    slot = self.gen.uniform_int(first as i64, last as i64) as usize;
                    attempts += 1;
                }
                if attempts >= n_slots * 10 {
                    return Err(Unsupported::Error(115, "No available contiguous slots found for constrained pin group.".into()));
                }
                group_slot = slot;
            } else {
                let mut slot = *slot_indices.get(slot_idx).ok_or(Unsupported::Empty)?;
                while !self.is_free_for_group(&mut slot, size, n_slots - 1)? && slot_idx < slot_indices.len() - 1 {
                    slot_idx += 1;
                    slot = slot_indices[slot_idx];
                }
                if !self.is_free_for_group(&mut slot, size, n_slots - 1)? {
                    return Err(Unsupported::Error(116, "No available contiguous slots found for unconstrained pin group.".into()));
                }
                group_slot = slot_indices[slot_idx];
            }
            for k in 0..size {
                let p = self.groups[g].pins[k];
                self.assignment[p] = group_slot;
                self.used[group_slot] = true;
                if let Some(m) = self.info[p].mirror {
                    let ms = self.get_mirrored_slot_idx(group_slot)?;
                    self.assignment[m] = ms;
                    self.used[ms] = true;
                    placed_pins.insert(m);
                }
                group_slot += 1;
                placed_pins.insert(p);
            }
            slot_idx += 1;
        }
        Ok(slot_idx)
    }

    /// `getAssignmentCost`: every pin's own cost, summed.
    fn get_assignment_cost(&self) -> i64 {
        (0..self.pins.len()).map(|p| self.get_pin_cost(p)).sum()
    }

    /// `getDeltaCost`: what the touched pins cost now, less what they cost before.
    fn get_delta_cost(&self, prev_cost: i64) -> i64 {
        let new_cost: i64 = self.moved.iter().map(|&p| self.get_pin_cost(p)).sum();
        new_cost - prev_cost
    }

    /// `getPinCost`: the pin's OWN net cost at its slot — a mirrored partner is costed separately.
    fn get_pin_cost(&self, pin: usize) -> i64 {
        let s = &self.slots[self.assignment[pin]];
        crate::sections::io_net_cost(&self.pins[pin], (s.x, s.y))
    }

    /// `getGroupCost`.
    fn get_group_cost(&self, g: usize) -> i64 {
        self.groups[g].pins.iter().map(|&p| self.get_pin_cost(p)).sum()
    }

    /// `perturbAssignment`: one draw picks swap or move; a failed swap falls through to a move,
    /// and a failed move to a lone-pin move.
    fn perturb_assignment(&mut self) -> Result<i64, Unsupported> {
        let mv = self.gen.uniform_real();
        let mut prev_cost = 0;
        if mv < SWAP_PINS && self.swappable_pins > 1 {
            prev_cost = self.swap_pins();
        }
        if mv >= SWAP_PINS || self.swappable_pins <= 1 || prev_cost == MOVE_FAIL {
            prev_cost = self.move_pin_to_free_slot(false)?;
            if prev_cost == MOVE_FAIL {
                prev_cost = self.move_pin_to_free_slot(true)?;
            }
        }
        Ok(prev_cost)
    }

    /// `swapPins`.
    ///
    /// Rule: `pin1` is redrawn over ALL pins until it is neither grouped nor mirrored. A
    /// constrained `pin1` takes its partner from its constraint's own pin list — a count of
    /// eligible partners first (failing when none), then draws over that list's length; an
    /// unconstrained one fails unless two unconstrained swappable pins exist, then redraws over all
    /// pins until the partner is swappable and unconstrained.
    fn swap_pins(&mut self) -> i64 {
        self.path.push('S');
        let n = self.pins.len() as i64;
        let mut pin1 = self.gen.uniform_int(0, n - 1) as usize;
        while self.in_group(pin1) || self.mirrored(pin1) {
            pin1 = self.gen.uniform_int(0, n - 1) as usize;
        }
        let swappable = |st: &Self, p: usize| !st.in_group(p) && !st.mirrored(p);
        let pin2;
        if let Some(c) = self.info[pin1].constraint {
            let list = &self.regions[c].pins;
            let eligible = list.iter().filter(|&&p| p != pin1 && swappable(self, p)).count();
            if eligible < 1 {
                return MOVE_FAIL;
            }
            let len = list.len() as i64;
            let mut p2 = list[self.gen.uniform_int(0, len - 1) as usize];
            while pin1 == p2 || !swappable(self, p2) {
                p2 = list[self.gen.uniform_int(0, len - 1) as usize];
            }
            pin2 = p2;
        } else {
            if self.unconstrained_swappable_pins < 2 {
                return MOVE_FAIL;
            }
            let mut p2 = self.gen.uniform_int(0, n - 1) as usize;
            while pin1 == p2 || !swappable(self, p2) || self.info[p2].constraint.is_some() {
                p2 = self.gen.uniform_int(0, n - 1) as usize;
            }
            pin2 = p2;
        }
        self.moved.push(pin1);
        self.moved.push(pin2);
        self.prev_slots.push(self.assignment[pin1]);
        self.prev_slots.push(self.assignment[pin2]);
        let prev_cost = self.get_pin_cost(pin1) + self.get_pin_cost(pin2);
        self.assignment.swap(pin1, pin2);
        prev_cost
    }

    /// `movePinToFreeSlot`.
    ///
    /// Rule: a grouped pin — or a mirrored pin whose partner is grouped — moves its group instead
    /// (unless this is the lone-pin retry, which redraws until the pin is neither grouped nor
    /// mirrored). Otherwise draws in the pin's range until a free slot (for a mirrored pin: free
    /// with a free reflection; for a plain one: free and not its own) or `10 x slots` attempts.
    fn move_pin_to_free_slot(&mut self, lone_pin: bool) -> Result<i64, Unsupported> {
        self.path.push(if lone_pin { 'L' } else { 'M' });
        let n = self.pins.len() as i64;
        let mut pin = self.gen.uniform_int(0, n - 1) as usize;
        if lone_pin {
            if self.swappable_pins == 0 {
                return Err(Unsupported::Hangs);
            }
            while self.in_group(pin) || self.mirrored(pin) {
                pin = self.gen.uniform_int(0, n - 1) as usize;
            }
        } else {
            if self.in_group(pin) {
                return self.move_group(pin);
            }
            if let Some(m) = self.info[pin].mirror {
                if self.in_group(m) {
                    return self.move_group(m);
                }
            }
        }

        self.moved.push(pin);
        let prev_slot = self.assignment[pin];
        self.prev_slots.push(prev_slot);
        let mirror = self.info[pin].mirror;
        let mut prev_cost = self.get_pin_cost(pin);
        if let Some(m) = mirror {
            prev_cost += self.get_pin_cost(m);
        }

        let (first, last) = self.get_slots_range(pin);
        let mut free_slot = false;
        let mut new_slot = 0;
        let mut mirrored_slot = 0;
        let mut attempts = 0;
        while !free_slot && attempts < self.slots.len() * 10 {
            new_slot = self.gen.uniform_int(first as i64, last as i64) as usize;
            free_slot = if mirror.is_some() {
                self.is_free_for_mirrored(new_slot, &mut mirrored_slot)?
            } else {
                self.available(new_slot) && new_slot != prev_slot
            };
            attempts += 1;
        }
        if !free_slot {
            self.prev_slots.clear();
            self.moved.clear();
            return Ok(MOVE_FAIL);
        }
        self.new_slots.push(new_slot);
        self.assignment[pin] = new_slot;
        if let Some(m) = mirror {
            self.moved.push(m);
            self.prev_slots.push(self.assignment[m]);
            self.new_slots.push(mirrored_slot);
            self.assignment[m] = mirrored_slot;
        }
        Ok(prev_cost)
    }

    /// `moveGroupToFreeSlots`.
    ///
    /// Rule: the group's slots are released, then a start is drawn in the range of the FIRST
    /// touched pin until the whole run is free (and reflected-free when that pin is mirrored) and
    /// on one edge; a start whose run would pass the end of the slot list costs an attempt and is
    /// not examined. Success re-orders the group for the edge it lands on; failure re-marks the
    /// released slots and clears the scratch lists.
    fn move_group_to_free_slots(&mut self, g: usize) -> Result<i64, Unsupported> {
        self.path.push('F');
        let prev_cost = self.compute_group_prev_cost(g);
        self.update_slots_from_group(false);
        let pin = self.moved[0];
        let pin_mirrored = self.mirrored(pin);
        let size = self.groups[g].pins.len();
        let n_slots = self.slots.len();
        let (first, last) = self.get_slots_range(pin);

        let (mut free_slot, mut same_edge) = (false, false);
        let mut new_slot = 0;
        let mut iter = 0;
        while (!free_slot || !same_edge) && iter < n_slots * 10 {
            new_slot = self.gen.uniform_int(first as i64, last as i64) as usize;
            if new_slot + size > n_slots {
                iter += 1;
                continue;
            }
            let edge = self.slots[new_slot].edge;
            for slot in new_slot..new_slot + size {
                free_slot = self.available(slot);
                if pin_mirrored && free_slot {
                    let mut ms = 0;
                    free_slot = self.is_free_for_mirrored(slot, &mut ms)?;
                }
                same_edge = self.slots[slot].edge == edge;
                if !free_slot || !same_edge {
                    break;
                }
            }
            iter += 1;
        }
        if free_slot && same_edge {
            self.sort_pins_from_group(g, self.slots[new_slot].edge);
            let pins = self.groups[g].pins.clone();
            self.update_group_slots(&pins, &mut new_slot)?;
        } else {
            self.update_slots_from_group(true);
            self.prev_slots.clear();
            self.new_slots.clear();
            self.moved.clear();
            return Ok(MOVE_FAIL);
        }
        Ok(prev_cost)
    }

    /// `shiftGroup`.
    ///
    /// Rule: release the group's slots, count the free same-edge run below its lowest slot and
    /// above it (the upward count starts beside the lowest slot, so it includes the group's own
    /// released slots, and the group's size is then taken off, floored at zero); shift toward the
    /// longer side (upward on a tie) by a draw in `[1, count]`. With no room it returns the
    /// failure WITHOUT re-marking its slots or clearing what it recorded.
    fn shift_group(&mut self, g: usize) -> Result<i64, Unsupported> {
        self.path.push('H');
        let pins = self.groups[g].pins.clone();
        let is_mirrored = pins.iter().any(|&p| self.mirrored(p));
        let mut prev_cost = self.compute_group_prev_cost(g);
        self.update_slots_from_group(false);

        let (front, back) = (self.assignment[pins[0]], self.assignment[pins[pins.len() - 1]]);
        let min_slot = front.min(back);
        let edge = self.slots[min_slot].edge;
        let mut min_count: i64 = 0;
        let mut slot = min_slot as i64 - 1;
        while slot >= 0 {
            let s = slot as usize;
            let mut free = self.available(s);
            if is_mirrored && free {
                let mut ms = 0;
                free = self.is_free_for_mirrored(s, &mut ms)?;
            }
            if !free || self.slots[s].edge != edge {
                break;
            }
            min_count += 1;
            slot -= 1;
        }
        let mut max_count: i64 = 0;
        let mut s = min_slot + 1;
        while s < self.slots.len() {
            let mut free = self.available(s);
            if is_mirrored && free {
                let mut ms = 0;
                free = self.is_free_for_mirrored(s, &mut ms)?;
            }
            if !free || self.slots[s].edge != edge {
                break;
            }
            max_count += 1;
            s += 1;
        }
        max_count = (max_count - pins.len() as i64).max(0);

        if min_count + max_count > 0 {
            if min_count > max_count {
                self.shift_group_to_position(&pins, min_count, min_slot, false)?;
            } else {
                self.shift_group_to_position(&pins, max_count, min_slot, true)?;
            }
        } else {
            prev_cost = MOVE_FAIL;
        }
        Ok(prev_cost)
    }

    /// `shiftGroupToPosition`: a draw in `[1, count]` slots away from the lowest slot.
    fn shift_group_to_position(&mut self, pins: &[usize], count: i64, min_slot: usize, move_to_max: bool) -> Result<(), Unsupported> {
        let move_count = self.gen.uniform_int(1, count);
        let mut new_slot = if move_to_max { min_slot as i64 + move_count } else { min_slot as i64 - move_count } as usize;
        self.update_group_slots(pins, &mut new_slot)
    }

    /// `rearrangeConstrainedGroups`.
    ///
    /// Rule: with several groups in the constraint, each group's extent (lowest, highest slot of
    /// its first and last pin) is collected in ascending group order and the extents sorted; the
    /// group order is shuffled ON THE SEARCH ENGINE, and the groups are laid back in that order
    /// starting at the lowest extent, keeping the gaps between consecutive sorted extents. The
    /// slots are NOT released first and no availability is checked. With one group, it shifts.
    fn rearrange_constrained_groups(&mut self, c: usize) -> Result<i64, Unsupported> {
        self.path.push('R');
        let mut prev_cost = 0;
        let groups: Vec<usize> = self.regions[c].groups.iter().copied().collect();
        if groups.len() > 1 {
            let mut aux: Vec<usize> = (0..groups.len()).collect();
            let mut limits: Vec<(usize, usize)> = Vec::new();
            for &g in &groups {
                prev_cost += self.compute_group_prev_cost(g);
                let pins = &self.groups[g].pins;
                let (a, b) = (self.assignment[pins[0]], self.assignment[pins[pins.len() - 1]]);
                limits.push((a.min(b), a.max(b)));
            }
            limits.sort();
            self.gen.shuffle(&mut aux);
            let mut cnt = 0;
            let mut new_slot = limits[0].0;
            for &idx in &aux {
                let g = groups[idx];
                self.sort_pins_from_group(g, self.slots[new_slot].edge);
                let pins = self.groups[g].pins.clone();
                self.update_group_slots(&pins, &mut new_slot)?;
                cnt += 1;
                if cnt < limits.len() {
                    new_slot = (new_slot as i64 + limits[cnt].0 as i64 - limits[cnt - 1].1 as i64 - 1) as usize;
                }
            }
        } else {
            for &g in &groups {
                prev_cost = self.shift_group(g)?;
            }
        }
        Ok(prev_cost)
    }

    /// `moveGroup`.
    ///
    /// Rule: an unconstrained group moves to free slots. A constrained one draws once: in a roomy
    /// constraint (pins per slot at most 0.5) at or below 0.7 it moves to free slots, else shifts;
    /// otherwise above 0.8 with more than one group in the constraint it rearranges, else shifts.
    fn move_group(&mut self, pin: usize) -> Result<i64, Unsupported> {
        self.path.push('G');
        let g = self.info[pin].group.expect("a grouped pin");
        let Some(c) = self.info[pin].constraint else {
            return self.move_group_to_free_slots(g);
        };
        if self.regions[c].pins_per_slots <= PINS_PER_SLOT_LIMIT {
            let mv = self.gen.uniform_real();
            if mv <= GROUP_TO_FREE_SLOTS {
                return self.move_group_to_free_slots(g);
            }
            return self.shift_group(g);
        }
        let mv = self.gen.uniform_real();
        if mv > SHIFT_GROUP && self.regions[c].groups.len() > 1 {
            return self.rearrange_constrained_groups(c);
        }
        self.shift_group(g)
    }

    /// `restorePreviousAssignment`: each touched pin back to the slot recorded beside it.
    fn restore_previous_assignment(&mut self) {
        if !self.prev_slots.is_empty() {
            for (cnt, &pin) in self.moved.iter().enumerate() {
                self.assignment[pin] = self.prev_slots[cnt];
            }
        }
    }

    /// `isFreeForGroup`: walks `slot_idx` forward to the first start whose `size` slots are all
    /// free, giving up once `slot_idx + size` reaches `last_slot`.
    ///
    /// Rule: the reflection checked is that of the START slot only, for every slot of the run, and
    /// it is looked up whether or not the group is mirrored — so a start with no reflection stops
    /// the run with the reference's PPL-112.
    fn is_free_for_group(&self, slot_idx: &mut usize, size: usize, last_slot: usize) -> Result<bool, Unsupported> {
        loop {
            if *slot_idx + size >= last_slot {
                return Ok(false);
            }
            let mirrored = self.get_mirrored_slot_idx(*slot_idx)?;
            let mut free = true;
            for slot in *slot_idx..*slot_idx + size {
                if !self.available(slot) || !self.available(mirrored) {
                    free = false;
                    break;
                }
            }
            if free {
                return Ok(true);
            }
            *slot_idx += 1;
        }
    }

    /// `getSlotsRange`: the pin's constraint's range; for a mirrored pin with no constraint of its
    /// own, the reflection of its partner's range (ends swapped if reversed); else every slot.
    fn get_slots_range(&self, pin: usize) -> (usize, usize) {
        let (mut first, mut last) = (0, self.slots.len() - 1);
        if let Some(c) = self.info[pin].constraint {
            return (self.regions[c].first_slot, self.regions[c].last_slot);
        }
        if let Some(m) = self.info[pin].mirror {
            if let Some(c) = self.info[m].constraint {
                let r = &self.regions[c];
                self.get_mirrored_slot_range(r.first_slot, r.last_slot, &mut first, &mut last);
                if first > last {
                    std::mem::swap(&mut first, &mut last);
                }
            }
        }
        (first, last)
    }

    /// `isFreeForMirrored`: the slot and its reflection both free.
    fn is_free_for_mirrored(&self, slot: usize, mirrored: &mut usize) -> Result<bool, Unsupported> {
        *mirrored = self.get_mirrored_slot_idx(slot)?;
        Ok(self.available(slot) && self.available(*mirrored))
    }

    /// `getMirroredSlotIdx`: the reflection's slot, or the reference's PPL-112.
    fn get_mirrored_slot_idx(&self, slot: usize) -> Result<usize, Unsupported> {
        self.mirror_slot[slot].ok_or_else(|| {
            let s = &self.slots[slot];
            Unsupported::Error(112, format!("Mirrored position of ({}, {}) at layer {} is not a valid position.", s.x, s.y, s.layer))
        })
    }

    /// `getMirroredSlotRange`: approximate the reflected range's ends by position.
    ///
    /// Rule: with the range on bottom or right, the first end is the LAST slot (scanning down)
    /// at or past the first end's reflection along the edge and level with it, the second end the
    /// FIRST slot (scanning up) at or before the second's; on top or left the scans run the other
    /// way. Only positions are compared — any edge, any layer — and an end with no match keeps the
    /// value it came in with.
    fn get_mirrored_slot_range(&self, s1: usize, s2: usize, m1: &mut usize, m2: &mut usize) {
        let (a, b) = (&self.slots[s1], &self.slots[s2]);
        let edge = a.edge;
        let die = self.die;
        let p1 = crate::sections::mirrored_position((a.x, a.y), die);
        let p2 = crate::sections::mirrored_position((b.x, b.y), die);
        let at_or_past = |s: &Slot, p: (i32, i32)| match edge {
            Edge::Bottom | Edge::Top => s.x >= p.0 && s.y == p.1,
            _ => s.x == p.0 && s.y >= p.1,
        };
        let at_or_before = |s: &Slot, p: (i32, i32)| match edge {
            Edge::Bottom | Edge::Top => s.x <= p.0 && s.y == p.1,
            _ => s.x == p.0 && s.y <= p.1,
        };
        let n = self.slots.len();
        if matches!(edge, Edge::Bottom | Edge::Right) {
            if let Some(i) = (0..n).rev().find(|&i| at_or_past(&self.slots[i], p1)) {
                *m1 = i;
            }
            if let Some(i) = (0..n).find(|&i| at_or_before(&self.slots[i], p2)) {
                *m2 = i;
            }
        } else if matches!(edge, Edge::Top | Edge::Left) {
            if let Some(i) = (0..n).find(|&i| at_or_past(&self.slots[i], p1)) {
                *m1 = i;
            }
            if let Some(i) = (0..n).rev().find(|&i| at_or_before(&self.slots[i], p2)) {
                *m2 = i;
            }
        }
    }

    /// `updateSlotsFromGroup`: every recorded previous slot marked `used = block`.
    fn update_slots_from_group(&mut self, block: bool) {
        for i in 0..self.prev_slots.len() {
            self.used[self.prev_slots[i]] = block;
        }
    }

    /// `computeGroupPrevCost`: the group's cost (and its partners'), each pin and partner recorded
    /// with its slot, in the group's current order.
    fn compute_group_prev_cost(&mut self, g: usize) -> i64 {
        let mut prev_cost = self.get_group_cost(g);
        for k in 0..self.groups[g].pins.len() {
            let p = self.groups[g].pins[k];
            self.prev_slots.push(self.assignment[p]);
            self.moved.push(p);
            if let Some(m) = self.info[p].mirror {
                self.prev_slots.push(self.assignment[m]);
                self.moved.push(m);
                prev_cost += self.get_pin_cost(m);
            }
        }
        prev_cost
    }

    /// `updateGroupSlots`: the pins on consecutive slots from `new_slot`, partners reflected;
    /// `new_slot` is left one past the last.
    fn update_group_slots(&mut self, pins: &[usize], new_slot: &mut usize) -> Result<(), Unsupported> {
        for &p in pins {
            self.assignment[p] = *new_slot;
            self.new_slots.push(*new_slot);
            if let Some(m) = self.info[p].mirror {
                let ms = self.get_mirrored_slot_idx(*new_slot)?;
                self.assignment[m] = ms;
                self.new_slots.push(ms);
            }
            *new_slot += 1;
        }
        Ok(())
    }

    /// `sortPinsFromGroup`: an ORDERED group landing on top or left is reversed in place.
    fn sort_pins_from_group(&mut self, g: usize, edge: Edge) {
        if self.groups[g].ordered && matches!(edge, Edge::Top | Edge::Left) {
            self.groups[g].pins.reverse();
        }
    }

    /// `countLonePins`.
    ///
    /// Rule: grouped pins count once each and twice when mirrored (the partner rides along);
    /// swappable pins are those neither grouped nor mirrored, and of those the unconstrained.
    fn count_lone_pins(&mut self) {
        let mut in_groups = 0;
        for group in &self.groups {
            for &p in &group.pins {
                in_groups += 1 + self.mirrored(p) as usize;
            }
        }
        let mut swappable = 0;
        let mut unconstrained = 0;
        for p in 0..self.pins.len() {
            if !self.in_group(p) && !self.mirrored(p) {
                swappable += 1;
                if self.info[p].constraint.is_none() {
                    unconstrained += 1;
                }
            }
        }
        self.lone_pins = self.pins.len().saturating_sub(in_groups);
        self.swappable_pins = swappable;
        self.unconstrained_swappable_pins = unconstrained;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sections::net_hpwl;
    use crate::slots::Edge;

    fn slots(n: usize) -> Vec<Slot> {
        (0..n)
            .map(|i| Slot {
                x: i as i32 * 100,
                y: 0,
                layer: "met1".into(),
                edge: Edge::Bottom,
                blocked: false,
            })
            .collect()
    }

    fn pin(name: &str, sinks: &[(i32, i32)]) -> Pin {
        Pin { name: name.into(), sinks: sinks.to_vec(), mirror: None, initial: None }
    }

    const DIE: (i32, i32, i32, i32) = (0, 0, 10_000, 10_000);

    #[test]
    fn the_default_effort_scales_with_the_problem() {
        assert_eq!(perturbations_for(54, 0), 43, "54 * 0.8");
        assert_eq!(perturbations_for(0, 3), 30, "groups count for more");
        assert_eq!(perturbations_for(0, 0), 0);
    }

    #[test]
    fn a_run_places_every_pin_in_a_distinct_slot() {
        let s = slots(40);
        let pins: Vec<Pin> =
            (0..12).map(|i| pin(&format!("p{i}"), &[(i * 300, 900)])).collect();
        let cfg = Anneal { max_iterations: 50, ..Default::default() };
        let placed = run(&s, &pins, DIE, &cfg).unwrap();

        assert_eq!(placed.len(), 12);
        let mut used: Vec<usize> = placed.iter().map(|p| p.slot).collect();
        used.sort_unstable();
        used.dedup();
        assert_eq!(used.len(), 12, "a slot was handed out twice");
        assert!(placed.iter().all(|p| !s[p.slot].blocked));
    }

    #[test]
    fn it_actually_improves_on_where_it_started() {
        // The whole point: the search has to do better than its random opening. Compared against
        // the same starting assignment the run itself begins from.
        let s = slots(60);
        let pins: Vec<Pin> =
            (0..20).map(|i| pin(&format!("p{i}"), &[(i * 290, 700)])).collect();
        let cfg = Anneal { max_iterations: 300, ..Default::default() };

        let mut start = State::new(&s, &pins, &[], &[], DIE, &cfg);
        start.random_assignment().unwrap();
        let before = start.get_assignment_cost();

        let placed = run(&s, &pins, DIE, &cfg).unwrap();
        let after: i64 = placed
            .iter()
            .map(|p| net_hpwl(&pins[p.pin].sinks, (s[p.slot].x, s[p.slot].y)))
            .sum();
        assert!(after < before, "annealing did not improve: {before} -> {after}");
    }

    #[test]
    fn the_same_seed_gives_the_same_placement_and_a_different_one_does_not() {
        // Reproducibility is the property the whole RNG effort exists to buy.
        let s = slots(40);
        let pins: Vec<Pin> =
            (0..15).map(|i| pin(&format!("p{i}"), &[(i * 250, 800)])).collect();
        let cfg = Anneal { max_iterations: 40, ..Default::default() };

        let a = run(&s, &pins, DIE, &cfg).unwrap();
        let b = run(&s, &pins, DIE, &cfg).unwrap();
        assert_eq!(a, b, "the same seed must give the same answer");

        let other = Anneal { seed: 7, ..cfg.clone() };
        let c = run(&s, &pins, DIE, &other).unwrap();
        assert_ne!(a, c, "a different seed should explore differently");
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn a_blocked_slot_is_never_used() {
        let mut s = slots(30);
        for i in 0..20 {
            s[i].blocked = true;
        }
        let pins: Vec<Pin> = (0..8).map(|i| pin(&format!("p{i}"), &[])).collect();
        let cfg = Anneal { max_iterations: 30, ..Default::default() };
        let placed = run(&s, &pins, DIE, &cfg).unwrap();
        assert!(placed.iter().all(|p| !s[p.slot].blocked), "a pin landed on a blocked slot");
    }

    #[test]
    fn cooling_makes_the_search_settle() {
        // A high temperature accepts almost anything; a low one accepts almost nothing worse. The
        // cold run should end up no worse than the hot one on a problem this small.
        let s = slots(50);
        let pins: Vec<Pin> =
            (0..16).map(|i| pin(&format!("p{i}"), &[(i * 300, 600)])).collect();
        let cost = |cfg: &Anneal| -> i64 {
            run(&s, &pins, DIE, cfg)
                .unwrap()
                .iter()
                .map(|p| net_hpwl(&pins[p.pin].sinks, (s[p.slot].x, s[p.slot].y)))
                .sum()
        };
        let hot = cost(&Anneal { init_temperature: 1e9, alpha: 1.0, max_iterations: 200, ..Default::default() });
        let cold = cost(&Anneal { max_iterations: 200, ..Default::default() });
        assert!(cold <= hot, "cooling did not help: cold {cold} vs hot {hot}");
    }

    // ── The reference's rules, one test each. The rule is stated as the reference has it; the
    // whole sequence is witnessed against the instrumented reference by
    // `correlation/instruments/ppl/ppl-anneal-trace.py` (31 of 31 cases identical, 2026-11-05).

    /// `n` slots on the bottom edge (x = 100 + 100·i) then `n` on the top, descending — each
    /// bottom slot's reflection is a top slot. None sits at x = 0: a corner position reflects
    /// across the LEFT edge first (`getMirroredPosition` tests x before y) and has no slot.
    fn two_edges(n: usize) -> Vec<Slot> {
        let mut v: Vec<Slot> = (0..n)
            .map(|i| Slot { x: 100 + i as i32 * 100, y: 0, layer: "m".into(), edge: Edge::Bottom, blocked: false })
            .collect();
        v.extend((0..n).rev().map(|i| Slot { x: 100 + i as i32 * 100, y: 10_000, layer: "m".into(), edge: Edge::Top, blocked: false }));
        v
    }

    fn pins(n: usize) -> Vec<Pin> {
        (0..n).map(|i| pin(&format!("p{i}"), &[(i as i32 * 37, 500)])).collect()
    }

    fn state<'a>(s: &'a [Slot], p: &'a [Pin], g: &[Group], r: &'a [Region]) -> State<'a> {
        let mut st = State::new(s, p, g, r, DIE, &Anneal::default());
        st.trace = false;
        st
    }

    fn region(pins: Vec<usize>, groups: &[usize], first: usize, last: usize, pps: f32) -> Region {
        Region { pins, groups: groups.iter().copied().collect(), first_slot: first, last_slot: last, pins_per_slots: pps }
    }

    #[test]
    fn an_unconstrained_group_takes_the_shuffled_entry_not_the_start_the_walk_found() {
        // Rule (randomAssignmentForGroups): isFreeForGroup walks its start forward to a free run,
        // but an unconstrained group is placed at slot_indices[slot_idx] — the shuffled entry,
        // even when its own run is taken.
        let mut s = two_edges(30);
        let mut order: Vec<usize> = (0..s.len()).collect();
        Mt19937::new(42).shuffle(&mut order);
        let start = order[0];
        s[start].blocked = true;
        let p = pins(3);
        let g = vec![Group { pins: vec![0, 1, 2], ordered: false }];
        let mut st = state(&s, &p, &g, &[]);
        if start + 4 >= s.len() - 1 {
            return; // this seed's first entry has no room to walk; the rule is not observable
        }
        let mut placed = BTreeSet::new();
        let next = st.random_assignment_for_groups(&mut placed, &order).unwrap();
        assert_eq!(next, 1, "every group advances the shuffled index once");
        assert_eq!(st.assignment[..3], [start, start + 1, start + 2]);
    }

    #[test]
    fn a_group_run_checks_only_its_starts_reflection() {
        // Rule (isFreeForGroup): the reflection looked up is the START slot's, and it is checked
        // for every slot of the run; the walk gives up once start + size reaches last_slot.
        let s = two_edges(10);
        let p = pins(1);
        let mut st = state(&s, &p, &[], &[]);
        let refl = |i: usize| st.mirror_slot[i].unwrap();
        let (r0, r1) = (refl(0), refl(1));
        st.used[r1] = true; // the reflection of slot 1 — not looked at for a run starting at 0
        let mut at = 0;
        assert!(st.is_free_for_group(&mut at, 3, 19).unwrap());
        assert_eq!(at, 0);
        st.used[r0] = true; // the start's own reflection: the run moves on
        let mut at = 0;
        assert!(st.is_free_for_group(&mut at, 3, 19).unwrap());
        assert_eq!(at, 2, "slot 1's run is refused through its own reflection");
        let mut at = 16;
        assert!(!st.is_free_for_group(&mut at, 3, 19).unwrap(), "16 + 3 reaches the last slot");
    }

    #[test]
    fn an_ordered_group_is_reversed_each_time_it_lands_on_top_or_left() {
        // Rule (Netlist::sortPinsFromGroup): reversed IN PLACE, so two landings undo each other;
        // an unordered group, or a landing on bottom/right, is left alone.
        let s = two_edges(4);
        let p = pins(4);
        let g = vec![Group { pins: vec![0, 1], ordered: true }, Group { pins: vec![2, 3], ordered: false }];
        let mut st = state(&s, &p, &g, &[]);
        st.sort_pins_from_group(0, Edge::Top);
        assert_eq!(st.groups[0].pins, vec![1, 0]);
        st.sort_pins_from_group(0, Edge::Left);
        assert_eq!(st.groups[0].pins, vec![0, 1]);
        st.sort_pins_from_group(0, Edge::Bottom);
        st.sort_pins_from_group(1, Edge::Top);
        assert_eq!((st.groups[0].pins.clone(), st.groups[1].pins.clone()), (vec![0, 1], vec![2, 3]));
    }

    #[test]
    fn a_shift_with_no_room_fails_leaving_its_records_and_its_slots_released() {
        // Rule (shiftGroup): with no free same-edge slot either side it returns the failure
        // without re-marking the slots it released or clearing pins_/prev_slots_.
        let s = two_edges(6);
        let p = pins(2);
        let g = vec![Group { pins: vec![0, 1], ordered: false }];
        let mut st = state(&s, &p, &g, &[]);
        st.assignment = vec![2, 3];
        for i in 0..s.len() {
            st.used[i] = true;
        }
        assert_eq!(st.shift_group(0).unwrap(), MOVE_FAIL);
        assert_eq!((st.moved.clone(), st.prev_slots.clone()), (vec![0, 1], vec![2, 3]));
        assert!(!st.used[2] && !st.used[3], "released and not re-marked");
    }

    #[test]
    fn a_shift_counts_upward_through_its_own_slots_less_its_size() {
        // Rule (shiftGroup): the upward run starts beside the lowest slot — through the group's
        // own released slots — and the group's size is taken off; downward stops at a taken
        // slot. Here down = 1, up = 6 - 3 = 3, so the group moves up by a draw in [1, 3].
        let s = two_edges(20);
        let p = pins(3);
        let g = vec![Group { pins: vec![0, 1, 2], ordered: false }];
        for seed in 0..40 {
            let mut st = state(&s, &p, &g, &[]);
            st.gen = Mt19937::new(seed);
            st.assignment = vec![5, 6, 7];
            for i in [3, 5, 6, 7, 12] {
                st.used[i] = true;
            }
            st.shift_group(0).unwrap();
            let start = st.assignment[0];
            assert!((6..=8).contains(&start), "seed {seed}: start {start}");
        }
    }

    #[test]
    fn a_roomy_constraints_group_moves_to_free_slots_at_or_below_the_split() {
        // Rule (moveGroup): constrained, at most 0.5 pins per slot: one draw, <= 0.7 moves the
        // group to free slots, above shifts; tighter: > 0.8 with company rearranges, else shifts;
        // unconstrained: always to free slots (no draw).
        let s = two_edges(20);
        let p = pins(4);
        let g = vec![Group { pins: vec![0, 1], ordered: false }, Group { pins: vec![2, 3], ordered: false }];
        let roomy = [region(vec![0, 1, 2, 3], &[0, 1], 0, 19, 0.2)];
        let tight = [region(vec![0, 1, 2, 3], &[0, 1], 0, 19, 0.9)];
        for seed in 0..30 {
            for (r, roomy_case) in [(&roomy, true), (&tight, false)] {
                let mut st = state(&s, &p, &g, r);
                st.gen = Mt19937::new(seed);
                st.assignment = vec![2, 3, 10, 11];
                for i in [2, 3, 10, 11] {
                    st.used[i] = true;
                }
                let draw = st.gen.clone().uniform_real();
                let _ = st.move_group(0).unwrap();
                let want = if roomy_case {
                    if draw <= GROUP_TO_FREE_SLOTS { "GF" } else { "GH" }
                } else if draw > SHIFT_GROUP {
                    "GR"
                } else {
                    "GH"
                };
                assert_eq!(st.path, want, "seed {seed} draw {draw}");
            }
        }
        let mut st = state(&s, &p, &g, &[]);
        st.assignment = vec![2, 3, 10, 11];
        let before = st.gen.draws();
        let _ = st.move_group(0);
        assert!(st.path.starts_with("GF"));
        assert!(st.gen.draws() > before);
    }

    #[test]
    fn a_group_moved_to_free_slots_never_straddles_two_edges() {
        // Rule (moveGroupToFreeSlots): every slot of the run must be free AND on the start slot's
        // edge; a start whose run turns a corner is rejected and drawn again.
        let s = two_edges(5); // 0..5 bottom, 5..10 top
        let p = pins(3);
        let g = vec![Group { pins: vec![0, 1, 2], ordered: false }];
        for seed in 0..40 {
            let mut st = state(&s, &p, &g, &[]);
            st.gen = Mt19937::new(seed);
            st.assignment = vec![0, 1, 2];
            for i in [0, 1, 2, 8, 9] {
                st.used[i] = true;
            }
            if st.move_group_to_free_slots(0).unwrap() == MOVE_FAIL {
                continue;
            }
            let edges: BTreeSet<u8> = st.assignment.iter().map(|&k| s[k].edge as u8).collect();
            assert_eq!(edges.len(), 1, "seed {seed}: group at {:?}", st.assignment);
        }
    }

    #[test]
    fn a_rearrangement_lays_the_groups_back_keeping_the_gaps_between_extents() {
        // Rule (rearrangeConstrainedGroups): extents sorted, groups laid back in shuffled order
        // from the lowest extent, each next start = end + the gap between consecutive extents.
        let s = two_edges(20);
        let p = pins(5);
        let g = vec![Group { pins: vec![0, 1, 2], ordered: false }, Group { pins: vec![3, 4], ordered: false }];
        let r = [region(vec![0, 1, 2, 3, 4], &[0, 1], 0, 19, 0.9)];
        for seed in 0..10 {
            let mut st = state(&s, &p, &g, &r);
            st.gen = Mt19937::new(seed);
            st.assignment = vec![10, 11, 12, 2, 3];
            st.rearrange_constrained_groups(0).unwrap();
            let (a, b) = (st.assignment[0], st.assignment[3]);
            // gap = 10 - 3 - 1 = 6: either [0..3) at 2 then [3..5) at 2+3+6 = 11, or [3..5) at 2
            // then [0..3) at 2+2+6 = 10.
            assert!((a, b) == (2, 11) || (a, b) == (10, 2), "seed {seed}: {a} {b}");
            assert_eq!(st.moved.len(), 5);
        }
    }

    #[test]
    fn grouped_pins_count_twice_when_mirrored_and_only_lone_ones_swap() {
        // Rule (countLonePins): a grouped pin counts once, twice when mirrored; swappable means
        // neither grouped nor mirrored; unconstrained swappable additionally has no constraint.
        let s = two_edges(10);
        let mut p = pins(6);
        p[0].mirror = Some(5);
        let g = vec![Group { pins: vec![0, 1], ordered: false }];
        let r = [region(vec![2], &[], 0, 9, 0.1)];
        let st = state(&s, &p, &g, &r);
        assert_eq!((st.lone_pins, st.swappable_pins, st.unconstrained_swappable_pins), (3, 3, 2));
        assert_eq!(st.perturb_per_iter, perturbations_for(3, 1));
    }

    #[test]
    fn a_constrained_swap_takes_its_partner_from_its_own_constraint() {
        // Rule (swapPins): a constrained pin1's partner is drawn from that constraint's pin list;
        // with no eligible partner the swap fails before drawing it.
        let s = two_edges(10);
        let p = pins(4);
        let r = [region(vec![0, 1], &[], 0, 9, 0.1), region(vec![2], &[], 0, 9, 0.1)];
        for seed in 0..20 {
            let mut st = state(&s, &p, &[], &r);
            st.gen = Mt19937::new(seed);
            st.assignment = vec![0, 1, 2, 3];
            let pin1 = st.gen.clone().uniform_int(0, 3) as usize;
            let got = st.swap_pins();
            match pin1 {
                0 | 1 => assert_eq!(st.moved, vec![pin1, 1 - pin1], "seed {seed}"),
                2 => assert_eq!(got, MOVE_FAIL, "seed {seed}: pin 2 is alone in its constraint"),
                _ => assert_eq!(got, MOVE_FAIL, "seed {seed}: one unconstrained swappable pin"),
            }
        }
    }

    #[test]
    fn a_grouped_pin_or_the_partner_of_one_moves_its_group() {
        // Rule (movePinToFreeSlot): not the lone retry — a grouped pin, or a mirrored pin whose
        // partner is grouped, delegates to moveGroup.
        let s = two_edges(10);
        let mut p = pins(3);
        p[0].mirror = Some(2);
        let g = vec![Group { pins: vec![0, 1], ordered: false }];
        for seed in 0..20 {
            let mut st = state(&s, &p, &g, &[]);
            st.gen = Mt19937::new(seed);
            st.assignment = vec![2, 3, st.mirror_slot[2].unwrap()];
            let pin = st.gen.clone().uniform_int(0, 2);
            let _ = st.move_pin_to_free_slot(false).unwrap();
            assert!(st.path.starts_with("MG"), "seed {seed}: pin {pin} path {}", st.path);
        }
    }

    #[test]
    fn a_mirrored_partners_range_is_its_partners_reflected() {
        // Rule (getSlotsRange/getMirroredSlotRange): a mirrored pin with no constraint of its own
        // uses its partner's range reflected — on bottom, the first end is the LAST slot at or
        // past the reflection of the first, the second the FIRST at or before the second's —
        // with the ends swapped when reversed.
        let s = two_edges(10); // bottom 0..10 (x ascending), top 10..20 (x descending)
        let mut p = pins(2);
        p[0].mirror = Some(1);
        let r = [region(vec![0], &[], 2, 5, 0.1)];
        let st = state(&s, &p, &[], &r);
        assert_eq!(st.get_slots_range(0), (2, 5));
        // bottom x 300..600 reflect to top x 300..600: top index of x is 10 + (9 - (x - 100)/100).
        assert_eq!(st.get_slots_range(1), (14, 17));
    }

    #[test]
    fn a_constraint_range_is_half_open_at_its_far_end_in_list_order() {
        // Rule (createSectionsPerConstraint): bottom begins at along >= begin and stops at the
        // first along >= end; top (descending) begins at along <= end and stops at along <= begin.
        let s = two_edges(10);
        let bottom = Constraint { interval: crate::slots::Interval { edge: Edge::Bottom, begin: 300, end: 600 }, pins: vec![0] };
        assert_eq!(constraint_slot_range(&s, &bottom), Some((2, 4)));
        let top = Constraint { interval: crate::slots::Interval { edge: Edge::Top, begin: 300, end: 600 }, pins: vec![0] };
        // top indices 10.. hold x 1000, 900, ...: x <= 600 first at index 14, stop at x <= 300 (17).
        assert_eq!(constraint_slot_range(&s, &top), Some((14, 16)));
    }

    #[test]
    fn an_annealing_run_keeps_the_constraints_in_the_order_read() {
        // Rule (initConstraints(true)): sortConstraints is skipped when annealing.
        let s = two_edges(10);
        let c = |b: i32, e: i32, p: usize| Constraint { interval: crate::slots::Interval { edge: Edge::Bottom, begin: b, end: e }, pins: vec![p] };
        let r = regions(&s, &[c(0, 900, 0), c(100, 300, 1)], &[], 200);
        assert_eq!(r.iter().map(|x| x.pins[0]).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(r[1].pins_per_slots, 0.5, "1 pin over 2 usable slots, in f32");
    }

    #[test]
    fn a_constrained_mirrored_pin_places_its_partner_at_its_reflection() {
        // Rule (randomAssignment): constrained pins first, drawn in their range on the search
        // engine; a mirrored one also needs its reflection free, its partner takes it and is
        // recorded as placed (so the second pass skips it).
        let s = two_edges(10);
        let mut p = pins(3);
        p[0].mirror = Some(1);
        let r = [region(vec![0], &[], 0, 9, 0.1)];
        let mut st = state(&s, &p, &[], &r);
        st.random_assignment().unwrap();
        let a = st.assignment.clone();
        assert!(a[0] <= 9);
        assert_eq!(a[1], st.mirror_slot[a[0]].unwrap());
        assert!(st.gen.draws() >= 1, "the constrained draw is on the search engine");
        assert!(a[2] != a[0] && a[2] != a[1]);
    }

    #[test]
    fn nothing_to_place_or_nowhere_to_put_it_is_refused_not_guessed() {
        let s = slots(4);
        let many: Vec<Pin> = (0..9).map(|i| pin(&format!("p{i}"), &[])).collect();
        assert_eq!(run(&s, &many, DIE, &Anneal::default()), Err(Unsupported::Empty));
        assert_eq!(run(&s, &[], DIE, &Anneal::default()), Err(Unsupported::Empty));
    }
}
