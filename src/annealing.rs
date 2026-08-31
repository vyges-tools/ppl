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
//! Plain pins only — no groups, no constraints, no mirrored pairs. [`run`] refuses instead.
//!
//! ⚠️ **This scope note used to say all three "add four more move types". That is FALSE for
//! constraints, and was corrected on 2026-08-31 by reading the reference.** Only a GROUP reaches
//! the extra moves: `movePinToFreeSlot` delegates to `moveGroup` — and thence to `shiftGroup`,
//! `moveGroupToFreeSlots`, `rearrangeConstrainedGroups` — **only when `io_pin.isInGroup()`**.
//!
//! A constraint changes one thing: `getSlotsRange` sets `first_slot`/`last_slot` from
//! `constraints_[idx]`, and the very same `uniform_int_distribution` is then constructed over that
//! narrower range. Same move, same one draw per attempt, different bounds. It is a bounding rule,
//! not a move type, and the desynchronisation argument does not apply to it — an unconstrained run
//! already draws from `0..num_slots-1` through that same distribution.
//!
//! ⟹ **The refusal is therefore OVER-BROAD**, and measurably so: 6 of the 29 upstream annealing
//! cases (`annealing_constraint1..5`, `8`) carry constraints with no groups and no mirroring, and
//! we refuse all six. Kept for now because honouring a constraint means carrying a per-pin slot
//! range through the move generators, which is real work — but kept with the true reason, not a
//! borrowed one. See `docs/openroad/ppl/ppl-audit.md` finding 1.
//!
//! Nothing here touches a database.

use crate::rng::Mt19937;
use crate::sections::{net_hpwl, Pin, Placement};
use crate::slots::Slot;

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
/// A move that could not be made. Distinct from a zero-cost move, which is a perfectly good one.
const MOVE_FAIL: i64 = -1;

/// How many perturbations one temperature step gets, when the caller does not say.
///
/// Proportional to the number of pins free to move, so a bigger problem gets a longer search at
/// each temperature rather than the same fixed effort.
pub fn perturbations_for(lone_pins: usize, groups: usize) -> i32 {
    (lone_pins as f64 * 0.8 + groups as f64 * 10.0) as i32
}

/// Why a design cannot be annealed by this implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// A pin group or a mirrored pair — each does bring its own moves or slot-pairing, and
    /// implementing some of them would put the random stream out of step with the reference,
    /// which is worse than not running at all because the result would look plausible.
    ///
    /// ⚠️ **A constraint does NOT belong in that list** — it bounds the slot range and adds no
    /// move. The variant is still returned for constrained designs because the range is not
    /// carried through the move generators yet; that is a gap, not an impossibility.
    NeedsExtraMoves,
    /// Nothing to place, or nowhere to put it.
    Empty,
}

/// One annealing run.
///
/// Returns the final assignment, or why it was refused.
pub fn run(
    slots: &[Slot],
    pins: &[Pin],
    die: (i32, i32, i32, i32),
    cfg: &Anneal,
) -> Result<Vec<Placement>, Unsupported> {
    let _ = die;
    if pins.iter().any(|p| p.mirror.is_some()) {
        return Err(Unsupported::NeedsExtraMoves);
    }
    let usable: Vec<usize> = (0..slots.len()).filter(|&i| !slots[i].blocked).collect();
    if pins.is_empty() || usable.len() < pins.len() {
        return Err(Unsupported::Empty);
    }

    let mut state = State::new(slots, pins, cfg);
    state.random_assignment();
    state.anneal(cfg);
    Ok(state.assignment())
}

struct State<'a> {
    slots: &'a [Slot],
    pins: &'a [Pin],
    /// Which slot each pin currently holds.
    assignment: Vec<usize>,
    used: Vec<bool>,
    /// The generator the *search* draws from. Separate from the one that seeds the starting
    /// assignment: the reference uses two engines, and sharing one would consume the other's draws.
    gen: Mt19937,
    /// The seed both engines start from — the search's, and the separate one that shuffles the
    /// opening assignment.
    seed: u32,
    /// Scratch for one perturbation: which pins moved, what they held, what they took.
    moved: Vec<usize>,
    prev_slots: Vec<usize>,
    new_slots: Vec<usize>,
}

impl<'a> State<'a> {
    fn new(slots: &'a [Slot], pins: &'a [Pin], cfg: &Anneal) -> State<'a> {
        State {
            slots,
            pins,
            assignment: vec![0; pins.len()],
            used: vec![false; slots.len()],
            gen: Mt19937::new(cfg.seed),
            seed: cfg.seed,
            moved: Vec::new(),
            prev_slots: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    fn pin_cost(&self, pin: usize) -> i64 {
        let s = &self.slots[self.assignment[pin]];
        net_hpwl(&self.pins[pin].sinks, (s.x, s.y))
    }

    fn total_cost(&self) -> i64 {
        (0..self.pins.len()).map(|p| self.pin_cost(p)).sum()
    }

    fn available(&self, slot: usize) -> bool {
        !self.used[slot] && !self.slots[slot].blocked
    }

    /// **A1** — the starting assignment: a shuffled slot order, handed out in pin order.
    ///
    /// ⚠️ **A second, independent engine.** The reference shuffles with an engine of its own,
    /// seeded the same way as the search's; reusing one generator would consume draws the search
    /// expects and desynchronise everything after.
    fn random_assignment(&mut self) {
        let mut order: Vec<usize> = (0..self.slots.len()).collect();
        let mut shuffler = Mt19937::new(self.seed);
        shuffler.shuffle(&mut order);

        let mut at = 0usize;
        for pin in 0..self.pins.len() {
            let mut slot = order[at];
            // Walk forward past anything taken or blocked. The walk does not draw, so it costs
            // nothing in the stream.
            while !self.available(slot) && at < order.len() - 1 {
                at += 1;
                slot = order[at];
            }
            self.assignment[pin] = slot;
            self.used[slot] = true;
            at += 1;
        }
    }

    /// **A2** — one perturbation: either swap two pins, or move one to a free slot.
    ///
    /// The choice is a draw, and it is taken **before** either move is attempted — so it costs a
    /// draw whichever way it goes. If the swap fails, the move is tried on the *same* iteration,
    /// consuming its own draws on top.
    fn perturb(&mut self, swappable: usize) -> i64 {
        let choice = self.gen.uniform_real();
        let mut prev_cost = MOVE_FAIL;

        if choice < SWAP_PINS && swappable > 1 {
            prev_cost = self.swap_pins(swappable);
        }
        if choice >= SWAP_PINS || swappable <= 1 || prev_cost == MOVE_FAIL {
            prev_cost = self.move_pin();
        }
        prev_cost
    }

    /// **A3** — exchange the slots of two pins.
    ///
    /// A swap changes no slot's *occupancy*, only who occupies it — which is why it records no new
    /// slots, and why accepting one leaves the used-slot bookkeeping alone.
    fn swap_pins(&mut self, swappable: usize) -> i64 {
        let n = self.pins.len() as i64;
        let pin1 = self.gen.uniform_int(0, n - 1) as usize;
        if swappable < 2 {
            return MOVE_FAIL;
        }
        let mut pin2 = self.gen.uniform_int(0, n - 1) as usize;
        while pin1 == pin2 {
            pin2 = self.gen.uniform_int(0, n - 1) as usize;
        }

        self.moved.push(pin1);
        self.moved.push(pin2);
        self.prev_slots.push(self.assignment[pin1]);
        self.prev_slots.push(self.assignment[pin2]);

        let prev_cost = self.pin_cost(pin1) + self.pin_cost(pin2);
        self.assignment.swap(pin1, pin2);
        prev_cost
    }

    /// **A4** — move one pin to a free slot, chosen by repeated draws.
    ///
    /// ⚠️ **Every rejected candidate costs a draw.** The search gives up after `10 × slots`
    /// attempts; both the attempt limit and the fact that a failed attempt still consumes a draw
    /// are part of the algorithm, because they decide where the stream is when the next
    /// perturbation starts.
    fn move_pin(&mut self) -> i64 {
        let n = self.pins.len() as i64;
        let pin = self.gen.uniform_int(0, n - 1) as usize;

        self.moved.push(pin);
        let prev_slot = self.assignment[pin];
        self.prev_slots.push(prev_slot);
        let prev_cost = self.pin_cost(pin);

        let last = self.slots.len() as i64 - 1;
        let max_attempts = self.slots.len() * 10;
        let mut attempts = 0;
        let mut chosen = None;
        while chosen.is_none() && attempts < max_attempts {
            let candidate = self.gen.uniform_int(0, last) as usize;
            if self.available(candidate) && candidate != prev_slot {
                chosen = Some(candidate);
            }
            attempts += 1;
        }
        let Some(new_slot) = chosen else {
            // Nothing was changed, so nothing is recorded — a failed move must leave no trace for
            // the accept/reject step to undo.
            self.prev_slots.clear();
            self.moved.clear();
            return MOVE_FAIL;
        };

        self.new_slots.push(new_slot);
        self.assignment[pin] = new_slot;
        prev_cost
    }

    /// **A5** — the cooling loop.
    ///
    /// A change that helps is always kept. One that hurts is kept with probability
    /// `exp(-Δ / temperature)`, which starts near 1 and falls as the temperature does — so the
    /// search explores early and settles late.
    fn anneal(&mut self, cfg: &Anneal) {
        let swappable = self.pins.len();
        let perturb_per_iter = if cfg.perturb_per_iter != 0 {
            cfg.perturb_per_iter
        } else {
            perturbations_for(self.pins.len(), 0)
        };

        let mut cost = self.total_cost();
        let mut temperature = cfg.init_temperature;
        // Set VYGES_PPL_ANNEAL_TRACE to compare this run against the reference's own debug trace,
        // perturbation by perturbation. It is the only way to localise a divergence in a stream
        // that is millions of draws long.
        let trace = std::env::var_os("VYGES_PPL_ANNEAL_TRACE").is_some();

        for iter in 0..cfg.max_iterations {
            for perturb in 0..perturb_per_iter {
                let prev_cost = self.perturb(swappable);
                let new_cost: i64 = self.moved.iter().map(|&p| self.pin_cost(p)).sum();
                let delta = new_cost - prev_cost;

                // ⚠️ The acceptance draw is taken EVERY perturbation, before it is known whether
                // the answer needs it. Taking it only when `delta > 0` would save a draw and put
                // the stream out of step from the first improving move onward.
                if trace {
                    eprintln!(
                        "iteration: {iter}; perturb: {perturb}; cost: {}; delta cost: {}",
                        cost + delta,
                        delta
                    );
                }
                let roll = self.gen.uniform_real();
                let accept = delta <= 0 || (-(delta as f32) / temperature).exp() > roll;

                if accept {
                    cost += delta;
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
                    for (i, &pin) in self.moved.iter().enumerate() {
                        self.assignment[pin] = self.prev_slots[i];
                    }
                }
                self.prev_slots.clear();
                self.new_slots.clear();
                self.moved.clear();
            }
            temperature *= cfg.alpha;
        }
        let _ = cost;
    }

    fn assignment(&self) -> Vec<Placement> {
        (0..self.pins.len()).map(|p| Placement { pin: p, slot: self.assignment[p] }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        Pin { name: name.into(), sinks: sinks.to_vec(), mirror: None }
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
            (0..12).map(|i| pin(&format!("p{i}"), &[(i as i32 * 300, 900)])).collect();
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
            (0..20).map(|i| pin(&format!("p{i}"), &[(i as i32 * 290, 700)])).collect();
        let cfg = Anneal { max_iterations: 300, ..Default::default() };

        let mut start = State::new(&s, &pins, &cfg);
        start.random_assignment();
        let before = start.total_cost();

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
            (0..15).map(|i| pin(&format!("p{i}"), &[(i as i32 * 250, 800)])).collect();
        let cfg = Anneal { max_iterations: 40, ..Default::default() };

        let a = run(&s, &pins, DIE, &cfg).unwrap();
        let b = run(&s, &pins, DIE, &cfg).unwrap();
        assert_eq!(a, b, "the same seed must give the same answer");

        let other = Anneal { seed: 7, ..cfg.clone() };
        let c = run(&s, &pins, DIE, &other).unwrap();
        assert_ne!(a, c, "a different seed should explore differently");
    }

    #[test]
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
            (0..16).map(|i| pin(&format!("p{i}"), &[(i as i32 * 300, 600)])).collect();
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

    #[test]
    fn a_design_needing_moves_this_does_not_implement_is_refused() {
        // Refusing is the honest answer: a partial implementation would put the random stream out
        // of step and produce a plausible-looking placement that is not the reference's.
        let s = slots(20);
        let mirrored = vec![
            Pin { name: "a".into(), sinks: vec![], mirror: Some(1) },
            Pin { name: "b".into(), sinks: vec![], mirror: None },
        ];
        assert_eq!(run(&s, &mirrored, DIE, &Anneal::default()), Err(Unsupported::NeedsExtraMoves));
    }

    #[test]
    fn nothing_to_place_or_nowhere_to_put_it_is_refused_not_guessed() {
        let s = slots(4);
        let many: Vec<Pin> = (0..9).map(|i| pin(&format!("p{i}"), &[])).collect();
        assert_eq!(run(&s, &many, DIE, &Anneal::default()), Err(Unsupported::Empty));
        assert_eq!(run(&s, &[], DIE, &Anneal::default()), Err(Unsupported::Empty));
    }
}
