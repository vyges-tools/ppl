// SPDX-License-Identifier: Apache-2.0
//! `vyges-ppl` — IO pin placement.
//!
//! Pins on a block's boundary are placed where the wires that reach them are cheapest to route.
//! The engine does that in two halves, and the split is worth knowing before reading further:
//!
//! - **[`slots`] decides where a pin *may* go.** Routing tracks on the die edge, minus the
//!   corners, minus the boundary overhang of a wide pin, minus any spacing the caller asked for.
//!   Pure arithmetic over the technology's track patterns.
//! - **[`sections`] then decides which pin goes where**, in two stages: cut the slot list into
//!   fixed-size runs and send each pin greedily to the cheapest run with room, then solve the
//!   optimal pairing *inside* each run with [`hungarian`]. Optimal within a section, greedy
//!   between them — which is upstream's decomposition, and the reason the slot list's order is a
//!   contract rather than a presentation detail.
//! - **[`groups`] complicate both**, because a bus has to arrive together: its pins need a
//!   *contiguous* run of slots, and groups are therefore placed before any individual pin.
//! - **[`constraints`] runs first when the design demands it.** A pin restricted to a region is
//!   placed before any free pin, into sections cut from that region alone, and the slots it takes
//!   are withdrawn. Free pins first would let one sit where a constrained pin has no alternative.
//!
//! Nothing in this module reads a database; the binary does that and hands values in.
//!
//! # Tracing a divergence
//!
//! Three environment variables turn on step-by-step traces. Each exists because a difference of a
//! few tenths of a percent is otherwise almost impossible to localise, and every one of them has
//! found a real defect:
//!
//! | variable | prints |
//! | --- | --- |
//! | `VYGES_PPL_ANNEAL_TRACE` | every annealing perturbation: cost and delta |
//! | `VYGES_PPL_SECTION_TRACE` | each section's slot range and the pins routed to it |
//! | `VYGES_PPL_GROUP_TRACE` | a pin group's candidate start positions and their costs |

/// This crate's version, as Cargo knows it — the single number the whole suite is released on.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The copyright line `--version` prints.
pub const COPYRIGHT: &str = "© 2026 Vyges. All Rights Reserved.  https://vyges.com";

pub mod annealing;
pub mod constraints;
pub mod groups;
pub mod hungarian;
pub mod sections;
pub mod polygon;
pub mod rng;
pub mod slots;
pub mod toplayer;

pub use constraints::{
    collect as collect_constraints, interval_from_rect, overlapping, place_with_constraints,
    run_round, sort_constraints, Constraint,
};
pub use groups::Group;
pub use sections::{
    assign_groups_to_sections, assign_pins_to_sections, create_sections, find_sections,
    match_section, net_hpwl, place, solve_sections, Pin, Placement, Section, SLOTS_PER_SECTION,
};
pub use slots::{
    corner_avoidance, define_slots, enforce_spacing, is_blocked, layer_slots, slot_step, Boundary,
    Edge, Interval, LayerTracks, Params, Slot, TrackPattern, DEFAULT_MIN_DIST, NUM_TRACKS_OFFSET,
};
