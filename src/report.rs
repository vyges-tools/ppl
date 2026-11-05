// SPDX-License-Identifier: Apache-2.0
//! The run's message lines: what `place_pins` says about the problem it was given and the
//! placement it made, as `(id, text)` pairs the caller hands to the logger (tool `PPL`).

use crate::sections::{net_hpwl, Pin};

/// What `printConfig` reports, read before any pin is assigned.
pub struct Config {
    /// Boundary slots neither blocked nor taken (`Slot::isAvailable`); nothing is taken yet.
    pub available_slots: usize,
    /// The top-layer lattice, blocked slots included; empty when the design declares none.
    pub top_layer_slots: usize,
    /// Ports placed by this run: not fixed, and on a net.
    pub ios: usize,
    /// Those whose net has no instance pin.
    pub ios_without_sink: usize,
    pub groups: usize,
    /// `None` when annealing, which does not cut sections.
    pub slots_per_section: Option<usize>,
    /// `Netlist::getMinimizeDisplacement`: PPL-0007 last.
    pub minimize_displacement: bool,
}

/// `printConfig`: the slot and I/O counts, in the reference's order and spacing.
pub fn config(c: &Config) -> Vec<(u16, String)> {
    let mut out = vec![(1, format!("Number of available slots {}", c.available_slots))];
    if c.top_layer_slots > 0 {
        out.push((62, format!("Number of top layer slots {}", c.top_layer_slots)));
    }
    out.push((2, format!("Number of I/O             {}", c.ios)));
    out.push((3, format!("Number of I/O w/sink      {}", c.ios - c.ios_without_sink)));
    out.push((4, format!("Number of I/O w/o sink    {}", c.ios_without_sink)));
    if c.groups > 0 {
        out.push((6, format!("Number of I/O Groups    {}", c.groups)));
    }
    if let Some(n) = c.slots_per_section {
        out.push((5, format!("Slots per section         {n}")));
    }
    if c.minimize_displacement {
        out.push((7, "Displacement minimization enabled".into()));
    }
    out
}

/// A pin's move, `(from, to)`.
pub type Move = ((i32, i32), (i32, i32));

/// `runHungarianMatching`'s debug line in a displacement-minimising run (`PPL displacement 1`):
/// over every assigned pin `(from, to)`, how many moved, their mean distance — the integer mean in
/// database units, then microns — and the largest.
pub fn displacement(moves: &[Move], dbu_per_micron: i32) -> String {
    let (mut moved, mut total, mut worst) = (0i64, 0i64, 0i64);
    for &(from, to) in moves {
        let d = i64::from((to.0 - from.0).abs()) + i64::from((to.1 - from.1).abs());
        if d > 0 {
            moved += 1;
            total += d;
            worst = worst.max(d);
        }
    }
    let um = |v: i64| v as f64 / f64::from(dbu_per_micron);
    let mean = if moved > 0 { um(total / moved) } else { 0.0 };
    format!("Moved {moved} of {} pins, mean {mean:.2} um, max {:.2} um.", moves.len(), um(worst))
}

/// `assignPinsToSections`' outcome: every pin found a section (PPL-8), or how many did (PPL-9).
pub fn sections_assigned(assigned: usize, ios: usize) -> (u16, String) {
    if assigned == ios {
        (8, "Successfully assigned pins to sections.".into())
    } else {
        (9, format!("Unsuccessfully assigned pins to sections ({assigned} out of {ios})."))
    }
}

/// `reportHPWL`: each placed pin's net, half-perimeter from the pin's position, summed in database
/// units and printed in microns through a `float`.
pub fn io_hpwl(pins: &[Pin], placed: impl IntoIterator<Item = (usize, (i32, i32))>, dbu_per_micron: i32) -> (u16, String) {
    let total: i64 = placed.into_iter().map(|(i, at)| net_hpwl(&pins[i].sinks, at)).sum();
    let microns = (total as f64 / dbu_per_micron as f64) as f32;
    (12, format!("I/O nets HPWL: {microns:.2} um."))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rule (printConfig): the top-layer and group lines appear only when there are any; the
    // section size only when the run cuts sections (not annealing).
    #[test]
    fn config_lines_follow_the_reference_order() {
        let c = Config { available_slots: 866, top_layer_slots: 900, ios: 54, ios_without_sink: 0, groups: 2, slots_per_section: Some(200), minimize_displacement: false };
        let ids: Vec<u16> = config(&c).iter().map(|l| l.0).collect();
        assert_eq!(ids, [1, 62, 2, 3, 4, 6, 5]);
        let plain = Config { top_layer_slots: 0, groups: 0, slots_per_section: None, ..c };
        assert_eq!(config(&plain), [
            (1, "Number of available slots 866".to_string()),
            (2, "Number of I/O             54".to_string()),
            (3, "Number of I/O w/sink      54".to_string()),
            (4, "Number of I/O w/o sink    0".to_string()),
        ]);
    }

    // Rule (printConfig): a displacement-minimising run (runHungarianMatching(true), as global
    // placement calls it) says so last, PPL-0007.
    #[test]
    fn a_displacement_minimising_run_says_so_last() {
        let c = Config { available_slots: 356, top_layer_slots: 0, ios: 54, ios_without_sink: 0, groups: 0, slots_per_section: Some(200), minimize_displacement: true };
        let lines = config(&c);
        assert_eq!(lines.iter().map(|l| l.0).collect::<Vec<_>>(), [1, 2, 3, 4, 5, 7]);
        assert_eq!(lines[5].1, "Displacement minimization enabled");
    }

    // Rule (reportHPWL): PPL-0012 measures each net's box even when the run minimises displacement
    // — getBB, not computeIONetHPWL.
    #[test]
    fn the_hpwl_report_ignores_displacement() {
        let pins = vec![Pin { name: "a".into(), sinks: vec![(1000, 3000)], mirror: None, initial: Some((9000, 9000)) }];
        assert_eq!(io_hpwl(&pins, [(0, (0, 0))], 1000).1, "I/O nets HPWL: 4.00 um.");
    }

    // Rule (runHungarianMatching, debug "displacement"): pins that did not move are not counted;
    // the mean is the moved pins' total over their count in whole dbu, then in microns.
    #[test]
    fn the_displacement_line_counts_only_moved_pins() {
        let moves = [((0, 0), (0, 0)), ((0, 0), (1000, 0)), ((0, 0), (0, 2001))];
        assert_eq!(displacement(&moves, 1000), "Moved 2 of 3 pins, mean 1.50 um, max 2.00 um.");
        assert_eq!(displacement(&moves[..1], 1000), "Moved 0 of 1 pins, mean 0.00 um, max 0.00 um.");
    }

    #[test]
    fn sections_line_names_the_shortfall() {
        assert_eq!(sections_assigned(54, 54).0, 8);
        assert_eq!(sections_assigned(50, 54), (9, "Unsuccessfully assigned pins to sections (50 out of 54).".to_string()));
    }

    // Rule (reportHPWL): the sum is over each pin's own net box, in dbu, then dbu / dbu-per-micron.
    #[test]
    fn hpwl_sums_each_pins_net_box() {
        let pins = vec![
            Pin { name: "a".into(), sinks: vec![(1000, 3000)], mirror: None, initial: None },
            Pin { name: "b".into(), sinks: vec![], mirror: None, initial: None },
        ];
        // a: |0-1000| + |0-3000| = 4000; b: no sinks, 0.
        assert_eq!(io_hpwl(&pins, [(0, (0, 0)), (1, (500, 500))], 2000), (12, "I/O nets HPWL: 2.00 um.".to_string()));
    }
}
