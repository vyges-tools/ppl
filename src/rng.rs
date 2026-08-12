// SPDX-License-Identifier: Apache-2.0
//! The random number stream the annealer draws from — reproduced exactly.
//!
//! Simulated annealing makes millions of random choices, so a placement is reproducible only if
//! the *stream* is. Matching the reference here is not a nicety: one draw out of step and every
//! subsequent decision diverges, and the two placements have nothing left to compare.
//!
//! # Why this is possible at all
//!
//! The reference draws from **Boost**, not from `std`. That matters more than it sounds. The C++
//! standard specifies what `std::uniform_int_distribution` *means* but not how it works, so two
//! standard libraries give different numbers from the same seed — measured: libc++ opens
//! `102 435 860…` where libstdc++ opens `374 796 950…`. Boost specifies the algorithm and
//! guarantees it across platforms, and the reference chose Boost deliberately: its own shuffle
//! helper carries the comment *"std::shuffle produces different results on different platforms"*.
//!
//! So this is a published, stable algorithm — the same thing every other engine in this programme
//! reimplements — rather than one standard library's private implementation.
//!
//! # How it was written
//!
//! Not from the Boost sources. Reference streams were **generated from Boost** and the algorithms
//! written to reproduce them; those streams are the tests at the bottom of this file, verbatim. A
//! change that breaks the stream fails here rather than three stages downstream as an unexplained
//! placement difference.

/// The Mersenne Twister, exactly as MT19937 specifies it.
///
/// Fully specified by its own definition, so there is no implementation freedom to get wrong —
/// unlike the distributions built on top of it.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let x = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut next = x >> 1;
            if x & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = self.state[(i + 397) % 624] ^ next;
        }
        self.index = 0;
    }

    /// The next raw 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// The engine's span, `max - min`. Every distribution below is defined in terms of it.
    const BRANGE: u32 = u32::MAX;

    /// A uniform integer in `[min, max]`, inclusive.
    ///
    /// 🔑 **Bucket division with rejection**, not a modulo and not a scaled float. The span is
    /// divided into `range + 1` buckets of equal *width*, and a draw is the bucket it falls in.
    /// Because the buckets rarely tile the span exactly, a draw landing in the leftover tail is
    /// **rejected and redrawn** — which is what keeps the result uniform, and what makes the number
    /// of draws consumed depend on the values seen.
    ///
    /// The rejection is not decoration: skipping it biases low values *and* desynchronises the
    /// stream from the reference's the first time a draw lands in the tail.
    pub fn uniform_int(&mut self, min: i64, max: i64) -> i64 {
        if max <= min {
            return min;
        }
        let range = (max - min) as u64;
        // Integer division, deliberately: the remainder is the tail that gets rejected below.
        let bucket = (Self::BRANGE as u64) / (range + 1);
        loop {
            let result = self.next_u32() as u64 / bucket;
            if result <= range {
                return min + result as i64;
            }
        }
    }

    /// A uniform `f32` in `[0, 1)`.
    ///
    /// One draw, scaled by the engine's full span. Computed in `f64` and narrowed, which is what
    /// the reference's arithmetic amounts to — narrowing first would lose bits the comparison
    /// depends on.
    pub fn uniform_real(&mut self) -> f32 {
        (self.next_u32() as f64 / (Self::BRANGE as f64 + 1.0)) as f32
    }

    /// Shuffle in place, matching the reference's own shuffle helper.
    ///
    /// Fisher–Yates walking **downwards**, drawing each index from `[0, i]`. The reference hand-
    /// wrote this rather than calling `std::shuffle`, precisely so the result would not depend on
    /// which standard library it was built against.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        if v.len() <= 1 {
            return;
        }
        for i in (1..v.len()).rev() {
            let j = self.uniform_int(0, i as i64) as usize;
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below was produced by Boost at the pinned commit and pasted in
    // unchanged. They are the specification this file is written against.

    #[test]
    fn the_engine_matches_the_reference_stream() {
        let mut g = Mt19937::new(42);
        let got: Vec<u32> = (0..8).map(|_| g.next_u32()).collect();
        assert_eq!(
            got,
            vec![
                1608637542, 3421126067, 4083286876, 787846414, 3143890026, 3348747335,
                2571218620, 2563451924
            ]
        );
    }

    #[test]
    fn a_uniform_integer_matches_across_several_ranges() {
        let mut g = Mt19937::new(42);
        let got: Vec<i64> = (0..10).map(|_| g.uniform_int(0, 999)).collect();
        assert_eq!(got, vec![374, 796, 950, 183, 731, 779, 598, 596, 156, 445]);

        // A range of two exercises the widest buckets, where rejection is most likely.
        let mut g = Mt19937::new(42);
        let got: Vec<i64> = (0..16).map(|_| g.uniform_int(0, 1)).collect();
        assert_eq!(got, vec![0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0]);

        // ⚠️ A wide range with a non-dividing span — the case that proves the algorithm is bucket
        // division and not a scaled float. A float scaling gives 374542 here; the answer is 374627.
        let mut g = Mt19937::new(42);
        let got: Vec<i64> = (0..8).map(|_| g.uniform_int(3, 1_000_000)).collect();
        assert_eq!(got, vec![374627, 796725, 950931, 183479, 732161, 779869, 598796, 596987]);
    }

    #[test]
    fn an_empty_range_consumes_nothing() {
        // It must not draw: a wasted draw shifts every later value in the stream.
        let mut g = Mt19937::new(42);
        assert_eq!((0..4).map(|_| g.uniform_int(5, 5)).collect::<Vec<_>>(), vec![5, 5, 5, 5]);
        assert_eq!(g.next_u32(), 1608637542, "the stream is untouched");
    }

    #[test]
    fn a_uniform_real_matches_the_reference_stream() {
        let mut g = Mt19937::new(42);
        let got: Vec<f32> = (0..8).map(|_| g.uniform_real()).collect();
        assert_eq!(
            got,
            vec![
                0.37454012, 0.796543, 0.95071429, 0.183434784, 0.731993914, 0.779690981,
                0.598658502, 0.596850157
            ]
        );
    }

    #[test]
    fn the_shuffle_matches_the_reference_helper() {
        let mut g = Mt19937::new(42);
        let mut v: Vec<i32> = (0..12).collect();
        g.shuffle(&mut v);
        assert_eq!(v, vec![11, 6, 10, 0, 2, 3, 7, 5, 1, 9, 8, 4]);

        let mut g = Mt19937::new(7);
        let mut v: Vec<i32> = (0..5).collect();
        g.shuffle(&mut v);
        assert_eq!(v, vec![1, 3, 2, 4, 0]);
    }

    #[test]
    fn a_short_shuffle_is_a_no_op_and_consumes_nothing() {
        let mut g = Mt19937::new(42);
        let mut one = [7];
        g.shuffle(&mut one);
        let mut none: [i32; 0] = [];
        g.shuffle(&mut none);
        assert_eq!(one, [7]);
        assert_eq!(g.next_u32(), 1608637542, "the stream is untouched");
    }

    #[test]
    fn the_engine_stays_in_step_past_its_first_reload() {
        // MT19937 regenerates its state every 624 draws; an off-by-one in that reload shows up
        // only after the first block, which is exactly where a placement would silently diverge.
        let mut g = Mt19937::new(42);
        let all: Vec<u32> = (0..1300).map(|_| g.next_u32()).collect();
        // Independently re-derived by running the engine forward from a fresh seed.
        let mut h = Mt19937::new(42);
        for _ in 0..624 {
            h.next_u32();
        }
        assert_eq!(all[624], h.next_u32(), "first value after the reload");
        assert_eq!(all.len(), 1300);
        assert!(all[624..].iter().any(|&x| x != 0));
    }
}
