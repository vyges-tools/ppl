// SPDX-License-Identifier: Apache-2.0
//! Optimal assignment — the Hungarian algorithm, as a shortest-augmenting-path search.
//!
//! Given a cost for putting each pin in each slot, find the pairing that minimises the **total**
//! cost. Greedily giving every pin its own cheapest slot does not do this: two pins wanting the
//! same slot must be separated, and the cheaper separation is often to move the pin that seemed
//! settled. The Hungarian algorithm finds the global optimum in O(n²m).
//!
//! ⚠️ **The optimal cost is unique; the optimal assignment is not.** When two pairings cost the
//! same total, which one comes out is decided by iteration order, and two correct implementations
//! can disagree. Placement therefore uses [`munkres`] — Buehren's star-and-prime Munkres, step for
//! step, on the matrix the way round the reference builds it — whose pairing among equal totals is
//! the reference's; [`solve`], the potentials search, finds the same TOTAL but its own pairing.
//! Transcribing the solver took the placement comparison from 34 to 87 exact of 88 cases: the
//! "ties" were the solver, not the cost model.
//!
//! Nothing here knows what a pin or a slot is. It is a matrix and a permutation.

/// No pairing is possible at this cost — used to forbid a cell.
pub const FORBIDDEN: i64 = i64::MAX / 4;

/// Assign every row to a distinct column, minimising the total cost.
///
/// `cost[r][c]` is the cost of pairing row `r` with column `c`. Requires `rows <= cols`; the
/// caller decides which side is which, and pins-as-rows is the orientation that fits, since there
/// are always at least as many slots as pins in a section.
///
/// Returns `assignment[r] = c`. Rows are all assigned; surplus columns are left over.
///
/// Returns `None` if the matrix is ragged or has more rows than columns — a caller that built the
/// matrix wrongly gets an error rather than a confidently wrong pairing.
pub fn solve(cost: &[Vec<i64>]) -> Option<Vec<usize>> {
    let n = cost.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let m = cost[0].len();
    if m < n || cost.iter().any(|r| r.len() != m) {
        return None;
    }

    // The classic potentials formulation, 1-indexed with a virtual row 0 as the search's start.
    // `u` and `v` are the dual variables: a cell is "tight" when cost - u[row] - v[col] is zero,
    // and the algorithm keeps raising potentials until enough tight cells exist to extend the
    // matching by one row.
    let inf = i64::MAX / 2;
    let mut u = vec![0i64; n + 1];
    let mut v = vec![0i64; m + 1];
    // `row_for_col[c]` is the row currently matched to column c (0 = none, columns 1-indexed).
    let mut row_for_col = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for row in 1..=n {
        row_for_col[0] = row;
        let mut col0 = 0usize;
        let mut minv = vec![inf; m + 1];
        let mut used = vec![false; m + 1];

        loop {
            used[col0] = true;
            let cur_row = row_for_col[col0];
            let mut delta = inf;
            let mut next_col = 0usize;

            for col in 1..=m {
                if used[col] {
                    continue;
                }
                let c = cost[cur_row - 1][col - 1].saturating_sub(u[cur_row]).saturating_sub(v[col]);
                if c < minv[col] {
                    minv[col] = c;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    next_col = col;
                }
            }
            // Raise the potentials by the slack we just found. Every tight cell stays tight, and
            // at least one more becomes tight — which is what makes the search terminate.
            for col in 0..=m {
                if used[col] {
                    u[row_for_col[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = next_col;
            if row_for_col[col0] == 0 {
                break;
            }
        }
        // Walk the augmenting path back, flipping each edge along it.
        while col0 != 0 {
            let prev = way[col0];
            row_for_col[col0] = row_for_col[prev];
            col0 = prev;
        }
    }

    let mut out = vec![usize::MAX; n];
    for col in 1..=m {
        if row_for_col[col] != 0 {
            out[row_for_col[col] - 1] = col - 1;
        }
    }
    if out.contains(&usize::MAX) {
        return None;
    }
    Some(out)
}

/// Optimal assignment as Markus Buehren's `assignmentoptimal` finds it (Munkres: starred and primed
/// zeros, column covers) — the pairing it returns among EQUAL-cost optima, which [`solve`]'s
/// potentials search does not reproduce.
///
/// `cost[r][c]` with `rows >= cols` here (slots as rows, pins as columns), C `int` arithmetic:
/// unset cells are `i32::MAX` and the covered-row additions wrap as the C build does. Returns
/// `assignment[r] = Some(c)`, `None` for a row left over.
///
/// Transcribed from the BSD-licensed original:
/// Copyright (c) 2014, Markus Buehren; Copyright (c) 2016, mcximing. All rights reserved.
/// Redistribution and use in source and binary forms, with or without modification, are permitted
/// provided that the following conditions are met: redistributions of source code must retain the
/// above copyright notice, this list of conditions and the following disclaimer; redistributions in
/// binary form must reproduce the above copyright notice, this list of conditions and the
/// following disclaimer in the documentation and/or other materials provided with the
/// distribution. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
/// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
/// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
/// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
/// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
/// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
/// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
pub fn munkres(cost: &[Vec<i32>]) -> Vec<Option<usize>> {
    let rows = cost.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = cost[0].len();
    // Column-major, as the original stores the matrix: element (r, c) at r + rows * c.
    let at = |r: usize, c: usize| r + rows * c;
    let mut d = vec![0i32; rows * cols];
    for (r, row) in cost.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            d[at(r, c)] = v;
        }
    }
    let mut cov_c = vec![false; cols];
    let mut cov_r = vec![false; rows];
    let mut star = vec![false; rows * cols];
    let mut prime = vec![false; rows * cols];
    let min_dim;
    if rows <= cols {
        min_dim = rows;
        for r in 0..rows {
            let m = (0..cols).map(|c| d[at(r, c)]).min().expect("a column");
            for c in 0..cols {
                d[at(r, c)] = d[at(r, c)].wrapping_sub(m);
            }
        }
        // Steps 1 and 2a: each row's first zero in an uncovered column.
        for r in 0..rows {
            if let Some(c) = (0..cols).find(|&c| d[at(r, c)] == 0 && !cov_c[c]) {
                star[at(r, c)] = true;
                cov_c[c] = true;
            }
        }
    } else {
        min_dim = cols;
        for c in 0..cols {
            let m = (0..rows).map(|r| d[at(r, c)]).min().expect("a row");
            for r in 0..rows {
                d[at(r, c)] = d[at(r, c)].wrapping_sub(m);
            }
        }
        // Steps 1 and 2a: each column's first zero in an uncovered row.
        for c in 0..cols {
            if let Some(r) = (0..rows).find(|&r| d[at(r, c)] == 0 && !cov_r[r]) {
                star[at(r, c)] = true;
                cov_c[c] = true;
                cov_r[r] = true;
            }
        }
        cov_r.iter_mut().for_each(|x| *x = false);
    }
    // step2b → step3 ⇄ step5, step3 → step4 → step2a → step2b, until min_dim columns are covered.
    loop {
        // step2b
        if cov_c.iter().filter(|&&x| x).count() == min_dim {
            break;
        }
        // step3: prime uncovered zeros, columns then rows, until one has no star in its row.
        let found = 'step3: loop {
            let mut zeros_found = true;
            while zeros_found {
                zeros_found = false;
                for c in 0..cols {
                    if cov_c[c] {
                        continue;
                    }
                    for r in 0..rows {
                        if cov_r[r] || d[at(r, c)] != 0 {
                            continue;
                        }
                        prime[at(r, c)] = true;
                        match (0..cols).find(|&sc| star[at(r, sc)]) {
                            None => break 'step3 (r, c),
                            Some(sc) => {
                                cov_r[r] = true;
                                cov_c[sc] = false;
                                zeros_found = true;
                                break;
                            }
                        }
                    }
                }
            }
            // step5: the smallest uncovered value, added to covered rows, taken from uncovered columns.
            let mut h = i32::MAX;
            for r in (0..rows).filter(|&r| !cov_r[r]) {
                for c in (0..cols).filter(|&c| !cov_c[c]) {
                    h = h.min(d[at(r, c)]);
                }
            }
            for r in (0..rows).filter(|&r| cov_r[r]) {
                for c in 0..cols {
                    d[at(r, c)] = d[at(r, c)].wrapping_add(h);
                }
            }
            for c in (0..cols).filter(|&c| !cov_c[c]) {
                for r in 0..rows {
                    d[at(r, c)] = d[at(r, c)].wrapping_sub(h);
                }
            }
        };
        // step4: the alternating path of primes and stars from the found zero.
        let mut new_star = star.clone();
        let (row, col) = found;
        new_star[at(row, col)] = true;
        let mut star_col = col;
        let mut star_row = (0..rows).find(|&r| star[at(r, star_col)]);
        while let Some(sr) = star_row {
            new_star[at(sr, star_col)] = false;
            let prime_col = (0..cols).find(|&c| prime[at(sr, c)]).expect("a starred zero's row holds a prime");
            new_star[at(sr, prime_col)] = true;
            star_col = prime_col;
            star_row = (0..rows).find(|&r| star[at(r, star_col)]);
        }
        prime.iter_mut().for_each(|x| *x = false);
        star = new_star;
        cov_r.iter_mut().for_each(|x| *x = false);
        // step2a: cover every column holding a star.
        for c in 0..cols {
            if (0..rows).any(|r| star[at(r, c)]) {
                cov_c[c] = true;
            }
        }
    }
    (0..rows).map(|r| (0..cols).find(|&c| star[at(r, c)])).collect()
}

/// The total cost of an assignment — the number that is actually unique when the pairing is not.
pub fn total_cost(cost: &[Vec<i64>], assignment: &[usize]) -> i64 {
    assignment.iter().enumerate().map(|(r, &c)| cost[r][c]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The optimum by exhaustive search, for checking the real solver on small matrices.
    fn brute_force(cost: &[Vec<i64>]) -> i64 {
        let m = cost[0].len();
        let mut best = i64::MAX;
        let mut cols: Vec<usize> = (0..m).collect();
        // Every ordered choice of n columns out of m.
        fn go(cost: &[Vec<i64>], cols: &mut Vec<usize>, row: usize, acc: i64, best: &mut i64) {
            if row == cost.len() {
                *best = (*best).min(acc);
                return;
            }
            for i in row..cols.len() {
                cols.swap(row, i);
                go(cost, cols, row + 1, acc + cost[row][cols[row]], best);
                cols.swap(row, i);
            }
        }
        go(cost, &mut cols, 0, 0, &mut best);
        best
    }

    #[test]
    fn the_obvious_case_is_the_diagonal() {
        let c = vec![vec![1, 9, 9], vec![9, 1, 9], vec![9, 9, 1]];
        assert_eq!(solve(&c).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn it_moves_a_settled_row_when_that_is_globally_cheaper() {
        // The point of the algorithm, in three rows. Row 0's cheapest column is 0 (cost 1), but
        // rows 1 and 2 can ONLY use column 0 cheaply; giving it away costs far more than row 0
        // taking its second choice. A greedy pass takes column 0 for row 0 and pays for it.
        let c = vec![
            vec![1, 2, 3],
            vec![1, 50, 50],
            vec![1, 50, 3],
        ];
        let a = solve(&c).unwrap();
        assert_eq!(total_cost(&c, &a), brute_force(&c));
        assert_eq!(a[1], 0, "the row with no alternative gets the contested column");
    }

    #[test]
    fn surplus_columns_are_simply_left_over() {
        // A section always has at least as many slots as pins, so this is the normal shape.
        let c = vec![vec![5, 1, 9, 9], vec![9, 9, 2, 7]];
        let a = solve(&c).unwrap();
        assert_eq!(a.len(), 2, "one column per row, the rest unused");
        assert_eq!(total_cost(&c, &a), 3);
        assert_ne!(a[0], a[1], "no column is used twice");
    }

    #[test]
    fn a_forbidden_cell_is_avoided_when_anything_else_will_do() {
        let c = vec![vec![FORBIDDEN, 5], vec![3, FORBIDDEN]];
        let a = solve(&c).unwrap();
        assert_eq!(a, vec![1, 0]);
        assert_eq!(total_cost(&c, &a), 8, "no overflow from the forbidden cells");
    }

    #[test]
    fn it_matches_brute_force_on_many_random_matrices() {
        // The property that matters: the total is optimal. Which optimal pairing comes out is
        // deliberately NOT asserted — ties are resolved by iteration order, and two correct
        // implementations may differ.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..300 {
            let n = 1 + (next() % 5) as usize;
            let m = n + (next() % 4) as usize;
            // A small value range on purpose: it forces frequent ties, which is where a wrong
            // implementation stops being optimal.
            let c: Vec<Vec<i64>> =
                (0..n).map(|_| (0..m).map(|_| (next() % 9) as i64).collect()).collect();
            let a = solve(&c).unwrap();
            assert_eq!(
                total_cost(&c, &a),
                brute_force(&c),
                "case {case}: not optimal for {c:?} -> {a:?}"
            );
            let mut seen: Vec<usize> = a.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), a.len(), "case {case}: a column was used twice");
        }
    }

    #[test]
    fn a_large_cost_range_does_not_overflow_the_potentials() {
        // Real costs are HPWL in database units and run to millions; the potentials accumulate
        // across rows, so the arithmetic has to have room.
        let big = 2_000_000_000i64;
        let c = vec![vec![big, big / 2, big], vec![big, big, big / 3], vec![big / 4, big, big]];
        let a = solve(&c).unwrap();
        assert_eq!(total_cost(&c, &a), brute_force(&c));
    }

    #[test]
    fn a_ragged_or_too_tall_matrix_is_refused_rather_than_guessed() {
        assert!(solve(&[vec![1, 2], vec![3]]).is_none(), "ragged");
        assert!(solve(&[vec![1], vec![2], vec![3]]).is_none(), "more rows than columns");
        assert_eq!(solve(&[]).unwrap(), Vec::<usize>::new());
    }

    // `assignmentoptimal` on more rows (slots) than columns (pins): optimal, each column once,
    // every column assigned — over many small matrices with frequent ties.
    #[test]
    fn munkres_is_optimal_on_tall_matrices() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..300 {
            let pins = 1 + (next() % 5) as usize;
            let slots = pins + (next() % 4) as usize;
            let by_slot: Vec<Vec<i32>> = (0..slots).map(|_| (0..pins).map(|_| (next() % 9) as i32).collect()).collect();
            let a = munkres(&by_slot);
            let mut pin_slot = vec![usize::MAX; pins];
            for (s, c) in a.iter().enumerate() {
                if let Some(c) = *c {
                    assert_eq!(pin_slot[c], usize::MAX, "case {case}: pin {c} twice");
                    pin_slot[c] = s;
                }
            }
            assert!(!pin_slot.contains(&usize::MAX), "case {case}: a pin unassigned");
            let by_pin: Vec<Vec<i64>> = (0..pins).map(|p| (0..slots).map(|s| i64::from(by_slot[s][p])).collect()).collect();
            assert_eq!(total_cost(&by_pin, &pin_slot), brute_force(&by_pin), "case {case}: not optimal");
        }
    }

    // Among equal totals the pairing is the original's: slot costs [[1, 1], [0, 0]] tie at 1
    // either way. Column 1's zero in row 1 is primed, row 1's star moves, step 5 levels the
    // matrix, then column 0's zero in row 0 starts the path: slot 0 → pin 0, slot 1 → pin 1.
    #[test]
    fn munkres_breaks_a_tie_as_the_original_steps_do() {
        assert_eq!(munkres(&[vec![1, 1], vec![0, 0]]), vec![Some(0), Some(1)]);
        // A surplus slot is left over; with all costs equal the first slots, in pin order.
        assert_eq!(munkres(&[vec![5, 5], vec![5, 5], vec![5, 5]]), vec![Some(0), Some(1), None]);
    }
}
