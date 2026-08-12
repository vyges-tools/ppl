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
//! can disagree. Compare **cost** before calling a difference a bug — [`total_cost`] exists for
//! exactly that.
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
}
