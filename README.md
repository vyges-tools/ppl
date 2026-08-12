# vyges-ppl

IO pin placement over an OpenDB database.

A block's pins sit on its die boundary, and where they sit decides how much wire the rest of the
design needs to reach them. This engine works in two halves:

1. **Slots — where a pin *may* go.** Routing tracks on the die edge, minus the corners, minus the
   overhang a wide pin would have over the boundary, minus any spacing asked for between pins.
2. **Assignment — which pin goes where.** A choice *among* slots.

Assignment itself has two stages: the slot list is cut into fixed-size **sections**, each pin goes
to the cheapest section with room, and the optimal pairing is then solved *inside* each section by
the Hungarian algorithm. Optimal within a section, greedy between them.

**Constraints are honoured**: excluded regions, region constraints, pin groups (contiguous, and
optionally ordered), and mirrored pairs. Constrained pins are placed first, into their own region,
and the slots they take are withdrawn before anything else is placed.

Pins sent to the **top layer** (`define_pin_shape_pattern` + `-region up:`) are placed on a lattice
*inside* the die instead — a separate surface, where a position is legal only if a pin of the
declared size fits there and clears the routing on that layer.

A **rectilinear (polygon) die** is handled as a list of boundary segments rather than four named
edges, so a notched or L-shaped outline gets pins on every one of its faces.

`--annealing` selects the other optimiser: a randomised search over the whole boundary instead of
optimal matching within sections. It reproduces the reference's random stream exactly, so an
annealed placement is reproducible rather than merely comparable. Plain pins only — a design with
groups, constraints or mirrored pairs is refused rather than approximated.

## Use

```sh
vyges-ppl place-pins design.odb --hor-layers metal3 --ver-layers metal2
vyges-ppl slots      design.odb --ver-layers metal2 --min-distance 0.4 --corner-avoidance 1.5
vyges-ppl --describe        # machine-readable description of the command
vyges-ppl --help
```

`slots` reports where a pin *may* go; `place-pins` decides where each one *does* go.

`--hor-layers` carry pins on the **left and right** edges; `--ver-layers` on the **bottom and
top**. Distances are in microns, except `--min-distance` when `--min-distance-in-tracks` is given,
where it is a count of candidate positions.

Output is JSON: every slot in **generation order**, which runs counter-clockwise from the bottom
edge. The order is part of the contract — assignment works in contiguous runs of it.

Exit status: `0` slots generated · `1` the design offers none on the layers given · `2` usage or
read error.

## What it does not do yet

Read `--describe` for the authoritative list. In short: it reports the chosen positions but does
not yet write them to the database, because the pin *rectangle* depends on pin-length handling that
is not built. Macros and routing obstructions are not yet subtracted from slot availability. On a
polygon die, edge-named region constraints are reported and ignored rather than reinterpreted
against the bounding box.

Across all 62 comparable cases the total wirelength matches the reference or beats it — 25 of them
position-for-position, the rest by a cost-equal tie. No case is worse, none violates a constraint,
and none leaves a pin unplaced.

The simulated-annealing placement path is **deferred, not impossible**: the reference draws from
`boost::random`, which is specified and portable, so reproducing it exactly is feasible and simply
not built yet.

## Correctness

Validated against OpenROAD's IO placer at a pinned commit, on every test case that does not use a
feature listed above as unimplemented: **the total wirelength of this engine's placement equals the
reference's**, and every region constraint is independently verified to be satisfied.

Individual pins can still differ, and that is expected rather than tolerated. The optimal
assignment *cost* is unique; the optimal *pairing* is not, so where two pairings tie, two correct
implementations may swap a pair of pins. `--evaluate` scores a reference placement under the same
cost model, which is what makes "differs" distinguishable from "is worse".

The algorithm is reimplemented from published behaviour, not transliterated.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
