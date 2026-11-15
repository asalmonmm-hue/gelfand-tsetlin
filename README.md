# Gelfand–Tsetlin patterns for classical types

A Python library with C kernels for Gelfand–Tsetlin patterns and representation-theoretic computations in the classical Lie algebras, following Molev’s [§§2–4](https://arxiv.org/abs/math/0211289).

- **Type A (`gl_N`)** has pattern validation, matrix coefficients, the contravariant form, and sparse enveloping algebra actions. It also covers quantum minors, coproducts, the quadratic Casimir, and tensor products with the standard module and its dual.
- **Type B** has pattern validation and ordered basis factors.
- **Types B, C, and D** have exact matrix generators, normalized branching operators, and interpolation polynomials. The library also tests the string criterion for irreducibility of Yangian evaluation-module tensor products.
- **The alternating orthogonal chain** has pattern validation and lazy one-step branching for `o_N` to `o_(N-1)`.

See [Capabilities](CAPABILITIES.md) for the tool comparison, API examples, indexing conventions, and arithmetic limits.

## Run locally

Build with CPython and a C compiler for that Python installation.

```sh
python3 -m pip install .
python3 -m unittest discover -s tests
```

```python
from gelfand_tsetlin import weight

assert weight([[2, 1, 0], [1, 0], [1]]) == (1, 0, 2)
```

## Head-to-head benchmarks

These workloads use the same inputs and exact outputs on a single Linux host.

| Workload | gelfand-tsetlin | passageMath 10.8.13 | Macaulay2 1.26.06 |
| --- | ---: | ---: | ---: |
| Validate 59,808 raw integer arrays | 5.6 ms | 44.6 ms | — |
| Compute full weights for 3,106 prepared patterns | 2.2 ms | 29.5 ms | — |
| Multiply 375 prepared rational PBW pairs | 221 ms | 116 ms | — |
| Materialize eight exact dense `sl_3` matrices, dimension 64 | 12.0 ms | — | 171.2 ms |

Times are warm medians with startup excluded. Validation and weights use
seven interleaved runs. PBW uses nine interleaved runs with passageMath’s
basis-product cache populated. Matrix timings use 31 runs with patterns
already available. gelfand-tsetlin materializes dense output from sparse
columns. A dash indicates an unmeasured workload.

## License

This project is released under the [MIT License](LICENSE).
