# Gelfand–Tsetlin patterns for `gl_N`

A Python interface with C kernels for working through the Gelfand–Tsetlin basis in [Molev, §2](https://arxiv.org/abs/math/0211289). The package covers interlacing, weights, matrix coefficients, the contravariant form, and sparse actions of enveloping algebra elements.

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
