# Gelfand–Tsetlin patterns for `gl_N`

A Python interface with C kernels for working through the Gelfand–Tsetlin basis in [Molev, §2](https://arxiv.org/abs/math/0211289). The package covers interlacing, weights, matrix coefficients, the contravariant form, and sparse actions of enveloping algebra elements.

See [Capabilities](CAPABILITIES.md) for API examples, indexing conventions, and arithmetic limits.

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

## Performance

Built-in integer rows are checked directly in C. `weight(pattern)` validates once and sums each row once. Ordered PBW word pairs concatenate directly.

| Workload | Before | After |
| --- | ---: | ---: |
| Validate 59,808 raw integer arrays | 162.8 ms | 5.6 ms |
| Compute full weights for 3,106 prepared patterns | 45.7 ms | 2.2 ms |
| Multiply 375 ordered PBW word pairs | 2.68 ms | 1.32 ms |
| Multiply 375 mixed sparse PBW pairs | 214.5 ms | 214.2 ms |

Times are same-host medians of seven interleaved warm runs on identical inputs. The ordered PBW fixture uses two ordered words of equal length per pair, with lengths from 1 to 375. The mixed workload exercises both the ordered shortcut and normal ordering.
