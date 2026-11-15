# Repository guidance

- Follow Molev’s *Gelfand–Tsetlin bases for classical Lie algebras*, §2,
  for pattern indexing, operator order, and normalization.
- Keep computational kernels in C and the public interface in Python. Keep
  fixed-width coefficient kernels allocation-free. Preserve exact arithmetic
  and the documented overflow behavior.
- Keep dependencies light. Use CPython and its standard library by default.
  Justify any new dependency by the operation that needs it. Keep comparator
  installations, benchmark scratch files, and credentials outside this repo.
- Compare overlapping capabilities head to head using the same inputs,
  arithmetic domain, and output. Record the tool versions, workload, and
  warm-up or cache policy. Put timings in `README.md` and capability scope in
  `CAPABILITIES.md`.
- Delegate independent implementation, review, or comparison work to
  subagents when available.
- Build with `python3 -m pip install .` and run
  `python3 -m unittest discover -s tests`. Check the examples in both
  documentation files when changing the public interface.
