# Decode-worker sweep — encoder

sample: 768 gallery images (first N-1 in released order plus the last entry), batch 32, transform 224px, 3 steady trials plus 1 cold. Wall-clock only; not a whole-run figure.

| model | workers | trials | median img/s | min | max | cold img/s | peak RSS MiB | VRAM MiB | parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| encoder | 0 | 3 | 554.5 | 548.1 | 554.6 | 548.0 | 231.0 | 852 | IDENTICAL |
| encoder | 2 | 3 | 1032.8 | 998.1 | 1041.7 | 988.4 | 309.9 | 852 | IDENTICAL |
| encoder | 4 | 3 | 1756.2 | 1724.5 | 1826.1 | 1752.7 | 398.2 | 852 | IDENTICAL |
| encoder | 6 | 3 | 2282.1 | 2240.8 | 2288.9 | 2342.5 | 504.3 | 852 | IDENTICAL |
| encoder | 8 | 3 | 2688.1 | 2580.2 | 2688.9 | 2784.3 | 602.9 | 852 | IDENTICAL |

A setting marked REJECTED changed decoded content or order and must not be adopted for a canonical run, however fast it looks.
