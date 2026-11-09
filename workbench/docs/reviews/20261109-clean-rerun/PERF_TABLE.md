# Decode-worker sweep — encoder

sample: 768 gallery images (first N-1 in released order plus the last entry), batch 32, transform 224px, 3 steady trials plus 1 cold. Wall-clock only; not a whole-run figure.

| model | workers | trials | median img/s | min | max | cold img/s | peak RSS MiB | VRAM MiB | parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| encoder | 0 | 3 | 103.8 | 103.7 | 104.1 | 104.0 | 169.3 | 852 | IDENTICAL |
| encoder | 2 | 3 | 235.9 | 235.1 | 236.4 | 236.0 | 170.9 | 852 | IDENTICAL |
| encoder | 4 | 3 | 428.3 | 427.6 | 429.5 | 426.4 | 185.4 | 852 | IDENTICAL |
| encoder | 6 | 3 | 432.0 | 431.3 | 432.5 | 434.2 | 210.2 | 852 | IDENTICAL |
| encoder | 8 | 3 | 429.2 | 428.4 | 429.2 | 431.8 | 229.0 | 852 | IDENTICAL |

A setting marked REJECTED changed decoded content or order and must not be adopted for a canonical run, however fast it looks.
