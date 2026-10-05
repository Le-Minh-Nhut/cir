# Upstream and checkpoint audit

Audit date: 2026-10-05. No model artifacts were downloaded. Pins are GitHub `main` HEADs observed during audit; `sync_upstreams.py` checks out these exact commits.

| Model | Source pin | Official checkpoint evidence | Training noise | Protocol | Status |
| --- | --- | --- | --- | --- | --- |
| CSMCIR | `qzp2018/CSMCIR@774f94e2076ff17ea91703a6239d2a08f0e1a44e` | Author-linked `peng12138/CSMCIR`, `fashioniq_tuned_clip_best.pt` | 0% | original | URL verified |
| Air-Know | `iLearn-Lab/CVPR26-Air-Know@6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | Official iLearn-Lab Hub: `airknow_fiq_0.5.pt`, `airknow_fiq_0.8.pt` | 50%, 80% | original | No clean checkpoint |
| ConeSep | `iLearn-Lab/CVPR26-ConeSep@bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | Official iLearn-Lab Hub: `ConeSep-FIQ-N0.2.pt`, `N0.5.pt`, `N0.8.pt` | 20%, 50%, 80% | original | No clean checkpoint |
| PTHA + MTST | unavailable | No author-linked FashionIQ fine-tuned weight verified | n/a | n/a | BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT |
| HABIT | `iLearn-Lab/AAAI26-HABIT@6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | Official iLearn-Lab Hub: `HABIT-FIQ_N0.2.pt`, `N0.5.pt`, `N0.8.pt` | 20%, 50%, 80% | original | No clean checkpoint |
| INTENT | `iLearn-Lab/AAAI26-INTENT@2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` | Official iLearn-Lab Hub: `intent_fiq_0.2.pt`, `0.5.pt`, `0.8.pt` | 20%, 50%, 80% | original | No clean checkpoint |
| HINT | `iLearn-Lab/ICASSP26-HINT@bec50b6c8c19111893b617979502b948d1cea5b2` | Official iLearn-Lab Hub: `fashioniq.pt` | 0% | val | URL verified |
| ENCODER | `iLearn-Lab/AAAI25-ENCODER@29a2a31d6a56f677bf450c3be7cdaef423fb7018` | Official README public Google Drive folder; FashionIQ filename metadata known | 0% | val | direct file URL unresolved |
| PAIR | `iLearn-Lab/ICASSP25-PAIR@bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` | Existing audit names `pair-B1.pt`, `pair-B2.pt`; author source/mapping unverified | unknown | val | BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED |

## Protocol source observations

CSMCIR loads full FashionIQ validation split gallery and does not exclude reference. ENCODER/PAIR default `val-split` gallery is first-seen validation reference/target union and removes reference before recall. HINT source uses the same union and reference removal. Details and local implementation mapping: [PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md).

## Provenance policy

Registry stores only official/author-linked endpoints. `expected_sha256` stays `null` unless upstream publishes a hash. A locally computed hash belongs only in ignored `artifacts/checkpoints/download_manifest.json`; it never upgrades registry provenance.

Official paper scores are sanity references, never local reproductions. All listed runs remain `NOT_RUN` under laptop code-only policy.
