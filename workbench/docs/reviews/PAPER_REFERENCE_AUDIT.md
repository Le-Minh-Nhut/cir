# Paper reference audit

How the committed benchmark references were produced, what was verified, and by whom. The rows
themselves are in `workbench/registry/paper_benchmarks/papers.yaml`; the raw extraction payloads are in
`extracted/`; the independent verifier reports are in `verification/`.

## Two stages, and the reason for the split

**Offline extraction and verification.** A reader (human or model) downloads the paper, reads the
results table, and records what it actually says. A *second, independent* reader then re-derives the
same values from the primary source and reports CONFIRMED, PARTIAL, REFUTED or UNVERIFIABLE. Both
outputs are committed.

**Runtime comparison.** Plain Python loads the committed rows and compares numbers. No language model
is involved: see `workbench/backend/benchmark_references.py` and `workbench/backend/paper_gate.py`.

The split exists because a runtime judgement that consults a model is not reproducible. The extraction
is where judgement belongs, and it is frozen in Git with the evidence that produced it.

## What each row records

| Field group | Purpose |
|---|---|
| paper identity | title, authors, year, venue, DOI/arXiv id, version, URL, PDF sha256 and size |
| method identity | method name, variant, backbone |
| checkpoint condition | `checkpoint_training_noise_pct`, `checkpoint_mapping_status` |
| experiment identity | dataset, split, gallery protocol, reference policy, evaluation noise, preprocessing |
| metrics | `reported_metrics` (printed) and `derived_metrics` (computed, with formula) |
| provenance | table, row, page, column, and what the table actually says |
| verification | HOW each value was confirmed, and by which route |

A row that carries no comparable metric is `UNVERIFIED`, and the comparator skips it. A row that states
none of the conditions a run is compared on is refused outright
(`BLOCKED_REFERENCE_INCOMPLETE`), because it would otherwise certify any run whose numbers happened to
land nearby.

## Corrections this audit produced

| Correction | Evidence |
|---|---|
| Air-Know's published mean is **65.73**, not the 65.74 the registry recorded | two independent readers, four routes: arXiv HTML v1, arXiv HTML v2, the PDF text, the rendered page, and the author's own `assets/results-fiq.png`. The paper prints 65.73 and the printed Average columns give 65.735, so it truncates rather than rounds half-up; the discrepancy is recorded, not smoothed |
| **CLVC-Net's FashionIQ row exists** (avg R@10 30.70, R@50 58.41) | an earlier extraction reported the paper paywalled; a verifier found the first author's group page hosting the SIGIR camera-ready, and the implementer confirmed the same bytes and the same row by hand |
| **SPRC's R@10 is 54.92**, not `null` | the paper's own Average block prints it. The registry had recorded the omission; the paper's internal rounding difference (54.92 printed against a 54.72 category macro) is kept visible as a note |
| **DCNet's checkpoint condition is unknown**, not 0 % | the registry records `null` for the unresolved Drive bundle; 0 % was an inference from the table being a clean comparison, not a property of any observed artifact |
| The CSMCIR row had been written into SPRC's paper file | it inherited a wrong `paper_id` and no title; it now has its own paper file, with identity re-derived from the arXiv record and the table re-read |
| `mean_from_avg_columns` and `macro_r50_from_categories` were dead data | the comparator looks a metric up by name, so a derived entry under an arithmetic name was never read; Air-Know's 20 % row carried a derived R@50 of 75.72 against the printed 75.71. The builder now normalises the key and drops any derived copy of a printed value |

## What is not verified

- **`UNVERIFIABLE` is a legitimate outcome** and appears where a source could not be reached. It is not
  the same as "the numbers are wrong".
- **A paper that does not state its protocol** has `gallery_protocol` inferred from the authors' own
  code, and the row says so in its evidence. It is not a claim about the paper.
- **Checkpoint mapping is a separate claim from the metric.** Rows for Air-Know, ConeSep, HABIT and
  INTENT carry the paper's 0 % row while the authors publish only noise-trained weights; those rows are
  `NO_CHECKPOINT` for the mapping, and no noisy weight is comparable to them.
- **No payload was inspected.** Every mapping rests on author filenames, Hub metadata and published
  digests, never on loading a checkpoint.
