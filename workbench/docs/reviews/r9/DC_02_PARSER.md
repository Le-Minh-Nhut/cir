# DCNet Category-Order Handling and Parser Audit

## 1. Assignment and Origin of `trainer.categories`

`trainer.categories` is assigned in `BaseTrainer.__init__` at `workbench/third_party/DCNet/base/base_trainer.py:32`:

```python
self.categories = config['data_loader']['args']['categories']
```

It originates directly from the run configuration dictionary under key `config['data_loader']['args']['categories']`.

Tracing the configuration loading path:
1. In `workbench/third_party/DCNet/test.py:102`, `test_config = ConfigParser(args, 'eval')` is initialized.
2. In `workbench/third_party/DCNet/parse_config.py:24`, `ConfigParser` reads JSON from disk:
   ```python
   self._config = read_json(self.cfg_fname)
   ```
   where `self.cfg_fname` is `self.resume / 'config.json'` (when resuming checkpoint) or `Path(args.config)`.
3. In `workbench/third_party/DCNet/trainer/trainer_joint.py:137`, `TrainerJoint._valid_epoch` also extracts:
   ```python
   categories = self.config['data_loader']['args']['categories']
   ```

In the upstream reference config (`workbench/third_party/DCNet/configs/ce/fashioniq_dcnet.json:22-26`), the list is:
```json
            "categories": [
                "dress",
                "shirt",
                "toptee"
            ],
```

A run config can reorder this list arbitrarily:
```json
            "categories": [
                "toptee",
                "dress",
                "shirt"
            ],
```
Setting this in `config.json` sets `trainer.categories` and `TrainerJoint._valid_epoch` loop variable `categories` to `["toptee", "dress", "shirt"]`.

---

## 2. Proving Parser Order Mapping via Fixed Tuple

### Claim: PROVEN
The regex at `workbench/backend/metrics_extraction.py:613-615` matches only `dress|shirt|toptee`, and the `comb:` line parser at `workbench/backend/metrics_extraction.py:638-645` maps values strictly using the fixed tuple `("dress", "shirt", "toptee")` rather than the order printed by the evaluator.

### Evidence
In `workbench/backend/metrics_extraction.py:612-623`:
```python
    combined_pairs: dict[str, tuple[float, float]] = {}
    for match in re.finditer(
            r"\[(dress|shirt|toptee)\]\s+comp\s+corr\s+r@10,\s*r@50:\s*"
            r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", stdout_text):
        category = match.group(1)
        # `i[0..3]` is comp10, comp50, corr10, corr50 (trainer_joint.py:231 `metric['recall'][i] =`).
        comp10, comp50, corr10, corr50 = (float(match.group(n)) * 100.0 for n in (2, 3, 4, 5))
        combined_pairs[category] = (comp10, comp50)
        combined_pairs[f"{category}__correction"] = (corr10, corr50)
        combined_pairs[f"{category}__composition"] = (comp10, comp50)
```

In `workbench/backend/metrics_extraction.py:633-645`:
```python
    comb_line = re.search(r"(?:^|\n)\s*(?:[^\n]* - INFO - )?comb:\s*"
                          r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
                          r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", stdout_text)

    if comb_line:
        cells = [float(value) * 100.0 for value in comb_line.groups()]
        # The `comb:` line carries no names, so it is only usable when the named per-category lines
        # are present to establish the order and the count.
        named = [c for c in ("dress", "shirt", "toptee") if c in combined_pairs]
        if len(named) != 3:
            return unparsed
        for index, category in enumerate(named):
            combined_pairs[category] = (cells[index * 2], cells[index * 2 + 1])
```

### Analysis
1. The regex only matches `dress`, `shirt`, `toptee` via explicit alternation `(dress|shirt|toptee)`.
2. The `comb_line` regex captures 6 anonymous floats without category names.
3. At line 642, `named` is constructed with:
   `named = [c for c in ("dress", "shirt", "toptee") if c in combined_pairs]`
   This list comprehension iterates across the literal tuple `("dress", "shirt", "toptee")`. Membership check `c in combined_pairs` checks presence in a dict set. It completely discards the order in which matches appeared in `stdout_text`.
4. As a result, `named` is always exactly `["dress", "shirt", "toptee"]`.
5. At lines 644-645:
   - `index 0` -> `category = "dress"` <- `(cells[0], cells[1])`
   - `index 1` -> `category = "shirt"` <- `(cells[2], cells[3])`
   - `index 2` -> `category = "toptee"` <- `(cells[4], cells[5])`
   The parser assumes positional ordering matching the fixed tuple `("dress", "shirt", "toptee")`.

---

## 3. Reproduction of Category Order Transposition

Evaluator category order: `toptee, dress, shirt`.
Comb line values: `0.7 0.9 0.2 0.4 0.3 0.5`.
Expected category metrics from evaluator:
- `toptee`: R@10=70.0, R@50=90.0
- `dress`: R@10=20.0, R@50=40.0
- `shirt`: R@10=30.0, R@50=50.0

### Reproduction Script and Literal Output
```bash
.venv/bin/python -c "
import json
from workbench.backend.metrics_extraction import parse_dcnet_output

stdout = '''2026-10-19 10:00:00,000 - INFO - [toptee] comp corr r@10, r@50: 0.65	0.85	0.60	0.80
2026-10-19 10:00:00,001 - INFO - [dress] comp corr r@10, r@50: 0.15	0.35	0.10	0.30
2026-10-19 10:00:00,002 - INFO - [shirt] comp corr r@10, r@50: 0.25	0.45	0.20	0.40
2026-10-19 10:00:00,003 - INFO - comb: 0.7 0.9 0.2 0.4 0.3 0.5
2026-10-19 10:00:00,004 - INFO - [Avg Recall]     : 0.4417
2026-10-19 10:00:00,005 - INFO - [Avg Recall corr]: 0.4000
2026-10-19 10:00:00,006 - INFO - [comb_avg]       : 0.5
'''

res = parse_dcnet_output(stdout)
print('extraction_status =', repr(res.extraction_status))
print('observed_metrics =', json.dumps(res.observed_metrics, indent=2))
print('category_metrics =', json.dumps(res.category_metrics, indent=2))
"
```

### Literal Function Output
```text
extraction_status = 'EXTRACTED_VERIFIED'
observed_metrics = {
  "r10": 40.0,
  "r50": 60.0,
  "mean": 50.0
}
category_metrics = {
  "dress": {
    "r10": 70.0,
    "r50": 90.0
  },
  "shirt": {
    "r10": 20.0,
    "r50": 40.0
  },
  "toptee": {
    "r10": 30.0,
    "r50": 50.0
  },
  "dress__composition": {
    "r10": 15.0,
    "r50": 35.0
  },
  "shirt__composition": {
    "r10": 25.0,
    "r50": 45.0
  },
  "toptee__composition": {
    "r10": 65.0,
    "r50": 85.0
  },
  "dress__correction": {
    "r10": 10.0,
    "r50": 30.0
  },
  "shirt__correction": {
    "r10": 20.0,
    "r50": 40.0
  },
  "toptee__correction": {
    "r10": 60.0,
    "r50": 80.0
  }
}
```

### Transposition Status
The transposition is **100% silent**:
- `extraction_status` returned `EXTRACTED_VERIFIED`.
- `dress` received `toptee`'s combined values (`70.0, 90.0` instead of `20.0, 40.0`).
- `shirt` received `dress`'s combined values (`20.0, 40.0` instead of `30.0, 50.0`).
- `toptee` received `shirt`'s combined values (`30.0, 50.0` instead of `70.0, 90.0`).
- `dress` combined metrics (`70.0, 90.0`) completely contradict its own composition metrics (`15.0, 35.0`), yet no error or warning was raised.

---

## 4. Consistency Check Against `[comb_avg]`

### Appearance in Stream
`[comb_avg]` appears in the stdout stream via `workbench/third_party/DCNet/test.py:73`:
```python
elif key == 'comb_avg':
    logger.info(f'[comb_avg]       : {value}')
```
where `value` is computed in `workbench/third_party/DCNet/trainer/trainer_joint.py:245`:
```python
metric['comb_avg'] = metric['comb'].mean()
```
For comb cells `[0.7, 0.9, 0.2, 0.4, 0.3, 0.5]`, the printed value is:
```text
[comb_avg]       : 0.5
```

### Consistency Check Logic (`workbench/backend/metrics_extraction.py:657-662`)
```python
    macro_r10 = sum(category_metrics[c]["r10"] for c in ("dress", "shirt", "toptee")) / 3.0
    macro_r50 = sum(category_metrics[c]["r50"] for c in ("dress", "shirt", "toptee")) / 3.0
    macro_mean = (macro_r10 + macro_r50) / 2.0

    declared = re.search(r"\[comb_avg\]\s*:\s*([0-9.eE+-]+)", stdout_text)
    if declared and abs(float(declared.group(1)) * 100.0 - macro_mean) > 0.5:
        return unparsed
```

### Arithmetic
The macro mean is calculated as:
$$\text{macro\_r10} = \frac{\text{cells}[0] + \text{cells}[2] + \text{cells}[4]}{3} = \frac{70.0 + 20.0 + 30.0}{3} = 40.0$$
$$\text{macro\_r50} = \frac{\text{cells}[1] + \text{cells}[3] + \text{cells}[5]}{3} = \frac{90.0 + 40.0 + 50.0}{3} = 60.0$$
$$\text{macro\_mean} = \frac{\text{macro\_r10} + \text{macro\_r50}}{2} = \frac{40.0 + 60.0}{2} = 50.0$$

Equivalently, expansion shows:
$$\text{macro\_mean} = \frac{1}{6} \sum_{i=0}^5 \text{cells}[i] = \frac{70.0 + 90.0 + 20.0 + 40.0 + 30.0 + 50.0}{6} = \frac{300.0}{6} = 50.0$$

The declared evaluator average is:
$$\text{declared} \times 100.0 = 0.5 \times 100.0 = 50.0$$
$$\text{abs}(50.0 - 50.0) = 0.0 \le 0.5$$

Addition is commutative: permuting the cells among categories leaves their total sum and their arithmetic mean unchanged.
Therefore, **the consistency check cannot catch category transposition**. It evaluates to `0.0` error under any permutation of the categories.

---

## 5. Native Logging and Category Order Information

### Native Print Loop
In `workbench/third_party/DCNet/test.py:74-84` (and identically in `workbench/third_party/DCNet/base/base_trainer.py:73-84`):
```python
            elif key == 'recall':
                for i, category in zip(value, trainer.categories):
                    if len(i) == 2:
                        logger.info(f'[{category}] r@10, r@50: {i[0]}\t{i[1]}')
                    elif len(i) == 4:
                        logger.info(f'[{category}] comp corr r@10, r@50: {i[0]}\t{i[1]}\t{i[2]}\t{i[3]}')
            elif key == 'comb':
                combstr = "comb:"
                for i, category in zip(value, trainer.categories):
                    combstr += f' {i[0]} {i[1]}'
                logger.info(combstr)
```

### Additional Parseable Emission
In `workbench/third_party/DCNet/test.py:19-20`:
```python
    logger.info("Running test with configuration:")
    logger.info(config)
```
`ConfigParser.__repr__` at `workbench/third_party/DCNet/parse_config.py:94-95` pretty-prints `self.__dict__`, which echoes `_config['data_loader']['args']['categories']`.

### What an Order-Faithful Parser CAN Rely On
An order-faithful parser CAN rely on:
1. **The order of occurrences of the per-category lines in stdout.** Both the `key == 'recall'` loop and the `key == 'comb'` loop iterate over the exact same list (`trainer.categories`). Therefore, the order of names extracted from the sequential matches of `\[([a-z_]+)\]\s+comp\s+corr\s+r@10` (or `\[([a-z_]+)\]\s+r@10`) provides the exact category sequence corresponding to the pairs in `comb:`.
2. Extracting `order = [m.group(1) for m in re.finditer(r"\[(dress|shirt|toptee)\]...", stdout)]` preserves the stream's category sequence and maps `comb:` cells without transposition.

---

## 6. Paper Benchmarks Definition and Registry Comparison

### Paper Benchmarks Definition
In `workbench/registry/paper_benchmarks/papers.yaml:1467-1470` (and in `workbench/registry/paper_benchmarks/extracted/legacy_cir_fashioniq_bundle.json:3`):
```yaml
  aggregation_method: The printed category cells are per-category R@10/R@50; the paper's Avg column is
    the mean of the six printed cells. The workbench's macro R@10/R@50 (equal-weight mean over the three
    categories) is derived here and printed nowhere as such; the paper's printed Avg equals the rounding
    of (macro_R@10 + macro_R@50)/2.
```
The docstring of `parse_dcnet_output` at `workbench/backend/metrics_extraction.py:598-601` cites this verbatim:
```python
    * **which branch.** Three variants exist (combined, composition, correction). The `comb:` line is
      the one the registry's own paper row describes — `paper_benchmarks/papers.yaml` defines the row
      as "the mean of the six printed cells" — so that is what is read.
```

### Registry `models.yaml` dcnet Row
In `workbench/registry/models.yaml:626-665`, the `dcnet` model entry carries:
```yaml
    # Sanity row: the paper prints only the category-level mean (40.83). r10/r50 exist only as
    # WORKBENCH_DERIVED aggregates in the committed row, so they stay null here — a "reported_"
    # field must not carry a derived value. Feeds the legacy parity display; the gate compares
    # against the committed benchmark row dcnet_fashioniq_val_table1.
    reported_mean: 40.83
```
- The registry `models.yaml` entry carries **no `paper_metrics` block** and **no per-category numbers** for dcnet.
- When `_check_parity(observed, paper_metrics)` runs inside `parse_dcnet_output`, `_check_parity` only inspects `("r10", "r50", "mean")` (`workbench/backend/metrics_extraction.py:58-64`). It never checks per-category numbers.
- Because `models.yaml` only carries `reported_mean: 40.83`, `_check_parity` receives `p_val = paper.get("r10") == None` and returns `PARITY_NOT_EVALUATED`.
- **Conclusion:** The registry `models.yaml` dcnet entry cannot contradict a transposition. (Only the committed benchmark row `dcnet_fashioniq_val_table1` in `papers.yaml:1486-1498` carries per-category numbers: `dress: 28.95/56.07`, `shirt: 23.95/47.30`, `toptee: 30.44/58.29`, which are evaluated exclusively by `paper_gate.py`, not by `parse_dcnet_output`).

---

## Verified claims

| Claim | Command / Verification Route | Observed Output |
|---|---|---|
| `trainer.categories` is assigned from config | Code inspection `workbench/third_party/DCNet/base/base_trainer.py:32` | `self.categories = config['data_loader']['args']['categories']` |
| `parse_dcnet_output` maps `comb:` using fixed tuple | Code inspection `workbench/backend/metrics_extraction.py:642-645` | `named = [c for c in ("dress", "shirt", "toptee") if c in combined_pairs]` |
| Transposed category order produces silent transposition in production parser | `.venv/bin/python -c "from workbench.backend.metrics_extraction import parse_dcnet_output; ..."` | `extraction_status: EXTRACTED_VERIFIED`, dress got 70.0/90.0, shirt got 20.0/40.0, toptee got 30.0/50.0 |
| `[comb_avg]` check passes on transposed categories | Evaluated macro mean arithmetic vs `declared` in reproduction script | `abs(50.0 - 50.0) == 0.0 <= 0.5`, parser accepted stream |
| Per-category lines and `comb:` line both iterate over `trainer.categories` | Code inspection `workbench/third_party/DCNet/test.py:74-84` | Both loops use `for i, category in zip(value, trainer.categories)` |
| `paper_benchmarks` defines mean as mean of six cells | Code inspection `workbench/registry/paper_benchmarks/papers.yaml:1467-1470` | `the paper's Avg column is the mean of the six printed cells` |
| `models.yaml` dcnet entry carries no per-category numbers | Inspection of `workbench/registry/models.yaml:626-665` | Only `reported_mean: 40.83` present; no per-category numbers |
