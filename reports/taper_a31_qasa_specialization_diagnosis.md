# TAPER A3.1 QASA / No-NULL — Comprehensive Specialization Diagnosis

- Checkpoint: `outputs/2026-08-26/03-49-22/best.pt`
- Protocol: `fashioniq_original`
- Samples: `{'dress': 512, 'shirt': 512, 'toptee': 512}`

## 1. Headline verdict

**Overall:** FAILURE MODE: monopoly collapse

**Ownership:** MONOPOLY / WINNER COLLAPSE evidence

**Functional specialization:** STRONG shared-task / functional redundancy evidence

> Automatic verdicts are heuristic. Raw metrics + causal ablations are the evidence.

## 2. QASA failure-mode baseline

- Slots: **4**
- `tau=0.5`, `rho=0.8`, `mu=0.3`
- `qasa_apply_at_eval=True`
- Uniform Quality baseline `1/L`: **0.2500**
- Perfectly uniform attention needs only **2** slots to reach `tau`.

Therefore high QASA coverage by itself is **not evidence of decomposition**. If entropy is near 1 and quality is near `1/L`, QASA may simply be pruning a diffuse attention field.

## 3. Ownership decomposition

| Metric | Value | Interpretation |
|---|---:|---|
| Normalized token entropy | 0.8918 | 1≈uniform; 0≈sharp |
| Top-1 probability | 0.4329 | Higher = sharper winner |
| Top1-top2 margin | 0.1598 | Near 0 = ambiguous |
| Winner-active slots | 1.1810 | Slots winning ≥1 token |
| Winner balance entropy | 0.0476 | 1≈balanced winners |
| Dominant mass share | 0.4323 | Uniform mass baseline=1/L |
| Ownership cosine | 0.9806 | 1 = same token pattern |
| Token-map JS | 0.0070 | 0 = identical token maps |

## 4. QASA behavior

| Metric | Value |
|---|---:|
| Selected K | 1.9974 |
| Quality mean | 0.2519 |
| Final coverage | 0.9994 |
| Novelty skips | 0.0059 |
| Selected K - winner-active K | 0.8164 |
| Selected mask == winner mask | 0.1836 |
| QASA rank regret vs all slots | 8.8294 |
| Fraction all-slots rank better | 0.4167 |
| Fraction QASA rank better | 0.3034 |

## 5. Representation specialization

| Metric | Value | Same-task warning |
|---|---:|---|
| Slot semantic cosine | 0.9984 | High = same semantic content |
| Slot effect cosine | 0.7795 | High = teacher effects align |
| Raw Edit Slot cosine | 0.9907 | High = redundant representation |
| Edit Slot cosine | 0.9907 | High = redundant post-activity slots |
| Selected effect cosine | 0.6960 | High = QASA kept redundant slots |
| Slot-effect effective rank | 1.2777 | Near 1 = common direction |
| Raw-slot effective rank | 1.0421 | Near 1 = low-dimensional |
| Effect norm CV | 0.7869 | Near 0 = equal magnitude |
| Dataset effect-prototype cosine | 0.9101 | High = same dataset-level role |

## 6. Causal test — are the 4 Edit Slots doing the same job?

This section is the strongest evidence because it intervenes on execution rather than only comparing representations.

1. **Drop-one:** remove each currently selected slot; compare query-change directions.
2. **Forced-only:** force each slot to execute alone; compare its edit direction.
3. **Pair-drop:** remove pairs and test non-additivity/redundancy.

| Metric | Value | Interpretation |
|---|---:|---|
| Drop contribution direction cosine | 0.9820 | High = different drops perturb query similarly |
| Forced-only effect cosine | 0.9936 | High = different slots perform similar edit alone |
| Dataset forced-effect prototype cosine | 0.9948 | High = same causal role across dataset |
| Pair-drop redundancy index | -1.0330 | Positive = overlapping/non-additive contribution |
| Full edit effect norm | 1.0580 | Distance from reference-only query |

## 7. Execution / primitive routing

- Slot↔primitive MI: **0.0069**
- Slot↔primitive NMI: **0.0075**
- State-change cosine: **0.9890**
- Route confidence: **0.1178**
- Transition strength: **0.9999**

Primitive counts by slot:

```text
S0: [66, 0, 0, 107, 102, 892, 0, 88]
S1: [5, 0, 0, 21, 8, 239, 0, 4]
S2: [0, 0, 0, 0, 0, 0, 0, 0]
S3: [73, 0, 0, 137, 96, 1145, 0, 85]
```

## 8. Retrieval counterfactuals

| Variant | R@10 | R@50 | Mean Recall | Mean Rank |
|---|---:|---:|---:|---:|
| qasa | 45.44 | 70.83 | 58.14 | 124.85 |
| all_slots | 45.51 | 71.22 | 58.37 | 116.02 |
| winner_active | 43.82 | 67.77 | 55.79 | 147.42 |
| reference_only | 4.56 | 12.89 | 8.72 | 1406.56 |

- `qasa`: actual current A3.1 policy.
- `all_slots`: diagnostic counterfactual; directly tests whether QASA pruning hurts.
- `winner_active`: diagnostic preview of the proposed next experiment only.
- `reference_only`: no execution baseline.

## 9. Per-slot panel

| Slot | Selected rate | Winner-active rate | Mass share | Winner count | Q | Only-effect norm | Only→full cos | Drop target loss | Only target gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.8171 | 0.0007 | 0.2316 | 0.0007 | 0.0001 | 0.8413 | 0.9874 | 0.0082 | 0.2261 |
| S1 | 0.1803 | 0.1803 | 0.2593 | 0.3288 | 0.0291 | 0.8421 | 0.9874 | 0.0128 | 0.2257 |
| S2 | 0.0000 | 0.0000 | 0.0768 | 0.0000 | 0.0000 | 0.7825 | 0.9714 | nan | 0.2162 |
| S3 | 1.0000 | 1.0000 | 0.4323 | 11.7630 | 0.9786 | 0.8429 | 0.9877 | 0.0094 | 0.2259 |

## 10. Red-team flags

### Diffuse / symmetric ownership
- FLAG `token_entropy_near_uniform` = `True`
- pass `top1_margin_small` = `False`
- FLAG `quality_near_uniform_baseline` = `True`
- FLAG `ownership_maps_highly_similar` = `True`

### Monopoly collapse
- pass `dominant_mass_large` = `False`
- FLAG `winner_active_slots_low` = `True`
- FLAG `winner_balance_entropy_low` = `True`

### Four-slots-same-task / functional redundancy
- pass `slot_effects_highly_aligned` = `False`
- FLAG `slot_effect_effective_rank_low` = `True`
- FLAG `forced_single_slot_effects_aligned` = `True`
- FLAG `drop_contribution_directions_aligned` = `True`
- FLAG `executed_state_changes_aligned` = `True`
- FLAG `slot_primitive_dependence_low` = `True`

## 11. Worst-case samples

Full top-N records are in the JSON. Below are the first five from each view. Use them for manual trace inspection.

### highest_ownership_entropy

- `fashioniq:val:shirt:305` — is luminous green and is bright green | H=0.9497 | own_cos=0.9881 | effect_cos=0.9839 | forced_cos=0.9969 | rank QASA/all=2/2
- `fashioniq:val:shirt:97` — is blue with sea turtles and desired item is dark blue with sea creatures pictured | H=0.9495 | own_cos=0.9907 | effect_cos=0.9307 | forced_cos=0.9997 | rank QASA/all=1/1
- `fashioniq:val:shirt:483` — is a blue nasa t shirt and is colored blue | H=0.9447 | own_cos=0.9947 | effect_cos=0.9126 | forced_cos=0.9998 | rank QASA/all=13/11
- `fashioniq:val:toptee:237` — the tank top is black with fringe and is solid black and has fringes | H=0.9435 | own_cos=0.9862 | effect_cos=0.8846 | forced_cos=0.9994 | rank QASA/all=2/2
- `fashioniq:val:shirt:268` — is a green t shirt with balls on it and is green and has more text | H=0.9421 | own_cos=0.9928 | effect_cos=0.9349 | forced_cos=0.9997 | rank QASA/all=46/45

### highest_ownership_similarity

- `fashioniq:val:shirt:465` — is baby outfit and is green and is a baby outfit | H=0.9419 | own_cos=0.9968 | effect_cos=0.9613 | forced_cos=0.9999 | rank QASA/all=2/1
- `fashioniq:val:toptee:452` — is identical and is same | H=0.8875 | own_cos=0.9952 | effect_cos=0.8597 | forced_cos=0.9209 | rank QASA/all=1/1
- `fashioniq:val:toptee:0` — is the same and appears to be exactly the same | H=0.8657 | own_cos=0.9951 | effect_cos=0.7830 | forced_cos=0.9918 | rank QASA/all=3/2
- `fashioniq:val:shirt:279` — its exactly what i want and is the same product | H=0.8659 | own_cos=0.9951 | effect_cos=0.7946 | forced_cos=0.9912 | rank QASA/all=1/1
- `fashioniq:val:toptee:226` — is more revealing and is a blue crop top | H=0.9314 | own_cos=0.9950 | effect_cos=0.9413 | forced_cos=0.9982 | rank QASA/all=91/77

### highest_functional_redundancy_forced_only

- `fashioniq:val:shirt:465` — is baby outfit and is green and is a baby outfit | H=0.9419 | own_cos=0.9968 | effect_cos=0.9613 | forced_cos=0.9999 | rank QASA/all=2/1
- `fashioniq:val:shirt:25` — is a long sleeved shirt and has long sleeve and is grey color | H=0.9034 | own_cos=0.9920 | effect_cos=0.8816 | forced_cos=0.9999 | rank QASA/all=35/41
- `fashioniq:val:toptee:131` — the long sleeved shirt is olive green and is a button down grey long sleeve shirt | H=0.9334 | own_cos=0.9936 | effect_cos=0.9327 | forced_cos=0.9999 | rank QASA/all=40/29
- `fashioniq:val:toptee:190` — is lighter in color and more lacy and is white with lace | H=0.9237 | own_cos=0.9885 | effect_cos=0.8737 | forced_cos=0.9998 | rank QASA/all=1/1
- `fashioniq:val:toptee:103` — is a white lacy long sleeve to the hips and a sexy white long sleeve shirt with see through on the back | H=0.9417 | own_cos=0.9936 | effect_cos=0.9338 | forced_cos=0.9998 | rank QASA/all=1/1

### lowest_slot_effect_effective_rank

- `fashioniq:val:shirt:333` — is lighter and less colorful and is lighter in color | H=0.8144 | own_cos=0.9684 | effect_cos=0.6320 | forced_cos=0.9916 | rank QASA/all=4/4
- `fashioniq:val:shirt:421` — has less colors and less pockets | H=0.7629 | own_cos=0.9501 | effect_cos=0.3756 | forced_cos=0.8921 | rank QASA/all=636/592
- `fashioniq:val:toptee:467` — is no sleeves and more lighter and is lighter colored with less shoulder straps | H=0.8302 | own_cos=0.9798 | effect_cos=0.5309 | forced_cos=0.9966 | rank QASA/all=434/438
- `fashioniq:val:shirt:305` — is luminous green and is bright green | H=0.9497 | own_cos=0.9881 | effect_cos=0.9839 | forced_cos=0.9969 | rank QASA/all=2/2
- `fashioniq:val:shirt:121` — is white with no stripes and is more solid colored | H=0.8473 | own_cos=0.9765 | effect_cos=0.6322 | forced_cos=0.9923 | rank QASA/all=1/1

### largest_qasa_rank_regret_vs_all

- `fashioniq:val:toptee:100` — is black and has short sleeves and is grey with image | H=0.8786 | own_cos=0.9746 | effect_cos=0.7090 | forced_cos=0.9935 | rank QASA/all=2219/1627
- `fashioniq:val:toptee:438` — is more casual and is darker and has shorter sleeves | H=0.8650 | own_cos=0.9772 | effect_cos=0.5872 | forced_cos=0.9936 | rank QASA/all=3599/3016
- `fashioniq:val:dress:386` — with pale green trim and pale green belt and is less black | H=0.8791 | own_cos=0.9561 | effect_cos=0.7226 | forced_cos=0.9970 | rank QASA/all=2841/2268
- `fashioniq:val:dress:91` — has a brighter color and is more of a boot with shorter heels and is a shoe | H=0.9148 | own_cos=0.9892 | effect_cos=0.8460 | forced_cos=0.9987 | rank QASA/all=759/219
- `fashioniq:val:toptee:367` — has shorter sleeves with an image in the center and is darker in color and more casual | H=0.8674 | own_cos=0.9728 | effect_cos=0.6596 | forced_cos=0.9971 | rank QASA/all=1364/849

## 12. Rule for claiming real specialization

Do **not** claim success from one metric. A convincing decomposition should agree across:

1. **Ownership:** multiple real winners, low ambiguity, non-identical token maps.
2. **Representation:** slot effects are not nearly collinear; effective rank meaningfully >1.
3. **Causality:** forced-only slots create different edit directions; drop-one effects are distinct.
4. **Execution:** state changes / primitive routing are not effectively identical.
5. **QASA:** high coverage comes with sharp ownership and quality above the uniform baseline.
