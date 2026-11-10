
"""Phase A: generate PROTOCOL_MATRIX.csv from measured + source-verified facts.

Every cell is either measured in raw/*.json (recomputed by scripts/pa_*.py), read from a pinned
file at the cited line, or the literal NOT_RECORDED. No value is inferred from a paper.
"""
import csv, json
from pathlib import Path

BASE = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261110-e2e-repair")
cohorts = json.load(open(BASE / "raw/protocol_cohorts.json"))
circus = json.load(open(BASE / "raw/circus_cohorts.json"))
mem = json.load(open(BASE / "raw/membership.json"))
hashes = {}
for line in (BASE / "raw/checkpoint_hashes.tsv").read_text().splitlines()[1:]:
    parts = line.split("\t")
    model_key = parts[0].split("/")[0]
    ck_key = parts[1]
    entry = {"model": model_key, "checkpoint_id": ck_key, "path": parts[2],
             "bytes": parts[3], "sha256": parts[4], "match": parts[6],
             "member": parts[0].split("/")[1] if "/" in parts[0] else None}
    hashes[(model_key, ck_key)] = entry
    hashes.setdefault(model_key, entry)
    if entry["member"]:
        hashes.setdefault((model_key, "bundle_members"), []).append(entry)

TP = "workbench/third_party"
COLS = [
    "row_id", "row_kind", "model_id", "protocol_label_recorded", "protocol_label_validated",
    "label_verdict", "upstream_repo", "upstream_commit", "vendor_patch", "checkpoint_id",
    "checkpoint_sha256", "checkpoint_resolution", "architecture_config", "noise_transfer",
    "interpreter", "python", "torch", "precision_device", "seed_recorded", "dataset_release",
    "split", "category_scope", "query_count", "query_membership_sha256", "gallery_kind",
    "gallery_count", "gallery_sha256", "gallery_order_source", "category_filtered_gallery",
    "reference_policy", "gallery_duplicates", "image_pipeline", "dtype", "tokenizer",
    "caption_join", "text_input_rule", "image_input_rule", "eval_mode", "forward", "fusion",
    "cache", "score_orientation", "score_normalize", "mask", "tie_policy", "rank_definition",
    "multiple_positives", "truncation", "metric_recall", "metric_ndcg", "metric_macro",
    "metric_coverage", "raw_units", "rounding", "missing_vs_unscored_vs_miss", "gap_formula",
    "evidence_path", "notes",
]

def cohort(protocol, cat):
    return cohorts["protocols"][protocol]["categories"][cat]

PAIR_UNION_COUNTS = "dress 2628 / shirt 3089 / toptee 2902"
FULL_COUNTS = "dress 3817 / shirt 6346 / toptee 5373 (union of three = 15415)"

def gsha(protocol):
    return " | ".join(f"{c}:{cohort(protocol, c)['gallery_sha256_ordered']}" for c in ("dress", "shirt", "toptee"))

def qsha(protocol):
    return " | ".join(f"{c}:{cohort(protocol, c)['query_identity_sha256']}" for c in ("dress", "shirt", "toptee"))

PAIR = "pair union (first-seen ref then target) over cap.<cat>.val.json"
FULL = "full split.<cat>.val.json in file order"

rows = []

def native(model_id, recorded, validated, verdict, repo, commit, ck, arch, noise, interp, py, torch,
           prec, seed, gallery_kind, gallery_count, gallery_source, ref_policy, img, dtype, tok,
           join, eval_mode, fwd, score_orient, score_norm, mask, rank, metric, units,
           evidence, notes, patch="none", extra=None, kind="native_config"):
    r = {c: "" for c in COLS}
    r.update({
        "row_id": f"native:{model_id}", "row_kind": kind, "model_id": model_id,
        "protocol_label_recorded": recorded, "protocol_label_validated": validated,
        "label_verdict": verdict, "upstream_repo": repo, "upstream_commit": commit,
        "vendor_patch": patch, "checkpoint_id": ck[0], "checkpoint_sha256": ck[1],
        "checkpoint_resolution": ck[2], "architecture_config": arch, "noise_transfer": noise,
        "interpreter": interp, "python": py, "torch": torch, "precision_device": prec,
        "seed_recorded": seed, "dataset_release": "FashionIQ validation (local data/FashionIQ)",
        "split": "val", "category_scope": "dress, shirt, toptee (separate gallery per category)",
        "query_count": "2017 / 2038 / 1961 = 6016",
        "query_membership_sha256": qsha("fashioniq_val_split"),
        "gallery_kind": gallery_kind, "gallery_count": gallery_count,
        "gallery_sha256": gsha("fashioniq_val_split" if "pair union" in gallery_kind else "fashioniq_original_split"),
        "gallery_order_source": gallery_source,
        "category_filtered_gallery": "yes - each category ranked against its own split file",
        "reference_policy": ref_policy, "gallery_duplicates": "0",
        "image_pipeline": img, "dtype": dtype, "tokenizer": tok, "caption_join": join,
        "text_input_rule": "two captions, correction dict applied where the family declares one",
        "image_input_rule": "reference image only (no target image at eval)",
        "eval_mode": eval_mode, "forward": fwd, "fusion": "see architecture_config",
        "cache": "none for the native evaluator; features computed in-process",
        "score_orientation": score_orient, "score_normalize": score_norm, "mask": mask,
        "tie_policy": "np.argsort(-sims) default (quicksort, not stable) - ties unresolved by upstream",
        "rank_definition": "1 + #{scores greater than target}; ties may order either way",
        "multiple_positives": "1 positive per query in both galleries (assert in upstream evaluator where present)",
        "truncation": f"top-50 window ({rank}); no score truncation beyond that", "metric_recall": metric, "metric_ndcg": "none in upstream evaluator",
        "metric_macro": "unweighted mean of the three per-category values (workbench macro, cited line)",
        "metric_coverage": "all 6016 queries; coverage 1.0",
        "raw_units": units, "rounding": "none - rounding belongs to the UI only",
        "missing_vs_unscored_vs_miss": "upstream: a missing asset raises/crashes, so a miss is a real rank>k "
                                         "and no query is 'unscored'; workbench replay records parity_mismatches separately",
        "gap_formula": "n/a - native evaluator has no composition gap",
        "evidence_path": evidence, "notes": notes,
    })
    if extra:
        r.update(extra)
    rows.append(r)

# ---------------- 15 active native configs ----------------
native("airknow", "fashioniq_full_gallery_ref_excluded", "fashioniq_full_gallery_ref_excluded", "CONFIRMED",
       "https://github.com/iLearn-Lab/CVPR26-Air-Know", "6ab6fa95b746f85b08eea4cb66eb51448bce99b7",
       ("fiq_n05", hashes["airknow"]["sha256"], "workbench.backend.registry.checkpoint_path -> " + hashes["airknow"]["path"]),
       "BLIP-2 Q-Former CIR; EKI set built by datasets.FashionIQ", "training noise 50% (fiq_n05); eval 0%",
       "NOT_RUN - no executed run", "3.8.10 (registry); no interpreter pinned on this host", "2.1.0 (registry)",
       "NOT_MEASURED", "NOT_RECORDED in artifact (no run)",
       "full split", FULL_COUNTS, FULL, "reference masked with -10e10 before argsort",
       "targetpad 1.25 / 224 (BICUBIC resize, center crop, CLIP mean/std)",
       "float32", "LAVIS blip_caption + BertTokenizerWrapper max_length=32 (model side)",
       '"{} and {}".format(correct_text(cap0), correct_text(cap1))',
       "model.train() at native test entry (test_BLIP2.py:10)", "BLIP-2 Q-Former fusion + EKI",
       "higher cosine better", "each side L2-normalized",
       "sims[i, ref_index] = -10e10 at test_BLIP2.py:151",
       "nn_result = argsort(-sims)[:50]",
        "r = 100 * r / len(nn_result) -> PERCENT", "percent",
       "workbench/docs/reviews/20261109-native15-composition-limn/NATIVE_EVAL_MATRIX.csv (airknow row)",
       "No run in this campaign: registry outcomes are source-verified only; G1/G2 NOT_RUN.",
       patch="2 local_fixes in provenance.json (data_utils.py, lavis/models/__init__.py)")

native("conesep", "fashioniq_full_gallery_ref_excluded", "fashioniq_full_gallery_ref_excluded", "CONFIRMED",
       "https://github.com/Lee-zixu/ConeSep", "bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b",
       ("fiq_n02", hashes["conesep"]["sha256"], "checkpoint_path -> " + hashes["conesep"]["path"]),
       "BLIP-2 Q-Former CIR (ConeSep)", "training noise 20%; eval 0%",
       "workbench/envs/ilearn39-adapted/bin/python", "3.9.25 (measured 3.9.25 in campaign meta)",
       "2.7.1+cu128", "float32, cuda", "NOT_RECORDED in raw/native/conesep.json",
       "full split", FULL_COUNTS, FULL, "reference masked -10e10 before argsort",
       "targetpad 1.25 / 224", "float32", "LAVIS blip_caption, max_words=50",
       'literal " and " after correct_text', "model.eval() (test.py:10)", "BLIP-2 Q-Former",
       "higher cosine better", "L2 normalized", "sims[i, ref] = -10e10 at test.py:68",
       "top-50 window", "100 * hits / n -> PERCENT", "percent",
       "raw/native/conesep.json + conesep.meta.txt",
       "macro R@10 54.7613 / R@50 75.2151 (percent). complete=true, parity_mismatches=[].")

native("ptha_mtst", "fashioniq_original_split", "fashioniq_original_split", "CONFIRMED",
       "https://github.com/zyinan99/MTST_PTHA", "6a02ee4763fc2260b6abfb18be78853facb91d86",
       ("cirr_transfer", hashes["ptha_mtst"]["sha256"], "checkpoint_path -> " + hashes["ptha_mtst"]["path"]),
       "MTST+PTHA BLIP-2 transfer; train dataset CIRR", "checkpoint trained on CIRR; eval noise 0%",
       "workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python", "3.10 (adapted)", "2.7.1+cu128",
       "float32, cuda", "NOT_RECORDED",
       "full split + exists() filter", FULL_COUNTS + "; 0 dropped locally (all 15536 .png present)", FULL,
       "reference eligible (no mask on the FashionIQ path)",
       "targetpad 1.25 / 224", "float32", "LAVIS blip_caption + BERT max_length=32",
       'strip/capitalize first + " and " + second', "eval (blip_validate)", "BLIP-2 Q-Former",
       "lower distance better (1 - cos)", "L2 normalized", "none",
       "top-50 window", "100 * hits / n -> PERCENT", "percent",
       "raw/native/ptha_mtst.result.json",
       "Label 'exploratory transfer': the reported number is cross-dataset transfer, not a reproduction.",
       patch="3 local_fixes in provenance.json (gitattributes, data_utils.py base_path, lavis guard)")

native("habit", "fashioniq_full_gallery_ref_excluded", "fashioniq_full_gallery_ref_excluded", "CONFIRMED",
       "https://github.com/iLearn-Lab/AAAI26-HABIT", "6ddc2cd63eac1f65693f4e821c34fb3b42aee306",
       ("fiq_n02", hashes["habit"]["sha256"], "checkpoint_path -> " + hashes["habit"]["path"]),
       "BLIP-2 Q-Former CIR (HABIT)", "training noise 20%; eval 0%",
       "workbench/envs/ilearn39-adapted/bin/python", "3.9", "2.7.1+cu128", "float32, cuda",
       "NOT_RECORDED", "full split", FULL_COUNTS, FULL, "reference masked -10e10",
       "targetpad 1.25 / 224", "float32", "LAVIS blip_caption max_words=50", 'literal " and "',
       "model.eval()", "BLIP-2 Q-Former", "higher cosine better", "L2 normalized",
       "sims[i, ref] = -10e10 at test.py:151", "top-50", "100 * hits / n -> PERCENT", "percent",
       "raw/native/habit.json", "macro R@10 54.8733 / R@50 74.5034 (percent).")

native("intent", "fashioniq_full_gallery_ref_excluded", "fashioniq_full_gallery_ref_excluded", "CONFIRMED",
       "https://github.com/iLearn-Lab/CVPR25-INTENT", "2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3",
       ("fiq_n02", hashes["intent"]["sha256"], "checkpoint_path -> " + hashes["intent"]["path"]),
       "BLIP-2 Q-Former CIR (INTENT)", "training noise 20%; eval 0%",
       "workbench/envs/ilearn39-adapted/bin/python", "3.9", "2.7.1+cu128", "float32, cuda",
       "NOT_RECORDED", "full split", FULL_COUNTS, FULL, "reference masked -10e10",
       "targetpad 1.25 / 224", "float32", "LAVIS blip_caption max_words=50", 'literal " and "',
       "model.eval()", "BLIP-2 Q-Former", "higher cosine better", "L2 normalized",
       "sims[i, ref] = -10e10 at test.py:150", "top-50", "100 * hits / n -> PERCENT", "percent",
       "raw/native/intent.json", "macro R@10 54.8662 / R@50 75.4319 (percent).")

native("hint", "fashioniq_val_split", "fashioniq_val_split", "CONFIRMED",
       "https://github.com/iLearn-Lab/TPAMI26-HINT", "bec50b6c8c19111893b617979502b948d1cea5b2",
       ("fashioniq", hashes["hint"]["sha256"], "checkpoint_path -> " + hashes["hint"]["path"]),
       "BLIP-2 Q-Former CIR (HINT)", "training noise not declared; eval 0%",
       "workbench/envs/ilearn39-adapted/bin/python", "3.9", "2.7.1+cu128", "float32, cuda",
       "NOT_RECORDED", "pair union", PAIR_UNION_COUNTS, PAIR, "reference masked -10e10",
       "targetpad 1.25 / 224", "float32", "LAVIS blip_caption max_words=50", 'literal " and "',
       "model.eval() (test.py:12)", "BLIP-2 Q-Former", "higher cosine better", "L2 normalized",
       "sims[i, ref] = -10e10 at test.py:83", "top-50", "100 * hits / n -> PERCENT", "percent",
       "raw/native/hint.json", "macro R@10 59.8034 / R@50 80.1308 (percent).",
       extra={"gallery_count": PAIR_UNION_COUNTS + " (recomputed; identical order to upstream union, see raw/membership.json)"})

native("encoder", "fashioniq_val_split", "fashioniq_val_split", "CONFIRMED",
       "https://github.com/iLearn-Lab/AAAI25-ENCODER", "29a2a31d6a56f677bf450c3be7cdaef423fb7018",
       ("fashioniq", hashes["encoder"]["sha256"], "checkpoint_path -> " + hashes["encoder"]["path"]),
       "ViT-B-32 open_clip + relation binding (state_dict, 415 entries)", "training noise not declared; eval 0%",
       "workbench/envs/encoder310-adapted/bin/python", "3.10 (adapted)",
       "2.7.1+cu128 + open_clip_torch 2.24.0", "float32, cuda", "dataset construction seeded (replay/common.py:347) - native run does not record a seed",
       "pair union", PAIR_UNION_COUNTS, PAIR, "reference masked -10e10 at test.py:68",
       "preprocess_train (RandomResizedCrop) for ALL images incl. gallery - see raw/scout/ilearn.txt",
       "float32", "open_clip ViT-B-32 tokenizer", "correct_text + ' and ' + correct_text, correction dict applied",
       "model.eval() (test.py)", "extract_retrieval_compose / extract_retrieval_target",
       "higher cosine better", "both sides L2 normalized", "sims[i, ref] = -10e10 at test.py:68",
       "top-50", "100 * hits / n -> PERCENT", "percent",
       "raw/native/encoder.json + ENCODER_PARITY.vi.md",
       "Transform asymmetry is a REAL upstream property: get_img(stage=0) for every query-side and gallery-side call in the val-split branch uses preprocess_train. G1/G2 recorded PASS / PASS_WITH_DECLARED_FAILURES.",
       patch="2 local_fixes (datasets1.py, evaluate_model.py)")

native("pair", "fashioniq_val_split", "fashioniq_val_split", "CONFIRMED",
       "https://github.com/Drutheseeker/PAIR", "bfb0b98cdb4c44f6b9190baaa84d783a77dd888f",
       ("pair_b1", "c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2",
        "pair_compat exploratory mapping, NOT the registry path"),
       "PAIR ViT-B-32 + MHTransformer", "training noise not declared; eval 0%",
       "NOT_RUN in this campaign", "3.8.10 (registry)", "2.0.0 (registry)", "NOT_MEASURED",
       "NOT_RECORDED", "pair union", PAIR_UNION_COUNTS, PAIR, "reference masked -10e10 (test.py:68)",
       "preprocess_train for query, preprocess_val for gallery (train.py:66-73)",
       "float32", "open_clip ViT-B-32 tokenizer", "correct_text + ' and '", "eval", "extract_retrieval_compose",
       "higher cosine better", "L2 normalized", "-10e10 at test.py:68", "top-50",
       "100 * hits / n -> PERCENT", "percent",
       "workbench/artifacts/checkpoint_selection/20261106T002617+0700/pair_compat/measurements/pair_b1_H_normal.json",
       "Reversed-query semantics are a HYPOTHESIS only (H_rev vs H_normal, EXPLORATORY_PUBLIC_MAPPING). Not a native run; not admitted as acceptance evidence.",
       patch="1 local_fix (models/MHTransformer_rev.py added)",
       extra={"row_kind": "native_config", "protocol_label_validated": "fashioniq_val_split (cohort identical; checkpoint mapping UNVERIFIED)"})

native("clvc_net", "fashioniq_val_split", "fashioniq_val_split", "CONFIRMED",
       "https://github.com/NeuralCollage/CLVC-Net", "bd9b6889489806cb60676597880f91d918b279cb",
       ("author_drive_unresolved", "NOT_HASHED (6 per-category members)", "checkpoint_path -> bundle directory"),
       "CLVC-Net local+global ResNet, 6 members", "training noise not declared; eval 0%",
       "NOT_RUN (no interpreter declared)", "3.7.6 (registry)", "1.6.0 (registry)", "NOT_MEASURED",
       "NOT_RECORDED", "pair union (gallery_all=False)", PAIR_UNION_COUNTS, PAIR,
       "reference masked -10e10 (test.py:135)",
       "Resize(256) -> CenterCrop(224) -> ImageNet norm; ONE transform for query and gallery",
       "float32", "SimpleVocab (ascii, punctuation deleted, specials 0-3)",
       'template "<BOS> {} <AND> {} <EOS>", NO correction dict', "eval", "concat(local, global) features",
       "higher cosine better", "each side L2 normalized before concat", "-10e10 at test.py:135",
       "top-50", "100 * hits / n -> PERCENT", "percent",
       "workbench/docs/reviews/20261109-native15-composition-limn/NATIVE_EVAL_MATRIX.csv (clvc_net row)",
       "BLOCKED_NO_SOURCE_REPLAY; pinned test.py has no CLI, replay shell is workbench-owned.",
       patch="1 local_fix (resnet.py)")

native("dcnet", "fashioniq_full_gallery_ref_excluded", "fashioniq_full_gallery_ref_excluded", "CONFIRMED",
       "https://github.com/HejingYin/DCNet", "68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb",
       ("fashioniq_run_directory", hashes[("dcnet","fashioniq_run_directory")]["sha256"],
        "checkpoint_path -> run directory (config.json + trained_model.pth)"),
       "DCNet 7-expert CE + correction branches", "training noise not declared; eval 0%",
       "workbench/artifacts/environments/dcnet-377/bin/python", "3.7.7", "1.4.0+cpu (CPU run)",
       "float32, cpu", "--seeds '0' on the command line (stdout.log:2), passed where upstream expects a list: EFFECTIVE_SEED_UNPROVEN", "full split (val_trg loader)", FULL_COUNTS, FULL,
       "reference masked in BOTH score matrices (trainer_joint.py:217-218)",
       "images pre-resized to 256 then CenterCrop(224) -> ImageNet norm (no TargetPad)",
       "float32", "GloVe token vectors from cap.<cat>.glove.val.pkl, max_text_words=30",
       "NO text join - precomputed GloVe vectors", "eval", "matmul over 7 expert subspaces",
       "higher inner product better", "none per score; fusion uses log_softmax over gallery",
       "-10e10 at trainer_joint.py:217 and :218", "top-50", "sum(hits)/n -> FRACTION", "fraction 0..1",
       "raw/native/dcnet.json", "macro R@10 0.27930 / R@50 0.53761 FRACTION == 27.93 / 53.76 percent. Do not compare the raw field with a percent field.",
       patch="1 approved patch (dcnet_checkpoint_weights_only_optout.patch)")

native("combiner_rn50x4_noft", "fashioniq_original_split", "fashioniq_original_split", "CONFIRMED",
       "https://github.com/ABaldrati/CLIP4Cir", "dfed9f748a8a4d05abf613164e06d59b22dbdf13",
       ("rn50x4_noft_unresolved", hashes["combiner_rn50x4_noft"]["sha256"],
        "checkpoint_path -> " + hashes["combiner_rn50x4_noft"]["path"]),
       "CLIP RN50x4 frozen + Combiner(640,2560,5120)", "training noise 0%; eval 0%",
       "workbench/artifacts/environments/clip4cir-38/bin/python", "3.8.20", "1.11.0 (cpu run)",
       "float32, cpu", "NOT_RECORDED", "full split", FULL_COUNTS, FULL,
       "reference ELIGIBLE - no mask on the FashionIQ path (compute_fiq_val_metrics :21-57)",
       "targetpad 1.25 / input_resolution 288", "float32",
       "openai clip.tokenize context_length=77", 'cap0.strip(".?, ").capitalize() + " and " + cap1.strip(".?, ")',
       "eval (clip_model.float().eval())", "combiner(reference image features, text features)",
       "lower distance better: distances = 1 - cos", "gallery and composed query L2 normalized",
       "none", "top-50 window", "(sum labels[:k] / n) * 100 -> PERCENT", "percent",
       "workbench/artifacts/reports/combiner_rn50x4_noft_..._020a70b0_aggregate.json",
       "Reported r10 35.1078 / r50 58.4756 (percent) from a pre-campaign run, re-verified not re-run.")

native("clip4cir_rn50x4_fullft", "fashioniq_full_gallery_ref_excluded", "fashioniq_original_split", "MISLABELED",
       "https://github.com/ABaldrati/CLIP4Cir", "dfed9f748a8a4d05abf613164e06d59b22dbdf13",
       ("rn50x4_fullft_unresolved",
        " + ".join(f"{m['member']}={m['sha256']}" for m in hashes[("clip4cir_rn50x4_fullft", "bundle_members")]),
        "checkpoint_path -> bundle directory (2 members)"),
       "CLIP RN50x4 fine-tuned (669 tensors) + Combiner(640,2560,5120)", "training noise 0%; eval 0%",
       "workbench/artifacts/environments/clip4cir-38/bin/python", "3.8.20", "1.11.0+cpu (CPU run)",
       "float32, cpu", "NOT_RECORDED", "full split", FULL_COUNTS, FULL,
       "reference ELIGIBLE - the FashionIQ branch has no mask; the only mask in validate.py is CIRR-only",
       "targetpad 1.25 / input_resolution 288", "float32",
       "openai clip.tokenize context_length=77", 'cap0.strip(".?, ").capitalize() + " and " + cap1.strip(".?, ")',
       "eval (clip_model.float().eval())", "combiner(reference image features, text features)",
       "lower distance better: distances = 1 - cos", "both sides L2 normalized", "none",
       "top-50 window", "(sum labels[:k] / n) * 100 -> PERCENT", "percent",
       "raw/native/clip4cir_fullft.json + .stdout.log",
       "R1: recorded label is wrong. Evidence: third_party/CLIP4Cir/src/validate.py:21-57 has no reference write; the only reference_mask is :162-166 inside compute_cirr_val_metrics. Registry agrees (models.yaml:1184 fashioniq_original_split). Metrics themselves are unaffected by the label: the cohort (3817/6346/5373) is unchanged; only the exclusion rule differs.")

native("tgcir", "fashioniq_val_split", "fashioniq_val_split", "CONFIRMED",
       "https://github.com/StarRainArt/TG-CIR", "65fa78eaf8cabe8197fcc22ccab42207c591bdee",
       ("tgcir_archive_unresolved", "c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966",
        "checkpoint_path -> " + "downloads/tgcir/x/checkpoints/Fashioniq.pt"),
       "TG-CIR ViT-B/16 + clip.tokenize", "training noise not declared; eval 0%",
       "NOT_RUN (PREFLIGHT_ONLY)", "UNKNOWN", "1.7.0 (registry)", "NOT_MEASURED", "NOT_RECORDED",
       "pair union", PAIR_UNION_COUNTS, PAIR, "reference masked -10e10 (test.py:67)",
       "OpenAI CLIP preprocess for ViT-B/16; one transform for query and gallery", "float32",
       "clip.tokenize (OpenAI CLIP)", "correct_text + ' and ' + correct_text", "eval",
       "extract_retrieval_compose", "higher cosine better", "L2 normalized",
       "-10e10 at test.py:67", "top-50", "100 * hits / n -> PERCENT", "percent",
       "raw/native/* (absent) + registry models.yaml:1275",
       "No native run: replay is preflight-only and the pinned test.py has no __main__.")

native("limn", "fashioniq_val_split", "fashioniq_val_split", "CONFIRMED",
       "https://github.com/BITT-based-LIMN/LIMN", "7d7bc9b116f594a65ac22457491edf28a88d3c3e",
       ("base_iter0_all_categories",
        hashes[("limn", "base_iter0_dress")]["sha256"] + " | shirt " + hashes[("limn", "base_iter0_shirt")]["sha256"] +
        " | toptee " + hashes[("limn", "base_iter0_toptee")]["sha256"],
        "checkpoint_path per category -> cohort/downloads/limn/"),
       "ViT-L-14 open_clip + LIMN base/iter-0 (whole model per category)", "training noise not declared; eval 0%",
       "workbench/envs/limn310-adapted/bin/python", "3.10 (adapted)",
       "2.7.1+cu128 + open_clip_torch 2.20.0", "float32, cuda", "seeded dataset construction (replay/common.py:347)",
       "pair union", PAIR_UNION_COUNTS, PAIR, "reference masked -10e10 for name != 'birds' (LIMN/test.py:82)",
       "preprocess_val (transform[1]) for query AND gallery images", "float32",
       "open_clip ViT-L-14 tokenizer", "correct_text + ' and ' + correct_text, correction dict applied",
       "eval", "compose_query / compose_target", "higher cosine better", "L2 normalized",
       "-10e10 at LIMN/test.py:82", "top-50", "100 * hits / n -> PERCENT", "percent",
       "limn_native.json + limn_native.{cat}.json",
       "macro R@1 23.3394 / R@10 55.7919 / R@50 77.6378 (percent). Instrumentation parity VERIFIED against the native aggregate.")

native("tme", "fashioniq_original_split", "fashioniq_original_split", "CONFIRMED",
       "https://github.com/li-shuxian/TME", "394aad79df2c104feb32a2c96132582012765062",
       ("fashioniq_0", hashes["tme"]["sha256"], "checkpoint_path -> " + hashes["tme"]["path"]),
       "TME BLIP-2 Q-Former, per-category image list", "training noise 0%; eval 0%",
       "workbench/envs/tme-adapted/bin/python", "3.10", "2.7.1+cu128", "float32, cuda",
       "NOT_RECORDED", "full split, per category", FULL_COUNTS, FULL,
       "reference ELIGIBLE on FashionIQ - the mask is guarded by `if category == 'CIRR'` (precompute_evaluation.py:89)",
       "targetpad 1.25 / 224", "float32", "LAVIS txt_processors['eval']",
       'cap0.strip(".?, ").capitalize() + " and " + cap1.strip(".?, ")', "eval",
       "model.inference -> max over 32 target tokens", "lower distance better (1 - cos)",
       "query and target L2 normalized; max over tokens", "none on the FIQ path",
       "top-50 window", "100 * hits / n -> PERCENT", "percent",
       "raw/native/tme.json", "macro R@10 54.9229 / R@50 74.8384 (percent).",
       patch="2 approved patches (precompute.py bounded allocation, eval batch override)")

# ---------------- deferred but declared ----------------
native("csmcir", "fashioniq_original_split", "fashioniq_original_split", "CONFIRMED (deferred: not in active 15)",
       "https://github.com/qzp2018/CSMCIR", "774f94e2076ff17ea91703a6239d2a08f0e1a44e",
       ("fashioniq", "c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e",
        "checkpoint_path -> cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt"),
       "BLIP-2 CSMCIR", "training noise 0%; eval 0%", "workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python",
       "3.10 (adapted)", "2.7.1+cu128", "float32, cuda", "NOT_RECORDED", "full split", FULL_COUNTS, FULL,
       "reference ELIGIBLE (no mask; CIRR-only mask at :266-268)", "targetpad 1.25 / 224", "float32",
       "LAVIS txt_processors['eval']", 'strip/capitalize first + " and " + second', "eval",
       "blip_model.inference max over 32 target tokens", "lower distance better", "L2 normalized", "none",
       "top-50", "100 * hits / n -> PERCENT", "percent",
       "workbench/docs/reviews/20261109-native15-composition-limn/NATIVE_EVAL_MATRIX.csv (csmcir row)",
       "Deferred by the operator; excluded from the 15 active. Protocol row retained for label validation only.")

native("sprc", "fashioniq_original_split", "fashioniq_original_split", "CONFIRMED (deferred: not in active 15)",
       "https://github.com/chunmeifeng/SPRC", "2935a5397732260d1db6fa577e5926f963e36f0f",
       ("sprc_fiq_vitl", "19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08",
        "checkpoint_path -> cohort/downloads/sprc/sprc_fiq_vitl.pt"),
       "SPRC BLIP-2 rerank", "training noise not declared; eval 0%",
       "workbench/artifacts/environments/sprc-39/bin/python", "3.9", "2.0.1 (registry)", "NOT_MEASURED",
       "NOT_RECORDED", "full split", FULL_COUNTS, FULL,
       "reference ELIGIBLE (no mask; CIRR-only mask at :258-260)", "targetpad 1.25 / 224", "float32",
       "LAVIS txt_processors['eval']", 'strip/capitalize first + " and " + second', "eval",
       "blip_model.inference max over tokens", "lower distance better", "L2 normalized", "none",
       "top-50", "100 * hits / n -> PERCENT", "percent",
       "workbench/docs/reviews/20261109-native15-composition-limn/NATIVE_EVAL_MATRIX.csv (sprc row)",
       "Deferred; no composition adapter exists.")

# ---------------- protocol labels ----------------
def label_row(pid, lit, gallery, ref, query_count, cohort_hash, accepted, notes, evidence):
    r = {c: "" for c in COLS}
    r.update({
        "row_id": f"protocol:{pid}", "row_kind": "protocol_label", "model_id": "",
        "protocol_label_recorded": pid, "protocol_label_validated": pid,
        "label_verdict": accepted, "dataset_release": "FashionIQ validation (local data/FashionIQ)",
        "split": "val", "category_scope": "dress, shirt, toptee",
        "query_count": query_count, "query_membership_sha256": cohort_hash,
        "gallery_kind": gallery, "reference_policy": ref,
        "literature_split_label": lit, "gallery_count": "", "raw_units": "percent",
        "rounding": "none - rounding belongs to the UI only",
        "evidence_path": evidence, "notes": notes,
    })
    r["literature_split_label"] = lit
    rows.append(r)

label_row("fashioniq_original_split", "original",
          "full split.<cat>.val.json in file order", "reference ELIGIBLE (no mask)",
          "6016", qsha("fashioniq_val_split"),
          "CONFIRMED",
          "Registry label 'FashionIQ - Original Split (Full Gallery, Reference Included)'. Gallery 3817/6346/5373. "
          "Aliases accepted evaluator-side: fashioniq_original, fashioniq_val_split_ref_included "
          "(src/evaluation/fashioniq.py:186). The alias is NOT persisted-admissible: PROTOCOL_IDS derives from the "
          "registry (workbench/backend/schemas/results.py:12-16) and RunMetadata.validate_metadata refuses it (:136).",
          "workbench/registry/models.yaml:15-18; src/evaluation/fashioniq.py:172-186")
label_row("fashioniq_full_gallery_ref_excluded", "original",
          "SAME full split order as fashioniq_original_split", "reference MASKED before R@K",
          "6016", qsha("fashioniq_val_split"),
          "CONFIRMED",
          "Identical candidate universe to fashioniq_original_split (same gallery_sha256); only the per-query "
          "reference exclusion differs. Mask and column-removal are equivalent for the top-k hit test whenever all "
          "other scores exceed -10e10 (documented at src/evaluation/fashioniq.py:35-56).",
          "workbench/registry/models.yaml:19-22; src/evaluation/fashioniq.py:187-189")
label_row("fashioniq_val_split", "val",
          "pair union (first-seen ref then target) over cap.<cat>.val.json", "reference MASKED before R@K",
          "6016", qsha("fashioniq_val_split"),
          "CONFIRMED",
          "Gallery 2628/3089/2902. Alias accepted: fashioniq_val (fashioniq.py:192). Rebuilt gallery is "
          "ORDER-IDENTICAL to the upstream union built as images.index(candidate)/index(target) - measured on all "
          "three categories (raw/membership.json + this row's script output).",
          "workbench/registry/models.yaml:23-26; src/evaluation/fashioniq.py:190-192")
label_row("circus_fashioniq_release_v1", "n/a (release protocol; not admitted by RunMetadata)", 
          "released CIRCUS gallery_paths order, 74381 identities", "reference ELIGIBLE by construction (reference is in the gallery and is not the positive)",
          "6003 full / 4069 sf / 586 validated",
          circus["queries"]["full"]["identity_sha256"],
          "CONFIRMED",
          "Gallery 74381 with 0 duplicates and sha256 " + circus["gallery"]["sha256_ordered"] + " (ordered). "
          "Paper states 18853 (data.py:12); the pinned audit artifact holds 74381 (data.py:13) and the loader "
          "refuses a mismatch (data.py:218-224). 9 queries carry more than one positive; 0 have reference == positive. "
          "Category resolution: 5884 resolved, 119 ambiguous (left None, never inferred from target).",
          "workbench/composition/data.py:11-13,196-315; workbench/backend/composition.py:1020-1030")
label_row("fashioniq_val_split_ref_included", "val (as TME once declared)",
          "NOT a cohort: resolves to fashioniq_original_split", "reference ELIGIBLE",
          "6016", qsha("fashioniq_val_split"),
          "CONFIRMED AS ALIAS / DISPROVED AS A DISTINCT PROTOCOL",
          "Declared for TME in an earlier revision; the eponymous split does not exist. It survives only as an "
          "evaluator-side alias. workbench/tests/fixtures/pc_launcher/models.json:1723-1725 still declares it as "
          "TME's native_protocol - a STALE fixture that no longer matches the registry (models.yaml:1669 area).",
          "src/evaluation/fashioniq.py:175-186; workbench/tests/fixtures/pc_launcher/models.json:1723")
label_row("p1", "n/a",
          "n/a - mock fixture", "n/a",
          "n/a", "n/a",
          "DISPROVED as a protocol (it is a mock artifact field)",
          "Found in workbench/artifacts/reports/mock_c1_burn-probe-final_{aggregate,invocation_outcome}.json "
          "with model_id=mock, checkpoint_id=c1. Not a FashionIQ cohort; must never be counted as one. "
          "RunMetadata would refuse it (:136).",
          "workbench/artifacts/reports/mock_c1_burn-probe-final_aggregate.json")

# ---------------- execution paths needing acceptance ----------------
def path_row(pid, name, proto, how, gallery, ref, gallery_count, query_count, hash_, units, notes, evidence,
             kind="execution_path"):
    r = {c: "" for c in COLS}
    r.update({
        "row_id": pid, "row_kind": kind, "model_id": name,
        "protocol_label_recorded": proto, "protocol_label_validated": proto,
        "label_verdict": ("CONFIRMED" if proto in (
            "fashioniq_original_split", "fashioniq_full_gallery_ref_excluded", "fashioniq_val_split",
            "circus_fashioniq_release_v1") else "N/A (a path names cohorts, not one label)"),
        "dataset_release": "FashionIQ validation" if proto != "circus_fashioniq_release_v1" else "CIRCUS FashionIQ release v1",
        "split": "val", "category_scope": "dress, shirt, toptee" if proto != "circus_fashioniq_release_v1" else "resolved by reference identity (5884) or adapter scope",
        "query_count": query_count, "query_membership_sha256": hash_,
        "gallery_kind": gallery, "gallery_count": gallery_count, "reference_policy": ref,
        "score_orientation": "higher is better (cosine / normalized dot)",
        "tie_policy": "descending score, ties broken by earlier gallery position (workbench/composition/topk.py:39,98,101)",
        "rank_definition": "1 + #greater + #earlier-equal (workbench/composition/adapters.py:371-384)",
        "metric_recall": "100 * hits / scored (workbench/composition/metrics.py:73-74)",
        "metric_ndcg": "1/log2(rank+1) per query, mean over scored (metrics.py:27-30,75)",
        "metric_macro": "n/a - cohort is the whole release subset, not a category macro",
        "metric_coverage": "scored_count / query_count (metrics.py:72)",
        "raw_units": units, "rounding": "none - rounding belongs to the UI only",
        "missing_vs_unscored_vs_miss": "three states kept apart: unscored (no exact rank: pipeline/mapping fact), "
                                        "retrieval_failure (rank > K), hit (rank <= K); see backend/composition.py:1181-1232 "
                                        "and metrics.py:168-172",
        "gap_formula": "1 - max(nDCG_text, nDCG_image) / nDCG_multimodal on the cohort scored in ALL THREE conditions "
                       "(metrics.py:126-148); gap_cohort_count is that intersection",
        "evidence_path": evidence, "notes": notes,
    })
    rows.append(r)

path_row("path:workbench-evaluator", "src/evaluation/fashioniq.py (library cohort resolver)",
         "all three registry ids + evaluator aliases", "resolve_fashioniq_protocol -> build_fashioniq_gallery -> recall",
         "per protocol (full split or pair union)", "per protocol",
         FULL_COUNTS + " | " + PAIR_UNION_COUNTS, "6016", qsha("fashioniq_val_split"),
         "percent", "Unknown protocol names raise; there is no default (:202-213). Reference policy comes from the "
                    "protocol, never from the caller (:300-310).", "src/evaluation/fashioniq.py:155-232,269-356")
path_row("path:native-replay-shell", "workbench/replay/*.py (per-model invocation shells)",
         "each model's PROTOCOL constant", "preflight -> load checkpoint -> call pinned test()",
         "per protocol", "per protocol", FULL_COUNTS + " | " + PAIR_UNION_COUNTS, "6016",
         qsha("fashioniq_val_split"), "percent where the upstream multiplies by 100, fraction for DCNet",
         "The shell is workbench-owned code: it must not re-implement ranking. LIMN/ENCODER/LM etc. call the "
         "pinned function directly.", "workbench/replay/{hint,encoder,limn,dcnet,tme}.py:1-60")
path_row("path:composition-adapter", "workbench/composition (9 adapters, CIRCUS conditions)",
         "circus_fashioniq_release_v1", "PIL intervention -> adapter-native transform -> one score() call -> rank+topk",
         "released gallery_paths order", "eligible by construction", "74381",
         "6003 full / 4069 sf / 586 validated", circus["queries"]["full"]["identity_sha256"],
         "recall percent, nDCG/MRR fraction",
         "One scoring call yields both the rank and the stored window (adapters.py:392-409). DCNet refuses "
         "(`ScoringNotReproducible`) rather than faking a fixed matrix (model_dcnet.py:145-155). "
         "mask_scope: adapters do NOT inherit the native reference mask (model_encoder.py:227, model_dcnet.py:96).",
         "workbench/composition/{data,inputs,adapters,topk,metrics,runner}.py")

# instrumentation rows for each measured acceptance artifact
for tag, proto, note in [
    ("native:hint", "fashioniq_val_split", "raw/native/hint.json"),
    ("native:encoder", "fashioniq_val_split", "raw/native/encoder.json"),
]:
    pass

OUT = BASE / "PROTOCOL_MATRIX.csv"
with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=COLS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print(f"wrote {OUT} rows={len(rows)} cols={len(COLS)}")
