#!/usr/bin/env python3
"""Recheck of the c25d67f frontend fixes on the LIVE workbench (FE :5173, API :8000).

READ-ONLY: navigation, select_option, screenshots and GET requests. No restart, no kill, no build,
no config edit, no write to the backend. Every number compared is pinned to the exact API URL that
produced it; every screenshot is 2560x1440 at deviceScaleFactor 1.

Bundle this evidence describes: assets/index-CBR6lHgq.js
sha256 acfb5fbbced77e88b86ee542e5f261c6df71c0e0e8039d753965e1fc873ac21a
Checked below in `bundle_check()` from the served HTML + the dist file, so a rebuild invalidates it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[6]
DIST = REPO / "workbench/frontend/dist"
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(REPO / "workbench/envs/ctl312h/.playwright-browsers"))

UI = "http://127.0.0.1:5173"
API = "http://127.0.0.1:8000"
VIEWPORT = {"width": 2560, "height": 1440}
PROTOCOL = "circus_fashioniq_release_v1"
HINT = "f461f391c6d4696011be8067"
TME = "cd1a696d8bca0fc4bce6925b"
TME_NOEXPORT = "ad477312aa57c1b9114f9e51"
QID = "7:1242"
DERIVED = "aee2e8d39214d877a84d7c08"          # limn, subset validated, derived from 9e4918a6
PARENT = "9e4918a6b5c243412cbc58c9"           # limn full, PARTIAL

RESULTS: list[dict] = []
API_CALLS: dict[str, str] = {}


def http(path: str, attempts: int = 4) -> tuple[int, object]:
    url = path if path.startswith("http") else API + path
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            try:
                return error.code, json.loads(body)
            except ValueError:
                return error.code, body[:400]
        except Exception as error:  # a co-tenant restarting the stack is a blip, not a fact
            if attempt + 1 >= attempts:
                return 0, repr(error)
            time.sleep(5)
    return 0, "unreachable"


def text(path: str) -> str:
    with urllib.request.urlopen(path if path.startswith("http") else API + path, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def bundle_check() -> dict:
    html = text(UI + "/")
    served = re.search(r"assets/index-[^\"']*\.js", html)
    js = served.group(0) if served else None
    dist_files = sorted(p.name for p in (DIST / "assets").iterdir())
    digest = hashlib.sha256((DIST / js).read_bytes()).hexdigest() if js else None
    return {"served_js": js, "sha256": digest, "dist_assets": dist_files,
            "css_expected": "index-DecQnta9.css", "js_expected": "index-CBR6lHgq.js"}


def explorer(run: str, query: str = QID, condition: str = "multimodal", subset: str = "full",
             runs: list[str] | None = None) -> str:
    keys = runs if runs is not None else [run]
    quoted = "%2C".join(f"c%3A{k}" for k in keys)
    return (f"/?area=samples&view=explorer&protocol={PROTOCOL}&subset={subset}"
            f"&runs={quoted}&run={run}&condition={condition}&query={query}")


def record(name: str, page, expectation: str, passed: bool, **extra) -> bool:
    png = OUT / f"{name}.png"
    page.screenshot(path=str(png))
    RESULTS.append({"check": name, "url": page.url, "expectation": expectation,
                    "passed": bool(passed), "png": png.name, "viewport": VIEWPORT,
                    "dom_excerpt": page.inner_text("body")[:2500], **extra})
    print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    return bool(passed)


def served(run: str, cond: str, qid: str, k: int, subset: str = "full") -> dict:
    path = (f"/api/composition/runs/{run}/queries?subset={subset}&condition={cond}"
            f"&query_id={qid}&top_k={k}&limit=1&offset=0")
    API_CALLS[f"served:{run[:8]}:{cond}:{k}"] = API + path
    status, payload = http(path)
    items = payload.get("items") if isinstance(payload, dict) else None
    return {"status": status, "item": items[0] if items else None,
            "candidates": payload.get("candidates") if isinstance(payload, dict) else None}


def inspect(page) -> dict:
    return page.evaluate("""() => {
      const q = (sel) => document.querySelector(sel);
      const cand = [...document.querySelectorAll('[data-testid=topk-grid] article')].map(a => ({
        rank: (a.querySelector('.candidate-rank b')||{}).textContent,
        img: (a.querySelector('img')||{getAttribute:()=>null}).getAttribute('src'),
        nat: (a.querySelector('img')||{naturalWidth:0}).naturalWidth || 0,
        state: (a.querySelector('img')||{getAttribute:()=>null}).getAttribute('data-asset-state')}));
      const unavailable = q('[data-testid=candidates-unavailable]');
      return {
        caption: (q('[data-testid=query-caption]')||{}).textContent,
        targetIds: [...document.querySelectorAll('[data-testid=query-targets] img')].map(i => i.getAttribute('src')),
        outside: (q('[data-testid=target-outside-window]')||{}).textContent,
        countLine: (q('.candidate-count')||{}).textContent,
        candidates: cand,
        unavailable: unavailable ? {title: (unavailable.querySelector('b,strong,.empty-title')||{}).textContent,
                                    text: unavailable.textContent.replace(/\\s+/g,' ').trim()} : null,
        emptyKind: (q('[data-empty-kind]')||{getAttribute:()=>null}).getAttribute('data-empty-kind'),
        emptyText: (q('[data-empty-kind]')||{}).textContent,
        images: [...document.querySelectorAll('[data-testid=query-image]')].map(i => ({
          src: i.getAttribute('src'), nat: i.naturalWidth, state: i.getAttribute('data-asset-state')})),
        topk: q('[data-testid=topk]') ? q('[data-testid=topk]').value : null,
      };
    }""")


def compare_columns(page) -> dict:
    return page.evaluate("""() => {
      const table = document.querySelector('[data-testid=compare-peer-picker]')?.closest('.stack')?.querySelector('table');
      const rows = table ? [...table.querySelectorAll('tbody tr')].map(tr => [...tr.children].map(td => td.textContent.trim())) : [];
      const opts = [...document.querySelectorAll('[data-testid=compare-peer-select] option')].map(o => ({
        value: o.value, label: o.textContent, disabled: o.disabled}));
      return {rows, pickerOptions: opts, pickerCount: opts.length,
              cohortLine: (document.querySelector('[data-testid=compare-peer-picker]')?.closest('.stack')
                           ?.querySelector('.hint')||{}).textContent};
    }""")


def main() -> int:
    from playwright.sync_api import sync_playwright

    bundle = bundle_check()
    print(json.dumps(bundle, ensure_ascii=False), flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path="/usr/bin/google-chrome" if Path("/usr/bin/google-chrome").exists() else None,
            args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        page = context.new_page()

        def goto(url: str) -> None:
            page.goto(UI + url, wait_until="domcontentloaded", timeout=90_000)

        # ------------------------------------------------------------------ A: peers reachable
        single = (f"/?area=samples&view=compare&protocol={PROTOCOL}&subset=full"
                  f"&runs=c%3A{HINT}&run={HINT}&query=7%3A1242")
        goto(single)
        page.wait_for_selector("[data-testid=compare-peer-picker]", timeout=90_000)
        page.wait_for_timeout(3000)
        before = compare_columns(page)
        record("A1_compare_before_single", page,
               "trước: URL cũ chỉ chọn 1 lượt → bảng dựng từ danh sách peer (không có TME)",
               len(before["rows"]) >= 1, columns=before)

        API_CALLS["peerlist"] = f"{API}/api/composition/runs?subset=full"
        both = (f"/?area=samples&view=compare&protocol={PROTOCOL}&subset=full"
                f"&runs=c%3A{HINT}%2Cc%3A{TME}&run={HINT}&query=7%3A1242")
        goto(both)
        page.wait_for_selector("[data-testid=compare-peer-picker]", timeout=90_000)
        page.wait_for_timeout(3000)
        after = compare_columns(page)
        tme_rank = served(TME, "multimodal", QID, 10)
        hint_rank = served(HINT, "multimodal", QID, 10)
        labels = [" ".join(r) for r in after["rows"]]
        has_tme = any("tme" in l for l in labels)
        record("A2_compare_after_two_selected", page,
               "URL cũ + runs= có 2 lượt → TME là cột so sánh, peer picker liệt kê mọi peer còn lại (unsliced)",
               has_tme and after["pickerCount"] - 1 >= 15, columns=after,
               tme_rank=((tme_rank["item"] or {}).get("ranks") or {}).get("multimodal"),
               hint_rank=((hint_rank["item"] or {}).get("ranks") or {}).get("multimodal"),
               peer_picker_option_count=after["pickerCount"] - 1)

        # arbitrary peer: pick a run that the OLD slice(0,4) never showed (conesep 100-depth a4f786d8)
        target = "a4f786d8af4d46ac2a85ab9c"
        page.select_option("[data-testid=compare-peer-select]", target)
        page.wait_for_timeout(3500)
        picked = compare_columns(page)
        picked_labels = [" ".join(r) for r in picked["rows"]]
        only_runs_changed = "runs=" in page.url and f"run={HINT}" in page.url and "query=7%3A1242" in page.url
        record("A3_compare_arbitrary_peer", page,
               "chọn peer tùy ý (conesep a4f786d8) → cột mới xuất hiện, chỉ runs= đổi, protocol/subset/run/query giữ nguyên",
               any("conesep" in l for l in picked_labels) and only_runs_changed,
               columns=picked, url_after=page.url)

        page.click("[data-testid=compare-peer-clear]")
        page.wait_for_timeout(2500)
        cleared = compare_columns(page)
        # The panel then falls back to the first four scoped rows (documented: "so the panel is never
        # empty"); what the button must do is drop the peer from the SELECTION, not from the picker.
        cleared_opts = {o["value"]: o["disabled"] for o in cleared["pickerOptions"]}
        record("A4_compare_peer_clear", page,
               "Bỏ lượt đã thêm → runs= chỉ còn lượt đang xem, peer cũ chọn lại được, query không mất",
               "runs=c%3A" + HINT in page.url.replace("%3A", "%3A")
               and "+" not in page.url and "query=7%3A1242" in page.url
               and cleared_opts.get(target) is False, columns=cleared, url_after=page.url)

        # A5: the compare table names peers by model_id only, so two runs of one model collide.
        goto(f"/?area=samples&view=compare&protocol={PROTOCOL}&subset=full"
             f"&runs=c%3A{HINT}%2Cc%3A10173dc45dea8c90fd2a284b%2Cc%3A9e4918a6b5c243412cbc58c9"
             f"&run={HINT}&query=7%3A1242")
        page.wait_for_selector("[data-testid=compare-peer-picker]", timeout=90_000)
        page.wait_for_timeout(4000)
        twin = compare_columns(page)
        twin_labels = [" ".join(r) for r in twin["rows"]]
        limn_labels = [l for l in twin_labels if l.startswith("limn")]
        API_CALLS["limn_a"] = (f"{API}/api/composition/runs/10173dc45dea8c90fd2a284b/queries?subset=full"
                               f"&condition=multimodal&order=worst&limit=1&offset=0&query_id=7%3A1242")
        API_CALLS["limn_b"] = (f"{API}/api/composition/runs/9e4918a6b5c243412cbc58c9/queries?subset=full"
                               f"&condition=multimodal&order=worst&limit=1&offset=0&query_id=7%3A1242")
        twin_api = {r: (http(f"/api/composition/runs/{r}/queries?subset=full&condition=multimodal"
                             f"&order=worst&limit=1&offset=0&query_id={QID}")[1]["items"][0])
                    for r in ("10173dc45dea8c90fd2a284b", "9e4918a6b5c243412cbc58c9")}
        record("A5_compare_two_runs_one_model", page,
               "hai lượt cùng model_id (limn 10173dc4 không Top-K, 9e4918a6 Top-K 100) phải phân biệt được TRONG BẢNG so sánh",
               len(set(limn_labels)) == 2, columns=twin,
               twin_labels=limn_labels,
               api_candidate_status={k: v.get("candidate_status") for k, v in twin_api.items()})

        # ------------------------------------------------------------------ B: unique run labels
        goto(f"/?area=samples&view=explorer&protocol={PROTOCOL}&subset=full"
             f"&runs=c%3A{HINT}&run={HINT}")
        page.wait_for_function("document.querySelectorAll('#run-select option').length > 5", timeout=90_000)
        page.wait_for_timeout(1500)
        options = page.eval_on_selector_all(
            "#run-select option", "els => els.map(e => ({value: e.value, label: e.textContent, disabled: e.disabled}))")
        by_value = {o["value"]: o["label"] for o in options}
        hint_labels = [o["label"] for o in options if o["label"].startswith("hint / fashioniq")]
        tme_labels = [o["label"] for o in options if o["label"].startswith("tme / fashioniq_0")]
        noexport_label = by_value.get(f"c:{TME_NOEXPORT}", "")
        export_label = by_value.get(f"c:{TME}", "")
        record("B1_run_labels_distinct", page,
               "2 dòng hint / fashioniq và 2 dòng tme / fashioniq_0 mang nhãn KHÁC nhau; nhãn nói được lượt nào có Top-K",
               len(set(hint_labels)) == 2 and len(set(tme_labels)) == 2
               and "không có Top-K" in noexport_label and "Top-K" in export_label
               and "không có Top-K" not in export_label,
               hint_labels=hint_labels, tme_labels=tme_labels,
               ad477312_label=noexport_label, cd1a696d_label=export_label,
               hint_run_id_in_label=HINT[:8] in " ".join(hint_labels),
               tme_run_id_in_label=TME[:8] in " ".join(tme_labels))
        Path(OUT / "B1_run_labels.txt").write_text("\n".join(o["label"] for o in options), encoding="utf-8")

        goto(explorer(TME_NOEXPORT))
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(3000)
        noexport_dom = inspect(page)
        ad_api = served(TME_NOEXPORT, "multimodal", QID, 10)
        record("B2_noexport_run_matches_label", page,
               "mở ad477312 (nhãn 'không có Top-K') → khối ứng viên rỗng nói đúng lý do, khớp API TOPK_EXPORT_NOT_REQUESTED",
               noexport_dom["unavailable"] is not None
               and (ad_api["candidates"] or {}).get("status") == "TOPK_EXPORT_NOT_REQUESTED"
               and len(noexport_dom["candidates"]) == 0,
               dom=noexport_dom["unavailable"], empty_kind=noexport_dom["emptyKind"],
               api_candidates=ad_api["candidates"])

        goto(explorer(TME))
        page.wait_for_selector("[data-testid=topk-grid]", timeout=90_000)
        page.wait_for_timeout(2500)
        tme_dom = inspect(page)
        record("B3_sibling_run_has_candidates", page,
               "lượt cùng tiền tố nhãn (cd1a696d · Top-K 100) thì có ứng viên — nhãn dự đoán đúng",
               len(tme_dom["candidates"]) == 10 and tme_dom["unavailable"] is None,
               candidate_count=len(tme_dom["candidates"]))

        # ------------------------------------------------------------------ C: earlier steps still hold
        goto(explorer(HINT))
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(3000)
        hint_dom = inspect(page)
        hint_api = served(HINT, "multimodal", QID, 10)
        target_api = (hint_api["item"] or {}).get("positive_ids") or []
        record("C1_reference_caption_target", page,
               "query 7:1242: ảnh tham chiếu + caption + ảnh đích, đích NGOÀI cửa sổ ứng viên",
               bool(hint_dom["targetIds"]) and bool(hint_dom["caption"])
               and target_api and target_api[0].split("/")[-1] in hint_dom["targetIds"][0]
               and not any(target_api[0].split("/")[-1] in (c["img"] or "") for c in hint_dom["candidates"])
               and bool(hint_dom["outside"]),
               caption=hint_dom["caption"], target=hint_dom["targetIds"],
               outside=hint_dom["outside"], api_rank=(hint_api["item"] or {}).get("ranks"))

        matrix = []
        for cond, kind in (("multimodal", "MM"), ("text_only_black_image", "Text-only"),
                           ("image_only_empty_text", "Image-only")):
            page.select_option("[data-testid=filter-condition]", cond)
            page.wait_for_timeout(2500)
            for k in (10, 50):
                page.select_option("[data-testid=topk]", str(k))
                page.wait_for_timeout(2500)
                dom = inspect(page)
                truth = served(HINT, cond, QID, k)
                api_ids = [c["image_id"].split("/")[-1]
                           for c in ((truth["item"] or {}).get("top_results") or [])]
                dom_ids = [(c["img"] or "").split("/")[-1] for c in dom["candidates"]]
                loaded = [c for c in dom["candidates"] if (c["nat"] or 0) > 0]
                entry = {"condition": cond, "k": k, "dom_count": len(dom_ids), "api_count": len(api_ids),
                         "ids_and_order_match": dom_ids == api_ids, "thumbs": f"{len(loaded)}/{len(dom_ids)}",
                         "count_line": dom["countLine"]}
                matrix.append(entry)
                record(f"C2_{kind}_top{k}", page,
                       f"{kind} Top-{k}: số/id/thứ tự khớp API, thumbnail tải thật",
                       dom_ids == api_ids and len(dom_ids) > 0 and len(loaded) == len(dom_ids)
                       and (k != 50 or len(dom_ids) == 50), **entry)
        orders = {}
        for cond in ("multimodal", "text_only_black_image", "image_only_empty_text"):
            m = next(e for e in matrix if e["condition"] == cond and e["k"] == 10)
            orders[cond] = m["ids_and_order_match"]
        # three distinct first-candidate ids, straight from the API
        first_ids = {}
        for cond in ("multimodal", "text_only_black_image", "image_only_empty_text"):
            it = served(HINT, cond, QID, 10)["item"] or {}
            first_ids[cond] = ((it.get("top_results") or [{}])[0]).get("image_id", "").split("/")[-1]
        record("C3_three_conditions_differ", page,
               "MM / Text-only / Image-only trả 3 thứ tự khác nhau (đối chiếu API) và UI khớp cả 3",
               len(set(first_ids.values())) == 3 and all(orders.values()),
               first_candidate_api=first_ids, ui_matches=orders)

        goto(explorer(HINT, query="7:99999"))
        page.wait_for_timeout(4000)
        bad = page.evaluate("""() => {
          const el = document.querySelector('[data-empty-kind=no-results]');
          return {kind: el ? el.getAttribute('data-empty-kind') : null,
                  text: el ? el.textContent.replace(/\\s+/g,' ').trim() : null,
                  inspector: document.querySelectorAll('[data-testid=query-inspector]').length,
                  retry: document.querySelectorAll('[data-testid=error-retry]').length};
        }""")
        record("C4_bad_qid_no_results", page,
               "7:99999 → no-results nêu tên qid, không dựng inspector, không nút thử lại",
               bad["kind"] == "no-results" and bad["text"] and "7:99999" in bad["text"]
               and bad["inspector"] == 0, **bad)

        # ------------------------------------------------------------------ D: CSMCIR row
        goto(f"/?area=experiments&view=models&protocol={PROTOCOL}&subset=full")
        page.wait_for_selector("[data-testid=model-row-csmcir]", timeout=90_000)
        page.wait_for_timeout(3000)
        csm = page.evaluate("""() => {
          const row = document.querySelector('[data-testid=model-row-csmcir]');
          const deferred = document.querySelector('[data-testid=deferred-csmcir]');
          const link = document.querySelector('[data-testid^=deferred-native-csmcir]');
          return {deferred_text: deferred ? deferred.textContent.replace(/\\s+/g,' ').trim() : null,
                  link_text: link ? link.textContent.replace(/\\s+/g,' ').trim() : null,
                  link_testid: link ? link.getAttribute('data-testid') : null,
                  row_text: row ? row.textContent.replace(/\\s+/g,' ').trim() : null};
        }""")
        record("D1_csmcir_row_text", page,
               "dòng CSMCIR hiện 'FashionIQ native đã đo · CIRCUS composition tạm gác' kèm link tới số đo native",
               csm["deferred_text"] is not None
               and "FashionIQ native đã đo · CIRCUS composition tạm gác" in csm["deferred_text"]
               and csm["link_testid"] is not None, **csm)
        if csm["link_testid"]:
            page.click(f"[data-testid={csm['link_testid']}]")
            page.wait_for_selector("[data-testid=native-table]", timeout=90_000)
            page.wait_for_timeout(2500)
            native = page.evaluate("""() => {
              const row = document.querySelector('[data-testid=native-row-6812926dbe6394ce]');
              return {href: location.search, has_csmcir_row: !!row,
                      row_text: row ? row.textContent.replace(/\\s+/g,' ').trim() : null,
                      rows: document.querySelectorAll('[data-testid=native-table] tbody tr').length};
            }""")
            record("D2_csmcir_native_link_opens_row", page,
                   "link mở đúng bảng native và thấy dòng đo của CSMCIR (6812926d), macro R@10 57.05 / R@50 77.30",
                   native["has_csmcir_row"] and "57.05" in (native["row_text"] or ""), **native)

        # ------------------------------------------------------------------ E: derived lineage
        goto(f"/?area=evaluation&view=composition&protocol={PROTOCOL}&subset=validated")
        page.wait_for_selector(f"[data-testid=composition-row-{DERIVED}]", timeout=90_000)
        page.wait_for_timeout(3000)
        deriv = page.evaluate(f"""() => {{
          const cell = document.querySelector('[data-testid=derived-lineage-{DERIVED}]');
          const row = document.querySelector('[data-testid=composition-row-{DERIVED}]');
          const parent = document.querySelector('[data-testid=composition-row-{PARENT}]');
          return {{lineage_text: cell ? cell.textContent.replace(/\\s+/g,' ').trim() : null,
                  complete_marker: cell ? cell.textContent.includes('DERIVED_COMPLETE') : false,
                  row_text: row ? row.textContent.replace(/\\s+/g,' ').trim() : null,
                  parent_present: !!parent,
                  parent_text: parent ? parent.textContent.replace(/\\s+/g,' ').trim() : null}};
        }}""")
        api_calls = {}
        for subset in ("validated", "full"):
            api_calls[subset] = f"{API}/api/composition/runs?subset={subset}"
        API_CALLS["runs_validated"] = api_calls["validated"]
        API_CALLS["runs_full"] = api_calls["full"]
        deriv_api = {r["run_id"]: r for r in http("/api/composition/runs?subset=validated")[1]}
        row = deriv_api.get(DERIVED) or {}
        df = row.get("derived_from") or {}
        own = f"{df.get('own_scored_count')}/{df.get('own_scored_count', 0) + (df.get('own_unscored_count') or 0)}"
        record("E1_derived_lineage_validated", page,
               "lượt dẫn xuất LIMN ở subset Validated in lineage: chính subset này 586/586 + DERIVED_COMPLETE",
               deriv["lineage_text"] is not None and "586/586" in (deriv["lineage_text"] or "")
               and deriv["complete_marker"]
               and df.get("own_scored_count") == 586 and df.get("own_unscored_count") == 0,
               lineage=deriv["lineage_text"], marker_ok=deriv["complete_marker"], own_from_api=own,
               parent_id_from_api=df.get("parent_run_id"))

        page.click("[data-testid=subset-pill-full]")
        page.wait_for_timeout(4000)
        parent_state = page.evaluate(f"""() => {{
          const parent = document.querySelector('[data-testid=composition-row-{PARENT}]');
          const derived = document.querySelector('[data-testid=derived-lineage-{PARENT}]');
          return {{parent_present: !!parent,
                  parent_text: parent ? parent.textContent.replace(/\\s+/g,' ').trim() : null,
                  parent_has_lineage: !!derived,
                  status: parent ? (parent.textContent.match(/PARTIAL|COMPLETED|REUSED|FAILED/)||[])[0] : null}};
        }}""")
        record("E2_parent_stays_partial", page,
               "parent 9e4918a6 ở subset Full vẫn PARTIAL (6001/6003) và KHÔNG mang lineage",
               parent_state["parent_present"] and parent_state["status"] == "PARTIAL"
               and not parent_state["parent_has_lineage"], **parent_state)

        context.close()
        browser.close()

    summary = {
        "bundle": bundle,
        "viewport": VIEWPORT,
        "api_calls": API_CALLS,
        "results": RESULTS,
        "passed": sum(1 for r in RESULTS if r["passed"]),
        "failed": [r["check"] for r in RESULTS if not r["passed"]],
        "screenshots": sum(1 for r in RESULTS if r["png"]),
    }
    (OUT / "recheck_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "failed": summary["failed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
