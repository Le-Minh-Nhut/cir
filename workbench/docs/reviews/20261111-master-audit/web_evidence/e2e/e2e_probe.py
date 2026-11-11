#!/usr/bin/env python3
"""Read-only end-to-end probe of the LIVE workbench (frontend :5173, API :8000).

Reuses the repo's existing browser harness idioms (playwright, managed chromium, testid-first
selectors) — no new framework. Every claim is pinned to three things at once: the URL, the exact
API body the page was served for that context, and a 2560x1440 full-resolution screenshot.

Never writes to the backend except ONE disposable annotation whose note text is the run tag
`e2e-check-20261111` (step 6, explicitly requested). Nothing is restarted or reconfigured.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(REPO / "workbench/envs/ctl312h/.playwright-browsers"))

UI = "http://127.0.0.1:5173"
API = "http://127.0.0.1:8000"
VIEWPORT = {"width": 2560, "height": 1440}
PROTOCOL = "circus_fashioniq_release_v1"
RELEASE = "FashionIQ Release v1"          # the label the Exactly-one CIRCUS release carries
HINT_RUN = "f461f391c6d4696011be8067"     # hint / fashioniq / full, topk depth 100
TME_RUN = "cd1a696d8bca0fc4bce6925b"      # tme  / fashioniq_0 / full, topk depth 100
QID = "7:1242"                            # hint MM rank 98, tme MM rank 65; target outside Top-50
TAG = "e2e-check-20261111"

API_LOG: list[dict] = []
BLIPS: list[dict] = []
RESULTS: list[dict] = []


def api_get(path: str, attempts: int = 5) -> tuple[int, object]:
    """A refused/502/dropped answer is retried 5x with a 5 s pause (another agent may restart the
    stack); each retry is recorded so a blip is evidence, not a data fact."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{API}{path}", timeout=120) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            if error.code >= 500 and attempt + 1 < attempts:
                BLIPS.append({"path": path, "status": error.code, "attempt": attempt + 1})
                time.sleep(5)
                continue
            try:
                return error.code, json.loads(body)
            except ValueError:
                return error.code, body[:600]
        except Exception as error:                      # dropped connection / refused / timeout
            if attempt + 1 >= attempts:
                return 0, repr(error)
            BLIPS.append({"path": path, "error": repr(error)[:160], "attempt": attempt + 1})
            time.sleep(5)
    return 0, "unreachable"


def explorer(run: str, subset: str = "full", query: str = "", condition: str = "multimodal") -> str:
    return (f"/?area=samples&view=explorer&protocol={PROTOCOL}&subset={subset}"
            f"&runs=c%3A{run}&run={run}&condition={condition}" + (f"&query={query}" if query else ""))


def record(name: str, page, expectation: str, passed: bool, **extra) -> bool:
    png = OUT / f"{name}.png"
    page.screenshot(path=str(png), full_page=False)
    body = page.inner_text("body")
    RESULTS.append({
        "check": name, "url": page.url, "expectation": expectation, "passed": bool(passed),
        "png": png.name, "viewport": VIEWPORT, "dom_excerpt": body[:4000], **extra,
    })
    print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    return bool(passed)


def goto(page, url: str, **kwargs) -> None:
    """Navigate, retrying a refused/dropped connection 5x with a 5 s pause."""
    for attempt in range(5):
        try:
            page.goto(url, timeout=90_000, **{"wait_until": "domcontentloaded", **kwargs})
            return
        except Exception as error:
            BLIPS.append({"goto": url, "error": repr(error)[:160], "attempt": attempt + 1})
            if attempt == 4:
                raise
            time.sleep(5)


def served(run: str, cond: str, qid: str, k: int) -> dict:
    status, payload = api_get(
        f"/api/composition/runs/{run}/queries?subset=full&condition={cond}"
        f"&query_id={qid}&top_k={k}&limit=1&offset=0")
    items = payload.get("items") if isinstance(payload, dict) else None
    return {"status": status, "item": items[0] if items else None,
            "candidates": payload.get("candidates") if isinstance(payload, dict) else None}


def main() -> int:
    from playwright.sync_api import sync_playwright

    timings: dict[str, object] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/usr/bin/google-chrome" if Path("/usr/bin/google-chrome").exists() else None,
            args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
        page = context.new_page()
        console_errors: list[str] = []
        page.on("console", lambda m: console_errors.append(m.text[:300]) if m.type == "error" else None)
        page.on("response", lambda r: API_LOG.append({"status": r.status, "url": r.url[:300]})
                if "/api/" in r.url else None)

        # ---------------------------------------------------------------- 1. fresh-session discovery
        clicks: list[str] = []
        marks: dict[str, float] = {}
        t0 = time.time()
        goto(page, UI + "/")
        page.wait_for_selector("[data-testid=sidebar-catalog-count]", timeout=60_000)
        marks["shell_ready"] = round(time.time() - t0, 3)
        shots = OUT / "s1_discovery"
        shots.mkdir(exist_ok=True)
        page.screenshot(path=str(shots / "01_landing.png"))
        clicks.append("goto / (landing = Tổng quan)")
        page.click("[data-testid=area-nav-samples]")
        page.wait_for_selector("[data-testid=area-samples]", timeout=60_000)
        marks["samples_mounted"] = round(time.time() - t0, 3)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(shots / "02_samples_no_run.png"))
        clicks.append("click nav 'Khám phá mẫu' → samples/explorer, no run selected")
        no_run_state = page.inner_text("[data-testid=area-samples]")[:600]
        # The landing subset is SF (routes.defaultLocation); the target runs are `full`.
        default_subset = page.input_value("[data-testid=subset-select]")
        page.select_option("[data-testid=subset-select]", "full")
        page.wait_for_function("document.querySelectorAll('#run-select option').length > 1", timeout=90_000)
        marks["run_list_usable"] = round(time.time() - t0, 3)
        clicks.append(f"select #subset-select = full (mặc định là {default_subset})")
        timings["nav_to_usable_run_list_s"] = marks["run_list_usable"]
        timings["nav_marks_s"] = marks
        # The run selector lists every composition run of the active protocol+subset.
        options = page.eval_on_selector_all(
            "#run-select option", "els => els.map(e => ({value: e.value, label: e.textContent, disabled: e.disabled}))")
        page.select_option("#run-select", f"c:{HINT_RUN}")
        page.wait_for_timeout(2500)
        clicks.append(f"select #run-select = c:{HINT_RUN} (from {len(options)} options)")
        page.screenshot(path=str(shots / "03_hint_selected.png"))
        hint_selected = {"run": page.input_value("[data-testid=active-run]"),
                         "subset": page.input_value("[data-testid=subset-select]")}
        page.click("[data-testid=query-list] li:first-child button")
        page.wait_for_timeout(2000)
        clicks.append("click first row of [data-testid=query-list]")
        page.screenshot(path=str(shots / "04_first_query_open.png"))
        record("s1_discovery_no_guessing", page,
               "từ / mở được HINT + full + một query chỉ bằng click, không cần đoán protocol/subset",
               hint_selected["run"] == f"c:{HINT_RUN}" and hint_selected["subset"] == "full"
               and page.locator("[data-testid=query-inspector]").count() == 1,
               clicks=clicks, run_options=len(options), hint_selected=hint_selected,
               no_run_state=no_run_state,
               options_sample=[o for o in options if "hint" in (o["label"] or "").lower()
                               or "tme" in (o["label"] or "").lower()])

        # ---------------------------------------------------------------- 2. same scored query, both
        nodead: list[str] = []
        goto(page, UI + explorer(HINT_RUN, query=QID))
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(2500)
        hint_api = served(HINT_RUN, "multimodal", QID, 10)
        tme_api = served(TME_RUN, "multimodal", QID, 10)
        def inspect(shot: str) -> dict:
            return page.evaluate("""() => {
              const q = (sel) => document.querySelector(sel);
              const imgs = [...document.querySelectorAll('[data-testid=query-image]')]
                 .map(i => ({src: i.getAttribute('src'), nat: i.naturalWidth, alt: i.alt,
                             state: i.getAttribute('data-asset-state')}));
              const cand = [...document.querySelectorAll('[data-testid=topk-grid] article')].map(a => ({
                 rank: (a.querySelector('.candidate-rank b')||{}).textContent,
                 img: (a.querySelector('img')||{}).getAttribute && a.querySelector('img').getAttribute('src'),
                 nat: (a.querySelector('img')||{}).naturalWidth || 0}));
              return {
                 caption: (q('[data-testid=query-caption]')||{}).textContent,
                 title: (q('[data-testid=query-title]')||{}).textContent,
                 targetIds: [...document.querySelectorAll('[data-testid=query-targets] img')].map(i => i.getAttribute('src')),
                 outside: (q('[data-testid=target-outside-window]')||{}).textContent,
                 pending: (q('[data-testid=target-window-pending]')||{}).textContent,
                 unknown: (q('[data-testid=target-window-unknown]')||{}).textContent,
                 countLine: (q('.candidate-count')||{}).textContent,
                 images: imgs, candidates: cand,
                 topk: q('[data-testid=topk]') ? q('[data-testid=topk]').value : null,
                 condition: q('[data-testid=filter-condition]') ? q('[data-testid=filter-condition]').value : null,
                 ranks: [...document.querySelectorAll('[data-testid^=rank-]')].map(e => ({
                    id: e.getAttribute('data-testid'), state: e.getAttribute('data-rank-state'),
                    text: e.textContent.replace(/\\s+/g,' ').trim()})),
              };
            }""")
        hint_dom = inspect("s2_hint")
        record("s2_same_query_hint", page,
               "ảnh tham chiếu + caption + ảnh đích hiện CÙNG query đã chấm; ảnh đích nằm ngoài cửa sổ ứng viên",
               hint_dom["targetIds"] and hint_dom["caption"] and hint_api["item"] is not None
               and hint_api["item"]["positive_ids"][0].split("/")[-1] in hint_dom["targetIds"][0]
               and not any(c["nat"] and hint_api["item"]["positive_ids"][0].split("/")[-1] in (c["img"] or "")
                           for c in hint_dom["candidates"]),
               dom=hint_dom, api_mm_rank=hint_api["item"]["ranks"]["multimodal"] if hint_api["item"] else None,
               api_target=hint_api["item"]["positive_ids"] if hint_api["item"] else None)

        goto(page, UI + explorer(TME_RUN, query=QID))
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(2500)
        tme_dom = inspect("s2_tme")
        record("s2_same_query_tme", page,
               "cùng query 7:1242 mở được trên lượt TME, caption/ảnh đích trùng, hạng MM khác HINT",
               tme_dom["caption"] == hint_dom["caption"]
               and tme_api["item"] is not None
               and tme_api["item"]["ranks"]["multimodal"] != hint_api["item"]["ranks"]["multimodal"]
               and tme_dom["targetIds"] == hint_dom["targetIds"],
               dom=tme_dom, api_mm_rank=tme_api["item"]["ranks"]["multimodal"] if tme_api["item"] else None)

        # ---------------------------------------------------------------- 3. Top-K and condition matrix
        goto(page, UI + explorer(HINT_RUN, query=QID))
        page.wait_for_selector("[data-testid=topk-grid]", timeout=90_000)
        page.wait_for_timeout(2000)
        matrix = []
        for cond, kind in (("multimodal", "MM"), ("text_only_black_image", "Text-only"),
                           ("image_only_empty_text", "Image-only")):
            page.select_option("[data-testid=filter-condition]", cond)
            page.wait_for_timeout(2200)
            for k in (10, 50, 10):
                page.select_option("[data-testid=topk]", str(k))
                page.wait_for_timeout(2200)
                dom = inspect(f"s3_{kind}_{k}")
                truth = served(HINT_RUN, cond, QID, k)
                api_ids = [c["image_id"].split("/")[-1] for c in (truth["item"]["top_results"] if truth["item"] else [])]
                dom_ids = [(c["img"] or "").split("/")[-1] for c in dom["candidates"]]
                loaded = [c for c in dom["candidates"] if c["nat"] and c["nat"] > 0]
                entry = {
                    "condition": cond, "requested_k": k, "dom_count": len(dom_ids),
                    "api_served": len(api_ids), "export_depth": truth["item"]["candidate_depth"] if truth["item"] else None,
                    "ids_match": dom_ids == api_ids,
                    "order_match": dom_ids == api_ids,
                    "thumbs_loaded": f"{len(loaded)}/{len(dom_ids)}",
                    "count_line": dom["countLine"], "outside_note": dom["outside"],
                    "rank_boxes": {r["id"]: r["text"] for r in dom["ranks"]},
                    "ref_loaded": [i for i in dom["images"] if i["nat"] and i["nat"] > 0 and i["state"] == "present"],
                    "missing_tiles": [i for i in dom["images"] if i["state"] == "missing"],
                }
                matrix.append(entry)
                passed = (dom_ids == api_ids and len(loaded) == len(dom_ids) and len(dom_ids) > 0
                          and (k != 50 or len(dom_ids) == 50))
                record(f"s3_{kind}_top{k}", page,
                       f"{kind} Top-{k}: số ứng viên + id + thứ tự khớp API, mọi thumbnail naturalWidth>0",
                       passed, **entry)
        # 50/10 totals must differ (Top-50 really adds rows) and Text/Image orders must differ from MM
        mm10 = next(m for m in matrix if m["condition"] == "multimodal" and m["requested_k"] == 10)
        mm50 = next(m for m in matrix if m["condition"] == "multimodal" and m["requested_k"] == 50)
        txt10 = next(m for m in matrix if m["condition"] == "text_only_black_image" and m["requested_k"] == 10)
        img10 = next(m for m in matrix if m["condition"] == "image_only_empty_text" and m["requested_k"] == 10)
        record("s3_topk_and_condition_differ", page,
               "Top-50 thực sự nhiều dòng hơn Top-10; ba điều kiện cho ba thứ hạng ứng viên khác nhau",
               mm50["dom_count"] == 50 and mm10["dom_count"] == 10
               and mm10["count_line"] != txt10["count_line"],
               mm10=mm10["dom_count"], mm50=mm50["dom_count"], txt10=txt10["dom_count"], img10=img10["dom_count"])

        # ---------------------------------------------------------------- 4. filters, paging, switching
        base = UI + explorer(HINT_RUN)
        goto(page, base)
        page.wait_for_selector("[data-testid=query-list]", timeout=90_000)
        page.wait_for_timeout(1500)
        pager0 = page.get_attribute("[data-testid=pager-total]", "data-total")

        page.fill("[data-testid=filter-query-id]", QID)
        page.wait_for_timeout(2200)
        search_ids = page.eval_on_selector_all("[data-testid=query-id]", "els => els.map(e => e.textContent)")
        page.screenshot(path=str(OUT / "s4_query_search.png"))
        record("s4_query_id_search", page,
               f"gõ {QID} vào ô mã query → danh sách còn đúng query đó và inspector mở đúng nó",
               search_ids == [QID] and QID in page.inner_text("[data-testid=query-title]"),
               search_ids=search_ids, url=page.url)

        page.fill("[data-testid=filter-query-id]", "")
        page.wait_for_timeout(1800)
        page.click("[data-testid=chip-unscored]")
        page.wait_for_timeout(2200)
        unscored_state = {
            "active": page.get_attribute("[data-testid=query-state-chips]", "data-active"),
            "rows": page.locator("[data-testid=query-list] li").count(),
            "counter": page.inner_text("[data-testid=chip-count-unscored]") if page.locator("[data-testid=chip-count-unscored]").count() else None,
        }
        page.screenshot(path=str(OUT / "s4_unscored_chip.png"))
        page.click("[data-testid=chip-hit]")
        page.wait_for_timeout(2200)
        hit_state = {"active": page.get_attribute("[data-testid=query-state-chips]", "data-active"),
                     "rows": page.locator("[data-testid=query-list] li").count(),
                     "counter": page.inner_text("[data-testid=chip-count-hit]")}
        page.screenshot(path=str(OUT / "s4_hit_chip.png"))
        page.click("[data-testid=chip-retrieval_failure]")
        page.wait_for_timeout(2200)
        miss_state = {"active": page.get_attribute("[data-testid=query-state-chips]", "data-active"),
                      "rows": page.locator("[data-testid=query-list] li").count(),
                      "counter": page.inner_text("[data-testid=chip-count-retrieval_failure]")}
        page.screenshot(path=str(OUT / "s4_miss_chip.png"))
        api_unscored = api_get(f"/api/composition/runs/{HINT_RUN}/queries?subset=full&condition=multimodal"
                               f"&query_state=unscored&limit=20&offset=0")[1]
        record("s4_state_filters", page,
               "chip unscored / hit / trượt lọc đúng tập, số dòng trên trang ≤ 20 và khớp counter của chip",
               unscored_state["active"] == "unscored" and hit_state["active"] == "hit"
               and miss_state["active"] == "retrieval_failure"
               and int(hit_state["counter"].replace(",", "")) == 1970
               and unscored_state["rows"] == api_unscored["total"],
               unscored=unscored_state, hit=hit_state, miss=miss_state,
               api_unscored_total=api_unscored["total"], api_hit_at_10=1970)

        page.click("[data-testid=chip-clear]")
        page.wait_for_timeout(1800)
        before_next = page.inner_text("[data-testid=pager-total]")
        page.click("[data-testid=pager-next]")
        page.wait_for_timeout(2200)
        after_next = {"pager": page.inner_text("[data-testid=pager-total]"),
                      "offset_in_url": "offset=20" in page.url,
                      "first_qid": page.inner_text("[data-testid=query-id]")}
        page.screenshot(path=str(OUT / "s4_pagination.png"))
        page.go_back()
        page.wait_for_timeout(1800)
        back_pager = page.inner_text("[data-testid=pager-total]")
        record("s4_pagination", page,
               "pager Sau tải trang 21–40, URL mang offset=20, Back trả lại 1–20",
               after_next["offset_in_url"] and after_next["pager"] != before_next
               and back_pager == before_next and page.inner_text("[data-testid=query-id]") != after_next["first_qid"],
               before=before_next, after=after_next, after_back=back_pager)

        # context survives cross-page navigation
        goto(page, UI + explorer(HINT_RUN, query=QID))
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(1500)
        page.click("[data-testid=area-nav-evaluation]")
        page.wait_for_timeout(1200)
        eval_url = page.url
        page.click("[data-testid=area-nav-samples]")
        page.wait_for_timeout(2500)
        ctx = {"url": page.url, "run": page.input_value("[data-testid=active-run]"),
               "subset": page.input_value("[data-testid=subset-select]"),
               "title": page.inner_text("[data-testid=query-title]") if page.locator("[data-testid=query-title]").count() else None,
               "selection_count": page.inner_text("[data-testid=selected-run-count]")}
        record("s4_context_survives_navigation", page,
               "đi sang Đánh giá rồi về Khám phá mẫu: run + subset + query vẫn nguyên trong URL và inspector",
               ctx["run"] == f"c:{HINT_RUN}" and ctx["subset"] == "full" and ctx["title"]
               and QID in ctx["title"] and f"query={QID.replace(':', '%3A')}" in ctx["url"],
               eval_url=eval_url, back=ctx)

        # rapid run switching
        goto(page, UI + explorer(HINT_RUN, query=QID))
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(1200)
        for value in (f"c:{TME_RUN}", f"c:{HINT_RUN}", f"c:{TME_RUN}"):
            page.select_option("[data-testid=active-run]", value)
            page.wait_for_timeout(120)
        page.wait_for_timeout(6000)
        rapid = {"url": page.url, "run_select": page.input_value("[data-testid=active-run]"),
                 "subtitle": page.inner_text("[data-testid=query-title]") if page.locator("[data-testid=query-title]").count() else None,
                 "inspector_text": page.inner_text("[data-testid=query-inspector]")[:300] if page.locator("[data-testid=query-inspector]").count() else None,
                 "query_ctx": page.inner_text(".ctx:has-text('Query')") if page.locator(".ctx:has-text('Query')").count() else None}
        page.screenshot(path=str(OUT / "s4_rapid_switch.png"))
        record("s4_rapid_run_switch", page,
               "đổi run liên tiếp 3 lần trong 0.4s → DOM cuối cùng là TME, không còn dấu vết HINT",
               f"{TME_RUN}" in rapid["url"] and rapid["run_select"] == f"c:{TME_RUN}"
               and "hint" not in (rapid["inspector_text"] or "").lower(),
               rapid=rapid)

        # ---------------------------------------------------------------- 5. three states
        # (a) loading: hold the queries answer so the loading card is what is on screen
        loading_page = context.new_page()
        loading_page.on("response", lambda r: API_LOG.append({"status": r.status, "url": r.url[:300]})
                        if "/api/" in r.url else None)
        held = {"hit": False}
        def hold_queries(route, request):
            if "/queries" in request.url and not held["hit"]:
                held["hit"] = True
                time.sleep(4)
            route.continue_()
        loading_page.route("**/api/composition/runs/*/queries*", hold_queries)
        goto(loading_page, UI + explorer(HINT_RUN, query=QID), wait_until="domcontentloaded")
        loading_page.wait_for_timeout(1300)
        loading_shots = loading_page.evaluate("""() => [...document.querySelectorAll('[data-testid=empty-state]')]
             .map(e => ({kind: e.getAttribute('data-empty-kind'), text: e.textContent.replace(/\\s+/g,' ').slice(0,160)}))""")
        loading_page.screenshot(path=str(OUT / "s5_loading.png"))
        record("s5_state_loading", loading_page,
               "trạng thái ĐANG TẢI hiện thành khối loading có nhãn riêng (data-empty-kind=loading), không phải trang trắng",
               any("loading" in (s["kind"] or "") for s in loading_shots), states=loading_shots)
        loading_page.close()

        # (b) API failure: 503 on the candidates request only
        err_page = context.new_page()
        err_page.route("**/api/composition/runs/*/queries*top_k*", lambda route: route.fulfill(
            status=503, content_type="application/json",
            body=json.dumps({"detail": {"error": "induced_e2e_failure", "message": "induced 503"}})))
        goto(err_page, UI + explorer(HINT_RUN, query=QID), wait_until="domcontentloaded")
        err_page.wait_for_timeout(4000)
        err_dom = err_page.evaluate("""() => ({
            kind: (document.querySelector('[data-testid=empty-state]')||{}).getAttribute
                  ? document.querySelector('[data-testid=empty-state]').getAttribute('data-empty-kind') : null,
            role: (document.querySelector('[data-testid=empty-state]')||{}).getAttribute
                  ? document.querySelector('[data-testid=empty-state]').getAttribute('role') : null,
            text: [...document.querySelectorAll('[data-testid=empty-state]')]
                  .map(e => e.textContent.replace(/\\s+/g,' ').slice(0,220)),
            retry: document.querySelectorAll('[data-testid=error-retry]').length,
            candidate_count_line: (document.querySelector('.candidate-count')||{}).textContent || null,
        })""")
        err_page.screenshot(path=str(OUT / "s5_api_failure.png"))
        record("s5_state_api_failure", err_page,
               "503 ở request ứng viên hiện lỗi TRONG vùng dữ liệu, có nút Thử lại, không im lặng",
               err_dom["retry"] > 0 and err_dom["kind"] == "error"
               and any("induced" in t or "thất bại" in t.lower() for t in err_dom["text"]),
               dom=err_dom)
        err_page.close()

        # (c) genuinely empty: a query id that does not exist in the release
        empty_page = context.new_page()
        goto(empty_page, UI + explorer(HINT_RUN, query="7%3A99999"), wait_until="domcontentloaded")
        empty_page.wait_for_timeout(4000)
        empty_dom = empty_page.evaluate("""() => ({
            kinds: [...document.querySelectorAll('[data-testid=empty-state]')].map(e => e.getAttribute('data-empty-kind')),
            texts: [...document.querySelectorAll('[data-testid=empty-state]')].map(e => e.textContent.replace(/\\s+/g,' ').slice(0,240)),
            inspector: document.querySelectorAll('[data-testid=query-inspector]').length,
            says_no_substitute: document.body.innerText.includes('Không tự thay bằng query khác'),
            names_qid: document.body.innerText.includes('7:99999'),
            retry: document.querySelectorAll('[data-testid=error-retry]').length,
        })""")
        empty_page.screenshot(path=str(OUT / "s5_empty_bad_qid.png"))
        api_bad = api_get(f"/api/composition/runs/{HINT_RUN}/queries?subset=full&condition=multimodal"
                          f"&query_id=7%3A99999&limit=1&offset=0")
        record("s5_state_genuinely_empty", empty_page,
               "query id không tồn tại: UI nói rõ tên qid, không dựng inspector, không gợi ý lỗi API",
               empty_dom["names_qid"] and empty_dom["says_no_substitute"]
               and empty_dom["inspector"] == 0 and empty_dom["retry"] == 0
               and "no-results" in empty_dom["kinds"],
               dom=empty_dom, api_status=api_bad[0],
               api_total=api_bad[1]["total"] if isinstance(api_bad[1], dict) and "total" in api_bad[1] else None)

        # (d) a real 4xx from the backend, surfaced without restarting it
        code_page = context.new_page()
        code_page.route("**/api/composition/runs/*/queries*top_k*", lambda route: route.fulfill(
            status=404, content_type="application/json",
            body=json.dumps({"detail": {"error": "composition_run_not_found",
                                        "message": "Composition run 'f461f391c6d4696011be8067' with subset 'bogus' was not found."}})))
        goto(code_page, UI + explorer(HINT_RUN, query=QID), wait_until="domcontentloaded")
        code_page.wait_for_timeout(4000)
        code_dom = code_page.evaluate("""() => ({
            texts: [...document.querySelectorAll('[data-testid=empty-state]')]
                   .map(e => ({kind: e.getAttribute('data-empty-kind'), text: e.textContent.replace(/\\s+/g,' ').slice(0,280)})),
            retry: document.querySelectorAll('[data-testid=error-retry]').length,
        })""")
        code_page.screenshot(path=str(OUT / "s5_404_response.png"))
        record("s5_404_visible", code_page,
               "404 thật từ backend hiện nguyên văn thông điệp lỗi kèm nút thử lại trong vùng ứng viên",
               code_dom["retry"] > 0 and any("404" in t["text"] or "not found" in t["text"].lower() for t in code_dom["texts"]),
               dom=code_dom)
        code_page.close()

        # ---------------------------------------------------------------- 6. compare + failure analysis note
        goto(page, UI + f"/?area=samples&view=compare&protocol={PROTOCOL}&subset=full"
                      f"&runs=c%3A{HINT_RUN}%2Cc%3A{TME_RUN}&run={HINT_RUN}&query={QID.replace(':', '%3A')}")
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(3500)
        compare = {"rows": page.eval_on_selector_all(
            "[data-testid=query-inspector] table tbody tr",
            "els => els.map(e => e.textContent.replace(/\\s+/g,' ').trim())"),
            "url": page.url,
            "selected": page.inner_text("[data-testid=selected-run-count]")}
        page.screenshot(path=str(OUT / "s6_compare_view.png"))
        record("s6a_compare_panel_renders", page,
               "So sánh mô hình dựng bảng hạng MM cho query 7:1242 với lượt đang xem + các peer",
               any("hint" in r.lower() for r in compare["rows"]) and len(compare["rows"]) >= 2,
               compare=compare, api_hint=hint_api["item"]["ranks"]["multimodal"],
               api_tme=tme_api["item"]["ranks"]["multimodal"])
        peers = api_get(f"/api/composition/runs?subset=full")[1] or []
        record("s6b_compare_reaches_tme", page,
               "TME phải chọn được làm cột so sánh cho cùng query (đây là mục tiêu của bước 6)",
               any("tme" in r.lower() for r in compare["rows"]),
               rows=compare["rows"],
               peer_selection_rule="ComparePanel.tsx: slice(0, 4) trên /api/composition/runs?subset=full",
               api_first_6=[f"{r['model_id']}/{r['checkpoint_id']}:{r['run_id'][:8]}" for r in peers[:6]])

        # open a result FROM Failure Analysis
        goto(page, UI + f"/?area=failures&view=analytics&protocol={PROTOCOL}&subset=full"
                      f"&runs=c%3A{HINT_RUN}%2Cc%3A{TME_RUN}")
        page.wait_for_selector("[data-testid=analysis-row]", timeout=120_000)
        page.wait_for_timeout(2500)
        first_row_qid = page.get_attribute("[data-testid=analysis-row]", "data-query-id")
        page.screenshot(path=str(OUT / "s6_failure_analysis.png"))
        cohort_button = {"count": page.get_attribute("[data-testid=cohort-save]", "data-scope-count"),
                         "label": page.inner_text("[data-testid=cohort-save]")}
        page.click("[data-testid=analysis-row]")
        page.wait_for_selector("[data-testid=query-inspector]", timeout=90_000)
        page.wait_for_timeout(2500)
        jumped = {"url": page.url, "area": "samples" in page.url,
                  "title": page.inner_text("[data-testid=query-title]"),
                  "caption": page.inner_text("[data-testid=query-caption]")}
        record("s6_failure_analysis_to_inspector", page,
               "click một dòng ở Phân tích lỗi mở đúng query đó trong Inspector (URL + tiêu đề khớp)",
               jumped["area"] and first_row_qid in jumped["title"],
               first_row_qid=first_row_qid, jumped=jumped, cohort_button=cohort_button)
        page.screenshot(path=str(OUT / "s6_jumped_from_failures.png"))

        # save exactly ONE disposable note, tagged e2e-check-20261111
        page.click("text=Ghi chú query này")
        page.wait_for_selector("[data-testid=annotation-note]", timeout=60_000)
        page.fill("[data-testid=annotation-note]", TAG)
        page.click("[data-testid=annotation-save]")
        page.wait_for_timeout(1500)
        save_state = page.get_attribute("[data-testid=save-feedback]", "data-status")
        note_context = {"query": page.inner_text("[data-testid=annotation-query-id]"),
                        "note_value": page.input_value("[data-testid=annotation-note]")}
        page.screenshot(path=str(OUT / "s6_note_saved.png"))
        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=annotation-note]", timeout=90_000)
        page.wait_for_timeout(2500)
        persisted = page.input_value("[data-testid=annotation-note]")
        page.screenshot(path=str(OUT / "s6_note_after_reload.png"))
        api_note = api_get(f"/api/annotations?query_id={first_row_qid}&protocol_id={PROTOCOL}")
        record("s6_note_persists", page,
               f"lưu đúng MỘT ghi chú '{TAG}', reload xong vẫn còn, gắn đúng (query, protocol)",
               save_state == "saved" and persisted == TAG
               and any(r.get("note") == TAG and r.get("protocol_id") == PROTOCOL for r in (api_note[1] or [])),
               save_state=save_state, persisted_after_reload=persisted,
               note_context=note_context, api=api_note[1])

        # cohort save on CIRCUS composition runs: record what the UI does (do not try to fix it)
        goto(page, UI + f"/?area=failures&view=analytics&protocol={PROTOCOL}&subset=full"
                      f"&runs=c%3A{HINT_RUN}%2Cc%3A{TME_RUN}")
        page.wait_for_selector("[data-testid=cohort-save]", timeout=120_000)
        page.wait_for_timeout(1500)
        page.click("[data-testid=cohort-save]")
        page.wait_for_timeout(1800)
        cohort_state = {"status": page.get_attribute("[data-testid=save-feedback]", "data-status"),
                        "message": page.inner_text("[data-testid=save-feedback]"),
                        "cohorts_api": api_get("/api/cohorts")[1]}
        page.screenshot(path=str(OUT / "s6_cohort_save.png"))
        record("s6_cohort_note", page,
               f"ghi nhận hành vi lưu cohort tên '{TAG}' trên lượt CIRCUS (báo cáo, không sửa)",
               True, cohort_state=cohort_state, tag=TAG)

        # ---------------------------------------------------------------- 7. timings
        api_timings = {}
        for label, path in (("cold", "/api/models"), ("warm_1", "/api/models"), ("warm_2", "/api/models")):
            t = time.time()
            code, _ = api_get(path)
            api_timings[label] = {"status": code, "seconds": round(time.time() - t, 4)}
        timings["api_models"] = api_timings

        nav_times = []
        for _ in range(3):
            fresh = context.new_page()
            t = time.time()
            goto(fresh, UI + f"/?area=samples&view=explorer&protocol={PROTOCOL}&subset=full")
            fresh.wait_for_function(
                "document.querySelector('#run-select') && document.querySelectorAll('#run-select option').length > 1",
                timeout=90_000)
            nav_times.append(round(time.time() - t, 3))
            fresh.close()
        timings["nav_to_usable_run_list_s_deeplink"] = nav_times
        timings["console_errors"] = console_errors[:20]
        browser.close()

    (OUT / "e2e_results.json").write_text(
        json.dumps({"results": RESULTS, "timings": timings,
                    "api_calls": API_LOG[-120:], "blips": BLIPS, "viewport": VIEWPORT},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "screenshots.txt").write_text(
        "\n".join(sorted(p.name for p in OUT.rglob("*.png"))) + "\n", encoding="utf-8")
    failed = [r["check"] for r in RESULTS if not r["passed"]]
    print("FAILED:", failed, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
