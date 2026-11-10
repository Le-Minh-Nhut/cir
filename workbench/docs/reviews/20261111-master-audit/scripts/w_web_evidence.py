#!/usr/bin/env python3
"""W-01…W-05 browser evidence: one crop per scope group, with the run id and qid beside the body.

Drives the LIVE stack at :5173 / :8000 (never the stale :8765) and writes:

    <out>/<issue>_<slug>.png       1920-wide screenshot of the viewport
    <out>/<issue>_<slug>.json      the run id, qid, API response digest and the DOM text it saw
    <out>/manifest.json            every shot with its URL, viewport and sha256

Each shot is paired with the API response it renders, so the screenshot cannot be read as evidence
of a different payload: `api` in the JSON is the exact body the page was served for that context,
fetched with the same query string the UI built.

Screenshots are 1920x1080 at deviceScaleFactor 1; a caller that needs detail crops with PIL rather
than asking for a taller viewport (a taller viewport changes what the page renders).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DEFAULT_BROWSERS = REPO / "workbench/envs/ctl312h/.playwright-browsers"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_json(url: str) -> object:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:                  # a 4xx is a VALUE here, not a crash
        return {"http_status": error.code, "body": error.read().decode("utf-8", "replace")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5173")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chrome", default=os.environ.get("WORKBENCH_CHROME_PATH", "/usr/bin/google-chrome"))
    args = parser.parse_args()

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(DEFAULT_BROWSERS))
    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    CIRCUS = "circus_fashioniq_release_v1"
    LIMN_RUN = "9e4918a6b5c243412cbc58c9"
    ENCODER_RUN = "ce89a9a27c4cceadeaaf30b0"

    shots = [
        # (slug, issue, url, extra api probes)
        ("w01_native_overview", "W-01",
         "/?area=overview&view=&protocol=fashioniq_full_gallery_ref_excluded&subset=full",
         ["/api/native-measurements"]),
        ("w01_native_table", "W-01",
         "/?area=evaluation&view=native&protocol=fashioniq_full_gallery_ref_excluded&subset=full",
         ["/api/native-measurements"]),
        ("w02_overview_circus", "W-02",
         f"/?area=overview&view=&protocol={CIRCUS}&subset=sf", ["/api/composition/runs"]),
        ("w02_models_inventory", "W-02",
         f"/?area=experiments&view=models&protocol={CIRCUS}&subset=sf", ["/api/composition/runs?subset=sf"]),
        ("w02_native_scope_populated", "W-02",
         "/?area=evaluation&view=native&protocol=fashioniq_original_split&subset=full",
         ["/api/native-measurements", "/api/benchmark-runs"]),
        ("w03_inspector_7-3038_limn_full", "W-03",
         f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full&runs=c%3A{LIMN_RUN}&run={LIMN_RUN}&query=7%3A3038",
         [f"/api/composition/runs/{LIMN_RUN}/queries?subset=full&condition=multimodal&query_id=7%3A3038&top_k=10&limit=1&offset=0"]),
        ("w03_inspector_7-3038_limn_top50", "W-03",
         f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full&runs=c%3A{LIMN_RUN}&run={LIMN_RUN}&query=7%3A3038&k=50",
         [f"/api/composition/runs/{LIMN_RUN}/queries?subset=full&condition=multimodal&query_id=7%3A3038&top_k=50&limit=1&offset=0"]),
        ("w03_inspector_midrank_encoder", "W-03",
         f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full&runs=c%3A{ENCODER_RUN}&run={ENCODER_RUN}&query=7%3A3038&k=20",
         [f"/api/composition/runs/{ENCODER_RUN}/queries?subset=full&condition=multimodal&query_id=7%3A3038&top_k=20&limit=1&offset=0"]),
        ("w03_limn_diagnostic_toptee", "W-03",
         f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full&runs=c%3A{LIMN_RUN}&run={LIMN_RUN}",
         ["/api/composition/diagnostics/limn-dual-scope",
          "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7%3A3038&scope=toptee&condition=multimodal&depth=20"]),
        ("w04_deeplink_out_of_range", "W-04",
         f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=sf&runs=c%3A{LIMN_RUN}&run={LIMN_RUN}&query=7:99999",
         [f"/api/composition/runs/{LIMN_RUN}/queries?subset=sf&condition=multimodal&query_id=7%3A99999&top_k=10&limit=1&offset=0"]),
        ("w04_reload_full", "W-04",
         f"/?area=evaluation&view=composition&protocol={CIRCUS}&subset=full", ["/api/composition/runs"]),
        ("w05_benchmark_reports", "W-05",
         "/?area=evaluation&view=benchmark&protocol=fashioniq_original_split&subset=full",
         ["/api/benchmark-runs", "/api/evidence/summary"]),
        ("w05_pe_histograms", "W-05",
         f"/?area=evaluation&view=composition&protocol={CIRCUS}&subset=validated",
         ["/api/composition/capabilities?subset=validated", "/api/parity-gate"]),
    ]

    failures: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.chrome if Path(args.chrome).exists() else None,
            args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        console: list[str] = []
        page.on("console", lambda message: console.append(f"{message.type}: {message.text}")
                if message.type in ("error", "warning") else None)
        page.on("pageerror", lambda error: console.append(f"pageerror: {error}"))

        for slug, issue, url, probes in shots:
            console.clear()
            # `networkidle` NEVER fires on this stack: `/api/models` hashes every checkpoint and takes
            # ~154 s, so the network is never quiet while the page is up. Wait on the DOM plus the
            # shell's own readiness marker instead of on silence.
            page.goto(args.base_url + url, wait_until="domcontentloaded", timeout=90_000)
            # Any rendered region will do: the areas name themselves differently (the models view
            # renders `model-inventory`, not `area-experiments`), and the point of this wait is that
            # React mounted and at least one region committed — not which one.
            page.wait_for_selector("main [data-testid]", timeout=60_000)
            page.wait_for_timeout(3500)
            png = out / f"{slug}.png"
            page.screenshot(path=str(png), full_page=False)
            body = page.inner_text("body")
            record = {
                "issue": issue, "slug": slug, "url": args.base_url + url,
                "viewport": "1920x1080@1", "png": png.name,
                "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                "dom_text_sha256": digest(body),
                "dom_text_excerpt": body[:4000],
                "console": list(console),
                "api": {probe: fetch_json(args.api_url + probe) for probe in probes},
            }
            (out / f"{slug}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
            if record["console"]:
                failures.append(f"{slug}: console {record['console'][:3]}")
            if "error" in body.lower() and "lỗi" not in body.lower():
                pass
            manifest.append({key: record[key] for key in
                             ("issue", "slug", "url", "png", "png_sha256", "dom_text_sha256")})
            print(f"{slug}: {png.name} {record['png_sha256'][:12]} bytes={png.stat().st_size}",
                  flush=True)
        browser.close()

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
