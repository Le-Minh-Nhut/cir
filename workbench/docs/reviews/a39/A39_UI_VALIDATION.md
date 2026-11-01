# A39 UI Validation

## Mobile timeout investigation

**No browser session was executed this round. No screenshot was captured or verified.** The mobile phase was investigated statically and with one in-process latency probe only (`STATIC` diagnosis plus diagnostic execution).

Harness executes `cold` → `fixture` → `mobile`. Cold phase waits until dashboard reports 17 registered models before opening Models; mobile phase uses a fresh isolated context and waits only for `/api/runs`, then opens Models while browser `/api/models` and Python HTTP client's second `/api/models` request are both in flight. Models endpoint is synchronous and CPU-heavy; competing work under Python GIL likely extends latency beyond the 90s deadline. Its exact threshold failure is a ranked diagnosis, not reproduced by Playwright.

Measured by in-process `workbench.backend.main.models()` probe:

```text
COLD_LATENCY_SEC 43.383 entries 17
WARM_LATENCY_SEC 8.891 entries 17
```

Prior A38 cold 96.602s and warm 27.65–28.52s exceeded/approached the configured 90s navigation and locator timeout. A39 values are after the A38 optimization and are not browser timings.

### Ranked causes

1. **Primary diagnosis, race + duplicated cold `/api/models` calls:** mobile skips the strict registered-models barrier and fires browser and Python API requests concurrently; warm/cold in-process latency makes timeout plausible, but not reproduced in browser.
2. **Contributing hypothesis, Vite proxy/gateway timeout:** delayed/dropped response could leave `models === null`, so page remains at one loading row until timeout.
3. **Minor hypothesis, Vite dev-mode transform overhead:** mobile context cold-loads dynamic frontend modules while backend is already contended.

Source: `.scratch-tmp/a39-pipe/playwright-timeout.md:23-96`.

## Viewports and requested gap

Harness actually supports two configured viewports: desktop `1440×900`, mobile `390×844`. Brief requests four viewports; only two are present, so two requested viewport profiles are unsupported/not established by the harness. Their exact dimensions are `NOT_ESTABLISHED` in this evidence. No WebKit run and no physical iPhone run.

## Build

`export TMPDIR=/dev/shm/a39-freeze; npx vite build` succeeded (`vite v8.3.2`, built in 181ms); see `/dev/shm/a39-freeze/build.md`. Build success does not substitute for browser validation.