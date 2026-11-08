# QA flow g: why `g.annotation_note_restored` reports fail, and proof the UI is correct

`workbench/scripts/browser_qa/run_flows.py` (owned by the browser-QA agent) asserts:

```python
restored = H.texts(ctx.page.locator("textarea"))      # H.texts -> locator.all_inner_texts()
flow.check("g.annotation_note_restored", note_text in restored, ...)
```

`H.texts` uses Playwright's `all_inner_texts()`, which reads `HTMLElement.innerText`. For a
`<textarea>`, `innerText` is **always the empty string in Chromium** — the control's content lives in
its *value*, not in its text children. So this assertion cannot pass for any correct textarea, and it
is not evidence about the app.

## Measured, same page, same moment (1440x900, note saved and reloaded)

```
URL: /?area=notebook&view=annotations&protocol=circus_fashioniq_release_v1&query=7%3A603

all_inner_texts  : ['']                                <-- what the harness reads
input_value      : 'qa annotation 20261108T141922'     <-- what the user sees in the box
evaluate e=>e.value       : 'qa annotation 20261108T141922'
evaluate e=>e.textContent : 'qa annotation 20260808T141922'  (initial/default value)
```

## The rest of flow g passes at all three viewports

- `g.annotation_persisted_or_refused` — PASS. Server row exists:
  `{"query_id":"7:603","protocol_id":"circus_fashioniq_release_v1","labels":[],"note":"qa annotation 20261108T141922"}`
- `g.annotation_form_bound_to_query` — PASS (form names `7:603`).
- `g.annotation_bound_after_reload` — PASS (reloaded page names the query).
- `g.annotation_protocol_bound` — PASS (reloaded page names the protocol).
- `browser.failed_requests` 0, `browser.http_ge_400` 0, console/page errors 0.

Exactly one assertion fails per viewport, and it is the textarea `innerText` read.

## One-line fix on the harness side

In `run_flows.py` flow g, replace the value read:

```python
restored = [ctx.page.locator("textarea").first.input_value()]   # or .evaluate("e => e.value")
```

Alternatively teach `harness.texts` to use `input_value()` when the element is an `input`/`textarea`.
No UI change can fix it: switching the note field away from `<textarea>` (e.g. to `contenteditable`)
would make `innerText` non-empty, but that is a worse control for a multi-line note and would be
changing the app to satisfy a broken probe.

## Annotation contract, verified live with the new backend

| call | result |
|---|---|
| `PUT /api/annotations` `{"query_id":"7:603","protocol_id":"circus_fashioniq_release_v1",...}` | `200`, echoed `query_id` byte-for-byte |
| `GET /api/annotations?query_id=7%3A603&protocol_id=circus_fashioniq_release_v1` | the record, with the note |
| reload of the deep link above | textarea value restored from the server |

The client-side duplicate validator that used to block this (`/^(dress|shirt|toptee):\d+:[^:]+:[^:]+$/`
in `NotebookPage.tsx`) has been removed: the request is always sent and the server's own named error
(`invalid_query_id`, `query_id_protocol_mismatch`, `unknown_query_id`, `unknown_dataset`,
`unknown_protocol`) is surfaced verbatim as `error: message`.
