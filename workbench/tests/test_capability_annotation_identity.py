"""Annotations must bind to a real query identity, and reload as the same query.

The backend used to accept only the canonical `category:index:reference:target` form, so every
CIRCUS release query id (`7:603`) was refused -- the annotation flow could not be exercised on the
main dataset the workbench serves. Loosening validation to "anything goes" would be worse: an
annotation is keyed on `(query_id, protocol_id)`, so an id that describes no query of that protocol
would persist and be read back as if it were about real data.

What these tests pin:

* the CIRCUS form is admitted ONLY under the release's own protocol AND only for a qid the loaded
  release actually contains;
* a genuine cross-protocol collision is structurally impossible -- the same numeric shape under a
  different protocol is refused;
* the `query_id` round-trips byte for byte, so a reload binds to the query the UI sent;
* malformed ids are still refused with a named reason.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from workbench.backend.composition import release_protocol_ids, release_query_ids
from workbench.backend.main import app, annotation_query_id_error

CIRCUS_PROTOCOL = "circus_fashioniq_release_v1"


def _release_ready() -> bool:
    return bool(release_protocol_ids()) and bool(release_query_ids())


def _first_release_qid() -> str:
    ids = release_query_ids()
    assert ids is not None
    return sorted(ids, key=lambda value: (int(value.split(":")[0]), int(value.split(":")[1])))[0]


def _annotations_path(monkeypatch, tmp_path):
    """Point the annotation store at a temp file so the test never touches real annotations."""
    import workbench.backend.main as main

    path = tmp_path / "annotations.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(main, "ANNOTATIONS_PATH", path)
    return path


def test_release_is_loaded_for_these_checks():
    if not _release_ready():
        pytest.skip("CIRCUS release not available locally")
    assert CIRCUS_PROTOCOL in release_protocol_ids()
    assert "7:603" in (release_query_ids() or frozenset())


def test_circus_query_id_is_accepted_under_the_release_protocol():
    if not _release_ready():
        pytest.skip("CIRCUS release not available locally")
    qid = _first_release_qid()
    assert annotation_query_id_error(qid, CIRCUS_PROTOCOL) is None
    assert annotation_query_id_error("7:603", CIRCUS_PROTOCOL) is None


def test_circus_shape_is_refused_under_a_different_protocol():
    """`7:603` under another protocol must not be admitted: that is where a collision would live."""
    if not _release_ready():
        pytest.skip("CIRCUS release not available locally")
    problem = annotation_query_id_error("7:603", "fashioniq_original_split")
    assert problem is not None
    assert problem["error"] == "query_id_protocol_mismatch"
    assert CIRCUS_PROTOCOL in problem["details"]["release_protocols"]


def test_a_qid_the_release_does_not_contain_is_refused():
    if not _release_ready():
        pytest.skip("CIRCUS release not available locally")
    problem = annotation_query_id_error("7:999999999", CIRCUS_PROTOCOL)
    assert problem is not None
    assert problem["error"] == "unknown_query_id"
    assert problem["details"]["known_query_count"] == 6003


@pytest.mark.parametrize("bad", ["7", ":", "7:", ":603", "abc:603", "7:603:extra",
                                 "7:603 ", " 7:603", "7:60a", "dress:1", ""])
def test_malformed_ids_are_refused_with_a_reason(bad):
    problem = annotation_query_id_error(bad, CIRCUS_PROTOCOL)
    assert problem is not None
    assert problem["error"] in {"invalid_query_id", "query_id_protocol_mismatch",
                                "unknown_query_id"}
    assert problem["message"]


def test_canonical_fashioniq_identity_is_still_accepted():
    local = ("dress:12:B001X.jpg:B002Y.jpg", "shirt:0:ref.png:tgt.png",
             "toptee:999:r/t.jpg:t/t.jpg")
    for query_id in local:
        assert annotation_query_id_error(query_id, CIRCUS_PROTOCOL) is None


def test_save_and_reload_bind_to_the_same_circus_query(monkeypatch, tmp_path):
    """The round-trip the UI depends on: save, reload, still the same query."""
    if not _release_ready():
        pytest.skip("CIRCUS release not available locally")
    path = _annotations_path(monkeypatch, tmp_path)
    client = TestClient(app)
    qid = _first_release_qid()
    note = "qa annotation circus round-trip"

    saved = client.put("/api/annotations", json={
        "query_id": qid, "protocol_id": CIRCUS_PROTOCOL,
        "labels": ["wrong_attribute"], "note": note})
    assert saved.status_code == 200, saved.text
    assert saved.json()["query_id"] == qid, "the response must echo the identity the caller sent"
    assert saved.json()["protocol_id"] == CIRCUS_PROTOCOL

    restored = client.get(f"/api/annotations?query_id={qid}&protocol_id={CIRCUS_PROTOCOL}")
    assert restored.status_code == 200
    rows = restored.json()
    assert len(rows) == 1, rows
    assert rows[0]["query_id"] == qid
    assert rows[0]["note"] == note
    # Persisted on disk under exactly that pair, so a reload after a restart binds the same way.
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert [(item["query_id"], item["protocol_id"]) for item in on_disk] == [(qid, CIRCUS_PROTOCOL)]


def test_the_same_numeric_shape_under_another_protocol_does_not_collide(monkeypatch, tmp_path):
    """`7:603` under the release and a local `7:603` are different keys, and the local one is refused."""
    if not _release_ready():
        pytest.skip("CIRCUS release not available locally")
    path = _annotations_path(monkeypatch, tmp_path)
    client = TestClient(app)
    qid = "7:603"

    ok = client.put("/api/annotations", json={
        "query_id": qid, "protocol_id": CIRCUS_PROTOCOL, "labels": ["other"], "note": "release"})
    assert ok.status_code == 200, ok.text

    refused = client.put("/api/annotations", json={
        "query_id": qid, "protocol_id": "fashioniq_original_split",
        "labels": ["other"], "note": "should not exist"})
    assert refused.status_code == 400
    assert refused.json()["detail"]["error"] == "query_id_protocol_mismatch"

    # Only the release-bound row exists; reading by the other protocol returns nothing.
    rows = client.get(f"/api/annotations?query_id={qid}&protocol_id={CIRCUS_PROTOCOL}").json()
    assert [row["note"] for row in rows] == ["release"]
    other = client.get(f"/api/annotations?query_id={qid}"
                       "&protocol_id=fashioniq_original_split").json()
    assert other == []
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 1


def test_unknown_protocol_is_still_refused(monkeypatch, tmp_path):
    _annotations_path(monkeypatch, tmp_path)
    client = TestClient(app)
    response = client.put("/api/annotations", json={
        "query_id": "dress:1:a.jpg:b.jpg", "protocol_id": "not_a_protocol",
        "labels": ["other"]})
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "unknown_protocol"


def test_unknown_label_is_still_refused(monkeypatch, tmp_path):
    if not _release_ready():
        pytest.skip("CIRCUS release not available locally")
    _annotations_path(monkeypatch, tmp_path)
    client = TestClient(app)
    response = client.put("/api/annotations", json={
        "query_id": _first_release_qid(), "protocol_id": CIRCUS_PROTOCOL,
        "labels": ["not_a_real_label"]})
    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "invalid_annotation_labels"
