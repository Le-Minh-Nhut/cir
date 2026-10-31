from types import SimpleNamespace

import pytest

from workbench.composition.model_csmcir import _load_csmcir_checkpoint


class FakeModel:
    def __init__(self, report):
        self.report = report

    def load_state_dict(self, state, strict):
        assert state == {"weight": 1}
        assert strict is False
        return self.report


def test_csmcir_checkpoint_missing_keys_fails_and_old_discard_path_does_not(monkeypatch):
    from workbench.composition import model_csmcir

    model = FakeModel(SimpleNamespace(missing_keys=["weight"], unexpected_keys=[]))
    monkeypatch.setattr(model_csmcir, "_check_checkpoint_load", lambda report: None)
    # With old report-discard behavior, this assertion fails: _load_csmcir_checkpoint returns None.
    with pytest.raises(AssertionError):
        assert _load_csmcir_checkpoint(model, {"weight": 1}) is not None
    monkeypatch.undo()
    with pytest.raises(ValueError, match=r"1 missing \['weight'\].*0 unexpected"):
        _load_csmcir_checkpoint(model, {"weight": 1})


def test_csmcir_checkpoint_report_rejects_missing_keys():
    with pytest.raises(ValueError) as exc_info:
        _load_csmcir_checkpoint(
            FakeModel(SimpleNamespace(missing_keys=["weight"], unexpected_keys=[])), {"weight": 1}
        )
    assert "1 missing ['weight']" in str(exc_info.value)
    assert "0 unexpected keys" in str(exc_info.value)


def test_csmcir_checkpoint_report_accepts_empty_report():
    assert _load_csmcir_checkpoint(
        FakeModel(SimpleNamespace(missing_keys=[], unexpected_keys=[])), {"weight": 1}
    ) is None


def test_csmcir_checkpoint_tolerates_unexpected_keys():
    # Policy B: unexpected keys represent benign training artifacts, wrapper prefixes, or unused
    # heads (e.g. from pretraining or multi-task tuning). Model has all required parameters, so
    # evaluation remains scientifically valid without failing closed.
    assert _load_csmcir_checkpoint(
        FakeModel(SimpleNamespace(missing_keys=[], unexpected_keys=["extra_head.weight"])),
        {"weight": 1},
    ) is None


def test_csmcir_checkpoint_report_truncates_offending_key_names():
    keys = [f"layer.{i}.weight" for i in range(7)]
    with pytest.raises(ValueError) as exc_info:
        _load_csmcir_checkpoint(
            FakeModel(SimpleNamespace(missing_keys=keys, unexpected_keys=[])), {"weight": 1}
        )
    message = str(exc_info.value)
    assert "7 missing" in message
    assert "'layer.0.weight'" in message
    assert "'layer.4.weight'" in message
    assert "'layer.5.weight'" not in message
    assert "(and 2 more)" in message
