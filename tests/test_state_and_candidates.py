import torch

from models.iag_srme.context import GroundedEditContext
from models.iag_srme.editor import SharedTokenEditor
from models.iag_srme.grounded_reader import GroundedStateReader
from models.iag_srme.readout import TokenStateReadout
from models.iag_srme.scorer import ConsequenceScorer


def test_current_state_changes_dynamic_tensors() -> None:
    torch.manual_seed(2)
    width, retrieval_dim = 16, 12
    reader = GroundedStateReader(width)
    context = GroundedEditContext(width)
    editor = SharedTokenEditor(width)
    readout = TokenStateReadout(width, retrieval_dim)
    scorer = ConsequenceScorer(width, retrieval_dim)
    anchor = torch.randn(2, 7, width)
    state_a = anchor
    state_b = anchor + 0.3 * torch.randn_like(anchor)
    support = torch.rand(2, 4, 7).softmax(-1)
    intents = torch.randn(2, 4, width)
    reference = torch.nn.functional.normalize(torch.randn(2, retrieval_dim), dim=-1)
    text = torch.randn(2, width)

    def run(state):
        g0, gt, dt = reader(support, anchor, state)
        ctx = context(intents, g0, gt, dt)
        dz, states = editor(ctx, support, anchor, state)
        q = readout(state, anchor, text, reference)
        q_hat = readout(states, anchor, text, reference)
        scores = scorer(ctx, dz, q_hat - q[:, None], dt, support)
        return gt, dt, ctx, dz, scores

    for left, right in zip(run(state_a), run(state_b), strict=True):
        assert not torch.allclose(left, right)


def test_all_previews_have_identical_parent(core, synthetic_encoded) -> None:
    output = core.eval()(synthetic_encoded)
    for step in output.trace:
        expected = step.current_state[:, None] + step.delta_z
        assert torch.equal(step.candidate_states, expected)
