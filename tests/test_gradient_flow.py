import torch

from losses.retrieval import TerminalRetrievalLoss
from models.iag_srme.outputs import BackboneOutput


def test_terminal_gradient_reaches_intent_grounding_editor_and_inputs(
    core, synthetic_encoded
) -> None:
    with torch.no_grad():
        final = core.scorer.score_head[-1]
        final.weight.zero_()
        final.bias.fill_(1.0)
    encoded = BackboneOutput(
        anchor=synthetic_encoded.anchor.detach().requires_grad_(),
        reference_global=synthetic_encoded.reference_global.detach().requires_grad_(),
        text_tokens=synthetic_encoded.text_tokens.detach().requires_grad_(),
        text_global=synthetic_encoded.text_global.detach().requires_grad_(),
        text_semantic_global=synthetic_encoded.text_semantic_global.detach().requires_grad_(),
        text_content_mask=synthetic_encoded.text_content_mask,
    )
    output = core.train()(encoded)
    targets = torch.nn.functional.normalize(torch.randn_like(output.final_query), dim=-1)
    loss = TerminalRetrievalLoss()(output.final_query, targets)
    loss.backward()
    expected = [
        core.intent_encoder.query_bank,
        core.grounder.intent_projection.weight,
        core.editor.direction.weight,
        core.readout.output_projection.weight,
        encoded.anchor,
        encoded.text_tokens,
    ]
    assert all(tensor.grad is not None and tensor.grad.abs().sum() > 0 for tensor in expected)
