from __future__ import annotations

import torch

from models.taper_mag.contracts import EncodedPolicyBatch
from models.taper_mag.executor import SharedLocalExecutor
from models.taper_mag.model import TaperMAG, TaperMAGConfig
from models.taper_mag.readout import ChangeAwareReadout
from models.taper_mag.state import LocalState
from models.taper_mag.text_reader import SharedQueryTextReader
from models.taper_mag.visual_grounding import EditAwareVisualGrounding


def encoded_batch(batch: int = 2) -> EncodedPolicyBatch:
    torch.manual_seed(4)
    return EncodedPolicyBatch(
        reference_local=torch.randn(batch, 6, 24),
        reference_local_mask=torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)[:batch],
        reference_global=torch.nn.functional.normalize(torch.randn(batch, 32), dim=-1),
        text_tokens=torch.randn(batch, 7, 20),
        text_attention_mask=torch.ones(batch, 7, dtype=torch.bool),
        text_content_mask=torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)[:batch],
    )


def test_noncompetitive_query_attention_and_content_mask() -> None:
    reader = SharedQueryTextReader(d_model=256, num_queries=4, dropout=0).eval()
    with torch.no_grad():
        reader.block.attention.in_proj_weight.zero_()
        reader.block.attention.in_proj_bias.zero_()
    text = torch.randn(2, 5, 256)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    output = reader(text, mask)
    torch.testing.assert_close(output.attention.sum(dim=-1), torch.ones(2, 4))
    assert (output.attention.masked_select(~mask[:, None].expand_as(output.attention)) == 0).all()
    # Zero Q/K yields uniform independent distributions: every query may read token zero.
    assert (output.attention[:, :, 0] > 0).all()
    assert not torch.allclose(output.attention.sum(dim=1), torch.ones_like(output.attention[:, 0]))


def test_visual_grounding_mask_and_query_permutation_equivariance() -> None:
    grounding = EditAwareVisualGrounding(d_model=256, dropout=0).eval()
    queries = torch.randn(2, 4, 256)
    local = torch.randn(2, 6, 256)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    normal = grounding(queries, local, mask)
    permuted = grounding(queries[:, permutation], local, mask)
    torch.testing.assert_close(permuted.grounded, normal.grounded[:, permutation])
    torch.testing.assert_close(permuted.attention, normal.attention[:, permutation])
    assert (normal.attention.masked_select(~mask[:, None].expand_as(normal.attention)) == 0).all()
    assert torch.isfinite(normal.attention).all()


def _state() -> LocalState:
    local = torch.randn(2, 5, 256)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    local = local.masked_fill(~mask.unsqueeze(-1), 0)
    return LocalState(
        local=local,
        initial_local=local.clone(),
        local_mask=mask,
        reference_global=torch.nn.functional.normalize(torch.randn(2, 32), dim=-1),
        reference_anchor=torch.randn(2, 256),
        alive=torch.ones(2, dtype=torch.bool),
    )


def test_executor_vectorization_order_parent_identity_and_no_aliasing() -> None:
    executor = SharedLocalExecutor(d_model=256).eval()
    state = _state()
    parent = state.local.clone()
    context = torch.randn(2, 256)
    operators = torch.randn(2, 4, 256)
    features = executor.encode_state(state, context)
    vectorized = executor.enumerate(state, features, operators)
    loop = torch.cat(
        [executor.enumerate(state, features, operators[:, index : index + 1]).local for index in range(4)],
        dim=1,
    )
    torch.testing.assert_close(vectorized.local, loop)
    permutation = torch.tensor([3, 1, 0, 2])
    reordered = executor.enumerate(state, features, operators[:, permutation])
    torch.testing.assert_close(reordered.local, vectorized.local[:, permutation])
    torch.testing.assert_close(state.local, parent)
    assert vectorized.local.data_ptr() != state.local.data_ptr()
    zero = executor.enumerate(state, features, operators, delta_scale=0.0)
    torch.testing.assert_close(zero.local, state.local[:, None].expand_as(zero.local), rtol=0, atol=0)


def test_readout_exact_reference_identity_and_retrieval_contract() -> None:
    state = _state()
    readout = ChangeAwareReadout(d_model=256, retrieval_dim=32)
    output = readout(state)
    torch.testing.assert_close(output.query, state.reference_global, atol=1e-6, rtol=1e-6)
    assert output.query.shape == (2, 32)
    torch.testing.assert_close(output.query.norm(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(output.query).all()


def test_actor_gradients_reach_queries_executor_and_retrieval_head() -> None:
    model = TaperMAG(
        TaperMAGConfig(text_dim=20, vision_dim=24, retrieval_dim=32, dropout=0, max_steps=1)
    )
    from models.taper_mag.rollout import RolloutConfig

    output = model(encoded_batch(), RolloutConfig(max_steps=1, selection_mode="uniform"))
    output.final_query.square().mean().backward()
    assert model.operator_generator.text_reader.queries.grad is not None
    assert model.executor.film.weight.grad is not None
    assert model.readout.retrieval_projection.weight.grad is not None
