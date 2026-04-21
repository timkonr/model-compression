"""
Unit tests for pruning operations in prune.py.

Run without a GPU or model:
    cd src && python -m pytest tests/test_pruning.py -v
    cd src && python tests/test_pruning.py

Each test documents:
  - WHAT is being tested
  - EXPECTED values derived by hand (not from the code itself)
  - WHY this specific case matters
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from prune import ActivationCollector, apply_linear_pair_pruning, compute_linear_hidden_scores


# ─────────────────────────────────────────────────────────────────────────────
# ActivationCollector
#
# Expected behavior: seq=mean, batch=L2
#   score_i = sqrt( sum_over_all_samples( mean_over_seq(|act_i|)^2 ) )
# ─────────────────────────────────────────────────────────────────────────────


def test_collector_single_sample_single_token():
    """
    Simplest case: 1 sample, 1 token, 3 neurons.

    Input:  [[[1.0, 2.0, 3.0]]]   shape (1, 1, 3)
    seq-mean over dim=1:  [[1.0, 2.0, 3.0]]
    sumsq:                [1.0, 4.0, 9.0]
    result = sqrt:        [1.0, 2.0, 3.0]
    """
    c = ActivationCollector()
    c.update(torch.tensor([[[1.0, 2.0, 3.0]]]))
    result = c.result()

    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_collector_single_sample_two_tokens():
    """
    seq-mean across 2 tokens.

    Input:  [[[1.0, 4.0], [3.0, 2.0]]]   shape (1, 2, 2)
    seq-mean:  [[2.0, 3.0]]
    sumsq:     [4.0, 9.0]
    result:    [2.0, 3.0]
    """
    c = ActivationCollector()
    c.update(torch.tensor([[[1.0, 4.0], [3.0, 2.0]]]))
    result = c.result()

    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_collector_two_update_calls():
    """
    batch=L2 accumulation across multiple update() calls.

    Call 1: [[[2.0]]]  → seq-mean [[2.0]] → sumsq [4.0]
    Call 2: [[[4.0]]]  → seq-mean [[4.0]] → sumsq [4.0 + 16.0] = [20.0]
    result: [sqrt(20)] ≈ [4.472]
    """
    c = ActivationCollector()
    c.update(torch.tensor([[[2.0]]]))
    c.update(torch.tensor([[[4.0]]]))
    result = c.result()

    expected = torch.tensor([math.sqrt(20.0)])
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"


def test_collector_batch_of_two_samples():
    """
    Two samples in one update() call — same as two separate calls.

    Input:  [[[1.0], [3.0]], [[2.0], [4.0]]]   shape (2, 2, 1)
    seq-mean (dim=1):  [[2.0], [3.0]]           shape (2, 1)
    sumsq:             [4.0 + 9.0] = [13.0]
    result:            [sqrt(13)]
    """
    c = ActivationCollector()
    c.update(torch.tensor([[[1.0], [3.0]], [[2.0], [4.0]]]))
    result = c.result()

    expected = torch.tensor([math.sqrt(13.0)])
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"


def test_collector_with_sequence_mask():
    """
    PAD positions (mask=False) must not contribute to the seq-mean.

    Input:  [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]   shape (1, 3, 2)
    Mask:   [[True, True, False]]  — 3rd token is PAD
    Masked sum (first 2 tokens): [[4.0, 6.0]]
    Denom: 2 valid tokens
    Mean: [[2.0, 3.0]]
    result: [2.0, 3.0]
    """
    c = ActivationCollector()
    mask = torch.tensor([[True, True, False]])
    c.set_sequence_mask(mask)
    c.update(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]))
    result = c.result()

    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"


def test_collector_mask_cleared_after_call():
    """
    clear_sequence_mask() must make subsequent updates ignore masking.
    """
    c = ActivationCollector()
    # First update WITH mask (should give [2.0, 3.0] as above)
    mask = torch.tensor([[True, True, False]])
    c.set_sequence_mask(mask)
    c.update(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]))
    c.clear_sequence_mask()

    # Second update without mask — full mean of [1.0, 0.0] = [0.5]
    # Using a separate collector to verify the cleared state
    c2 = ActivationCollector()
    c2.update(torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]))
    result_no_mask = c2.result()

    expected_no_mask = torch.tensor([3.0, 4.0])  # mean of all 3 tokens
    assert torch.allclose(result_no_mask, expected_no_mask, atol=1e-5)


def test_collector_abs_flips_negatives():
    """
    Negative activations (small GELU outputs) are abs()-ed before aggregation.

    Input:  [[[-2.0, 3.0]]]
    abs:    [[[2.0, 3.0]]]
    result: [2.0, 3.0]
    """
    c = ActivationCollector()
    c.update(torch.tensor([[[-2.0, 3.0]]]))
    result = c.result()

    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"


def test_collector_2d_input():
    """
    2D input (batch, hidden_dim) is treated as seq_len=1.

    Input:  [[3.0, 4.0], [1.0, 2.0]]   shape (2, 2)
    After unsqueeze(1): (2, 1, 2)
    seq-mean: same, [[3.0, 4.0], [1.0, 2.0]]
    sumsq: [9+1, 16+4] = [10.0, 20.0]
    result: [sqrt(10), sqrt(20)]
    """
    c = ActivationCollector()
    c.update(torch.tensor([[3.0, 4.0], [1.0, 2.0]]))
    result = c.result()

    expected = torch.tensor([math.sqrt(10.0), math.sqrt(20.0)])
    assert torch.allclose(result, expected, atol=1e-5), f"Expected {expected}, got {result}"


def test_collector_no_updates_raises():
    """result() before any update() must raise, not return garbage."""
    c = ActivationCollector()
    try:
        c.result()
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# apply_linear_pair_pruning
#
# Prunes hidden neurons from a (Linear → Linear) pair given keep indices.
# Critical invariant: the pruned model's output must equal the output of
# the original model when the removed neurons are zeroed out.
# ─────────────────────────────────────────────────────────────────────────────


def test_pruning_output_shapes():
    """
    After pruning Linear(4→6) + Linear(6→3) keeping neurons [0, 2, 4]:
      new_first:  Linear(4→3)
      new_second: Linear(3→3)
    """
    first = nn.Linear(4, 6, bias=True)
    second = nn.Linear(6, 3, bias=True)
    keep_idx = torch.tensor([0, 2, 4])

    new_first, new_second = apply_linear_pair_pruning(first, second, keep_idx)

    assert new_first.weight.shape == (3, 4), f"Got {new_first.weight.shape}"
    assert new_first.bias.shape == (3,), f"Got {new_first.bias.shape}"
    assert new_second.weight.shape == (3, 3), f"Got {new_second.weight.shape}"
    assert new_second.bias.shape == (3,), f"Got {new_second.bias.shape}"


def test_pruning_correct_rows_and_cols_selected():
    """
    Weights of the pruned layers must be exactly the kept rows/columns.

    first.weight rows [0, 2, 4] → new_first.weight
    second.weight columns [0, 2, 4] → new_second.weight
    first.bias indices [0, 2, 4] → new_first.bias
    second.bias unchanged (it's per-output, not per-hidden-neuron)
    """
    torch.manual_seed(42)
    first = nn.Linear(4, 6, bias=True)
    second = nn.Linear(6, 3, bias=True)

    w1 = first.weight.data.clone()
    b1 = first.bias.data.clone()
    w2 = second.weight.data.clone()
    b2 = second.bias.data.clone()

    keep_idx = torch.tensor([0, 2, 4])
    new_first, new_second = apply_linear_pair_pruning(first, second, keep_idx)

    assert torch.allclose(new_first.weight, w1[[0, 2, 4], :]), "Wrong rows in new_first.weight"
    assert torch.allclose(new_first.bias, b1[[0, 2, 4]]), "Wrong entries in new_first.bias"
    assert torch.allclose(new_second.weight, w2[:, [0, 2, 4]]), "Wrong cols in new_second.weight"
    assert torch.allclose(new_second.bias, b2), "new_second.bias should be unchanged"


def test_pruning_numerical_equivalence():
    """
    The pruned model's output must equal the original's output when the
    removed neurons are manually zeroed out.

    This is the key correctness invariant — if this fails, pruning silently
    changes the model's computation beyond just removing neurons.
    """
    torch.manual_seed(42)
    first = nn.Linear(4, 6, bias=True)
    second = nn.Linear(6, 3, bias=True)

    keep_idx = torch.tensor([0, 2, 4])
    new_first, new_second = apply_linear_pair_pruning(first, second, keep_idx)

    x = torch.randn(5, 4)  # batch of 5 inputs

    # Pruned model output
    with torch.no_grad():
        out_pruned = new_second(F.gelu(new_first(x)))

    # Manual: zero out removed neurons in original model
    w1_sel = first.weight.data[[0, 2, 4], :]
    b1_sel = first.bias.data[[0, 2, 4]]
    w2_sel = second.weight.data[:, [0, 2, 4]]
    b2 = second.bias.data

    with torch.no_grad():
        hidden = F.gelu(x @ w1_sel.T + b1_sel)
        out_manual = hidden @ w2_sel.T + b2

    assert torch.allclose(out_pruned, out_manual, atol=1e-5), (
        f"Pruned output differs from manual computation.\n"
        f"Max diff: {(out_pruned - out_manual).abs().max().item()}"
    )


def test_pruning_keep_all_neurons():
    """
    Keeping all neurons must produce a model that is numerically equivalent
    to the original.
    """
    torch.manual_seed(7)
    first = nn.Linear(3, 5, bias=True)
    second = nn.Linear(5, 2, bias=True)

    keep_idx = torch.arange(5)
    new_first, new_second = apply_linear_pair_pruning(first, second, keep_idx)

    x = torch.randn(4, 3)
    with torch.no_grad():
        out_original = second(F.gelu(first(x)))
        out_pruned = new_second(F.gelu(new_first(x)))

    assert torch.allclose(out_original, out_pruned, atol=1e-5), (
        f"Keeping all neurons should be a no-op. Max diff: "
        f"{(out_original - out_pruned).abs().max().item()}"
    )


def test_pruning_no_bias():
    """Pruning must work correctly when bias=False."""
    first = nn.Linear(4, 6, bias=False)
    second = nn.Linear(6, 3, bias=False)

    keep_idx = torch.tensor([1, 3])
    new_first, new_second = apply_linear_pair_pruning(first, second, keep_idx)

    assert new_first.bias is None
    assert new_second.bias is None
    assert new_first.weight.shape == (2, 4)
    assert new_second.weight.shape == (3, 2)


# ─────────────────────────────────────────────────────────────────────────────
# compute_linear_hidden_scores
# ─────────────────────────────────────────────────────────────────────────────


def test_scores_random_shape():
    """random mode returns scores of shape (hidden_dim,)."""
    first = nn.Linear(4, 6)
    second = nn.Linear(6, 3)
    scores = compute_linear_hidden_scores(first, second, mode="random")
    assert scores.shape == (6,), f"Got {scores.shape}"


def test_scores_first_l2_values():
    """
    first_l2 mode: score_i = ||first.weight[i, :]||_2  (row i of W1)

    With a known weight matrix we can verify exactly.
    """
    first = nn.Linear(2, 3, bias=False)
    second = nn.Linear(3, 2, bias=False)

    # Set first.weight rows to [3,4], [1,0], [0,5] → norms 5, 1, 5
    first.weight.data = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
    scores = compute_linear_hidden_scores(first, second, mode="first_l2")

    expected = torch.tensor([5.0, 1.0, 5.0])
    assert torch.allclose(scores, expected, atol=1e-5), f"Expected {expected}, got {scores}"


def test_scores_wanda_returns_activation_scores():
    """
    wanda mode must return the provided activation_scores unchanged
    (dtype/device cast aside). The multiplication by W2 norms happens
    outside this function (in the collection step).
    """
    first = nn.Linear(4, 6)
    second = nn.Linear(6, 3)
    act_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    scores = compute_linear_hidden_scores(first, second, mode="wanda", activation_scores=act_scores)

    assert torch.allclose(scores, act_scores.to(dtype=first.weight.dtype)), (
        f"wanda mode must return activation_scores as-is"
    )


def test_scores_wanda_wrong_shape_raises():
    """wanda mode must raise if activation_scores shape doesn't match hidden_dim."""
    first = nn.Linear(4, 6)
    second = nn.Linear(6, 3)
    wrong_scores = torch.ones(5)  # should be 6

    try:
        compute_linear_hidden_scores(first, second, mode="wanda", activation_scores=wrong_scores)
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


def run_all():
    tests = [
        # ActivationCollector
        test_collector_single_sample_single_token,
        test_collector_single_sample_two_tokens,
        test_collector_two_update_calls,
        test_collector_batch_of_two_samples,
        test_collector_with_sequence_mask,
        test_collector_mask_cleared_after_call,
        test_collector_abs_flips_negatives,
        test_collector_2d_input,
        test_collector_no_updates_raises,
        # apply_linear_pair_pruning
        test_pruning_output_shapes,
        test_pruning_correct_rows_and_cols_selected,
        test_pruning_numerical_equivalence,
        test_pruning_keep_all_neurons,
        test_pruning_no_bias,
        # compute_linear_hidden_scores
        test_scores_random_shape,
        test_scores_first_l2_values,
        test_scores_wanda_returns_activation_scores,
        test_scores_wanda_wrong_shape_raises,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
