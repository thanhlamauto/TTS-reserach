import itertools

import pytest
import torch

from otlab.geometry import block_down, block_up, lowpass, rms, trusted_state
from otlab.ot import (epsilon_scale, projections, quadratic, regularized_ot,
                      sinkhorn_divergence, sliced_w2, solve, weighted_w2_1d)
from otlab.p1 import assignment, utilities
from otlab.proximal import feature_vjp
from otlab.target import cluster, target_weights


@pytest.mark.parametrize("epsilon", [0.03, 0.2, 2.0])
def test_sinkhorn_marginals_and_entropy(epsilon):
    c = torch.tensor([[0.3, 1., 0.2], [0.8, 0., 0.6]], dtype=torch.float64)
    a, b = torch.tensor([0.3, 0.7]), torch.tensor([0.2, 0.3, 0.5])
    p, value, info = solve(c, a, b, epsilon)
    a, b = a.double() / a.sum().double(), b.double() / b.sum().double()
    a, b = a / a.sum(), b / b.sum()
    assert torch.allclose(p.sum(1), a, atol=1e-9, rtol=0)
    assert torch.allclose(p.sum(0), b, atol=1e-9, rtol=0)
    expected = (p * c).sum() + epsilon * (p * (p.log() - (a[:, None] * b[None]).log())).sum()
    assert torch.allclose(value, expected, atol=1e-9)
    assert info["marginal_error"] < 1e-9


def test_zero_mass_and_solver_failure():
    c = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    p, _, _ = solve(c, b=torch.tensor([1., 0.]))
    assert p[:, 1].eq(0).all()
    with pytest.raises(RuntimeError):
        solve(c * 7, a=torch.tensor([0.4, 0.6]), epsilon=0.01, max_iter=10, tol=1e-30)


def test_envelope_gradient_matches_fd():
    c = torch.tensor([[0.2, 0.7], [0.5, 0.1]], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([[0.3, -0.2], [0.1, 0.2]], dtype=torch.float64)
    gradient = torch.autograd.grad(regularized_ot(c, epsilon=.3), c)[0]
    h = 1e-5
    fd = (regularized_ot(c.detach() + h * direction, epsilon=.3) -
          regularized_ot(c.detach() - h * direction, epsilon=.3)) / (2 * h)
    assert float((gradient * direction).sum()) == pytest.approx(float(fd), abs=1e-7)


def test_sinkhorn_self_value_and_both_gradients():
    x = torch.tensor([[0., .2], [1., .4], [.1, .7]], dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    value = sinkhorn_divergence(x, y, epsilon=.4)
    gx, gy = torch.autograd.grad(value, [x, y])
    assert abs(float(value.detach())) < 1e-10
    assert gx.abs().max() < 1e-8 and gy.abs().max() < 1e-8
    self_cost = regularized_ot(quadratic(x, x), epsilon=.4)
    grad = torch.autograd.grad(self_cost, x)[0]
    h = 1e-5
    direction = torch.arange(6, dtype=torch.float64).reshape(3, 2) / 10
    xp, xm = x.detach() + h * direction, x.detach() - h * direction
    fd = (regularized_ot(quadratic(xp, xp), epsilon=.4) -
          regularized_ot(quadratic(xm, xm), epsilon=.4)) / (2 * h)
    assert float((grad * direction).sum()) == pytest.approx(float(fd), abs=1e-7)


def test_weighted_unequal_1d_known_transport():
    # 0(.25),2(.75) against 1(.5),3(.25),4(.25): cost .25+.25+.25+1 =1.75.
    x = torch.tensor([0., 2.], requires_grad=True)
    y = torch.tensor([1., 3., 4.])
    value = weighted_w2_1d(x, y, torch.tensor([.25, .75]), torch.tensor([.5, .25, .25]))
    assert float(value.detach()) == pytest.approx(1.75)
    assert torch.allclose(torch.autograd.grad(value, x)[0], torch.tensor([-.5, -1.]))


def test_sliced_fixed_directions_and_translation():
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    d = projections(2, 32, 91)
    assert torch.equal(d, projections(2, 32, 91))
    assert sliced_w2(x, x, directions=d) == 0
    shift = torch.tensor([2., -1.])
    assert float(sliced_w2(x, x + shift, directions=d)) == pytest.approx(float((d @ shift.double()).square().mean()))


def test_epsilon_centering_does_not_change_cost():
    x, y = torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[.2, .3], [1., 0.]])
    c = quadratic(x, y).clone()
    assert epsilon_scale(x, y) > 0
    assert torch.equal(c, quadratic(x, y))


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_orthonormal_block_geometry(factor):
    g = torch.Generator().manual_seed(9)
    z = torch.randn(2, 3, 2, 2, generator=g, dtype=torch.float64)
    x = block_up(z, factor)
    assert torch.allclose(block_down(x, factor), z)
    assert torch.allclose(x.square().sum(), z.square().sum())
    assert torch.allclose(lowpass(x, factor), x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bound", [0., .002, .025])
def test_actual_postcast_trust(dtype, bound):
    g = torch.Generator().manual_seed(45)
    x = (torch.randn(4, 2, 8, 8, generator=g) * 3).to(dtype)
    delta = torch.randn(4, 2, 8, 8, generator=g) * 9
    y, audit = trusted_state(x, delta, bound)
    assert (rms(y.float() - x.float()) <= bound).all()
    assert y.dtype == dtype and max(audit["actual_rms"]) <= bound
    if bound == 0:
        assert torch.equal(x, y)


@pytest.mark.parametrize("seed", range(5))
def test_exact_allocator_against_bruteforce(seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(4, 3, generator=g)
    labels = torch.randint(0, 3, (4, 3), generator=g)
    feasible = torch.rand(4, 3, generator=g) > .2
    feasible[:, 0] = True
    choice, value = assignment(u, labels, feasible, penalty=.7, capacity=1)
    candidates = []
    for indices in itertools.product(range(3), repeat=4):
        if not all(feasible[i, j] for i, j in enumerate(indices)):
            continue
        selected = [int(labels[i, j]) for i, j in enumerate(indices)]
        score = sum(float(u[i, j]) for i, j in enumerate(indices)) - .7 * sum(max(selected.count(k) - 1, 0) for k in range(3))
        candidates.append((score, indices))
    best = min(candidates, key=lambda p: (-p[0], p[1]))
    assert value == pytest.approx(best[0]) and tuple(choice) == best[1]
    independent, _ = assignment(u, labels, feasible, mode="independent")
    zero, _ = assignment(u, labels, feasible, penalty=0)
    assert zero == independent
    _, greedy_value = assignment(u, labels, feasible, penalty=.7, mode="greedy", capacity=1)
    assert value >= greedy_value - 1e-7


def test_quality_guard_and_noop_ties():
    p = torch.tensor([[2., 1., 3.], [2., 2., 1.]])
    active = torch.ones(2, 3, dtype=torch.bool)
    u, feasible = utilities(p, active)
    assert not feasible[0, 1] and not feasible[1, 2]
    assert assignment(u, torch.zeros(2, 3, dtype=torch.long), feasible, penalty=0)[0] == [2, 0]


def test_two_pass_vjp_against_direct_autograd():
    g = torch.Generator().manual_seed(20)
    matrix = torch.randn(8, 5, generator=g)
    encode = lambda x: torch.tanh(x.flatten(1) @ matrix)
    objective = lambda f: (f.mean(0) - .3).square().sum()
    x = torch.randn(3, 2, 2, 2, generator=g, requires_grad=True)
    expected = torch.autograd.grad(objective(encode(x)), x)[0]
    actual, _ = feature_vjp(encode, x.detach(), objective)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_quality_target_and_identical_clusters():
    f = torch.ones(5, 3)
    assert cluster(f).tolist() == [0] * 5
    weights, good = target_weights(torch.tensor([1., 2., 3., 4., 5.]), cluster(f))
    assert good.tolist() == [4, 3, 2]
    assert weights.sum() == pytest.approx(1.) and weights[:2].eq(0).all()
