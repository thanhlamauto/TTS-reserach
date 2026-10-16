from collections import Counter
import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from otlab.adapter import FrozenXlaAdapter
from otlab.analysis import prompt_bootstrap, terminal_metrics
from otlab.io import validate_checkpoint
from otlab.noise import NoiseTape, new_tape
from otlab.proximal import proximal_edit
from otlab.runner import Cache, checked_inputs, cold_accounting, make_plan, replay
from otlab.toy import ToyAdapter, ToyEncoder, trusted_loop
from otlab.worker import WorkerClient


@pytest.fixture
def cp():
    a = ToyAdapter(0)
    g = torch.Generator().manual_seed(81)
    x = torch.randn(8, 2, 8, 8, generator=g).to(torch.bfloat16)
    return dict(schema="otlab.checkpoint.v2", meta=dict(prompt="toy", prompt_id=0, bank_id="b", prefix_id="p",
                next_index=50, clean_prediction_index=50, schedule_sha=a.schedule_sha,
                model_id="toy", model_revision="toy-v1", vae_id="toy", state_dtype=str(x.dtype)),
                source_xt=x, source_clean=x.float(), bank_clean=torch.randn(16, 2, 8, 8, generator=g),
                bank_reward=torch.linspace(-1., 1., 16), signal_scale=.5)


@pytest.mark.parametrize("key", ["source_terminal_reward", "source_terminal_image", "none_reward"])
def test_terminal_leakage_rejected(cp, key):
    cp[key] = torch.ones(8)
    with pytest.raises(ValueError):
        validate_checkpoint(cp)
    cp.pop(key)
    cp["meta"][key] = "not permitted even in metadata"
    with pytest.raises(ValueError):
        validate_checkpoint(cp)


def test_stale_clean_rejected(cp):
    validate_checkpoint(cp)
    cp["meta"]["clean_prediction_index"] = 49
    with pytest.raises(ValueError, match="stale"):
        validate_checkpoint(cp)


def test_shared_noise_and_method_independence(cp):
    tape = new_tape(9, 0, 0, [0, 1], (2, 4, 4), cp["meta"]["schedule_sha"])
    selected = tape.at(51, [1, 0, 1, 1])
    assert torch.equal(selected[0], selected[2]) and torch.equal(selected[2], selected[3])
    assert not torch.equal(selected[0], selected[1])
    assert torch.equal(tape.variance_noise,
                       new_tape(9, 0, 0, [0, 1], (2, 4, 4), cp["meta"]["schedule_sha"]).variance_noise)
    assert not torch.equal(tape.variance_noise,
                           new_tape(9, 0, 1, [0, 1], (2, 4, 4), cp["meta"]["schedule_sha"]).variance_noise)
    with pytest.raises(ValueError):
        tape.at(49, [0])
    with pytest.raises(ValueError):
        tape.at(51, [99])


def test_terminal_pool_ties_and_no_cross_method_pooling():
    bank, source = torch.ones(16), torch.ones(8)
    m = terminal_metrics(source, bank)
    assert m["top8_source_contribution"] == 0
    assert m["top8_indices"] == list(range(8))
    better = terminal_metrics(source + 1, bank)
    assert better["pool_top8_mean"] == 2
    assert terminal_metrics(source, bank)["pool_top8_mean"] == 1


def test_bootstrap_clusters_prompts_before_sampling():
    rows = []
    for p in range(2):
        for rep in range(4):
            for method in ["none", "ot"]:
                rows.append(dict(prompt_id=p, bank_id="b", prefix_id="x", replicate=rep, method=method,
                                 reward=(p + 1) if method == "ot" else 0))
    r = prompt_bootstrap(rows, "ot", "none", "reward", count=1000)
    assert r["prompts"] == 2 and r["conditional_units"] == 8 and r["delta"] == 1.5
    assert r["ci95"] == [1., 2.]
    with pytest.raises(ValueError, match="duplicate"):
        prompt_bootstrap(rows + [rows[0]], "ot", "none", "reward")
    with pytest.raises(ValueError, match="unpaired"):
        prompt_bootstrap(rows[1:], "ot", "none", "reward")


def test_hash_locked_plan_and_trusted_replay(tmp_path, cp):
    adapter = ToyAdapter(0)
    tape = new_tape(10, 0, 0, range(8), (2, 8, 8), adapter.schedule_sha)
    torch.save(cp, tmp_path / "cp.pt")
    torch.save(tape.bundle(), tmp_path / "noise.pt")
    torch.save({"latent": trusted_loop(cp["source_xt"], 50, 100, range(8), tape)}, tmp_path / "trusted.pt")
    row = dict(prompt_id=0, bank_id="b", prefix_id="p", replicate=0,
               checkpoint="cp.pt", noise="noise.pt", trusted_none="trusted.pt")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    plan = make_plan(manifest, tmp_path / "plan")
    assert len(plan["jobs"]) == 9
    proof = replay(plan, lambda _: adapter, tmp_path / "proof.json")
    assert proof["passed"] and proof["records"][0]["trusted_latent_bitwise"]
    cp["source_xt"][0, 0, 0, 0] += 1
    torch.save(cp, tmp_path / "cp.pt")
    with pytest.raises(ValueError, match="changed"):
        checked_inputs(plan["jobs"][0]["input"])


def test_cache_fingerprint_and_corruption(tmp_path):
    c = Cache(tmp_path, {"encoder": "A"})
    value, hit = c.get("stage", {"noise": 1}, lambda: {"x": torch.ones(1)})
    assert not hit
    _, hit = c.get("stage", {"noise": 1}, lambda: pytest.fail("cache miss"))
    assert hit
    _, hit = Cache(tmp_path, {"encoder": "B"}).get("stage", {"noise": 1}, lambda: {"x": torch.zeros(1)})
    assert not hit
    for path in tmp_path.glob("*.pt"):
        path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        c.get("stage", {"noise": 1}, lambda: None)


@pytest.mark.parametrize("loss", ["sinkhorn", "mean", "sliced"])
def test_proximal_active_finite_and_postcast_bound(cp, loss):
    result = proximal_edit(cp, ToyEncoder().encode_clean, loss=loss)
    assert torch.isfinite(result["entry_states"]).all()
    assert max(result["trust"]["actual_rms"]) <= .025
    assert any(result["trust"]["active"])
    assert len(result["losses"]) == 3


class Config(dict):
    __getattr__ = dict.__getitem__


class FakeSampler:
    def __init__(self):
        self.batch = 4
        self.cfg = dict(steps=100, ddim_eta=1, cfg_scale=7.5)
        self.timesteps = list(range(99, -1, -1))
        self.scheduler = SimpleNamespace(alphas_cumprod=torch.linspace(.9, .1, 100), final_alpha_cumprod=torch.tensor(1.),
                                         config=Config(prediction_type="epsilon", clip_sample=False))
        self.coefficients = [torch.tensor([.8 + i / 1000, .6 - i / 1000, .9, .25, .05]) for i in range(100)]
        self.rt = SimpleNamespace(device=torch.device("cpu"), xm=SimpleNamespace(mark_step=lambda: None), sync=lambda: None)
        self.pipe = SimpleNamespace(unet=lambda model_input, timestep, **kwargs: ((model_input.float() * .05 + float(timestep) * .001).to(torch.bfloat16),))

    def embedding(self, prompt):
        return torch.zeros(8, 1, 1)


def update(x, eps, c, noise):
    x0 = (x.float() - c[1] * eps) / c[0]
    return c[2] * x0 + c[3] * eps + c[4] * noise


def test_frozen_adapter_matches_independent_legacy_loop_and_current_clean():
    sampler = FakeSampler()
    adapter = FrozenXlaAdapter(sampler, None, "test", update, dict(backend="fake"))
    x = torch.randn(4, 2, 4, 4, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    tape = new_tape(32, 0, 0, range(4), (2, 4, 4), adapter.schedule_sha)
    legacy = x.clone()
    for k in range(50, 100):
        pred = sampler.pipe.unet(torch.cat([legacy, legacy]), torch.tensor(sampler.timesteps[k]))[0]
        unconditional, conditional = pred.chunk(2)
        eps = unconditional.float() + 7.5 * (conditional.float() - unconditional.float())
        legacy = update(legacy, eps, sampler.coefficients[k], tape.at(k, range(4))).to(torch.bfloat16)
    actual = adapter.advance(x, 50, 100, range(4), tape)
    assert torch.equal(legacy, actual)
    assert adapter.counters["denoise_sample_steps"] == 200
    current = adapter.predict_clean(x, 50)
    assert not torch.equal(current, adapter.predict_clean(x, 49))
    duplicated = adapter.advance(x[[0, 0]], 50, 54, [0, 0], tape)
    assert torch.equal(duplicated[0], duplicated[1])
    assert adapter.counters["denoise_padding_sample_steps"] == 8


def test_actual_file_worker_roundtrip_and_closed_schema(tmp_path, cp):
    config = tmp_path / "worker.json"
    config.write_text(json.dumps(dict(mode="synthetic_test_only", vae=dict(identity="toy"))))
    worker = WorkerClient(sys.executable, config, tmp_path / "requests")
    try:
        result = worker.call("proximal", cp, dict(loss="mean"))
        assert result["entry_states"].shape == cp["source_xt"].shape
        assert worker.identity["mode"] == "synthetic_test_only"
        leaked = copy.deepcopy(cp)
        leaked["source_terminal_reward"] = torch.ones(8)
        with pytest.raises(RuntimeError, match="schema"):
            worker.call("proximal", leaked, dict(loss="mean"))
    finally:
        worker.close()
    assert worker.process.poll() == 0


def test_budget_auxiliary_padding_and_noop_accounting():
    assert cold_accounting("none", initial_aux=24)["cold_nfe"] == 2400
    assert cold_accounting("p1", active_proposals=24, initial_aux=8)["cold_nfe"] == 2496
    assert cold_accounting("p1", active_proposals=24, initial_aux=24)["cold_nfe"] == 2512
    assert cold_accounting("p2", initial_aux=8)["cold_nfe"] == 2408
    assert cold_accounting("p3")["cold_nfe"] is None
    padded = cold_accounting("p1", active_proposals=17, initial_aux=24, physical_batch=4)
    assert padded["cold_padding_nfe"] == 15
    assert padded["cold_nfe_with_padding"] == padded["cold_nfe"] + 15
