"""N6 research campaign tests.

These tests execute real NumPy forwards over generated architectures and real
weight transforms. Nothing here mocks a metric, a scheduler or a corpus record.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from uzen.core.campaign import (
    CampaignEngine,
    CampaignSpec,
    Candidate,
    Evaluator,
    apply_transforms,
    dominates,
    pareto_front,
    score_trials,
)
from uzen.core.campaign.corpus import compare_records
from uzen.core.campaign.models import Trial
from uzen.core.campaign.space import GridExhausted, grid_candidate, grid_size, propose
from uzen.core.nws.families import build_architecture
from uzen.errors import (
    ArtifactIntegrityError,
    CampaignConflict,
    InvalidOperationParameters,
)

BASE_SPEC = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "d_ff": 32,
    "vocab_size": 64,
    "max_seq_len": 8,
}


def make_spec(**overrides):
    payload = {
        "name": "test-campaign",
        "family": "transformer_decoder",
        "base_spec": dict(BASE_SPEC),
        "axes": [
            {"name": "d_model", "kind": "int", "low": 8, "high": 16, "step": 8},
            {"name": "n_layers", "kind": "int", "low": 1, "high": 2, "step": 1},
        ],
        "objectives": [
            {"name": "parameters", "direction": "min"},
            {"name": "output_diversity", "direction": "max"},
        ],
        "budget": 4,
        "batch_size": 2,
        "strategy": "random",
        "seed": 7,
        "evaluation": {"batch": 2, "seq": 3},
    }
    payload.update(overrides)
    return CampaignSpec.from_dict(payload)


# --------------------------------------------------------------------------- #
# specification and planning
# --------------------------------------------------------------------------- #


def test_spec_digest_is_stable_and_ignores_campaign_id():
    first = make_spec()
    second = make_spec()
    assert first.digest() == second.digest()
    assert first.candidate_id == second.candidate_id
    renamed = make_spec(campaign_id="explicit")
    assert renamed.digest() == first.digest()
    assert renamed.candidate_id == "explicit"


def test_spec_rejects_axis_not_in_base_spec():
    with pytest.raises(InvalidOperationParameters, match="not a base_spec field"):
        make_spec(
            axes=[{"name": "width_typo", "kind": "int", "low": 8, "high": 16, "step": 8}]
        )


def test_spec_rejects_unknown_objective_and_bad_budget():
    with pytest.raises(InvalidOperationParameters, match="unknown campaign objective"):
        make_spec(objectives=[{"name": "quality", "direction": "max"}])
    with pytest.raises(InvalidOperationParameters, match="batch_size cannot exceed budget"):
        make_spec(budget=2, batch_size=4)


def test_spec_rejects_weight_axis_in_architecture_mode():
    with pytest.raises(InvalidOperationParameters, match="collides with a weight"):
        make_spec(
            axes=[{"name": "scale", "kind": "float", "low": 0.5, "high": 1.5, "step": 0.5}]
        )


def test_grid_candidates_are_deterministic_and_exhaust():
    spec = make_spec(strategy="grid")
    assert grid_size(spec) == 4
    first = grid_candidate(spec, 0)
    assert first.digest == grid_candidate(spec, 0).digest
    assert grid_candidate(spec, 3).digest != first.digest
    with pytest.raises(GridExhausted):
        grid_candidate(spec, 4)
    assert [item.digest for item in propose(spec, [], round_index=0, count=4)] == [
        grid_candidate(spec, index).digest for index in range(4)
    ]


def test_random_proposals_are_seeded_and_skip_completed():
    spec = make_spec()
    first = propose(spec, [], round_index=0, count=2)
    repeated = propose(spec, [], round_index=0, count=2)
    assert [item.digest for item in first] == [item.digest for item in repeated]
    later = propose(spec, first, round_index=1, count=2)
    assert {item.digest for item in first}.isdisjoint({item.digest for item in later})
    skipped = propose(spec, first[:1], round_index=0, count=2)
    assert first[0].digest not in {item.digest for item in skipped}


def test_evolution_uses_ranked_parent_and_changes_a_coordinate():
    spec = make_spec(strategy="evolution")
    parent = grid_candidate(spec, 0)
    trial = Trial(candidate=parent, round_index=0, sequence=0, status="ok", metrics={})
    children = propose(spec, [trial], round_index=1, count=2, ranked=[trial])
    assert children
    assert all(child.digest != parent.digest for child in children)


# --------------------------------------------------------------------------- #
# evaluation, scoring and Pareto
# --------------------------------------------------------------------------- #


def test_evaluator_metrics_match_real_package_and_forward():
    spec = make_spec()
    candidate = grid_candidate(spec, 0)
    metrics, fingerprint, graph_digest = Evaluator(spec).evaluate(candidate)
    definition = build_architecture(spec.family, candidate.spec)
    package = definition.build_package(seed=0)
    assert metrics["parameters"] == package.numel()
    assert metrics["parameter_bytes"] == package.nbytes()
    assert metrics["nodes"] == len(definition.graph.nodes)
    assert metrics["output_rms"] > 0
    assert np.isfinite(metrics["output_diversity"])
    assert len(fingerprint) == 64
    assert len(graph_digest) == 64
    assert metrics["quantization_rmse"] >= 0


def test_invalid_architecture_is_recorded_not_hidden(tmp_path):
    spec = make_spec(
        base_spec={**BASE_SPEC, "n_heads": 5},
        axes=[
            {"name": "d_model", "kind": "int", "low": 15, "high": 16, "step": 1},
            {"name": "n_layers", "kind": "int", "low": 1, "high": 1, "step": 1},
        ],
        budget=1,
        batch_size=1,
    )
    report = CampaignEngine(tmp_path).run(spec)
    assert len(report.trials) == 1
    assert report.trials[0].status in ("invalid", "error")
    assert report.trials[0].metrics == {}
    assert report.best is None
    assert report.corpus_digest  # corpus directory exists even with no records
    assert CampaignEngine(tmp_path).corpus.digests() == []


def _trial(metrics, digest, status="ok"):
    candidate = Candidate.build("architecture", "x", (digest,), {"spec": {}, "transforms": {}})
    return Trial(
        candidate=candidate,
        round_index=0,
        sequence=0,
        status=status,
        metrics=dict(metrics),
    )


def test_score_and_pareto_use_objective_directions():
    spec = make_spec()
    a = _trial({"parameters": 10, "output_diversity": 1.0}, "a")
    b = _trial({"parameters": 20, "output_diversity": 1.0}, "b")
    c = _trial({"parameters": 20, "output_diversity": 2.0}, "c")
    ranked = score_trials([a, b, c], spec.objectives)
    assert ranked[0] is a  # tied weighted score with c, digest breaks the tie
    assert a.score == pytest.approx(0.5)
    assert c.score == pytest.approx(0.5)
    assert b.score is not None and b.score < a.score
    assert dominates(a.metrics, b.metrics, spec.objectives)
    assert not dominates(a.metrics, c.metrics, spec.objectives)
    front = pareto_front([a, b, c], spec.objectives)
    assert front == [a, c]


def test_constraints_mark_trials_and_leave_no_best(tmp_path):
    spec = make_spec(constraints={"parameters": {"max": 1}})
    report = CampaignEngine(tmp_path).run(spec)
    assert len(report.trials) == 4
    assert all(trial.status == "constraint" for trial in report.trials)
    assert report.best is None
    assert all(trial.constraint_failures for trial in report.trials)


# --------------------------------------------------------------------------- #
# persistence, resume, determinism
# --------------------------------------------------------------------------- #


def test_pause_and_resume_reproduce_a_fresh_run(tmp_path):
    spec = make_spec()
    resumed_root = tmp_path / "resumed"
    fresh_root = tmp_path / "fresh"
    engine = CampaignEngine(resumed_root)
    paused = engine.run(spec, max_rounds=1)
    assert paused.status == "paused"
    assert len(paused.trials) == 2
    resumed = engine.run(spec)
    fresh = CampaignEngine(fresh_root).run(spec)
    assert resumed.status == "completed"
    assert resumed.stop_reason == "budget"
    assert resumed.deterministic_digest() == fresh.deterministic_digest()


def test_completed_campaign_does_not_spend_more_budget(tmp_path):
    spec = make_spec()
    engine = CampaignEngine(tmp_path)
    first = engine.run(spec)
    second = engine.run(spec)
    assert len(first.trials) == len(second.trials) == 4
    assert second.deterministic_digest() == first.deterministic_digest()


def test_parallel_batches_match_sequential_results(tmp_path):
    spec = make_spec()
    sequential = CampaignEngine(tmp_path / "one").run(make_spec(parallel=1))
    parallel = CampaignEngine(tmp_path / "two").run(make_spec(parallel=2))
    assert [trial.candidate.digest for trial in sequential.trials] == [
        trial.candidate.digest for trial in parallel.trials
    ]
    for left, right in zip(sequential.trials, parallel.trials):
        assert left.status == right.status
        assert left.metrics == right.metrics


def test_same_id_with_different_spec_conflicts(tmp_path):
    engine = CampaignEngine(tmp_path)
    engine.run(make_spec(campaign_id="fixed"))
    with pytest.raises(CampaignConflict):
        engine.run(make_spec(campaign_id="fixed", seed=99))


def test_no_resume_refuses_an_existing_trial_history(tmp_path):
    engine = CampaignEngine(tmp_path)
    engine.run(make_spec(campaign_id="fixed"))
    with pytest.raises(CampaignConflict, match="already has 4 trials"):
        engine.run(make_spec(campaign_id="fixed"), resume=False)


def test_events_record_creation_and_trials(tmp_path):
    spec = make_spec()
    engine = CampaignEngine(tmp_path)
    report = engine.run(spec)
    events = engine.events(report.campaign_id)
    kinds = [event["kind"] for event in events]
    assert "campaign.created" in kinds
    assert kinds.count("trial.recorded") == 4
    assert kinds[-1] == "campaign.updated"


# --------------------------------------------------------------------------- #
# weight transforms and parameter campaigns
# --------------------------------------------------------------------------- #


def test_weight_transforms_are_measured_and_do_not_mutate_source():
    definition = build_architecture("transformer_decoder", BASE_SPEC)
    package = definition.build_package(seed=0)
    before = package.fingerprint()
    transformed, diagnostics = apply_transforms(
        package, {"scale": 0.5, "sparsity": 0.5, "quantize": "int8"}
    )
    assert package.fingerprint() == before
    assert transformed.fingerprint() != before
    assert diagnostics["realized_sparsity"] > 0.4
    assert diagnostics["quantization_rmse"] > 0
    assert transformed.numel() == package.numel()


def test_weight_campaign_runs_real_forward_and_corpus(tmp_path):
    spec = CampaignSpec.from_dict(
        {
            "name": "weight-campaign",
            "family": "transformer_decoder",
            "base_spec": dict(BASE_SPEC),
            "mode": "weights",
            "axes": [
                {"name": "scale", "kind": "float", "low": 0.5, "high": 1.5, "step": 0.5},
                {"name": "sparsity", "kind": "float", "low": 0.0, "high": 0.5, "step": 0.25},
                {"name": "quantize", "kind": "choice", "values": ["none", "int8"]},
            ],
            "objectives": [
                {"name": "activation_agreement", "direction": "max"},
                {"name": "weight_sparsity", "direction": "max"},
            ],
            "budget": 4,
            "batch_size": 2,
            "strategy": "grid",
            "evaluation": {"batch": 2, "seq": 3},
        }
    )
    engine = CampaignEngine(tmp_path)
    report = engine.run(spec)
    assert report.status == "completed"
    assert all(trial.status == "ok" for trial in report.trials)
    assert all("activation_agreement" in trial.metrics for trial in report.trials)
    assert all(
        trial.candidate.transforms["quantize"] in ("none", "int8")
        for trial in report.trials
    )
    assert len(engine.corpus.digests()) == 4


# --------------------------------------------------------------------------- #
# benchmark corpus
# --------------------------------------------------------------------------- #


def test_corpus_replay_and_integrity(tmp_path):
    spec = make_spec(write_corpus=True)
    engine = CampaignEngine(tmp_path)
    engine.run(spec)
    digests = engine.corpus.digests()
    assert len(digests) == 4
    replay = engine.replay(digests[0])
    assert replay["matches"], replay["differences"]
    assert replay["record"]["schema"] == "uzen.benchmark/1"
    assert engine.verify_corpus()["matched"] == 4

    path = engine.corpus.path_for(digests[0])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["metrics"]["parameters"] += 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="does not hash"):
        engine.corpus.get(digests[0])


def test_compare_records_honours_tolerance():
    recorded = {"status": "ok", "metrics": {"parameters": 100, "output_rms": 1.0}}
    recomputed = {"status": "ok", "metrics": {"parameters": 100, "output_rms": 1.0000001}}
    assert compare_records(recorded, recomputed, tolerance=1e-6) == []
    recorded["metrics"]["parameters"] = 101
    differences = compare_records(recorded, recomputed, tolerance=1e-9)
    assert {item["field"] for item in differences} == {
        "metrics.parameters",
        "metrics.output_rms",
    }


# --------------------------------------------------------------------------- #
# interfaces
# --------------------------------------------------------------------------- #


def test_cli_run_list_verify(tmp_path, capsys):
    from uzen.interfaces.cli.main import main

    spec_json = json.dumps(make_spec().to_dict())
    assert main(["--root", str(tmp_path), "campaign", "run", "--spec", spec_json]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["trials_evaluated"] == 4
    assert main(["--root", str(tmp_path), "campaign", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["campaigns"][0]["campaign_id"] == output["campaign_id"]
    assert main(["--root", str(tmp_path), "campaign", "verify"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["matched"] == 4


def test_rest_campaign_roundtrip(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from uzen.interfaces.api.app import create_app

    client = TestClient(create_app(root=tmp_path))
    response = client.post("/v1/campaigns", json={"spec": make_spec().to_dict()})
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["status"] == "completed"
    campaign_id = report["campaign_id"]
    assert client.get(f"/v1/campaigns/{campaign_id}").status_code == 200
    listed = client.get("/v1/campaigns").json()["campaigns"]
    assert listed[0]["campaign_id"] == campaign_id
    verified = client.post("/v1/campaigns/verify", json={}).json()
    assert verified["matched"] == 4
    assert client.get(f"/v1/campaigns/{campaign_id}/events").status_code == 200


def test_sdk_local_campaign_roundtrip(tmp_path):
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path)
    report = client.run_campaign(make_spec().to_dict())
    assert report["status"] == "completed"
    assert client.campaign(report["campaign_id"])["campaign_id"] == report["campaign_id"]
    assert client.campaigns()[0]["campaign_id"] == report["campaign_id"]
    assert client.verify_campaign_corpus()["matched"] == 4
