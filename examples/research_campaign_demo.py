#!/usr/bin/env python
"""N6 demo: resumable architecture and weight-transform research campaigns.

Run:

    python examples/research_campaign_demo.py --root /tmp/uzen-n6-demo

The demo performs real NumPy forward passes. It does not train, fine-tune or
optimize weights, and it does not claim that a proxy metric is model quality.
The gate fails if pause/resume does not reproduce a fresh run, if a candidate is
duplicated, or if any benchmark record fails replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uzen.core.campaign import CampaignEngine, CampaignSpec


def _architecture_spec(seed: int, budget: int) -> CampaignSpec:
    return CampaignSpec.from_dict(
        {
            "name": "n6-architecture-campaign",
            "family": "transformer_decoder",
            "base_spec": {
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "d_ff": 32,
                "vocab_size": 64,
                "max_seq_len": 16,
            },
            "axes": [
                {"name": "d_model", "kind": "int", "low": 16, "high": 32, "step": 8},
                {"name": "n_layers", "kind": "int", "low": 1, "high": 2, "step": 1},
                {"name": "d_ff", "kind": "int", "low": 32, "high": 64, "step": 32},
            ],
            "objectives": [
                {"name": "parameters", "direction": "min"},
                {"name": "output_diversity", "direction": "max"},
                {"name": "quantization_rmse", "direction": "min"},
            ],
            "constraints": {"output_rms": {"min": 1e-6}},
            "budget": budget,
            "batch_size": 4,
            "strategy": "evolution",
            "seed": seed,
            "patience": 4,
            "parallel": 2,
            "evaluation": {"batch": 4, "seq": 8, "init_seed": 0, "input_seed": 0},
        }
    )


def _weight_spec(seed: int, budget: int) -> CampaignSpec:
    return CampaignSpec.from_dict(
        {
            "name": "n6-weight-transform-campaign",
            "family": "transformer_decoder",
            "base_spec": {
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 2,
                "d_ff": 32,
                "vocab_size": 64,
                "max_seq_len": 16,
            },
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
            "budget": budget,
            "batch_size": 4,
            "strategy": "grid",
            "seed": seed,
            "parallel": 2,
            "evaluation": {"batch": 4, "seq": 8, "init_seed": 0, "input_seed": 0},
        }
    )


def _print_front(title: str, report, limit: int = 4) -> None:
    print(f"\n{title}: {len(report.pareto)} Pareto candidate(s)")
    for entry in report.pareto[:limit]:
        metrics = entry["metrics"]
        values = ", ".join(
            f"{name}={metrics[name]:.6g}"
            for name in ("parameters", "output_diversity", "quantization_rmse",
                         "activation_agreement", "weight_sparsity")
            if name in metrics
        )
        print(f"  {entry['candidate']['digest'][:12]}  score={entry['score']:.4f}  {values}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UZEN N6 campaign demo")
    parser.add_argument("--root", default="/tmp/uzen-n6-demo")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    if args.budget < 4:
        parser.error("--budget must be at least 4")

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print("UZEN N6 - autonomous research campaigns")
    print(f"workspace: {root}")
    print(f"budget per campaign: {args.budget}; strategy: evolution / grid; no training")

    architecture_spec = _architecture_spec(args.seed, args.budget)
    engine = CampaignEngine(root / "campaigns-resume")
    paused = engine.run(architecture_spec, max_rounds=1)
    resumed = engine.run(architecture_spec)
    print(
        f"\narchitecture pause/resume: status={paused.status} "
        f"rounds={paused.rounds} -> {resumed.status} "
        f"trials={len(resumed.trials)} stop={resumed.stop_reason}"
    )
    _print_front("architecture Pareto", resumed)

    fresh_engine = CampaignEngine(root / "campaigns-fresh")
    fresh = fresh_engine.run(architecture_spec)
    print(
        f"fresh replay digest: {fresh.deterministic_digest()[:24]}... "
        f"resumed digest: {resumed.deterministic_digest()[:24]}..."
    )
    print(f"pause/resume deterministic: {fresh.deterministic_digest() == resumed.deterministic_digest()}")

    weight_spec = _weight_spec(args.seed, args.budget)
    weight_engine = CampaignEngine(root / "campaigns-weights")
    weight_report = weight_engine.run(weight_spec)
    print(
        f"\nweight-transform campaign: status={weight_report.status} "
        f"trials={len(weight_report.trials)} stop={weight_report.stop_reason}"
    )
    _print_front("weight-transform Pareto", weight_report)

    all_digests = engine.corpus.digests() + weight_engine.corpus.digests()
    verification = engine.corpus.verify(engine.corpus.digests())
    weight_verification = weight_engine.corpus.verify(weight_engine.corpus.digests())
    print(
        f"\ncorpus: {len(all_digests)} records; "
        f"architecture verified={verification['matched']}/{verification['checked']}; "
        f"weights verified={weight_verification['matched']}/{weight_verification['checked']}"
    )
    replay = engine.corpus.replay(engine.corpus.digests()[0])
    print(f"sample replay: digest={replay['digest'][:16]}... matches={replay['matches']}")

    architecture_digests = [trial.candidate.digest for trial in resumed.trials]
    gates = {
        "architecture_completed": resumed.status == "completed",
        "architecture_no_duplicate_candidates": len(set(architecture_digests)) == len(architecture_digests),
        "architecture_best_selected": resumed.best is not None,
        "architecture_pareto_present": bool(resumed.pareto),
        "pause_resume_matches_fresh": fresh.deterministic_digest() == resumed.deterministic_digest(),
        "architecture_corpus_replays": verification["matched"] == verification["checked"] == len(engine.corpus.digests()),
        "weight_campaign_completed": weight_report.status == "completed",
        "weight_trials_ok": all(trial.status == "ok" for trial in weight_report.trials),
        "weight_corpus_replays": weight_verification["matched"] == weight_verification["checked"] == len(weight_engine.corpus.digests()),
        "sample_replay_matches": replay["matches"],
    }
    report = {
        "schema": "uzen.n6.demo/1",
        "root": str(root),
        "architecture": resumed.to_dict(),
        "fresh_deterministic_digest": fresh.deterministic_digest(),
        "weight_campaign": weight_report.to_dict(),
        "corpus": {
            "records": len(all_digests),
            "architecture_verification": verification,
            "weight_verification": weight_verification,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "strict_plus": {
            "training": False,
            "fine_tuning": False,
            "backpropagation": False,
            "gradient_updates": False,
            "metrics": "inference-only proxy metrics, not model quality",
        },
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "n6_campaigns.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {report_path}")
    print("gates:")
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\ngate: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
