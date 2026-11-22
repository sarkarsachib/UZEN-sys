# N6 — Autonomous research campaigns

N6 adds a deterministic, resumable research loop on top of the existing engines. A
campaign proposes architecture or weight-transform candidates, evaluates each one with
a real NumPy forward pass, records every outcome, and reports the Pareto front. It never
trains, fine-tunes, backpropagates or applies gradient updates.

```text
campaign spec
    │
    ├── seeded planner ──► candidate batch ──► NumPy forward + package metrics
    │        ▲                                      │
    │        └──────── completed trials ◄───────────┘
    │
    ├── SQLite campaign/trial/event store (resume at batch boundaries)
    └── content-addressed benchmark corpus (replay/verify)
```

## Status

Implemented:

* architecture-field search and weight-transform search;
* deterministic `grid`, `random` and `evolution` planners;
* bounded batch evaluation with a real thread pool (independent NumPy passes);
* constraints, min-max weighted scoring and multi-objective Pareto filtering;
* durable SQLite campaign state, idempotent trials, pause/resume and event history;
* content-addressed benchmark records with replay and integrity checks;
* SDK, CLI, REST and MCP exposure;
* a demo that fails its gate if resume does not reproduce a fresh run or any record
  fails replay.

Not implemented:

* distributed campaign execution inside N6 itself (the local scheduler is in-process);
  see [N7_DISTRIBUTED_CAMPAIGNS.md](N7_DISTRIBUTED_CAMPAIGNS.md) and
  [N8_CAMPAIGN_CONTROL.md](N8_CAMPAIGN_CONTROL.md) for the N7/N8 extensions;
* task-quality metrics or any learned surrogate model;
* training, gradient-based search or fine-tuning of any kind;
* a latency objective (wall-clock latency is environment-dependent and is not part of
  the reproducible corpus).

## Campaign specification

```json
{
  "name": "transformer-width-study",
  "family": "transformer_decoder",
  "mode": "architecture",
  "base_spec": {
    "d_model": 16, "n_heads": 2, "n_layers": 1,
    "d_ff": 32, "vocab_size": 64, "max_seq_len": 16
  },
  "axes": [
    {"name": "d_model", "kind": "int", "low": 16, "high": 32, "step": 8},
    {"name": "n_layers", "kind": "int", "low": 1, "high": 2, "step": 1},
    {"name": "activation", "kind": "choice", "values": ["gelu", "relu", "silu"]}
  ],
  "objectives": [
    {"name": "parameters", "direction": "min"},
    {"name": "output_diversity", "direction": "max"}
  ],
  "constraints": {"output_rms": {"min": 1e-6}},
  "budget": 24,
  "batch_size": 4,
  "strategy": "evolution",
  "seed": 11,
  "patience": 4,
  "parallel": 2,
  "evaluation": {"batch": 4, "seq": 8, "init_seed": 0, "input_seed": 0},
  "write_corpus": true
}
```

`mode: "weights"` keeps the architecture fixed and searches `scale`, `sparsity` and
`quantize` knobs instead:

```json
{
  "name": "sparsification-study",
  "family": "transformer_decoder",
  "mode": "weights",
  "base_spec": {"d_model": 16, "n_heads": 2, "n_layers": 2,
                "d_ff": 32, "vocab_size": 64, "max_seq_len": 16},
  "axes": [
    {"name": "scale", "kind": "float", "low": 0.5, "high": 1.5, "step": 0.5},
    {"name": "sparsity", "kind": "float", "low": 0.0, "high": 0.5, "step": 0.25},
    {"name": "quantize", "kind": "choice", "values": ["none", "int8"]}
  ],
  "objectives": [
    {"name": "activation_agreement", "direction": "max"},
    {"name": "weight_sparsity", "direction": "max"}
  ],
  "budget": 12,
  "batch_size": 4,
  "strategy": "grid"
}
```

Architecture axes must name a real `base_spec` field; a typo is rejected rather than
silently ignored. Weight axes are restricted to the three transform knobs. Objectives
and constraints are restricted to computed metrics.

## Planners and determinism

| Strategy | Proposal source | Determinism |
| --- | --- | --- |
| `grid` | mixed-radix index over the finite axis domains | a pure function of the flat index |
| `random` | seeded per-axis sampling | a pure function of `(seed, round)` |
| `evolution` | mutation/crossover of best-first completed trials plus 25% immigrants | a pure function of ranked trials and `(seed, round)` |

Completed candidate digests are excluded from new proposals, and a candidate digest is
SHA-256 over `{mode, family, config}`. A fresh run and a run resumed at a batch boundary
produce the same candidate order, metrics and deterministic report digest. Timestamps
and wall-clock latency are excluded from that digest.

`max_rounds` pauses a campaign (status `paused`); calling `run` again resumes it.
`budget`, `patience` and grid exhaustion complete it (status `completed`). SQLite trial
keys are `(campaign_id, candidate_digest)`, so a retry cannot duplicate a trial.

## Metrics

All metrics are measured, not fabricated:

| Metric | Meaning |
| --- | --- |
| `parameters` | `WeightPackage.numel()` |
| `parameter_bytes` | actual tensor bytes in the package |
| `nodes` | NIR node count |
| `output_rms` | root-mean-square over all forward outputs |
| `output_diversity` | mean normalized pairwise distance between per-example output rows (0 for batch 1) |
| `output_bytes` | actual output array bytes |
| `weight_sparsity` | measured fraction of exact zeros in the package |
| `quantization_rmse` | relative L2 error of a symmetric int8 round-trip over matrices |
| `distance_from_base` | relative L2 parameter distance from the base package (weight campaigns) |
| `activation_agreement` | `1 - relative L2` of candidate outputs versus the base package on the same inputs (weight campaigns) |

These are **inference-only proxy metrics**. `output_diversity` is not task accuracy,
`activation_agreement` is not semantic equivalence, and a Pareto-optimal candidate is
not claimed to be a better model. The corpus records that distinction in its schema and
the docs repeat it.

Constraints are checked after metrics are measured. A failed constraint is a recorded
`constraint` trial, not a discarded one. Evaluator errors are recorded as `invalid`
(malformed architecture / binding) or `error` (execution failure), with the exception
text. Only `ok` trials enter scoring and the corpus.

## Scoring and Pareto

Each objective is min-max normalized over currently eligible trials, direction is
applied (`min` becomes `1 - normalized`), and the weighted sum is divided by total
weight. The best candidate is the highest score, with candidate digest as a stable
tie-break. Pareto dominance uses raw metrics and supports any mix of min/max objectives.
A candidate is dominated only if another is no worse on every objective and strictly
better on at least one.

## Persistence

`<root>/campaigns/campaigns.sqlite3` contains campaigns, trials and append-only events.
Trial rows store the candidate, status, metrics, error, package fingerprint, graph digest
and constraint failures. Resume reconstructs the spec, trials and scores from SQLite.
The store uses WAL and one short-lived connection per operation.

A process killed in the middle of a batch resumes at the next batch boundary: completed
trials are never repeated, but the partially written batch is not replayed. This is an
explicit batch-granularity guarantee, not a transaction-per-candidate claim.

## Benchmark corpus

Successful trials are written to `<root>/campaigns/benchmarks/<digest>.json`. The digest
covers the campaign spec, candidate, evaluation settings, measured metrics, package
fingerprint, graph digest, engine version and environment; `recorded_at` and the digest
field itself are excluded. Writes are atomic and idempotent.

`verify` reads each record, rebuilds the candidate from the stored spec, re-executes it
and compares status, fingerprints and metrics with a relative tolerance of `1e-9`
(integers must match exactly). `replay` does the same for one record. A modified file
that no longer hashes to its filename is rejected with `artifact.integrity`.

Cross-host BLAS differences can change the last bits of a float; verification reports
the measured delta instead of claiming bitwise identity. Environment versions are stored
so a mismatch is visible.

## Interfaces

CLI:

```bash
uzen campaign run --spec-file campaign.json
uzen campaign run --spec-file campaign.json --max-rounds 2
uzen campaign list
uzen campaign show <campaign_id>
uzen campaign events <campaign_id>
uzen campaign verify
uzen campaign replay <benchmark_digest>
```

SDK:

```python
from uzen import Uzen

client = Uzen(root="./uzen-workspace")
report = client.run_campaign(spec_dict)
print(report["best"], report["pareto"])
print(client.verify_campaign_corpus())
```

REST:

```text
POST /v1/campaigns
GET  /v1/campaigns
GET  /v1/campaigns/{campaign_id}
GET  /v1/campaigns/{campaign_id}/events
GET  /v1/campaigns/{campaign_id}/corpus
POST /v1/campaigns/verify
GET  /v1/benchmarks/{digest}
```

MCP tools: `uzen.run_campaign`, `uzen.get_campaign`, `uzen.list_campaigns`,
`uzen.verify_campaign_corpus`, `uzen.replay_benchmark`.

The REST run endpoint executes synchronously in the control-plane process. For a large
budget, prefer the CLI/SDK with a pause (`max_rounds`) or run it as a normal job; there
is no background campaign daemon in N6.

## Architecture engineering connectivity

The audit added the missing NASE surfaces rather than leaving them internal:

* `compose_architectures` composes stored architectures sequentially, in parallel or
  with residuals, validates the merged graph, executes it, persists the new artifact and
  records the component artifacts as lineage parents. Component weights are not
  transplanted; the new graph is initialized with labelled structured random weights.
* `edit_architecture` performs `bypass`, `truncate` or `substitute` on a stored graph.
  It preserves surviving tensors, refuses a substitution that would need silently
  generated parameters, executes the result and persists it with parent lineage.
* The reconstructed graph now preserves its family spec, so `truncate_layers` sees the
  real `n_layers` after a store round-trip.
* Composition selects the semantic output (`out`, `logits`, `y`, `reconstruction`,
  `eps_pred`) instead of whichever output name sorted first after JSON serialization.

## Verification performed

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_research_campaigns.py
.venv/bin/python -m pytest -o addopts="" -q tests/test_architecture_editing.py
.venv/bin/python examples/research_campaign_demo.py --root /tmp/uzen-n6-demo
```

Results in this workspace:

* `24 passed` campaign tests (spec/planner determinism, real metrics, invalid trials,
  constraints, scoring/Pareto, pause/resume equivalence, parallel equivalence, conflict,
  events, weight transforms, corpus integrity, CLI, REST, SDK).
* `7 passed` architecture-editing tests (composition, shape-mismatch refusal, bypass,
  truncation spec preservation, substitution, REST, MCP campaign/tool exposure).
* demo: 12 architecture trials, 12 weight-transform trials, 24/24 corpus records replayed,
  pause/resume digest equal to a fresh run, gate `PASS`.

The full suite is the final authority; see the repository's test output for the current
count and skips. The four CUDA GPU tests still skip on hosts without a driver/device.

## Honest limitations

1. **No model-quality signal.** There is no dataset, loss or accuracy metric; the
   objectives measure size, signal diversity, quantization error and weight
   transformations. Do not present a Pareto front as benchmark superiority.
2. **In-process scheduler.** Batches use a thread pool and NumPy. There is no campaign
   distribution across N4 workers, no multi-host store and no resumable worker lease for
   a single candidate.
3. **Batch-boundary resume.** A crash inside a batch loses that batch's unwritten trials;
   it never duplicates completed ones.
4. **No latency objective.** Timing is intentionally excluded from scoring and corpus
   digests; use MAR's exploratory timing separately.
5. **Corpus reproducibility is version-sensitive.** Same engine + same NumPy usually
   matches at `1e-9`; different BLAS builds may differ in low bits and are reported.
6. **The corpus is not a held-out benchmark.** It is a replayable record of measured
   proxy evaluations for the exact campaign spec.
7. **Weight transforms are lossy by design.** Pruning and int8 simulation change
   parameters; agreement and distance are reported, not hidden.
8. **REST runs synchronously.** A very large budget can exceed an HTTP timeout.
