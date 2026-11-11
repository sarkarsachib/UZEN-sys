# STRICT_PLUS enforcement audit

This audit covers every route that can produce or alter weights. The rule is
unchanged: no training, fine-tuning, backpropagation, optimizer state or
gradient-based parameter update is permitted, and a caller cannot silently widen
the requested scope.

| Route | Enforcement point | Audit evidence |
| --- | --- | --- |
| Operation registry (NWPC) | `OperationRegistry.invoke` calls `StrictPlusEngine.authorize` before `spec.fn`; unknown training operations are refused. | `tests/test_core_substrate.py::test_registry_rejects_unregistered_operation`, `tests/test_strict_plus_routes.py::test_registry_route_rejects_training_operation` |
| NWPC tensor writes | `OperationContext.emit` calls `guard_mutation`; autograd-tracked tensors are rejected. | `tests/test_strict_plus_routes.py::test_nwpc_context_rejects_grad_tracking_write` |
| WPL compiler / runtime | `compile_stages` authorizes each stage's declared effect for the program scope before execution; the pipeline executes through the same registry. | `tests/test_strict_plus_routes.py::test_wpl_route_rejects_transform_outside_scope`, `tests/test_wpl_compiler.py` |
| Job service / SDK | `JobService.submit` accepts only `policy == "STRICT_PLUS"`, validates every operation against the registry and requested scope, and persists the policy. | `tests/test_jobs_and_interfaces.py::test_job_rejects_prohibited_operation`, `tests/test_jobs_and_interfaces.py::test_sdk_generate_modify_evaluate` |
| REST API | Submissions delegate to `JobService`; policy/scope violations surface as typed HTTP errors. | `tests/test_jobs_and_interfaces.py::test_api_job_lifecycle`, `tests/test_observability.py` |
| CLI | CLI commands delegate to the same service/SDK; there is no direct tensor-write path. `uzen policy` reports the active policy. | `uzen/interfaces/cli/main.py` (no separate mutation path) |
| ANE planner / search | Planner authorizes proposed stages and refuses prohibited effects before search or execution. | `tests/test_ane_and_search.py`, `tests/test_core_substrate.py::test_policy_rejects_plan_with_prohibited_stage` |
| Worker | Workers advertise `strict_plus: true`; jobs carry the policy and are refused by the service otherwise. | `tests/test_strict_plus_routes.py::test_worker_capabilities_declare_strict_plus` |
| PyTorch runtime | Forward passes run under `torch.inference_mode`; tensors requiring gradients are refused at conversion and execution. | `tests/test_strict_plus_routes.py::test_runtime_route_refuses_grad_tracking_tensor`, `tests/test_pytorch_runtime.py` |
| Artifact store | Verification rejects policy records that deny a stage or contain a prohibited effect. | `ArtifactStore._integrity_problems`, `tests/test_core_substrate.py::test_provenance_lineage_records_parents` |

## Remaining audit gaps

* The CLI does not have a dedicated policy-violation test because every CLI
  execution route delegates to `JobService`; the service test is the authoritative
  check.
* GPU/native backends are not implemented, so there is no GPU execution route to
  audit. Capability reporting says so explicitly.
* A malicious caller with process-level access can bypass Python-level guards by
  editing files directly; artifact checksums and the hash-chained audit log make
  such tampering detectable, not impossible.
