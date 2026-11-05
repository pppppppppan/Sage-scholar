# Router evaluation

The Router uses Ollama Qwen3.5:2b for semantic intent. SourceResolver validates actual document identities; PolicyValidator enforces permissions. The model cannot change hard permissions by returning a different classification.

The recorded **2026-11-04 full-final** regression used 144 candidate-labeled behavior cases on macOS ARM64, with local Qwen3.5:2b. Accuracy was 90.28%; Clarify precision/recall/F1 were 73.68% / 82.35% / 77.78%. There were 17 Clarify, 58 Research, 32 Direct and 37 QA gold cases. No cloud inference was used.

This dataset was inspected during prompt changes and is not independent holdout evidence. Relative to the earlier V2 snapshot, QA lost one correct prediction and Research lost three; higher overall accuracy does not eliminate those regressions. Other experimental snapshots are retained but are not silently substituted for the selected result.

- [Error analysis, per-class results and commands](../evals/reports/router-clarify/README.md)
- [Full-final machine-readable report, model/prompt/code hashes](../evals/reports/router-clarify/full-final/20261104T005230235546-router.json)
- [Full-final readable report](../evals/reports/router-clarify/full-final/20261104T005230235546-router.md)
- [Router V2 evaluation setup](../evals/ROUTER_V2.md)
- [Eval modes and quality gates](../evals/README.md)

Deployment changes only the allowlist of trusted self-managed Ollama hosts. It does not alter the classification prompt, model settings or labels. Historical results are not re-measured by a Docker healthcheck.
