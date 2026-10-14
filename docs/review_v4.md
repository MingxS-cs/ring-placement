# V4 independent review

Read-only implementation review and a separate method/requirement review both passed without remaining blocking findings.

The implementation review reproduced an early-transfer-completion bug with a legal 25 mm longitudinal residual on a 45 mm transfer. The V4-specific completion guard fixes it, and a real contact-physics regression verifies the entire reference completes before lowering. Review confirmed that vetoing the stage transition changes no physical state and preserves the historical simulator/V3 binding.

Review verified 96-case stratification and conditional bearing-minus-ring workspace intervals, shared execution for all three profiles, all failures/exception denominators, and separate physical ring/tool/command/action records. Development parsing uses 13 original training and 3 validation sources; all 78 selection folds fit only the other 12 training sources. New reserved labels are not training input.

The method review checked all 288 raw outcomes against the published 96/96 results, all per-cell scores, point counts, frozen hashes and fresh fidelity/sensitivity summaries. It confirmed the README states the worse-than-straight mean fidelity result, geometric grasp/Z/gripper stages, simulator-known positions and deferred public repository accurately. The official challenge page was reread on 2026-10-14.

The reported training distance span was added following review: 66.2–201.1 mm, making the 45–160 mm test partly short-distance extrapolation. The separate fidelity source was completed/tested before unsealing and recorded in the output, but it lacks an independent pre-unseal source receipt. This scope limitation is disclosed; no fidelity source change or model retuning occurred after label access.

Clean-clone execution is recorded separately in [reproduction_v4.md](reproduction_v4.md). Review does not replace executed verification.
