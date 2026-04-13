This branch is `codex/dc-h200-sweep`.

Purpose:

- keep the full runnable `codex/dc-training` training bundle
- add a DC-specific H200 smoke / pilot / extended sweep layer
- avoid changing the trainer's DC/V task logic just for sweep orchestration

This branch intentionally keeps the normal DC training files so the H200 sweep
commands run against the real DC training flow.
