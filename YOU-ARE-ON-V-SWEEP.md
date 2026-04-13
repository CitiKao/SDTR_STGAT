This branch is `codex/v-sweep`.

Purpose:

- keep the full runnable `codex/v-training` training bundle
- add a V-specific H200 smoke / pilot / extended sweep layer
- avoid changing the trainer's DC/V task logic just for sweep orchestration

This branch intentionally keeps the normal V training files so the H200 sweep
commands run against the real V training flow.
