# Official Code Sources

This directory is reserved for upstream official repositories used by the DC benchmark official-code lane.
The upstream folders should remain unmodified unless a change is explicitly limited to a data loader or output head.

| Method | Repository | License | Local Commit |
| --- | --- | --- | --- |
| DCRNN | https://github.com/liyaguang/DCRNN | MIT | 602afd9d767d3aa1c9b3eac51710d6aeee12c227 |
| STGCN | https://github.com/VeritasYin/STGCN_IJCAI-18 | BSD-2-Clause | 747007ab01d56762c1a6bad89b255700cedf204d |
| Graph WaveNet | https://github.com/nnzhan/Graph-WaveNet | MIT | 6b162e80c59a1d494809252eca055cff93dc66b1 |

The project-owned adapter code lives in `dc_benchmark/official_*.py`. Its manifest records the upstream URL,
local commit, target channel, data shape, and exact command templates used for a benchmark run.
