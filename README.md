# SDTR_STGAT

`SDTR_STGAT` 是 `STDR` 專案中獨立維護的 predictor 訓練元件，專門負責改版 STGAT 的 `DC` 與 `V` 訓練。

如果之後要：

- 修正 predictor 架構
- 單獨訓練 `DC`
- 單獨訓練 `V`
- 在 H200 上跑 Slurm
- 在本地 RTX 3060 Laptop 上重跑

都應該以這個目錄為準。

## 目前的核心檔案

- `train_predictor.py`
  主訓練入口，支援 `joint`、`dc`、`v` 三種訓練模式。
- `stgat_model.py`
  改版 STGAT predictor。
- `predictor_normalization.py`
  `D/C/V` 的 normalization 與反正規化。
- `data_loader.py`
  讀取 NYC 真實路網、建立時間特徵、組成 dataset。
- `recompute_predictor_metrics.py`
  從既有 checkpoint 補算 `stgat_meta.json` 與 `predictor_test_metrics.json`。
- `submit_nano5_stgat_dc.slurm`
  H200 的 `DC` 訓練入口。
- `submit_nano5_stgat_v.slurm`
  H200 的 `V` 訓練入口。
- `run_local_dc_3060.ps1`
  本地 RTX 3060 的 `DC` 訓練入口。
- `run_local_v_3060.ps1`
  本地 RTX 3060 的 `V` 訓練入口。

## 模型與任務

### 任務定義

- `DC`
  預測節點層的 `demand` 與 `supply`
- `V`
  預測邊層的 `speed`

### 架構摘要

模型遵循 STGAT 的核心思想，由多個 `ST-block` 組成：

- 每個 `ST-block` 先做 temporal convolution（`GTCN`）
- 再做 graph attention

可調參數包括：

- `--num-st-blocks`
- `--num-gtcn-layers`
- `--hidden-dim`
- `--num-heads`
- `--kernel-size`

### 圖結構

#### DC 分支

- 有固定 node adjacency path
- 也有 adaptive node adjacency path

adaptive adjacency 現在是：

- 由 learnable embeddings 生成節點間 score
- 再做 `top-k` 稀疏化
- 預設 `--adaptive-topk 16`

如果設成：

```bash
--adaptive-topk 0
```

才會回到 dense learned graph。

#### V 分支

- 使用 fixed line-graph adjacency
- 目前主線沒有額外的 adaptive edge graph

## 時間特徵

目前主線會使用這 6 個時間特徵：

- `month_sin`
- `month_cos`
- `weekday_sin`
- `weekday_cos`
- `slot_sin`
- `slot_cos`

這組特徵是為了讓模型學到：

- 一天內相鄰時間窗的變化
- 星期幾之間的差異
- 月份之間的 recurring pattern

## 資料切分

主線切分規則是：

- 每月 `1-20`：train
- 每月 `21-24`：val
- 每月 `25+`：test

但現在真正使用的規則不是只看 target，而是：

- **整個 `history + target` window 必須完整落在同一個 split**

這樣可以避免：

- train sample 吃到上個月 `test` history
- val / test window 跨 split 邊界

## Normalization

- `demand`：`log1p + z-score`
- `supply`：`log1p + z-score`
- `speed`：`per-edge z-score`

統計量只來自：

- train windows 實際覆蓋到的時間點

## 訓練模式

### `DC-only`

```bash
python train_predictor.py --train-task dc --monitor-task dc
```

### `V-only`

```bash
python train_predictor.py --train-task v --monitor-task v
```

### `joint`

```bash
python train_predictor.py --train-task joint --monitor-task raw_dc
```

`raw_dc` 代表：

- `demand_rmse + supply_rmse`

## H200 用法

### 訓練 DC

```bash
NUM_ST_BLOCKS=2 sbatch submit_nano5_stgat_dc.slurm
```

常用可覆寫參數：

- `BATCH_SIZE`
- `EPOCHS`
- `PRECISION`
- `NUM_ST_BLOCKS`
- `ADAPTIVE_TOPK`
- `TRAIN_TASK`
- `MONITOR_TASK`
- `RUN_PREFIX`

### 訓練 V

```bash
NUM_ST_BLOCKS=2 sbatch submit_nano5_stgat_v.slurm
```

常用可覆寫參數：

- `BATCH_SIZE`
- `EPOCHS`
- `PRECISION`
- `NUM_ST_BLOCKS`
- `ADAPTIVE_TOPK`
- `RUN_PREFIX`

目前兩支 Slurm 都是：

- H200 單卡
- `#SBATCH --time=07:00:00`

## 本地 RTX 3060 用法

### 本地訓練 DC

```powershell
.\run_local_dc_3060.ps1 -NumStBlocks 2 -BatchSize 16 -Epochs 100
```

### 本地訓練 V

```powershell
.\run_local_v_3060.ps1 -NumStBlocks 2 -BatchSize 16 -Epochs 100
```

本地腳本預設：

- `--device cuda`
- `--precision fp32`
- `--num-workers 0`

這是為了讓 3060 Laptop 比較穩定，不先假設本地 CUDA / AMP 環境和 H200 完全一致。

## 輸出檔案

每次訓練的 run 目錄通常會包含：

- `stgat_best.pt`
- `stgat_final.pt`
- `stgat_meta.json`
- `predictor_log.json`
- `predictor_test_metrics.json`

其中：

- `stgat_meta.json`
  記錄模型設定、split、normalization、checkpoint 選模資訊
- `predictor_test_metrics.json`
  記錄 normalized loss 與 raw metrics

## Recompute 工具

`recompute_predictor_metrics.py` 用於：

- 舊 run 缺 JSON
- 需要從 checkpoint 補算 metadata / metrics

它會優先讀取既有 run 的 metadata，避免補算時偷偷改用不同的 split 或 normalization。

## 使用原則

如果 README 與程式細節有出入，請以這些檔案為準：

- `train_predictor.py`
- `stgat_model.py`
- `predictor_normalization.py`

這份 README 只描述目前保留的主線方法，不再記錄已經否決的舊做法。
