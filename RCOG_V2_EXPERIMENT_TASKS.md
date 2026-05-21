# RCOG-v2 训练与实验任务清单

目标：在不使用 TopK attention 的前提下，优化 RCOG gate，使 DDQN 尽量抓到更多真实有收益场景，同时减少误抓和坏覆盖。

当前基线：

| 版本 | run dir | activation | benefit precision | benefit recall | bad >1% | bad >5% | mean ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Conservative RCOG | `runs/no_topk_rcog_base_oppddqn_train2000_eval1000` | 7.5% | 76.0% | 56.4% | 0.7% | 0.1% | 0.9971 |
| High-recall RCOG | `runs/no_topk_rcog_high_recall_benefit_eval1000` | 34.3% | 26.2% | 89.1% | 16.4% | 2.3% | 1.0016 |
| Balanced RCOG | `runs/no_topk_rcog_balanced_recall_eval1000` | 14.9% | 51.7% | 76.2% | 4.1% | 0.3% | 0.9975 |

最终主目标：

| 指标 | 目标 |
|---|---:|
| benefit recall | >= 70% |
| benefit precision | >= 55% |
| bad >5% | <= 0.5% |
| bad >1% | <= 3.0% |
| activation | 12%-20% |
| mean ratio vs Dijkstra | <= 0.9973，越低越好 |

## Task 0: 固定公平比较协议

目的：避免不同实验因为 candidate pool 不同而比较不公平。

要求：

- 固定 `seed=121` 作为主表。
- 所有主实验使用同一组 train/val/test split。
- 主测试先使用 `eval_pool_size=1000`，选出候选方案后再扩大到 `3000` 或全资料。
- test 只用于最终报告，不用 test 调阈值。
- gate 输入只能使用决策时可见的预测值、历史速度、候选路线、DDQN Q 值；未来真实速度只能作为训练 label 和最终评价。

产出：

- `runs/rcog_v2_protocol_summary.json`
- 主表字段统一为：`activation_rate`, `opportunity_precision`, `opportunity_recall`, `benefit_precision`, `benefit_recall`, `bad_gt_1pct`, `bad_gt_5pct`, `ddqn_over_pred`, `ddqn_time`, `dijkstra_real_time`

## Task 1: 阈值细扫实验

对应优化意见 1：细调 gate 阈值。

目的：在现有 RCOG 模型不重训的情况下，只优化选择规则，寻找比 Balanced RCOG 更好的 precision/recall/bad trade-off。

需要修改：

- 在 `no_topk_rcog_gate_experiment.py` 增加 dense threshold search。
- 保存完整 sweep，而不是只保存 top 200。
- 新增 objective：`constrained_balanced`。

建议搜索范围：

```text
score_threshold: 0.06 - 0.22
p_o_min:         0.20 - 0.55
p_b_min:         0.20 - 0.60
p_d_max:         0.10 - 0.35
q_margin_min:    [-inf, 0, q50, q70]
```

约束：

```text
benefit_recall >= 70
benefit_precision >= 55
bad_gt_5pct <= 0.5
activation_rate between 12 and 20
```

排序目标：

```text
maximize benefit_recall
then maximize benefit_precision
then minimize bad_gt_1pct
then minimize ddqn_over_pred
```

推荐 run dir：

```text
runs/rcog_v2_dense_threshold_balanced_eval1000
```

判断标准：

- 如果只靠阈值细扫就能达到 `benefit recall >= 70%`, `precision >= 55%`, `bad >5% <= 0.5%`，说明 RCOG-v1 已经足够。
- 如果 recall/precision 无法同时满足，进入 Task 2。

当前结果：

| run dir | strict target | selected activation | benefit precision | benefit recall | bad >1% | bad >5% | mean ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `runs/rcog_v2_dense_threshold_balanced_eval1000` | 未满足 | 7.0% | 74.3% | 51.5% | 0.7% | 0.0% | 0.9972 | 细扫只能找到更安全的保守点，不能在高 recall 下同时压住误抓 |

验证发现：

- 满足 `benefit_recall >= 70`, `benefit_precision >= 55`, `bad_gt_5pct <= 0.5`, `bad_gt_1pct <= 3`, `activation 12%-20%` 的配置数量为 0。
- 即使不约束 `bad_gt_1pct`，满足 `benefit_recall >= 70`, `benefit_precision >= 55`, `bad_gt_5pct <= 0.5`, `activation 12%-20%` 的配置数量仍为 0。
- 高 recall 区域可达到 `benefit_recall ~= 81.5%`，但 `benefit_precision ~= 42%`, `bad_gt_1pct ~= 8.2%`, `bad_gt_5pct ~= 1.3%`。
- 因此下一步应进入 Task 2：Utility Gate，用收益/损失幅度替代纯分类 gate。

## Task 2: Utility Gate 实验

对应优化意见 2：从分类 gate 改为期望收益/损失 gate。

目的：解决目前分类 label 只知道“赢/输”，不知道“赢多少/输多少”的问题。

需要新增或修改脚本：

```text
rcog_utility_gate_experiment.py
```

模型：

- `gain_model`: 预测 `max(0, dijkstra_real_time - ddqn_real_time)`
- `loss_model`: 预测 `max(0, ddqn_real_time - dijkstra_real_time)`
- 可选 `bad_model`: 预测 `ddqn_real_time > dijkstra_real_time * 1.05`

激活规则：

```text
expected_utility = predicted_gain - lambda_loss * predicted_loss
activate if expected_utility > threshold
```

建议搜索：

```text
lambda_loss: [1.0, 1.5, 2.0, 3.0, 5.0]
utility_threshold: validation quantiles
bad_prob_max: [0.15, 0.20, 0.25, 0.30]
```

推荐 run dir：

```text
runs/rcog_v2_utility_gate_train2000_eval1000
```

主要比较：

- Utility Gate vs Balanced RCOG
- 是否能在 `benefit recall >= 70%` 时把 `bad >1%` 从 4.1% 压到 3% 以下。

判断标准：

- 若 mean ratio 低于 Balanced RCOG，并且 bad 指标更低，则 Utility Gate 成为主方案。
- 若 recall 提升但 mean ratio 变差，则只能作为 high-recall ablation。

## Task 3: 扩大训练池实验

对应优化意见 3：把 gate 训练集从 2000 扩到 5000 或 10000。

目的：确认 RCOG 概率估计是否因为训练样本太少而不稳定。

实验组：

| 实验 | gate 类型 | train_pool_size | eval_pool_size | run dir |
|---|---|---:|---:|---|
| 3A | RCOG 分类 gate | 5000 | 1000 | `runs/rcog_v2_classifier_train5000_eval1000` |
| 3B | RCOG 分类 gate | 10000 | 1000 | `runs/rcog_v2_classifier_train10000_eval1000` |
| 3C | Utility gate | 5000 | 1000 | `runs/rcog_v2_utility_train5000_eval1000` |
| 3D | Utility gate | 10000 | 1000 | `runs/rcog_v2_utility_train10000_eval1000` |

优先级：

1. 先跑 3A 和 3C。
2. 如果 5000 明显优于 2000，再跑 10000。
3. 如果 5000 已经没有提升，10000 可作为附录或跳过。

预估时间：

| train_pool_size | 单次实验预估 |
|---:|---:|
| 2000 | 35-50 分钟 |
| 5000 | 60-80 分钟 |
| 10000 | 90-130 分钟 |

判断标准：

- 5000 比 2000 的 mean ratio 至少改善 `0.0002`，才值得继续 10000。
- 如果只是 precision/recall 波动，但 mean ratio 不变，不作为主提升。

## Task 4: 消融实验表

目的：把三个优化意见拆开论证，证明每一步是否有效。

推荐主表：

| 方法 | dense threshold | utility gate | train size | activation | benefit precision | benefit recall | bad >1% | bad >5% | mean ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Conservative RCOG | no | no | 2000 | | | | | | |
| Balanced RCOG | coarse | no | 2000 | | | | | | |
| Dense-threshold RCOG | yes | no | 2000 | | | | | | |
| Utility RCOG | yes | yes | 2000 | | | | | | |
| Utility RCOG large | yes | yes | 5000 | | | | | | |
| Utility RCOG larger | yes | yes | 10000 | | | | | | |

## Task 5: 稳健性验证

目的：避免单一 seed 下偶然好看。

只对最终候选模型跑：

```text
seed = 121, 222, 333
eval_pool_size = 1000
```

若三组 seed 均满足：

```text
mean ratio < 1.0
bad_gt_5pct <= 0.5
benefit_recall >= 65
```

则可以作为论文主结果。

## Task 6: 扩大测试规模

目的：把最终方法从 1000 样本扩到更接近完整资料集。

推荐顺序：

1. `eval_pool_size=3000`
2. `eval_pool_size=5000`
3. 全 split 或尽可能大规模

只跑最终 1-2 个候选方案，不再把所有 ablation 全部扩大。

## 推荐执行顺序

1. Task 1：dense threshold search，最快验证是否还有阈值空间。
2. Task 2：Utility Gate 2000，验证新设计是否有效。
3. Task 3A/3C：5000 训练池，比较 classifier vs utility。
4. Task 4：整理消融表。
5. Task 5：多 seed 稳健性。
6. Task 6：扩大测试规模。

## 当前最可能成为主方案的路线

```text
Dijkstra default
-> RCOG Utility Gate 判断是否进入动态不确定机会场景
-> DDQN reranker 在候选路径中做 correction
-> 若 gate 不激活则保持 Dijkstra
```

这条路线的论文表述最清楚：强化学习不是取代 Dijkstra，而是在预测不稳定、候选路径存在潜在收益时做 future-value correction。

## 已完成实验总表

汇总文件：

```text
runs/rcog_v2_completed_experiment_summary.json
```

| 方法 | activation | benefit precision | benefit recall | bad >1% | bad >5% | mean ratio |
|---|---:|---:|---:|---:|---:|---:|
| Conservative RCOG 2000 | 7.5% | 76.0% | 56.4% | 0.7% | 0.1% | 0.9971 |
| High-recall RCOG 2000 | 34.3% | 26.2% | 89.1% | 16.4% | 2.3% | 1.0016 |
| Balanced RCOG 2000 | 14.9% | 51.7% | 76.2% | 4.1% | 0.3% | 0.9975 |
| Dense-threshold RCOG 2000 | 7.0% | 74.3% | 51.5% | 0.7% | 0.0% | 0.9972 |
| Utility RCOG 2000 | 11.1% | 45.0% | 49.5% | 2.9% | 0.0% | 0.9980 |
| Classifier RCOG 5000 | 10.3% | 65.0% | 66.3% | 1.7% | 0.1% | 0.9972 |
| Utility RCOG 5000 | 14.1% | 47.5% | 66.3% | 3.2% | 0.2% | 0.9976 |
| Classifier RCOG 10000 | 12.7% | 58.3% | 73.3% | 2.8% | 0.2% | 0.9974 |
| Utility RCOG 10000 | 15.7% | 48.4% | 75.2% | 3.7% | 0.2% | 0.9974 |
| Final eval3000 Classifier 10000 | 13.1% | 61.6% | 78.1% | 2.8% | 0.4% | 0.9974 |
| Stability seed222 Classifier 10000 | 11.5% | 61.7% | 71.7% | 2.0% | 0.3% | 0.9965 |
| Stability seed333 Classifier 10000 | 12.3% | 58.5% | 75.8% | 2.6% | 0.3% | 0.9982 |

当前主结论：

- `Classifier RCOG 10000` 是最适合作为论文主方案的 gate。
- `Utility Gate` 没有超过 classifier gate，可作为消融：证明直接预测收益/损失并不优于三头分类式 RCOG。
- 扩大训练池有效：从 2000 到 10000 后，recall 在安全约束下明显提高。
- 大样本 `eval_pool=3000` 下，最终 classifier gate 保持 `benefit recall=78.1%`, `benefit precision=61.6%`, `bad_gt_5pct=0.4%`, `mean ratio=0.9974`。
- 多 seed 结果均保持 `mean ratio < 1.0`, `bad_gt_5pct <= 0.5%`, `benefit recall >= 70%`，稳定性通过。

## Precision-focused 后续实验

用户目标：在 recall 不降低或提升的前提下，提高 precision。

汇总文件：

```text
runs/rcog_v2_precision_focus_summary.json
```

| 方法 | activation | benefit precision | benefit recall | bad >1% | bad >5% | mean ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Final classifier 10000 eval3000 | 13.1% | 61.6% | 78.1% | 2.8% | 0.4% | 0.9974 | 当前主方案 |
| Precision-at-recall eval1000 | 10.9% | 67.0% | 72.3% | 1.5% | 0.1% | 0.9970 | 小样本 precision 提升，但 recall 略低 |
| Two-stage precision recall eval1000 | 18.2% | 45.6% | 82.2% | 5.4% | 0.5% | 0.9981 | recall 提升但 precision 失败 |
| Weighted light eval1000 | 11.3% | 65.5% | 73.3% | 1.6% | 0.1% | 0.9970 | 小样本 precision 提升且 recall 持平 |
| Weighted light eval3000 | 13.4% | 61.3% | 79.4% | 2.8% | 0.4% | 0.9974 | 大样本 recall 提升，但 precision 未提升 |
| Weighted strong eval1000 | 12.4% | 61.3% | 75.2% | 2.4% | 0.1% | 0.9971 | 小样本 precision/recall 均提升 |
| Weighted strong eval3000 | 13.4% | 60.3% | 78.4% | 2.9% | 0.4% | 0.9975 | 大样本 precision 未提升 |

结论：

- 在 `eval1000` 小样本上，reweight 可以提高 precision，同时保持或提高 recall。
- 在 `eval3000` 大样本上，precision 提升没有稳定复现；当前主方案仍是最稳妥的最终结果。
- 当前特征和 DDQN checkpoint 下，precision/recall 已接近 Pareto 边界：要继续稳定提高 precision 且不降 recall，需要新增能区分 false-positive override 的特征，或重训 DDQN/candidate scorer，而不是继续调 gate 阈值。

## Precision + Recall 同时提升尝试

用户进一步目标：recall 和 precision 都继续提升。

新增脚本：

```text
rcog_meta_filter_experiment.py
```

汇总文件：

```text
runs/rcog_v2_precision_recall_push_summary.json
```

| 方法 | activation | benefit precision | benefit recall | bad >1% | bad >5% | mean ratio | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Main classifier eval3000 | 13.1% | 61.6% | 78.1% | 2.8% | 0.4% | 0.9974 | 主方案 |
| Weighted light eval3000 | 13.4% | 61.3% | 79.4% | 2.8% | 0.4% | 0.9974 | recall 提升，precision 未提升 |
| Weighted strong eval3000 | 13.4% | 60.3% | 78.4% | 2.9% | 0.4% | 0.9975 | recall 微升，precision 下降 |
| Meta-filter eval1000 | 13.8% | 55.8% | 76.2% | 3.2% | 0.4% | 0.9975 | 二层 false-positive filter 失败 |

结论：

- 阈值重选、二阶段 OR gate、hard-negative reweight、meta-filter 都未能在大样本上同时稳定提升 precision 和 recall。
- 目前 `Classifier RCOG 10000 eval3000` 仍是最可靠主结果。
- 若要继续提高两者，需要进入模型/特征层改造：
  1. 新增 route-level regret magnitude predictor。
  2. 新增 historical OD/time override success-rate 特征。
  3. 新增 local perturbation sensitivity 特征。
  4. 重训 DDQN 或 supervised reranker，降低 DDQN raw action 的 false-positive 倾向。

## Enhanced Feature / Reranker Ablation

目标：按顺序验证四个增强方向是否比当前最佳 RCOG gate 有用。

新增脚本：

```text
rcog_enhanced_ablation_experiment.py
```

汇总文件：

```text
runs/rcog_enhanced_ablation_comparison_summary.json
```

`eval1000_same_pool` 使用同一批 train/val/test candidates，对四个方向做公平比较：

| variant | activation | benefit precision | benefit recall | opportunity precision | opportunity recall | bad >5% | mean ratio | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base_reproduced | 12.2% | 60.7% | 67.3% | 67.2% | 59.4% | 0.50% | 0.9986 | 同池 baseline |
| regret_magnitude | 5.6% | 80.4% | 40.9% | 82.1% | 33.3% | 0.00% | 0.9984 | precision 有效，但 recall 掉太多 |
| history_success | 12.9% | 58.1% | 68.2% | 67.4% | 63.0% | 0.60% | 0.9988 | 无效，precision 下降且 bad 增加 |
| perturb_sensitivity | 12.2% | 58.2% | 64.5% | 67.2% | 59.4% | 0.40% | 0.9985 | 无效，recall/precision 都不优 |
| supervised_reranker | 6.8% | 72.1% | 100.0% | 75.0% | 37.0% | 0.10% | 0.9982 | 有用，显著提高 precision 和安全性，但 activation/机会覆盖较低 |

对最有希望的 `supervised_reranker` 进一步做 `eval3000` 大样本验证：

| scope | variant | activation | benefit precision | benefit recall | opportunity precision | opportunity recall | bad >5% | mean ratio | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| eval3000 same pool | base_reproduced | 12.5% | 57.0% | 70.8% | 64.7% | 60.8% | 0.47% | 0.9985 | 同池 baseline |
| eval3000 same pool | supervised_reranker | 5.6% | 75.6% | 95.5% | 77.4% | 32.7% | 0.03% | 0.9983 | 有用：precision 和安全性大幅提升，mean ratio 略优，但 opportunity recall 低 |

注意：

- `supervised_reranker` 的 benefit recall 是相对于 supervised reranker 自己能产生 benefit 的场景；它和 DDQN raw-action benefit recall 的分母不同，不能直接等同于原 DDQN gate 的 recall。
- 从最终系统角度看，`supervised_reranker` 更像“低激活、高精度、安全替代器”，不是“高覆盖 DDQN correction gate”。

当前四个方向结论：

1. `route-level regret magnitude predictor`：对 precision 有明显帮助，但 recall 牺牲太大，适合作为高精度安全模式。
2. `historical OD/time success-rate`：本轮无效。
3. `local perturbation sensitivity`：本轮无效。
4. `supervised reranker`：最有用，显著提高 precision、降低 bad override，但会降低 activation 和 opportunity 覆盖。
