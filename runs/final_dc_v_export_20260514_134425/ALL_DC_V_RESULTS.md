# STGAT Final DC and V Experiment Results

Generated: 2026-05-14T13:44:25

This export contains the selected final/best STGAT results for all current DC city datasets and V speed datasets. Metrics are raw-scale test-set values.

Metric convention: DC_RMSE = D_RMSE + C_RMSE. DC_MSE_SUM = D_MSE + C_MSE. STGAT final JSONs consistently provide MSE/RMSE/MAE; MAPE is intentionally not used as the unified table metric.

## DC Overall Results

| Rank | Dataset | D_MSE | D_RMSE | D_MAE | C_MSE | C_RMSE | C_MAE | DC_MSE_SUM | DC_RMSE | Selected Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Shanghai | 29.130323 | 5.397251 | 1.761081 | 70.280154 | 8.383326 | 3.210450 | 99.410477 | 13.780577 | dc_shanghai |
| 2 | NYC | 67.646410 | 8.224744 | 2.705992 | 538.264104 | 23.200519 | 6.011228 | 605.910515 | 31.425263 | dc_nyc |
| 3 | Shenzhen | 153.987644 | 12.409176 | 1.912070 | 537.416281 | 23.182241 | 4.290065 | 691.403925 | 35.591416 | dc_shenzhen |
| 4 | Chengdu | 2357.741255 | 48.556578 | 13.345404 | 4537.511521 | 67.361053 | 21.266208 | 6895.252776 | 115.917631 | dc_chengdu |

## DC Horizon Results

| Dataset | Horizon | D_MSE | D_RMSE | D_MAE | C_MSE | C_RMSE | C_MAE | DC_MSE_SUM | DC_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shanghai | 15min | 24.144851 | 4.913741 | 1.631895 | 45.424181 | 6.739746 | 2.677054 | 69.569032 | 11.653487 |
| Shanghai | 30min | 27.787070 | 5.271344 | 1.735828 | 61.620487 | 7.849872 | 3.081416 | 89.407557 | 13.121216 |
| Shanghai | 60min | 33.860873 | 5.819010 | 1.872916 | 96.350421 | 9.815825 | 3.692339 | 130.211293 | 15.634835 |
| NYC | 15min | 52.348861 | 7.235251 | 2.451981 | 398.085214 | 19.952073 | 4.271019 | 450.434076 | 27.187324 |
| NYC | 30min | 63.013583 | 7.938110 | 2.632336 | 406.807538 | 20.169470 | 5.321542 | 469.821121 | 28.107580 |
| NYC | 60min | 82.711103 | 9.094564 | 2.950787 | 799.690152 | 28.278793 | 7.947479 | 882.401255 | 37.373358 |
| Shenzhen | 15min | 90.111402 | 9.492703 | 1.615641 | 253.611651 | 15.925189 | 3.298070 | 343.723053 | 25.417892 |
| Shenzhen | 30min | 136.333144 | 11.676178 | 1.844988 | 439.964307 | 20.975326 | 3.980840 | 576.297450 | 32.651505 |
| Shenzhen | 60min | 215.688107 | 14.686324 | 2.185905 | 824.133069 | 28.707718 | 5.258973 | 1039.821176 | 43.394042 |
| Chengdu | 15min | 1452.084908 | 38.106232 | 8.915346 | 2309.752275 | 48.059882 | 13.384645 | 3761.837183 | 86.166114 |
| Chengdu | 30min | 2082.614191 | 45.635668 | 12.141938 | 3925.825629 | 62.656409 | 19.321327 | 6008.439821 | 108.292077 |
| Chengdu | 60min | 3211.137219 | 56.666897 | 17.341381 | 6453.557524 | 80.334037 | 27.880441 | 9664.694743 | 137.000934 |

## V Overall Results

| Rank | Dataset | V_MSE | V_RMSE | V_MAE | Selected Run |
| --- | --- | --- | --- | --- | --- |
| 1 | NYC V | 3.622625 | 1.903320 | 1.079752 | v_nyc |
| 2 | PEMS-BAY | 10.960525 | 3.310668 | 1.589988 | v_pems_bay |
| 3 | METR-LA | 34.254402 | 5.852726 | 3.276046 | v_metr_la |

## V Horizon Results

| Dataset | Horizon | V_MSE | V_RMSE | V_MAE |
| --- | --- | --- | --- | --- |
| NYC V | 15min | 3.544046 | 1.882564 | 1.071679 |
| NYC V | 30min | 3.605916 | 1.898925 | 1.071766 |
| NYC V | 60min | 3.568368 | 1.889012 | 1.079592 |
| PEMS-BAY | 15min | 6.908007 | 2.628309 | 1.330906 |
| PEMS-BAY | 30min | 11.621642 | 3.409053 | 1.646597 |
| PEMS-BAY | 60min | 15.996541 | 3.999568 | 1.946799 |
| METR-LA | 15min | 25.459736 | 5.045764 | 2.878571 |
| METR-LA | 30min | 35.014818 | 5.917332 | 3.310712 |
| METR-LA | 60min | 46.339960 | 6.807346 | 3.829510 |

## Exported Checkpoints

| ID | Task | Dataset | Export Folder | Copied Best Checkpoints |
| --- | --- | --- | --- | --- |
| dc_nyc | DC | NYC | runs\final_dc_v_export_20260514_134425\checkpoints\dc_nyc | stgat_best.pt, stgat_best_raw_dc.pt, training_state_best.pt |
| dc_shenzhen | DC | Shenzhen | runs\final_dc_v_export_20260514_134425\checkpoints\dc_shenzhen | stgat_best.pt, stgat_best_raw_dc.pt |
| dc_chengdu | DC | Chengdu | runs\final_dc_v_export_20260514_134425\checkpoints\dc_chengdu | stgat_best.pt, stgat_best_raw_dc.pt |
| dc_shanghai | DC | Shanghai | runs\final_dc_v_export_20260514_134425\checkpoints\dc_shanghai | stgat_best.pt, stgat_best_raw_dc.pt |
| v_nyc | V | NYC V | runs\final_dc_v_export_20260514_134425\checkpoints\v_nyc | stgat_best.pt, stgat_best_v.pt, training_state_best.pt |
| v_metr_la | V | METR-LA | runs\final_dc_v_export_20260514_134425\checkpoints\v_metr_la | stgat_best.pt |
| v_pems_bay | V | PEMS-BAY | runs\final_dc_v_export_20260514_134425\checkpoints\v_pems_bay | stgat_best.pt |

## Source Runs

| ID | Label | Metrics JSON | Original Run Dir | Note |
| --- | --- | --- | --- | --- |
| dc_nyc | NYC DC | D:\Citi\STDR\STDR_STGAT\runs\local_4090_dc\autopilot_sessions\DC_autopilot_25h_from_speed_20260430_153706\s04_final_002_tk16_h16_lr1em3_wd1em5_hd48_b2_tfbase_s42_ep45\predictor_test_metrics.json | D:\Citi\STDR\STDR_STGAT\runs\local_4090_dc\autopilot_sessions\DC_autopilot_25h_from_speed_20260430_153706\s04_final_002_tk16_h16_lr1em3_wd1em5_hd48_b2_tfbase_s42_ep45 | final continue to 200; selected from DC_final_002_continue_to200 summary |
| dc_shenzhen | Shenzhen DC | runs\shenzhen_stgat_continue_20260508_162201\006_lr0008_bs8_h64_topk16_c15_continue_100_from_final\predictor_test_metrics.json | runs\shenzhen_stgat_continue_20260508_162201\006_lr0008_bs8_h64_topk16_c15_continue_100_from_final | val-selected group6; continued to 200 epochs |
| dc_chengdu | Chengdu DC | runs\chengdu_dc_full_20260513_093044\stgat\predictor_test_metrics.json | runs\chengdu_dc_full_20260513_093044\stgat | formal full 100-epoch STGAT; continue80 was not better on test DC |
| dc_shanghai | Shanghai DC | runs\shanghai_stgat_continue80_20260514_065707\stgat\predictor_test_metrics.json | runs\shanghai_stgat_continue80_20260514_065707\stgat | Shanghai final selected graph; alternate graph validation runs excluded by user decision |
| v_nyc | NYC V speed | runs\local_4090_v\nyc_pemsbest_transfer_topk20_seed19_hist14_ep60_20260426_185250\predictor_test_metrics.json | runs\local_4090_v\nyc_pemsbest_transfer_topk20_seed19_hist14_ep60_20260426_185250 | NYC local V best selected run |
| v_metr_la | METR-LA speed | runs\external_speed\METR-LA_project_monthly_bs32_ep100_resumeable_20260420_142901\predictor_test_metrics.json | runs\external_speed\METR-LA_project_monthly_bs32_ep100_resumeable_20260420_142901 | best scanned METR-LA speed RMSE run |
| v_pems_bay | PEMS-BAY speed | runs\external_speed\autopilot_sessions\PEMS-BAY_distance_paired_autopilot_20260424_213850\batch_10_final_best_60epoch\final_best_repeat1_legacy_map_route_topk20_seed19_lr0p001_wd0p00001_hist14_ep60\predictor_test_metrics.json | runs\external_speed\autopilot_sessions\PEMS-BAY_distance_paired_autopilot_20260424_213850\batch_10_final_best_60epoch\final_best_repeat1_legacy_map_route_topk20_seed19_lr0p001_wd0p00001_hist14_ep60 | best scanned PEMS-BAY speed RMSE run |
