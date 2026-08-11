# Results (metric JSONs)

Aggregate / derived **metric outputs** behind the reported tables and figures, organised
by experiment directory (the same layout the scripts write to under `OUTPUTS`). These are
summary statistics and anonymous per-field arrays only — **no images, no patient
identifiers, no clinical per-sample records**. One per-field diagnostic file that listed
raw field filenames (`cluster_probe/stats.json`) was deliberately excluded.

## Key files by report area

| Report area | Files |
|---|---|
| **Representation / encoder** (§4.1) | `e2_tanzania/paired_thickdino_vs_reddino.json` |
| **Faintness-protected filter** (§3.4, §4.3) | `s3_filter/final.json`, `four_way.json`, `byband.json`, `sensitivity.json`, `protected.json`, `multi_analysis2.json`, `joint_calib.json`, `space_intrinsic.json` |
| **Detectors** (§4.3) | `s5_yolo/yolo_3seed.json`, `yolo_results.json`; `s6_retinanet/base3_retinanet_results.json`, `a0_3seed_results.json`, `protected_multi_results.json`, `protected_space_results.json`, `sweep_results.json` |
| **Focal-loss isolation** (§4.3, Table 4.10) | `s6_retinanet/focal_iso_results.json` (note: the reported Table 4.10 uses the 3-seed rerun; the 2-seed json is here for reference) |
| **FilterDistil / teacher–student** (§4.3, Tables 4.9a,b) | `s9_teacher_student/`, `s7_bootstrap*/` |
| **Joint MIL–detector** (§4.4) | `s12_joint/joint_results.json` |
| **Counting → parasitaemia** (§4.6) | `s11_counting*/counting_results.json`, `s11_counting_refined/scatter.json` |
| **Limit of detection** (§4.7) | `lod_mehanian/results.json`, `results_filter.json`, `space_results.json` |
| **Label efficiency** (§4.7) | `s10_labeleff*/labeleff_results.json` |
| **Intra-field aggregation & space signal** (§3.4.5, §3.4.8) | `cluster_probe/ripley_results.json`, `aggreg_stats.json`, `spatial_signal.json`, `filter_cluster.json` |
| **Cross-site decision / adaptation** (§4.2) | `moco_adapt/results.json`, `reddino_dinoadapt/results.json`, `dc_detcount/decision_results.json` |
| **Domain B (Chittagong-2)** (§4.4) | `s8_chitt2/` |

Other directories hold intermediate / ablation metric dumps from the same runs. See the
top-level `MANIFEST.md` for which script produces each experiment directory.
