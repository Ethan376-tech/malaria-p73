# GitHub upload manifest — P73 experiment code

All scripts here are the **latest / authoritative versions** as run on the GPU (`/root/A_Dissertation`) — the exact code that produced the reported experiments and results. Report-building scripts (the `.docx` builder, figure/equation renderers, slide deck), superseded prior versions, and one-off helpers are **excluded** and listed at the bottom with the reason.

**Upload set: 123 scripts.** Integrity: every experiment script md5-matches its GPU source. (G19, the intra-field-aggregation / space-signal work, added 2026-08-11.)

**Repository scaffolding (added 2026-08-12, not experiment scripts):** `README.md`, `requirements.txt`, `LICENSE`, `.gitignore`, and `code/common/paths.py` — the shared path config that every experiment script imports (`from common.paths import ...`; env-configurable via `MALARIA_DATA_ROOT`). `common/paths.py` is the repo config version (env-var override added for portability), not md5-matched to the GPU absolute-path original.

## G01 — Data & bag construction

| script | date | role |
|---|---|---|
| `build_bagset_v2.py` | 2026-06-27 | final 441-bag set (241 Ibadan+200 Chitt-1), 4-encoder embeddings + robustness seeds |
| `build_bagset_v2_extraseeds.py` | 2026-06-27 | final 441-bag set (241 Ibadan+200 Chitt-1), 4-encoder embeddings + robustness seeds |

## G02 — Encoder (ThickDINO) training & probes

| script | date | role |
|---|---|---|
| `build_a2_tile_paired.py` | 2026-07-01 | paired significance test for tile-level cross-site ranking (reported stats) |
| `build_framework_probe_nb.py` | 2026-06-14 | framework ablation DINO+iBOT vs fair MoCo (5 seeds) |
| `build_moco_fair_nb.py` | 2026-06-14 | fair MoCo (queue+predictor), reported framework ablation |
| `build_seedconfirm_probe_nb.py` | 2026-06-14 | multi-seed tile-probe confirmation (canonical cross-site probe) |
| `build_stagea_nb.py` | 2026-06-13 | Stage A: SSL patch pool build |
| `build_stageb_supcon_nb.py` | 2026-06-13 | SupCon encoder (reported Table 3.4 baseline) |
| `build_stageb_v3_nb.py` | 2026-06-13 | Stage B FINAL: DINO+iBOT adaptation, init_values=1.0 (=Table 3.4 ThickDINO) |
| `build_stagecredux_final_nb.py` | 2026-06-13 | FINAL clean B->A encoder comparison probe |
| `build_supvit_dinoadapt_nb.py` | 2026-06-14 | ThickDINO = SupViT-init + DINO+iBOT + softmax centring |
| `build_thickdino_red_nb.py` | 2026-06-14 | ThickDINO-Red arm (RedDino-init), reported Table 3.4 |

## G03 — MIL bag classification / aggregation

| script | date | role |
|---|---|---|
| `build_bandnums_v2.py` | 2026-06-27 | by-band spec/recall + FP/FN counts (reported §3/§4 numbers) |
| `build_context_v2.py` | 2026-06-27 | context-aware aggregation probe (max-pool/ABMIL/TransMIL) |
| `build_e2_stats.py` | 2026-06-17 | E2 Tanzania probe + in-domain bag stats hardening |
| `build_indomain_v2_multiseed.py` | 2026-06-27 | seed-averaged in-domain repeated 5x3 CV (Table 3.2 numbers) |
| `build_pooled_milca_cv.py` | 2026-06-21 | pooled-MILCA CV row of Table 3.2 |
| `build_xsite_v2.py` | 2026-06-27 | cross-site B->A + top-k aggregations (Table 3.2/3.3) |

## G04 — End-to-end MIL fine-tune

| script | date | role |
|---|---|---|
| `build_b2_frozen_robust_v2.py` | 2026-06-27 | protocol-matched frozen baseline (repeated 5x3) |
| `build_b2_tilecache_v2.py` | 2026-06-27 | raw-tile cache for end-to-end MIL FT (v2 441-bag set) |
| `build_b2_train_v2run.py` | 2026-06-27 | end-to-end MIL fine-tune run (SupViT/RedDino/ThickDINO-FT) |

## G05 — Pseudo-box generation (ViT-native)

| script | date | role |
|---|---|---|
| `build_ft_chitt.py` | 2026-06-28 | in-domain MIL FT of ThickDINO on Chittagong-1 (domain-B generator) |
| `build_iii_byband.py` | 2026-06-28 | B-domain generation + by-parasitaemia-band recall/precision |
| `build_s1_chitt1.py` | 2026-06-22 | S1 ViT localiser comparison (A and B), reported Table 3.5 |
| `build_s1_localization.py` | 2026-06-21 | S1 ViT localiser comparison (A and B), reported Table 3.5 |
| `build_s2_1_calibrate.py` | 2026-06-28 | ViT-native ICAM parameter calibration (Alg-1 constants) |
| `build_s2_2_v3.py` | 2026-06-28 | FINAL class-discriminative single-pass pseudo-box generator (in-domain FT ThickDINO) |
| `build_s8_chitt2.py` | 2026-06-28 | domain-B pseudo-label generation (Chittagong-1, the other train/test design) |

## G06 — Faintness / multi-signal filter

| script | date | role |
|---|---|---|
| `build_a3_filter_sensitivity.py` | 2026-08-07 | filter two-threshold sensitivity region (test perbox) |
| `build_s3_final.py` | 2026-06-28 | FINAL single-signal faintness-protected filter, fully dev-calibrated |
| `build_sweep_signals.py` | 2026-07-27 | per-box multi-signal features (MC mean/var, TTA var, typicality) for all A0 boxes |
| `dev_faint_4way.py` | 2026-07-27 | 4-way A0/naive/MC-protected/multi-signal on one protocol |
| `dev_faint_final.py` | 2026-07-27 | consolidated FINAL multi-signal filter numbers (one pipeline) |
| `dev_faint_joint_calib.py` | 2026-07-27 | joint constrained threshold calibration (multi-signal, §3.4.6) |
| `dev_faint_multi_analyze2.py` | 2026-07-27 | multi-signal analysis: physical proxy, field bootstrap CI, 3-split |
| `dev_faint_multi_extract2.py` | 2026-07-27 | multi-signal per-box extraction w/ coords/field_id (final) |
| `gt_per_field.py` | 2026-07-27 | per-field GT parasite counts aligned to extract2 field_id |

## G07 — Detector training (RetinaNet / MicroYOLO)

| script | date | role |
|---|---|---|
| `build_a0_3seed.py` | 2026-07-27 | matched 3-seed A0 baseline (fair vs HNM) |
| `build_s4_detector.py` | 2026-06-28 | patch-level detector extrinsic validation of pseudo-labels (single/multi) |
| `build_s4_detector_multi.py` | 2026-07-28 | patch-level detector extrinsic validation of pseudo-labels (single/multi) |
| `build_s5_yolo_data.py` | 2026-06-28 | YOLO dataset prep from pseudo-labels (single/multi) |
| `build_s5_yolo_data_multi.py` | 2026-07-27 | YOLO dataset prep from pseudo-labels (single/multi) |
| `build_s5_yolo_train.py` | 2026-06-28 | YOLOv8 training on pseudo-labels vs GT baseline |
| `build_s6_protected_multi.py` | 2026-07-27 | RetinaNet on MULTI-signal-filtered pseudo-boxes |
| `build_s6_retinanet.py` | 2026-06-28 | RetinaNet on pseudo-labels (A0 vs protected), reported detector |
| `build_y10_boxgain_3seed.py` | 2026-07-30 | box-loss-gain tuning 3-seed (P2 box7.5/9/11) -> box9 choice |
| `build_y11_microyolo_table47.py` | 2026-07-30 | MicroYOLO (P2+box9) Table 4.7 row (A0/protected/GT) |
| `build_y9_microyolo_specialize.py` | 2026-07-29 | MicroYOLO = YOLOv8n + P2 stride-4 small-object head (final spec) |

## G08 — Refinement (teacher-student / bootstrap / dynamic)

| script | date | role |
|---|---|---|
| `build_s7_adaptive.py` | 2026-07-28 | adaptive multi-signal retain at bootstrap stage |
| `build_s7_bootstrap.py` | 2026-07-11 | MILCA bootstrap refinement + uncertainty (2nd retain place) |
| `build_s7_bootstrap_multi.py` | 2026-07-27 | bootstrap refinement with multi-signal retain |
| `build_s7_typ95_3seed.py` | 2026-07-28 | 3-seed typ95 (retain + moderate typicality gate) |
| `build_s9_complete.py` | 2026-08-06 | completes RetinaNet TS retain grid to match MicroYOLO (Table 4.9) |
| `build_s9_teacher_student.py` | 2026-07-11 | RetinaNet teacher-student refinement (FilterDistil, §3.5.2) |
| `build_s9_teacher_student_multi.py` | 2026-07-27 | teacher-student with multi-signal retain |
| `build_y12_microyolo_bootstrap.py` | 2026-07-30 | MicroYOLO bootstrap 3-seed switch-gate |
| `build_y13_microyolo_ts.py` | 2026-07-30 | MicroYOLO teacher-student refinement (Table 4.9 analog) |
| `build_y14_microyolo_grid.py` | 2026-07-30 | MicroYOLO reverse-schedule dynamic-filter grid (3-seed, Phase 2B) |
| `build_y5_dynamic_3seed.py` | 2026-07-29 | YOLOv8n reverse-schedule dynamic filter (3-seed confirm) |
| `build_y6_v2_faintaware.py` | 2026-07-29 | faint-aware dynamic gate (avoids late-round faint eviction) |

## G09 — Joint MIL-detector

| script | date | role |
|---|---|---|
| `build_s12_joint.py` | 2026-06-29 | generates neg-crop bag data reused by verify_joint (kept as dependency) |
| `build_y16_joint_native.py` | 2026-07-31 | AUTHORITATIVE MicroYOLO joint (subclass DetectionModel, LAM=8, Table 4.11) |
| `verify_joint.py` | 2026-07-20 | AUTHORITATIVE RetinaNet joint 3-arm/3-seed (Table 4.11 RetinaNet rows) |

## G10 — Source-free adaptation / calibration

| script | date | role |
|---|---|---|
| `build_t7_calib.py` | 2026-08-02 | post-hoc calibration recovery under cross-site shift (temp/Platt) |
| `build_ws1a_labelshift.py` | 2026-07-03 | label/prior-shift diagnosis (BBSE), §3.6/§4.5 |
| `build_ws1b_selftrain.py` | 2026-07-03 | source-free self-training decision adaptation |
| `build_ws1c_featalign.py` | 2026-07-03 | source-free feature alignment (full-covariance CORAL) |

## G11 — Conformal selective prediction

| script | date | role |
|---|---|---|
| `build_ws2_selective.py` | 2026-07-03 | conformal selective prediction / abstention |

## G12 — Resolution-aware detection

| script | date | role |
|---|---|---|
| `build_ws3_resdet.py` | 2026-07-03 | resolution-aware detection base (domain-B ~19px) |
| `build_ws3_v3.py` | 2026-07-03 | FINAL: is 21px detection fundamentally hard (C1->C2 scale) |

## G13 — Label efficiency

| script | date | role |
|---|---|---|
| `build_s10_labeleff_microyolo.py` | 2026-07-30 | AUTHORITATIVE MicroYOLO label-efficiency (Table 4.14) |
| `verify_labeleff.py` | 2026-07-20 | verify supervised>hybrid is not a schedule artifact (§4.7) |

## G14 — Counting / LoD / parasitaemia

| script | date | role |
|---|---|---|
| `build_s11_counting_microyolo.py` | 2026-07-30 | counting propagated to MicroYOLO (Table 4.15 companion) |
| `build_s11_counting_refined.py` | 2026-07-30 | FINAL counting with refined MicroYOLO (Table 4.15) |
| `dev_lod_filter_v2.py` | 2026-08-07 | does the faintness filter lower clinical LoD (efficient rewrite) |
| `dev_lod_full.py` | 2026-07-14 | per-sample count over K=40 fields (LoD robustness) |
| `dev_lod_mehanian.py` | 2026-08-07 | FINAL LoD (Mehanian 2017) + Manescu parasitaemia |
| `dev_parasitemia.py` | 2026-07-14 | parasitaemia = 8000*MP/WBC on FASTMAL (Manescu 2020) |
| `dev_wbc_train.py` | 2026-07-14 | WBC detector on Lacuna WBC boxes (block B) |

## G15 — Diagnosis: detection-vs-bag decision

| script | date | role |
|---|---|---|
| `build_dc1_detcount.py` | 2026-07-31 | detection-count decision signals per Ibadan sample |
| `build_dc2_analysis.py` | 2026-07-31 | detection-vs-bag analysis + leak check (support dc4/dc5) |
| `build_dc3_leakcheck.py` | 2026-08-01 | detection-vs-bag analysis + leak check (support dc4/dc5) |
| `build_dc4_clean.py` | 2026-08-01 | per-field leakage-clean detection signals (final) |
| `build_dc5_bootstrap.py` | 2026-08-01 | detection-vs-bag AUC + paired bootstrap 95% CI (final) |
| `dev_block_d.py` | 2026-07-14 | sample-level MIL vs Detection vs MIL+Det (Table 4.12) |
| `dev_lowpara_ci.py` | 2026-07-12 | AUTHORITATIVE low-para cross-site AUC 95% CI (bootstraps roc_auc) |

## G16 — Leakage-clean GT-supervised (LOFO)

| script | date | role |
|---|---|---|
| `build_dc10_yolov8n_lofo.py` | 2026-08-01 | LOFO GT-sup stock YOLOv8n (Table 4.7) |
| `build_dc11_torchvision_lofo.py` | 2026-08-01 | LOFO GT-sup torchvision detectors RetinaNet/FCOS (Table 4.7) |
| `build_dc12_patch_lofo.py` | 2026-08-01 | LOFO GT-sup patch head (Table 4.7) |
| `build_dc13_map50_agg.py` | 2026-08-01 | aggregate LOFO YOLO mAP50 (Table 4.7) |
| `build_dc14_retinanet_lofo.py` | 2026-08-01 | LOFO GT-sup RetinaNet-R50 (Table 4.7) |
| `build_dc6_devtest_leak.py` | 2026-08-01 | FASTMAL dev/test film-overlap leakage quantification |
| `build_dc7_lofo_gtsup.py` | 2026-08-01 | LOFO-clean GT-sup MicroYOLO reference (custom AP) |
| `build_dc8_lofo_map50.py` | 2026-08-01 | LOFO-clean GT-sup mAP50 (Table 4.7) |
| `build_dc9_lofo_counting.py` | 2026-08-01 | LOFO-clean GT-sup counting (Table 4.15) |

## G17 — External validation (Lacuna/Tanzania)

| script | date | role |
|---|---|---|
| `build_b2c_robust.py` | 2026-06-17 | robust B->C Tanzania transfer (5 seeds, top-k) |
| `dev_lacuna_detect.py` | 2026-07-14 | Lacuna cross-site parasite detection |
| `dev_lacuna_probe.py` | 2026-07-14 | Lacuna (Ghana) tile-probe external validation |

## G18 — Ablations & detector-choice screens

| script | date | role |
|---|---|---|
| `build_hnm_data.py` | 2026-07-27 | MILCA hard-negative mining data + RetinaNet train (A0+HNM) |
| `build_hnm_train.py` | 2026-07-27 | MILCA hard-negative mining data + RetinaNet train (A0+HNM) |
| `build_ng1_gate1c.py` | 2026-07-31 | finer-CAM generator gates (centroid-refined; does it train a better MicroYOLO) |
| `build_ng2_gate2.py` | 2026-07-31 | finer-CAM generator gates (centroid-refined; does it train a better MicroYOLO) |
| `build_notyp_3seed.py` | 2026-07-27 | filter WITHOUT typicality gate (ablation, 3-seed) |
| `build_notyp_data.py` | 2026-07-27 | filter WITHOUT typicality gate (ablation, 3-seed) |
| `build_s5b_centroid.py` | 2026-07-31 | new-generator data prep (centroid / bigbox variants) |
| `build_s5c_bigbox.py` | 2026-07-31 | new-generator data prep (centroid / bigbox variants) |
| `build_sweep_datasets.py` | 2026-07-27 | detector-oriented filter sweep (data + RetinaNet screen) |
| `build_sweep_train.py` | 2026-07-27 | detector-oriented filter sweep (data + RetinaNet screen) |
| `build_t1_hnm.py` | 2026-08-02 | does standalone HNM help the strong MicroYOLO |
| `build_t2_data.py` | 2026-08-02 | soft uncertainty-WEIGHTED pseudo-box loss (MicroYOLO) |
| `build_t2_fullaug.py` | 2026-08-02 | soft uncertainty-WEIGHTED pseudo-box loss (MicroYOLO) |
| `build_t2_train.py` | 2026-08-02 | soft uncertainty-WEIGHTED pseudo-box loss (MicroYOLO) |
| `build_t4_rtdetr.py` | 2026-08-02 | RT-DETR transformer-detector ablation |
| `build_yolo_3seed.py` | 2026-07-28 | 3-seed top YOLO contenders on A0 pseudo-labels |
| `build_yolo_versions.py` | 2026-07-28 | YOLO-version screen (does newer YOLO beat v8n) |
| `dev_base3_retinanet.py` | 2026-08-08 | RetinaNet base D1 at 3 seeds (Table 4.9 base row) |

## G19 — Intra-field aggregation & space-signal filtering (added 2026-08-11)

| script | date | role |
|---|---|---|
| `dev_ripley.py` | 2026-08-10 | Ripley's L test of parasite spatial clustering over all positive fields (§3.4.4, Fig 3.5b; global p=0.001) |
| `dev_overdisp_gr.py` | 2026-08-10 | pair-correlation g(r) + quadrat variance-to-mean over-dispersion characterisation (§3.4.4 scale/VMR) |
| `dev_spatial_signal.py` | 2026-08-11 | intrinsic TP-vs-FP AUC of the local-density (space) signal (§3.4.7; 0.84 vs 0.81 per-box) |
| `dev_lod_space.py` | 2026-08-11 | space-signal filter at counting time → per-µL LoD τ-sweep (§4.7, Fig 4.10; LoD 30k→10k, −67%) |
| `dev_s9_space_sweep.py` | 2026-08-11 | RetinaNet teacher–student space-signal retain, τ operating-point sweep (Table 4.9a) |
| `dev_y13_sweep.py` | 2026-08-11 | MicroYOLO teacher–student space-signal retain, τ sweep (Table 4.9b) |
| `dev_focal_ts.py` | 2026-08-11 | focal-loss isolation: FPF retain advantage at γ=0 vs γ=2 (§4.3; proves FPF ≡ focal loss) |

---
## Excluded (NOT uploaded)

### Report / figure / equation / slide builders — produce the write-up, not results (33)

- `build_alg1_pipeline.py` — report schematic/qualitative figure
- `build_classviz.py` — classification-performance figures
- `build_dc15_counting_fig.py` — figure generator (reads results, draws report PNG)
- `build_eda_nb.py` — EDA / figure notebook (report-support)
- `build_fig1_abc.py` — figure generator (reads results, draws report PNG)
- `build_fig2_2.py` — figure generator (reads results, draws report PNG)
- `build_fig34_v2.py` — figure generator (reads results, draws report PNG)
- `build_fig_4way.py` — figure generator (reads results, draws report PNG)
- `build_fig_confusion_tri.py` — classification-performance figures
- `build_fig_domains.py` — figure generator (reads results, draws report PNG)
- `build_fig_e2.py` — figure generator (reads results, draws report PNG)
- `build_fig_filter_compare.py` — figure generator (reads results, draws report PNG)
- `build_fig_filter_concept.py` — figure generator (reads results, draws report PNG)
- `build_fig_filter_field_compare.py` — figure generator (reads results, draws report PNG)
- `build_fig_filter_plane.py` — figure generator (reads results, draws report PNG)
- `build_fig_joint.py` — figure generator (reads results, draws report PNG)
- `build_fig_labeleff_micro.py` — figure generator (reads results, draws report PNG)
- `build_fig_microyolo.py` — figure generator (reads results, draws report PNG)
- `build_fig_multifilter.py` — figure generator (reads results, draws report PNG)
- `build_fig_parasitaemia.py` — figure generator (reads results, draws report PNG)
- `build_fig_pipeline.py` — figure generator (reads results, draws report PNG)
- `build_fig_s1_AB.py` — figure generator (reads results, draws report PNG)
- `build_figs_notitle.py` — figure generator (reads results, draws report PNG)
- `build_figures_nb.py` — EDA / figure notebook (report-support)
- `build_report_docx.py` — builds the report/deck itself
- `build_report_pdf.py` — builds the report/deck itself
- `build_s3_figs.py` — report schematic/qualitative figure
- `build_slides.py` — builds the report/deck itself
- `build_ws1_fig.py` — WS1 diagnostic figure for §4.5
- `dev_filterfig.py` — figure generator (reads results, draws report PNG)
- `plot_labeleff.py` — figure generator (reads results, draws report PNG)
- `render_eqs.py` — renders report equation PNGs
- `render_eqs2.py` — renders report equation PNGs

### Superseded prior versions — a newer script produces the reported result (73)

- `build_a1_reddino_sanity.py` — one-off RedDino checkpoint-artifact sanity (diagnostic, not reported result)
- `build_b2_frozen_robust_v2_128.py` — 128-tile variant; v2 (64-tile) is the reported frozen baseline
- `build_b2_tilecache.py` — superseded by build_b2_tilecache_v2.py
- `build_b2_tilecache_v2_128.py` — superseded by build_b2_tilecache_v2.py
- `build_b2_train.py` — superseded by build_b2_train_v2run.py
- `build_b2_train_v2.py` — superseded by build_b2_train_v2run.py
- `build_b2_train_v2_128.py` — superseded by build_b2_train_v2run.py
- `build_dbag1_nb.py` — early bag-MIL notebooks -> superseded by bagset_v2/indomain_v2_multiseed
- `build_dbag2_nb.py` — early bag-MIL notebooks -> superseded by bagset_v2/indomain_v2_multiseed
- `build_dbag3_nb.py` — early bag-MIL notebooks -> superseded by bagset_v2/indomain_v2_multiseed
- `build_dbag3b_nb.py` — early bag-MIL notebooks -> superseded by bagset_v2/indomain_v2_multiseed
- `build_indomain_v2.py` — superseded by build_indomain_v2_multiseed.py
- `build_manifest_nb.py` — data-pipeline notebook gens (splits.csv already shipped as artifact)
- `build_moco_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_moco_probe_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_ng1_gate1.py` — generator-gate dev -> superseded by build_ng1_gate1c.py
- `build_ng1_gate1b.py` — generator-gate dev -> superseded by build_ng1_gate1c.py
- `build_path2_fromscratch_nb.py` — from-scratch DINOv2 tile export/training (excluded encoder branch)
- `build_path2_tiles_big_nb.py` — from-scratch DINOv2 tile export/training (excluded encoder branch)
- `build_path2_tiles_nb.py` — from-scratch DINOv2 tile export/training (excluded encoder branch)
- `build_s10_ct.py` — RetinaNet/old label-eff -> build_s10_labeleff_microyolo.py
- `build_s10_labeleff.py` — RetinaNet/old label-eff -> build_s10_labeleff_microyolo.py
- `build_s10b_hybrid_matched.py` — RetinaNet/old label-eff -> build_s10_labeleff_microyolo.py
- `build_s11_counting.py` — superseded by build_s11_counting_refined.py
- `build_s11_ct.py` — superseded by build_s11_counting_refined.py
- `build_s2_2_generate.py` — superseded by build_s2_2_v3.py
- `build_s2_2_v2.py` — superseded by build_s2_2_v3.py
- `build_s2_field_icam.py` — early ICAM prototype (iterative erasure dropped) -> build_s2_2_v3.py
- `build_s2_icam.py` — early ICAM prototype (iterative erasure dropped) -> build_s2_2_v3.py
- `build_s3_2_protected.py` — superseded by build_s3_final.py (dev-calibrated)
- `build_s3_filter.py` — superseded by build_s3_final.py (dev-calibrated)
- `build_s6_phase3.py` — duplicate of build_s6_retinanet.py (same RetinaNet stage)
- `build_splits_nb.py` — data-pipeline notebook gens (splits.csv already shipped as artifact)
- `build_stageb_nb.py` — StageB v1/v2 -> superseded by build_stageb_v3_nb.py
- `build_stageb_v2_nb.py` — StageB v1/v2 -> superseded by build_stageb_v3_nb.py
- `build_stagec_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_stagecredux_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_stagecredux_supcon_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_stagecredux_v2_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_stagecredux_v3_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_staged1_nb.py` — early bag-MIL notebooks -> superseded by bagset_v2/indomain_v2_multiseed
- `build_staged2_nb.py` — early bag-MIL notebooks -> superseded by bagset_v2/indomain_v2_multiseed
- `build_t99_3seed.py` — duplicate matched-3seed A0 baseline -> build_a0_3seed.py
- `build_thickdino_probe_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_thickdino_red_probe_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_thickdino_v2_nb.py` — from-scratch DINOv2 exploration (not in reported encoder set)
- `build_thickdino_v2_probe_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_thickdino_v3_nb.py` — from-scratch DINOv2 exploration (not in reported encoder set)
- `build_thickdino_v3_probe_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_transmil_probe.py` — superseded by build_context_v2.py
- `build_ws3_v2.py` — superseded by build_ws3_v3.py
- `build_xdomain_nb.py` — encoder probe/variant iterations -> superseded by stagecredux_final / framework_probe
- `build_y13_ct.py` — superseded by build_y13_microyolo_ts.py
- `build_y13_ctb.py` — superseded by build_y13_microyolo_ts.py
- `build_y15_joint_apitest.py` — joint-on-MicroYOLO dev/test -> superseded by build_y16_joint_native.py
- `build_y15_joint_microyolo.py` — joint-on-MicroYOLO dev/test -> superseded by build_y16_joint_native.py
- `build_y15b_patchtest.py` — joint-on-MicroYOLO dev/test -> superseded by build_y16_joint_native.py
- `build_y15v2_joint_ft.py` — joint-on-MicroYOLO dev/test -> superseded by build_y16_joint_native.py
- `build_y16_ct.py` — joint-on-MicroYOLO dev/test -> superseded by build_y16_joint_native.py
- `build_y16_native_test.py` — joint-on-MicroYOLO dev/test -> superseded by build_y16_joint_native.py
- `build_y1_reeval.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `build_y1_yolo_bootstrap.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `build_y2_yolo_bootstrap_3seed.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `build_y3_yolo_teacher_student.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `build_y4_dynamic_screen.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `build_y4b_reverse.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `build_y7_customize.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `build_y8_customize_3seed.py` — YOLOv8n dev ladder -> superseded by y5/y6/y9/y11/y12/y13/y14
- `dev_faint_agg.py` — prototype v1 -> *_extract2/_analyze2/_final
- `dev_faint_multi_analyze.py` — prototype v1 -> *_extract2/_analyze2/_final
- `dev_faint_multi_extract.py` — prototype v1 -> *_extract2/_analyze2/_final
- `dev_lod.py` — superseded by dev_lod_mehanian.py / dev_lod_full.py / dev_lod_filter_v2.py
- `dev_lod_filter.py` — superseded by dev_lod_mehanian.py / dev_lod_full.py / dev_lod_filter_v2.py

### Readers / inspectors / progress helpers — not experiments (12)

- `build_manifest.py` — reader/inspector/progress/helper — not an experiment
- `build_progress2.py` — reader/inspector/progress/helper — not an experiment
- `build_progress4.py` — reader/inspector/progress/helper — not an experiment
- `evalsanity.py` — reader/inspector/progress/helper — not an experiment
- `extract_headers_remote.py` — reader/inspector/progress/helper — not an experiment
- `inspect_fastmal.py` — reader/inspector/progress/helper — not an experiment
- `read_dataset_pdfs.py` — reader/inspector/progress/helper — not an experiment
- `read_docx.py` — reader/inspector/progress/helper — not an experiment
- `read_manescu.py` — reader/inspector/progress/helper — not an experiment
- `read_milca_formal.py` — reader/inspector/progress/helper — not an experiment
- `read_reddino.py` — reader/inspector/progress/helper — not an experiment
- `search_milca_split.py` — reader/inspector/progress/helper — not an experiment
