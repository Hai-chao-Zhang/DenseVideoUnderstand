# DIVE-Bench Leaderboard

- generated_at: `2026-08-20T06:20:21`
- include_closed_source: `true`
- source_artifacts: `21`
- open_mos_judge: `Qwen/Qwen3-VL-32B-Instruct`

## DIVE-Bench Educational Dense Video

| rank | model | method_id | samples | open_mos ↑ | token_f1 ↑ | cer ↓ | wer ↓ | exact_match ↑ | patch_projection_recompute_ratio ↓ | reference_patch_compute_ratio ↓ | sampling_density_fps | throughput_fps ↑ | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Gemini 3.1 Pro Preview (gemini-3.1-pro-preview) | gemini-3.1-pro-preview | 634 | 1.6041 | 0.159633 | 1.26881 | 1.24166 | 0 |  |  |  |  | gemini |
| 2 | GRT (Qwen2.5-VL 7B, route floor 0.80 / 0.55, cap 48) | grt_qwen2_5_vl_7b_dual_floor_s080_o055_cap48 | 634 | 1.59148 | 0.0489087 | 1.06401 | 1.05377 | 0 | 0.847719 | 0.847719 | 0.00747917 | 2.48043 | open |
| 3 | GRT (Qwen2.5-VL 3B, threshold 0.3) | grt_qwen2_5_vl_3b_t03 | 634 | 1.58991 | 0.0995837 | 1.13915 | 1.08351 | 0 | 0.885659 | 0.885659 | 0.00747917 | 1.67264 | open |
| 4 | Qwen3-VL 8B Instruct | qwen3_vl_8b | 634 | 1.29811 | 0.0360398 | 1.05889 | 1.05198 | 0 |  |  |  |  | open |
| 5 | Qwen2.5-VL 7B Instruct | qwen2_5_vl_7b | 634 | 1.28391 | 0.0347846 | 1.04624 | 1.03975 | 0 |  |  |  |  | open |
| 6 | Gemini 3.6 Flash (gemini-3.6-flash) | gemini-3.6-flash | 634 | 1.24921 | 0.0814611 | 1.10168 | 1.06796 | 0 |  |  |  |  | gemini |
| 7 | Qwen3-VL 2B Instruct | qwen3_vl_2b | 634 | 1.14511 | 0.0331448 | 1.0455 | 1.0365 | 0 |  |  |  |  | open |
| 8 | Qwen3-VL 4B Instruct | qwen3_vl_4b | 634 | 1.1388 | 0.0348606 | 1.05593 | 1.05119 | 0 |  |  |  |  | open |
| 9 | Qwen2.5-VL 3B Instruct | qwen2_5_vl_3b | 634 | 1.13565 | 0.0343388 | 1.04393 | 1.03399 | 0 |  |  |  |  | open |
| 10 | Qwen2.5-VL 72B Instruct | qwen2_5_vl_72b | 634 | 0.944795 | 0.0268628 | 1.05934 | 1.04789 | 0 |  |  |  |  | open |
| 11 | Qwen2-VL 2B Instruct | qwen2_vl_2b | 634 | 0.790221 | 0.0200933 | 1.03692 | 1.0334 | 0 |  |  |  |  | open |
| 12 | Qwen2.5-VL 32B Instruct | qwen2_5_vl_32b | 634 | 0.5 | 0.0250276 | 1.05448 | 1.04856 | 0 |  |  |  |  | open |
| 13 | Qwen3-VL 32B Instruct | qwen3_vl_32b | 634 | 0.175079 | 0.0184676 | 1.05239 | 1.05085 | 0 |  |  |  |  | open |
| 14 | GRT (LLaVA-OneVision Qwen2 0.5B, verified Route31) | grt_llava_onevision_0_5b_hf_route31_t0001 | 634 | 0.119874 | 0.0141964 | 1.11093 | 1.10273 | 0 | 0.867154 | 0.867154 | 0.00747917 | 4.86514 | open |
| 15 | LLaVA-OneVision Qwen2 0.5B | llava_onevision_0_5b | 634 | 0.116719 | 0.0140879 | 1.03765 | 1.03756 | 0 |  |  |  |  | open |
| 16 | LLaVA-OneVision 1.5 8B Instruct | llava_onevision_1_5_8b | 634 |  | 0.0349748 | 1.07243 | 1.05686 | 0 |  |  |  |  | open |
| 17 | Qwen2-VL 7B Instruct | qwen2_vl_7b | 634 |  | 0.0305096 | 1.05309 | 1.04788 | 0 |  |  |  |  | open |
| 18 | Phi-4 Multimodal Instruct | phi4_multimodal | 634 |  | 0.0275287 | 1.03008 | 1.02762 | 0 |  |  |  |  | open |
| 19 | InternVL3 1B | internvl3_1b | 634 |  | 0.0271622 | 1.062 | 1.0606 | 0 |  |  |  |  | open |
| 20 | InternVL3 8B | internvl3_8b | 634 |  | 0.0250303 | 1.07046 | 1.06601 | 0 |  |  |  |  | open |
| 21 | VideoLLaMA3 7B | videollama3_7b | 634 |  | 0.0249369 | 1.05106 | 1.04975 | 0 |  |  |  |  | open |
| 22 | LongVA 7B | longva_7b | 634 |  | 0.0246616 | 1.04678 | 1.04241 | 0 |  |  |  |  | open |
| 23 | InternVL3 2B | internvl3_2b | 634 |  | 0.0223091 | 1.03408 | 1.03574 | 0 |  |  |  |  | open |
| 24 | InternVL2.5 8B | internvl2_5_8b | 634 |  | 0.0188533 | 1.05738 | 1.05719 | 0 |  |  |  |  | open |
| 25 | InternVL2.5 4B | internvl2_5_4b | 634 |  | 0.01792 | 1.02051 | 1.02362 | 0 |  |  |  |  | open |
| 26 | InternVL2.5 1B | internvl2_5_1b | 634 |  | 0.0178427 | 1.03147 | 1.0336 | 0 |  |  |  |  | open |
| 27 | LLaVA-OneVision Qwen2 7B | llava_onevision_original | 634 |  | 0.0175202 | 1.04056 | 1.03982 | 0 |  |  |  |  | open |
| 28 | VideoLLaMA3 2B | videollama3_2b | 634 |  | 0.0158476 | 1.0245 | 1.02683 | 0 |  |  |  |  | open |
| 29 | InternVL2.5 2B | internvl2_5_2b | 634 |  | 0.0119298 | 1.01729 | 1.02037 | 0 |  |  |  |  | open |

## DIVE-Bench High-Motion Dense Video

| rank | model | method_id | samples | grid_acc ↑ | grid_ade ↓ | grid_fde ↓ | transition_acc ↑ | token_f1 ↑ | sampling_density_fps | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | GRT (LLaVA-OneVision Qwen2 0.5B) | grt_llava_ov_0_5b | 1000 | 0.10125 | 0.712142 | 0.686488 | 0.0148571 | 0.181281 | 0.988599 | open |
| 2 | Gemini 3.6 Flash (gemini-3.6-flash) | gemini-3.6-flash | 1000 | 0.0885007 | 0.502699 | 0.571509 | 0.863788 | 0 |  | gemini |
| 3 | Gemini 3.1 Pro Preview (gemini-3.1-pro-preview) | gemini-3.1-pro-preview | 1000 | 0.066019 | 0.779814 | 0.849685 | 0.603178 | 0 |  | gemini |
