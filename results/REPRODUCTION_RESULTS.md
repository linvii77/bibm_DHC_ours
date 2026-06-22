# DHC 复现结果汇总

本文件汇总 [bibm_DHC_baseline](https://github.com/linvii77/bibm_DHC_baseline) 仓库中两个数据集的测试集指标。

| 数据集 | 标签比例 | Fold | Seed | 权重路径 |
|--------|----------|------|------|----------|
| Synapse | 20% | 3-fold (1/2/3) | 0 / 1 / 666 | `weights/fold{1,2,3}/ckpts/best_model.pth` |
| AMOS | 5% | fold1 | 0 | `weights/amos_5p/fold1/ckpts/best_model.pth` |

**指标说明**

- **Dice / ASD**：官方 `code/evaluate_Ntimes_npy.py`（`results/evaluation_res.txt`、`results/amos_5p_fold1_evaluation_res.txt`）
- **HD95 / NSD**：`scripts/evaluate_hd95_nsd.py` 从 `logs/.../predictions_AB/*.nii.gz` 计算；NSD 容差 1.0 mm
- 预测/GT 均为空：HD95=0，NSD=100%；仅一方为空：HD95=128，NSD=0

---

## Synapse 20% labeled（3-fold 平均，6 test cases）

| Organ | Dice% | HD95 (mm) | NSD% | ASD (mm) |
|-------|------:|----------:|-----:|---------:|
| spleen | 77.9 | 14.0 | 75.5 | 3.7 |
| R.kidney | 51.6 | 7.3 | 59.4 | 2.4 |
| L.kidney | 48.7 | 20.3 | 54.0 | 7.9 |
| gallbladder | 55.5 | 23.9 | 65.1 | 8.8 |
| esophagus | 52.6 | 5.0 | 70.7 | 1.7 |
| liver | 86.8 | 10.7 | 73.0 | 3.2 |
| stomach | 35.9 | 24.8 | 35.7 | 7.6 |
| aorta | 63.2 | 15.7 | 68.2 | 4.3 |
| IVC | 55.0 | 9.5 | 60.1 | 2.5 |
| PSV | 35.6 | 26.2 | 48.0 | 1.8 |
| pancreas | 22.6 | 17.8 | 30.5 | 4.7 |
| R.adrenal | 21.1 | 12.6 | 54.3 | 8.6 |
| L.adrenal | 4.8 | 56.7 | 19.2 | 52.6 |
| **Mean ± std** | **47.03 ± 1.31** | **18.82 ± 2.32** | **54.90 ± 1.83** | **8.46 ± 2.45** |

各 fold 官方 Mean Dice：fold1 45.50% · fold2 46.89% · fold3 48.69% → 3-fold 平均 **47.03%**

---

## AMOS 5% labeled（fold1，120 test cases）

| Organ | Dice% | HD95 (mm) | NSD% | ASD (mm) |
|-------|------:|----------:|-----:|---------:|
| spleen | 69.7 | 35.3 | 59.3 | 13.4 |
| R.kidney | 54.2 | 69.2 | 43.5 | 29.4 |
| L.kidney | 58.9 | 59.2 | 50.2 | 19.5 |
| gallbladder | 6.7 | 119.5 | 6.7 | 119.5 |
| esophagus | 20.3 | 43.4 | 25.1 | 27.9 |
| liver | 75.1 | 52.7 | 46.4 | 15.3 |
| stomach | 41.4 | 56.5 | 27.5 | 20.6 |
| aorta | 74.2 | 17.9 | 72.4 | 4.0 |
| IVC | 57.1 | 22.4 | 52.2 | 11.4 |
| pancreas | 47.7 | 23.6 | 40.4 | 8.8 |
| R.adrenal | 0.0 | 128.0 | 0.0 | 128.0 |
| L.adrenal | 0.0 | 128.0 | 0.0 | 128.0 |
| duodenum | 28.6 | 29.1 | 29.1 | 11.7 |
| bladder | 62.6 | 15.3 | 55.2 | 7.7 |
| prostate/uterus | 44.8 | 30.2 | 39.7 | 24.5 |
| **Mean** | **42.75** | **55.34** | **36.52** | **37.99** |

训练验证集（eval, 24 cases）best Mean Dice：**37.44%** @ epoch 270  
官方 `evaluate_Ntimes_npy.py` Mean Dice：**40.11%**（与上表 Dice 列算法略有差异，见 `amos_5p_fold1_evaluation_res.txt`）

---

## 复现所需仓库文件

| 类别 | 路径 |
|------|------|
| 权重 (Git LFS) | `weights/fold{1,2,3}/`, `weights/amos_5p/fold1/` |
| 数据划分 | `splits/`, `amos_data/splits/`（AMOS 需自备 `amos_data/npy/`） |
| 评估结果 | `results/evaluation_res.txt`, `results/amos_5p_fold1_*`, `results/metrics_hd95_nsd_summary.txt` |
| 推理脚本 | `code/test.py`, `code/evaluate_Ntimes_npy.py`, `scripts/evaluate_hd95_nsd.py` |
| 权重链接 | `scripts/setup_for_inference.sh`, `scripts/setup_amos_weights.sh` |
| 说明 | `REPRODUCE.md` |

## 重新计算 HD95 / NSD

需先完成 `test.py` 推理生成 `logs/.../predictions_AB/`，再运行：

```bash
source env.sh
python scripts/evaluate_hd95_nsd.py
# 输出: results/metrics_hd95_nsd_summary.txt
```
