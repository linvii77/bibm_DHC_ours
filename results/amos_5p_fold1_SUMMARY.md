# AMOS 5% DHC — Fold1 复现结果

**任务**: AMOS22，5% 有标签半监督（labeled_5p=10, unlabeled_5p=206, test=120, eval=24）  
**配置**: seed=0, lr=3e-2, w=0.1, 300 epochs, CPS A+B  
**权重**: `weights/amos_5p/fold1/ckpts/best_model.pth`（epoch 270, val best）  
**完成时间**: 2026-06-22

## 1. 训练阶段 — 验证集 (eval, 24 例)

| 指标 | 值 |
|------|-----|
| 最佳 Mean Dice | **37.44%** (epoch 270) |
| 最终 epoch 300 Dice | 37.13% |

## 2. 测试集 (test, 120 例) — 官方评估

| 指标 | 值 |
|------|-----|
| **Mean Dice** | **40.11%** |
| Mean ASD (mm) | 37.99 |

### 各类 Dice (%)

| # | 器官 | Dice |
|---|------|------|
| 1 | spleen | 69.7 |
| 2 | right kidney | 54.2 |
| 3 | left kidney | 58.9 |
| 4 | gallbladder | 0.1 |
| 5 | esophagus | 20.3 |
| 6 | liver | 75.1 |
| 7 | stomach | 41.4 |
| 8 | aorta | 74.2 |
| 9 | IVC | 57.1 |
| 10 | pancreas | 47.7 |
| 11 | right adrenal | 0.0 |
| 12 | left adrenal | 0.0 |
| 13 | duodenum | 28.6 |
| 14 | bladder | 46.9 |
| 15 | prostate/uterus | 27.4 |

## 3. 文件清单

- `evaluation_res.txt` — 完整逐病例 Dice/ASD + 汇总
- `train_val_summary.txt` — 训练验证集 best dice 记录
- `best_model.pth` — `weights/amos_5p/fold1/ckpts/`（Git LFS）

## 4. 说明

- 仅 **fold1 (seed=0)**，非论文 3-fold 平均。
- 权重路径：`weights/amos_5p/fold1/ckpts/best_model.pth`
- 论文 AMOS 5% 参考图见官方 README `images/amos-5.png`。
