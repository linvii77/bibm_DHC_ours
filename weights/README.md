# DHC 预训练权重（Git LFS）

每个 `best_model.pth` 约 289MB，内含双网络 A+B state_dict。

## Synapse 20% labeled — 3-fold

| 文件 | Seed | 说明 |
|------|------|------|
| `fold1/ckpts/best_model.pth` | 0 | 双网络 A+B |
| `fold2/ckpts/best_model.pth` | 1 | 双网络 A+B |
| `fold3/ckpts/best_model.pth` | 666 | 双网络 A+B |

测试集 Mean Dice ≈ **47.03%**（3-fold 评估，见 `results/evaluation_res.txt`）。

## AMOS 5% labeled — fold1（本地复现）

| 文件 | Seed | 说明 |
|------|------|------|
| `amos_5p/fold1/ckpts/best_model.pth` | 0 | 双网络 A+B，epoch 270 val best |

测试集 Mean Dice = **40.11%**（120 cases，见 `results/amos_5p_fold1_evaluation_res.txt`）。

## 使用

克隆后若权重未自动下载：

```bash
git lfs install
git lfs pull
```

链接到 `logs/` 布局（推理脚本所需路径）：

```bash
bash scripts/setup_for_inference.sh          # Synapse 3-fold
bash scripts/setup_amos_weights.sh           # AMOS fold1
```
