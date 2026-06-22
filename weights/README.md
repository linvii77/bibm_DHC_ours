# DHC Synapse 20% labeled — 3-fold best checkpoints

预训练权重，无需重新训练即可直接推理/评估。

| 文件 | Seed | 说明 |
|------|------|------|
| `fold1/ckpts/best_model.pth` | 0 | 双网络 A+B |
| `fold2/ckpts/best_model.pth` | 1 | 双网络 A+B |
| `fold3/ckpts/best_model.pth` | 666 | 双网络 A+B |

每个文件约 289MB，通过 **Git LFS** 托管。克隆后若权重未自动下载：

```bash
git lfs install
git lfs pull
```

链接到官方测试目录：

```bash
bash scripts/setup_for_inference.sh
```
