# 复现指南 — DHC + FusedProxyLoss

本仓库包含两组实验的完整复现材料：
1. **Synapse 20%** — DHC baseline（3-fold）& DHC+FusedProxy（3-fold）
2. **AMOS 5%** — DHC+FusedProxy（3-fold，fold2/3 权重待补）

完整指标见 `results/ALL_EXPERIMENTS_DETAILED.txt`。

---

## 0. 环境

```bash
pip install torch torchvision medpy SimpleITK surface-distance h5py tqdm tensorboard scipy
source env.sh   # export PYTHONPATH=$(pwd)/code:$PYTHONPATH
```

推荐：Python 3.8+，CUDA GPU（训练需 ≥16GB 显存，推理 ≥8GB）。

---

## 1. 数据准备

### 1.1 Synapse

官方预处理数据（推荐直接下载）：
https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/EmOL8Cn-GTBJtOjg6zNgsPABdF6TgoWtRac4FwGqfFxLvQ?e=a1xaDJ

解压到项目根目录，目录结构：
```
./synapse_data/
├── npy/
│   ├── 0001_image.npy
│   ├── 0001_label.npy
│   └── ...
└── splits/   (可直接使用本仓库 splits/ 目录替代)
```

data splits 已包含在本仓库 `splits/` 目录：
```
splits/
├── labeled_20p.txt
├── unlabeled_20p.txt
├── train.txt
├── eval.txt
└── test.txt
```

若 `synapse_data/splits/` 不存在，将 `splits/` 复制过去：
```bash
cp -r splits/* synapse_data/splits/
```

### 1.2 AMOS

官方预处理数据：
https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/En8eq9ClytlAi8ZJaJBLswoB5tfJElLm1yd86gF2WIZVGw?e=7LhcfH

解压到项目根目录：
```
./amos_data/
├── npy/
│   ├── amos_0001_image.npy
│   └── ...
└── splits/
```

AMOS 5% splits 已包含在本仓库 `splits/amos/`：
```bash
cp splits/amos/* amos_data/splits/
```

---

## 2. 权重下载（Git LFS）

```bash
git lfs install
git lfs pull
```

权重目录结构：
```
weights/
├── synapse_baseline/
│   ├── fold1/ckpts/best_model.pth   # Synapse DHC baseline, seed=0
│   ├── fold2/ckpts/best_model.pth   # Synapse DHC baseline, seed=1
│   └── fold3/ckpts/best_model.pth   # Synapse DHC baseline, seed=666 ← BEST (Dice=48.69)
├── synapse_fused/
│   └── fold1/ckpts/best_model.pth   # Synapse DHC+FusedProxy, seed=0
└── amos_fused/
    └── fold1/ckpts/best_model.pth   # AMOS 5% DHC+FusedProxy, seed=0
```

链接到推理路径：
```bash
bash scripts/setup_for_inference.sh
```

---

## 3. 推理（已有权重）

### 3.1 Synapse — DHC Baseline（3-fold）

```bash
source env.sh

# 逐 fold 推理（生成 predictions_AB/）
for fold in 1 2 3; do
  python code/test.py --task synapse \
    --exp Task_synapse_20p/dhc/fold${fold} -g 0 --cps AB
done

# 官方 3-fold Dice 评估
python code/evaluate_Ntimes_npy.py \
  --task synapse --exp Task_synapse_20p/dhc --folds 3 --cps AB

# 4 指标评估（Dice/HD95/NSD/ASD）
python code/evaluate_4metrics.py \
  --task synapse --exp Task_synapse_20p/dhc --folds 3 --cps AB
```

期望结果：`Dice = 47.03 ± 1.31`

### 3.2 Synapse — DHC+FusedProxy（fold1）

```bash
python code/test.py --task synapse \
  --exp Task_synapse_20p/dhc_fused/fold1 -g 0 --cps AB

python code/evaluate_4metrics.py \
  --task synapse --exp Task_synapse_20p/dhc_fused --folds 1 --cps AB
```

fold1 期望：`Dice ≈ 46.11`（3-fold 均值 48.58，fold2/3 权重待补）

### 3.3 AMOS 5% — DHC+FusedProxy（fold1）

```bash
python code/test.py --task amos \
  --exp Task_amos_5p/dhc/fold1 -g 0 --cps AB

python code/evaluate_4metrics.py \
  --task amos --exp Task_amos_5p/dhc --folds 1 --cps AB
```

fold1 期望：`Dice ≈ 40.02`（3-fold 均值 41.67，fold2/3 权重待补）

---

## 4. 从头训练

### 4.1 Synapse 20% — DHC Baseline（3-fold）

```bash
bash train3times_seeds_20p.sh \
  -c 0 -t synapse -m dhc -e '' -l 3e-2 -w 0.1
```

### 4.2 Synapse 20% — DHC+FusedProxy（3-fold）

```bash
bash train3times_seeds_20p.sh \
  -c 0 -t synapse -m dhc -e '_fused' -l 3e-2 -w 0.1 \
  -x "--lambda_cdba 0.05 --lambda_sac 0.2 --lambda_var 0.2 --vapl_warmup 3000"
```

生成实验目录：`logs/Task_synapse_20p/dhc_fused/fold{1,2,3}/`

### 4.3 AMOS 5% — DHC+FusedProxy（3-fold）

```bash
bash train3times_seeds_20p.sh \
  -c 0 -t amos -m dhc -e '' -l 3e-2 -w 0.1 -p 5 \
  -x "--lambda_cdba 0.05 --lambda_sac 0.2 --lambda_var 0.2 --vapl_warmup 3000"
```

生成实验目录：`logs/Task_amos_5p/dhc/fold{1,2,3}/`

训练约需 300 epochs × 24 iters/epoch，A800 GPU 约 8-10 小时/fold。

---

## 5. 关键文件说明

| 文件 | 说明 |
|------|------|
| `code/train_dhc.py` | DHC 训练脚本，已集成 FusedProxyLoss |
| `code/models/vnet.py` | VNet 模型（添加了 `return_features` 接口） |
| `code/utils/fused_proxy.py` | FusedProxyLoss 模块（SAC + CDBA + Variation Vectors） |
| `code/test.py` | 推理脚本，输出 `predictions_AB/*.nii.gz` |
| `code/evaluate_4metrics.py` | 4 指标评估（Dice/HD95/NSD/ASD，支持 Synapse+AMOS） |
| `code/evaluate_Ntimes_npy.py` | 原始官方 Dice 评估 |
| `code/utils/config.py` | 数据路径与超参数配置 |
| `splits/` | Synapse 20% 数据划分 |
| `splits/amos/` | AMOS 5% 数据划分 |
| `results/ALL_EXPERIMENTS_DETAILED.txt` | 全实验逐类别完整指标 |

---

## 6. 实验结果汇总

| 实验 | Dice | HD95 | NSD | ASD |
|------|------|------|-----|-----|
| Synapse 20% DHC baseline (3-fold) | 47.03±1.31 | 18.82±2.32 | 62.13±1.59 | 8.46±2.45 |
| Synapse 20% DHC+FusedProxy (3-fold) | **48.58±1.88** | **18.56±0.27** | **63.88±2.02** | 9.46±0.31 |
| AMOS 5% DHC+FusedProxy (3-fold) | **41.67±1.17** | 53.87±2.01 | 45.07±0.86 | 39.01±0.89 |

详细逐类别数据见 `results/ALL_EXPERIMENTS_DETAILED.txt`。
