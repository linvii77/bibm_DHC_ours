"""Evaluate Dice / HD95 / NSD / ASD for DHC experiments (Synapse, 3-fold)."""
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from medpy import metric
from scipy.ndimage import binary_erosion, distance_transform_edt

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp',  type=str, required=True)
parser.add_argument('--folds', type=int, default=3)
parser.add_argument('--cps',  type=str, default='AB')
parser.add_argument('--nsd_tau', type=float, default=2.0,
                    help='NSD tolerance in voxels')
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(__file__))
from utils import read_list, read_nifti
from utils.config import Config
config = Config(args.task)


def surface_distances(pred, label):
    """Return (d_pred→label, d_label→pred) in voxels."""
    pred  = pred.astype(bool)
    label = label.astype(bool)
    pred_surf  = pred  ^ binary_erosion(pred,  iterations=1)
    label_surf = label ^ binary_erosion(label, iterations=1)
    dt_label = distance_transform_edt(~label)
    dt_pred  = distance_transform_edt(~pred)
    d_p2l = dt_label[pred_surf].astype(np.float32)
    d_l2p = dt_pred[label_surf].astype(np.float32)
    return d_p2l, d_l2p


def nsd(pred, label, tau):
    if pred.sum() == 0 and label.sum() == 0:
        return 100.0
    if pred.sum() == 0 or label.sum() == 0:
        return 0.0
    d_p2l, d_l2p = surface_distances(pred, label)
    if len(d_p2l) == 0 and len(d_l2p) == 0:
        return 100.0
    n = (np.sum(d_p2l <= tau) + np.sum(d_l2p <= tau))
    d = (len(d_p2l) + len(d_l2p))
    return float(n) / float(d) * 100.0


def metrics_one_class(pred_i, label_i, tau):
    """Return (dice, hd95, nsd_val, asd) as floats."""
    p = pred_i.astype(bool)
    l = label_i.astype(bool)
    if not p.any() and not l.any():
        return 100.0, 0.0, 100.0, 0.0
    if not p.any() or not l.any():
        return 0.0, 128.0, 0.0, 128.0
    dc   = metric.binary.dc(p, l) * 100
    hd   = metric.binary.hd95(p, l)
    asd_ = metric.binary.asd(p, l)
    nsd_ = nsd(p, l, tau)
    return float(dc), float(hd), float(nsd_), float(asd_)


if __name__ == '__main__':
    ids_list  = read_list('test', task=args.task)
    test_cls  = list(range(1, config.num_cls))
    n_cls     = len(test_cls)

    txt_path = f'./logs/{args.exp}/evaluation_4metrics.txt'
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    fw = open(txt_path, 'w')

    if args.task == 'amos':
        CLASS_NAMES = [
            'Spleen','Right Kidney','Left Kidney','Gallbladder','Esophagus',
            'Liver','Stomach','Aorta','IVC','Pancreas',
            'Right Adrenal','Left Adrenal','Duodenum','Bladder','Prostate/Uterus'
        ]
    else:
        CLASS_NAMES = [
            'Aorta','Gallbladder','Spleen','Left Kidney','Right Kidney',
            'Liver','Stomach','Pancreas','Duodenum',
            'Portal Vein','Vena Cava','Left Adrenal','Right Adrenal'
        ]
    # values: [n_cases, n_cls, 4]  (dice, hd95, nsd, asd)
    results_all_folds = []

    for fold in range(1, args.folds + 1):
        print(f'\n========== Fold {fold} ==========')
        fw.write(f'\n========== Fold {fold} ==========\n')

        pred_dir = f'./logs/{args.exp}/fold{fold}/predictions_{args.cps}'
        values   = np.zeros((len(ids_list), n_cls, 4))

        for idx, data_id in enumerate(tqdm(ids_list, desc=f'fold{fold}')):
            pred  = read_nifti(os.path.join(pred_dir, f'{data_id}.nii.gz'))
            label = np.load(os.path.join(config.save_dir, 'npy',
                                         f'{data_id}_label.npy')).astype(np.int8)

            for ci, cls in enumerate(test_cls):
                dc, hd, ns, asd_ = metrics_one_class(pred == cls, label == cls,
                                                      args.nsd_tau)
                values[idx, ci] = [dc, hd, ns, asd_]

        vm = values.mean(0)  # [n_cls, 4]
        results_all_folds.append(values)

        for mi, mname in enumerate(['Dice','HD95','NSD','ASD']):
            line = f'  {mname}: ' + str([round(x,1) for x in vm[:,mi].tolist()])
            print(line); fw.write(line+'\n')
        summary = (f'  Mean  Dice={vm[:,0].mean():.2f}  HD95={vm[:,1].mean():.2f}'
                   f'  NSD={vm[:,2].mean():.2f}  ASD={vm[:,3].mean():.2f}')
        print(summary); fw.write(summary+'\n')

    # ── 3-fold average ──
    results_all_folds = np.array(results_all_folds)   # [3, n_cases, n_cls, 4]
    mean_folds = results_all_folds.mean(0)             # [n_cases, n_cls, 4]
    cls_mean   = mean_folds.mean(0)                    # [n_cls, 4]

    print('\n============ 3-Fold Average ============')
    fw.write('\n============ 3-Fold Average ============\n')

    header = f"{'Class':<20} {'Dice':>7} {'HD95':>7} {'NSD':>7} {'ASD':>7}"
    print(header); fw.write(header+'\n')
    fw.write('-'*50+'\n')
    for ci, cname in enumerate(CLASS_NAMES[:n_cls]):
        row = (f'{cname:<20} {cls_mean[ci,0]:>7.1f} {cls_mean[ci,1]:>7.1f}'
               f' {cls_mean[ci,2]:>7.1f} {cls_mean[ci,3]:>7.1f}')
        print(row); fw.write(row+'\n')
    fw.write('-'*50+'\n')

    # per-fold summary for std
    fold_means = results_all_folds.mean(1).mean(1)  # [3, 4]
    stds = fold_means.std(0)
    means = cls_mean.mean(0)

    final = (f'\nFinal (mean±std over 3 folds)\n'
             f'  Dice : {means[0]:.2f} ± {stds[0]:.2f}\n'
             f'  HD95 : {means[1]:.2f} ± {stds[1]:.2f}\n'
             f'  NSD  : {means[2]:.2f} ± {stds[2]:.2f}\n'
             f'  ASD  : {means[3]:.2f} ± {stds[3]:.2f}')
    print(final); fw.write(final+'\n')
    fw.close()
    print(f'\nSaved → {txt_path}')
