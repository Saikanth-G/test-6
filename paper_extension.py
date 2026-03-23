"""
QHSA-Net Paper Extension Script
================================
Covers 4 gaps to make the paper publication-ready:

  GAP 1 — Cross-dataset generalisation
           Indian Pines + Salinas: all 6 models, seed=42

  GAP 2 — Multi-seed statistical credibility
           All 3 datasets, seeds=[42,7,21] for QHSA-Net/SSRN/3D-CNN-Only
           (PaviaU seed=42 reused from existing CSV; 2 extra seeds added)

  GAP 3 — Data efficiency experiment
           QHSA-Net vs SSRN vs 3D-CNN-Only on all 3 datasets
           Train fractions: 1%, 2%, 5%, 10%
           Tests whether quantum advantage emerges at low data

  GAP 4 — Training stability (FREE from GAP 2)
           Per-epoch loss/acc saved for every multi-seed run
           Plotted as mean ± std shaded bands

All results saved to CSV after every section.
All figures saved as PNG.
"""

# ============================================================
# 0. IMPORTS & CONFIG
# ============================================================
import os, sys, time, json, logging, warnings, copy
warnings.filterwarnings('ignore')

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pennylane as qml

from sklearn.svm import SVC
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_score,
                             recall_score, f1_score,
                             roc_auc_score)

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

WORKDIR    = r'c:/Users/saika/OneDrive/Desktop/test 6'
LOG_PATH   = os.path.join(WORKDIR, 'paper_extension.log')

# Best config from Pavia U ablation
BEST_DR          = 'FactorAnalysis'
BEST_N_QUBITS    = 4
BEST_N_LAYERS    = 2
BEST_MEASUREMENT = 'softmax_z'
N_COMP           = BEST_N_QUBITS   # DR components == n_qubits

PATCH_SIZE   = 9
EPOCHS       = 30
BATCH_SIZE   = 64
DEVICE       = torch.device('cpu')
SEEDS        = [42, 7, 21]
KEY_MODELS   = ['QHSA-Net', 'SSRN', '3D-CNN-Only']
ALL_MODELS   = ['QHSA-Net', 'SSRN', 'DBDA', '3D-CNN-Only', 'HybridSN', 'SVM']
DEFRACTIONS  = [0.01, 0.02, 0.05, 0.10]

DATASETS = {
    'PaviaU': {
        'data_path': os.path.join(WORKDIR, 'pavia u data', 'PaviaU.mat'),
        'gt_path':   os.path.join(WORKDIR, 'pavia u data', 'PaviaU_gt.mat'),
        'data_key':  'paviaU',
        'gt_key':    'paviaU_gt',
        'n_classes': 9,
        'class_names': ['Asphalt','Meadows','Gravel','Trees',
                        'Painted metal sheets','Bare soil','Bitumen',
                        'Self-blocking bricks','Shadows'],
    },
    'IndianPines': {
        'data_path': os.path.join(WORKDIR, 'indian pines data', 'Indian_pines_corrected.mat'),
        'gt_path':   os.path.join(WORKDIR, 'indian pines data', 'Indian_pines_gt.mat'),
        'data_key':  'indian_pines_corrected',
        'gt_key':    'indian_pines_gt',
        'n_classes': 16,
        'class_names': ['Alfalfa','Corn-notill','Corn-mintill','Corn',
                        'Grass-pasture','Grass-trees','Grass-pasture-mowed',
                        'Hay-windrowed','Oats','Soybean-notill','Soybean-mintill',
                        'Soybean-clean','Wheat','Woods',
                        'Buildings-Grass-Trees','Stone-Steel-Towers'],
    },
    'Salinas': {
        'data_path': os.path.join(WORKDIR, 'salinas data', 'Salinas_corrected.mat'),
        'gt_path':   os.path.join(WORKDIR, 'salinas data', 'Salinas_gt.mat'),
        'data_key':  'salinas_corrected',
        'gt_key':    'salinas_gt',
        'n_classes': 16,
        'class_names': ['Weeds_1','Weeds_2','Fallow','Fallow_rough','Fallow_smooth',
                        'Stubble','Celery','Grapes','Soil','Corn_senesced',
                        'Lettuce_4wk','Lettuce_5wk','Lettuce_6wk','Lettuce_7wk',
                        'Vineyard_untrained','Vineyard_vertical'],
    },
}

# ============================================================
# 1. LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.FileHandler(LOG_PATH, 'w', 'utf-8'),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger()

def section(title):
    bar = '=' * 60
    log.info(f'\n{bar}\n  {title}\n{bar}')

# ============================================================
# 2. DATA LOADING
# ============================================================
def load_dataset(name, seed=42):
    """Load a dataset, extract 9x9 patches, fit FactorAnalysis, return splits."""
    cfg = DATASETS[name]
    raw    = sio.loadmat(cfg['data_path'])
    gt_raw = sio.loadmat(cfg['gt_path'])

    # handle variable key names
    data_key = cfg['data_key']
    gt_key   = cfg['gt_key']
    if data_key not in raw:
        data_key = [k for k in raw if not k.startswith('_')][0]
    if gt_key not in gt_raw:
        gt_key = [k for k in gt_raw if not k.startswith('_')][0]

    HSI = raw[data_key].astype(np.float32)
    GT  = gt_raw[gt_key].astype(np.int32)
    H, W, B = HSI.shape
    n_classes = cfg['n_classes']

    # per-band normalisation
    mn = HSI.min(axis=(0,1), keepdims=True)
    mx = HSI.max(axis=(0,1), keepdims=True)
    HSI = (HSI - mn) / (mx - mn + 1e-8)

    # 10/90 stratified split
    rng = np.random.default_rng(seed)
    rows, cols = np.where(GT > 0)
    labels = GT[rows, cols] - 1   # 0-indexed

    tr_idx_list, te_idx_list = [], []
    for c in range(n_classes):
        cidx = np.where(labels == c)[0]
        if len(cidx) == 0:
            continue
        n_tr = max(int(0.10 * len(cidx)), 3)
        perm = rng.permutation(len(cidx))
        tr_idx_list.extend(cidx[perm[:n_tr]].tolist())
        te_idx_list.extend(cidx[perm[n_tr:]].tolist())

    tr_idx = np.array(tr_idx_list)
    te_idx = np.array(te_idx_list)

    # patch extraction
    PAD = PATCH_SIZE // 2
    hsi_pad = np.pad(HSI, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')

    def extract(ridx, cidx_):
        out = np.empty((len(ridx), B, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for i, (r, c_) in enumerate(zip(rows[ridx], cols[ridx])):
            p = hsi_pad[r:r+PATCH_SIZE, c_:c_+PATCH_SIZE, :]
            out[i] = p.transpose(2, 0, 1)
        return out

    X_tr = extract(tr_idx, None)
    X_te = extract(te_idx, None)
    y_tr = labels[tr_idx].astype(np.int64)
    y_te = labels[te_idx].astype(np.int64)

    # centre spectra
    spec_tr = X_tr[:, :, PAD, PAD]
    spec_te = X_te[:, :, PAD, PAD]

    # FactorAnalysis DR
    fa = FactorAnalysis(n_components=N_COMP, random_state=42)
    fa_tr = fa.fit_transform(spec_tr).astype(np.float32)
    fa_te = fa.transform(spec_te).astype(np.float32)

    log.info(f'  {name} | H={H} W={W} B={B} | classes={n_classes} | '
             f'train={len(y_tr)} test={len(y_te)} | seed={seed}')

    return dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                fa_tr=fa_tr, fa_te=fa_te, spec_tr=spec_tr, spec_te=spec_te,
                n_bands=B, n_classes=n_classes, fa=fa)


def load_dataset_fraction(name, fraction, seed=42):
    """Like load_dataset but with a smaller train fraction (stratified)."""
    cfg = DATASETS[name]
    raw    = sio.loadmat(cfg['data_path'])
    gt_raw = sio.loadmat(cfg['gt_path'])

    data_key = cfg['data_key']
    gt_key   = cfg['gt_key']
    if data_key not in raw:
        data_key = [k for k in raw if not k.startswith('_')][0]
    if gt_key not in gt_raw:
        gt_key = [k for k in gt_raw if not k.startswith('_')][0]

    HSI = raw[data_key].astype(np.float32)
    GT  = gt_raw[gt_key].astype(np.int32)
    H, W, B = HSI.shape
    n_classes = cfg['n_classes']

    mn = HSI.min(axis=(0,1), keepdims=True)
    mx = HSI.max(axis=(0,1), keepdims=True)
    HSI = (HSI - mn) / (mx - mn + 1e-8)

    rng = np.random.default_rng(seed)
    rows, cols = np.where(GT > 0)
    labels = GT[rows, cols] - 1

    tr_idx_list, te_idx_list = [], []
    for c in range(n_classes):
        cidx = np.where(labels == c)[0]
        if len(cidx) == 0:
            continue
        # min 2 samples per class for training, rest for test
        n_total_c = len(cidx)
        n_tr = max(int(fraction * n_total_c), 2)
        n_tr = min(n_tr, n_total_c - 1)   # always leave at least 1 for test
        perm = rng.permutation(n_total_c)
        tr_idx_list.extend(cidx[perm[:n_tr]].tolist())
        te_idx_list.extend(cidx[perm[n_tr:]].tolist())

    tr_idx = np.array(tr_idx_list)
    te_idx = np.array(te_idx_list)

    PAD = PATCH_SIZE // 2
    hsi_pad = np.pad(HSI, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')

    def extract(ridx):
        out = np.empty((len(ridx), B, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        for i, (r, c_) in enumerate(zip(rows[ridx], cols[ridx])):
            p = hsi_pad[r:r+PATCH_SIZE, c_:c_+PATCH_SIZE, :]
            out[i] = p.transpose(2, 0, 1)
        return out

    X_tr = extract(tr_idx)
    X_te = extract(te_idx)
    y_tr = labels[tr_idx].astype(np.int64)
    y_te = labels[te_idx].astype(np.int64)

    spec_tr = X_tr[:, :, PAD, PAD]
    spec_te = X_te[:, :, PAD, PAD]

    fa = FactorAnalysis(n_components=N_COMP, random_state=42)
    fa_tr = fa.fit_transform(spec_tr).astype(np.float32)
    fa_te = fa.transform(spec_te).astype(np.float32)

    return dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                fa_tr=fa_tr, fa_te=fa_te,
                n_bands=B, n_classes=n_classes, n_train=len(y_tr))


# ============================================================
# 3. DATASET CLASS & LOADERS
# ============================================================
class HSIDataset(Dataset):
    def __init__(self, patches, fa_feats, labels):
        self.patches = torch.from_numpy(patches)
        self.fa      = torch.from_numpy(fa_feats)
        self.labels  = torch.from_numpy(labels).long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.patches[i], self.fa[i], self.labels[i]

def make_loaders(d):
    tr = DataLoader(HSIDataset(d['X_tr'], d['fa_tr'], d['y_tr']),
                    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    te = DataLoader(HSIDataset(d['X_te'], d['fa_te'], d['y_te']),
                    batch_size=256,        shuffle=False, num_workers=0)
    return tr, te


# ============================================================
# 4. METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_prob, n_classes):
    oa    = accuracy_score(y_true, y_pred) * 100
    cm    = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    pc_acc = np.where(cm.sum(1) > 0, cm.diagonal() / cm.sum(1) * 100, 0.0)
    aa    = float(np.mean(pc_acc))
    kappa = cohen_kappa_score(y_true, y_pred) * 100
    f1    = f1_score(y_true, y_pred, average=None, zero_division=0,
                     labels=list(range(n_classes))) * 100
    mac_f1 = float(np.mean(f1))
    mac_prec = precision_score(y_true, y_pred, average='macro',
                               zero_division=0, labels=list(range(n_classes))) * 100
    macro_auc = None
    if y_prob is not None:
        try:
            yb = label_binarize(y_true, classes=list(range(n_classes)))
            macro_auc = roc_auc_score(yb, y_prob, average='macro',
                                      multi_class='ovr') * 100
        except Exception:
            macro_auc = None
    return dict(OA=oa, AA=aa, kappa=kappa, mac_f1=mac_f1,
                mac_prec=mac_prec, macro_auc=macro_auc,
                pc_f1=f1.tolist())


# ============================================================
# 5. MODEL DEFINITIONS (parameterised for any dataset)
# ============================================================
def make_vqc(n_qubits, n_layers, measurement='softmax_z'):
    dev = qml.device('default.qubit', wires=n_qubits)
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    out_dim = n_qubits
    return circuit, {'weights': wshape}, out_dim, measurement


class QuantumBranch(nn.Module):
    def __init__(self, n_qubits=BEST_N_QUBITS, n_layers=BEST_N_LAYERS,
                 proj_dim=64, measurement=BEST_MEASUREMENT):
        super().__init__()
        circuit, wshape, out_dim, self.mtype = make_vqc(n_qubits, n_layers, measurement)
        self.qlayer = qml.qnn.TorchLayer(circuit, wshape)
        self.proj   = nn.Sequential(nn.Linear(out_dim, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, x):
        x = torch.tanh(x) * np.pi
        q_out = self.qlayer(x)
        if self.mtype == 'softmax_z':
            q_out = torch.softmax(q_out, dim=-1)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out, dim=-1)
        return self.proj(q_out)


class ClassicalBranch(nn.Module):
    def __init__(self, n_bands, proj_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1,  8, (7,3,3), padding=(3,1,1)), nn.BatchNorm3d(8),  nn.ReLU(),
            nn.Conv3d(8, 16, (5,3,3), padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32, (3,3,3), padding=(1,1,1)), nn.BatchNorm3d(32), nn.ReLU(),
        )
        dummy = torch.zeros(1, 1, n_bands, PATCH_SIZE, PATCH_SIZE)
        flat  = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)).flatten(1))


class GatedFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())

    def forward(self, fc, fq):
        alpha = self.gate(torch.cat([fc, fq], dim=-1))
        return alpha * fq + (1 - alpha) * fc


class QHSANet(nn.Module):
    def __init__(self, n_bands, n_classes, proj_dim=64):
        super().__init__()
        self.classical = ClassicalBranch(n_bands, proj_dim)
        self.quantum   = QuantumBranch(BEST_N_QUBITS, BEST_N_LAYERS, proj_dim, BEST_MEASUREMENT)
        self.fusion    = GatedFusion(proj_dim)
        self.clf = nn.Sequential(nn.Linear(proj_dim, 128), nn.GELU(),
                                 nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, patch, fa):
        fc = self.classical(patch)
        fq = self.quantum(fa)
        return self.clf(self.fusion(fc, fq))


class CNN3DOnly(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, (7,3,3), padding=(3,1,1)), nn.BatchNorm3d(8),  nn.ReLU(),
            nn.Conv3d(8,16, (5,3,3), padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3), padding=(1,1,1)), nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 128), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, patch, fa=None):
        return self.head(self.conv(patch.unsqueeze(1)))


class HybridSN(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.conv3d_1 = nn.Sequential(nn.Conv3d(1,  8,(7,3,3),padding=(3,1,1)), nn.ReLU())
        self.conv3d_2 = nn.Sequential(nn.Conv3d(8, 16,(5,3,3),padding=(2,1,1)), nn.ReLU())
        self.conv3d_3 = nn.Sequential(nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)), nn.ReLU())
        self.band_pool = nn.AdaptiveAvgPool3d((1, PATCH_SIZE, PATCH_SIZE))
        self.conv2d_1  = nn.Sequential(nn.Conv2d(32, 64,3,padding=1), nn.ReLU())
        self.conv2d_2  = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.ReLU())
        self.pool2d    = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128,256), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, patch, fa=None):
        x = patch.unsqueeze(1)
        x = self.conv3d_3(self.conv3d_2(self.conv3d_1(x)))
        x = self.band_pool(x).squeeze(2)
        return self.head(self.pool2d(self.conv2d_2(self.conv2d_1(x))))


class SpectralResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, (1,1,7), padding=(0,0,3)),
            nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, (1,1,7), padding=(0,0,3)),
        )
    def forward(self, x): return x + self.block(x)


class SpatialResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, (3,3,1), padding=(1,1,0)),
            nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, (3,3,1), padding=(1,1,0)),
        )
    def forward(self, x): return x + self.block(x)


class SSRN(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        self.spec_conv  = nn.Conv3d(1, 24, (1,1,7), padding=(0,0,3))
        self.spec_res1  = SpectralResBlock(24)
        self.spec_res2  = SpectralResBlock(24)
        self.spec2spat  = nn.Conv3d(24, 128, (1, 1, n_bands))
        self.spat_res1  = SpatialResBlock(128)
        self.spat_res2  = SpatialResBlock(128)
        self.pool       = nn.AdaptiveAvgPool3d(1)
        self.fc         = nn.Linear(128, n_classes)

    def forward(self, x, fa=None):
        x = x.permute(0,2,3,1).unsqueeze(1)   # (N,B,H,W)→(N,1,H,W,B)
        x = self.spec_conv(x)
        x = self.spec_res2(self.spec_res1(x))
        x = self.spec2spat(x)
        x = self.spat_res2(self.spat_res1(x))
        return self.fc(self.pool(x).flatten(1))


class ChannelAttn(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(1, ch//r)), nn.ReLU(),
                                nn.Linear(max(1, ch//r), ch), nn.Sigmoid())
    def forward(self, x):
        gap = x.flatten(2).mean(dim=2)
        w = self.fc(gap).view(x.shape[0], x.shape[1], *([1]*(x.dim()-2)))
        return x * w


class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, (3,3,1), padding=(1,1,0))
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class DBDA(nn.Module):
    def __init__(self, n_bands, n_classes):
        super().__init__()
        GR = 12
        self.sc0 = nn.Sequential(nn.Conv3d(1,   GR,(1,1,7),padding=(0,0,3)), nn.BatchNorm3d(GR), nn.ReLU())
        self.sc1 = nn.Sequential(nn.Conv3d(GR,  GR,(1,1,7),padding=(0,0,3)), nn.BatchNorm3d(GR), nn.ReLU())
        self.sc2 = nn.Sequential(nn.Conv3d(GR*2,GR,(1,1,7),padding=(0,0,3)), nn.BatchNorm3d(GR), nn.ReLU())
        self.sc3 = nn.Sequential(nn.Conv3d(GR*3,GR,(1,1,7),padding=(0,0,3)), nn.BatchNorm3d(GR), nn.ReLU())
        self.spec_ca   = ChannelAttn(GR*4)
        self.spec_pool = nn.AdaptiveAvgPool3d(1)
        self.tc0 = nn.Sequential(nn.Conv3d(1,   GR,(3,3,1),padding=(1,1,0)), nn.BatchNorm3d(GR), nn.ReLU())
        self.tc1 = nn.Sequential(nn.Conv3d(GR,  GR,(3,3,1),padding=(1,1,0)), nn.BatchNorm3d(GR), nn.ReLU())
        self.tc2 = nn.Sequential(nn.Conv3d(GR*2,GR,(3,3,1),padding=(1,1,0)), nn.BatchNorm3d(GR), nn.ReLU())
        self.tc3 = nn.Sequential(nn.Conv3d(GR*3,GR,(3,3,1),padding=(1,1,0)), nn.BatchNorm3d(GR), nn.ReLU())
        self.spat_sa   = SpatialAttn()
        self.spat_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Linear(GR*8, 128), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x, fa=None):
        x3 = x.permute(0,2,3,1).unsqueeze(1)
        s0 = self.sc0(x3)
        s1 = self.sc1(s0)
        s2 = self.sc2(torch.cat([s0,s1],1))
        s3 = self.sc3(torch.cat([s0,s1,s2],1))
        sf = self.spec_pool(self.spec_ca(torch.cat([s0,s1,s2,s3],1))).flatten(1)
        t0 = self.tc0(x3)
        t1 = self.tc1(t0)
        t2 = self.tc2(torch.cat([t0,t1],1))
        t3 = self.tc3(torch.cat([t0,t1,t2],1))
        tf = self.spat_pool(self.spat_sa(torch.cat([t0,t1,t2,t3],1))).flatten(1)
        return self.head(torch.cat([sf,tf],1))


MODEL_BUILDERS = {
    'QHSA-Net':   QHSANet,
    'SSRN':       SSRN,
    'DBDA':       DBDA,
    '3D-CNN-Only':CNN3DOnly,
    'HybridSN':   HybridSN,
}


# ============================================================
# 6. TRAINING & EVALUATION
# ============================================================
def train_model(model, loader, n_epochs, tag='', save_curves=True):
    """Train model; returns (train_time_s, epoch_records)."""
    q_params  = [p for n,p in model.named_parameters() if 'quantum' in n or 'qlayer' in n]
    cl_params = [p for n,p in model.named_parameters() if 'quantum' not in n and 'qlayer' not in n]
    groups = []
    if cl_params: groups.append({'params': cl_params, 'lr': 1e-3})
    if q_params:  groups.append({'params': q_params,  'lr': 1e-2})
    opt   = optim.Adam(groups or model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.CrossEntropyLoss()
    epoch_records = []
    model.train()
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        tot_loss, tot_corr, tot_n = 0., 0, 0
        for pb, fb, lb in loader:
            pb, fb, lb = pb.to(DEVICE), fb.to(DEVICE), lb.to(DEVICE)
            opt.zero_grad()
            out  = model(pb, fb)
            loss = crit(out, lb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(lb)
            tot_corr += (out.argmax(1) == lb).sum().item()
            tot_n    += len(lb)
        sched.step()
        ep_loss = tot_loss / tot_n
        ep_acc  = tot_corr / tot_n * 100
        if save_curves:
            epoch_records.append({'epoch': ep, 'loss': ep_loss, 'train_acc': ep_acc})
        if ep % 5 == 0 or ep == 1:
            log.info(f'  [{tag}] ep {ep:3d}/{n_epochs}  '
                     f'loss={ep_loss:.4f}  acc={ep_acc:.1f}%  ({time.time()-t0:.0f}s)')
    return time.time() - t0, epoch_records


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    yt_all, yp_all, prob_all = [], [], []
    for pb, fb, lb in loader:
        pb, fb = pb.to(DEVICE), fb.to(DEVICE)
        out  = model(pb, fb)
        prob = torch.softmax(out, dim=-1)
        yp_all.append(out.argmax(1).cpu().numpy())
        yt_all.append(lb.numpy())
        prob_all.append(prob.cpu().numpy())
    return np.concatenate(yt_all), np.concatenate(yp_all), np.concatenate(prob_all)


def run_neural_model(model_name, d, seed, tag):
    """Train and evaluate one neural model on dataset dict d."""
    torch.manual_seed(seed); np.random.seed(seed)
    n_bands   = d['n_bands']
    n_classes = d['n_classes']
    model = MODEL_BUILDERS[model_name](n_bands, n_classes).to(DEVICE)
    tr_loader, te_loader = make_loaders(d)
    t_train, curves = train_model(model, tr_loader, EPOCHS, tag=tag)
    t0 = time.time()
    yt, yp, yprob = eval_model(model, te_loader)
    t_inf = time.time() - t0
    m = compute_metrics(yt, yp, yprob, n_classes)
    log.info(f'  {tag}: OA={m["OA"]:.2f}%  AA={m["AA"]:.2f}%  '
             f'kappa={m["kappa"]:.2f}  AUC={m["macro_auc"]:.2f}%  '
             f'train={t_train/60:.1f}min')
    del model
    return m, t_train, t_inf, curves


def run_svm(d, seed, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    spec_tr = d['spec_tr'] if 'spec_tr' in d else d['X_tr'][:, :, PATCH_SIZE//2, PATCH_SIZE//2]
    spec_te = d['spec_te'] if 'spec_te' in d else d['X_te'][:, :, PATCH_SIZE//2, PATCH_SIZE//2]
    scaler = StandardScaler().fit(spec_tr)
    t0 = time.time()
    svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=seed)
    svm.fit(scaler.transform(spec_tr), d['y_tr'])
    t_train = time.time() - t0
    t0 = time.time()
    yp    = svm.predict(scaler.transform(spec_te))
    yprob = svm.predict_proba(scaler.transform(spec_te))
    t_inf = time.time() - t0
    m = compute_metrics(d['y_te'], yp, yprob, d['n_classes'])
    log.info(f'  {tag}: OA={m["OA"]:.2f}%  AA={m["AA"]:.2f}%  '
             f'kappa={m["kappa"]:.2f}  train={t_train:.1f}s')
    return m, t_train, t_inf, []


def row_from_metrics(model, dataset, seed, m, t_train, t_inf):
    r = dict(model=model, dataset=dataset, seed=seed,
             OA=m['OA'], AA=m['AA'], kappa=m['kappa'],
             mac_f1=m['mac_f1'], mac_prec=m['mac_prec'],
             macro_auc=m['macro_auc'],
             train_time_s=t_train, infer_time_s=t_inf)
    # add per-class F1
    for i, v in enumerate(m['pc_f1']):
        r[f'pc_f1_{i}'] = v
    return r


# ============================================================
# 7. SECTION 1+2 — CROSS-DATASET + MULTI-SEED
# ============================================================
section('Sections 1+2: Cross-dataset & Multi-seed Evaluation')

# Reuse existing PaviaU seed=42 results for all 6 models
pavia_s6 = pd.read_csv(os.path.join(WORKDIR, 's6_baseline_results.csv'))
pavia_s5 = pd.read_csv(os.path.join(WORKDIR, 's5_final_qhsa.csv'))

eval_rows  = []    # main results table
curves_all = []    # for training stability

# Map existing PaviaU results (seed=42)
model_name_map = {
    'QHSA-Net (Optimised)': 'QHSA-Net',
    'SVM (RBF)': 'SVM',
    '3D-CNN-Only': '3D-CNN-Only',
    'HybridSN': 'HybridSN',
    'SSRN': 'SSRN',
    'DBDA': 'DBDA',
}
for _, row in pavia_s6.iterrows():
    mname = model_name_map.get(row['model'], row['model'])
    pc_f1 = {f'pc_f1_{i}': row.get(f'pc_f1_{i}', np.nan)
             for i in range(DATASETS['PaviaU']['n_classes'])}
    r = dict(model=mname, dataset='PaviaU', seed=42,
             OA=row['OA'], AA=row['AA'], kappa=row['kappa'],
             mac_f1=row.get('mac_f1', np.nan), mac_prec=row.get('mac_prec', np.nan),
             macro_auc=row.get('macro_auc', np.nan),
             train_time_s=row.get('train_time_s', np.nan),
             infer_time_s=row.get('infer_time_s', np.nan))
    r.update(pc_f1)
    eval_rows.append(r)
# Also add QHSA-Net from s5
for _, row in pavia_s5.iterrows():
    mname = model_name_map.get(row['model'], row['model'])
    pc_f1 = {f'pc_f1_{i}': row.get(f'pc_f1_{i}', np.nan)
             for i in range(DATASETS['PaviaU']['n_classes'])}
    r = dict(model=mname, dataset='PaviaU', seed=42,
             OA=row['OA'], AA=row['AA'], kappa=row['kappa'],
             mac_f1=row.get('mac_f1', np.nan), mac_prec=row.get('mac_prec', np.nan),
             macro_auc=row.get('macro_auc', np.nan),
             train_time_s=row.get('train_time_s', np.nan),
             infer_time_s=row.get('infer_time_s', np.nan))
    r.update(pc_f1)
    eval_rows.append(r)

log.info('  PaviaU seed=42 results loaded from existing CSVs.')

# Run Indian Pines and Salinas
for ds_name in ['IndianPines', 'Salinas']:
    section(f'Dataset: {ds_name}')
    n_classes = DATASETS[ds_name]['n_classes']

    for seed in SEEDS:
        log.info(f'\n--- {ds_name} | seed={seed} ---')
        d = load_dataset(ds_name, seed=seed)

        for mname in ALL_MODELS:
            tag = f'{ds_name[:2]}-{mname[:6]}-s{seed}'
            # Key models: run all seeds; other models: only seed=42
            if mname not in KEY_MODELS and seed != 42:
                continue
            try:
                if mname == 'SVM':
                    m, t_tr, t_inf, _ = run_svm(d, seed, tag)
                else:
                    m, t_tr, t_inf, curves = run_neural_model(mname, d, seed, tag)
                    if mname in KEY_MODELS:
                        for c in curves:
                            curves_all.append({'model': mname, 'dataset': ds_name,
                                               'seed': seed, **c})
                eval_rows.append(row_from_metrics(mname, ds_name, seed, m, t_tr, t_inf))
            except Exception as e:
                log.error(f'  {tag} FAILED: {e}')
                import traceback; log.error(traceback.format_exc())

# Run PaviaU extra seeds (42, 7, 21) for KEY_MODELS only
section('PaviaU — extra seeds [7, 21] for key models')
for seed in [7, 21]:
    log.info(f'\n--- PaviaU | seed={seed} ---')
    d = load_dataset('PaviaU', seed=seed)
    for mname in KEY_MODELS:
        tag = f'PU-{mname[:6]}-s{seed}'
        try:
            if mname == 'SVM':
                m, t_tr, t_inf, _ = run_svm(d, seed, tag)
            else:
                m, t_tr, t_inf, curves = run_neural_model(mname, d, seed, tag)
                for c in curves:
                    curves_all.append({'model': mname, 'dataset': 'PaviaU',
                                       'seed': seed, **c})
            eval_rows.append(row_from_metrics(mname, 'PaviaU', seed, m, t_tr, t_inf))
        except Exception as e:
            log.error(f'  {tag} FAILED: {e}')
            import traceback; log.error(traceback.format_exc())

# Save
df_eval = pd.DataFrame(eval_rows)
df_eval.to_csv(os.path.join(WORKDIR, 'paper_eval_results.csv'), index=False)
df_curves = pd.DataFrame(curves_all)
df_curves.to_csv(os.path.join(WORKDIR, 'paper_training_curves.csv'), index=False)
log.info('\nSections 1+2 results saved.')
log.info('\n' + df_eval[['model','dataset','seed','OA','AA','kappa','macro_auc']].to_string())


# ============================================================
# 8. SECTION 3 — DATA EFFICIENCY
# ============================================================
section('Section 3: Data Efficiency Experiment')

eff_rows = []

for ds_name in ['PaviaU', 'IndianPines', 'Salinas']:
    log.info(f'\n=== Data Efficiency: {ds_name} ===')
    for frac in DEFRACTIONS:
        log.info(f'\n--- Fraction={frac*100:.0f}% ---')
        d = load_dataset_fraction(ds_name, fraction=frac, seed=42)
        log.info(f'  Train samples: {d["n_train"]}')
        for mname in ['QHSA-Net', 'SSRN', '3D-CNN-Only']:
            tag = f'Eff-{ds_name[:2]}-{frac*100:.0f}pct-{mname[:6]}'
            try:
                m, t_tr, t_inf, _ = run_neural_model(mname, d, 42, tag)
                eff_rows.append(dict(
                    model=mname, dataset=ds_name,
                    fraction=frac, n_train=d['n_train'],
                    OA=m['OA'], AA=m['AA'], kappa=m['kappa'],
                    mac_f1=m['mac_f1'], macro_auc=m['macro_auc'],
                    train_time_s=t_tr
                ))
            except Exception as e:
                log.error(f'  {tag} FAILED: {e}')
                import traceback; log.error(traceback.format_exc())

df_eff = pd.DataFrame(eff_rows)
df_eff.to_csv(os.path.join(WORKDIR, 'paper_data_efficiency.csv'), index=False)
log.info('\nSection 3 results saved.')
log.info('\n' + df_eff[['model','dataset','fraction','OA']].to_string())


# ============================================================
# 9. SECTION 4 — FIGURES
# ============================================================
section('Section 4: Generating Figures')

# ── Helper ────────────────────────────────────────────────────
def save(fname):
    path = os.path.join(WORKDIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'  Saved: {fname}')


# ── Fig 1: Cross-dataset OA comparison (all 6 models × 3 datasets, seed=42) ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
ds_order  = ['PaviaU', 'IndianPines', 'Salinas']
clr_map   = {'QHSA-Net':'#2196F3','SSRN':'#FF5722','DBDA':'#4CAF50',
             '3D-CNN-Only':'#9C27B0','HybridSN':'#FF9800','SVM':'#607D8B'}

for ax, ds in zip(axes, ds_order):
    sub = df_eval[(df_eval['dataset']==ds) & (df_eval['seed']==42)]
    # drop duplicate QHSA-Net rows (s5 + s6)
    sub = sub.drop_duplicates(subset=['model'], keep='last')
    sub = sub[sub['model'].isin(ALL_MODELS)].set_index('model')
    sub = sub.reindex([m for m in ALL_MODELS if m in sub.index])
    bars = ax.bar(sub.index, sub['OA'],
                  color=[clr_map.get(m,'#999') for m in sub.index], alpha=0.85)
    ax.bar_label(bars, fmt='%.1f', fontsize=7.5, padding=2)
    ax.set_title(ds.replace('IndianPines','Indian Pines'), fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (%)' if ds == 'PaviaU' else '')
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Cross-Dataset Overall Accuracy Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
save('fig_paper_cross_dataset_oa.png')


# ── Fig 2: Multi-seed mean ± std bar chart (KEY_MODELS × 3 datasets) ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ds in zip(axes, ds_order):
    sub = df_eval[(df_eval['dataset']==ds) & (df_eval['model'].isin(KEY_MODELS))]
    stats = sub.groupby('model')['OA'].agg(['mean','std']).reindex(KEY_MODELS).reset_index()
    bars = ax.bar(stats['model'], stats['mean'],
                  yerr=stats['std'].fillna(0), capsize=5,
                  color=[clr_map[m] for m in stats['model']], alpha=0.85)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in stats['mean']], fontsize=8, padding=3)
    ax.set_title(ds.replace('IndianPines','Indian Pines'), fontsize=12, fontweight='bold')
    ax.set_ylabel('OA mean ± std (%)' if ds == 'PaviaU' else '')
    ax.set_ylim(0, 108)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Multi-Seed Evaluation (3 seeds: 42, 7, 21)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('fig_paper_multi_seed.png')


# ── Fig 3: Summary table heatmap (OA across models × datasets) ───────────────
fig, ax = plt.subplots(figsize=(10, 5))
pivot_data = {}
for ds in ds_order:
    sub = df_eval[(df_eval['dataset']==ds) & (df_eval['seed']==42)]
    sub = sub.drop_duplicates(subset=['model'], keep='last')
    sub = sub[sub['model'].isin(ALL_MODELS)].set_index('model')
    pivot_data[ds.replace('IndianPines','Indian\nPines')] = sub['OA']

pivot_df = pd.DataFrame(pivot_data).reindex(ALL_MODELS)
im = ax.imshow(pivot_df.values, aspect='auto', cmap='RdYlGn', vmin=70, vmax=100)
plt.colorbar(im, ax=ax, label='OA (%)')
ax.set_xticks(range(3)); ax.set_xticklabels(pivot_df.columns, fontsize=11)
ax.set_yticks(range(len(ALL_MODELS))); ax.set_yticklabels(ALL_MODELS, fontsize=11)
for i in range(len(ALL_MODELS)):
    for j in range(3):
        v = pivot_df.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if v < 85 else 'black')
ax.set_title('Overall Accuracy Heatmap — All Models × All Datasets', fontsize=12, fontweight='bold')
plt.tight_layout()
save('fig_paper_oa_heatmap.png')


# ── Fig 4: Data efficiency curves ────────────────────────────────────────────
if not df_eff.empty:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    eff_clr = {'QHSA-Net':'#2196F3','SSRN':'#FF5722','3D-CNN-Only':'#9C27B0'}
    eff_style = {'QHSA-Net':'-o','SSRN':'-s','3D-CNN-Only':'-^'}

    for ax, ds in zip(axes, ds_order):
        sub = df_eff[df_eff['dataset']==ds]
        for mname in ['QHSA-Net','SSRN','3D-CNN-Only']:
            msub = sub[sub['model']==mname].sort_values('fraction')
            if msub.empty: continue
            ax.plot(msub['fraction']*100, msub['OA'],
                    eff_style[mname], color=eff_clr[mname],
                    label=mname, linewidth=2, markersize=7)
        ax.set_title(ds.replace('IndianPines','Indian Pines'), fontsize=12, fontweight='bold')
        ax.set_xlabel('Training fraction (%)')
        ax.set_ylabel('OA (%)' if ds == 'PaviaU' else '')
        ax.set_xlim(0, 11); ax.set_ylim(40, 105)
        ax.set_xticks([1, 2, 5, 10])
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle('Data Efficiency: OA vs Training Set Size', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save('fig_paper_data_efficiency.png')


# ── Fig 5: Training stability (mean ± std shaded curves) ─────────────────────
if not df_curves.empty:
    fig, axes = plt.subplots(len(KEY_MODELS), 3, figsize=(16, 4*len(KEY_MODELS)))
    if len(KEY_MODELS) == 1:
        axes = [axes]

    for row_i, mname in enumerate(KEY_MODELS):
        for col_j, ds in enumerate(ds_order):
            ax = axes[row_i][col_j]
            sub = df_curves[(df_curves['model']==mname) & (df_curves['dataset']==ds)]
            if sub.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            pivot = sub.pivot_table(index='epoch', columns='seed', values='loss')
            if not pivot.empty:
                mn = pivot.mean(axis=1)
                sd = pivot.std(axis=1).fillna(0)
                ax.plot(mn.index, mn.values, color='#2196F3', linewidth=2, label='Mean loss')
                ax.fill_between(mn.index, mn-sd, mn+sd, alpha=0.25, color='#2196F3')
                for s in pivot.columns:
                    ax.plot(pivot.index, pivot[s], alpha=0.3, linewidth=0.8, color='gray')
            ax.set_title(f'{mname} — {ds.replace("IndianPines","Indian Pines")}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Train Loss' if col_j == 0 else '')
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Training Stability: Loss Curves Across 3 Seeds\n'
                 '(Grey lines = individual seeds, Blue band = mean ± std)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('fig_paper_training_stability.png')


# ── Fig 6: Kappa comparison across datasets ───────────────────────────────────
if not df_eval.empty:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, ds in zip(axes, ds_order):
        sub = df_eval[(df_eval['dataset']==ds) & (df_eval['seed']==42)]
        sub = sub.drop_duplicates(subset=['model'], keep='last')
        sub = sub[sub['model'].isin(ALL_MODELS)].set_index('model')
        sub = sub.reindex([m for m in ALL_MODELS if m in sub.index])
        bars = ax.bar(sub.index, sub['kappa'],
                      color=[clr_map.get(m,'#999') for m in sub.index], alpha=0.85)
        ax.bar_label(bars, fmt='%.1f', fontsize=7.5, padding=2)
        ax.set_title(ds.replace('IndianPines','Indian Pines'), fontsize=12, fontweight='bold')
        ax.set_ylabel('Kappa (%)' if ds == 'PaviaU' else '')
        ax.set_ylim(0, 105)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Cross-Dataset Kappa Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save('fig_paper_cross_dataset_kappa.png')


# ── Fig 7: Multi-seed Kappa ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, ds in zip(axes, ds_order):
    sub = df_eval[(df_eval['dataset']==ds) & (df_eval['model'].isin(KEY_MODELS))]
    stats = sub.groupby('model')['kappa'].agg(['mean','std']).reindex(KEY_MODELS).reset_index()
    bars = ax.bar(stats['model'], stats['mean'],
                  yerr=stats['std'].fillna(0), capsize=5,
                  color=[clr_map[m] for m in stats['model']], alpha=0.85)
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in stats['mean']], fontsize=8, padding=3)
    ax.set_title(ds.replace('IndianPines','Indian Pines'), fontsize=12, fontweight='bold')
    ax.set_ylabel('Kappa mean ± std (%)' if ds == 'PaviaU' else '')
    ax.set_ylim(0, 108)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Multi-Seed Kappa (3 seeds: 42, 7, 21)', fontsize=13, fontweight='bold')
plt.tight_layout()
save('fig_paper_multi_seed_kappa.png')


# ── Fig 8: Macro summary table ────────────────────────────────────────────────
# Build mean ± std summary across all seeds for key models, OA/AA/Kappa/F1
rows_summary = []
for ds in ds_order:
    sub_ds = df_eval[df_eval['dataset']==ds]
    for mname in ALL_MODELS:
        sub = sub_ds[sub_ds['model']==mname]
        if sub.empty: continue
        for metric in ['OA','AA','kappa','mac_f1']:
            vals = sub[metric].dropna()
            rows_summary.append(dict(
                model=mname, dataset=ds, metric=metric,
                mean=vals.mean(), std=vals.std() if len(vals)>1 else 0.0,
                n_seeds=len(vals)
            ))
df_summary = pd.DataFrame(rows_summary)
df_summary.to_csv(os.path.join(WORKDIR, 'paper_summary_stats.csv'), index=False)


# ── LaTeX table: cross-dataset OA mean ± std for key models ──────────────────
latex_rows = []
for mname in ALL_MODELS:
    row = {'Model': mname}
    for ds in ds_order:
        sub = df_eval[(df_eval['dataset']==ds) & (df_eval['model']==mname)]
        if sub.empty:
            row[ds] = '—'
            continue
        vals = sub['OA'].dropna()
        if len(vals) > 1:
            row[ds] = f'{vals.mean():.2f}±{vals.std():.2f}'
        else:
            row[ds] = f'{vals.mean():.2f}'
    latex_rows.append(row)

df_latex = pd.DataFrame(latex_rows)
df_latex.columns = ['Model','Pavia U (OA%)','Indian Pines (OA%)','Salinas (OA%)']
latex_str = df_latex.to_latex(index=False,
    caption='Cross-dataset OA (\\%) comparison. Mean±std over 3 seeds for QHSA-Net, SSRN, 3D-CNN-Only; single seed for others.',
    label='tab:cross_dataset')

with open(os.path.join(WORKDIR, 'paper_latex_table.tex'), 'w') as f:
    f.write(latex_str)
log.info('  LaTeX table saved: paper_latex_table.tex')

log.info('\n' + '='*60)
log.info('  ALL DONE')
log.info('='*60)
log.info('\nOutput files:')
log.info('  paper_eval_results.csv      — all model × dataset × seed results')
log.info('  paper_training_curves.csv   — per-epoch data for stability plots')
log.info('  paper_data_efficiency.csv   — data efficiency experiment')
log.info('  paper_summary_stats.csv     — mean ± std summary table')
log.info('  paper_latex_table.tex       — LaTeX table for paper')
log.info('  fig_paper_cross_dataset_oa.png')
log.info('  fig_paper_multi_seed.png')
log.info('  fig_paper_oa_heatmap.png')
log.info('  fig_paper_data_efficiency.png')
log.info('  fig_paper_training_stability.png')
log.info('  fig_paper_cross_dataset_kappa.png')
log.info('  fig_paper_multi_seed_kappa.png')
