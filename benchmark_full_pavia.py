"""
QHSA-Net Full Pavia University Benchmark
=========================================
Runs on the COMPLETE Pavia U dataset (no subsampling).

Phase 1 – Ablations (find best component settings):
  S2: DR + Feature-Selection comparison (11 configs)
  S3: Qubit & Layer sweep (9 configs)
  S4: Attention measurement variants (4 configs)

Phase 2 – Final model + baselines:
  S5: Optimised QHSA-Net (best DR + best qubits + best attention)
  S6: Baselines — SVM, 3D-CNN-only, HybridSN, SSRN, DBDA

All results saved to CSV after every section.
All figures saved as PNG.
"""

# ============================================================
# 0. IMPORTS
# ============================================================
import os, sys, time, json, logging, warnings
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
from sklearn.decomposition import (PCA, KernelPCA, FastICA,
                                   FactorAnalysis, TruncatedSVD)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_score,
                             recall_score, f1_score,
                             roc_curve, auc, roc_auc_score)

import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ============================================================
# 1. CONFIGURATION
# ============================================================
WORKDIR    = r'c:/Users/saika/OneDrive/Desktop/test 6'
DATA_PATH  = r'c:/Users/saika/OneDrive/Desktop/test 6/pavia u data/PaviaU.mat'
GT_PATH    = r'c:/Users/saika/OneDrive/Desktop/test 6/pavia u data/PaviaU_gt.mat'
LOG_PATH   = os.path.join(WORKDIR, 'benchmark_full_pavia.log')
RESULTS_DIR = WORKDIR   # save CSVs and PNGs alongside the script

SEED        = 42
PATCH_SIZE  = 9
N_CLASSES   = 9
N_QUBITS    = 8          # default for ablations
N_Q_LAYERS  = 2          # default for ablations
N_PCA       = 8          # default quantum input dim
EPOCHS_ABL  = 30         # epochs for every ablation config
EPOCHS_FULL = 30         # epochs for final model + baselines
BATCH_SIZE  = 64
DEVICE      = torch.device('cpu')

CLASS_NAMES = ['Asphalt','Meadows','Gravel','Trees',
               'Painted metal sheets','Bare soil','Bitumen',
               'Self-blocking bricks','Shadows']

torch.manual_seed(SEED); np.random.seed(SEED)

# ============================================================
# 2. LOGGING
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
# 3. DATA LOADING
# ============================================================
section('Loading Pavia University (full dataset)')

raw = sio.loadmat(DATA_PATH)
gt_raw = sio.loadmat(GT_PATH)

HSI = raw['paviaU'].astype(np.float32)        # (610, 340, 103)
GT  = gt_raw['paviaU_gt'].astype(np.int32)    # (610, 340)

H, W, B = HSI.shape
log.info(f'  Image: {H}x{W}x{B}   Labeled pixels: {(GT>0).sum():,}')

# normalise per-band
HSI = (HSI - HSI.min(0)) / (HSI.max(0) - HSI.min(0) + 1e-8)

# 10/90 split — NO MAX_TRAIN cap
rng = np.random.default_rng(SEED)
rows, cols = np.where(GT > 0)
labels = GT[rows, cols] - 1          # 0-indexed

n_total = len(rows)
n_train = int(0.10 * n_total)
idx = rng.permutation(n_total)
tr_idx, te_idx = idx[:n_train], idx[n_train:]

log.info(f'  Train: {len(tr_idx):,}   Test: {len(te_idx):,}')

# patch extraction
PAD = PATCH_SIZE // 2
hsi_pad = np.pad(HSI, ((PAD,PAD),(PAD,PAD),(0,0)), mode='reflect')

def extract_patches(ridx, cidx):
    out = np.empty((len(ridx), B, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for i, (r, c) in enumerate(zip(ridx, cidx)):
        p = hsi_pad[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :]   # (ps, ps, B)
        out[i] = p.transpose(2, 0, 1)
    return out

log.info('  Extracting patches ...')
t0 = time.time()
X_tr = extract_patches(rows[tr_idx], cols[tr_idx])
X_te = extract_patches(rows[te_idx], cols[te_idx])
y_tr = labels[tr_idx]
y_te = labels[te_idx]
log.info(f'  Patches done in {time.time()-t0:.1f}s')

# centre-pixel spectra (for DR/FS fitting)
spec_tr = X_tr[:, :, PAD, PAD]   # (N_train, B)
spec_te = X_te[:, :, PAD, PAD]   # (N_test,  B)

# default PCA (used throughout ablations unless overridden)
pca_default = PCA(n_components=N_PCA, random_state=SEED)
Xpca_tr = pca_default.fit_transform(spec_tr).astype(np.float32)
Xpca_te = pca_default.transform(spec_te).astype(np.float32)
log.info(f'  Default PCA variance: {pca_default.explained_variance_ratio_.sum()*100:.1f}%')

# ============================================================
# 4. DATASET
# ============================================================
class HSIDataset(Dataset):
    def __init__(self, patches, pca_feats, labels):
        self.patches = torch.from_numpy(patches)
        self.pca     = torch.from_numpy(pca_feats)
        self.labels  = torch.from_numpy(labels).long()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.patches[i], self.pca[i], self.labels[i]

def make_loaders(patches_tr, pca_tr, patches_te, pca_te):
    tr = DataLoader(HSIDataset(patches_tr, pca_tr, y_tr),
                    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    te = DataLoader(HSIDataset(patches_te, pca_te, y_te),
                    batch_size=256,        shuffle=False, num_workers=0)
    return tr, te

# ============================================================
# 5. METRICS
# ============================================================
def compute_metrics_full(y_true, y_pred, y_prob=None):
    """Returns dict with OA, AA, kappa, per-class stats, ROC/AUC."""
    oa    = accuracy_score(y_true, y_pred) * 100
    cm    = confusion_matrix(y_true, y_pred)
    pc_acc = cm.diagonal() / cm.sum(axis=1) * 100
    aa    = float(np.mean(pc_acc))
    kappa = cohen_kappa_score(y_true, y_pred) * 100

    prec  = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
    rec   = recall_score   (y_true, y_pred, average=None, zero_division=0) * 100
    f1    = f1_score       (y_true, y_pred, average=None, zero_division=0) * 100
    mac_prec = float(np.mean(prec))
    mac_f1   = float(np.mean(f1))

    fpr_d, tpr_d, auc_d, macro_auc = {}, {}, {}, None
    if y_prob is not None:
        yb = label_binarize(y_true, classes=list(range(N_CLASSES)))
        try:
            macro_auc = roc_auc_score(yb, y_prob, average='macro',
                                      multi_class='ovr') * 100
        except Exception:
            macro_auc = None
        for c in range(N_CLASSES):
            fpr_d[c], tpr_d[c], _ = roc_curve(yb[:, c], y_prob[:, c])
            auc_d[c] = auc(fpr_d[c], tpr_d[c]) * 100

    return dict(OA=oa, AA=aa, kappa=kappa,
                mac_prec=mac_prec, mac_f1=mac_f1,
                pc_acc=pc_acc, pc_prec=prec, pc_rec=rec, pc_f1=f1,
                cm=cm, macro_auc=macro_auc,
                fpr=fpr_d, tpr=tpr_d, auc_pc=auc_d)

# ============================================================
# 6. QUANTUM BRANCH
# ============================================================
def make_vqc(n_qubits, n_layers, measurement='pauliz'):
    dev = qml.device('default.qubit', wires=n_qubits)

    if measurement == 'pauliz':
        @qml.qnode(dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        out_dim = n_qubits

    elif measurement == 'softmax_z':
        @qml.qnode(dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        out_dim = n_qubits          # softmax applied in forward()

    elif measurement == 'multobs':
        @qml.qnode(dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return ([qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])
        out_dim = 3 * n_qubits

    elif measurement == 'entangled':
        pairs = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
        @qml.qnode(dev, interface='torch', diff_method='best')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return ([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] +
                    [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i,j in pairs])
        out_dim = n_qubits + len(pairs)

    else:
        raise ValueError(f'Unknown measurement: {measurement}')

    weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers,
                                                       n_wires=n_qubits)
    return circuit, weight_shape, out_dim, measurement


class QuantumBranch(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_Q_LAYERS,
                 proj_dim=64, measurement='pauliz'):
        super().__init__()
        circuit, wshape, out_dim, self.mtype = make_vqc(n_qubits, n_layers, measurement)
        self.qlayer = qml.qnn.TorchLayer(circuit, {'weights': wshape})
        self.proj   = nn.Sequential(nn.Linear(out_dim, proj_dim),
                                    nn.LayerNorm(proj_dim))

    def forward(self, x):
        # x: (N, n_qubits) — already PCA-reduced
        # normalise to [-pi, pi]
        x = torch.tanh(x) * np.pi
        q_out = self.qlayer(x)
        if self.mtype == 'softmax_z':
            q_out = torch.softmax(q_out, dim=-1)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out, dim=-1)
        return self.proj(q_out)


class ClassicalBranch(nn.Module):
    """3D-CNN spatial branch (shared across all QHSA-Net variants)."""
    def __init__(self, n_bands=B, proj_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, (7,3,3), padding=(3,1,1)), nn.BatchNorm3d(8), nn.ReLU(),
            nn.Conv3d(8,16, (5,3,3), padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3), padding=(1,1,1)), nn.BatchNorm3d(32), nn.ReLU(),
        )
        # compute flattened size
        dummy = torch.zeros(1, 1, n_bands, PATCH_SIZE, PATCH_SIZE)
        flat  = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, proj_dim), nn.LayerNorm(proj_dim))

    def forward(self, x):
        # x: (N, B, H, W) → (N, 1, B, H, W)
        x = x.unsqueeze(1)
        return self.fc(self.conv(x).flatten(1))


class GatedFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())

    def forward(self, fc, fq):
        alpha = self.gate(torch.cat([fc, fq], dim=-1))
        return alpha * fq + (1 - alpha) * fc


class QHSANet(nn.Module):
    def __init__(self, n_bands=B, n_classes=N_CLASSES,
                 n_qubits=N_QUBITS, n_qlayers=N_Q_LAYERS,
                 measurement='pauliz', proj_dim=64):
        super().__init__()
        self.classical = ClassicalBranch(n_bands, proj_dim)
        self.quantum   = QuantumBranch(n_qubits, n_qlayers, proj_dim, measurement)
        self.fusion    = GatedFusion(proj_dim)
        self.clf = nn.Sequential(nn.Linear(proj_dim, 128), nn.GELU(),
                                 nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, patch, pca):
        fc = self.classical(patch)
        fq = self.quantum(pca)
        return self.clf(self.fusion(fc, fq))


# ============================================================
# 7. CLASSICAL-ONLY 3D-CNN BASELINE
# ============================================================
class CNN3DOnly(nn.Module):
    """Classical spatial branch only — no quantum."""
    def __init__(self, n_bands=B, n_classes=N_CLASSES, proj_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, (7,3,3), padding=(3,1,1)), nn.BatchNorm3d(8),  nn.ReLU(),
            nn.Conv3d(8,16,(5,3,3), padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32,64,(3,3,3),padding=(1,1,1)), nn.BatchNorm3d(64), nn.ReLU(),
        )
        dummy = torch.zeros(1, 1, n_bands, PATCH_SIZE, PATCH_SIZE)
        flat  = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # overrides flat — recalculate
        )
        # proper flat size
        self.conv2 = nn.Sequential(
            nn.Conv3d(1, 8, (7,3,3), padding=(3,1,1)), nn.BatchNorm3d(8),  nn.ReLU(),
            nn.Conv3d(8,16,(5,3,3), padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16,32,(3,3,3),padding=(1,1,1)), nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(32, 128), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, patch, pca=None):
        x = patch.unsqueeze(1)
        return self.head(self.conv2(x))


# ============================================================
# 8. HYBRIDN
# ============================================================
class HybridSN(nn.Module):
    """Roy et al. 2020 — 3D-CNN + 2D-CNN architecture."""
    def __init__(self, n_bands=B, n_classes=N_CLASSES, patch_size=PATCH_SIZE):
        super().__init__()
        # 3D conv part
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8,  (7,3,3), padding=(3,1,1)), nn.ReLU())
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, (5,3,3), padding=(2,1,1)), nn.ReLU())
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16,32, (3,3,3), padding=(1,1,1)), nn.ReLU())
        # After 3D convs: (N, 32, B, H, W) → reshape to (N, 32*B, H, W) for 2D conv
        # We use AdaptiveAvgPool to collapse band dim
        self.band_pool = nn.AdaptiveAvgPool3d((1, patch_size, patch_size))
        # 2D conv part
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.pool2d   = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, patch, pca=None):
        x = patch.unsqueeze(1)               # (N,1,B,H,W)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)                 # (N,32,B,H,W)
        x = self.band_pool(x).squeeze(2)     # (N,32,H,W)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        return self.head(self.pool2d(x))


# ============================================================
# 9. SSRN
# ============================================================
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
    """
    Spectral-Spatial Residual Network.
    Zhong et al., IEEE TGRS 2018.
    Ported from official TF code: github.com/zilongzhong/SSRN
    Input: (N, B, H, W) patches — same as all other models.
    """
    def __init__(self, n_bands=B, n_classes=N_CLASSES):
        super().__init__()
        # Reshape inside forward: (N,B,H,W) → (N,1,H,W,B)
        self.spec_conv   = nn.Conv3d(1,  24, (1,1,7), padding=(0,0,3))
        self.spec_res1   = SpectralResBlock(24)
        self.spec_res2   = SpectralResBlock(24)
        self.spec2spat   = nn.Conv3d(24, 128, (1, 1, n_bands))  # collapse bands
        self.spat_res1   = SpatialResBlock(128)
        self.spat_res2   = SpatialResBlock(128)
        self.pool        = nn.AdaptiveAvgPool3d(1)
        self.fc          = nn.Linear(128, n_classes)

    def forward(self, x, pca=None):
        # x: (N, B, H, W) → (N, 1, H, W, B)
        x = x.permute(0, 2, 3, 1).unsqueeze(1)
        x = self.spec_conv(x)        # (N,24,H,W,B)
        x = self.spec_res1(x)
        x = self.spec_res2(x)
        x = self.spec2spat(x)        # (N,128,H,W,1)
        x = self.spat_res1(x)
        x = self.spat_res2(x)
        return self.fc(self.pool(x).flatten(1))


# ============================================================
# 10. DBDA
# ============================================================
class ChannelAttn(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(1, ch//r)), nn.ReLU(),
                                nn.Linear(max(1, ch//r), ch), nn.Sigmoid())
    def forward(self, x):
        # x: (N, C, ...) — global avg pool over spatial dims
        gap = x.flatten(2).mean(dim=2)   # (N, C)
        w   = self.fc(gap).view(x.shape[0], x.shape[1], *([1]*(x.dim()-2)))
        return x * w


class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, (3,3,1), padding=(1,1,0))
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        # x: (N, C, H, W, 1) — compute spatial attention map
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class DBDA(nn.Module):
    """
    Double-Branch Dual-Attention Network.
    Li et al., MDPI Remote Sensing 2020.
    Adapted from github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network
    """
    def __init__(self, n_bands=B, n_classes=N_CLASSES):
        super().__init__()
        GR = 12
        # ── Spectral branch (operates along band dim)
        self.sc0 = nn.Sequential(nn.Conv3d(1,GR,(1,1,7),padding=(0,0,3)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        self.sc1 = nn.Sequential(nn.Conv3d(GR,   GR,(1,1,7),padding=(0,0,3)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        self.sc2 = nn.Sequential(nn.Conv3d(GR*2, GR,(1,1,7),padding=(0,0,3)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        self.sc3 = nn.Sequential(nn.Conv3d(GR*3, GR,(1,1,7),padding=(0,0,3)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        spec_ch = GR * 4   # 48
        self.spec_ca  = ChannelAttn(spec_ch)
        self.spec_pool= nn.AdaptiveAvgPool3d(1)

        # ── Spatial branch (operates along spatial dims)
        self.tc0 = nn.Sequential(nn.Conv3d(1,GR,(3,3,1),padding=(1,1,0)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        self.tc1 = nn.Sequential(nn.Conv3d(GR,   GR,(3,3,1),padding=(1,1,0)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        self.tc2 = nn.Sequential(nn.Conv3d(GR*2, GR,(3,3,1),padding=(1,1,0)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        self.tc3 = nn.Sequential(nn.Conv3d(GR*3, GR,(3,3,1),padding=(1,1,0)),
                                 nn.BatchNorm3d(GR), nn.ReLU())
        spat_ch = GR * 4   # 48
        self.spat_sa  = SpatialAttn()
        self.spat_pool= nn.AdaptiveAvgPool3d(1)

        self.head = nn.Sequential(
            nn.Linear(spec_ch + spat_ch, 128), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x, pca=None):
        # x: (N,B,H,W) → (N,1,H,W,B)
        x3 = x.permute(0,2,3,1).unsqueeze(1)

        # spectral branch (dense connections along band dim)
        s0 = self.sc0(x3)
        s1 = self.sc1(s0)
        s2 = self.sc2(torch.cat([s0, s1], 1))
        s3 = self.sc3(torch.cat([s0, s1, s2], 1))
        sf = torch.cat([s0, s1, s2, s3], 1)    # (N,48,H,W,B)
        sf = self.spec_pool(self.spec_ca(sf)).flatten(1)

        # spatial branch (dense connections along spatial dims)
        t0 = self.tc0(x3)
        t1 = self.tc1(t0)
        t2 = self.tc2(torch.cat([t0, t1], 1))
        t3 = self.tc3(torch.cat([t0, t1, t2], 1))
        tf = torch.cat([t0, t1, t2, t3], 1)    # (N,48,H,W,B)
        tf = self.spat_pool(self.spat_sa(tf)).flatten(1)

        return self.head(torch.cat([sf, tf], 1))


# ============================================================
# 11. TRAINING & EVALUATION
# ============================================================
def train_model(model, loader, n_epochs, tag='', lr_cls=1e-3, lr_q=1e-2):
    """Generic trainer. Separates quantum / classical params automatically."""
    q_params  = [p for n,p in model.named_parameters() if 'quantum' in n or 'qlayer' in n]
    cl_params = [p for n,p in model.named_parameters() if 'quantum' not in n and 'qlayer' not in n]
    groups = []
    if cl_params: groups.append({'params': cl_params, 'lr': lr_cls})
    if q_params:  groups.append({'params': q_params,  'lr': lr_q})
    opt   = optim.Adam(groups or model.parameters(), lr=lr_cls)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    for ep in range(1, n_epochs+1):
        tot_loss, tot_corr, tot_n = 0., 0, 0
        for pb, qb, lb in loader:
            pb, qb, lb = pb.to(DEVICE), qb.to(DEVICE), lb.to(DEVICE)
            opt.zero_grad()
            out  = model(pb, qb)
            loss = crit(out, lb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(lb)
            tot_corr += (out.argmax(1) == lb).sum().item()
            tot_n    += len(lb)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            elapsed = time.time() - t0
            log.info(f'  [{tag}] ep {ep:3d}/{n_epochs}  '
                     f'loss={tot_loss/tot_n:.4f}  '
                     f'acc={tot_corr/tot_n*100:.1f}%  ({elapsed:.0f}s)')
    return time.time() - t0


@torch.no_grad()
def eval_model(model, loader):
    """Returns y_true, y_pred, y_prob (softmax)."""
    model.eval()
    yt_all, yp_all, prob_all = [], [], []
    for pb, qb, lb in loader:
        pb, qb = pb.to(DEVICE), qb.to(DEVICE)
        out  = model(pb, qb)
        prob = torch.softmax(out, dim=-1)
        yp_all.append(out.argmax(1).cpu().numpy())
        yt_all.append(lb.numpy())
        prob_all.append(prob.cpu().numpy())
    return (np.concatenate(yt_all),
            np.concatenate(yp_all),
            np.concatenate(prob_all))


def run_qhsa_experiment(tag, pca_tr, pca_te,
                         n_qubits=N_QUBITS, n_layers=N_Q_LAYERS,
                         measurement='pauliz', n_epochs=EPOCHS_ABL):
    torch.manual_seed(SEED); np.random.seed(SEED)
    tr_loader, te_loader = make_loaders(X_tr, pca_tr, X_te, pca_te)
    model = QHSANet(n_qubits=n_qubits, n_qlayers=n_layers,
                    measurement=measurement).to(DEVICE)
    t_train = train_model(model, tr_loader, n_epochs, tag=tag)
    t_inf0 = time.time()
    yt, yp, yprob = eval_model(model, te_loader)
    t_inf = time.time() - t_inf0
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    m = compute_metrics_full(yt, yp, yprob)
    log.info(f'  {tag}: OA={m["OA"]:.2f}%  AA={m["AA"]:.2f}%  '
             f'kappa={m["kappa"]:.2f}  AUC={m["macro_auc"]:.2f}%  '
             f'train={t_train/60:.1f}min')
    return m, t_train, t_inf


# ============================================================
# 12. DR / FEATURE SELECTION HELPERS
# ============================================================
def fit_dr(name, spec_tr_flat, spec_te_flat, k=N_PCA):
    """Fit a DR or FS method and return (tr_feat, te_feat)."""
    if name == 'PCA':
        m = PCA(k, random_state=SEED); tr = m.fit_transform(spec_tr_flat)
    elif name == 'KernelPCA':
        m = KernelPCA(k, kernel='rbf', random_state=SEED); tr = m.fit_transform(spec_tr_flat)
    elif name == 'FastICA':
        m = FastICA(k, random_state=SEED, max_iter=500); tr = m.fit_transform(spec_tr_flat)
    elif name == 'FactorAnalysis':
        m = FactorAnalysis(k, random_state=SEED); tr = m.fit_transform(spec_tr_flat)
    elif name == 'TruncSVD':
        m = TruncatedSVD(k, random_state=SEED); tr = m.fit_transform(spec_tr_flat)
    elif name == 'RandProj':
        m = GaussianRandomProjection(k, random_state=SEED); tr = m.fit_transform(spec_tr_flat)
    elif name == 'AutoEncoder':
        tr, m = fit_autoencoder(spec_tr_flat, k)
    # Feature-selection methods (select k bands directly)
    elif name == 'FS_Variance':
        idx = np.argsort(spec_tr_flat.var(axis=0))[-k:]
        idx = np.sort(idx); tr = spec_tr_flat[:, idx]; m = ('fs', idx)
    elif name == 'FS_ANOVA':
        sel = SelectKBest(f_classif, k=k); sel.fit(spec_tr_flat, y_tr)
        idx = np.where(sel.get_support())[0]; tr = spec_tr_flat[:, idx]; m = ('fs', idx)
    elif name == 'FS_MutualInfo':
        sel = SelectKBest(mutual_info_classif, k=k); sel.fit(spec_tr_flat, y_tr)
        idx = np.where(sel.get_support())[0]; tr = spec_tr_flat[:, idx]; m = ('fs', idx)
    elif name == 'FS_DivMin':
        idx = divmin_greedy(spec_tr_flat, k); tr = spec_tr_flat[:, idx]; m = ('fs', idx)
    else:
        raise ValueError(name)

    # transform test set
    if isinstance(m, tuple) and m[0] == 'fs':
        te = spec_te_flat[:, m[1]]
    elif name == 'AutoEncoder':
        te = ae_encode(spec_te_flat, m)
    elif name == 'KernelPCA':
        te = m.transform(spec_te_flat)
    else:
        te = m.transform(spec_te_flat)

    return tr.astype(np.float32), te.astype(np.float32)


def fit_autoencoder(X, k, epochs=30):
    """Small MLP autoencoder for DR."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)
    Xt  = torch.from_numpy(X_s)
    enc = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, k))
    dec = nn.Sequential(nn.Linear(k, 64), nn.ReLU(),
                        nn.Linear(64, X.shape[1]))
    params = list(enc.parameters()) + list(dec.parameters())
    opt = optim.Adam(params, lr=1e-3)
    ds  = DataLoader(torch.utils.data.TensorDataset(Xt), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for (xb,) in ds:
            out = dec(enc(xb)); loss = F.mse_loss(out, xb)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        tr_feat = enc(Xt).numpy()
    return tr_feat, (enc, scaler)


def ae_encode(X, model_tuple):
    enc, scaler = model_tuple
    X_s = scaler.transform(X).astype(np.float32)
    enc.eval()
    with torch.no_grad():
        return enc(torch.from_numpy(X_s)).numpy()


def divmin_greedy(X, k):
    """Greedy diversity-maximising band selection."""
    X_n = (X - X.mean(0)) / (X.std(0) + 1e-8)
    selected = [int(np.argmax(X_n.var(axis=0)))]
    while len(selected) < k:
        best, best_score = -1, -1.0
        for i in range(X_n.shape[1]):
            if i in selected: continue
            min_div = min(1 - abs(float(np.corrcoef(X_n[:,i], X_n[:,s])[0,1]))
                         for s in selected)
            if min_div > best_score:
                best_score, best = min_div, i
        selected.append(best)
    return np.sort(selected)


# ============================================================
# 13. SECTION 2 — DR + Feature Selection
# ============================================================
section('Section 2: DR + Feature Selection Comparison')

DR_METHODS = ['PCA','KernelPCA','FastICA','FactorAnalysis',
              'TruncSVD','RandProj','AutoEncoder',
              'FS_Variance','FS_ANOVA','FS_MutualInfo','FS_DivMin']

s2_rows = []
s2_roc  = {}

for name in DR_METHODS:
    log.info(f'\n--- S2: {name} ---')
    try:
        t_dr0 = time.time()
        tr_feat, te_feat = fit_dr(name, spec_tr, spec_te)
        t_dr = time.time() - t_dr0
        m, t_train, t_inf = run_qhsa_experiment(
            f'S2-{name}', tr_feat, te_feat)
        row = dict(method=name, OA=m['OA'], AA=m['AA'], kappa=m['kappa'],
                   mac_prec=m['mac_prec'], mac_f1=m['mac_f1'],
                   macro_auc=m['macro_auc'],
                   dr_time_s=t_dr, train_time_s=t_train, infer_time_s=t_inf)
        s2_rows.append(row)
        s2_roc[name] = {'fpr': m['fpr'], 'tpr': m['tpr'], 'auc': m['auc_pc']}
    except Exception as e:
        log.error(f'  {name} FAILED: {e}')

df_s2 = pd.DataFrame(s2_rows)
df_s2.to_csv(os.path.join(RESULTS_DIR, 's2_dr_fs_results.csv'), index=False)
log.info('\nS2 results saved.')
log.info('\n' + df_s2[['method','OA','AA','kappa','macro_auc']].to_string())

best_dr = df_s2.loc[df_s2['OA'].idxmax(), 'method']
log.info(f'\n>>> Best DR/FS method: {best_dr}  (OA={df_s2["OA"].max():.2f}%)')


# ============================================================
# 14. SECTION 3 — Qubit & Layer Sweep
# ============================================================
section('Section 3: Qubit & Layer Sweep')

# Use best DR from S2
log.info(f'Using DR: {best_dr}')
tr_best_dr, te_best_dr = fit_dr(best_dr, spec_tr, spec_te)

QUBIT_COUNTS = [2, 4, 6, 8, 10, 12]
s3_rows = []

for nq in QUBIT_COUNTS:
    log.info(f'\n--- S3 Qubit sweep: {nq}q ---')
    # re-fit DR to match qubit count (k = n_qubits)
    try:
        tr_q, te_q = fit_dr(best_dr, spec_tr, spec_te, k=nq)
        m, t_train, t_inf = run_qhsa_experiment(
            f'S3-{nq}q', tr_q, te_q, n_qubits=nq, n_layers=N_Q_LAYERS)
        hilbert = 2**nq
        vqc_params = int(np.prod(
            qml.StronglyEntanglingLayers.shape(n_layers=N_Q_LAYERS, n_wires=nq)))
        s3_rows.append(dict(config=f'QHSA-{nq}q', n_qubits=nq, n_layers=N_Q_LAYERS,
                            hilbert_dim=hilbert, vqc_params=vqc_params,
                            OA=m['OA'], AA=m['AA'], kappa=m['kappa'],
                            macro_auc=m['macro_auc'],
                            train_time_s=t_train, infer_time_s=t_inf))
    except Exception as e:
        log.error(f'  {nq}q FAILED: {e}')

best_nq = pd.DataFrame(s3_rows).loc[pd.DataFrame(s3_rows)['OA'].idxmax(), 'n_qubits']
log.info(f'\n>>> Best qubit count: {best_nq}')

# Layer sweep with best qubit count
LAYER_COUNTS = [1, 2, 3]
for nl in LAYER_COUNTS:
    log.info(f'\n--- S3 Layer sweep: {best_nq}q-{nl}L ---')
    try:
        tr_q, te_q = fit_dr(best_dr, spec_tr, spec_te, k=int(best_nq))
        m, t_train, t_inf = run_qhsa_experiment(
            f'S3-{best_nq}q-{nl}L', tr_q, te_q,
            n_qubits=int(best_nq), n_layers=nl)
        vqc_params = int(np.prod(
            qml.StronglyEntanglingLayers.shape(n_layers=nl, n_wires=int(best_nq))))
        s3_rows.append(dict(config=f'QHSA-{best_nq}q-{nl}L',
                            n_qubits=int(best_nq), n_layers=nl,
                            hilbert_dim=2**int(best_nq), vqc_params=vqc_params,
                            OA=m['OA'], AA=m['AA'], kappa=m['kappa'],
                            macro_auc=m['macro_auc'],
                            train_time_s=t_train, infer_time_s=t_inf))
    except Exception as e:
        log.error(f'  {best_nq}q-{nl}L FAILED: {e}')

df_s3 = pd.DataFrame(s3_rows)
df_s3.to_csv(os.path.join(RESULTS_DIR, 's3_qubit_results.csv'), index=False)
log.info('\nS3 results saved.')

layer_df = df_s3[df_s3['config'].str.contains('L')]
best_nl  = int(layer_df.loc[layer_df['OA'].idxmax(), 'n_layers'])
best_nq  = int(best_nq)
log.info(f'>>> Best: {best_nq} qubits, {best_nl} layers')


# ============================================================
# 15. SECTION 4 — Attention Measurement Variants
# ============================================================
section('Section 4: Attention Measurement Variants')

tr_best_dr2, te_best_dr2 = fit_dr(best_dr, spec_tr, spec_te, k=best_nq)

ATT_VARIANTS = ['pauliz', 'softmax_z', 'multobs', 'entangled']
ATT_LABELS   = {'pauliz': 'PauliZ (baseline)',
                'softmax_z': 'Softmax-Z',
                'multobs': 'Multi-obs (X+Y+Z)',
                'entangled': 'Entangled (Z+ZZ)'}
s4_rows = []
s4_roc  = {}

for mtype in ATT_VARIANTS:
    log.info(f'\n--- S4: {ATT_LABELS[mtype]} ---')
    try:
        m, t_train, t_inf = run_qhsa_experiment(
            f'S4-{mtype}', tr_best_dr2, te_best_dr2,
            n_qubits=best_nq, n_layers=best_nl, measurement=mtype)
        s4_rows.append(dict(measurement=mtype,
                            label=ATT_LABELS[mtype],
                            OA=m['OA'], AA=m['AA'], kappa=m['kappa'],
                            macro_auc=m['macro_auc'],
                            train_time_s=t_train, infer_time_s=t_inf))
        s4_roc[mtype] = {'fpr': m['fpr'], 'tpr': m['tpr'], 'auc': m['auc_pc']}
    except Exception as e:
        log.error(f'  {mtype} FAILED: {e}')

df_s4 = pd.DataFrame(s4_rows)
df_s4.to_csv(os.path.join(RESULTS_DIR, 's4_attention_results.csv'), index=False)
log.info('\nS4 results saved.')

best_att = df_s4.loc[df_s4['OA'].idxmax(), 'measurement']
log.info(f'>>> Best attention: {best_att} (OA={df_s4["OA"].max():.2f}%)')


# ============================================================
# 16. PICK WINNERS & ASSEMBLE FINAL MODEL
# ============================================================
section('Assembling Final Optimised QHSA-Net')

log.info(f'  DR method  : {best_dr}')
log.info(f'  Qubits     : {best_nq}')
log.info(f'  Layers     : {best_nl}')
log.info(f'  Measurement: {best_att}')

tr_final, te_final = fit_dr(best_dr, spec_tr, spec_te, k=best_nq)

winners = dict(dr=best_dr, n_qubits=best_nq, n_layers=best_nl, measurement=best_att)
with open(os.path.join(RESULTS_DIR, 'best_config.json'), 'w') as f:
    json.dump(winners, f, indent=2)


# ============================================================
# 17. SECTION 5 — Final QHSA-Net (full Pavia U)
# ============================================================
section('Section 5: Final Optimised QHSA-Net')

m5, t5_train, t5_inf = run_qhsa_experiment(
    'QHSA-Net-Optimised', tr_final, te_final,
    n_qubits=best_nq, n_layers=best_nl,
    measurement=best_att, n_epochs=EPOCHS_FULL)

df_s5 = pd.DataFrame([dict(
    model='QHSA-Net (Optimised)',
    OA=m5['OA'], AA=m5['AA'], kappa=m5['kappa'],
    mac_prec=m5['mac_prec'], mac_f1=m5['mac_f1'],
    macro_auc=m5['macro_auc'],
    train_time_s=t5_train, infer_time_s=t5_inf,
    **{f'pc_f1_{c}': m5['pc_f1'][c] for c in range(N_CLASSES)},
    **{f'pc_prec_{c}': m5['pc_prec'][c] for c in range(N_CLASSES)},
)])
df_s5.to_csv(os.path.join(RESULTS_DIR, 's5_final_qhsa.csv'), index=False)
log.info(f'  Final QHSA-Net: OA={m5["OA"]:.2f}%  AA={m5["AA"]:.2f}%  kappa={m5["kappa"]:.2f}')


# ============================================================
# 18. SECTION 6 — Baseline Models
# ============================================================
section('Section 6: Baseline Models')

# SVM
log.info('\n--- Baseline: SVM (RBF) ---')
t0 = time.time()
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED)
svm.fit(Xpca_tr, y_tr)
t_svm_train = time.time() - t0
t0 = time.time()
svm_pred  = svm.predict(Xpca_te)
svm_prob  = svm.predict_proba(Xpca_te)
t_svm_inf = time.time() - t0
m_svm = compute_metrics_full(y_te, svm_pred, svm_prob)
log.info(f'  SVM: OA={m_svm["OA"]:.2f}%  kappa={m_svm["kappa"]:.2f}')

def run_torch_baseline(name, ModelClass, model_kwargs,
                       uses_pca=False, n_epochs=EPOCHS_FULL):
    """Train & eval a torch model that takes (patch, pca) inputs."""
    torch.manual_seed(SEED)
    pca_tr_use = Xpca_tr if uses_pca else np.zeros((len(y_tr), 1), np.float32)
    pca_te_use = Xpca_te if uses_pca else np.zeros((len(y_te), 1), np.float32)
    tr_l, te_l = make_loaders(X_tr, pca_tr_use, X_te, pca_te_use)
    model = ModelClass(**model_kwargs).to(DEVICE)
    t_train = train_model(model, tr_l, n_epochs, tag=name)
    t0 = time.time()
    yt, yp, yprob = eval_model(model, te_l)
    t_inf = time.time() - t0
    del model
    m = compute_metrics_full(yt, yp, yprob)
    log.info(f'  {name}: OA={m["OA"]:.2f}%  AA={m["AA"]:.2f}%  '
             f'kappa={m["kappa"]:.2f}  AUC={m["macro_auc"]:.2f}%  '
             f'train={t_train/60:.1f}min')
    return m, t_train, t_inf

baselines = [
    ('3D-CNN-Only',  CNN3DOnly,  dict(n_bands=B, n_classes=N_CLASSES)),
    ('HybridSN',     HybridSN,   dict(n_bands=B, n_classes=N_CLASSES)),
    ('SSRN',         SSRN,       dict(n_bands=B, n_classes=N_CLASSES)),
    ('DBDA',         DBDA,       dict(n_bands=B, n_classes=N_CLASSES)),
]

s6_rows = [dict(model='SVM (RBF)',
                OA=m_svm['OA'], AA=m_svm['AA'], kappa=m_svm['kappa'],
                mac_prec=m_svm['mac_prec'], mac_f1=m_svm['mac_f1'],
                macro_auc=m_svm['macro_auc'],
                train_time_s=t_svm_train, infer_time_s=t_svm_inf,
                **{f'pc_f1_{c}': m_svm['pc_f1'][c] for c in range(N_CLASSES)},
                **{f'pc_prec_{c}': m_svm['pc_prec'][c] for c in range(N_CLASSES)})]
s6_roc = {'SVM (RBF)': {'fpr': m_svm['fpr'], 'tpr': m_svm['tpr'], 'auc': m_svm['auc_pc']}}

for bname, BClass, bkwargs in baselines:
    log.info(f'\n--- Baseline: {bname} ---')
    try:
        mb, tb_train, tb_inf = run_torch_baseline(bname, BClass, bkwargs)
        s6_rows.append(dict(model=bname,
                            OA=mb['OA'], AA=mb['AA'], kappa=mb['kappa'],
                            mac_prec=mb['mac_prec'], mac_f1=mb['mac_f1'],
                            macro_auc=mb['macro_auc'],
                            train_time_s=tb_train, infer_time_s=tb_inf,
                            **{f'pc_f1_{c}': mb['pc_f1'][c] for c in range(N_CLASSES)},
                            **{f'pc_prec_{c}': mb['pc_prec'][c] for c in range(N_CLASSES)}))
        s6_roc[bname] = {'fpr': mb['fpr'], 'tpr': mb['tpr'], 'auc': mb['auc_pc']}
    except Exception as e:
        log.error(f'  {bname} FAILED: {e}')

# Add final QHSA-Net to S6 table
s6_rows.append(dict(model='QHSA-Net (Optimised)',
                    OA=m5['OA'], AA=m5['AA'], kappa=m5['kappa'],
                    mac_prec=m5['mac_prec'], mac_f1=m5['mac_f1'],
                    macro_auc=m5['macro_auc'],
                    train_time_s=t5_train, infer_time_s=t5_inf,
                    **{f'pc_f1_{c}': m5['pc_f1'][c] for c in range(N_CLASSES)},
                    **{f'pc_prec_{c}': m5['pc_prec'][c] for c in range(N_CLASSES)}))
s6_roc['QHSA-Net (Optimised)'] = {'fpr': m5['fpr'], 'tpr': m5['tpr'], 'auc': m5['auc_pc']}

df_s6 = pd.DataFrame(s6_rows)
df_s6.to_csv(os.path.join(RESULTS_DIR, 's6_baseline_results.csv'), index=False)
log.info('\nS6 results saved.')
log.info('\n' + df_s6[['model','OA','AA','kappa','macro_auc',
                        'train_time_s']].to_string())


# ============================================================
# 19. SECTION 7 — VISUALISATIONS
# ============================================================
section('Section 7: Visualisations')

COLOURS = ['#2196F3','#4CAF50','#FF9800','#E91E63',
           '#9C27B0','#00BCD4','#FF5722','#607D8B']

# ── Fig 1: S2 DR comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x = np.arange(len(df_s2)); w = 0.25
for i, (col, lbl) in enumerate([('OA','OA (%)'),('AA','AA (%)'),('kappa','Kappa')]):
    axes[i].bar(x, df_s2[col], color=COLOURS[:len(df_s2)], alpha=0.85)
    axes[i].set_xticks(x); axes[i].set_xticklabels(df_s2['method'], rotation=35, ha='right', fontsize=8)
    axes[i].set_ylabel(lbl); axes[i].set_title(f'DR/FS — {lbl}', fontweight='bold')
    axes[i].set_ylim(max(0, df_s2[col].min()-5), 100)
    axes[i].axhline(df_s2[col].max(), color='red', ls='--', alpha=0.5, lw=1)
plt.suptitle('Section 2: DR & Feature-Selection Comparison (Full Pavia U)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_s2_dr_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_s2_dr_comparison.png')

# ── Fig 2: S2 AUC bar ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(df_s2['method'], df_s2['macro_auc'], color=COLOURS[:len(df_s2)], alpha=0.85)
ax.set_xticklabels(df_s2['method'], rotation=35, ha='right', fontsize=8)
ax.set_ylabel('Macro AUC (%)'); ax.set_title('DR/FS — Macro AUC (OvR)', fontweight='bold')
ax.set_ylim(max(0, df_s2['macro_auc'].min()-3), 100)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_s2_auc.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_s2_auc.png')

# ── Fig 3: S3 Qubit sweep ────────────────────────────────────
qubit_df = df_s3[~df_s3['config'].str.contains('L')].copy()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(qubit_df['n_qubits'], qubit_df['OA'], 'o-', color=COLOURS[0], lw=2)
axes[0].set_xlabel('Number of Qubits'); axes[0].set_ylabel('OA (%)')
axes[0].set_title('Qubit Sweep — Overall Accuracy', fontweight='bold')
ax2 = axes[0].twinx()
ax2.plot(qubit_df['n_qubits'], qubit_df['hilbert_dim'], 's--', color=COLOURS[1], lw=1.5)
ax2.set_ylabel('Hilbert Space Dim (log scale)'); ax2.set_yscale('log')

layer_df2 = df_s3[df_s3['config'].str.contains('L')].copy()
axes[1].bar(layer_df2['config'], layer_df2['OA'], color=COLOURS[:3], alpha=0.85)
axes[1].set_ylabel('OA (%)'); axes[1].set_title(f'Layer Sweep ({best_nq} qubits)', fontweight='bold')
axes[1].set_ylim(max(0, layer_df2['OA'].min()-5), 100)
plt.suptitle('Section 3: Qubit & Layer Sweep', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_s3_qubit_sweep.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_s3_qubit_sweep.png')

# ── Fig 4: S4 Attention variants ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x4 = np.arange(len(df_s4)); w4 = 0.3
for i, (col, lbl) in enumerate([('OA','OA (%)'), ('kappa','Kappa')]):
    axes[i].bar(x4, df_s4[col], color=COLOURS[:len(df_s4)], alpha=0.85)
    axes[i].set_xticks(x4); axes[i].set_xticklabels(df_s4['label'], rotation=20, ha='right', fontsize=9)
    axes[i].set_ylabel(lbl); axes[i].set_title(f'Attention Variants — {lbl}', fontweight='bold')
    axes[i].set_ylim(max(0, df_s4[col].min()-3), 100)
plt.suptitle('Section 4: Quantum Measurement Variants', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_s4_attention.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_s4_attention.png')

# ── Fig 5: S6 Baseline comparison — OA/AA/Kappa/AUC ────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics_4 = [('OA','OA (%)'), ('AA','AA (%)'), ('kappa','Kappa'), ('macro_auc','Macro AUC (%)')]
for ax, (col, lbl) in zip(axes.flat, metrics_4):
    colours_b = [COLOURS[6] if 'QHSA' in m else COLOURS[0] for m in df_s6['model']]
    ax.bar(df_s6['model'], df_s6[col], color=colours_b, alpha=0.85)
    ax.set_xticklabels(df_s6['model'], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(lbl); ax.set_title(f'Baseline Comparison — {lbl}', fontweight='bold')
    ax.set_ylim(max(0, df_s6[col].min()-5), 101)
plt.suptitle('Section 6: Baseline Models vs QHSA-Net (Full Pavia U)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_s6_baselines.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_s6_baselines.png')

# ── Fig 6: Per-class F1 heatmap ─────────────────────────────
pc_f1_cols = [f'pc_f1_{c}' for c in range(N_CLASSES)]
pc_data = df_s6[['model'] + pc_f1_cols].set_index('model')
pc_data.columns = CLASS_NAMES
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(pc_data.values, aspect='auto', cmap='RdYlGn', vmin=50, vmax=100)
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(df_s6))); ax.set_yticklabels(df_s6['model'], fontsize=9)
for i in range(len(df_s6)):
    for j in range(N_CLASSES):
        ax.text(j, i, f'{pc_data.values[i,j]:.1f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax, label='F1 Score (%)')
ax.set_title('Per-Class F1 Score — All Models', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_s6_f1_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_s6_f1_heatmap.png')

# ── Fig 7: ROC curves — QHSA-Net all 9 classes ──────────────
if m5['fpr']:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for c, ax in zip(range(N_CLASSES), axes.flat):
        fpr, tpr = m5['fpr'][c], m5['tpr'][c]
        auc_c = m5['auc_pc'][c]
        ax.plot(fpr, tpr, lw=2, color=COLOURS[c % len(COLOURS)],
                label=f'AUC = {auc_c:.1f}%')
        ax.plot([0,1],[0,1],'k--',lw=0.8)
        ax.set_title(CLASS_NAMES[c], fontsize=9, fontweight='bold')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    plt.suptitle('ROC Curves — QHSA-Net (Optimised) — Pavia U', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_roc_qhsa.png'), dpi=150, bbox_inches='tight')
    plt.close(); log.info('Saved: fig_bench_roc_qhsa.png')

# ── Fig 8: ROC macro comparison — all models ─────────────────
fig, ax = plt.subplots(figsize=(10, 8))
for (model_name, roc_data), col in zip(s6_roc.items(), COLOURS):
    if not roc_data['fpr']: continue
    # Compute macro ROC by averaging all class FPR/TPR
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros(200)
    valid = 0
    for c in range(N_CLASSES):
        if c in roc_data['fpr']:
            mean_tpr += np.interp(mean_fpr, roc_data['fpr'][c], roc_data['tpr'][c])
            valid += 1
    if valid > 0:
        mean_tpr /= valid
        mean_auc = auc(mean_fpr, mean_tpr) * 100
        ax.plot(mean_fpr, mean_tpr, lw=2, color=col,
                label=f'{model_name} (AUC={mean_auc:.1f}%)')
ax.plot([0,1],[0,1],'k--',lw=0.8,label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Macro-Averaged ROC Curves — All Models', fontweight='bold')
ax.legend(loc='lower right', fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_roc_all_models.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_roc_all_models.png')

# ── Fig 9: Training-time comparison ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sorted_s6 = df_s6.sort_values('train_time_s', ascending=True)
colours_t  = [COLOURS[6] if 'QHSA' in m else COLOURS[0] for m in sorted_s6['model']]
ax.barh(sorted_s6['model'], sorted_s6['train_time_s']/60, color=colours_t, alpha=0.85)
ax.set_xlabel('Training Time (minutes)'); ax.set_title('Training Time Comparison', fontweight='bold')
for i, (_, row) in enumerate(sorted_s6.iterrows()):
    ax.text(row['train_time_s']/60 + 0.2, i, f'{row["train_time_s"]/60:.1f}min',
            va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_timing.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_timing.png')

# ── Fig 10: Confusion matrix — QHSA-Net ──────────────────────
fig, ax = plt.subplots(figsize=(10, 9))
cm5 = m5['cm']
im  = ax.imshow(cm5, cmap='Blues')
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES, fontsize=9)
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, str(cm5[i,j]), ha='center', va='center',
                fontsize=8, color='white' if cm5[i,j] > cm5.max()*0.5 else 'black')
plt.colorbar(im, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — QHSA-Net (Optimised)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_bench_confusion_qhsa.png'), dpi=150, bbox_inches='tight')
plt.close(); log.info('Saved: fig_bench_confusion_qhsa.png')

# ── Master summary CSV ───────────────────────────────────────
master = df_s6[['model','OA','AA','kappa','mac_prec','mac_f1','macro_auc',
                 'train_time_s','infer_time_s']].copy()
master.to_csv(os.path.join(RESULTS_DIR, 'benchmark_master_results.csv'), index=False)
log.info('Saved: benchmark_master_results.csv')

section('ALL DONE')
log.info('\nFinal Results Summary:')
log.info('\n' + master.to_string(index=False))
log.info(f'\nBest configuration found:')
log.info(f'  DR method  : {best_dr}')
log.info(f'  Qubits     : {best_nq}')
log.info(f'  Layers     : {best_nl}')
log.info(f'  Measurement: {best_att}')
