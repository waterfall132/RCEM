# RCEM
Official code of the paper *Rejectable Component Evidence Matching for Few-Shot Image Classification*

## File list

```
RCEM_code/
├── README.md                         This document
├── sgd_pair.sh                       Train a pair of models (RCEM or baseline) under the DN4 protocol
├── sgd_eval.sh                       Evaluate all trained models and output paired comparisons
├── summarize_results.py              Summarize all results: accuracy, confidence intervals, paired differences, p-values
├── utils.py                          Utility functions from the original DN4 code (needed by models/classifier.py)
├── Prepare_Cars.py                   Unpack and organize Stanford Cars
├── dataset/
│   ├── general_dataloader.py         DN4 episode data loader (used in training, includes data augmentation)
│   └── Prepare_csv_CubBird_NDCN.py   CUB split and organization
├── models/
│   ├── network.py                    Backbone registry encoder_dict
│   ├── backbone.py                   Original DN4 backbones (Conv-4, ResNet-12, etc.)
│   ├── backbone_relu.py              Version with LeakyReLU replaced by ReLU (non-negative features, required by NDM)
│   ├── classifier.py                 Original DN4 classification head (needed by network.py, not used by RCEM)
│   ├── ndcn.py                       NDM non-negative decomposition module
│   └── ndcn_local.py                 Early local head (needed by network.py, not used by RCEM)
├── routeA/
│   ├── common.py                     Model loading, data splits, feature extraction and caching, fixed episode sampling
│   ├── disc_pool.py                  Per-image component computation, GAP prototype scoring
│   └── train_routeA.py               Baseline training (--gamma 0 is Cosine ProtoNet / GAP + CE)
└── routeB/
    ├── rcem.py                       The RCEM method itself: components, class-contrastive evidence, partial optimal transport, scoring of the two branches
    ├── fusion.py                     Temperature calibration, probability fusion, λ selection
    ├── train_routeB.py               RCEM training
    ├── eval_b.py                     Evaluation: validation-set calibration + 5×2000 test episodes
    ├── compare_pair.py               Paired comparison of two models on identical episodes
    ├── timing.py                     Inference time and GPU memory measurement
    └── selftest_b.py                 Unit self-test on CPU
```

`routeA` and `routeB` are early-project names. They are kept because all scripts and result folders are named after this structure. The paper contains only two models:

- **RCEM**: the model trained by `routeB/train_routeB.py`;
- **Baseline** (GAP + CE / Cosine ProtoNet in the paper): the model trained by `routeA/train_routeA.py --gamma 0`.

---

## 3. Environment

- Python ≥ 3.8, PyTorch ≥ 1.10, torchvision;
- numpy, scipy, matplotlib, pillow. scipy and matplotlib are required by `utils.py`, and the Wilcoxon test in `summarize_results.py` also needs scipy;
- Training requires a GPU; a CPU is only enough to run the self-test.

The verification for this release used PyTorch 2.15 (CPU).

```
pip install torch torchvision numpy scipy matplotlib pillow
```

Run all commands from the code root directory (`RCEM_code/`). The scripts add the root directory to `sys.path` themselves.

---

## 4. Data format and preparation

Each dataset is a folder (called `dataset_dir` below) with the following structure:

```
dataset_dir/
├── images/          All images, flat in this folder, with unique file names
├── train.csv        Header: filename,label
├── val.csv
└── test.csv
```

`train.csv` is used for training, `val.csv` for model selection and temperature calibration, and final results use only `test.csv`. The classes of the three splits do not overlap.

**CUB-200-2011**

```
python dataset/Prepare_csv_CubBird_NDCN.py --data_dir /data/CUB_200_2011 --save_dir /root/data/CUB/For_FewShot
```

- The default split is 100 / 50 / 50 classes (changeable via `--n_train --n_val --n_test`).
- `--bbox` crops images by the annotated bounding box. The experiments in the paper use the existing `/root/data/CUB/For_FewShot` on the server; whether it is cropped is determined by that copy of the data.

**Stanford Cars**

```
python Prepare_Cars.py --archive /root/StanfordCar.tar.gz --raw /root/raw/Cars --out /root/data/Cars
```

This script does only two things: safe extraction, and keeping or restoring the existing few-shot CSVs. It does not guess annotations, re-split randomly, or recompress images.

**Stanford Dogs, miniImageNet**

The archives on the server are already in the format above (`images/` plus three CSVs). Just extract them to the following locations:
- Dogs: `$DATA/Dogs/For_FewShot`
- mini: `$DATA/miniImageNet`

**Preprocessing**

| Stage | Processing |
|---|---|
| Training (`dataset/general_dataloader.py`) | RandomResizedCrop(84), ColorJitter(0.4, 0.4, 0.4), horizontal flip, normalization with (0.5, 0.5, 0.5) |
| Validation and test (`eval_transform` in `routeA/common.py`) | Resize(92), CenterCrop(84), same normalization, no randomness |

---

## 5. Correspondence between method and code

The notation below follows the paper. B is the batch size, L = H×W is the number of feature-map positions, C is the number of channels, and r is the number of components.

### 5.1 Backbone and non-negative features

`models/backbone_relu.py`:
- `ResNet12_ReLU`: outputs 640×5×5;
- `Conv64F_Local_ReLU`: outputs 64×21×21.

The last activation of both is ReLU, so the features are non-negative, which is a prerequisite for NDM. The features are flattened into Z ∈ ℝ^{B×L×C} (`flatten` in `routeA/disc_pool.py`), followed by one more ReLU.

### 5.2 NDM component decomposition (`NDM` in `models/ndcn.py`)

Decompose Z into Z ≈ WH, where W ≥ 0 (size L×r) and H ≥ 0 (size r×C):

- **Initialization**: NNDSVD (`_nndsvd_init`), deterministic and independent of the random seed.
  - The first component takes the absolute value of the principal singular vector;
  - For the others, each singular vector contributes whichever of its positive and negative parts has the larger product of norms.
- **Update**: only **one** Lee–Seung multiplicative update (`_mu_step`), i.e. `PAPER_N_ITER = 1`. Gradients flow through this single update.
- **Output**: A = W normalized by row; each position distributes one unit of mass over the r components.
- **SVD implementation**: when the environment variable `NDM_FAST_SVD=1` is set, the top r singular vectors are computed from the smaller Gram matrix with `eigh` (`_topr_svd`). The result agrees with the truncated full SVD and is much faster on the 441×64 feature maps of Conv-4. `sgd_pair.sh` turns it on by default.
- **Error handling**: when the full SVD does not converge, `_robust_svd` falls back to `gesvd`, and then to CPU double-precision computation. If the input contains NaN, it raises an error immediately.

### 5.3 Per-image components (`image_components` in `routeA/disc_pool.py`, and `finalize` in `routeB/rcem.py`)

| Quantity | Definition |
|---|---|
| Component mass | m_v = Σ_l A_lv |
| Component vector | e_v = Σ_l A_lv z_l / m_v, written ê_v after normalization to unit length |
| Global vector | g = mean of Z over all positions (GAP) |
| Activation strength | s_v = Σ_l A_lv ‖z_l‖ / m_v, used only in the `evidence norm` control |

Components whose mass or vector norm is close to 0 are marked invalid (`valid = False`) and do not take part in later computation.

### 5.4 Global branch (`global_scores` in `routeB/rcem.py`)

- The g of each support image is first normalized to unit length;
- Within each class, the vectors are averaged and normalized again to give the class prototype;
- The cosine similarity between the normalized g of the query image and each class prototype gives s^G.

The baseline and the global branch of RCEM use **the same function**.

### 5.5 Class-contrastive evidence (`evidence`, `image_similarity`, `class_means`, `contrast`)

1. **Component similarity between images**: m(e, F) = Σ_u π_u eᵀf_u, where π_u = softmax_u(eᵀf_u / t) and t = 0.1.
2. **Class mean**: R_c(e) = average of m between component e and all support images of class c.
3. **Class-contrastive evidence**: d_v^c = R_c(e_v) − t_n · log[ 1/(N−1) · Σ_{b≠c} exp(R_b(e_v) / t_n) ], with t_n = 0.1.
4. **Support-side evidence** d_s: computed with the support image itself excluded, i.e. R_c^{(−i)}. In 1-shot there is no other image in the same class, so β_s is automatically 0.

Values of `--evidence`:

| Value | Meaning |
|---|---|
| `disc` | The class-contrastive evidence above, i.e. the method of this paper |
| `same` | Uses only R_c, with no contrast |
| `norm` | Uses the activation strength s_v in place of the evidence |
| `shuffle` | Evidence computed after shuffling the class labels (used only in the controls in `eval_b.py`) |
| `none` | No evidence |

### 5.6 Cost and fixed-mass partial optimal transport (`local_scores`, `sinkhorn_partial`, `class_marginals`)

**Cost**:

C_vj^c = 1 − ê_vᵀê_j + β_q · h(d_v^c) + β_s · h(d_j), where h(u) = softplus(−u / t_d) and t_d = 0.1.

The weaker the evidence, the higher the cost, and the easier it is for that component to be rejected.

**Marginals**:
- Query side: each valid component has a_v = 1 / (number of valid components);
- Support side (`class_marginals`): each support image of class c has capacity 1/K, which is split evenly among its valid components.

**Partial transport**: only a mass of μ is transported in total, and the remaining 1 − μ is rejected. This is implemented by adding one "dummy" row and one "dummy" column to the cost matrix, with marginals [a, 1−μ] and [b, 1−μ] respectively and cost 0. The edge from the dummy row to the dummy column is masked out rather than assigned a large value.

**Solver**: log-domain Sinkhorn with ε = 0.05. 30 iterations in training (`--iters 30`, unrolled so that it is differentiable) and 100 iterations at evaluation.

**Local score**: s_c^L = −⟨T*, C⟩ / ⟨T*, 1⟩, i.e. the negative of the average cost over the transported part.

**Special cases**:
- When μ ≥ 1 it reduces to ordinary full optimal transport, which is the "no rejection" control (`transport='full'`).
- `eval_b.py` also contains three transport variants used only for control experiments:
  - `randdrop`: randomly drops the same proportion of components;
  - `topk`: keeps the top k components by evidence;
  - `indep`: independent coupling, with no optimal transport.

### 5.7 Training objective (`RouteBNet` in `routeB/train_routeB.py`)

- Each of the two branches is first multiplied by a learnable temperature and then passed through softmax: p_G = softmax(τ_G s^G), p_L = softmax(τ_L s^L). τ is parameterized with softplus to keep it positive, and its initial value is given by `--tau_G` and `--tau_L`.
- Fused probability: p_λ = (1 − λ) p_G + λ p_L. During training λ is fixed at `--lam`.
- Loss: L = CE(p_λ) + η_G · CE(p_G) + η_L · CE(p_L). The paper uses η_G = η_L = 0.5 (hard-coded in `sgd_pair.sh`).
- `--local_head dense|dn4` replaces the local branch with per-position prototype classification or DN4 k-nearest neighbours, for the control experiments on the form of local supervision. The default is `rcem`.

### 5.8 Baseline (`routeA/train_routeA.py --gamma 0`)

- Global branch only: normalized GAP, class prototypes, and cosine similarity multiplied by a learnable temperature τ (initial value 10); the loss is cross-entropy.
- When `gamma = 0`, the NDM components are not computed, so the speed is the same as ProtoNet.
- It shares the backbone code, data loader, optimizer settings, episode budget and random-seed procedure with RCEM.
- The random seed of the training episodes is `seed × 1000 + epoch`, so under the same seed the two models see the same training episodes in every epoch.

### 5.9 Test-time calibration and fusion (`routeB/fusion.py`, `routeB/eval_b.py`)

All choices are made using **only the validation set**. These choices are then frozen before the test set is read.

1. **Temperature calibration**: on 1000 fixed validation episodes (seed 1000 + K), for each branch the temperature is chosen from `np.geomspace(1, 300, 41)` to minimize the validation negative log-likelihood. Because the temperature comes from this grid, it is normal for different models to obtain the same temperature value.
2. **1-shot fusion weight λ₀**: chosen from {0, 0.25, 0.5, 0.75, 1} by the highest validation accuracy; ties are broken by the smaller negative log-likelihood. λ₀ = 0 means that only the global branch is used at test time.
3. **5-shot support-set selector**: λ_S is chosen separately for each episode:
   - λ_S = argmin_λ [ R̂_S(λ) + ζ λ² ];
   - R̂_S is the negative log-likelihood obtained by taking each support image in turn as the query (leave-one-out); it uses only the support set and never looks at the query images;
   - The shrinkage coefficient ζ is chosen from {0, 0.02, 0.05, 0.1, 0.2, 0.5} by validation accuracy.
4. **Main reported results**: in 1-shot it is `fixed` (using λ₀), in 5-shot it is `select` (using λ_S).

---

## 6. Training

### 6.1 DN4 training protocol (`sgd_pair.sh`)

| Item | Setting |
|---|---|
| Training volume | 30 epochs, 10,000 episodes per epoch; 5-way, 15 queries per class |
| Optimizer | SGD, momentum 0.9, dampening 0.9, no Nesterov, weight decay 5e-4 |
| Learning rate | 0.01 for ResNet-12, 0.05 for Conv-4; cosine annealing down to 1e-3 of the initial value |
| Gradient clipping | None by default (`CLIP=0`); Conv-4 on Cars and miniImageNet uses `CLIP=5` for both methods |
| Model selection | After each epoch, test on 1000 fixed validation episodes (seed 777) and save the model with the highest fused accuracy as `model_best.pth.tar` |
| Pre-training | None; all models are trained from random initialization |
| Resuming | If the script is interrupted, run the same command again to resume exactly from `model_last.pth.tar`, including the optimizer, learning-rate scheduler and random-number state; models that have already finished are skipped automatically |

**Test protocol**: 5 rounds of 2000 episodes each, with seed `round × 100 + K`, for 10,000 episodes in total. All models use **exactly the same** test episodes, so paired comparisons are possible.

**Three modes**:

| Mode | What it does |
|---|---|
| `--fixed --no_controls` | Uses the training-time settings and evaluates only RCEM itself (main results of the paper) |
| `--refs_only` | Evaluates only the reference results, used for baseline checkpoints |
| Without `--fixed` | Full diagnosis with the backbone frozen: first search μ × β_q × β_s on the validation set, then run three groups of controls (grouping, rejection mode, evidence form) (the screening experiments of phase one) |

Other parameters:
- `--val_episodes` (default 1000), `--test_episodes` (default 2000), `--test_seeds` (default 0–4);
- `--occlude`: occlusion robustness test, in which each image has the specified fraction of its area masked;
- `--cache_on_cpu`: keeps the feature cache in main memory rather than GPU memory;
- `--device`: the device to run on.

**Output**: one subdirectory per checkpoint, named "parent directory name + run name (last few characters of each) + first 8 characters of the checkpoint hash".

| File | Content |
|---|---|
| `feat_<split>_r5_it1_maps.pt` | Feature cache, read directly when evaluation is repeated |
| `results_<K>shot.json` | All choices made on the validation set: `choices[selected]` holds `tau_G`, `tau_L`, `lam_fixed` (i.e. λ₀) and `zeta` (5-shot), plus the `lambda` field (distribution and correlation coefficients of λ_S) |
| `report_<K>shot.md` | Human-readable report |
| `test_<K>shot/B_fixed.npz` | Main RCEM 1-shot result (an `accuracy` array of size 5×2000) |
| `test_<K>shot/B_select.npz` | Main RCEM 5-shot result |
| `test_<K>shot/gap.npz` | In an RCEM directory, the global branch of RCEM; in a baseline directory, the baseline itself |
| `test_<K>shot/<setting>__global/local/fixed/select/oracle/random_gate.npz` | Results of each branch alone, the upper bound of λ (oracle), and the result of choosing λ at random |
| `test_<K>shot/ndcp.npz`, `routeA.npz` | Reference results of early methods, not used in the paper |

6. **Randomness.** Training episodes are determined by `seed × 1000 + epoch`, validation and test episodes by fixed seeds, and the NDM initialization is deterministic. On CPU, the same seed can be reproduced bit for bit; on GPU, cuDNN non-determinism may cause small differences.
