# RCEM
official code of the paper Rejectable Component Evidence Matching for Few-Shot Image Classification

## 文件清单

```
RCEM_code/
├── README.md                         本文档
├── sgd_pair.sh                       按 DN4 协议训练一对模型（RCEM 或基线）
├── sgd_eval.sh                       评测所有训练完的模型，并输出配对比较
├── summarize_results.py              汇总全部结果：准确率、置信区间、配对差值、p 值
├── utils.py                          原 DN4 代码的工具函数（models/classifier.py 需要）
├── Prepare_Cars.py                   Stanford Cars 解压与整理
├── dataset/
│   ├── general_dataloader.py         DN4 的 episode 数据加载器（训练时用，含数据增强）
│   └── Prepare_csv_CubBird_NDCN.py   CUB 划分与整理
├── models/
│   ├── network.py                    骨干网络注册表 encoder_dict
│   ├── backbone.py                   DN4 原版骨干网络（Conv-4、ResNet-12 等）
│   ├── backbone_relu.py              把 LeakyReLU 换成 ReLU 的版本（特征非负，NDM 需要）
│   ├── classifier.py                 DN4 原版分类头（network.py 需要，RCEM 不用）
│   ├── ndcn.py                       NDM 非负分解模块
│   └── ndcn_local.py                 早期局部头（network.py 需要，RCEM 不用）
├── routeA/
│   ├── common.py                     模型加载、数据划分、特征提取与缓存、固定 episode 采样
│   ├── disc_pool.py                  每张图的成分（component）计算、GAP 原型打分
│   └── train_routeA.py               基线训练（--gamma 0 即 Cosine ProtoNet / GAP + CE）
└── routeB/
    ├── rcem.py                       RCEM 方法本体：成分、类对比证据、部分最优传输、两个分支的打分
    ├── fusion.py                     温度标定、概率融合、λ 选择
    ├── train_routeB.py               RCEM 训练
    ├── eval_b.py                     评测：验证集标定 + 5×2000 个测试 episode
    ├── compare_pair.py               两个模型在相同 episode 上的配对比较
    ├── timing.py                     推理耗时与显存测量
    └── selftest_b.py                 CPU 上的单元自检
```

`routeA` 和 `routeB` 是项目早期的命名，保留下来是因为所有脚本和结果文件夹都按这个结构命名。论文里只有两个模型：

- **RCEM**：`routeB/train_routeB.py` 训练出的模型；
- **基线**（论文中的 GAP + CE / Cosine ProtoNet）：`routeA/train_routeA.py --gamma 0` 训练出的模型。

---

## 3. 环境

- Python ≥ 3.8，PyTorch ≥ 1.10，torchvision；
- numpy、scipy、matplotlib、pillow。scipy 和 matplotlib 是 `utils.py` 需要的，`summarize_results.py` 的 Wilcoxon 检验也要 scipy；
- 训练需要 GPU，CPU 只够跑自检。

本次验证用的是 PyTorch 2.15（CPU）。

```
pip install torch torchvision numpy scipy matplotlib pillow
```

所有命令都在代码根目录（`RCEM_code/`）下执行。脚本会自己把根目录加入 `sys.path`。

---

## 4. 数据格式与准备

每个数据集是一个文件夹（下文记为 `dataset_dir`），结构如下：

```
dataset_dir/
├── images/          所有图片平铺在这里，文件名唯一
├── train.csv        表头 filename,label
├── val.csv
└── test.csv
```

训练用 `train.csv`，模型选择和温度标定用 `val.csv`，最终结果只用 `test.csv`。三个划分的类别互不重叠。

**CUB-200-2011**

```
python dataset/Prepare_csv_CubBird_NDCN.py --data_dir /data/CUB_200_2011 --save_dir /root/data/CUB/For_FewShot
```

- 默认划分为 100 / 50 / 50 类（`--n_train --n_val --n_test` 可改）。
- `--bbox` 表示按标注框裁剪。论文实验用的是服务器上已有的 `/root/data/CUB/For_FewShot`，是否裁剪以那份数据为准。

**Stanford Cars**

```
python Prepare_Cars.py --archive /root/StanfordCar.tar.gz --raw /root/raw/Cars --out /root/data/Cars
```

这个脚本只做两件事：安全解压，以及保留或恢复已有的 few-shot CSV。它不猜标注、不重新随机划分、不重新压缩图片。

**Stanford Dogs、miniImageNet**

服务器上的压缩包已经是上面的格式（`images/` 加三个 CSV）。解压到下面的位置即可：
- Dogs：`$DATA/Dogs/For_FewShot`
- mini：`$DATA/miniImageNet`

**预处理**

| 阶段 | 处理 |
|---|---|
| 训练（`dataset/general_dataloader.py`） | RandomResizedCrop(84)、ColorJitter(0.4, 0.4, 0.4)、水平翻转，按 (0.5, 0.5, 0.5) 归一化 |
| 验证、测试（`routeA/common.py` 的 `eval_transform`） | Resize(92)、CenterCrop(84)，同样的归一化，没有随机性 |

---

## 5. 方法与代码的对应关系

下面的符号和论文一致。B 是批大小，L = H×W 是特征图位置数，C 是通道数，r 是成分数。

### 5.1 骨干网络与非负特征

`models/backbone_relu.py`：
- `ResNet12_ReLU`：输出 640×5×5；
- `Conv64F_Local_ReLU`：输出 64×21×21。

两者的最后一层激活都是 ReLU，所以特征非负，这是 NDM 的前提。特征被展平为 Z ∈ ℝ^{B×L×C}（`routeA/disc_pool.py` 的 `flatten`），再取一次 ReLU。

### 5.2 NDM 成分分解（`models/ndcn.py` 的 `NDM`）

把 Z 分解为 Z ≈ WH，其中 W ≥ 0（大小 L×r），H ≥ 0（大小 r×C）：

- **初始化**：NNDSVD（`_nndsvd_init`），确定性，不依赖随机种子。
  - 第一个成分取主奇异向量的绝对值；
  - 其余成分在每个奇异向量的正部和负部中取范数乘积更大的一半。
- **更新**：只做 **1 次** Lee–Seung 乘性更新（`_mu_step`），即 `PAPER_N_ITER = 1`。梯度只穿过这一次更新。
- **输出**：A = W 按行归一化，每个位置把一份质量分给 r 个成分。
- **SVD 实现**：环境变量 `NDM_FAST_SVD=1` 时，改用较小的 Gram 矩阵加 `eigh` 求前 r 个奇异向量（`_topr_svd`），结果与完整 SVD 截断后一致，在 Conv-4 的 441×64 特征图上快很多。`sgd_pair.sh` 默认打开它。
- **出错处理**：完整 SVD 不收敛时，`_robust_svd` 依次退回 `gesvd`，再退回 CPU 双精度计算。输入含 NaN 时直接报错。

### 5.3 每张图的成分（`routeA/disc_pool.py` 的 `image_components`，以及 `routeB/rcem.py` 的 `finalize`）

| 量 | 定义 |
|---|---|
| 成分质量 | m_v = Σ_l A_lv |
| 成分向量 | e_v = Σ_l A_lv z_l / m_v，单位化后记为 ê_v |
| 全局向量 | g = Z 在所有位置上的平均（GAP） |
| 激活强度 | s_v = Σ_l A_lv ‖z_l‖ / m_v，只在 `evidence norm` 对照里使用 |

质量接近 0 或向量范数接近 0 的成分标记为无效（`valid = False`），不参与后续计算。

### 5.4 全局分支（`routeB/rcem.py` 的 `global_scores`）

- 每张支持图的 g 先单位化；
- 同类取平均，再单位化，得到类原型；
- 查询图的单位化 g 与各类原型做余弦相似度，得到 s^G。

基线和 RCEM 的全局分支用的是**同一个函数**。

### 5.5 类对比证据（`evidence`、`image_similarity`、`class_means`、`contrast`）

1. **图与图的成分相似度**：m(e, F) = Σ_u π_u eᵀf_u，其中 π_u = softmax_u(eᵀf_u / t)，t = 0.1。
2. **类平均**：R_c(e) = 成分 e 与 c 类所有支持图的 m 的平均。
3. **类对比证据**：d_v^c = R_c(e_v) − t_n · log[ 1/(N−1) · Σ_{b≠c} exp(R_b(e_v) / t_n) ]，t_n = 0.1。
4. **支持侧证据** d_s：计算时排除支持图自己，也就是 R_c^{(−i)}。1-shot 时同类没有其他图，所以 β_s 自动为 0。

`--evidence` 的几种取值：

| 取值 | 含义 |
|---|---|
| `disc` | 上面的类对比证据，即本文方法 |
| `same` | 只用 R_c，不做对比 |
| `norm` | 用激活强度 s_v 代替证据 |
| `shuffle` | 打乱类别标签后的证据（只在 `eval_b.py` 的对照里使用） |
| `none` | 不用证据 |

### 5.6 代价与固定质量的部分最优传输（`local_scores`、`sinkhorn_partial`、`class_marginals`）

**代价**：

C_vj^c = 1 − ê_vᵀê_j + β_q · h(d_v^c) + β_s · h(d_j)，其中 h(u) = softplus(−u / t_d)，t_d = 0.1。

证据越弱，代价越高，这个成分越容易被拒绝。

**边缘分布**：
- 查询侧：每个有效成分 a_v = 1 / (有效成分数)；
- 支持侧（`class_marginals`）：c 类的每张支持图容量为 1/K，再平均分给它的有效成分。

**部分传输**：总共只运输 μ 的质量，其余 1 − μ 被拒绝。实现方式是给代价矩阵加一行一列"虚拟"节点，边缘分别为 [a, 1−μ] 和 [b, 1−μ]，代价为 0；虚拟行到虚拟列这条边被屏蔽，而不是赋一个大数。

**求解**：对数域 Sinkhorn，ε = 0.05。训练时迭代 30 次（`--iters 30`，展开后可以求导），评测时迭代 100 次。

**局部分数**：s_c^L = −⟨T*, C⟩ / ⟨T*, 1⟩，即被运输部分的平均代价取负。

**几种特殊情况**：
- μ ≥ 1 时退化为普通的完全最优传输，也就是"不拒绝"的对照（`transport='full'`）。
- `eval_b.py` 里还有三种只用于对照实验的传输方式：
  - `randdrop`：随机丢弃同样比例的成分；
  - `topk`：按证据保留前 k 个成分；
  - `indep`：独立耦合，不做最优传输。

### 5.7 训练目标（`routeB/train_routeB.py` 的 `RouteBNet`）

- 两个分支先各自乘以可学习温度，再取 softmax：p_G = softmax(τ_G s^G)，p_L = softmax(τ_L s^L)。τ 用 softplus 参数化，保证为正，初始值由 `--tau_G`、`--tau_L` 给定。
- 融合概率：p_λ = (1 − λ) p_G + λ p_L。训练时 λ 固定为 `--lam`。
- 损失：L = CE(p_λ) + η_G · CE(p_G) + η_L · CE(p_L)，论文设置为 η_G = η_L = 0.5（`sgd_pair.sh` 中写死）。
- `--local_head dense|dn4` 把局部分支换成逐位置原型分类或 DN4 的 k 近邻，用于"局部监督形式"的对照实验。默认是 `rcem`。

### 5.8 基线（`routeA/train_routeA.py --gamma 0`）

- 只有全局分支：单位化 GAP、类原型、余弦相似度乘以可学习温度 τ（初始值 10），损失为交叉熵。
- `gamma = 0` 时不计算 NDM 成分，速度与 ProtoNet 相同。
- 和 RCEM 共用同一份骨干网络代码、数据加载器、优化器设置、episode 预算和随机种子流程。
- 训练 episode 的随机种子为 `seed × 1000 + epoch`，所以同一个种子下，两个模型每个 epoch 看到的训练 episode 相同。

### 5.9 测试时的标定与融合（`routeB/fusion.py`、`routeB/eval_b.py`）

所有选择都**只用验证集**完成。之后冻结这些选择，再去读测试集。

1. **温度标定**：在 1000 个固定的验证 episode（种子 1000 + K）上，对每个分支从 `np.geomspace(1, 300, 41)` 中选出使验证集负对数似然最小的温度。因为温度来自这个网格，不同模型得到相同的温度值是正常的。
2. **1-shot 融合权重 λ₀**：从 {0, 0.25, 0.5, 0.75, 1} 中选验证集准确率最高的；准确率相同时选负对数似然更小的。λ₀ = 0 表示测试时只用全局分支。
3. **5-shot 支持集选择器**：每个 episode 单独选 λ_S：
   - λ_S = argmin_λ [ R̂_S(λ) + ζ λ² ]；
   - R̂_S 是把每张支持图轮流当作查询（留一法）得到的负对数似然，只用到支持集，不看查询图；
   - 收缩系数 ζ 从 {0, 0.02, 0.05, 0.1, 0.2, 0.5} 中按验证集准确率选出。
4. **报告的主结果**：1-shot 是 `fixed`（用 λ₀），5-shot 是 `select`（用 λ_S）。

---

## 6. 训练

### 6.1 DN4 训练协议（`sgd_pair.sh`）

| 项目 | 设置 |
|---|---|
| 训练量 | 30 个 epoch，每个 epoch 10,000 个 episode；5-way，每类 15 个查询 |
| 优化器 | SGD，momentum 0.9，dampening 0.9，不用 Nesterov，weight decay 5e-4 |
| 学习率 | ResNet-12 为 0.01，Conv-4 为 0.05；余弦退火到初始值的 1e-3 |
| 梯度裁剪 | 默认不裁剪（`CLIP=0`）；Conv-4 在 Cars 和 miniImageNet 上两种方法都用 `CLIP=5` |
| 模型选择 | 每个 epoch 结束后，在 1000 个固定验证 episode（种子 777）上测试，保存融合准确率最高的模型 `model_best.pth.tar` |
| 预训练 | 无，所有模型都从随机初始化开始训练 |
| 断点续训 | 脚本中断后，重新执行同一条命令即可，会从 `model_last.pth.tar` 精确恢复，包括优化器、学习率调度器和随机数状态；已训练完的会自动跳过 |



**测试协议**：5 轮，每轮 2000 个 episode，种子为 `轮次 × 100 + K`，共 10,000 个。所有模型使用**完全相同**的测试 episode，所以可以做配对比较。

**三种模式**：

| 模式 | 做什么 |
|---|---|
| `--fixed --no_controls` | 使用训练时的设置，只评测 RCEM 本身（论文主结果） |
| `--refs_only` | 只评测参照结果，用于基线检查点 |
| 不加 `--fixed` | 冻结骨干后的完整诊断：先在验证集上搜索 μ × β_q × β_s，再跑分组、拒绝方式、证据形式三类对照（第一阶段的筛选实验） |

其他参数：
- `--val_episodes`（默认 1000）、`--test_episodes`（默认 2000）、`--test_seeds`（默认 0–4）；
- `--occlude`：遮挡鲁棒性测试，每张图遮住指定比例的面积；
- `--cache_on_cpu`：特征缓存放内存，不放显存；
- `--device`：运行设备。

**输出**：每个检查点一个子目录，名字是"上级目录名 + 运行名（各取末尾若干字符）+ 检查点哈希前 8 位"。

| 文件 | 内容 |
|---|---|
| `feat_<split>_r5_it1_maps.pt` | 特征缓存，重复评测时直接读取 |
| `results_<K>shot.json` | 验证集上的所有选择：`choices[selected]` 里有 `tau_G`、`tau_L`、`lam_fixed`（即 λ₀）、`zeta`（5-shot），以及 `lambda` 字段（λ_S 分布和相关系数） |
| `report_<K>shot.md` | 可读的报告 |
| `test_<K>shot/B_fixed.npz` | RCEM 1-shot 主结果（`accuracy` 数组，大小为 5×2000） |
| `test_<K>shot/B_select.npz` | RCEM 5-shot 主结果 |
| `test_<K>shot/gap.npz` | 在 RCEM 目录下，是 RCEM 的全局分支；在基线目录下，是基线本身 |
| `test_<K>shot/<设置>__global/local/fixed/select/oracle/random_gate.npz` | 各个分支单独的结果、λ 的上限（oracle）、随机选 λ 的结果 |
| `test_<K>shot/ndcp.npz`、`routeA.npz` | 早期方法的参照结果，论文不用 |


6. **随机性。** 训练 episode 由 `seed × 1000 + epoch` 决定，验证和测试 episode 由固定种子决定，NDM 初始化是确定性的。在 CPU 上，同一种子可以逐位复现；GPU 上由于 cuDNN 的非确定性，可能有微小差异。
