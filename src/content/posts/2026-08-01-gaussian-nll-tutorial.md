---
title: Gaussian Negative Log-Likelihood 完全指南
author: 思喆
pubDatetime: 2026-08-01T08:02:31Z
featured: true
draft: true
tags:
  - 算法
description: Gaussian NLL 完全指南
---

> 因为极大似然估计 (MLE) 本身就是统计学里最基础的估计准则，而负对数似然 (NLL) 就是它的「最小化形式」。当你的模型假设输出服从高斯分布时，NLL 就成了 Gaussian NLL。 几乎所有你熟悉的 loss（MSE、MAE、交叉熵……）都是某种分布假设下的 NLL 特例。

---

## 目录

1. [概率基础：为什么 NLL 能当 loss](#1-概率基础为什么-nll-能当-loss)
2. [Gaussian NLL 的推导与两种形式](#2-gaussian-nll-的推导与两种形式)
3. [与 MSE 的关系：同源不同心](#3-与-mse-的关系同源不同心)
4. [统一视角：loss = 分布假设](#4-统一视角loss--分布假设)
5. [贝叶斯视角与统计性质](#5-贝叶斯视角与统计性质)
6. [与 VAE 的联系：重建项就是 NLL](#6-与-vae-的联系重建项就是-nll)
7. [与更多生成模型/算法的联系](#7-与更多生成模型算法的联系)
8. [NLL 作为评估指标](#8-nll-作为评估指标)
9. [学习不确定性：异方差回归](#9-学习不确定性异方差回归)
10. [鲁棒性与替代分布](#10-鲁棒性与替代分布)
11. [分类里的 NLL 变体与取舍](#11-分类里的-nll-变体与取舍)
12. [数值实现要点](#12-数值实现要点)
13. [何时用哪个 loss：决策指南](#13-何时用哪个-loss决策指南)

---

## 1. 概率基础：为什么 NLL 能当 loss

### 1.1 从 MLE 说起

给定数据集 $\{(x_i, y_i)\}_{i=1}^N$ 和带参数 $\theta$ 的概率模型 $p_\theta(y|x)$，极大似然估计选择让观测到的数据最可能的参数：

$$
\hat\theta_{MLE} = \arg\max_\theta \prod_{i=1}^N p_\theta(y_i | x_i)
$$

取对数（把连乘变连加，数值更稳定），最大化对数似然 (log-likelihood)：

$$
\hat\theta_{MLE} = \arg\max_\theta \sum_{i=1}^N \log p_\theta(y_i | x_i)
$$

习惯上优化器都在做最小化，于是取负号，得到 负对数似然 (Negative Log-Likelihood, NLL)：

$$
\boxed{\ \mathcal{L}_{NLL}(\theta) = - \sum_{i=1}^N \log p_\theta(y_i | x_i)\ }
$$

NLL 就是一个完全合格的 loss：它非负、可微（对大多数分布）、对每个样本有定义，而且——这是关键——它的优化目标有统计学意义。

### 1.2 MLE 为什么是一个"好"的估计准则

- 一致性 (Consistency)：样本量 $N \to \infty$ 时，MLE 收敛到真实参数。
- 渐近有效性 (Asymptotic efficiency)：MLE 的渐近方差达到 Cramér–Rao 下界，即所有一致估计量中最小的方差（在正则条件下）。
- 渐近正态性：$\sqrt{N}(\hat\theta - \theta^*) \xrightarrow{d} \mathcal{N}(0, \mathcal{I}(\theta^*)^{-1})$，其中 $\mathcal{I}$ 是 Fisher 信息矩阵。

所以最小化 NLL不是拍脑袋选的，而是统计推断的核心准则落到优化问题上的结果。

### 1.3 严格正确的评分规则：NLL 唯一地“奖励诚实”

这是 NLL 为什么能当 loss 最深刻的一个理由。假设真实的输出分布是 $P$，你的模型拟合的是 $Q$，则期望 NLL 是：

$$
\mathbb{E}_{y \sim P}\big[-\log Q(y)\big] = H(P) + \mathrm{KL}(P \| Q)
$$

其中 $H(P)$ 是 $P$ 的熵（与 $Q$ 无关的常数），$\mathrm{KL}(P\|Q)$ 是 KL 散度（$\ge 0$，当且仅当 $Q=P$ 时为 0）。

这个分解告诉我们：

1. 期望 NLL = 交叉熵 $H(P,Q)$。最小化 NLL 就是在最小化交叉熵。
2. 由于 KL $\ge 0$ 且唯一零点在 $Q=P$，只有当模型分布完全等于真实分布时，期望 NLL 才达到最小。
3. 这意味着 NLL 是一个严格正确的评分规则 (strictly proper scoring rule)：任何「装腔作势」的分布（高估不确定性、低估不确定性、说错均值）都会让期望 NLL 变大。

> 一个对不确定性撒谎的模型，在它自信但错误的点上会给真实数据分配极小的概率，log 之后是巨大的负值，NLL 惩罚它。因此 NLL 强制模型既说对均值、也说对方差——这是 MSE 做不到的。

### 1.4 用 R 亲手验证：最小化 NLL = 求 MLE

上面的抽象论证，用几行 R 就能亲手看到。我们先从已知的高斯分布里抽样本，然后把「负对数似然」当成一个普通 loss 交给优化器最小化——除了导数用数值方法算之外，和训练神经网络毫无区别：

```r
set.seed(42)
n <- 500
mu_true <- 2.0; sd_true <- 1.5
y <- rnorm(n, mean = mu_true, sd = sd_true)      # 观测数据

# NLL 作为 loss：参数是 (mu, log_sigma)，用 log 参数化保证 sigma > 0
nll <- function(par) {
  mu  <- par[1]
  sig <- exp(par[2])
  -sum(dnorm(y, mean = mu, sd = sig, log = TRUE))
}

fit <- optim(c(0, 0), nll)                        # 最小化 NLL
(est <- c(mu = fit$par[1], sd = exp(fit$par[2])))
```

输出大约是：

```
       mu        sd
1.955132  1.456880
```

现在把它和解析 MLE 对比（高斯分布的 MLE 就是样本均值，标准差取 n 分母的均方根，而不是 `sd()` 的 n−1 分母）：

```r
c(sample_mean = mean(y),
  mle_sd      = sqrt(mean((y - mean(y))^2)))
```

两个结果几乎重合。这说明最小化 NLL 和教科书里的 MLE 是同一个数——优化器不知道统计，但它把统计做对了。

再验证第 1.3 节的诚实性质：把分布参数换掉（方差报小 = 过度自信，方差报大 = 过度保守），平均 NLL 都会上升，真实参数处取到最小：

```r
nll_avg <- function(mu, sd) -sum(dnorm(y, mean = mu, sd = sd, log = TRUE)) / n
c(truth          = nll_avg(est["mu"], est["sd"]),            # 最小
  overconfident  = nll_avg(est["mu"], 0.5 * est["sd"]),      # 方差报小了
  underconfident = nll_avg(est["mu"], 2.0 * est["sd"]))      # 方差报大了
```

```
        truth   overconfident  underconfident
      1.79519         2.60189         2.11337
```

注意惩罚的不对称：方差报小（过度自信）比报大（过度保守）罚得更狠（2.60 vs 2.11）。这是因为 NLL 里方差项 $\frac12\log\sigma^2$ 随 $\sigma$ 只做对数增长（罚得温和），而第一项 $(y-\mu)^2/2\sigma^2$ 在 $\sigma$ 过小时会指数爆炸（罚得凶）。模型宁可保守，也不敢吹牛——这在不确定性估计里是理想性质。

三件事值得记住：

1. 上面这个 `nll` 函数就是 Gaussian NLL loss 本身——只不过这里参数是标量、用 `optim` 迭代，而深度学习里参数是网络的权重、用反向传播迭代。目标函数一字不差。
2. `optim` 求出的 $\mu$ 恰是样本均值——这对应后面第 5.3 节的估计方程：令 NLL 对 $\mu$ 的梯度为零，解出 $\sum(y_i-\mu)/\sigma^2=0 \Rightarrow \mu=\bar y$。
3. 撒谎方差让 NLL 升高，就是第 1.3 节「NLL 唯一地奖励诚实分布」的直接证据。

### 1.5 信息论解读：为什么是 log

§1.1 说取对数是为了「连乘变连加、数值更稳定」——这只是表面理由。信息论给了更本质的答案：`-log p(x)` 在信息论里就是事件 $x$ 的「惊讶度」或「自信息 (surprisal)」：

$$
I(x) = -\log p(x)
$$

- 单位取决于底数：以 $e$ 为底是 nat，以 2 为底是 bit。于是 log 把「概率」翻译成了「信息量」。
- 期望惊讶度就是熵：$\mathbb{E}_{x\sim p}[-\log p(x)] = H(p)$——正好呼应 §1.3 里的 $H(P)$。
- Shannon 最优编码：若用二进制码字编码事件，$x$ 的最优码长恰好是 $-\log_2 p(x)$ bit。这意味着 NLL 有一层更深的解释：

> 最小化 NLL = 最小化平均编码长度 = 找到最能「压缩」数据的模型（最小描述长度，MDL）。 一个好模型就是一台好的数据压缩器——这个观点在生成模型的评估里会反复出现（§8 的 bits/dim）。

```r
-log2(0.1)   # 概率 0.1 的事件携带 3.32 bit 的信息
-log(0.1)    # 同一件事，2.30 nat
```

- 这还解释了「log 里不能出现零概率」：$p=0$ 时惊讶度发散到无穷。NLL 对模型完全否定某个真实观测的惩罚是无穷大的——这是第 1.3 节「诚实」性质的信息论化身。

---

## 2. Gaussian NLL 的推导与两种形式

假设模型假设条件分布是高斯分布：

$$
y \sim \mathcal{N}\big(\mu(x), \sigma^2(x)\big)
$$

其中 $\mu(x), \sigma^2(x)$ 都是神经网络的输出（或任何参数化函数）。概率密度：

$$
p(y|x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

取负对数：

$$
-\log p(y|x) = \frac{(y-\mu)^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)
$$

批量版本（求和或取平均）：

$$
\boxed{\ \mathcal{L}_{GNLL} = \frac{1}{N}\sum_{i=1}^N \left[ \frac{(y_i - \mu_i)^2}{2\sigma_i^2} + \frac{1}{2}\log \sigma_i^2 \right] + \text{const}\ }
$$

注意两点：

- 常数项 $\frac{1}{2}\log(2\pi)$ 与参数无关，可以扔掉（PyTorch 的 `GaussianNLLLoss`、本教程 §12.2 的 `gaussian_nll()` 默认也不含它）。
- 第二项 $\frac{1}{2}\log\sigma_i^2$ 是正则项，它防止 $\sigma \to \infty$（那样第一项虽然趋近 0，但 log 项会无限大）。

### 两种形式

形式 A：同方差 (homoscedastic)，固定 $\sigma$。 假设所有样本共享同一个方差，此时 $\sigma^2$ 只是常数：

$$
\mathcal{L} = \frac{1}{2\sigma^2}\underbrace{\sum_i (y_i-\mu_i)^2}_{\text{MSE}} + \text{const}
$$

形式和 MSE 只差一个缩放因子 $\frac{1}{2\sigma^2}$ 和一个常数。这就是「MSE = 固定方差下的 Gaussian NLL」的出处。

形式 B：异方差 (heteroscedastic)，$\sigma$ 随输入变化。 网络同时输出 $\mu(x)$ 和 $\sigma(x)$（实践中输出 $\log \sigma^2$ 保证正性）：

$$
\mathcal{L} = \frac{1}{N}\sum_i \left[ \frac{(y_i-\mu_i)^2}{2\sigma_i^2} + \frac{1}{2}\log\sigma_i^2 \right]
$$

这是不确定性感知回归的标准目标，也是下面第 9 节的主角。

---

## 3. 与 MSE 的关系：同源不同心

### 3.1 数学上的等价性

从 2 节的推导看，固定方差时：

$$
\text{Gaussian NLL} \ \propto \ \text{MSE} + \text{const}
$$

所以 MSE 是不是 Gaussian NLL 的答案取决于你怎么问：

- 优化等价：固定 $\sigma$ 时，二者给出相同的 $\mu^*$。在「找最好的均值预测」这个意义上，MSE 就是 Gaussian NLL。
- 模型不等价：MSE 是「默认 $\sigma=1$（或常数）且不估计它」的 Gaussian NLL；Gaussian NLL 多了一个 $\sigma$ 的可移动参数，能额外描述不确定性。

### 3.2 隐式假设：MSE 暗中假设了什么

写 MSE 的训练目标，等于你已经在隐式地假设：

$$
y = f(x) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

即噪声是加性、独立、同方差的高斯噪声。这个假设有三大后果：

1. 对离群点极度敏感。高斯分布的尾巴很轻，平方项 $(y-\mu)^2$ 让大残差贡献巨量梯度。一个离群点就能扭曲回归线。
2. 无法表达有些点就是不可预测。MSE 强制所有点得到同样的信任度。
3. 当真实噪声是重尾时，MSE 不再是 MLE，统计性质（有效性）随之丧失。

### 3.3 信息论视角：为什么是平方

给定均值 $\mu$、方差 $\sigma^2$ 这两个约束，所有满足的分布中熵最大的是高斯分布（最大熵原理）。而 MSE 恰好对应「我除了均值和方差之外一无所知、因此假定最不偏不倚的高斯」的信息量度量。也就是说：

> MSE 是信息论上最诚实的二阶矩 loss——它在不额外假设任何高阶结构的前提下度量误差。 这也解释了为什么它在工程中如此通用：当你对噪声形态一无所知时，均值+方差是最合理的最低阶假设。

### 3.4 梯度视角：$\sigma$ 是每个样本的自动权重

对 $\mu$ 求导：

$$
\frac{\partial \mathcal{L}}{\partial \mu} = - \frac{y-\mu}{\sigma^2}
$$

和 MSE 的梯度 $-(y-\mu)$ 相比，Gaussian NLL 多了一个因子 $1/\sigma^2$。这个因子的意义极其深远：

> $\sigma^2$ 越大（模型越不确定），该样本在 $\mu$ 上的梯度越小。 网络自动学会了「少被不可预测的样本带偏」。

这就是异方差 Gaussian NLL 的自动加权 (adaptive weighting / heteroscedastic weighting) 性质，后面第 9 节会展开。

---

## 4. 统一视角：loss = 分布假设

可以这样理解所有 loss 关系：

> 每一种回归/分类 loss，都等价于「假设某个条件分布」后的 NLL。选 loss = 选分布的假设。

| 你写的 loss | 隐含的分布假设 | NLL 形式 |
|---|---|---|
| MSE / L2 | 高斯 $\mathcal{N}(\mu, \sigma^2)$，$\sigma$ 固定 | $\frac{(y-\mu)^2}{2\sigma^2}+\frac12\log(2\pi\sigma^2)$ |
| MAE / L1 | 拉普拉斯 $\mathrm{Laplace}(\mu, b)$ | $\frac{\lvert y-\mu \rvert}{b}+\log(2b)$ |
| Huber | 中心高斯、尾部拉普拉斯的分段分布 | 分段 NLL |
| 二分类交叉熵 (BCE) | 伯努利 $\mathrm{Bernoulli}(p)$ | $-y\log p-(1-y)\log(1-p)$ |
| 多分类交叉熵 (CE) | 类别分布 $\mathrm{Categorical}(\pi)$ | $-\sum_k y_k \log \pi_k$ |
| Poisson loss | 泊松 $\mathrm{Poisson}(\lambda)$ | $\lambda - y\log\lambda$（+阶乘项） |
| Gaussian NLL（异方差） | 高斯，$\mu,\sigma$ 都学 | $\frac{(y-\mu)^2}{2\sigma^2}+\frac12\log\sigma^2$ |
| Student-t NLL | 学生 t 分布（重尾） | $\frac{\nu+1}{2}\log\big(1+\frac{(y-\mu)^2}{\nu}\big)+\text{const}$ |

几个值得注意的推论：

- BCE/CE 也是 NLL。分类问题的交叉熵，就是伯努利/类别分布假设下的负对数似然。所以「回归用 NLL、分类用 CE」其实是一回事——**都是 NLL，只是分布假设不同**。
- L1 vs L2：拉普拉斯分布尾巴比高斯重，所以 L1 对离群点更鲁棒——这正是「选择 loss = 选择对异常值的容忍度」。
- Huber 是折中：中心用平方（对正常点光滑高效）、尾部用绝对值（抑制离群点爆炸），对应一个「高斯中心 + 拉普拉斯尾巴」的分布。
- 广义线性模型 (GLM) 早就这么干了：给定误差分布族（高斯、泊松、二项……），用 NLL 做拟合，就是「选分布 = 选 loss」的古典形态——而且 R 的 `glm()` 函数把这个思想封装得明明白白。详见下面的 §4.1。

<img src="/upload/pic/distribution_comp.png" width="65%" />


### 4.1 GLM：古典时代，选分布 = 选 loss

现代机器学习里换个 loss 这件事，统计学家 1972 年就系统化过了。Nelder & Wedderburn 的广义线性模型 (GLM) 用一句话概括就是：把「响应变量服从什么分布」和「预测值 = 参数化函数」解耦，然后用该分布的对数似然来拟合。GLM 有三个组成部分：

1. 随机成分 (random component)：$y$ 服从某个指数族分布——高斯、泊松、二项、伽马……这是你选的分布假设。
2. 系统成分 (systematic component)：线性预测器 $\eta = \beta^\top x$。
3. 连接函数 (link function)：$g(\mu) = \eta$，把均值 $\mu$ 和线性预测器连起来（如 logit、log、identity）。

而拟合方式，就是最小化所选分布族的 NLL——和现代深度学习一字不差。

#### R 的 `glm()`：选 loss 翻译成选 family

R 里这个思想落成了最直观的 API：同一个 `glm()` 函数，`family` 参数不同，内部用的 loss 就不同。

```r
fit_gauss <- glm(y     ~ x, family = gaussian())   # 高斯族   → 最小化 MSE / 高斯 NLL
fit_pois  <- glm(y_cnt ~ x, family = poisson())    # 泊松族   → 最小化 Poisson NLL
fit_binom <- glm(y_bin ~ x, family = binomial())   # 二项族   → 最小化二元交叉熵
```

三个调用看起来只是换了个 `family` 字符串，实际上换的是整个概率模型：

| R 的 family | 分布族 | 内部使用的 loss | 现代深度学习里的名字 |
|---|---|---|---|
| `gaussian()` | 高斯 $\mathcal{N}(\mu,\sigma^2)$ | 高斯 NLL（固定 $\sigma$） | MSE |
| `binomial()` | 伯努利 / 二项 | 伯努利 NLL | 二元交叉熵 (BCE) |
| `poisson()` | 泊松 | 泊松 NLL | Poisson loss |
| `Gamma()` | 伽马 | 伽马 NLL | — |
| `inverse.gaussian()` | 逆高斯 | 逆高斯 NLL | — |

Note 1：R 的 family 对象里就装着 NLL 公式。 每个 family 对象都有一个 `dev.resids` 字段，它逐点给出「负二倍的对数似然偏离量」——也就是逐点的 loss 贡献：

```r
gaussian()$dev.resids   # function(y, mu, wt) wt * ((y - mu)^2)  → 逐点 MSE
poisson()$dev.resids    # 2 * wt * (y*log(y/mu) - (y - mu))       → 逐点 Poisson NLL
binomial()$dev.resids   # 二项偏差，等价于逐点 -2 * log-likelihood
```

你选 `family = binomial()`，`glm()` 内部就用 `binomial()$dev.resids` 这条 NLL 做拟合；选 `family = gaussian()`，就用 `(y-mu)^2`。这不是接口上的巧合，而是选分布 = 选 loss 在代码层面的直接体现。

Note 2：GLM 的求解算法就是加权 NLL 优化。 `glm()` 默认用 IRLS（迭代加权最小二乘）：每一步都解一个加权最小二乘问题，权重来自所选分布的方差函数（高斯族权重恒定、泊松/二项族权重随 $\mu$ 变化）。这和神经网络的 SGD 是同一个 NLL 目标、两种优化器——IRLS 是牛顿法的变体（批量、精确二阶），SGD 是随机一阶法（适合大规模、非凸）。两者收敛到的解是同一个 MLE。

Note 3：连接函数 → 激活函数。 GLM 的逆连接函数 `linkinv`，就是现代神经网络里的输出激活函数。

- `binomial()$linkinv` 是 logit 的逆 = sigmoid；
- `gaussian()$linkinv` 是 identity（线性输出）；
- `poisson()$linkinv` 是 exp。

所以：

- logistic 回归 = 单神经元网络 + sigmoid 输出 + BCE；
- 线性回归 = 单神经元网络 + 线性输出 + MSE。

你在 torch 里手写的 `nn_linear(1,1)` + `nnf_binary_cross_entropy`，本质上就是在手写一个 `glm(..., family = binomial())`。

下面这段 R 代码把「现代 = 古典」落成可见的等价：高斯族 GLM 的系数和 `lm()`（最小二乘）完全一致——因为它们都在最小化同一个高斯 NLL：

```r
set.seed(1)
x <- rnorm(200)
y <- 1.5 + 0.8 * x + rnorm(200, sd = 0.5)

# 三条路径得到同一个 β：最小二乘、高斯族 GLM、手动最小化高斯 NLL（§1.4）
c(ls  = coef(lm(y ~ x))[2],
  glm = coef(glm(y ~ x, family = gaussian()))[2])
#      ls      glm
# 0.7884611 0.7884611
```

再把分类的例子显式化，说明 `glm` 的 `binomial()` = torch 里 sigmoid + BCE：

```r
set.seed(2)
x2 <- rnorm(300)
p <- plogis(0.5 - 1.2 * x2)          # 真实 logit 线性
label <- rbinom(300, 1, p)
fit <- glm(label ~ x2, family = binomial()) # 学习 logit 系数
round(coef(fit), 3)                         # 截距 0.54、斜率 -1.32 ≈ 真实 (0.5, -1.2)
```

#### 为什么说现代 ML 的 loss 是从这个体系来的

把历史捋直：Fisher (1922) 提出极大似然 → Nelder & Wedderburn (1972) 把不同分布的 NLL 统一进 GLM 框架 → 深度学习时代，人们只是在 GLM 的两端做了推广：

- 输入侧：线性预测器 $\eta = \beta^\top x$ 换成多层神经网络 $f_\theta(x)$（模型容量更大）；
- 输出侧：把 GLM 的「分布族 + 连接函数」换成任意输出层。sigmoid 输出 + BCE = 二项族 + logit；线性输出 + MSE = 高斯族 + identity；softmax 输出 + 交叉熵 = 类别分布（多分类的 GLM 推广，即多项 logit）；
- loss 侧：不变的还是那条 NLL——`MSE / BCE / Poisson loss` 就是 GLM 里那三族 NLL，一个字都没变。

所以当你今天在 torch 里纠结「回归该用 MSE 还是 Gaussian NLL」时，你其实是在做 50 年前 Fisher 和 Nelder 就定型的选择：给数据假设一个分布，然后最小化它的负对数似然。 界面从 `glm(family=...)` 变成了 `loss_fn(...)`，但统计内核完全相同。

### 4.2 Barron 2019：一个分布族统一所有鲁棒 loss

§4.1 说 GLM 用选分布来选 loss。能不能更进一步——用同一个分布族的连续形状参数，把几乎所有鲁棒 loss 串起来？Barron (2019) 的「通用自适应鲁棒损失」做到了。它对残差定义（$\alpha \ne 0, 2$，在 $\alpha = 0, 2, -\infty$ 处取极限，$c$ 控制中心「二次碗」的大小）：

$$
f(x; \alpha, c) = \frac{|\alpha-2|}{\alpha}\left[\left(\frac{(x/c)^2}{|\alpha-2|} + 1\right)^{\alpha/2} - 1\right]
$$

形状参数 $\alpha$ 连续扫过一族鲁棒 loss：

| $\alpha$ | 还原的 loss | 形式 |
|---|---|---|
| $2$ | L2 | $\frac12(x/c)^2$ |
| $1$ | Charbonnier / pseudo-Huber（smooth-L1 的近亲）** | $\sqrt{(x/c)^2+1}-1$ |
| $0$ | Cauchy（Lorentzian） | $\log\!\left(\frac12(x/c)^2+1\right)$ |
| $-2$ | Geman-McClure | $\frac{2(x/c)^2}{(x/c)^2+4}$ |
| $-\infty$ | Welsch | $1-\exp\!\left(-\frac12(x/c)^2\right)$ |

两个关键点，让它成为 §4 那张 loss = 分布假设表的加冕：

1. 它就是一族 NLL：把该损失解释为某单变量密度的负对数——$p(z) \propto \exp(-f(z))$，且归一化常数有解析形式。所以它确实是精确的负对数似然：$\alpha=2$ 时是正态（回到 Gaussian NLL），$\alpha=0$ 时是柯西。
2. $\alpha$ 可以学：把 $\alpha$（和 $c$）当作网络参数随数据自适应——等于让模型自己回答「我的噪声是高斯（$\alpha=2$）还是重尾（$\alpha \to -\infty$）」。论文还建议从高 $\alpha$ 退火到低 $\alpha$ 以避开局部最优。

于是 MSE、Cauchy、Welsch……不再是几个孤立的选择，而是一条一维流形上的采样点，每个点都对应一个真实分布——这比每种 loss 换个分布假设更统一。

---

## 5. 贝叶斯视角与统计性质

### 5.1 MAP 估计：NLL + 先验

贝叶斯公式给出后验：

$$
\log p(\theta|\mathcal{D}) = \underbrace{\log p(\mathcal{D}|\theta)}_{-\text{NLL}} + \underbrace{\log p(\theta)}_{\text{先验}} - \log p(\mathcal{D})
$$

最大化后验（MAP）等价于：

$$
\hat\theta_{MAP} = \arg\min_\theta \big[ \underbrace{-\sum_i \log p_\theta(y_i|x_i)}_{\text{NLL}} - \underbrace{\log p(\theta)}_{\text{正则项}} \big]
$$

这就是 NLL + 正则化 = MAP 的经典对应。L2 正则对应高斯先验，L1 正则对应拉普拉斯先验。所以你在深度学习里写的 `loss = NLL + λ||θ||²`，其实是一个完整的贝叶斯后验最大化问题。

### 5.2 Fisher 信息：$\sigma$ 告诉我们局部曲率

Gaussian NLL 对 $\mu$ 的二阶导（Fisher 信息）恰好是精度 (precision)：

$$
\mathcal{I}(\mu) = -\mathbb{E}\left[\frac{\partial^2 \log p}{\partial \mu^2}\right] = \frac{1}{\sigma^2}
$$

这意味着：

- $\sigma^2$ 直接刻画了「$\mu$ 这个估计的置信度」。$\sigma$ 小 → Fisher 信息大 → 目标函数在该点弯曲剧烈 → $\mu$ 被数据牢牢锁定。
- 这也是为什么预测方差本质上是预测置信度——它不是可有可无的装饰，而是模型对自身预测精确度的声明。
- 该观察是贝叶斯神经网络、Laplace 近似、以及各种不确定性估计方法的基础。

### 5.3 得分函数与估计方程

Gaussian NLL 的梯度 $\partial \log p/\partial \mu = (y-\mu)/\sigma^2$ 在统计上叫得分函数 (score)，它满足期望为零（模型正确时）：

$$
\mathbb{E}_{y\sim p_\theta}\left[\frac{\partial \log p_\theta}{\partial \theta}\right] = 0
$$

这就是估计方程 (estimating equation)：训练到收敛时，NLL 的梯度期望为零，模型给出的分布与数据分布「对齐」到了得分层面。

### 5.4 信息准则：NLL + 复杂度

§5.1 说「NLL + 正则 = MAP」。统计里还有一套更直接地把 NLL 用于模型选择的工具：信息准则 (information criteria)。它们都是「负对数似然 + 复杂度惩罚」：

$$
\mathrm{AIC} = 2k - 2\ln\hat{L} = 2k + 2\cdot\mathrm{NLL}_{\min}, \qquad
\mathrm{BIC} = k\ln n - 2\ln\hat{L} = k\ln n + 2\cdot\mathrm{NLL}_{\min}
$$

其中 $k$ 是参数个数，$n$ 是样本量，$\hat{L}$ 是最大化后的似然（$\mathrm{NLL}_{\min}$ 是最小训练 NLL）。选模型就是在这两个量之间取最小——惩罚项防止「参数越多、NLL 越低」的过拟合。

- AIC 惩罚是常数 2，BIC 惩罚是 $k\ln n$（随样本量增长更重）——所以 BIC 更保守，倾向于更简单的模型。
- BIC 可看成对边际似然（贝叶斯模型证据）的近似，AIC 可看成对「样本外期望 NLL」的近似——两者都是「NLL 最小化 + 复杂度权衡」这条主线的直接产物。

R 里这几乎免费：`glm()` 直接给出 `logLik`，算准则只需一行。下面用「真实为 2 阶多项式」的数据让 AIC/BIC 自己选阶数：

```r
set.seed(1); n <- 150
x <- runif(n, -1, 1)
y <- 1 + 2*x - 3*x^2 + rnorm(n, sd = 0.3)          # 真实是 2 阶
aic <- bic <- rep(NA, 5)
for (d in 0:4) {
  fm <- if (d == 0) glm(y ~ 1) else glm(y ~ poly(x, d, raw = TRUE))
  aic[d+1] <- 2*(d+1) - 2*as.numeric(logLik(fm))   # k = d+1
  bic[d+1] <- (d+1)*log(n) - 2*as.numeric(logLik(fm))
}
cat("AIC:", round(aic,1), "  → 最优阶数", which.min(aic)-1, "\n")
cat("BIC:", round(bic,1), "  → 最优阶数", which.min(bic)-1, "\n")
```

实际输出：

```
AIC: 519.5 395.1  58.1  57.7  59.7   → 最优阶数 3
BIC: 522.6 401.1  67.1  69.7  74.7   → 最优阶数 2
```

BIC 正确选出 2 阶；AIC 因惩罚轻（2 vs $\ln 150 \approx 5.0$）轻微过拟合到 3 阶（58.1 vs 57.7 差距很小）。这正是「复杂度惩罚如何决定模型选择」的直观演示——而驱动这一切的，仍是 NLL。

---

## 6. 与 VAE 的联系：重建项就是 NLL

### 6.1 ELBO 拆解

变分自编码器 (VAE) 优化的是证据下界 (ELBO)：

$$
\log p(x) \ \ge \ \underbrace{\mathbb{E}_{z\sim q_\phi(z|x)}\big[\log p_\theta(x|z)\big]}_{\text{重建项 / 负重建 NLL}} \ -\ \underbrace{\mathrm{KL}\big(q_\phi(z|x) \| p(z)\big)}_{\text{正则项}}
$$

这个重建项就是一个条件 NLL：

$$
\mathbb{E}_{q(z|x)}\big[\log p_\theta(x|z)\big]
$$

如果解码器假设 $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \sigma^2 I)$，那么：

$$
-\log p_\theta(x|z) = \frac{\|x-\mu_\theta(z)\|^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)
$$

固定 $\sigma$ 时，重建项就是负的 MSE（差个常数）。 所以：

> VAE 的重建 loss 是 MSE 这句话的完整版本是：VAE 假设了一个高斯基底解码器，其重建项是 Gaussian NLL；把方差固定为常数后，它化简为 MSE。 用 BCE 作重建 loss 的 VAE（二值图像），同理是假设伯努利解码器的 NLL。

### 6.2 为什么 VAE 用高斯

1. 连续数据建模的自然选择：高斯是连续空间上的最大熵分布，且解码器只需输出均值（方差可固定）。
2. 重参数化技巧友好：$x = \mu + \sigma\cdot\varepsilon$，$\varepsilon\sim\mathcal{N}(0,I)$，梯度可回传。
3. 解析可算：高斯之间的 KL 有闭式解，ELBO 可以直接写出来。

### 6.3 方差在这里同样重要

- 如果解码器固定 $\sigma=1$ 只学均值，模型被迫「对所有像素一视同仁」，重建会模糊（mean regression 的固有缺陷）。
- 若让解码器输出 $\sigma(x|z)$，重建项成为**异方差 Gaussian NLL**——模型可以表达「这部分像素就是生成不出来」。这缓解了 VAE 重建模糊的一个根因，也是部分改进工作（如学习感知不确定性）的动机。

### 6.4 一个更大的图景

VAE 的 ELBO 只是「最大化边际对数似然 $\log p(x)$ 的下界」。换句话说：

> VAE 的最终目标仍然是「负对数似然」，只不过对潜变量做近似积分，用一个下界（ELBO）来代替不可解的边际 NLL。 从这点看，Gaussian NLL 和 VAE 不是两个独立的东西，而是「似然目标」在不同复杂度下的两种形态：
>
> - 普通回归：条件 NLL，无潜变量；
> - VAE：带潜变量的边际 NLL，用 ELBO 逼近。

---

## 7. 与更多生成模型/算法的联系

### 7.1 归一化流 (Normalizing Flows)

流模型直接训练精确的 NLL：

$$
\log p(x) = \log p(z) - \sum_k \log\left|\det \frac{\partial f_k}{\partial z}\right|
$$

通过变量替换公式把简单分布映射到数据分布，目标是最大化精确 $\log p(x)$，即最小化 NLL。这是 Gaussian NLL「最小化负对数似然」理念最直接的继承者——只是把「条件高斯假设」换成了「学一个可逆映射」。

### 7.2 扩散模型 (Diffusion Models)

扩散模型的去噪目标来自对 ELBO（又一个对数似然下界）的化简。其简化的 DDPM 目标：

$$
\mathbb{E}_{t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

这看起来是 MSE，但它可以严格推导为每步高斯条件分布的 NLL 下界：反向过程 $p_\theta(x_{t-1}|x_t)$ 被假设为高斯，其 NLL 化简出噪声预测的 MSE。所以扩散模型的「MSE 训练」底层仍然是一个高斯 NLL 目标——和 VAE 同源。

### 7.3 评分匹配 (Score Matching)

对数似然的梯度（得分）$\nabla_x\log p(x)$ 是可学习的对象。高斯噪声扰动下的得分匹配、去噪评分匹配、以及扩散模型，共同点是用去噪 MSE 来估计高斯得分——这又回到了 Gaussian NLL 的梯度结构（第 5.3 节）。三者实际上是同一条线索：从 NLL → 得分 → 去噪。

### 7.4 为什么不提 GAN

GAN 是不以似然为目标的生成模型（用对抗损失）。这也是为什么 GAN 没有 NLL 的漂亮统计性质（无 MLE 一致性、训练不稳定、无显式密度）。对比之下，凡是以「最大似然」为信仰的生成模型（VAE、流、扩散），全部以 NLL 为核心——只是 Gaussian NLL 是最简单的那一格。

### 7.5 其他直系相关

| 领域 | 与 Gaussian NLL 的关系 |
|---|---|
| 分布强化学习 (Distributional RL) | 不预测 Q 的均值而预测回报的分布，常参数化为高斯/类别分布并用 NLL 训练 |
| DeepAR / 时序预测 | 预测未来值的均值+方差（高斯 NLL）作为不确定感知的 point forecast |
| 深度高斯过程 (Deep GP) | 网络输出的高斯 NLL 是天然的观测模型 |
| 变分推断 (VI) | 优化 ELBO 本质是最小化「负变分 NLL + KL 正则」 |
| 混合密度网络 (Mixture Density Network) | 把高斯 NLL 推广为高斯混合 NLL $\sum_k \pi_k \mathcal{N}(\mu_k,\sigma_k)$ |
| 贝叶斯神经网络 (BNN) | 高斯先验/后验，推断时用 NLL 的 Fisher 信息做 Laplace 近似 |

---

## 8. NLL 作为评估指标

前面几节把 NLL 当作**训练目标**；但 NLL 还有一个同样重要的身份：**概率模型的评估指标**。一个概率模型好不好，最终要看「它给真实数据分配的对数似然有多高」——这是 §1.3 严格正确评分的直接推论。它有三个贯穿全行业的标准化形态。

### 8.1 Perplexity：语言模型的 NLL

语言模型以「下一个 token 的类别分布」为输出，训练目标是交叉熵（= NLL）。评估时，行业习惯把平均 NLL 指数化成困惑度 (perplexity)：

$$
\mathrm{PPL} = \exp\!\left(-\frac{1}{N}\sum_i \log p_\theta(y_i)\right)
$$

- PPL 的直觉是模型对每个 token 平均给出多少个候选：若模型对每步都认为「下一个 token 从 10 个候选中均匀出现」，则 PPL = 10。完美模型 → PPL → 1，均匀瞎猜（词表 K）→ PPL ≈ K。
- LLM 训练曲线上你看到的 loss 就是 NLL，报告 ppl 只是把它指数化方便人读。

```r
mnll <- 2.3                  # 某模型在验证集上的平均 NLL（nats/token）
exp(mnll)                    # perplexity ≈ 9.97
```

### 8.2 bits/dim：生成模型的 NLL

生成模型（VAE、流、扩散）的比较标准是 bits per dimension：把 NLL 按「每个维度的比特数」归一化。

$$
\mathrm{bpd} = \frac{\text{总 NLL (nats)}}{\text{维度数} \times \ln 2}
$$

```r
nats_per_dim <- 55.5         # 某模型每维平均 NLL（nats）
nats_per_dim / log(2)        # ≈ 80.1 bits/dim
```

- 为什么 ×ln 2：把 nats 换成 bits（1 nat = 1/ln 2 bit），这样 28×28 的 MNIST 和 32×32 的 CIFAR 能在同一把尺子上比。
- 注意：VAE 报的 bpd 是 ELBO 上界（§6），真实 NLL 只会更小（更好）；流和扩散模型则报精确 NLL 或其可推导的上界。

### 8.3 CRPS：概率预测的另一个严格正确评分

NLL 不是唯一的严格正确评分规则（§1.3）。概率预测领域更常用连续分级概率评分 (CRPS)，它直接比较预测 CDF $F$ 与观测 $y$：

$$
\mathrm{CRPS}(F, y) = \mathbb{E}|X - y| - \tfrac12 \mathbb{E}|X - X'|, \qquad X, X' \stackrel{iid}{\sim} F
$$

对高斯分布有闭式解（$\Phi, \phi$ 为标准正态的 CDF/PDF，$z=(y-\mu)/\sigma$）：

$$
\mathrm{CRPS}(\mathcal{N}(\mu,\sigma^2), y) = \sigma\big[z(2\Phi(z)-1) + 2\phi(z) - \tfrac{1}{\sqrt\pi}\big]
$$

CRPS 与 NLL 同为严格正确评分，但有一个关键差异：NLL 是无界的——观测落在预测分布的极薄尾巴里时 $-\log p$ 可以无限大；CRPS 是 L1 型的，有界、对离群点温和。R 里闭式公式三行就能手写：

```r
crps_gauss <- function(y, mu, sd) {
  z <- (y - mu)/sd
  sd * (z*(2*pnorm(z)-1) + 2*dnorm(z) - 1/sqrt(pi))
}
crps_gauss(1.2, 1.2, 0.8)    # 在均值处 ≈ 0.187 = 0.234·sd
```

下面用「真实分布是重尾 $t_4$、模型却假设高斯」的数据，对比两者对离群点的反应：

```r
set.seed(2)
y_t <- rt(2000, df = 4)*1.5 + 2              # 重尾真实数据
-mean(dnorm(y_t, 2, 1.5, log = TRUE))        # 平均 NLL  ≈ 2.39
mean(crps_gauss(y_t, 2, 1.5))                # 平均 CRPS ≈ 1.13
y_out <- c(y_t, 100)                         # 塞进一个极端离群点
-mean(dnorm(y_out, 2, 1.5, log = TRUE))      # NLL  → 3.46（+1.07）
mean(crps_gauss(y_out, 2, 1.5))              # CRPS → 1.17（+0.05）
```

一个离群点让平均 NLL 暴涨 +1.07，而 CRPS 只动了 +0.05。这就是为什么概率预测里常把 CRPS 当 NLL 的鲁棒替代：模型被单个异常值「单点爆破」的风险更小。它属于更广的严格正确评分家族（能量分、变差分等，Gneiting & Raftery 2007）。

> 小结：NLL 有双重身份——训练时是 loss（§1–§7），评估时是指标（本节）。ppl、bpd、CRPS 都是 NLL（或其替代）在各自领域里的标准化形态。

---

## 9. 学习不确定性：异方差回归

### 9.1 为什么要预测 $\sigma$

真实世界的回归里，不确定性往往随输入变化：有的区域数据密集、噪声小，有的区域数据稀疏、噪声大。同方差 MSE 把不确定性当成常数，就永远学不到这一点。异方差 Gaussian NLL 让网络同时输出 $\mu(x)$ 和 $\sigma(x)$，从而学习输入依赖的噪声。

### 9.2 不确定性分两类

- 偶然不确定性 (Aleatoric)：数据本身不可约的噪声。由 Gaussian NLL 里的 $\sigma$ 建模。
- 认知不确定性 (Epistemic)：模型参数的不确定性（没见过这种数据）。由贝叶斯方法建模（MC dropout、Deep Ensemble、VI）。

两者互补。Gaussian NLL 建模的是前者；常见做法是两者都用：对 ensemble 的每个成员用异方差 NLL，再对成员的 $\mu,\sigma$ 做聚合得到总不确定性。

### 9.3 训练细节与反直觉现象

对 $\sigma^2$ 求导：

$$
\frac{\partial \mathcal{L}}{\partial \sigma^2} = -\frac{1}{2\sigma^2} + \frac{(y-\mu)^2}{2\sigma^4}
$$

令其为零，得到最优解：

$$
\sigma^2 = (y-\mu)^2
$$

即方差的最优值就是残差平方——网络会学习「这个点我残差多大，就报多大多大的方差」。这是完全合理的：方差度量残差的期望平方。

但要小心一个反直觉的坑：如果网络同时优化 $\mu$ 和 $\sigma$，它可能「偷懒」——把 $\sigma$ 调大来掩盖 $\mu$ 的误差，而不是把 $\mu$ 学好。这会导致：

- 期望 NLL 可能还在下降，但均值质量下降；
- $\sigma$ 系统性偏大（差校准）。

缓解手段：用验证集校准（见下）、给 $\sigma$ 加正则、或分阶段训练。

### 9.4 校准 (Calibration)

一个「诚实」的不确定性估计要求校准：声称 95% 置信区间应真的覆盖约 95% 的数据。Gaussian NLL 作为严格正确的评分规则会激励校准，但有限样本、模型误差下仍需验证。经典检查方法：

- 可靠性图：按预测 $\sigma$ 分箱，看真实覆盖率 vs 声称覆盖率；
- 如果过拟合了 NLL 的 log 正则项，常出现 $\sigma$ 偏小、覆盖率不足。

### 9.5 多任务不确定性加权：σ 自动给每个任务配权重

§9.1–§9.4 学的是样本级异方差 $\sigma(x)$。多任务学习里还有另一处 σ 大显身手：任务级同方差加权（Kendall et al. 2018）。假设两个回归任务共享一个网络，输出 $y_1, y_2$ 各自带同方差高斯噪声 $\sigma_1, \sigma_2$：

$$
p(y_1, y_2|x) = \mathcal{N}(y_1; f_1(x), \sigma_1^2)\cdot\mathcal{N}(y_2; f_2(x), \sigma_2^2)
$$

取 NLL，就得到多任务 loss：

$$
\mathcal{L} = \underbrace{\frac{1}{2\sigma_1^2}\mathrm{MSE}_1 + \frac{1}{2\sigma_2^2}\mathrm{MSE}_2}_{\text{被 }\sigma\text{ 缩放的每任务 loss}} + \underbrace{\log\sigma_1 + \log\sigma_2}_{\text{防止 }\sigma\!\to\!\infty}
$$

`1/σ²` 就是自动的任务权重：任务噪声大（σ 大）自动降权，噪声小（σ 小）自动升权——不用手调「两个任务的 loss 各乘多少」。这和 §3.4 的「每个样本自动加权」是同一件事的两个层次（样本级 vs 任务级）。R torch 里把 $\log\sigma_k$ 当成普通可学习参数即可：

```r
library(torch)
set.seed(0); n <- 500
x <- runif(n, -2, 2)
y1 <- 2*sin(x) + rnorm(n, sd = 0.3)   # 任务1：噪声小
y2 <- 0.5*x^2 + rnorm(n, sd = 1.5)    # 任务2：噪声大（≈28×）
x_t <- torch_tensor(x)$unsqueeze(2)
y1_t <- torch_tensor(y1)$unsqueeze(2)
y2_t <- torch_tensor(y2)$unsqueeze(2)

net <- nn_module(
  initialize = function(d = 32){
    self$fc <- nn_sequential(nn_linear(1,d), nn_relu(), nn_linear(d,d), nn_relu())
    self$h1 <- nn_linear(d,1); self$h2 <- nn_linear(d,1)
  },
  forward = function(x){ h <- self$fc(x); list(y1 = self$h1(h), y2 = self$h2(h)) }
)
model <- net()
ls1 <- nn_parameter(torch_tensor(0)); 
ls2 <- nn_parameter(torch_tensor(0))   # log σ
opt <- optim_adam(c(model$parameters, ls1, ls2), lr = 1e-2)

for (ep in 1:300) {
  opt$zero_grad(); out <- model(x_t)
  s1 <- torch_exp(ls1); s2 <- torch_exp(ls2)
  L1 <- (y1_t - out$y1)$pow(2)$mean(); L2 <- (y2_t - out$y2)$pow(2)$mean()
  loss <- L1/(2*s1^2) + torch_log(s1) + L2/(2*s2^2) + torch_log(s2)
  loss$backward(); opt$step()
}
```

训练后 σ 自动学会反映任务噪声：

```r
o <- model(x_t)
exp(c(log_sigma1 = as.numeric(ls1), log_sigma2 = as.numeric(ls2)))
#  σ1 ≈ 0.29（任务1 噪声小 → 权重 1/σ² 大）
#  σ2 ≈ 1.55（任务2 噪声大 → 自动降权）

# 加权后的两项贡献（应被拉到同一量级）：
c(t1 = as.numeric((y1_t-o$y1)$pow(2)$mean()/(2*torch_exp(ls1)^2)),
  t2 = as.numeric((y2_t-o$y2)$pow(2)$mean()/(2*torch_exp(ls2)^2)))
#  ≈ 0.50 / 0.50
```

即使任务 2 的原始 MSE 约是任务 1 的 28 倍，σ 加权后两项对梯度的贡献被拉到几乎相等——这就是「不确定性自动加权」的价值：省掉手工调 loss 权重，模型自己找到平衡。

### 9.6 Conformal Prediction：不靠分布假设也能给出覆盖率

§9.4 的校准是「事后检查」；保形预测 (conformal prediction) 则直接构造有覆盖率保证的预测区间，而且不需要分布假设——只用数据。最常用的分裂保形 (split conformal)：

1. 把数据分成训练集 + 校准集 + 测试集；用训练集学出预测器 $\hat\mu(x)$（及 σ 等）。
2. 在校准集上算**非一致性分数** $s_i = |y_i - \hat\mu(x_i)|$。
3. 取校准分数在水平 $1-\alpha$ 的分位数 $\hat q = \mathrm{Quantile}\big(s, (1-\alpha)(1+1/n_c)\big)$。
4. 测试点的区间：$[\hat\mu(x)-\hat q,\ \hat\mu(x)+\hat q]$。

保证（在交换性假设下）：区间以 ≥ $1-\alpha$ 的边际概率覆盖真值——不依赖误差分布、也不依赖模型正确。这正是 §9.4 想要的「可靠校准」，但由构造保证。

衔接 §9 的 σ：更聪明的做法用学生化分数 $s = |y-\mu|/\sigma$——把 §9.3 学到的异方差 σ 卷进分数，让区间局部自适应（噪声大的区域自动更宽）。复用 §12.3 的异方差网络：

```r
library(torch)
set.seed(2024); torch_manual_seed(2024)          # 同时固定 R 与 torch 的随机源
n <- 1000; nx <- 600; nc <- 200
xx <- runif(n, -3, 3); yy <- sin(xx) + (0.2 + 0.5*abs(xx))*rnorm(n)
x_t <- torch_tensor(xx)[, NULL]; y_t <- torch_tensor(yy)[, NULL]

# 异方差网络（结构同 §12.3），只用训练集训练
het <- nn_module(
  initialize = function(d = 32){
    self$fc <- nn_sequential(nn_linear(1,d), nn_relu(), nn_linear(d,d), nn_relu())
    self$mu <- nn_linear(d,1); self$lv <- nn_linear(d,1)
  },
  forward = function(x){
    h <- self$fc(x); 
    list(mu = self$mu(h), log_var = self$lv(h))
    }
)
m <- het(); 
opt <- optim_adam(m$parameters, lr = 1e-2);
itrain <- 1:nx
for (ep in 1:150) {
  opt$zero_grad(); out <- m(x_t[itrain,]);
  var <- torch_exp(out$log_var)
  loss <- ((y_t[itrain,] - out$mu)^2/var + out$log_var)$mean()/2
  loss$backward(); opt$step()
}

ical <- nx + 1:nc; itest <- (nx+nc+1):n
oc <- m(x_t[ical,]); mu_c <- as.numeric(oc$mu); 
sg_c <- as.numeric(torch_exp(oc$log_var/2))
s  <- abs(yy[ical] - mu_c)/sg_c                       # 学生化分数
q  <- as.numeric(quantile(s, 0.9*(1 + 1/nc)))         # α = 0.1，有限样本校正
ot <- m(x_t[itest,]); mu_t <- as.numeric(ot$mu); 
sg_t <- as.numeric(torch_exp(ot$log_var/2))
mean(abs(yy[itest] - mu_t) <= q*sg_t)                 # 经验覆盖率
```

实际运行：

```
q ≈ 1.73       经验覆盖率 = 0.90（目标 0.9）
```

覆盖率精确落在 0.9 目标上——而且这是无分布假设地做到的，区间宽度还随 σ 自动伸缩。（注：单次分裂的覆盖率会有波动；`set.seed` 固定 R 的数据源、`torch_manual_seed` 固定网络的初始化，上面的结果可复现。）这也是为什么「NLL 学出 σ」之后 conformal 是最自然的下一步：σ 把分布信息压缩进分数，conformal 把分数转成可验证的覆盖率保证。

---

## 10. 鲁棒性与替代分布

### 10.1 学生 t 分布：重尾鲁棒回归

当数据含大量离群点时，高斯假设会崩溃。改用学生 t 分布（自由度 $\nu$ 控制尾巴轻重，$\nu\to\infty$ 退化为高斯）：

$$
p(y) \propto \left(1+\frac{(y-\mu)^2}{\nu}\right)^{-(\nu+1)/2}
$$

其 NLL 对离群点的惩罚是对数级而非平方级，鲁棒得多。可学习的 $\nu$ 让模型自己决定尾巴多重。

### 10.2 混合密度网络 (MDN)：多模态输出

当条件分布是多峰的（如驾驶场景中「向左避让 or 向右避让」），单个高斯无能为力。高斯混合 NLL：

$$
p(y|x) = \sum_{k=1}^K \pi_k(x)\, \mathcal{N}\big(y;\, \mu_k(x), \sigma_k^2(x)\big)
$$

网络输出 $\{\pi_k, \mu_k, \sigma_k\}_{k=1}^K$，目标是高斯混合的 NLL。这是 Gaussian NLL 的自然推广，也是多模态预测的标准工具。

### 10.3 贝叶斯先验里的高斯

- 高斯先验 + 高斯似然 = 解析可解的高斯后验（共轭），这构成了贝叶斯线性回归、卡尔曼滤波的基础。
- 高斯先验的 MAP = L2 正则，这在前文已述。

---

## 11. 分类里的 NLL 变体与取舍

§1.3 论证了「严格正确评分 → 最小化 NLL 唯一地奖励诚实分布」。但现代分类工程经常故意偏离这个理想——不是不知道诚实的好处，而是为了换校准、类别平衡、或训练稳定。下面三个是流传最广的「被改过的 NLL」。

### 11.1 Label Smoothing：把目标软化

交叉熵 $\mathrm{CE} = -\sum_k y_k\log p_k$ 用 one-hot 目标 $y$。标签平滑把它换成软目标 $t = (1-\varepsilon)y + \varepsilon u$（$u$ 常取均匀分布），于是：

$$
-\sum_k t_k\log p_k = (1-\varepsilon)\,\mathrm{CE} + \varepsilon\,\mathrm{KL}(u \| p) + \text{const}
$$

分解揭示了本质：label smoothing 就是在 CE 上加了「把 $p$ 拉向均匀分布」的熵正则——抑制极端自信，防止模型对（可能有噪声的）训练标签过度自信。代价是它不再是严格正确评分，但它常换来更好的校准和泛化（Szegedy et al. 2016；Müller et al. 2019）。恒等式随手可验：

```r
softmax <- function(z) exp(z - max(z))/sum(exp(z - max(z)))
z <- c(2.5, 1.0, 0.2); p <- softmax(z); K <- length(p)
eps <- 0.1; y <- c(1, 0, 0); t <- (1-eps)*y + eps/K
-sum(t*log(p))                              # 平滑后的 CE ≈ 0.4069
(1-eps)*(-sum(y*log(p))) + eps*(-sum(log(p)/K))  # 右式 = 同一数（差 < 1e-15）
```

### 11.2 Focal Loss：为类别不平衡重加权

类别严重不平衡时（如目标检测的「背景 vs 前景」），CE 让大量「简单负例」主导梯度。Focal Loss（Lin et al. 2017）按置信度调制：

$$
\mathrm{FL}(p_t) = -(1-p_t)^\gamma \log p_t, \qquad p_t = p(\text{真实类别})
$$

- 样本越是「已分类正确且自信」（$p_t \to 1$），调制因子 $(1-p_t)^\gamma \to 0$，贡献被压下去；困难样本保留梯度。
- 代价：它故意不是严格正确评分（对正确但自信的样本另有偏好）。这是「为了训练效果，接受理论上的不诚实」的经典例子——因此 FL 的数值不能当概率模型的评估指标用（那是 NLL / CRPS 的活）。

### 11.3 Temperature Scaling：事后校准

温度缩放（Guo et al. 2017）是最简单的校准修复：把 logits 除以温度 $T$ 再 softmax，$T$ 在校验集上最小化 NLL：

$$
p_T(y|x) = \mathrm{softmax}(z/T), \qquad T^* = \arg\min_T \mathrm{NLL}_{\text{val}}(T)
$$

它只改置信度、不改预测类别（argmax 不变）。R 里用 `optimize()` 一行就能做：

```r
set.seed(3)
n <- 1000
x3 <- rnorm(n); lp <- 0.3 + 1.1*x3
label <- rbinom(n, 1, plogis(lp))
z_logit <- 1.6*(0.3 + 1.1*x3)                # 模拟过度自信模型：logit 被放大 1.6×
pm <- plogis(z_logit)

Tnll <- function(T){ 
  pp <- plogis(z_logit/T); 
  -mean(label*log(pp)+(1-label)*log(1-pp)) 
  }
Topt <- optimize(Tnll, c(0.05, 10))$minimum  # T ≈ 1.75 > 1：把过度自信拉回来
pm_T <- plogis(z_logit/Topt)

# ECE（期望校准误差）：分箱后 |平均置信度 − 实际频率| 的加权平均
ece <- function(prob, lab){
  b <- findInterval(prob, seq(0, 1, 0.1), rightmost.closed = TRUE)
  nk <- tabulate(b, nbins = 10); ok <- nk > 0
  acc <- as.numeric(tapply(lab, b, mean)); 
  conf <- as.numeric(tapply(prob, b, mean))
  sum((nk/sum(nk))[ok] * abs(acc - conf))
}
c(T_best     = round(Topt, 3),
  NLL_before = round(-mean(label*log(pm)   + (1-label)*log(1-pm)),   4),
  NLL_after  = round(-mean(label*log(pm_T) + (1-label)*log(1-pm_T)), 4),
  ECE_before = round(ece(pm,   label), 4),
  ECE_after  = round(ece(pm_T, label), 4))
```

实际输出：

```
T_best NLL_before NLL_after ECE_before ECE_after
1.751     0.6278    0.5932     0.0941    0.0353
```

温度 1.75 > 1 说明原模型过度自信；缩温后 NLL 和 ECE（校准）双双改善——代价只是学一个 $T$，因为真实分布并不支持那种极端自信。

### 11.4 何时值得「放弃」严格正确性

| 情况 | 做法 | 牺牲 | 换来 |
|---|---|---|---|
| 标签有噪声 | label smoothing | 严格正确性 | 校准、泛化 |
| 类别极不平衡 | focal loss | 严格正确性 | 困难样本学习、平衡梯度 |
| 模型过度自信 | temperature scaling | 无（仍是 NLL，只是多一个 $T$） | 校准 |
| 只需好均值 | MSE（固定 σ） | 不确定性表达 | 简单、稳定 |

一句话：严格正确性保证的是诚实模型在期望上最优，但真实工程还想要校准、平衡、稳定性这些「期望之外」的品质——所以人们会带着对代价的清醒去修改 NLL。而评估时，回到未修改的 NLL / CRPS（§8），才能和「诚实」对标。

---

## 12. 数值实现要点

### 12.1 参数化

永远不要直接输出 $\sigma^2$ 或 $\sigma$，否则训练中可能输出负值导致 NaN。标准做法是输出 $\log \sigma^2$（或 $\log\sigma$），再指数还原：

```r
log_var <- net(x)$log_var            # 网络输出 log σ²
var <- torch_exp(log_var)            # σ² > 0 恒成立
```

### 12.2 稳定实现

一个经典 bug：`(y-mu)**2 / var` 在 `var` 很小时爆炸；`log(var)` 在 var→0 时 → -∞。稳定写法（PyTorch 的 `GaussianNLLLoss` 内部就是这么做的）：

```r
gaussian_nll <- function(y, mu, log_var, full = FALSE) {
  var  <- torch_exp(log_var)           # 用 log-domain 运算避免除法和下溢
  loss <- (y - mu)^2 / var + log_var
  if (full) loss <- loss + log(2 * pi) # full = TRUE 时保留常数项
  0.5 * loss$mean()
}
```

注意：R 的 `torch` 包虽然镜像了 PyTorch 的 `nn_nll_loss`、`nn_poisson_nll_loss` 等，但至今没有内置 `nn_gaussian_nll_loss`（截至 0.17 版）。所以上面的手动函数不是备选，而是标准做法——它等价于 PyTorch 里的 `nn.GaussianNLLLoss(full = FALSE)`（即丢弃 $0.5\log(2\pi)$）。若你在 PyTorch 侧写，对应接口接收 `input`（即 $\mu$）、`target`（即 $y$）、`var`（即 $\sigma^2$，**注意不是 log-var**，且要求 >0）。

### 12.3 完整异方差回归示例

```r
library(torch)

# ---- 数据：异方差噪声的真实生成过程 ----
set.seed(42)
n <- 2000
x <- runif(n, -3, 3)
mu_true <- sin(x)                                 # 均值函数
sd_true <- 0.2 + 0.5 * abs(x)                     # 方差随 |x| 增大（异方差）
y <- mu_true + rnorm(n, mean = 0, sd = sd_true)

x_t <- torch_tensor(x, dtype = torch_float32())[, NULL]  # (n, 1)
y_t <- torch_tensor(y, dtype = torch_float32())[, NULL]

# ---- 网络：同时输出 μ 和 log σ² ----
net <- nn_module(
  initialize = function(d_hidden = 64) {
    self$fc <- nn_sequential(
      nn_linear(1, d_hidden), nn_relu(),
      nn_linear(d_hidden, d_hidden), nn_relu()
    )
    self$mu_head     <- nn_linear(d_hidden, 1)
    self$logvar_head <- nn_linear(d_hidden, 1)
  },
  forward = function(x) {
    h <- self$fc(x)
    list(mu = self$mu_head(h), log_var = self$logvar_head(h))  # 无激活，自由学习
  }
)

model <- net()
opt <- optim_adam(model$parameters, lr = 1e-2)

# ---- 训练：最小化 Gaussian NLL ----
for (epoch in 1:500) {
  opt$zero_grad()
  out <- model(x_t)
  var <- torch_exp(out$log_var)
  loss <- ((y_t - out$mu)^2 / var + out$log_var)$mean() / 2 # 与 1.4 节同一目标函数
  loss$backward()
  opt$step()
  if (epoch %% 100 == 0)
    cat(sprintf("epoch %3d   loss = %.4f\n", epoch, loss$item()))
}

# ---- 推理：每个 x 的预测 ± 1σ 区间（σ 直接来自网络）----
out <- model(x_t)
mu <- as.numeric(out$mu)
sigma <- as.numeric(torch_exp(out$log_var / 2))           # σ = sqrt(exp(log_var))
```

训练后可以用可靠性图检查校准（见 9.4）：把预测区间画出来，会看到 $\sigma$ 在 $\lvert x \rvert$ 大的地方自动变大——模型学到了「这里就是不可预测」。

### 12.4 几个实操提醒

- 训练初期 $\sigma$ 很大、loss 主要由 log 项主导，属正常现象；若收敛后 $\sigma$ 仍异常大，检查是否过度平滑。
- 多任务/多输出时，异方差 NLL 天然给出每个任务的自动权重（同分布族的优雅性质），可用来替代手动调 loss 权重。
- 做分布外检测或主动学习时，把 $\sigma$ 当作置信度是标准 trick。

---

## 13. 何时用哪个 loss：决策指南

| 你的情况 | 推荐 loss | 理由 |
|---|---|---|
| 同方差高斯噪声、只要好均值 | MSE | 简单、快、等价于固定方差的 NLL |
| 需要输出「预测值 + 可信区间」 | 异方差 Gaussian NLL | 学会输入依赖的不确定性 |
| 数据含大量离群点 | Huber / MAE / Student-t NLL | 重尾分布对离群点鲁棒 |
| 多模态输出（一条路多个可能结果） | 混合密度网络 (GMM NLL) | 单高斯表达不了多峰 |
| 分类问题 | 交叉熵 (Bernoulli/Categorical NLL) | 就是分类的 NLL |
| 计数值 | Poisson NLL | 计数数据专用 |
| 生成模型 | VAE ELBO / 流 NLL / 扩散 MSE 上界 | 底层全是「最大化对数似然」 |

---

## 结语：一条主线

把全文串起来，是一句话：

> Gaussian NLL 是「假设条件分布为高斯」时的负对数似然 loss。固定方差就退化为 MSE，学出方差就获得不确定性；换成别的分布假设就得到 MAE、Huber、交叉熵等其他 loss；把它当作似然目标嵌入潜变量模型就长出 VAE、流、扩散模型；而它同时还是拿来评估这些模型的标尺——perplexity、bits/dim 都是它的指数化或归一化形态。理解了「loss = 分布假设下的 NLL」这一个等式，Gaussian NLL 就不是一个孤立的函数，而是整个似然建模版图的枢纽。

---

## 参考文献与延伸阅读

**核心概念与统计基础**
- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2013（VAE / ELBO）
- Nix & Weigend, *Estimating the mean and variance of the target probability distribution*, 1994（异方差回归）
- Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, 2007（为什么 NLL 是诚实的评分规则）
- Gneiting, Raftery, Westveld & Goldman, *Calibrated Probabilistic Forecasting Using Ensemble MOS and Minimum CRPS Estimation*, 2005（CRPS）

**不确定性估计与多任务**
- Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*, 2017（aleatoric/epistemic，异方差 NLL 实证）
- Kendall, Gal & Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, 2018（任务级 σ 自动加权）
- Lakshminarayanan et al., *Deep Ensembles: A Loss Landscape Perspective*, 2017（不确定性聚合）
- Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, 2005（保形预测）
- Papadopoulos et al., *Inductive Confidence Machines for Regression*, 2002（学生化/归一化分数的保形预测）

**生成模型与鲁棒损失**
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, 2020（扩散模型的高斯 NLL 推导）
- Bishop, *Mixture Density Networks*, 1994（GMM NLL）
- Barron, *A General and Adaptive Robust Loss Function*, CVPR 2019（α 一族统一鲁棒 loss）

**分类里的 NLL 修改**
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, 2016（label smoothing）
- Müller, Kornblith & Hinton, *When Does Label Smoothing Help?*, 2019（label smoothing 的代价与收益）
- Lin, Goyal, Girshick, He & Dollár, *Focal Loss for Dense Object Detection*, 2017（focal loss）
- Guo, Pleiss, Sun & Weinberger, *On Calibration of Modern Neural Networks*, 2017（temperature scaling）
