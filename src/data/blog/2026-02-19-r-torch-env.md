---
title: R torch 环境安装
author: 思喆
pubDatetime: 2026-02-19T09:06:31Z
slug: r-torch-env
featured: false
draft: false
tags:
  - torch
description:
  在 Ubuntu 24.04 与 RTX 4090 服务器上从零配置 R 语言 Torch GPU 环境的完整记录
---


本文记录了在云服务器上从零配置 R 语言 Torch 深度学习环境的完整过程。主要涵盖底层显卡驱动安装、R 语言最新版安装以及支持 GPU 加速的 Torch 包配置。

## 服务器环境说明

- 云服务商：UCloud (ucloud.cn)
- 显卡：NVIDIA GeForce RTX 4090
- 操作系统：Ubuntu 24.04 (64位)
- 初始状态：纯净重装系统，未安装任何显卡驱动

> 提示：如果是重装了操作系统的云服务器，由于 IP 地址未变但服务器指纹已更新，本地 SSH 连接时可能会报错。在本地终端运行以下命令清除旧密钥（请将 IP 替换为您自己的服务器 IP）：
> ```bash
> ssh-keygen -R 117.50.46.224
> ```

---

# 一、 安装系统底层显卡驱动

Ubuntu 24.04 初始不带 Nvidia 专有驱动。我们需要手动安装适用于服务器的稳定版驱动（此处选择 `535-server` 分支）。

1. 执行以下命令安装驱动并重启服务器：

```bash
sudo apt install nvidia-driver-535-server
sudo reboot
```

2. 重启后，输入 nvidia-smi 验证驱动是否安装成功：

```bash
nvidia-smi
```

如果输出类似如下的显卡状态面板，说明底层驱动已就绪：

```
ubuntu@10-60-170-244:~$ nvidia-smi
Fri Feb 20 14:46:29 2026
+---------------------------------------------------------------------------------+
| NVIDIA-SMI 535.247.01       Driver Version: 535.247.01   CUDA Version: 12.2     |
|-----------------------------------+----------------------+----------------------+
| GPU  Name           Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf    Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                   |                      |               MIG M. |
|===================================+======================+======================|
|  0  NVIDIA GeForce RTX 4090  Off  | 00000000:00:03.0 Off |                  Off |
| 30%   31C    P0        47W / 450W |      0MiB / 24564MiB |      0%      Default |
|                                   |                      |                  N/A |
+-----------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------+
| Processes:                                                                      |
|  GPU   GI   CI      PID   Type   Process name                        GPU Memory |
|        ID   ID                                                       Usage      |
|=================================================================================|
|  No running processes found                                                     |
+---------------------------------------------------------------------------------+
```

(注：面板显示的 CUDA Version: 12.2 是该驱动支持的最高基础 API 版本，后续我们在 R 中安装的 CUDA 12.8 运行库自带完整的执行环境，向下兼容该驱动，互不冲突。)

# 二、 安装最新版 R 语言

系统默认的 APT 软件源中，R 的版本通常较旧。为了确保 Torch 的兼容性，我们需要添加 CRAN 的官方源来安装最新版本的 R。

依次在终端执行以下命令：

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common dirmngr wget apt-transport-https
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
sudo apt-get update
sudo apt-get install -y r-base r-base-dev
```

# 三、安装 R Torch (含 GPU 支持)

R 的 torch 包安装分为两步：首先在 R 环境中安装基础包（默认仅支持 CPU），然后再通过指定镜像源拉取包含 CUDA 支持的底层 C++ 二进制库。

1. 安装基础版 (CPU)
进入 R 的交互式命令行，执行标准安装：

```r
install.packages("torch")
library(torch)
```

此时安装完成的 R torch 为最新的 0.16.3 版本。

2. 下载并配置 GPU 依赖 (CUDA 12.8)

默认的 CPU 版本无法调用 RTX 4090 的算力。我们需要重新运行安装命令，并指定 cu128 (CUDA 12.8) 版本的二进制文件。

重要提醒：由于包含 CUDA 的二进制文件高达 2GB，R 默认的 60 秒下载超时限制极易导致下载失败。必须先调大 `timeout` 参数。

在 R 中继续执行：

```r
options(timeout = 6000)
# For Windows and Linux: "cpu", "cu128" are the only currently supported
# For MacOS the supported are: "cpu-intel" or "cpu-m1"
kind <- "cu128"
version <- available.packages()["torch","Version"]
options(repos = c(
  torch = sprintf("https://torch-cdn.mlverse.org/packages/%s/%s/", kind, version),
  CRAN = "https://cloud.r-project.org" 
))
install.packages("torch")
```

# 四、 验证环境

安装结束后，最后一步是确认 R 是否能够成功识别并调用显卡。

在 R 终端中输入：

```r
library(torch)
cuda_is_available()
```

如果返回结果为 TRUE，则恭喜你，R 语言的 GPU 深度学习环境已配置完毕。

```r
library(torch)
library(bench)

N <- 5000

# 1. R 原生矩阵 (默认是 Float64/Double)
matrix_r <- matrix(runif(N * N), nrow = N, ncol = N)

# 2. CPU 张量 (强制转为 Float32，模拟深度学习标准环境)
tensor_cpu <- torch_tensor(matrix_r, dtype = torch_float32())

# 3. 检查 CUDA 并创建 GPU 张量
if (!cuda_is_available()) {
  stop("CUDA 未就绪，请检查环境。")
}
tensor_cuda <- tensor_cpu$to(device = torch_device("cuda"))

# 4. 运行基准测试
results <- mark(
  Torch_CPU = torch_matmul(tensor_cpu, tensor_cpu),
  # 同步计算：强制 CPU 等待 GPU 计算完成（反映真实耗时）
  Torch_CUDA_Sync = {
    ans <- torch_matmul(tensor_cuda, tensor_cuda) # 补充了 ans 赋值
    ans[1, 1]$item() # 强制数据回传触发显式同步
  },

  iterations = 200,
  check = FALSE,  # R原生矩阵和torch张量精度不同故关闭检查
  memory = FALSE
)
```

T882Nd9kUZswmhhy

返回结果：

```r
coln <- c("expression", "min", "median", "itr/sec", "n_itr", "total_time")
results[, coln]
## A tibble: 2 × 6
#  expression           min   median `itr/sec` n_itr total_time
#  <bch:expr>      <bch:tm> <bch:tm>     <dbl> <int>   <bch:tm>
#1 Torch_CPU       317.65ms 336.25ms      2.87   200      1.16m
#2 Torch_CUDA_Sync   4.86ms   5.19ms    125.     200       1.6s
```

这就证明您的矩阵乘法是在 RTX 4090 上计算完成的。至此，您的深度学习环境已经彻底搭建完毕，可以开始跑大模型和复杂网络了！