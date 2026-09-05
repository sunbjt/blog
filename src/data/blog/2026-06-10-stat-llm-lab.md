---
title: 大语言模型的统计实验室
author: 思喆
pubDatetime: 2026-06-11T08:02:31Z
featured: true
draft: true
tags:
  - 大模型
description: 大语言模型统计实验室
---

1. BPE 分词
2. 传统意义的 LLM，包括 

在 NVIDIA 3090 24G 服务器上，约 17 分钟一个 epoch，加之微调，约 1 小时即可完整复现 LLM。

语料 120 万行，约 130m token，按照 DeepMind Chinchilla 定律，

> 每 1 个模型参数，大约需要配 20 个文本 token 的训练数据

因此最终模型参数规模：

- CausalLM: 10.49 M
- JepaVQ: 10.62 M

虽然已经将数据二进制化，但 IO 依然是 R 和 torch 间的瓶颈。

R torch 的劣势：出于线程安全和函数式编程的考虑，R 在 CPU 和 GPU 之间拷贝是瓶颈。
在这样密集 IO 交互的场景下，Pytorch 可以高速运行线程级数据拷贝。
3090 的 GPU 可以被打满，Pytorch 效率是 R torch 的 2-3 倍。
