---
title: "调用本地 Gemma 4 提取结构化数据"
author: 思喆
pubDatetime: 2026-04-03T09:06:31Z
slug: gemma4
featured: false
draft: false
tags:
  - 大模型
description:
  Google 开源的 Gemma 4 对本地部署非常友好。本文记录了如何在 Mac M4 16G 环境下运行 e4b 模型，并通过 R 语言的 httr2 包实现极速的结构化 JSON 数据提取。
---

昨天 Google 发布了开源大预言模型 Gemma 4 的系列模型。
这次开源协议有所变更，对企业应用更加友好，Google 显然也有其在开源生态上的目的。
但对普通开发者来说，最直观的利好是本地部署的门槛进一步降低了。

以我手头的 Mac M4 16G 版本为例，本地直接跑 e4b（40 亿参数）尺寸的小模型毫无压力，日常用它来执行一些简单的 NLP 任务刚好够用。

在终端通过 Ollama 启动服务之后，我们可以直接在 R 中进行调用。
不需要安装复杂的深度学习环境，只用原生的 httr2 包就能完成请求，并让模型输出规范的 JSON 数据。


```r
library(httr2)
ask_gemma_json <- function(prompt) {
  req <- request("http://localhost:11434/api/generate") |>
    req_body_json(list(
      model = "gemma4:e4b", 
      prompt = prompt,
      stream = FALSE,
      format = "json", # 强迫症福音：只允许输出合法的 JSON 格式
      options = list(temperature = 0.0)
    ))
  
  resp <- req_perform(req)
  
  # 因为返回的是标准 JSON 字符串，我们可以直接用 jsonlite 把它变成 R 的 List
  json_str <- resp_body_json(resp)$response
  jsonlite::fromJSON(json_str)
}

# 测试结构化提取
prompt_json <- "提取这句话中的公司名和情感得分(1-10分)，并严格以JSON格式输出，键名为 company 和 score。句子：苹果公司本季度营收大涨，超出华尔街预期。"

structured_data <- ask_gemma_json(prompt_json)
# > structured_data
# $company
# [1] "苹果公司"
#
# $score
# [1] 8
```

在上面的 API 调用中，能够实现快速响应，主要归功于以下两点参数控制：

- 利用 `temperature = 0.0` 让模型每次预测下一个词时只选概率最高的那个，直接跳过了概率采样计算的耗时，同时也保证了输出的确定性。
- 结合明确的 Prompt 和 JSON 格式锁，切断了模型生成“思考”和“分析”等中间 Token 的链条。原本可能要生成 200 个废话词汇（需要 10 秒），现在被强行压缩到只生成 5 个有效结果词汇（不到 1 秒）。