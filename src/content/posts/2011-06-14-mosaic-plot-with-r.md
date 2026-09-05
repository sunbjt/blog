---
title: 用 R 实现马赛克拼图 (更新版)
tags:
  - 不务正业的 R
pubDatetime: 2011-06-13T16:00:00.000Z
updateDatetime: 2024-05-20T10:00:00.000Z
author: 思喆
description: "三天前，在统计之都论坛上问到了如何做 Matrix67 博客上的平滑马赛克图，我是好事之徒，颠颠地跑去瞧了一眼。多年后，原来的包早没啦，重新用 magick 和矩阵黑魔法重构一版。"
---

三天前，在统计之都论坛上问到了如何做 [Matrix67](http://www.matrix67.com/blog/archives/519) 博客上的平滑马赛克图，我是好事之徒，颠颠地跑去瞧了一眼。

恩，蛮有意思的，而且非常黄，非常暴力！但比较悲剧的是我不会用 Mathematica，只好用 R 实现了一下。

本来标题党一些，叫做《一千二百个女人和我的故事》，想想还是算了吧，虽说是用了 1200 个漂亮女人组成了我的头像，但我一个也不认识，哈哈。


<img src="/upload/pic/me.png" alt="me" width="40%" />


用的原图我就不贴了，实际上我是戴着眼镜的，马赛克平滑以后，不明显了。当年初版的代码非常简单，不到 20 行，甚至丧心病狂地用了 `sqldf` 写 SQL 语句来对像素点排序，然后强行用 1200 次 `plot()` 循环画出来。


# 爷青回：多年后的代码重构

时光荏苒，当年用的图床会挂，古老的 `ReadImages` 包早就在 CRAN 上作古了，而且当年调系统 shell 用 ImageMagick 处理图片的姿势现在看来也确实不太优雅。

于是，我决定把这个“不务正业”的脚本翻新一下。

这次更新的几个重点：

1. 素材升级：把当年的“小姐姐”换成了现在的“喵星人”（使用了 afhq 猫咪数据集中的几千张照片），毕竟互联网的尽头是吸猫。
2. 工具换代：全面拥抱现代的 `magick`、`jpeg` 和 `png` 包。
3. 算法降维打击：抛弃了极其低效且容易导致白屏的 `par(mfcol)` 多图拼凑法，直接采用**底层的矩阵拼接大法**。把图片看作矩阵，像贴瓷砖一样把小图塞进大矩阵的对应位置，速度快了几十倍，而且做到了真正的像素级无缝！

下面是船新版本的实现步骤。

## 第一步：素材库准备（彩色转灰度并统一切图）

我们需要先把准备好的猫咪图片统一缩放并转为灰度图，存入一个专门的文件夹，作为我们的“马赛克瓷砖库”。

```r
library(magick)

# 设置输入和输出文件夹路径
input_dir <- "/Users/liusizhe/data/afhq/train/cat"   # 原始彩色图片文件夹
output_dir <- "/Users/liusizhe/data/afhq/cat_gray"   # 灰度瓷砖库文件夹

# 如果输出文件夹不存在，就创建它
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# 获取文件夹中的所有图片文件（常见格式）
files <- list.files(input_dir, pattern = "\\.(jpg|jpeg|png)$", full.names = TRUE)

# 批量转换为灰度并压缩到 50x50（作为底库备用）
for (f in files) {
  img <- image_read(f)         
     # 获取原始尺寸
     info <- image_info(img)
     w <- info$width
     h <- info$height
     
     # 裁剪掉四周边缘 70 像素，尽量展现的是猫的可爱圆脸
     margin <- 70
     img <- image_crop(img, 
     geometry = paste0(w-2*margin, "x", h-2*margin, "+", margin, "+", margin))
  img <- image_scale(img, "50x50")                      
  gray_img <- image_quantize(img, colorspace = "gray")  
  out_path <- file.path(output_dir, basename(f))
  image_write(gray_img, path = out_path)  
}

cat("所有图片已转换为灰度并保存到:", output_dir, "\n")
```


## 第二步：核心引擎（计算亮度、匹配并极速拼接矩阵）

这一步是灵魂。我们会读取目标图片（我的头像，缩小到 50x62 像素，刚好能够分辨人脸），计算每个像素的亮度（灰度值）。然后计算素材库里每张猫图的平均亮度。最后，用最近邻的方法给头像的每一个像素匹配一张亮度最接近的猫图，并用大矩阵一次性输出。

```r
library(jpeg)
library(magick)
library(png) 

# --- 1. 路径与图片读取 ---
output_dir <- "/Users/liusizhe/data/afhq/cat_gray"
source_file <- "/Users/liusizhe/ideas/image/me/grey.jpg"
output_file <- "me.png"

me <- readJPEG(source_file)
nr <- nrow(me)
nc <- ncol(me)

# --- 2. 计算素材库的平均灰度值 ---
files <- list.files(output_dir, pattern = "\\.(jpg|jpeg|png)$", full.names = TRUE)
tmp <- lapply(files, function(f) {
  img <- image_read(f)
})
names(tmp) <- basename(files)

# 计算每张小图的均值，转为 [0,1] 区间的数值
means <- sapply(tmp, function(img) {
  arr <- image_data(img)[1,,]
  mean(as.numeric(arr) / 255)
})
素材表 <- data.frame(n = names(tmp), m = means, stringsAsFactors = FALSE)

# --- 3. 最近邻匹配 ---
# 为源图的每一个像素寻找亮度最相近的猫咪图片
image_order <- sapply(as.vector(me), function(v) {
  素材表$n[which.min(abs(素材表$m - v))]
})

# --- 4. 矩阵拼接大魔法 ---
# 初始化一张巨大的空白矩阵，尺寸为 (原图行数*50) x (原图列数*50)
big_matrix <- matrix(0, nrow = nr * 50, ncol = nc * 50)
order_matrix <- matrix(image_order, nrow = nr, ncol = nc)

# 像贴瓷砖一样，将素材矩阵贴到大矩阵对应的位置
for (r in 1:nr) {
  for (c in 1:nc) {
    img_name <- order_matrix[r, c]
    
    # 提取素材灰度矩阵并归一化
    arr <- image_data(tmp[[img_name]])[1,,]
    arr <- as.numeric(arr) / 255
    dim(arr) <- c(50, 50) 
    
    # 注意：magick 导出的矩阵是 (宽, 高)，在 R 中需要转置一下变为 (高, 宽)
    block <- t(arr)
    
    # 计算当前这块瓷砖在大图中的索引位置
    row_idx <- ((r - 1) * 50 + 1):(r * 50)
    col_idx <- ((c - 1) * 50 + 1):(c * 50)
    
    # 严丝合缝地贴进去！
    big_matrix[row_idx, col_idx] <- block
  }
}

# 直接将大矩阵写出为 PNG 图片
writePNG(big_matrix, output_file)
cat("马赛克图已极速生成：", output_file, "\n")
```

旧时代的 plot 一去不复返，现在我们拥有了真正高效的像素艺术生成器。如果你也有想要拼起来的回忆，赶紧去试试吧！

