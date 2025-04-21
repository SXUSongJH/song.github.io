---
title: 生成模型简介
tags:
    - Machine Learning
    - Generative Model
---  

## toc

&emsp;&emsp;在过去三年里，人工智能领域发生了翻天覆地的变化，革命性的产品层出不穷，语言生成模型ChatGPT、Llama、Claude; 图片生成模型Midjourney、stable diffusion、DALL; 视频生成模型Sora。这些人工智能产品带给了世界前所未有的变化，深深影响了人们的生活，英伟达CEO黄仁勋在2023年的GTC大会上直言：“现在是AI的iphone时刻”。在这一轮人工智能狂潮中，以 Diffusion Model 为代表的生成模型扮演了至关重要的角色，stable diffusion、DALL、Sora等产品的底层模型便是 Diffusion Model，其对当今 AIGC 的发展起到了不可忽视的推动作用。  
&emsp;&emsp;早在今年的2月份，OpenAI发布Sora的演示视频时，笔者便想创作博客来介绍其底层的 Diffusion Model，但思来想去，还是认为自己的理解过于浅显，对于很多地方的 motivation 并没有比较清晰的认识，生成模型是笔者的研究方向之一，因此我并不想囫囵吞枣，浮于表面，遂埋头继续钻研。直到上个月，我阅读了 Google Brain 的 Calvin Luo 创作的介绍 Diffusion Model 原理的文章[1]，这篇文章从一个统一的视角介绍了VAE、Diffusion Model、Score-Based Model，解答了笔者心中的很多疑问，并将我过去半年在生成模型上的认识很好地串联了起来。在完成了对于所学知识的整合后，笔者终于有信心来开始生成模型这个系列。  

## 数据分布假设  

&emsp;&emsp;在生成模型中，我们通常假设数据点 $\boldsymbol{x}$ 来自于一个真实的分布 $p_{true}(x)$，生成模型的目的就是通过训练数据集 $T_{train}$ 去学习 $p_{true}(x)$ 的近似分布 $p_{g}(x)$，有了这个近似分布后，我们便可以利用这个分布进行采样，从而生成与真实数据点$\boldsymbol{x}$高度相似的样本。例如，现在我们有一批 $256 \times 256$ 的猫的图片，我们想要生成一些类似的猫的图片。已有的猫的图片便是我们的训练数据集$T_{cat}$，每一张图片便是一个数据点 $\boldsymbol{x} \in \mathbb{R}^{256 \times 256 \times 3}$。我们设 "猫的图片($256 \times 256$)" 是一个统计学上的总体$X_{cat}$，其分布为 $p_{cat}(x)$，则总体中的每一个样本，即每一张 $256 \times 256$ 的猫的图片便是服从 $p_{cat}(x)$ 的一个高维随机向量 $\boldsymbol{x} \in \mathbb{R}^{256 \times 256 \times 3}$。此时，我们的训练数据集 $T_{cat}$ 便可看作是从总体 $X_{cat}$ 中抽取的一个样本，其样本分布为$p_{data}(x)$，由大数定律可知，在大样本条件下，样本分布会趋向于总体分布，故我们可以认为 $p_{data} \rightarrow p_{cat}$，因此如果我们能够学习到 $p_{data}$ 的一个近似分布 $p_{g}$，则 $p_{g}$ 也是总体 $p_{cat}$ 的一个近似分布，我们便可以利用 $p_{g}$ 来进行采样，即生成相似的猫的图片。  
&emsp;&emsp;虽然我们希望能够直接学习数据的分布 $p_{data}$，但这在实际中是很难实现的，主要有以下两点原因：  

- **数据分布的复杂性:** 真实世界的数据通常具有复杂的高维分布，例如前文提到的猫的图片是便来自于 $256 \times 256 \times 3$ 维的分布，直接建模这种分布需要非常复杂的函数。  
- **计算复杂度:** 直接对高维数据进行概率密度估计需要大量的数据和计算资源，这往往是难以承受的。  

&emsp;&emsp;因此，在实际中，我们通常会引入隐变量假设。**隐变量假设认为存在一些无法直接观测的隐变量$\boldsymbol{z}$，这些变量可以解释观测数据的统计特性和依赖关系。隐变量与观测变量之间的依赖关系通常通过特定的概率分布来描述，通过引入隐变量，可以将观测数据的复杂分布特征简化为更为简单的分布特征。** 在生成模型中，我们所假设的隐变量通常比观测数据的维度要低，通过对隐变量进行建模，我们实际上是进行了数据压缩，对原始的特征空间进行降维，提取出数据的潜在特征，这些特征往往具有较好的可解释性。例如，在VAE中，隐变量可以表示数据的某些关键特征，如图像的形状、颜色等。  
&emsp;&emsp;还是以猫的图片为例，在观测数据的分布 $p_{data}$ 是一个高维分布，我们可以清晰地观察数据 $\boldsymbol{x}$ (猫)的各种细微特征，例如毛发、瞳孔的颜色、牙齿的形状等。而观测数据 $\boldsymbol{x}$ 背后的隐变量 $\boldsymbol{z}$ 是对原始数据特征的一种压缩，它的维度比原始数据更低，如果同样将 $\boldsymbol{z}$ 看作图片，那么这在这张图片上可能会有猫的大致轮廓、眼睛、耳朵的轮廓、模糊的毛发颜色等，虽然这张图片的细节特征并没有 $\boldsymbol{x}$ 丰富，但我们仍然可以看出这是一张猫的图片。这张图片 $\boldsymbol{z}$ 便是对观测数据 $\boldsymbol{x}$ 的分布特征一种抽象与压缩，它从现实世界中千奇百怪的猫的图片中抽象出猫的一般特征，如果我们的生成模型能够学习到猫的这种一般特征，便能从这种一般特征的基础上去生成各种猫的图片。

## 两种学习方式  

&emsp;&emsp;在生成模型，对于一些未知的分布，我们通常会采用参数化的概率模型或者神经网络模型是逼近，而对于参数的学习，目前两大主流的方法为：**最大似然法**以及**对抗训练**。最大似然法是统计推断中最经典的参数估计方法，其通过最大化数据分布的似然函数来确定分布中的未知参数，即：  

$$
\begin{equation}
    \hat{\theta} = \argmax_{\theta} \log{p_{\theta}(x)}
\end{equation}
$$  

&emsp;&emsp;现如今很多生成模型的学习策略便采用了最大似然法，例如高斯混合模型(GMM)、变分自编码(VAE)、扩散模型(Diffusion Model)。  
&emsp;&emsp; (1) 式的优化并不是一项简单的工作，在 GMM 中，由于模型较为简单，我们可以采用 EM 算法进行求解，而在 VAE、Diffusion Model 中，我们则是对似然函数 $\log{p(x)}$ 的证据下界(Evidence Lower Bound) 进行优化。然而在 2014 年，lan.Goodfellow 提出了一种完全不同的学习策略，可以绕过求解最大似然这个棘手的问题，这便是基于对抗思想的生成对抗网络(GAN)，对于 GAN 的主要思想，可以阅读笔者之前的一篇博客文章[《Generative Adversarial Network (GAN)》](https://sxusongjh.github.io/song.github.io/posts/zh/2024-05-18-generative-model-introduction/)，在这篇博客中我介绍了 GAN 的原始论文，感兴趣的读者可以自行阅读。  
&emsp;&emsp;由于本系列博客的初衷是介绍扩散模型，为了视角的统一，笔者会主要介绍几类基于最大似然的生成模型，包括GMM、VAE、Diffusion Model、Score-based Model。  

## Conference  

**[1] Paper: Luo C. Understanding diffusion models: A unified perspective[J]. arXiv preprint arXiv:2208.11970, 2022.**  












