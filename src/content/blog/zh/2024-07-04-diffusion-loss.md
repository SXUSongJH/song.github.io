---
title: Diffusion 损失函数的不同形式
tags:
    - Machine Learning
    - Generative Model
---

## toc 

&emsp;&emsp;在前一个章节，我们推导了扩散模型 ELBO 的理论形式，将其作为模型训练的损失函数，然后分析了损失函数中各项的具体计算方法，推导出了 **预测原始数据** 的损失函数形式。这一节我们将会介绍扩散模型另外两种损失函数的形式，即 **预测噪声** 与 **分数匹配**。扩散模型的原始论文 DDPM [2]，便是使用的 **预测噪声** 的损失函数形式，而之后的宋飏等的基于分数的生成模型[3] 则是使用了 **分数匹配** 的损失函数形式。在这一节，我们将会证明这三种损失函数是等价的。  

## 预测原始数据的损失函数形式  

&emsp;&emsp;首先，我们还是来回顾一下上一节的结论。在上一篇博客文章中，我们将损失函数分解为了 **重构似然损失 $L_{0}$ 、去噪匹配损失 $L_{t-1}$ 、先验匹配损失 $L_{T}$**。在这三项中，去噪匹配损失 $L_{t-1}$ 是损失函数中的主要部分，我们通过推导得到了其具体的计算公式：  

$$
\begin{align}
    D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) = \frac{1}{2\sigma_{q}^{2}(t)}\frac{\bar{\alpha}_{t-1}(1-\alpha_{t})^{2}}{(1-\bar{\alpha}_{t})^{2}} \left[ ||\hat{x}_{\theta}(x_{t-1},t) - x_{0}||_{2}^{2} \right] \tag{1}
\end{align}
$$  

&emsp;&emsp;从(1)式中我们可以看出，优化去噪匹配损失实际上是让模型在每一步尽可能地预测原始数据 $x_{0}$。通过多步的迭代，可以使得模型的输出值 $\hat{x}_{\theta}(x_{t-1},t)$ 与原始数据 $x_{0}$ 更加相似。  

## 预测噪声的损失函数形式  

&emsp;&emsp;通过上一篇博客的推导(15)，我们可以 $x_{0}$ 与 $x_{t}$ 之间所满足的等式：  

$$
\begin{align}
    x_{t} = \sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1-\bar{\alpha}_{t}}\epsilon_{0} \tag{2}
\end{align}
$$

&emsp;&emsp;基于 (2) 式，我们可以将 $x_{0}$ 用 $x_{t}$ 与 $\epsilon_{0}$ 来表示：  

$$
\begin{align}
    x_{0} = \frac{x_{t} - \sqrt{1-\bar{\alpha}_{t}}\epsilon_{0}}{\sqrt{\bar{\alpha}_{t}}} \tag{3}
\end{align}
$$  

&emsp;&emsp;通过 (3) 式，我们可以将编码器 $q(x_{t-1}|x_{t},x_{0})$ 所满足的高斯分布的均值 $\mu_{q}(x_{t},x_{0})$ 转化为关于原始数据 $x_{0}$ 与原始噪声 $\epsilon_{0}$ 的函数：  

$$
\begin{align}
    \mu_{q}(x_{t},x_{0}) = \frac{1}{\sqrt{\alpha_{t}}}x_{t} - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}(1-\bar{\alpha}_{t})}}\epsilon_{0} \tag{4} \\
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    \mu_{q}(x_{t},x_{0}) &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})x_{0}}{1-\bar{\alpha}_{t}} \notag \\
    &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})\frac{x_{t} - \sqrt{1-\bar{\alpha}_{t}}\epsilon_{0}}{\sqrt{\bar{\alpha}_{t}}}}{1-\bar{\alpha}_{t}} \notag \\
    &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + (1-\alpha_{t})\frac{x_{t} - \sqrt{1-\bar{\alpha}_{t}}\epsilon_{0}}{\sqrt{\alpha}_{t}}}{1-\bar{\alpha}_{t}} \notag \\
    &= \left( \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} + \frac{1-\alpha_{t}}{(1-\bar{\alpha}_{t})\sqrt{\alpha_{t}}} \right)x_{t} - \frac{(1-\alpha_{t})\sqrt{1-\bar{\alpha}_{t}}}{(1-\bar{\alpha}_{t})\sqrt{\alpha_{t}}}\epsilon_{0} \notag \\
    &= \frac{1}{\sqrt{\alpha_{t}}}x_{t} - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}(1-\bar{\alpha}_{t})}}\epsilon_{0} \notag \\
\end{align}
$$

&emsp;&emsp;在前一节中，我们将解码器 $p_{\theta}(x_{t-1}|x_{t})$ 同样设置为高斯分布 $N(x_{t-1}; \mu_{\theta}(x_{t},t),\Sigma_{q}(t))$，且高斯分布的均值 $\mu_{\theta}(x_{t},t)$ 具有与编码器的均值 $\mu_{q}(x_{t},x_{0})$ 相同的形式，故解码器 $p_{\theta}(x_{t-1}|x_{t})$ 同样可以用 (4) 式的形式表示，只是在解码过程中，我们没有 $\epsilon_{0}$ 的信息，故神经网络需要根据 $x_{t},t$ 的信息预测噪声 $\epsilon_{0}$。综上所述，我们可以解码器的均值重写为：  

$$
\begin{align}
    \mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{\alpha_{t}}}x_{t} - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}(1-\bar{\alpha}_{t})}}\hat{\epsilon}_{\theta}(x_{t},t) \tag{5} \\
\end{align}
$$  

&emsp;&emsp;这样，我们可以将编码器 $q(x_{t-1}|x_{t},x_{0})$ 与解码器 $p_{\theta}(x_{t-1}|x_{t})$ 之间的 KL Divergence (1) 式用噪声的预测误差重新表示：  

$$
\begin{align}
    D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) = \frac{1}{2\sigma_{q}^{2}(t)} \frac{(1-\alpha_{t})^{2}}{(1-\bar{\alpha}_{t})\alpha_{t}}\left[ || \hat{\epsilon}_{\theta}(x_{t},t) - \epsilon_{0} ||_{2}^{2} \right] \tag{6} \\
\end{align}
$$

**$Proof$**

$$
\begin{align}
    & D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[ ||\mu_{q} - \mu_{\theta} ||_{2}^{2} \right] \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[ ||\frac{1}{\sqrt{\alpha_{t}}}x_{t} - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}(1-\bar{\alpha}_{t})}}\epsilon_{0} - \frac{1}{\sqrt{\alpha_{t}}}x_{t} + \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}(1-\bar{\alpha}_{t})}}\hat{\epsilon}_{\theta}(x_{t},t) ||_{2}^{2} \right] \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[ || \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}(1-\bar{\alpha}_{t})}}\hat{\epsilon}_{\theta}(x_{t},t) - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}(1-\bar{\alpha}_{t})}}\epsilon_{0} ||_{2}^{2} \right] \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \frac{(1-\alpha_{t})^{2}}{(1-\bar{\alpha}_{t})\alpha_{t}}\left[ || \hat{\epsilon}_{\theta}(x_{t},t) - \epsilon_{0} ||_{2}^{2} \right] \notag \\
\end{align}
$$  

&emsp;&emsp;(6) 式表明在训练过程中，我们的解码器每一步都需要尽可能地去预测原始噪声 $\epsilon_{0}$。在前向加噪过程中，我们是将原始噪声 $\epsilon_{0}$ 不断地加到原始数据 $x_{0}$ 中(2)，直至原始数据变为近似高斯噪声，故如果编码器能够根据 $x_{t},t$ 很好地预测噪声 $\epsilon_{0}$，则在解码过程中，我们可以从高斯噪声开始，每一步逐步减去编码器所预测的原始噪声 $\hat{\epsilon}_{\theta}(x_{t},t)$，从而将高斯噪声还原回原始数据 $x_{0}$。以上过程可以用下图1表示。  

<center>
    <img src="https://s2.loli.net/2024/06/06/li8W17MkhEJOeBp.png" width=80% height=80%>
    <div align="center">Image1: 预测噪声训练过程</div>
</center>  

<center>
    <img src="https://s2.loli.net/2024/06/06/zvlHM8wZNYAcueU.png" width=80% height=80%>
    <div align="center">Image2: DDPM原始论文的训练与采样算法</div>
</center>  

&emsp;&emsp;在图2右侧的采样过程中，每一步解码，即减去预测的噪声后还加了一个噪声项 $\sigma_{t}z$，这一处理是模仿了前向扩散过程中的随机性，确保每一步生成的样本不是确定的，而是带有一定的随机性，从而可以生成多样化的样本。这是关键的，因为如果每一步都只是简单的去噪而不引入新的随机性，生成的样本将会缺乏多样性。  

## 分数匹配的损失函数形式  

&emsp;&emsp;在 DDPM 之后，Yang Song 等 [3] 在 2021 年提出了基于 VDM 的 Score-Based Generative Model (SGM)。在这篇论文中，作者使用 SDEs 建立起了扩散模型前向加噪与逆向去噪的一般框架，并利用得分函数作为损失函数来优化模型。关于 SGM 我们将会在下一篇博客中详细讨论，接下来我们主要来介绍一下基于分数的损失函数。  
&emsp;&emsp;为了得分函数函数，我们首先来介绍一下概率统计中的 Tweedie 公式。  
&emsp;&emsp;Tweedie 公式表明，给定从指数族分布中抽取的样本，其真实均值可以通过样本的最大似然估计（也称为经验均值）加上一些涉及估计得分的修正项来估计。在只有一个观测样本的情况下，经验均值就是该样本本身。Tweedie 公式通常用于减轻样本偏差；如果观测到的样本全部位于真实分布的一端，那么负得分会变大，并将样本的最大似然估计值校正到真实均值。  
&emsp;&emsp;具体来讲，假设我们从一个指数分布中抽取一些样本，但这些样本偏向分布的一端。单纯地使用这些样本的均值 (最大似然估计) 会偏离真实的分布均值。Tweedie 公式会通过添加一个修正项来纠正这种偏差。这个修正项会通过样本分布和得分函数来计算。得分函数是统计学中的一个概念，它是指对数似然函数关于参数的导数。如果样本都位于分布的一端，得分函数会变大，修正项会变大，从而将估计的均值向真实均值方向校正。
&emsp;&emsp;给定一个高斯随机变量 $z \sim N(z;\mu_{z},\Sigma_{z})$，由Tweedie 可以得到：  

$$
\begin{align}
    \mathbb{E}[\mu_{z}|z] = z + \Sigma_{z}\nabla_{z}\log{p(z)} \tag{7}
\end{align}
$$  

&emsp;&emsp;在 VDM 中，通过高斯假设，我们推导了前向加噪过程中 $x_{t}$ 所满足的高斯分布：  

$$
\begin{align}
    q(x_{t}|x_{0}) = N(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})\boldsymbol{I}) \tag{8}
\end{align}
$$  

&emsp;&emsp;利用 Tweedie 公式，我们可以给出 $x_{t}$ 在给定样本情况下的后验均值的修正估计：  

$$
\begin{align}
    \mathbb{E}[\mu_{x_{t}}|x_{t}] = x_{t} + (1 - \bar{\alpha}_{t})\nabla_{x_{t}}\log{p(x_{t})} \tag{9}
\end{align}
$$  

&emsp;&emsp;由于在逆向去噪过程中，我们是不知道原始数据 $x_{0}$ 的，我们需要对 $x_{0}$ 进行估计，前文中的预测原始数据的损失函数便是在做这件事。这里，通过结合 $x_{t}$ 的真实均值 $\sqrt{\bar{\alpha}_{t}}x_{0}$，我们可以给出后向去噪过程中 $x_{0}$ 的估计值：  

$$
\begin{align}
    x_{0} = \frac{x_{t} + (1 - \bar{\alpha}_{t})\nabla\log{p(x_{t})}}{\sqrt{\bar{\alpha}_{t}}} \tag{10} \\
\end{align}
$$

&emsp;&emsp;其中，$\nabla\log{p(x_{t})}$ 为 $\nabla_{x_{t}}\log{p(x_{t})}$ 的简写形式。需要指出的是，在后向去噪过程中，$x_{t}$ 所满足的高斯分布的均值是未知的，故解码过程中，(10)式中的得分函数 $\nabla\log{p(x_{t})}$ 是未知的。而在前向加噪过程中，基于高斯假设，我们已经得知 $x_{t}$ 的所满足的高斯分布(8)，故得分函数是可以计算的。  
&emsp;&emsp;现在我们需要将前向加噪过程中的 $x_{t-1}$ 的均值改写成含有得分函数的形式，将(10)带入 $\mu_{q}(x_{t},x_{0})$ 的原始表达式，通过推导可以得到以下等式成立：  

$$
\begin{align}
    \mu_{q}(x_{t},x_{0}) = \frac{1}{\sqrt{\alpha_{t}}}x_{t} + \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}\nabla\log{p(x_{t})} \tag{11} \\
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    \mu_{q}(x_{t},x_{0}) &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})x_{0}}{1-\bar{\alpha}_{t}} \notag \\
    &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})\frac{x_{t} + (1 - \bar{\alpha}_{t})\nabla\log{p(x_{t})}}{\sqrt{\bar{\alpha}_{t}}}}{1-\bar{\alpha}_{t}} \notag \\
    &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + (1-\alpha_{t})\frac{x_{t} + (1 - \bar{\alpha}_{t})\nabla\log{p(x_{t})}}{\sqrt{\alpha}_{t}}}{1-\bar{\alpha}_{t}} \notag \\
    &= \left( \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}} + \frac{1-\alpha_{t}}{(1-\bar{\alpha}_{t})\sqrt{\alpha_{t}}} \right)x_{t} - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}\nabla\log{p(x_{t})} \notag \\
    &= \frac{1}{\sqrt{\alpha_{t}}}x_{t} + \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}\nabla\log{p(x_{t})} \notag \\
\end{align}
$$  

&emsp;&emsp;因此，我们可以将逆向去噪过程中 $x_{t-1}$ 的均值设置成与前向过程相同的形式，只是得分函数需要由神经网络进行估计：  

$$
\begin{align}
    \mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{\alpha_{t}}}x_{t} + \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}s_{\theta}(x_{t},t) \tag{12} \\
\end{align}
$$  

&emsp;&emsp;结合 (11)、(12)式，我们可以得出基于得分函数的 $q(x_{t-1} | x_{t},x_{0})$ 与 $p_{\theta}(x_{t-1}|x_{t})$ 之间的KL Divergence 的表达式：  

$$
\begin{align}
    D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) = \frac{1}{2\sigma_{q}^{2}(t)} \frac{(1-\alpha_{t})^{2}}{\alpha_{t}}\left[ || s_{\theta}(x_{t},t) - \nabla\log{p(x_{t})} ||_{2}^{2} \right] \tag{13}
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    & D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[ ||\mu_{q} - \mu_{\theta} ||_{2}^{2} \right] \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[ ||\frac{1}{\sqrt{\alpha_{t}}}x_{t} + \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}s_{\theta}(x_{t},t) - \frac{1}{\sqrt{\alpha_{t}}}x_{t} - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}\nabla\log{p(x_{t})} ||_{2}^{2} \right] \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[ || \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}s_{\theta}(x_{t},t) - \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}\nabla\log{p(x_{t})} \right] \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \frac{(1-\alpha_{t})^{2}}{\alpha_{t}}\left[ || s_{\theta}(x_{t},t) - \nabla\log{p(x_{t})} ||_{2}^{2} \right] \notag \\
\end{align}
$$  

&emsp;&emsp;这样我们就可以得到基于得分函数的损失函数，在解码过程中，神经网络需要通过给定的 $x_{t}, t$ 去预测真实的得分函数。得分函数给出了似然函数的梯度，即使得似然函数最大的方向，去噪过程中数据由高斯噪声，沿着神经网络所预测出的这个方向移动，从而到达最大的重构似然。  
&emsp;&emsp;实际上，数据在去噪过程移动的方向应该是噪声的反向，即“去噪”，这是符合我们的直觉的。事实也的确如此，联立 (3) 式与 (10) 式，我们可以得到得分函数与原始噪声之间的联系：  

$$
\begin{align}
    \nabla\log{p(x_{t})} = -\frac{1}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{0} \tag{14}
\end{align}
$$  

&emsp;&emsp;得分函数衡量了如何在数据空间移动以使得对数似然最大化，由于在前向过程中，我们是将噪声不断地加入到图片中，因此在逆向过程中，我们很自然地应该向反方向移动，即逐渐去噪，以得到更高的对数似然，即与原始图片更加相似。  
&emsp;&emsp;以上我们推导了扩散模型损失函数的三种等价形式，包括 **预测原始数据、预测噪声、得分匹配**。关于得分匹配损失函数，在下一节关于 Score-Based Generative Model 中将会有更加详细的解释。


## Reference  

**[1] Paper: Luo C. Understanding diffusion models: A unified perspective[J]. arXiv preprint arXiv:2208.11970, 2022.**  
**[2] Paper: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.**  
**[3] Paper: Yang Song,  Jascha Sohl-Dickstein, et al, "Score-Based Generative Modeling through Stochastic Differential Equations," in International Conference on Learning Representations, 2021.**  
**[4] Video: 想不出来昵称又想改, 扩散模型-Diffusion Model【李宏毅2023】, Blibili**  
**[5] Blog: 苏剑林, 生成扩散模型漫谈(1-3), 科学空间** 
