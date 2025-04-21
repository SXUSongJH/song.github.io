---
title: Diffusion 推导
tags:
    - Machine Learning
    - Generative Model
---  

# 变分扩散模型(VDM)  

&emsp;&emsp;在上一节关于变分自编码的介绍中，我们已经讨论到了具有多层隐变量以及马尔可夫性质的变分自编码模型(MHVAE)，其基本形式与我们今天要介绍的变分扩散模型(Variational Diffusion Models)已经非常相似，在 MHVAE 的基础上， VDM的主要改进有三个方面：  

- **隐变量 $z$ 的维度:** VDM将隐变量 $z$ 的维度设置成与数据 $x$ 一致。  
- **编码器 $q(z|x)$ 的分布:** 每个时刻 $t$ 的编码器 $q(z_{t}|z_{t-1})$ 不再是一个需要学习的分布，而是由前一时刻所输入 $z_{t-1}$ 为中心的高斯分布。  
- **隐变量高斯分布的参数:** 隐变量高斯分布的参数随时间 $t$ 改变，经过 $T$ 步后最终变为标准高斯分布。  

&emsp;&emsp;我们来尝试理解一下这些改进的 motivations。在 VAE 中，我们首先将数据 $x$ 由数据空间通过编码器 $q_{\phi}(z|x)$ 映射到隐空间中，隐变量 $z$ 的维度比 $x$ 要小，这一步的目的是希望隐变量能够抽象出数据 $x$ 的一般分布特征，而忽略掉特殊细节；隐变量 $z$ 的分布为高斯分布 $N(z; \boldsymbol{\mu}_{\phi}(x),\sigma_{\phi}^{2}(x)\boldsymbol{I})$，参数由解码器计算出。同时我们希望该高斯分布与标准高斯分布的 KL Divergence 尽可能小，即尽可能相似，这一方面是因为我们希望提升模型的泛化性能，避免模型学习到的模式过于单一，另一方面是因为在采样时我们需要从标准高斯分布采样出隐变量 $z$，再由解码器生成新的样本 $x'$，在训练时要求隐变量 $z$ 的分布与标准高斯分布尽可能相似也是希望能够与采样过程匹配。但事与愿违，由于训练目标的对抗性，隐变量的分布无法与高斯分布非常相似，另外单个隐变量对于分布特征的抽象能力也十分有限，这造成原始 VAE 所生成的图片大多非常模糊，效果不佳。  
&emsp;&emsp;MHVAE 采用了多层次隐变量的架构，通过叠加多个隐变量，使得编码器一步一步地将原始数据的分布特征抽象出来，再通过解码器一步一步对隐变量进行解码，生成新样本。从一步到多步，虽然计算过程变得更复杂，但模型的学习数据分布的能力会变得更强，经过多次编码，隐变量的分布特征变得越发不明显(抽象)，其与标准高斯分布的相似程度也会越高，这样就能更加匹配采样过程。但 MHVAE 也具有缺陷，它虽然改善了分布不匹配问题，但由于训练目标的对抗性，仍无法彻底解决这个问题。同时，由于存在多个参数化的编码器与解码器，模型参数量较多，模型的训练需要很长时间。
&emsp;&emsp;现在我们来讨论 VDM 的想法，既然采样过程是先从标准高斯分布采样出 $z$ ，再经过解码器生成新样本 $x'$，VAE 与 MHVAE 均是希望应该将数据 $x$ 编码到与标准高斯分布相似的隐变量分布，以匹配采样过程，主要的困难在于很难学习出能够实现这一过程的编码器 $q_{\phi}(z|x)$。VDM 的想法便是，既然编码器很难学，那干脆不学了，人为设定编码器，通过更长的步骤，将数据 $x$ 逐步编码到近似标准高斯分布(随机噪声)。既然要将数据 $x$ 逐步编码为近似噪声，那干脆采用逐步加噪的方法，这种想法最为简洁，即：  

$$
x_{t} = \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1-\alpha_{t}}\epsilon,\quad \epsilon \sim N(\epsilon; \boldsymbol{0,I})
$$  

&emsp;&emsp;其中，$x_{0}$ 表示初始的数据 $x$，$x_{1:T}$ 表示编码后的隐变量，对应于MHVAE中的 $z_{1:T}$，基于这种形式，我们自然需要假设隐变量的维度与数据一致。同时，由于马尔可夫性质，$x_{t}$ 的分布自然是以 $x_{t-1}$ 为中心的高斯分布(不考虑系数)。  
&emsp;&emsp;这样设置的好处是显而易见的，在这种条件下，编码器没有参数要学习，是一个线性过程，速度较快，则可以用更长的加噪步骤使得最终得到的隐变量分布 $q(x_{T}|x_{T-1})$ 与噪声更加接近。同时，加噪的过程也可以视为将数据 $x$ 的分布特征进行抽象，例如一张清晰的猫的图片 $x_{0}$ 在经过多次加噪后，只能看见模糊的猫的轮廓了，这也是对猫的图片的分布特征的一种压缩与抽象。解码器则是从随机噪声生成新样本，与编码过程互逆。在正向加噪过程中已经产生了每个步骤加噪前与加噪后的图片对，如果解码器能够训练成编码器的逆过程，即利用正向过程得到的图片对，基于加噪后的图片预测噪声，从而得到加噪前的图片，则可以完成逐步去噪的过程，生成与原始图片相似的新样本。以猫的图片的例子类比，这个过程是从猫的一般特征(模糊)，去生成细节更加丰富的猫的图片(清晰)。这个过程大大改善了以往 VAE 所存在的分布不匹配问题。这便是我理解的 VDM 在 MHVAE 基础上的 motivations，接下来我们具体来介绍 VDM 的细节。  

## 概率模型  

&emsp;&emsp;VDM 的概率图与 MHVAE 基本相同，其概率图如下图1所示：  

<center>
    <img src="https://s2.loli.net/2024/05/27/F7RPJpD53LSaEVq.png" width=80% height=80%>
    <div align="center">Image1: VDM 概率图</div>
</center>  

&emsp;&emsp;其中 $x_0$ 表示原始数据，$x_{1:T}$ 表示隐变量。由前文的讨论可知，VDM 的前向编码过程是一个不需要学习的逐步加噪过程：  

$$
\begin{align}
    x_{t} = \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1-\alpha_{t}}\epsilon,\quad \epsilon \sim N(\epsilon; \boldsymbol{0,I}) \tag{1}
\end{align}
$$  

&emsp;&emsp;其中 $\alpha_{t}$ 是随层次 $t$ 变化的常数(潜在可学习)。这样第 $t$ 层的编码器便是以 $\sqrt{\alpha_{t}} x_{t-1}$ 为均值的高斯分布。同时，与 MHVAE 一样，VDM 各层的转移概率分布也满足马尔可夫性质，故有：  

$$
\begin{align}
    q(x_{1:T}|x_{0}) &= \prod_{t=1}^{T}q(x_{t} | x_{t-1}) \tag{2} \\
    q(x_{t} | x_{t-1}) &= N(x_{t}; \sqrt{\alpha_{t}} x_{t-1}, (1-\alpha_{t})\boldsymbol{I}) \tag{3}
\end{align}
$$  

&emsp;&emsp;从前文的第三个假设中，我们可以得知，最终层次的隐变量 $x_{T}$ 的先验分布为标准高斯分布。VDM 的逆向去噪过程需要通过参数化的解码器 $p_{\theta}(x_{t-1}|x_{t})$，逐步将图片由高斯噪声 $x_{T}$，还原回原始数据 $x_{0}$。通过马尔可夫性质，我们可以写出 VDM 的联合分布：  

$$
\begin{align}
    p(x_{0:T}) &= p(x_{T})\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t}) \tag{4} \\
    p(x_{T}) &= N(x_{T};\boldsymbol{0,I}) \tag{5} \\
\end{align}
$$  

&emsp;&emsp;与 MHVAE 不同的是，在 VDM 中，我们只需要学习解码器的参数 $\boldsymbol{\theta}$。当训练完成后，采样过程便是先从标准高斯分布 $p(x_{T})$ 中采样出高斯噪声 $x_{T}'$，再通过学习到的各层解码器 $p_{\theta}(x_{t-1}|x_{t})$ 经过 $T$ 步解码后，生成新的数据 $x_{0}'$。  

## 变分下界(ELBO)  

&emsp;&emsp;与 MHVAE 一样，VDM 同样是对似然函数的变分下界进行优化。  

**VDM's ELBO**  

$$
\begin{align}
    \log{p(x)} \ge \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] \tag{6}
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    \log{p(x)} &= \log{p(x)} \int q(x_{1:T}|x_{0})dx_{1:T} \notag \\
    &= \int \log{p(x)} q(x_{1:T}|x_{0})dx_{1:T} \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{p(x)} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})q(x_{1:T}|x_{0})}{p(x_{1:T}|x_{0})q(x_{1:T}|x_{0})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] + \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{q(x_{1:T}|x_{0})}{p(x_{1:T}|x_{0})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] + D_{KL}(q(x_{1:T}|x_{0}) || p(x_{1:T}|x_{0})) \tag{7} \\
    &\ge \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] \notag
\end{align}
$$  

&emsp;&emsp;由以上证明的(7)式我们可以得知，似然函数与ELBO之间的差为 $q(x_{1:T}|x_{0})$ 与 $p(x_{1:T}|x_{0})$ 之间的 KL Divergence，其表示给定原始数据 $x_{0}$ 后，编码过程的联合分布与解码过程的联合分布之间的KL距离。最大化 ELBO 等价于最小化这个 KL Divergence。这个距离越小，则说明正向加噪与逆向去噪越匹配，模型的生成效果越好。进一步地，利用马尔可夫性质，我们可以将这个 KL Divergence 分解成三项：  

$$
\begin{align}
    & D_{KL}(q(x_{1:T}|x_{0}) || p(x_{1:T}|x_{0})) \tag{8}\\
    =&\sum_{t=1}^{T-1}\mathbb{E}_{q(x_{t-1},x_{t+1} | x_{0})} \left[ D_{KL}(q(x_{t}|x_{t-1}) || p_{\theta}(x_{t}|x_{t+1})) \right] \tag{consistency term}\\
    &+ \mathbb{E}_{q(x_{T-1}|x_{0})}\left[ D_{KL}(q(x_{T}|x_{T-1}) || p(x_{T})) \right] \tag{prior matching term}\\
    &- \mathbb{E}_{q(x_{1}|x_{0})}\left[ \log{\frac{p_{\theta}(x_{0}|x_{1})}{p(x_{0})}} \right] \tag{reconstruction term}
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    & D_{KL}(q(x_{1:T}|x_{0}) || p(x_{1:T}|x_{0})) \notag \\
    & = \mathbb{E}_{q(x_{1:T}|x_{0})}\left[ \log{\frac{q(x_{1:T}|x_{0})}{p(x_{1:T}|x_{0})}} \right] \notag\\
    & = \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0})\prod_{t=1}^{T}q(x_{t}|x_{t-1})}{p(x_{T})\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t})}} \right] \notag \\
    & = \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0})q(x_{T}|x_{T-1})\prod_{t=1}^{T-1}q(x_{t}|x_{t-1})}{p(x_{T})p_{\theta}(x_{0}|x_{1})\prod_{t=1}^{T-1}p_{\theta}(x_{t}|x_{t+1})}} \right] \notag \\
    & = \sum_{t=1}^{T-1}\mathbb{E}_{q(x_{t-1},x_{t},x_{t+1}|x_{0})}\left[ \log{\frac{q(x_{t}|x_{t-1})}{p_{\theta}(x_{t}|x_{t+1})}} \right] + \mathbb{E}_{q(x_{T-1},x_{T}|x_{0})}\left[ \log{\frac{q(x_{T}|x_{T-1})}{p(x_{T})}} \right] - \mathbb{E}_{q(x_{1}|x_{0})}\left[ \log{\frac{p_{\theta}(x_{0}|x_{1})}{p(x_{0})}} \right] \notag \\
    & = \sum_{t=1}^{T-1}\underbrace{\mathbb{E}_{q(x_{t-1},x_{t+1}|x_{0})}\left[ D_{KL}(q(x_{t}|x_{t-1}) || p_{\theta}(x_{t}|x_{t+1})) \right]}_{consistency \ term} + \underbrace{\mathbb{E}_{q(x_{T-1}|x_{0})}\left[ D_{KL}( q(x_{T}|x_{T-1}) || p(x_{T})) \right]}_{prior \ matching \ term}  - \underbrace{\mathbb{E}_{q(x_{1}|x_{0})}\left[ \log{\frac{p_{\theta}(x_{0}|x_{1})}{p(x_{0})}} \right]}_{reconstruction \ term} \notag \\
\end{align}
$$  

&emsp;&emsp;我们对 KL Divergence 进行了分解，得到了 consistency term, prior matching term, reconstruction term 三项。 最小化 KL Divergence 等价于最小化这三项，即使得 consistency term, prior matching term 尽可能小，reconstruction term 尽可能大。我们先不详细解释这三项的含义，接下来我们同样对 VDM 的 ELBO(6) 进行分解，我们会发现 ELBO 分解后的结果与 KL Divergence 几乎一致，只是符号相反：  

$$
\begin{align}
   & \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] \tag{9} \\
   =& \mathbb{E}_{q(x_{1}|x_{0})}[\log{p_{\theta}(x_{0}|x_{1})}] \tag{reconstruction term} \\
   -& \mathbb{E}_{q(x_{T-1}|x_{0})}\left[ D_{KL}( q(x_{T}|x_{T-1}) || p(x_{T})) \right] \tag{prior matching term}\\
   -& \sum_{t=1}^{T-1}\mathbb{E}_{q(x_{t-1},x_{t+1}|x_{0})}\left[ D_{KL}(q(x_{t}|x_{t-1}) || p_{\theta}(x_{t}|x_{t+1})) \right] \tag{consistency term} \\
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    & \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] \notag \\
    & = \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t})}{\prod_{t=1}^{T}q(x_{t}|x_{t-1})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})p_{\theta}(x_{0}|x_{1})\prod_{t=2}^{T}p_{\theta}(x_{t-1}|x_{t})}{q(x_{T}|x_{T-1})\prod_{t=1}^{T-1}q(x_{t}|x_{t-1})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})p_{\theta}(x_{0}|x_{1})\prod_{t=1}^{T-1}p_{\theta}(x_{t}|x_{t+1})}{q(x_{T}|x_{T-1})\prod_{t=1}^{T-1}q(x_{t}|x_{t-1})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})p_{\theta}(x_{0}|x_{1})}{q(x_{T}|x_{T-1})}} \right] + \mathbb{E}_{q(x_{1:T}|x_{0})}\left[ \log{\prod_{i=1}^{T-1}} \frac{p_{\theta}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})} \right] \notag \\  
    &= \mathbb{E}_{q(x_{1}|x_{0})}[\log{p_{\theta}(x_{0}|x_{1})}] + \mathbb{E}_{q(x_{T-1},x_{T}|x_{0})} \left[ \log{\frac{p(x_{T})}{q(x_{T}|x_{T-1})}} \right] + \sum_{i=1}^{T-1}\mathbb{E}_{q(x_{1:T}|x_{0})}\left[ \log{ \frac{p_{\theta}(x_{t}|x_{t+1})}{q(x_{t}|x_{t-1})}} \right] \notag \\
    &= \underbrace{\mathbb{E}_{q(x_{1}|x_{0})}[\log{p_{\theta}(x_{0}|x_{1})}]}_{reconstruction \ term} - \underbrace{\mathbb{E}_{q(x_{T-1}|x_{0})}[D_{KL}(q(x_{T}|x_{T-1}) || p(x_{T}))]}_{prior \ matching \ term} - \sum_{i=1}^{T-1}\underbrace{\mathbb{E}_{q(x_{t-1},x_{t+1}|x_{0})}\left[ D_{KL}(q(x_{t}|x_{t-1}) || p_{\theta}(x_{t} || x_{t+1})) \right]}_{consistency \ term} \notag \\
\end{align}
$$  

&emsp;&emsp;由以上的推导我们将 ELBO 分解为与 KL Divergence 相似的三项，只是符号相反，这也从另一个方面说明了最大化 ELBO 实际上等价于最小化 KL Divergence。最大化 ELBO 的过程是使得 reconstruction term 尽可能大，prior matching term, consistency term 尽可能小。接下来以 ELBO 的分解结果为例，我们尝试理解一下这三项的含义。  

- **reconstruction term:** 重构项。这一项的含义是将原始数据 $x_{0}$ 编码一次后得到隐变量 $x_{1}$ 后再通过解码器还原回 $x_{0}$，所得到的对数似然。这一项在 VAE 中也存在，这一项的值越大，表明数据在编码与解码后与原始数据更相似，即生成的效果更好。  
- **prior matching term:** 先验匹配项。这一项的含义是最终隐变量 $x_{T}$ 的后验分布 $q(x_{T}|x_{T-1})$ 与其先验分布 $p(x_{T})$ 之间的 KL Divergence 的期望。这一项越小，则先验与后验越匹配，说明编码过程得到的最终隐变量的分布与标准高斯分布越接近，更能匹配采样过程。  
- **consistency term:** 一致性项。这一项的含义是从前向和后向两个过程努力使 $x_{t}$ 处的分布保持一致。如图2所示，对于每一个中间时间步 $t$，从噪声图像中得到的去噪图片的分布 $p_{\theta}(x_{t}|x_{t+1})$  应该与从干净图像中得到的相应加噪步骤得到的图片的分布 $q(x_{t}|x_{t-1})$ 相匹配，这在数学上通过KL Divergence得到了体现。这一项越小，说明解码器 $p_{\theta}(x_{t}|x_{t-})$ 被训练的越好。  

<center>
    <img src="https://s2.loli.net/2024/05/29/Aq8Eb5kvrWURtNX.png" width=80% height=80%>
    <div align="center">Image2: 一致性</div>
</center>  

&emsp;&emsp;通过以上的推导，VDM 的 ELBO 均分解成了期望的形式，我们可以使用蒙特卡洛方法来对这些项进行近似，然而，实际上使用我们刚才推导出的(9)式来优化ELBO可能是次优的；因为一致性项在每个时间步 $t$ 上都被计算为两个随机变量 $x_{t-1},x_{t+1}$ 的期望值，因此其蒙特卡洛估计的方差有可能高于每个时间步仅使用一个随机变量估计的项。由于它是通过对 $T-1$ 个时间步求和来计算的，因此对于大的T值，ELBO的最终估计值可能具有很高的方差。  
&emsp;&emsp;为了改善这个问题，我们尝试对 (9) 式做一些变形。基于马尔可夫性与贝叶斯公式，我们可以得到以下等式：  

$$
\begin{align}
    q(x_{t}|x_{t-1}) = q(x_{t}|x_{t-1},x_{0}) = \frac{q(x_{t-1})q(x_{t}|x_{0})}{q(x_{t-1}|x_{0})} \tag{10}
\end{align}
$$  

&emsp;&emsp;通过(10)式，我们可以将 VDM 的 ELBO 分解为如下形式：  

$$
\begin{align}
   & \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] \tag{11} \\
   =& \mathbb{E}_{q(x_{1}|x_{0})}[\log{p_{\theta}(x_{0}|x_{1})}] \tag{reconstruction term} \\
   -& D_{KL}( q(x_{T}|x_{0}) || p(x_{T}))  \tag{prior matching term}\\
   -& \sum_{t=2}^{T}\mathbb{E}_{q(x_{t}|x_{0})}\left[ D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) \right] \tag{denoising matching term} \\
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    & \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{0:T})}{q(x_{1:T}|x_{0})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t})}{\prod_{t=1}^{T}q(x_{t}|x_{t-1})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})p_{\theta}(x_{0}|x_{1})\prod_{t=2}^{T}p_{\theta}(x_{t-1}|x_{t})}{q(x_{1}|x_{0})\prod_{t=2}^{T}q(x_{t}|x_{t-1})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})p_{\theta}(x_{0}|x_{1})}{q(x_{1}|x_{0})}} + \log{\prod_{t=2}^{T}\frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_{t}|x_{t-1},x_{0})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})p_{\theta}(x_{0}|x_{1})}{q(x_{1}|x_{0})}} + \log{\prod_{t=2}^{T}\frac{p_{\theta}(x_{t-1}|x_{t})}{\frac{q(x_{t-1}|x_{t},x_{0})\cancel{q(x_{t}|x_{0})}}{\cancel{q(x_{t-1}|x_{0})}}}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1:T}|x_{0})} \left[ \log{\frac{p(x_{T})p_{\theta}(x_{0}|x_{1})}{\cancel{q(x_{1}|x_{0})}}} +\log{\frac{\cancel{q(x_{1}|x_{0})}}{q(x_{T}|x_{0})}} + \log{\prod_{t=2}^{T}\frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_{t-1}|x_{t},x_{0})}} \right] \notag \\
    &= \mathbb{E}_{q(x_{1}|x_{0})}[\log{p_{\theta}(x_{0}|x_{1})}] - \mathbb{E}_{q(x_{T}|x_{0})}\left[ \log{\frac{q(x_{T}|x_{0})}{p(x_{T})}} \right] -\sum_{t=2}^{T} \mathbb{E}_{q(x_{t},x_{t-1}|x_{0})}\left[ \log{\frac{q(x_{t-1}|x_{t},x_{0})}{p_{\theta}(x_{t-1}|x_{t})}} \right] \notag \\
    &= \underbrace{\mathbb{E}_{q(x_{1}|x_{0})}[\log{p_{\theta}(x_{0}|x_{1})}]}_{reconstruction \ term} - \underbrace{D_{KL}(q(x_{T}|x_{0}) || p(x_{T}))}_{prior \ matching \ term} - \sum_{t=2}^{T}\underbrace{\mathbb{E}_{q(x_{t}|x_{0})}[D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t}))]}_{denoising \ matching \ term} \notag \\
\end{align}
$$  

&emsp;&emsp;reconstruction term 与 prior matching term 的含义与(9)基本一致。差别较大的是(9)式中的 consistency term 与 (11)式中的 denoising matching term。与 consistency term 相比，denoising matching term 中的每一时间步 $t$，只需要计算一个随机变量 $x_{t}$ 的期望，显著改善了蒙特卡洛估计的方差较大的问题。同时，最小化 denoising matching term 意味着在每一个时间步 $t$，通过解码器去噪后的数据的分布 $p_{\theta}(x_{t-1}|x_{t})$ 与真实加噪过程中加入噪声前的图片的分布 $q(x_{t-1}|x_{t},x_{0})$ 相匹配，即 KL Divergence 尽可能小。这一项越小，说明解码器每一步预测噪声的能力越强，即从一般抽象特征去生成更加细节的特征的能力越强，生成图片与原始图片就会越相似，这一过程如图3所示。  

<center>
    <img src="https://s2.loli.net/2024/05/29/FAxezibV6GnNpv3.png" width=80% height=80%>
    <div align="center">Image3: 去噪匹配</div>
</center>  

## 损失函数  

&emsp;&emsp;在前文中，我们推导了 VDM's ELBO 的理论形式(11)，通过最大化 ELBO 来近似最大化对数似然，得到待估参数 $\theta$。现在我们要利用 VDM 的假设条件，根据 ELBO 的理论形式，来得到具体用于模型训练的损失函数。  
&emsp;&emsp;通过前文的分析，我们可以将VDM 的损失函数写作如下的三部分:   

$$
\begin{align}
    \boldsymbol{L}(\theta) = -ELBO = \mathbb{E}_{q}\left[ \underbrace{D_{KL}(q(x_{T}|x_{0}) || p(x_{T}))}_{L_{T}} + \sum_{t=2}^{T}\underbrace{D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t}))}_{L_{t-1}} - \underbrace{\log{p_{\theta}(x_{0}|x_{1})}}_{L_{0}} \right] \tag{12}
\end{align}
$$

&emsp;&emsp;接下来我们来逐个讨论这三项的具体形式。  

### 先验匹配损失 $L_{T}$  

&emsp;&emsp;这一项是衡量最终隐变量 $X_{T}$ 的先验分布与后验分布的相似程度，其中 $p(x_{T})$ 是标准高斯分布，当给定数据 $x_{0}$ 后， $x_{T}$ 的后验分布 $q(x_{T}|x_{0})$ 可以不依赖于参数 $\theta$ 计算出，故 $L_{T}$ 损失函数中相当于常数，可以不考虑。  

### 去噪匹配损失 $L_{t-1}$  

&emsp;&emsp;去噪匹配损失 $L_{t-1}$ 在损失函数中占主导地位，其衡量了每个时间步 $t$，编码器的去噪后得到的图片与加噪过程中该时刻的真实图片的相似程度。在 $L_{t-1}$ 中我们最主要是需要计算 $D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t}))$，对于其中的编码器 $q(x_{t-1}|x_{t},x_{0})$，由贝叶斯公式以及马尔可夫性质可以得到：  

$$
\begin{align}
    & q(x_{t-1}|x_{t},x_{0}) = \frac{q(x_{t}|x_{t-1},x_{0})q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})} = \frac{q(x_{t}|x_{t-1})q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})} \tag{13} \\
    & q(x_{t}|x_{t-1}) = N(x_{t}; \sqrt{\alpha_{t}}x_{t-1}, (1-\alpha_{t})\boldsymbol{I}) \tag{14}
\end{align}
$$  

&emsp;&emsp;在前文中，我们已经得知了正向加噪过程满足递推公式：  

$$x_{t} = \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1-\alpha_{t}}\epsilon,\quad \epsilon \sim N(\epsilon; \boldsymbol{0,I})$$  

&emsp;&emsp;利用递推公式，我们可以计算出 $q(x_{t}|x_{0})$ 所满足的高斯分布：  

$$
\begin{align}
    q(x_{t}|x_{0}) = N(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})\boldsymbol{I}),\quad \bar{\alpha}_{t} = \prod_{i=1}^{t}\alpha_{i} \tag{15} \\
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    x_{t} &= \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1-\alpha_{t}}\epsilon_{t-1}^{*} \notag \\
    &= \sqrt{\alpha_{t}} \left( \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}^{*} \right) + \sqrt{1-\alpha_{t}}\epsilon_{t-1}^{*} \notag \\
    &= \sqrt{\alpha_{t}\alpha_{t-1}} x_{t-2} + \sqrt{\alpha_{t}-\alpha_{t}\alpha_{t-1}}\epsilon_{t-2}^{*} + \sqrt{1-\alpha_{t}}\epsilon_{t-1}^{*} \notag \\
    &= \sqrt{\alpha_{t}\alpha_{t-1}} x_{t-2} + \sqrt{\sqrt{\alpha_{t}-\alpha_{t}\alpha_{t-1}}^{2} + \sqrt{1-\alpha_{t}}^{2}}\epsilon_{t-2} \notag \\
    &= \sqrt{\alpha_{t}\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t}\alpha_{t-1}}\epsilon_{t-2} \notag \\
    &= \dotsb \notag \\
    &= \sqrt{\prod_{i=1}^{t}\alpha_{i}}x_{0} + \sqrt{1-\prod_{i=1}^{t}\alpha_{i}} \epsilon_{0} \notag \\
    &= \sqrt{\bar{\alpha}_{t}}x_{0} + \sqrt{1-\bar{\alpha}_{t}}\epsilon_{0} \sim N(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})\boldsymbol{I}) \notag \\
\end{align}
$$  

&emsp;&emsp;利用 (15) 式，我们可以得到 $x_{t-1}$ 的分布：  

$$
\begin{align}
    q(x_{t-1}|x_{0}) = N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_{0}, (1 - \bar{\alpha}_{t-1})\boldsymbol{I}) \tag{16} \\
\end{align}
$$

&emsp;&emsp;联立(14)、(15)、(16)式，我们可以发现(13)式的分子分母均为高斯分布，由高斯分布的联合分布与边际分布均为高斯分布可知， $q(x_{t-1}|x_{t},x_{0})$，同样满足高斯分布，现在我们需要利用(13)式来计算其均值与方差，通过计算我们可以得到如下结论：  

$$
\begin{align}
    q(x_{t-1}|x_{t},x_{0}) = N(x_{t-1}; \mu_{q}(x_{t},x_{0}), \Sigma_{q}(t)) \tag{17} \\
\end{align}
$$  

$$
\begin{align}
    \mu_{q}(x_{t},x_{0}) &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})x_{0}}{1-\bar{\alpha}_{t}},\quad \Sigma_{q}(t) = \sigma_{q}^{2}(t)\boldsymbol{I} = \frac{1-\alpha_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}\boldsymbol{I} \tag{18} \\
\end{align}
$$  

**$Proof$**

$$
\begin{align}
    q(x_{t-1}|x_{t},x_{0}) &= \frac{q(x_{t}|x_{t-1})q(x_{t-1}|x_{0})}{q(x_{t}|x_{0})} \notag \\
    &= \frac{N(x_{t}; \sqrt{\alpha_{t}}x_{t-1}, (1-\alpha_{t})\boldsymbol{I})N(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_{0}, (1 - \bar{\alpha}_{t-1})\boldsymbol{I})}{N(x_{t}; \sqrt{\bar{\alpha}_{t}}x_{0}, (1 - \bar{\alpha}_{t})\boldsymbol{I}),\quad \bar{\alpha}_{t}} \notag \\
    & \propto \exp \left(-\frac{1}{2}\left[ \frac{(x_{t}-\sqrt{\alpha_{t}}x_{t-1})^2}{1-\alpha_{t}} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_{0})^2}{1 - \bar{\alpha}_{t-1}} - \frac{(x_{t} - \sqrt{\bar{\alpha}_{t}}x_{0})^2}{1 - \bar{\alpha}_{t}} \right] \right)  \notag \\
    &= \exp\left( -\frac{1}{2}\left[ \frac{(-2\sqrt{\alpha_{t}}x_{t}x_{t-1}+\alpha_{t}x_{t-1}^{2})}{1-\alpha_{t}} + \frac{(x_{t-1}^{2}-2\sqrt{\bar{\alpha}_{t-1}}x_{t-1}x_{0})}{1 - \bar{\alpha}_{t-1}} + C(x_{t},x_{0}) \right] \right)  \notag \\
    & \propto \exp\left( -\frac{1}{2}\left[ \frac{1-\bar{\alpha}_{t}}{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}x_{t-1}^{2} - 2\left( \frac{\sqrt{\alpha_{t}}x_{t}}{1-\alpha_{t}} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_{0}}{1-\bar{\alpha}_{t-1}} \right)x_{t-1} \right] \right)   \notag \\
    &= \exp\left( -\frac{1}{2}\left( \frac{1-\bar{\alpha}_{t}}{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})} \right)\left[ x_{t-1}^{2}-2\frac{\left( \frac{\sqrt{\alpha_{t}}x_{t}}{1-\alpha_{t}} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_{0}}{1-\bar{\alpha}_{t-1}} \right)(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}x_{t-1} \right] \right)  \notag \\
    &= \exp\left( -\frac{1}{2}\left( \frac{1}{\frac{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}} \right)\left[ x_{t-1}^{2}-2\frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})x_{0}}{1-\bar{\alpha}_{t}}x_{t-1} \right] \right)  \notag \\
    & \propto N(x_{t+1}; \underbrace{\frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})x_{0}}{1-\bar{\alpha}_{t}}}_{\mu_{q}(x_{t},x_{0})}, \underbrace{\frac{(1-\alpha_{t})(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}\boldsymbol{I}}_{\Sigma_{q}(t)=\sigma_{q}^{2}(t)\boldsymbol{I}}) \notag \\
\end{align}
$$  

&emsp;&emsp;通过以上的推导，我们得到了加噪过程中$x_{t-1}$所满足的高斯分布，要计算 $L_{t-1}$ 式中的 KL Divergence，我们还需要去噪过程中 $x_{t-1}$ 的分布。  
&emsp;&emsp;为了使得去噪过程与加噪过程尽可能匹配，我们同样将去噪过程建模为高斯过程，即 $p_{\theta}(x_{t-1}|x_{t})$ 满足高斯分布。去噪过程的高斯分布的方差与对应的加噪过程的方差一致，而均值是由参数化的神经网络计算得出。对于均值的计算，VDM将 $p_{\theta}(x_{t-1}|x_{t})$ 所满足的高斯分布的均值设定为与 $q(x_{t-1}|x_{t},x_{0})$ 具有相同的形式，即 $\mu_{q}(x_{t},x_{0})$，但在去噪过程中是没有给定 $x_{0}$ 的，故神经网络在均值计算中的实际作用是输出 $x_{0}$ 的预测值 $\hat{x}_{\theta}(x_{t-1},t)$ ，从而得到 $p_{\theta}(x_{t-1}|x_{t})$ 所满足的高斯分布的均值。以上建模过程总结的数学表达式如下：  

$$
\begin{align}
    p_{\theta}(x_{t-1}|x_{t}) = N(x_{t-1}; \mu_{\theta}(x_{t},t), \Sigma_{p}(t))  \tag{19}
\end{align}
$$  

$$
\begin{align}
    \mu_{\theta}(x_{t},t) &= \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})\hat{x}_{\theta}(x_{t-1},t)}{1-\bar{\alpha}_{t}},\quad \Sigma_{p}(t) = \Sigma_{q}(t) = \frac{1-\alpha_{t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}\boldsymbol{I} \tag{20} \\
\end{align}
$$

&emsp;&emsp;通过以上的建模，去噪匹配损失 $L_{t-1}$ 中的 KL Divergence 实际上是计算两个方差相同的高斯分布的 KL Divergence，这使得问题变得非常简单，因为高斯分布的 KL Divergence 是有显式表达式的，其表达式如下所示：  

$$
\begin{align}
    D_{KL}(N(x;\mu_{x},\Sigma_{x}) || N(y;\mu_{y},\Sigma_{y})) = \frac{1}{2}\left[ \log{\frac{|\Sigma_{y}|}{|\Sigma_{x}|}}-d + tr(\Sigma_{y}^{-1}\Sigma_{x}) + (\mu_{y} - \mu_{x})^{T}\Sigma_{y}^{-1}(\mu_{y}-\mu_{x}) \right] \tag{21}
\end{align}
$$  

&emsp;&emsp;其中，$d$ 是高斯分布的维度。结合(17)、(18)、(19)、(20)、(21)式，我们现在可以来计算 $L_{t-1}$ 中的 KL Divergence 的具体表达式了，其结果如下：  

$$
\begin{align}
    D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) = \frac{1}{2\sigma_{q}^{2}(t)}\frac{\bar{\alpha}_{t-1}(1-\alpha_{t})^{2}}{(1-\bar{\alpha}_{t})^{2}} \left[ ||\hat{x}_{\theta}(x_{t-1},t) - x_{0}||_{2}^{2} \right] \tag{22}
\end{align}
$$  

**$Proof$**  

$$
\begin{align}
    & D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) \notag \\
    &= D_{KL}(N(x_{t-1}; \mu_{q}, \Sigma_{q}(t)) || N(x_{t-1}; \mu_{\theta}, \Sigma_{p}(t)))  \notag \\
    &= \frac{1}{2}\left[ \log{\frac{|\Sigma_{q}(t)|}{|\Sigma_{q}(t)|}}-d + tr(\Sigma_{q}(t)^{-1}\Sigma_{q}(t)) + (\mu_{q} - \mu_{\theta})^{T}\Sigma_{q}(t)^{-1}(\mu_{q}-\mu_{\theta}) \right] \notag \\
    &= \frac{1}{2}\left[ \log{1}-d + d + (\mu_{q} - \mu_{\theta})^{T}\Sigma_{q}(t)^{-1}(\mu_{q}-\mu_{\theta}) \right] \notag \\
    &= \frac{1}{2}\left[(\mu_{q} - \mu_{\theta})^{T}(\sigma_{q}^{2}(t)\boldsymbol{I})^{-1}(\mu_{q}-\mu_{\theta}) \right] \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[ ||\mu_{q} - \mu_{\theta} ||_{2}^{2} \right] \notag \\
    &=  \frac{1}{2\sigma_{q}^{2}(t)} \left[|| \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})x_{0}}{1-\bar{\alpha}_{t}}- \frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})x_{t} + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})\hat{x}_{\theta}(x_{t-1},t)}{1-\bar{\alpha}_{t}}||_{2}^{2}\right]\notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)} \left[|| \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_{t})}{1-\bar{\alpha}_{t}} (\hat{x}_{\theta}(x_{t-1},t) - x_{0}) ||_{2}^{2}\right]  \notag \\
    &= \frac{1}{2\sigma_{q}^{2}(t)}\frac{\bar{\alpha}_{t-1}(1-\alpha_{t})^{2}}{(1-\bar{\alpha}_{t})^{2}} \left[ ||\hat{x}_{\theta}(x_{t-1},t) - x_{0}||_{2}^{2} \right] \notag \\
\end{align}
$$  

&emsp;&emsp;对于(12)式中的期望，可以在每个训练批次中使用 Monte Carlo estimate 方法来估计。通过 (22) 式 我们可以得知，VDM 在逆向去噪过程中，每一步的优化目标都是在给定噪声图片 $x_{t-1}$ 的情况下，预测出原始图片 $x_{0}$。这种损失函数的设定方式可以总结为**预测原始数据**，在之后的章节我们会讨论扩散模型损失函数的另外两种等价形式，分别为**预测噪声**以及**分数匹配**。  

### 重构似然损失 $L_{0}$  

&emsp;&emsp;对于损失函数中的 $L_{0}$ 项，在 DDPM 的原始论文[2] 中，采用了一个独立的离散编码器。这是因为在之前的加噪去噪步骤中，我们都将图片数据的取值由 $\{ 0,1,\dotsb,255 \}$ 的离散数值映射到 $[-1,1]$。这一方面是使数据标准化，便于神经网络处理；另一方面是因为采样过程是从标准高斯分布中进行采样，再由解码器逐步去噪，故需要在训练时的加噪去噪过程对离散数据进行映射后来匹配采样时的数值范围。  
&emsp;&emsp;具体来讲，由之前步骤训练得到的参数化的神经网络，以及去噪得到的数据 $x_{1}$，我们可以同样可以得到 $p_{\theta}(x_{0}|x_{1})$ 所满足的高斯分布 $N(x_{0}; \mu_{\theta}(x_{1},1),\sigma_{1}^{2})$，但在这里我们不能像去噪过程一样，直接使用该高斯分布来计算 $x_{0}$ 的对数似然，这是因为原始数据 $x_{0}$ 是离散数据，而高斯分布本身是连续的，直接使用高斯分布来计算离散对数本事是不够准确的。在 DDPM 的原始论文[2] 中，作者采用了一个离散边界函数计算高斯分布在离散区间上的积分，从而精确计算离散数据的对数似然。其计算公式如下：  

$$
\begin{align}
    p_{\theta}(x_{0}|x_{1}) = \prod_{i=1}^{d} \int_{\delta_{-}(x_{0}^{i})}^{\delta_{+}(x_{0}^{i})} N(x_{0}; \mu_{\theta}(x_{1},1),\sigma_{1}^{2}) dx \tag{22} \\
\end{align}
$$  

$$
\begin{align}
    \delta_{+}(x_{0}^{i}) = \left \{
\begin{array}{l}
\infty & if \ x = 1 \\
x + \frac{1}{255} & if \ x < 1 \\
\end{array} \right. \quad \delta_{-}(x_{0}^{i}) = \left \{
\begin{array}{l}
-\infty & if \ x = -1 \\
x - \frac{1}{255} & if \ x > -1 \\
\end{array} \right.
\tag{23}
\end{align}
$$  

&emsp;&emsp;其中，$i$ 表示数据 $x_{0}$ 的每一个维度，$\delta_{+}(x_{0}^{i}),\delta_{-}(x_{0}^{i})$ 表示每个数据维度的离散值的上下边界。由于在做数据映射时，是使用 $x' = (2x - 255) / 255$ 将数据取值范围由 $\{ 0,1,\dotsb,255 \}$ 映射到 $[0,1]$的，故 $x' \pm \frac{1}{255} = [2(x \pm 0.5) -255]/255$，故(23)式设置的离散边界实际上是等价于在离散值上下分别加上0.5后再计算高斯分布的积分。使用这种方法可以较为精确的计算离散数据的对数似然。  

## 总结  

&emsp;&emsp;在这篇博客中，我们首先讨论了 VDM 相较于之前的 MHVAE 又做了哪些假设，以及做出这些假设的 motivations。之后与 VAE 类似，我们讨论了 VDM 的 ELBO 的理论表达式，其可以分解为三项，在此基础上，我们计算了 ELBO 三个分解项在损失函数中的具体表达式。  
&emsp;&emsp;但实际上，DDPM 在训练中并不是使用**预测原始数据**，即(22)式作为损失函数，而是使用 **预测噪声** 作训损失函数，在之后的一些研究中，例如基于分数的生成模型，则是使用 **分数匹配** 作为损失函数，这三种扩散模型的损失函数实际上是等价的，这一点由于篇幅的限制我们不再做过多的讨论。在下一节，我们将重点讨论三种损失函数的等价关系，如何做训练以及采样，以及扩散模型系数该如何设置或学习。


## Reference  

**[1] Paper: Luo C. Understanding diffusion models: A unified perspective[J]. arXiv preprint arXiv:2208.11970, 2022.**  
**[2] Paper: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.**  
**[3] Video: 想不出来昵称又想改, 扩散模型-Diffusion Model【李宏毅2023】, Blibili**  
**[4] Blog: 苏剑林, 生成扩散模型漫谈(1-3), 科学空间**
