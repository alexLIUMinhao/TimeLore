# 多模态时序分析

## Motivation
1. 随着深度学习兴起，长短期记忆网络（LSTM）和卷积神经网络（CNN）等模型在建模这些复杂性方面取得了显著提升。然而，它们仍受限于仅依赖数值数据，无法融合外部情境信息——例如**专家见解**或**宏观经济**事件——而这些信息本可提高预测的准确率与可解释性。在金融、病患监护等混沌或高波动的系统中，这一缺陷尤为突出。
近年来，基于 **Transformer** 的架构以及**大语言模型（LLM）**的发展增强了捕获时间序列长程依赖的能力。但单独使用这些模型时，它们往往错失其他模态（如文本）所能提供的关键领域知识。在实际场景中，领域专家可能希望给出指令或洞见以引导预测过程，而现有模型难以灵活接纳此类交互。


---

## 论文列表
1. 

4. **Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization**  
   - **链接**: [https://icml.cc/virtual/2025/poster/46131](https://icml.cc/virtual/2025/poster/46131)  
   - **作者**: Luca Masserano, Abdul Fatir Ansari, Boran Han, Xiyuan Zhang, Christos Faloutsos, Michael Mahoney, Andrew Wilson, Youngsuk Park, Syama Sundar Yadav Rangapuram, Danielle Maddix, Yuyang Wang 论文由 Cornell University 和 多个工业团队（包括 AWS AI、Google Research、Stanford 等）合作完成 
   - **关键词**: 预测，基础模型，小波变换，token化
   - **Abstract**: How to best develop foundational models for time series forecasting remains an important open question. Tokenization is a crucial consideration in this effort: what is an effective discrete vocabulary for a real-valued sequential input? To address this question, we develop WaveToken, a wavelet-based tokenizer that allows models to learn complex representations directly in the space of time-localized frequencies. Our method first scales and decomposes the input time series, then thresholds and quantizes the wavelet coefficients, and finally pre-trains an autoregressive model to forecast coefficients for the forecast horizon. By decomposing coarse and fine structures in the inputs, wavelets provide an eloquent and compact language for time series forecasting that simplifies learning. Empirical results on a comprehensive benchmark, including 42 datasets for both in-domain and zeroshot settings, show that WaveToken: i) provides better accuracy than recently proposed foundation models for forecasting while using a much smaller vocabulary (1024 tokens), and performs on par or better than modern deep learning models trained specifically on each dataset; and ii) exhibits superior generalization capabilities, achieving the best average rank across all datasets for three complementary metrics. In addition, we show that our method can easily capture complex temporal patterns of practical relevance that are challenging for other recent pre-trained models, including trends, sparse spikes, and non-stationary time series with varying frequencies evolving over time.
   - **动机**：在构建 时间序列基础模型（foundation models） 时，关键挑战之一是如何将连续时间信号有效离散成可学习的 token。作者指出，现有 tokenization 方法往往要么过度依赖粗粒度采样、导致信息丢失；要么词表过大、难以训练和泛化。
   - **方法简述（Proposed Method）**：
    - 提出 WaveToken——一种基于小波变换的 tokenization 技术：将原始时序按不同时频尺度分解，量化阈值后形成有限词汇表（如 1024 tokens）；然后预训练自回归模型去预测未来小波系数，从而在频域结构上学习时间序列特征。该方法既保持了时频信息，又显著减少词表复杂度和显存占用。
   - **实验设置与数据集**
    - WaveToken 在包含 42（Electricity、Traffic（PEMS）、ETT、ETTm ） 个数据集的全面 benchmark 上评测，覆盖 in-domain 和 zero-shot 场景，结果显示其在多个常用任务中超越或匹配现有基础模型和针对性深度模型，并在泛化能力上表现优异。
   ![WaveToken](./img/WaveToken.png "WaveToken")

5. **Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting**  
   - **链接**: [https://icml.cc/virtual/2025/poster/44762](https://icml.cc/virtual/2025/poster/44762)  
   - **作者**: Siru Zhong, Weilin Ruan, Ming Jin, Huan Li, Qingsong Wen, Yuxuan Liang  
   - **关键词**: 预测，多模态，视觉语言模型
   - **Abstract**: Recent advancements in time series forecasting have explored augmenting models with text or vision modalities to improve accuracy. While text provides contextual understanding, it often lacks fine-grained temporal details. Conversely, vision captures intricate temporal patterns but lacks semantic context, limiting the complementary potential of these modalities.（虽然文本可以提供上下文理解，但它通常缺乏细粒度的时间细节。相反，视觉能够捕捉复杂的时间模式，但缺乏语义背景，从而限制了这两种模态的互补潜力。） To address this, we propose Time-VLM, a novel multimodal framework that leverages pre-trained Vision-Language Models (VLMs) to bridge temporal, visual, and textual modalities for enhanced forecasting. Our framework comprises three key components: (1) a Retrieval-Augmented Learner, which extracts enriched temporal features through memory bank interactions; (2) a Vision-Augmented Learner, which encodes time series as informative images; and (3) a Text-Augmented Learner, which generates contextual textual descriptions. These components collaborate with frozen pretrained VLMs to produce multimodal embeddings, which are then fused with temporal features for final prediction. Extensive experiments demonstrate that Time-VLM achieves superior performance, particularly in few-shot and zeroshot scenarios, thereby establishing a new direction for multimodal time series forecasting. Code is available at https://github.com/CityMind-Lab/ICML25-TimeVLM.
   - **动机**：时间序列预测在金融、气象、能源等领域具有重要应用。虽然已有模型尝试引入文本或视觉信息增强预测性能，但单一模态的方法仍存在语义理解不足或缺乏时序结构等局限。当前缺乏一个能同时整合文本、图像和时间序列数据的统一模型。因此，作者提出探索如何利用预训练视觉-语言模型（VLMs），统一三种模态信息，从而提升在数据稀缺场景下的预测性能与泛化能力。
   - **方法简述（Proposed Method）**：作者提出了Time-VLM，这是一个新颖的多模态预测框架，包含三个模块：（1）Retrieval-Augmented Learner (RAL) 用于从时间序列中提取丰富的时序特征；（2）Vision-Augmented Learner (VAL) 将时间序列转换为图像，以捕捉时空结构特征；（3）Text-Augmented Learner (TAL) 生成与时间序列相关的上下文语义文本。三个模块的输出通过**冻结的预训练VLM（如ViLT、CLIP）**进行融合，再输入至预测器生成最终预测结果。该框架无需外部图像或文本数据，能自行生成辅助模态以增强自身预测。
   - **实验设置与数据集**：
    作者在多个时间序列数据集上进行了实验，涵盖能源（ETTh1, ETTh2, ETTm1, ETTm2）、气象（Weather）、电力（ECL）、交通（Traffic）以及短期预测基准数据集M4，评估包括全监督、少样本（few-shot）与零样本（zero-shot）等场景。Time-VLM在多个指标（如MSE、MAE、SMAPE等）上均显著优于现有SOTA模型，尤其在数据稀缺条件下展现出强大的泛化能力。
   ![Time-VLM](./img/Time-VLM.png "Time-VLM")

