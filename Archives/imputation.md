## 缺失数据机制理论

在缺失数据研究中，Rubin（1976）提出的三大缺失机制是基础理论框架，广泛应用于统计学、机器学习和时间序列分析任务中：

1. **完全随机缺失（Missing Completely At Random, MCAR）**  
   缺失的发生与观测值或未观测值无关，即缺失数据在总体中是随机分布的。  
   - **数学表达**：  
     \[
     P(M \mid X_{obs}, X_{mis}) = P(M)
     \]  
     其中 \( M \) 表示缺失指示矩阵，\( X_{obs} \) 表示已观测值，\( X_{mis} \) 表示缺失值。
   - **示例**：传感器偶发网络掉线导致的测量丢失。
   - **插补方法适用性**：均值插补、KNN插补、随机森林插补等传统方法效果良好。

2. **随机缺失（Missing At Random, MAR）**  
   缺失的发生与已观测变量有关，但与缺失值本身无关。  
   - **数学表达**：  
     \[
     P(M \mid X_{obs}, X_{mis}) = P(M \mid X_{obs})
     \]
   - **示例**：医院在患者病情较轻时不测量某些指标，但具体缺失与该指标真实值无直接关系。
   - **插补方法适用性**：多重插补（MICE）、条件生成模型、变分自编码器等。

3. **非随机缺失（Missing Not At Random, MNAR）**  
   缺失的发生与缺失值本身有关，即缺失机制依赖于未观测的真实值。  
   - **数学表达**：  
     \[
     P(M \mid X_{obs}, X_{mis}) \neq P(M \mid X_{obs})
     \]
   - **示例**：患者病情严重时某项指标异常，医生选择不测量或测量失败。
   - **插补方法适用性**：需要机制建模（例如联合建模法、选择模型、模式混合模型）。

> **2025年研究进展**  
> 近年来，针对复杂应用（尤其是时间序列和多模态场景），研究者提出了**混合缺失机制（Hybrid Missingness）**、**因果驱动缺失（Causal Missingness）**等新概念，强调缺失机制可能在同一数据集内混合存在，并与外部因果关系相关。这一趋势推动了**机制感知（Mechanism-aware）Imputation**方法的发展。


## 常用时序 Imputation 数据集与缺失机制分析

| 领域         | 典型数据集                           | 主要缺失机制               | 备注                               |
| ------------ | ------------------------------------ | -------------------------- | ---------------------------------- |
| 电力能源     | Electricity, ETT, Solar              | MCAR、MAR                  | 主要为随机丢失                     |
| 交通         | PeMS, METR-LA                        | MAR、IRM、少量AM           | 具时空相关，缺失较复杂              |
| 空气质量     | BeijingAir, ItalyAir                 | MCAR、MAR、部分AM          | 极端环境下缺失机制复杂              |
| 医疗健康     | PhysioNet 2012/2019                   | MNAR                       | 与病情相关，非随机缺失              |
| 船舶/金融等  | Vessel AIS, Exchange Rate            | AM、PO、MCAR               | 包含设备异常与计划停机缺失          |

---

## 论文列表


1. **TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation**  
   - 链接: [https://icml.cc/virtual/2025/poster/43701](https://icml.cc/virtual/2025/poster/43701)  
   - 作者: Daoyu Wang, Mingyue Cheng, Zhiding Liu, Qi Liu  
   - 关键词: 预测，自回归，自监督
   - **Abstract**: Self-supervised learning has garnered increasing attention in time series analysis for benefiting various downstream tasks and reducing reliance on labeled data. Despite its effectiveness, existing methods often struggle to comprehensively capture both long-term dynamic evolution and subtle local patterns in a unified manner. In this work, we propose TimeDART, a novel self-supervised time series pre-training framework that unifies two powerful generative paradigms to learn more transferable representations. Specifically, we first employ a causal Transformer encoder, accompanied by a patch-based embedding strategy, to model the evolving trends from left to right. Building on this global modeling, we further introduce a denoising diffusion process to capture fine-grained local patterns through forward diffusion and reverse denoising. Finally, we optimize the model in an autoregressive manner. As a result, TimeDART effectively accounts for both global and local sequence features in a coherent way. We conduct extensive experiments on public datasets for time series forecasting and classification. The experimental results demonstrate that TimeDART consistently outperforms previous compared methods, validating the effectiveness of our approach. Our code is available at https://github.com/Melmaphother/TimeDART.
   - **动机**：当前自监督时间序列学习方法在捕捉长程动态与局部细节方面存在挑战：（1）基于自动回归（autoregressive）的方法虽符合时序自然趋势，但容易过拟合噪声、异常值；（2）扩散（diffusion）模型擅长恢复细节，但弱化全局依赖性。理想的自监督方法应同时兼具全局趋势建模和细粒度特征提取，以提升下游任务性能 。
   - **方法简述（Proposed Method）**：
    - 自回归 Transformer 编码器（causal Transformer）：采用 patch 级别嵌入与因果遮掩，捕捉序列全局动态；
    - 扩散+去噪机制：在每个 patch 中注入噪声，通过跨注意力（cross-attention）向经典 Transformer 提供修复信号，促进模型捕获局部结构；
   - **训练目标**
    - 使用扩散损失替代单纯 MSE，避免自动回归模型对高斯偏差的假设，允许对多模态分布更丰富建模 ；
    - 同时训练 autoregressive 与 diffusion 任务，以一致的特征学习目标提升表征质量。
   - **实验设置与数据集**
    - 时间序列预测（Forecasting）：PEMS（交通流量）、ETTh2, ETTm2（能源），Electricity 
    - 时间序列分类（Classification）：HAR（人体活动识别，来自可穿戴设备）、Epilepsy（癫痫发作 ECG 信号）、Sleep‑EEG（多通道 EEG 睡眠阶段分类
   ![TimeDART](./img/TimeDART.png "TimeDART")


11. **LSCD: Lomb--Scargle Conditioned Diffusion for Irregular Time series Imputation**  
    - 链接: [https://icml.cc/virtual/2025/poster/45821](https://icml.cc/virtual/2025/poster/45821)  
    - 作者: Elizabeth M Fons Etcheverry, Alejandro Sztrajman, Yousef El-Laham, Luciana Ferrer, Svitlana Vyetrenko, Manuela Veloso  
    - 关键词: 插补，不规则时间序列，扩散

12. **VerbalTS: Generating Time Series from Texts**  
    - 链接: [https://icml.cc/virtual/2025/poster/45631](https://icml.cc/virtual/2025/poster/45631)