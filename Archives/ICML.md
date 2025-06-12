# ICML 2025 | 时间序列(Time Series)论文总结


**会议时间**: 2025年7月13日至7月19日  
**地点**: 温哥华会议中心  
**论文总数**: 63篇  

## 时间序列主题
预测，分类，异常检测，生成，因果发现，基础模型，大语言模型等。

---

## 论文列表

### 预测相关
1. **TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation**  
   - 链接: [https://icml.cc/virtual/2025/poster/43701](https://icml.cc/virtual/2025/poster/43701)  
   - 作者: Daoyu Wang, Mingyue Cheng, Zhiding Liu, Qi Liu  
   - 关键词: 预测，自回归，自监督
   - Abstract: Self-supervised learning has garnered increasing attention in time series analysis for benefiting various downstream tasks and reducing reliance on labeled data. Despite its effectiveness, existing methods often struggle to comprehensively capture both long-term dynamic evolution and subtle local patterns in a unified manner. In this work, we propose TimeDART, a novel self-supervised time series pre-training framework that unifies two powerful generative paradigms to learn more transferable representations. Specifically, we first employ a causal Transformer encoder, accompanied by a patch-based embedding strategy, to model the evolving trends from left to right. Building on this global modeling, we further introduce a denoising diffusion process to capture fine-grained local patterns through forward diffusion and reverse denoising. Finally, we optimize the model in an autoregressive manner. As a result, TimeDART effectively accounts for both global and local sequence features in a coherent way. We conduct extensive experiments on public datasets for time series forecasting and classification. The experimental results demonstrate that TimeDART consistently outperforms previous compared methods, validating the effectiveness of our approach. Our code is available at https://github.com/Melmaphother/TimeDART.
   - 动机：当前自监督时间序列学习方法在捕捉长程动态与局部细节方面存在挑战：（1）基于自动回归（autoregressive）的方法虽符合时序自然趋势，但容易过拟合噪声、异常值；（2）扩散（diffusion）模型擅长恢复细节，但弱化全局依赖性。理想的自监督方法应同时兼具全局趋势建模和细粒度特征提取，以提升下游任务性能 。
   - 方法简述（Proposed Method）：
    - 自回归 Transformer 编码器（causal Transformer）：采用 patch 级别嵌入与因果遮掩，捕捉序列全局动态；
    - 扩散+去噪机制：在每个 patch 中注入噪声，通过跨注意力（cross-attention）向经典 Transformer 提供修复信号，促进模型捕获局部结构；
   - 训练目标
    - 使用扩散损失替代单纯 MSE，避免自动回归模型对高斯偏差的假设，允许对多模态分布更丰富建模 ；
    - 同时训练 autoregressive 与 diffusion 任务，以一致的特征学习目标提升表征质量。
   - 实验设置与数据集
    - 时间序列预测（Forecasting）：PEMS（交通流量）、ETTh2, ETTm2（能源），Electricity 
    - 时间序列分类（Classification）：HAR（人体活动识别，来自可穿戴设备）、Epilepsy（癫痫发作 ECG 信号）、Sleep‑EEG（多通道 EEG 睡眠阶段分类


2. **Towards a General Time Series Forecasting Model with Unified Representation and Adaptive Transfer**  
   - 链接: [https://icml.cc/virtual/2025/poster/46383](https://icml.cc/virtual/2025/poster/46383)  
   - 作者: Yihang Wang, Yuying Qiu, Peng Chen, Kai Zhao, Yang Shu, Zhongwen Rao, Lujia Pan, Bin Yang, Chenjuan Guo  
   - 关键词: 预测，少样本，零样本

3. **TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting**  
   - 链接: [https://icml.cc/virtual/2025/poster/46502](https://icml.cc/virtual/2025/poster/46502)  
   - 作者: Yifan Hu, Guibin Zhang, Peiyuan Liu, Disen Lan, Naiqi Li, Dawei Cheng, Tao Dai, Shutao Xia, Shirui Pan  
   - 关键词: 预测，时空图，通道关系

4. **Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization**  
   - 链接: [https://icml.cc/virtual/2025/poster/46131](https://icml.cc/virtual/2025/poster/46131)  
   - 作者: Luca Masserano, Abdul Fatir Ansari, Boran Han, Xiyuan Zhang, Christos Faloutsos, Michael Mahoney, Andrew Wilson, Youngsuk Park, Syama Sundar Yadav Rangapuram, Danielle Maddix, Yuyang Wang  
   - 关键词: 预测，基础模型，小波变换，token化

5. **Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting**  
   - 链接: [https://icml.cc/virtual/2025/poster/44762](https://icml.cc/virtual/2025/poster/44762)  
   - 作者: Siru Zhong, Weilin Ruan, Ming Jin, Huan Li, Qingsong Wen, Yuxuan Liang  
   - 关键词: 预测，多模态，视觉语言模型

6. **Lightweight Online Adaption for Time Series Foundation Model Forecasts**  
   - 链接: [https://icml.cc/virtual/2025/poster/44485](https://icml.cc/virtual/2025/poster/44485)  
   - 作者: Thomas Lee, William Toner, Rajkarn Singh, Artjom Joosen, Martin Asenov  
   - 关键词: 预测，基础模型，在线学习

7. **TimeStacker: A Novel Framework with Multilevel Observation for Capturing Nonstationary Patterns in Time Series Forecasting**  
   - 链接: [https://icml.cc/virtual/2025/poster/46428](https://icml.cc/virtual/2025/poster/46428)  
   - 作者: Qinglong Liu, Cong Xu, Wenhao Jiang, Kaixuan Wang, Lin Ma, Haifeng Li  
   - 关键词: 预测，非平稳性，多尺度

8. **AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting**  
   - 链接: [https://icml.cc/virtual/2025/poster/43518](https://icml.cc/virtual/2025/poster/43518)  
   - 作者: Abdelhakim Benechehab, Vasilii Feofanov, Giuseppe Paolo, Albert Thomas, Maurizio Filippone, Balázs Kégl  
   - 关键词: 预测，基础模型，单变量，概率预测

9. **TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state**  
   - 链接: [https://icml.cc/virtual/2025/poster/43851](https://icml.cc/virtual/2025/poster/43851)  
   - 作者: Xiaowen Ma, Zhen-Liang Ni, Shuai Xiao, Xinghao Chen  
   - 关键词: 长时预测，变量感知，时间感知

10. **Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting**  
    - 链接: [https://icml.cc/virtual/2025/poster/43827](https://icml.cc/virtual/2025/poster/43827)  
    - 作者: Zhining Liu, Ze Yang, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Yada Zhu, Hendrik Hamann, Jingrui He, Hanghang Tong  
    - 关键词: 预测，自适应

11. **TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting**  
    - 链接: [https://icml.cc/virtual/2025/poster/43973](https://icml.cc/virtual/2025/poster/43973)  
    - 作者: Peiyuan Liu, Beiliang Wu, Yifan Hu, Naiqi Li, Tao Dai, Jigang Bao, Shutao Xia  
    - 关键词: 长时预测，非平稳性

12. **HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting**  
    - 链接: [https://icml.cc/virtual/2025/poster/43741](https://icml.cc/virtual/2025/poster/43741)  
    - 作者: Boyuan Li, Yicheng Luo, Zhen Liu, Junhao Zheng, Jianming Lv, Qianli Ma  
    - 关键词: 预测，超图，不规则多元时序

13. **LETS Forecast: Learning Embedology for Time Series Forecasting**  
    - 链接: [https://icml.cc/virtual/2025/poster/45595](https://icml.cc/virtual/2025/poster/45595)  
    - 作者: Abrar Majeedi, Viswanatha Reddy Gajjala, Satya Sai Srinath Namburi GNVV, Nada Elkordi, Yin Li  
    - 关键词: 预测，经验动态建模

14. **Privacy Amplification by Structured Subsampling for Deep Differentially Private Time Series Forecasting**  
    - 链接: [https://icml.cc/virtual/2025/poster/44722](https://icml.cc/virtual/2025/poster/44722)  
    - 作者: Jan Schuchardt, Mina Dalirrooyfard, Jed Guzelkabaagac, Anderson Schneider, Yuriy Nevmyvaka, Stephan Günnemann  
    - 关键词: 预测，差分隐私

15. **TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting**  
    - 链接: [https://icml.cc/virtual/2025/poster/45815](https://icml.cc/virtual/2025/poster/45815)  
    - 作者: Qihe Huang, Zhengyang Zhou, Kuo Yang, Zhongchao Yi, Xu Wang, Yang Wang  
    - 关键词: 预测，极简主义，高效性

16. **CFPT: Empowering Time Series Forecasting through Cross-Frequency Interaction and Periodic-Aware Timestamp Modeling**  
    - 链接: [https://icml.cc/virtual/2025/poster/44425](https://icml.cc/virtual/2025/poster/44425)  
    - 作者: Feifei Kou, Jiahao Wang, Lei Shi, Yuhan Yao, Yawen Li, Suguo Zhu, Zhongbao Zhang, Junping Du  
    - 关键词: 预测，时间戳建模，跨频交互

17. **Winner-takes-all for Multivariate Probabilistic Time Series Forecasting**  
    - 链接: [https://icml.cc/virtual/2025/poster/46485](https://icml.cc/virtual/2025/poster/46485)  
    - 作者: Adrien Cortes, Remi Rehm, Victor Letzelter  
    - 关键词: 预测，多元概率预测

18. **Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting**  
    - 链接