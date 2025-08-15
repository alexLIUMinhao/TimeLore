# 多模态时序分析

## Motivation
1. 随着深度学习兴起，长短期记忆网络（LSTM）和卷积神经网络（CNN）等模型在建模这些复杂性方面取得了显著提升。然而，它们仍受限于仅依赖数值数据，无法融合外部情境信息——例如**专家见解**或**宏观经济**事件——而这些信息本可提高预测的准确率与可解释性。在金融、病患监护等混沌或高波动的系统中，这一缺陷尤为突出。
近年来，基于 **Transformer** 的架构以及**大语言模型(LLM)**的发展增强了捕获时间序列长程依赖的能力。但单独使用这些模型时，它们往往错失其他模态（如文本）所能提供的关键领域知识。在实际场景中，领域专家可能希望给出指令或洞见以引导预测过程，而现有模型难以灵活接纳此类交互。  

2. 传统时序预测普遍假设未来延续历史趋势与周期，强调被动预测——即回答“未来会怎样”。然而在气象、能源、金融等领域，分析师常需要主动塑形未来：注入经验或突发信息，调整曲线形态、引入特定事件或改变局部波动，以模拟不同情景，这超出了传统预测的范畴。现有文本辅助方法虽然允许自然语言输入，但多将其作为静态特征简单融合，缺乏两项核心能力：**（1）时间感知的动态权衡**，无法在不同时间段平衡历史模式与指令意图，既可人工调节权重，也可由模型自动调整；**（2）语义–时间的精细对齐**，难以将文本描述的趋势、事件和幅度精确映射到预测曲线上；通过多尺度语义–时间建模，将文本中全局趋势与局部事件映射到不同时间尺度，同时结合历史时序的统计特性与信号分解结果，实现文本语义与未来曲线的精细对齐，保证编辑结果的物理合理性和可解释性。**此外**，这些方法几乎不提供文本干预下的概率置信区间，使分析师难以评估指令影响与风险。为此，我们提出文本驱动的可控时序编辑框架，通过动态加权、语义–时间对齐及不确定性生成，使模型在保留历史规律的同时，能够根据自然语言指令主动“编辑”未来走势，并量化干预影响，实现从被动预测到可控生成与未来塑形的范式转变。

3. 在现实世界的时序预测任务中，研究者往往不仅需要被动地回答“未来会怎样”，还需要根据领域知识、突发事件或特定目标**主动塑形未来**，例如模拟政策干预后的经济走势，或基于运维策略调整能源系统的负荷曲线。现有自回归预测方法通常在局部时间窗口（look-back window）内，通过建模短期依赖来生成未来序列，但缺乏将人类高层语义意图直接融入预测过程的能力，因而难以支持**基于自然语言指令的时序编辑**。
    
    为此，我们提出一种**多层次动作描述驱动的可控时序生成范式**：首先，将连续的时间步（如 3–5 个点的预测区间）视为一个 token，并在真实数据上打上文本化的局部动作标签（local action description），这些标签通过预定义的规则或领域基准刻画局部动态模式（如“缓慢上升”“周期回落”）。在此基础上，训练一个自回归模型，使其能够根据局部文本动作指令预测下一个时序 token，实现可解释的短期生成。  
然而，单纯的局部生成难以捕捉长周期趋势、全局约束及领域知识。因此，我们进一步引入**全局动作描述（global action description）**，用于刻画长期模式及各种约束，包括：
    1. **物理约束**：如设备安全阈值、电力负荷或温度上下界，保证生成序列符合物理可能性；  
    2. **统计约束**：如均值、方差或波动范围，确保序列整体分布合理；  
    3. **领域逻辑规则**：如业务规则、操作限制或调度约束，使生成序列在业务场景中可行。  
    全局描述与局部动作指令联合作用，使模型能够在生成过程中同时满足局部可控性和全局一致性。例如，用户可以通过自然语言指令“保持当前震荡频率，但将幅度降低 20%”主动编辑未来曲线，同时确保序列不会超过安全阈值或违背领域规则。  
这一范式的核心创新性在于：
    - **从被动预测到主动编辑**：模型支持通过自然语言直接干预未来轨迹；  
    - **多尺度语义融合**：局部动作与全局约束联合建模，实现短期灵活性与长期一致性平衡；  
    - **跨模态可控生成**：自然语言作为控制信号嵌入时序预测，赋予模型可解释性与交互性。  
    这种方法为交互式时序预测和假设情景模拟提供了新的范式，使模型不仅能回答“未来会怎样”，还能基于高层语义主动塑形未来轨迹。


---

## 论文列表
1. **Instruction-Following LLMs for Time Series Prediction: A Two-Stage Multimodal Approach**
    - **链接**: https://openreview.net/forum?id=01wMplF8TL
    - **作者**: Anonymous authors (Paper under double-blind review)
    - **关键词**: Time series prediction, LLMs, multimodal, instruction-following, text integration
    - **abstract**: We introduce Text-Informed Time Series Prediction (TITSP), an innovative multimodal framework that integrates textual knowledge with temporal dynamics using Large Language Models (LLMs). TITSP employs a two-stage process that bridges numerical data with rich contextual information for enhanced forecasting accuracy and interpretability. In the first stage, we present AutoPrompter, which captures temporal dependencies from time series data and aligns them with semantically meaningful text embeddings. In the second stage, these aligned embeddings are refined by incorporating task-specific textual instructions through LLM. We evaluate TITSP on several multimodal time series prediction tasks, demonstrating substantial improvements over state-of-the-art baselines. Quantitative results reveal significant gains in predictive performance, while qualitative analyses show that textual context enhances interpretability and actionable insights. Our findings indicate that integrating multimodal inputs not only improves prediction accuracy but also fosters more intuitive, user-centered forecasting.
    - **动机**：时序预测在金融、医疗和气候科学等领域至关重要，用于支持决策。传统方法如ARIMA难以处理复杂的非线性模式和长距离依赖，而深度学习模型如LSTM和CNN虽有所改进，但仅依赖数值数据，限制了外部上下文信息（如专家见解或宏观经济事件）的整合。这在波动性强的系统中（如金融市场或患者健康监测）尤为问题。Transformer基模型和LLM的最新进展提升了依赖捕捉，但缺乏与领域特定文本见解的整合。TITSP通过结合深度学习处理时序与LLM整合文本输入，解决这些挑战，实现更准确、上下文感知且可解释的预测，尤其在需要专家输入的场景中。
    - **方法简述（Proposed Method）**：提出的方法Text-Informed Time Series Prediction (TITSP)  
        - 第一阶段，(Stage 1: AutoPrompter) 的工作机制：AutoPrompter 是 TITSP 框架的第一阶段，其核心目标是将时序数据转换为压缩的、语义丰富的文本嵌入空间。通过修改后的 Vector Quantized-Variational AutoEncoder (VQ-VAE) 结合交叉注意力机制和预训练语言码本，实现时序嵌入与文本嵌入的对齐，从而捕捉潜在的语义模式，为后续文本上下文整合奠定基础。该阶段采用自监督学习方式训练，强调时序数据的语义表示学习，而非直接预测。
        - 第二阶段，(Stage 2: Supervised Multimodal Fusion for Prediction) 的工作机制第二阶段通过整合任务特定文本指令（如“趋势上升”）来精炼第一阶段的嵌入，利用大型语言模型 (LLM) 进行监督时序预测。该阶段强调条件预测，提升准确性和可解释性，采用监督学习方式，焦点在于多模态融合和最终预测生成。

    - **实验设置与数据集**：使用的数据集包括ETTh1、ETTh2、ETTm1、ETTm2、天气、交通、电力、汇率，以及Lorenz时序，使用合成数据生成并融入文本指令如“增加”、“减少”或“稳定”。评估指标为均方误差 (MSE)、平均绝对误差 (MAE) 和遵守率（预测遵守指令的比例）。比较的基线包括Time-LLM、Qwen4MTS、UniTime、Llama-3.1-8B、GPT4MTS、itransformer和PatchTST。主要结果显示TITSP性能优越，具有高遵守率（如“保持稳定”为0.98）和低MSE（如“保持稳定”为0.35），比基线更好，证明了强大的零样本泛化和关键词提取能力，尤其在处理长序列和指令遵守方面。
    - **审稿人对论文的弱点评价**：
        - 创新性（普遍认可）：所有 reviewer 都肯定“把文本指令引入时间序列预测”这一思路本身有价值，认为把 LLM 当作可插拔的第二阶段来融合专家知识具有启发性。
        - 方法细节与可复现性（最大争议点）： R1 & R3 指出论文对 AutoPrompter 如何把数值序列映射为“语义有意义的文本嵌入”描述不足，缺少关键公式和消融实验；R2 提到代码与数据没有完全公开，只靠文字描述难以复现；R4 认为第二阶段的 LLM prompt 设计过于简单，担心对指令措辞敏感，建议给出 prompt 模板与消融。
        - 实验充分性（褒贬不一）：R2 & R4 质疑数据规模偏小（最大仅 50 k 点），且全部来自金融/能源领域，担心跨领域泛化能力；R3 建议增加 ablation：去掉文本、只用 LLM 等设置，以明确增益来源。
        - 可解释性与用户体验（亮点）：多位 reviewer 喜欢论文提供的“自然语言解释”示例，认为对业务人员友好；R1 建议把更多案例放进正文而不是附录。
    ![TITSP0](./img/TITSP0.png "TITSP0")
    ![TITSP1](./img/TITSP1.png "TITSP1")
    ![TITSP2](./img/TITSP2.png "TITSP2")

    - **数据集生成**
    ### 多模态时序预测数据生成过程

    #### 概述
    在论文 *Instruction-Following LLMs for Time Series Prediction: A Two-Stage Approach (TITSP)* 中，针对真实数据集（如 ETTh1）提出了一种**自动化的多模态数据合成方法**。  
    该方法以真实时序数据为基础，自动生成文本指令，并根据指令数学地修改未来预测值，形成多模态数据对 `(时序数据 + 文本 + 修改后标签)`，以模拟结合上下文指令的预测场景，无需人工标注。

    此过程是**数据预处理步骤**，用于构建训练/评估数据集，并不直接参与模型训练。

    ---

    #### 数据生成流程

    1. **基础时序数据获取**  
       - 从真实数据集加载多变量时序序列  
         $$
         X = \{x_1, x_2, \ldots, x_T\}
         $$
         其中 $x_t$ 为时间步 $t$ 的特征向量。  
       - 以 ETTh1 为例：包含 2 年、1 小时间隔的变压器温度与 6 个电力负载特征，数据集按 12/4/4 个月划分为训练/验证/测试集。  
       - 完全自动化加载，无需人工干预。

    2. **文本指令自动生成**  
       - 为每段时序自动分配文本指令 $S$，如 `"increase"`, `"decrease"`, `"stabilize"`, `"increase amplitude"` 等（详见论文 Table 6）。  
       - 指令从预定义列表中随机采样，确保覆盖多种模式（线性、指数、对数等）。  
       - 在 ETTh1 中，每个指令可生成约 14,307 条训练样本。

    3. **基于指令的预测值修改**  
       - 按指令 $S$ 自动修改未来预测值，得到标签序列：  
         $$
         \hat{X}(S) = \{\hat{x}(S)_{T+1}, \ldots, \hat{x}(S)_{T+H}\}
         $$
         其中 $H$ 为预测步长（horizon）。  
       - 修改规则基于最后观测值 $x_T$，并引入随机参数（如 $A \sim U(\cdot)$）模拟真实动态。  
       - 修改公式程序化实现（详见论文 Appendix C, Table 6）。

    4. **形成多模态数据对**  
       - 每条数据对为：  
         $$
         (X, S, \hat{X}(S))
         $$
         其中：
         - **$X$**：原始时序输入  
         - **$S$**：文本上下文  
         - **$\hat{X}(S)$**：指令修改后的标签序列  
       - 保证语义一致性，例如 `"trend up"` 对应上升趋势标签。

    ---

    #### 示例

    #### 示例 1
    **已知**：  
    - 原序列 $X = [45, 46, 47, 48, 49, 50]$  
    - 指令 $S = \text{"trend up"}$  
    - $A = 1$，预测步长 $H = 3$

    **计算**：  
    $$
    \hat{x}_{7} = 50 + 1 \times 1 = 51
    $$
    $$
    \hat{x}_{8} = 50 + 1 \times 2 = 52
    $$
    $$
    \hat{x}_{9} = 50 + 1 \times 3 = 53
    $$

    **标签**：$\hat{X}(S) = [51, 52, 53]$  
    **最终数据对**：([45,46,47,48,49,50], "trend up", [51,52,53])

    ---

| Action                          | Description                              | Mathematical Function                                     | Generated Dataset                              |
|---------------------------------|------------------------------------------|------------------------------------------------------------|-----------------------------------------------|
| Linear Trend Up                 | Linear increase over time                | see Equation 4                                             | weather, exchange rate                        |
| Linear Trend Down               | Linear decrease over time                | see Equation 5                                             | weather, exchange rate                        |
| Exponential Growth              | Exponential increase over time           | prediction * exp(B * np.arange(x))                         | weather, exchange rate, electricity           |
| Exponential Decay               | Exponential decrease over time           | prediction * exp(-B * np.arange(x))                        | weather, exchange rate, electricity           |
| Logarithmic Growth              | Logarithmic growth over time             | prediction + C * log(1 + np.arange(x))                     | weather, exchange rate, electricity           |
| Logarithmic Decay               | Logarithmic decay over time              | prediction - C * log(1 + np.arange(x))                     | weather, exchange rate, electricity           |
| Keep Stable                     | Constant value of last input point       | All                                                        |                                               |
| Linear Growth and Linear Decay  | Linear increase followed by decrease     | see Equation 6                                             | weather, exchange rate                        |
| Linear Decay and Linear Growth  | Linear decrease followed by increase     | see Equation 7                                             | weather, exchange rate                        |
| Increase Amplitude              | Scale up predictions by a factor         | prediction * (1 + A)                                       | ETTh1, ETTh2, ETTm1, ETTm2, traffic            |
| Decrease Amplitude              | Scale down predictions by a factor       | prediction * (1 - A)                                       | ETTh1, ETTh2, ETTm1, ETTm2, traffic            |

2. **VerbalTS: Generating Time Series from Texts**
   - **作者**: Shuqi Gu, Chuyue Li, Baoyu Jing, Kan Ren. 上海工业大学信息科学与技术学院;伊利诺伊大学厄巴纳-香槟分校
   - **关键词**: Time series generation, Text-to-time-series, Diffusion models, Multi-focal alignment, Multi-view noise estimator, Semantic reprogramming, VERBALTS
   - **Abstract**: Time series synthesis has become a foundational task in modern society, underpinning decisionmaking across various scenes. Recent approaches primarily generate time series from structured conditions, such as attribute-based metadata. However, these methods struggle to capture the full complexity of time series, as the predefined structures often fail to reflect intricate temporal dynamics or other nuanced characteristics. Moreover, constructing structured metadata requires expert knowledge, making large-scale data labeling costly and impractical. In this paper, we introduce VERBALTS, a novel framework for generating time series from unstructured textual descriptions, offering a more expressive and flexible solution to time series synthesis. To bridge the gap between unstructured text and time series data, VERBALTS employs a multi-focal alignment and generation framework, effectively modeling their complex relationships. Experiments on two synthetic and four real-world datasets demonstrate that VERBALTS outperforms existing methods in both generation quality and semantic alignment with textual conditions.
   - **相关工作**：
    - **无条件生成（unconditional generation）**：无条件生成就像“随机抽样”：模型先从真实数据中学到时间序列的整体分布（例如，通过训练捕捉数据的统计模式），然后随机生成新序列。但缺点是用户几乎无法控制生成的序列具体是什么样子（如趋势、周期或特定特征），生成的样本可能多样但不可预测或不精确。
    - **条件生成（conditional generation）**：条件生成允许用户通过“条件”来指导生成过程。这些条件是结构化的（structured），意思是固定格式、易于处理的（如键-值对或标签）。例如：元数据（metadata）：外部附加信息，如数据集标签或上下文（引用Narasimhan et al., 2024的TimeWeaver方法）；时间序列属性（attributes）：序列的内在特征，如趋势类型、季节周期（引用Jing et al., 2024a的TEdit方法）；类标签（class labels）：离散类别，用于分类指导生成（引用Li et al., 2022和Wang et al., 2023）。
    - **顺序条件**：结构化表示（如属性标签或元数据）虽然简单易用，但无法很好地处理时间序列的“顺序动态”（e.g., 事件的前后关系或时序演化）。因为结构化形式通常是静态的列表或类别，无法自然表达“顺序”（sequence），导致信息丢失。例如，一个天气序列的“先小雨后中雨”在结构化中可能只变成“Light rain, Moderate rain”，丢失了时间顺序，导致生成的序列无法准确反映真实动态。e.g.结构化： "Weather: Light rain ? Cloud ? Moderate rain ?"（无法捕捉事件顺序）；非结构化文本： "It‘s the morning of a day in June. The current weather is showing light rain. The weather overall is expected to be partly broken clouds with moderate rain later on."（明确表达顺序：当前小雨，后转为中雨）。
   - **动机**：现有时间序列生成方法主要依赖结构化条件（如元数据或属性），但这些条件难以捕捉时间序列的复杂动态（如局部形状或事件序列），且构建结构化元数据需要专家知识，导致大规模标注成本高昂且不切实际。因此，提出从非结构化文本描述生成时间序列的框架VERBALTS，提供更具表现力和灵活性的解决方案，能够更好地传达细粒度语义信息并桥接文本与时间序列间的模态差距。
        - 论文指出，结构化条件虽然易于学习和处理，但存在显著局限性，导致难以捕捉时间序列的复杂动态（如局部形状或事件序列）。主要原因如下：
            - **信息丢失（Information Loss）**：时间序列往往包含样本特定的独特信息（如特殊局部形状或不规则峰值），这些信息无法用统一的结构化框架封装。例如，论文图1中展示了“beginning part has double peaks”（开头部分有双峰），这是一种样本特定的局部形状（local shapelet），但结构化条件（如“local shapelet: ???”）难以精确描述或标准化，导致生成过程中忽略这些细微特征； 事件序列（sequence of events）是时间序列的常见特性，如天气变化的顺序（“light rain”后转为“moderate rain”），结构化条件（如简单列出“light rain, cloud, moderate rain”）无法有效捕捉时序依赖和顺序关系，容易丢失动态演化信息。
            - **预定义结构的局限性（Limitations of Predefined Structures）**：结构化条件依赖固定格式（如离散类别或数值属性），无法反映时间序列的复杂时序动态（intricate temporal dynamics）或其他细微特性（nuanced characteristics）。例如，真实世界时间序列可能涉及多变量交互、多尺度趋势或非线性变化，但结构化条件难以涵盖所有可能变异，导致生成的样本多样性不足；这些条件限制了生成模型的泛化能力。例如，论文提到DiffTime方法使用约束作为条件，但对于无法轻易表述为约束的条件（如复杂事件序列），其适用性受限。
            - **构建成本高昂（Costly Construction）**：从大规模无组织数据中提取结构化特征需要专家知识（expert knowledge），导致标注过程耗时且劳动力密集（time-consuming and labor-intensive）。这使得大规模数据集构建不切实际，进一步加剧了捕捉复杂动态的难度。
            - **举例**：
                - 结构化条件： "Trend: Up, Volatility: High"（静态属性，无法指定“开头上涨，中期波动大，后期稳定”）。
                - 非结构化文本（VERBALTS方式）： "The stock starts with a sharp rise in the morning, experiences high volatility around noon due to news, and stabilizes in the afternoon."（捕捉顺序事件序列，确保生成符合时序动态）。
    ![VerbalTS0](./img/VerbalTS0.png "VerbalTS0")

   - **方法简述（Proposed Method）**：
    - VERBALTS是一个基于扩散模型的多焦点对齐和生成框架，包括多视图噪声估计器和多焦点文本处理器。多视图噪声估计器从时间、空间和扩散三个视图处理噪声时间序列：使用多分辨率补丁编码器捕捉多尺度时序动态，通过自注意力机制建模时空交互，并将扩散过程分为多个阶段以渐进精炼生成。多焦点文本处理器通过语义重编程（基于可学习锚向量）将文本转换为多语义表示（对应时间、空间和扩散视图）。最终，通过适配器（使用门控、缩放和偏移参数）实现文本与时间序列的多模态语义对齐，确保生成过程受文本细粒度控制。

---
| 问题                     | VERBALTS 解决方案                                   | 技术机制与优势                                                                 |
|--------------------------|----------------------------------------------------|-------------------------------------------------------------------------------|
| 信息丢失                 | 利用非结构化文本提供更细粒度控制，避免忽略局部特征 | 利用文本丰富语义捕捉全局趋势与局部 shapelet，支持多变量时间序列生成           |
| 预定义结构局限性         | 提供更具表现力和灵活性的生成方式，动态应用条件信息 | 基于扩散模型的多焦点对齐框架，将文本转化为多语义组件（时间、空间、扩散过程） |
| 构建成本高昂             | 消除手动提取结构化信息需求，直接用文本控制生成     | 引入可学习锚向量直接桥接文本与时间序列模态，降低标注成本                     |
| 捕捉复杂动态（总体）     | 同时建模全局与局部动态，保证语义与时间特征对齐     | 多视图噪声估计器结合多焦点文本处理器，分阶段动态施加条件                     |
| 捕捉局部形状             | 聚焦局部时序特征与形状变化                        | 高分辨率阶段建模局部 shapelet，结合时间短语识别位置特征                     |
| 捕捉事件序列             | 精确表达事件顺序与状态变化                          | 不同扩散阶段处理不同粒度信息：早期聚焦全局，后期突出细节，可区分语义差异     |

   - **实验设置与数据集**
    - 实验分为单变量和多变量设置，使用合成数据集（Synth-U单变量、Synth-M多变量，手动构建文本-时间序列对）和真实世界数据集（Weather气候指标、BlindWays盲人轨迹，基于真实文本标注；增强数据集ETTm1电力和Traffic交通，使用外部工具标注文本）。评估指标包括保真度（FID和J-FTSD，测量生成分布与真实分布差异）和语义对齐（CTTP分数，通过对比学习计算时间序列与文本的相似度）。基线方法包括TimeVQVAE（类条件）、DiffTime（约束条件）、TimeWeaver和TEdit（属性条件），所有基线条件从文本转换而来。实验在三个随机运行中报告平均性能和标准差，验证生成质量、语义对齐及编辑能力。
   - **结构化条件(如元数据或属性)**

---
 | 类型                  | 描述                                                                 | 例子                                                                 |
|-----------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 元数据 (Metadata)     | 与时间序列相关的附加信息，如数据集标签或外部上下文，通常以键-值对形式呈现 | Weather: Light rain, Cloud, Moderate rain                           |
| 时间序列属性 (Attributes) | 时间序列的内在特征，如趋势、季节性或局部形状，通常预定义为固定属性   | Trend type: linear<br>Trend direction: up<br>Season cycle: 4<br>Local shapelet: ??? |
| 类标签 (Class Labels) | 离散类别，用于分类条件生成                                           | Discrete categories (e.g., class 1, class 2)                        |
| 约束 (Constraints)    | 预设约束条件，用于指导生成过程                                       | Preset constraints (e.g., min/max values, specific patterns)


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

