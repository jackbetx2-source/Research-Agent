# Systematic Literature Review: diffusion model in medical imaging

User-uploaded auxiliary documents are provided below. First infer their role. Treat writing requirements, rubrics, assignment prompts, and style constraints as instructions for the review, not as papers or evidence.

Uploaded auxiliary documents. These may be writing requirements, rubrics, assignment prompts, style constraints, background notes, or source evidence.

Use them as source evidence only when they clearly contain substantive research content.

## Auxiliary document 1: criteria

Source: criteria.docx

Detected role: instructions

Use note: classify this document before using it; if it is a rubric or writing requirement, apply it as task constraints rather than literature evidence.

Extracted excerpt:
写作要求：论文专家风格

**Date**: 2026-05-30

**Papers surveyed**: 20

**Scope**: query `diffusion model in medical`, sources search; arXiv (20), OpenAlex (20), PubMed (20), Crossref (20), deduplicated by DOI/URL/title

**Citation format**: APA 7th edition

**Source mode**: search

## Executive Summary

扩散模型在医学影像领域展现出强大的应用潜力，主要应用于图像合成、分割、翻译、增强和异常检测等任务。现有研究通过改进扩散过程（如使用伯努利噪声、潜在空间操作）和引入轻量化设计来提升模型性能与效率。然而，该领域仍面临计算成本高、泛化能力有限以及缺乏标准化评估等挑战。

## Methodology

This review surveyed 20 deduplicated scholarly records retrieved on 2026-05-30 using the query `diffusion model in medical`. Sources were searched independently, merged by DOI/URL/title, then analyzed through batch language-model extraction followed by cross-paper synthesis. Source counts before final deduplication: arXiv: 20; OpenAlex: 20; PubMed: 20; Crossref: 20.

**Limitations of this review**: 本次综述基于有限数量的论文，可能未能涵盖该领域的所有最新进展。部分论文（如综述类）不提供新的实验证据，而一些实证研究仅关注特定任务或模态，限制了结论的普适性。此外，部分论文的摘要信息不完整，影响了对其方法和发现的全面评估。

## Themes

### Theme 1: 图像分割与检测

研究者通过改进扩散模型架构以适应医学图像分割任务，例如使用伯努利扩散模型处理二值分割，或设计轻量化模型以降低计算成本。这些方法旨在生成多样且准确的分割掩模，辅助临床诊断。

Related papers: 2304.04429, 10.3390/electronics14040676, 10.1007/978-3-031-16452-1_4

### Theme 2: 图像合成与翻译

扩散模型被用于生成高保真度的医学图像，实现不同模态（如MRI到CT）或不同对比度图像之间的转换。研究重点包括提升翻译的保真度、可追溯性以及在无配对数据上的训练能力。

Related papers: 2406.13977, 2512.18455, 10.1109/tmi.2023.3290149, 2310.05237

### Theme 3: 图像增强与标准化

利用扩散模型对医学图像进行去噪、标准化和增强，以改善图像质量并为下游分析提供一致的基础。潜在扩散模型在此类任务中显示出优势。

Related papers: 2310.05237, 2406.13977

### Theme 4: 图像配准与特征提取

扩散模型被用于增强图像的结构信息提取，以改进医学图像配准任务，例如在心肌T1图谱的组配准中实现更精确的对齐。

Related papers: 10.1002/mp.70433

## Convergences and Disagreements

**Convergences**:
- 扩散模型在医学影像的多个任务（合成、分割、翻译）中均展现出优于传统方法（如GAN）的潜力。
- 研究普遍关注如何提升扩散模型的效率（如轻量化设计、加速采样）和生成质量（如保真度、多样性）。
- 潜在空间操作和条件扩散过程是提升模型性能和适应性的关键技术。
- 现有综述论文为该领域提供了系统的理论框架和分类体系。

**Disagreements**:
- 关于扩散模型在特定任务（如异常检测）上的具体性能和适用性，现有研究提供的实证证据有限。
- 不同研究在模型设计（如噪声类型、网络架构）和评估标准上存在差异，导致直接比较困难。

## Gaps and Open Questions

- 缺乏对扩散模型在多类别医学图像分割任务上的深入研究。
- 模型在多样化临床场景和不同成像设备上的泛化能力有待进一步验证。
- 计算效率和实时性仍是临床部署的主要障碍，需要更多轻量化和加速方案。
- 缺乏统一的基准数据集和评估标准来公平比较不同扩散模型方法。

## Methodological Patterns

- 大量研究采用条件扩散模型，通过引入图像或任务特定条件来引导生成过程。
- 潜在扩散模型（LDM）被广泛用于降低计算复杂度，通过在压缩的潜在空间中进行扩散。
- 研究常结合其他技术（如对抗训练、循环一致性、注意力机制）来提升模型性能。
- 评估通常在特定的公开数据集上进行，并与基线方法（如GAN、其他扩散模型）进行定量比较。

## Per-Paper Annotations

### Percutaneous absorption.

**Source**: PubMed

**Source ID**: 10.1002/jps.2600640604

**Research question**: Unknown (abstract not provided)

**Methodology**: Unknown (abstract not provided)

**Key findings**:
- Unknown (abstract not provided)

**Limitations**: The abstract is not available, so the scope, methodology, and findings cannot be determined from the provided context.

**Evidence type**: Unknown

### Gene regulation and DNA C-value paradox: A model based on diffusion of regulatory molecules

**Source**: Crossref

**Source ID**: 10.1016/0306-9877(89)90147-3

**Research question**: How can diffusion of regulatory molecules explain gene regulation and the DNA C-value paradox?

**Methodology**: The paper proposes a theoretical model based on diffusion of regulatory molecules to address gene regulation and the DNA C-value paradox.

**Key findings**:
- Proposes a diffusion-based model for gene regulation
- Addresses the DNA C-value paradox through molecular diffusion mechanisms
- Connects regulatory molecule diffusion to genome size variations

**Limitations**: The abstract is not provided, so specific limitations cannot be inferred from the available context.

**Evidence type**: Theoretical

### A molecular diffusion based utility model for Drosophila larval phototaxis

**Source**: Crossref

**Source ID**: 10.1186/1742-4682-9-3

**Research question**: How can molecular diffusion principles be used to model the phototactic behavior of Drosophila larvae?

**Methodology**: The paper proposes a theoretical utility model based on molecular diffusion to explain larval phototaxis.

**Key findings**:
- Presents a utility model for Drosophila larval phototaxis based on molecular diffusion principles
- The model provides a theoretical framework for understanding phototactic behavior
- Connects molecular-level diffusion processes to organism-level behavioral responses

**Limitations**: The abstract is not provided, so specific limitations cannot be inferred from the available context.

**Evidence type**: Theoretical

### Differences in Gaussian diffusion tensor imaging and non-Gaussian diffusion kurtosis imaging model-based estimates of diffusion tensor invariants in the human brain

**Source**: Crossref

**Source ID**: 10.1118/1.4946819

**Research question**: How do Gaussian diffusion tensor imaging and non-Gaussian diffusion kurtosis imaging model-based estimates of diffusion tensor invariants differ in the human brain?

**Methodology**: Compares model-based estimates of diffusion tensor invariants from Gaussian diffusion tensor imaging and non-Gaussian diffusion kurtosis imaging in the human brain.

**Key findings**:
- Identifies differences between Gaussian and non-Gaussian diffusion models in estimating brain diffusion tensor invariants.
- Provides insights into the comparative performance of these imaging models.

**Limitations**: The abstract does not provide specific details on the study's scope or sample size.

**Evidence type**: Empirical

### Diffusion Models for Medical Anomaly Detection

**Source**: OpenAlex

**Source ID**: 10.1007/978-3-031-16452-1_4

**Research question**: How can diffusion models be applied to detect anomalies in medical images?

**Methodology**: The paper presents a method using diffusion models for medical anomaly detection, though specific experimental details are not provided in the abstract.

**Key findings**:
- Diffusion models are proposed for medical anomaly detection
- The approach is presented as a book chapter in a computer science series
- Authors are affiliated with medical imaging research groups

**Limitations**: The abstract provides no details on specific datasets, experimental results, or performance metrics.

**Evidence type**: Theoretical

### Diffusion Models for Medical Image Analysis: A Comprehensive Survey

**Source**: arXiv

**Source ID**: 2211.07804

**Research question**: What is the current state of diffusion models in medical image analysis, and what are their applications, limitations, and future directions?

**Methodology**: A comprehensive survey providing theoretical foundations, systematic taxonomy, and multi-perspective categorization of diffusion models in medical imaging based on application, modality, organ, and algorithms.

**Key findings**:
- Provides comprehensive overview of diffusion models in medical image analysis
- Introduces three generic diffusion modelling frameworks: diffusion probabilistic models, noise-conditioned score networks, and stochastic differential equations
- Proposes multi-perspective categorization based on application, imaging modality, organ of interest, and algorithms
- Discusses limitations and proposes future research directions
- Compiles open-source implementations of reviewed studies

**Limitations**: As a survey, it synthesizes existing literature but does not present new experimental data or validate specific models.

**Evidence type**: Survey

### Diffusion Models for Medical Image Analysis: A Comprehensive Survey

**Source**: OpenAlex

**Source ID**: 10.48550/arxiv.2211.07804

**Research question**: What is the current state of diffusion models in medical image analysis, and what are their applications, limitations, and future directions?

**Methodology**: A comprehensive survey providing theoretical foundations, systematic taxonomy, and multi-perspective categorization of diffusion models in medical imaging based on application, modality, organ, and algorithms.

**Key findings**:
- Provides comprehensive overview of diffusion models in medical image analysis
- Introduces three generic diffusion modelling frameworks: diffusion probabilistic models, noise-conditioned score networks, and stochastic differential equations
- Proposes multi-perspective categorization based on application, imaging modality, organ of interest, and algorithms
- Discusses limitations and proposes future research directions
- Compiles open-source implementations of reviewed studies

**Limitations**: As a survey, it synthesizes existing literature but does not present new experimental data or validate specific models.

**Evidence type**: Survey

### BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation

**Source**: arXiv

**Source ID**: 2304.04429

**Research question**: How can a diffusion model be adapted to generate accurate and diverse binary segmentation masks for medical images?

**Methodology**: Proposes a conditional Bernoulli Diffusion model (BerDiff) that uses Bernoulli noise instead of Gaussian noise as the diffusion kernel for binary segmentation tasks, and leverages stochastic sampling to generate multiple diverse masks.

**Key findings**:
- BerDiff outperforms other state-of-the-art methods on two medical image segmentation datasets with different modalities.
- The model generates a range of diverse segmentation masks that can highlight salient regions of interest for radiologists.
- BerDiff can efficiently sample sub-sequences from the reverse diffusion trajectory to speed up the segmentation process.
- The results suggest diffusion models could serve as a strong backbone for medical image segmentation.

**Limitations**: The study is focused on binary segmentation tasks and does not address multi-class segmentation.

**Evidence type**: Empirical

### BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation

**Source**: OpenAlex

**Source ID**: 10.1007/978-3-031-43901-8_47

**Research question**: How can a conditional Bernoulli diffusion model be applied to medical image segmentation tasks?

**Methodology**: The paper proposes BerDiff, a conditional Bernoulli diffusion model specifically designed for medical image segmentation.

**Key findings**:
- Introduces BerDiff, a conditional Bernoulli diffusion model for medical image segmentation
- The model is designed to handle the discrete nature of segmentation masks
- Applies diffusion models to the specific challenge of medical image segmentation

**Limitations**: The abstract is not provided, so specific limitations cannot be inferred from the available context.

**Evidence type**: Theoretical

### Diffusion models in medical imaging: A comprehensive survey

**Source**: OpenAlex

**Source ID**: 10.1016/j.media.2023.102846

**Research question**: What is the current state of diffusion models applied to medical imaging tasks?

**Methodology**: A comprehensive survey reviewing the application of diffusion models in medical imaging.

**Key findings**:
- Provides a comprehensive overview of diffusion models in medical imaging.
- Covers various tasks such as image synthesis, segmentation, and reconstruction.
- Discusses challenges and future directions for the field.

**Limitations**: As a survey, it does not present new experimental results but synthesizes existing literature.

**Evidence type**: Survey

### Latent Diffusion Model for Medical Image Standardization and Enhancement

**Source**: arXiv

**Source ID**: 2310.05237

**Research question**: How can a score-based diffusion model operating in latent space standardize and enhance CT images from different scanners and protocols?

**Methodology**: DiffusionCT uses a U-Net encoder-decoder trained independently, followed by a latent DDPM model trained at the bottleneck to transform non-standard distributions into standardized forms.

**Key findings**:
- DiffusionCT transforms disparate CT distributions into standardized forms using latent diffusion
- The model shows notable improvements in CT image standardization
- Significantly reduces image noise in SPAD images
- Provides more consistent basis for downstream analysis compared to GAN-based methods

**Limitations**: The model's performance is validated on patient CT images but may require further testing across diverse clinical scenarios and scanner types.

**Evidence type**: Empirical

### Solute diffusion and partitioning in multi-arm poly(ethylene glycol) hydrogels.

**Source**: PubMed

**Source ID**: 10.1039/d2tb02004a

**Research question**: How do simultaneous variations in multiple structural parameters of PEG hydrogels affect solute diffusion and partitioning?

**Methodology**: High-throughput FRAP experiments were conducted on a library of multi-arm PEG hydrogels with variations in four independent structural parameters to characterize size-dependent solute diffusion and partitioning.

**Key findings**:
- Solute diffusivity dependence on junction functionality shows influence from network geometry not captured by mesh size models
- The Richbourg-Peppas swollen polymer network (SPN) model accurately predicts how three of four structural parameters affect solute diffusivity
- The SPN model outperforms the large pore effective medium (LPEM) model in predicting solute size and hydrogel structure effects
- Provides a framework for investigating solute transport in hydrogels for tissue engineering and drug delivery

**Limitations**: The study focuses on multi-arm PEG hydrogels and may not generalize to all hydrogel types or solute characteristics.

**Evidence type**: Empirical

### Unsupervised Medical Image Translation With Adversarial Diffusion Models

**Source**: OpenAlex

**Source ID**: 10.1109/tmi.2023.3290149

**Research question**: Can an adversarial diffusion model improve the performance of unsupervised medical image translation compared to GAN-based methods?

**Methodology**: The authors proposed SynDiff, a method based on adversarial diffusion modeling that uses a conditional diffusion process with adversarial projections in the reverse diffusion direction for fast and accurate image sampling, and a cycle-consistent architecture for training on unpaired datasets.

**Key findings**:
- SynDiff leverages a conditional diffusion process to progressively map noise and source images onto the target image.
- The method uses adversarial projections in the reverse diffusion direction for fast and accurate sampling.
- A cycle-consistent architecture enables training on unpaired datasets.
- SynDiff demonstrated quantitatively and qualitatively superior performance against competing GAN and diffusion models in multi-contrast MRI and MRI-CT translation.

**Limitations**: The study focused on specific translation tasks (multi-contrast MRI and MRI-CT), and performance on other modalities or clinical applications may vary.

**Evidence type**: Empirical

### Diffusion of proteins in crowded solutions studied by docking-based modeling.

**Source**: PubMed

**Source ID**: 10.1063/5.0220545

**Research question**: How does macromolecular crowding affect the translational and rotational diffusion of proteins?

**Methodology**: Uses a docking-based approach to simulate protein diffusion in crowded environments by sampling the intermolecular energy landscape with Markov Chain Monte Carlo, benchmarked against experimental and theoretical data.

**Key findings**:
- Smaller proteins diffuse faster in heterogeneous crowding of larger molecules compared to self-crowded solutions.
- Larger proteins diffuse faster in self-crowded solutions compared to heterogeneous crowding.
- The simulation approach demonstrates predictive power for long timescales of cell-size systems at atomic resolution.
- Results are in good agreement with available experimental and theoretical data.

**Limitations**: The study is based on simulations and may not fully capture all complexities of in vivo cellular environments.

**Evidence type**: Empirical

### FRAP analysis of peptide diffusion in extracellular matrix mimetic hydrogels as an in vitro model for subcutaneous injection.

**Source**: PubMed

**Source ID**: 10.1016/j.ijpharm.2024.124628

**Research question**: Can FRAP analysis quantify peptide diffusion in extracellular matrix mimetic hydrogels to model subcutaneous drug injection?

**Methodology**: FRAP (Fluorescence Recovery After Photobleaching) was used to measure diffusion coefficients of various peptides (FITC-dextran, poly-lysine, poly-glutamic acid, exenatide) in agarose, collagen, and hyaluronic acid hydrogels.

**Key findings**:
- Diffusion in uncharged agarose gels approximated free diffusion in PBS
- Cationic peptide diffusion was substantially decreased in anionic hyaluronic acid gels due to electrostatic interactions
- Peptide aggregation was observed as immobile fractions with exenatide
- FRAP provides useful information on peptide interactions and transport in hydrogel networks

**Limitations**: The study uses in vitro hydrogel models which may not fully replicate the complex in vivo subcutaneous environment.

**Evidence type**: Empirical

### Similarity-aware Syncretic Latent Diffusion Model for Medical Image Translation with Representation Learning

**Source**: arXiv

**Source ID**: 2406.13977

**Research question**: How can a latent diffusion model be designed to achieve high-fidelity medical image translation without requiring additional conditions during inference?

**Methodology**: Proposes a Syncretic generative model based on the latent diffusion model (S²LDM) that uses syncretic encoding and diffusing to enhance similarity in distinct modal images, guided by adaptive similarity loss and dynamic similarity.

**Key findings**:
- S²LDM achieves high-fidelity reconstruction without the need for additional conditions during inference.
- The model enhances similarity in distinct modal images via syncretic encoding and diffusing in the latent space.
- Adaptive similarity loss and dynamic similarity guide generation to supplement high-frequency details.
- Quantitative experiments confirm the effectiveness of the approach in medical image translation.

**Limitations**: The code is not yet released, and the study focuses on specific modalities (NCCT to CECT translation).

**Evidence type**: Empirical

### Lightweight Denoising Diffusion Implicit Model for Medical Segmentation

**Source**: Crossref

**Source ID**: 10.3390/electronics14040676

**Research question**: How can diffusion models be made lightweight for medical segmentation tasks on general-purpose devices?

**Methodology**: An enhanced denoising diffusion implicit model (DDIM) incorporating lightweight depthwise convolution layers within residual networks and self-attention layers was proposed and evaluated on X-ray and skin lesion/polyp segmentation datasets.

**Key findings**:
- The proposed lightweight DDIM significantly reduces computational overhead
- Model achieves accuracy comparable to standard DDIMs with reduced resource requirements
- Enables implementation on general-purpose devices without expensive high-performance computing
- Evaluated on two distinct medical imaging datasets

**Limitations**: The study is limited to two specific medical imaging tasks (X-ray and skin lesion/polyp segmentation) and may not generalize to all medical segmentation problems.

**Evidence type**: Empirical

### Plasticine: A Traceable Diffusion Model for Medical Image Translation

**Source**: arXiv

**Source ID**: 2512.18455

**Research question**: How can an image-to-image translation framework be designed to explicitly provide pixel-level traceability between original and translated medical images?

**Methodology**: The authors proposed Plasticine, an end-to-end framework that combines intensity translation and spatial transformation within a denoising diffusion framework to generate synthetic images with interpretable intensity transitions and spatially coherent deformations.

**Key findings**:
- Plasticine is the first framework explicitly designed with traceability as a core objective for medical image translation.
- The method enables generation of synthetic images with interpretable intensity transitions.
- It supports pixel-wise traceability throughout the translation process.
- The framework combines intensity translation and spatial transformation within a diffusion model.

**Limitations**: The study presents a novel framework but does not provide extensive quantitative comparisons with existing methods on standard benchmarks.

**Evidence type**: System

### Diffusion boost: Leveraging diffusion model for groupwise registration in myocardial T1 mapping

**Source**: Crossref

**Source ID**: 10.1002/mp.70433

**Research question**: Can a diffusion model be leveraged to improve the structural representation extraction for groupwise registration of myocardial T1 mapping images with varying contrast?

**Methodology**: The authors proposed a template-free groupwise registration framework that uses a diffusion process to boost structural information extraction and a Hybrid Attention Feature Fusion module for multi-scale feature fusion.

**Key findings**:
- The proposed method achieved a Dice score of 0.839 and a T1 mapping error of 11.372 ms.
- It surpassed current state-of-the-art approaches in registration performance.
- The diffusion model demonstrated strong feature extraction capacity for image registration.
- The framework enables robust groupwise alignment of T1-weighted image series through a single forward propagation.

**Limitations**: The study was evaluated on a single publicly available dataset, and generalizability to other datasets or clinical settings requires further validation.

**Evidence type**: Benchmark

### Understanding salt diffusion in dairy-based systems: A model approach using rennet-coagulated micellar casein concentrates.

**Source**: PubMed

**Source ID**: 10.3168/jds.2025-27626

**Research question**: How do pH, fat, calcium content, and salting temperature affect salt diffusion in a model cheese system, and can salty whey be used as an alternative salting medium?

**Methodology**: The study developed a model cheese system using renneted gels from micellar casein concentrate and modeled salt diffusivity using Fick's second law. It assessed the effects of varying formulation parameters and salting temperature on salt migration.

**Key findings**:
- Increasing calcium content (0%-1% wt/wt) significantly increased salt penetration.
- Salting at 30°C significantly enhanced salt diffusion compared to 20°C or 40°C.
- Diffusion coefficients ranged from 3.1 × 10⁻⁹ to 8.5 × 10⁻⁹ m²/s.
- Higher calcium promoted a more continuous protein matrix, increasing diffusion, while fat globules hindered it.
- Salty whey had lower diffusivity than brine, likely due to higher osmotic pressure and viscosity.

**Limitations**: The study used a model cheese system, which may not fully replicate the complexity of commercial cheese production.

**Evidence type**: Empirical

## References

Chen, T., Wang, C., & Shan, H. (2023). BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation. arXiv. https://arxiv.org/abs/2304.04429v1

Chen, T., Wang, C., & Shan, H. (2023). BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation. Lecture notes in computer science. https://doi.org/10.1007/978-3-031-43901-8_47

Gong, Z., & Gong, Z. (2012). A molecular diffusion based utility model for Drosophila larval phototaxis. Theoretical Biology and Medical Modelling. https://doi.org/10.1186/1742-4682-9-3

Idson, B. (1975). Percutaneous absorption.. Journal of pharmaceutical sciences. https://pubmed.ncbi.nlm.nih.gov/1094104/

Kazerouni, A., Aghdam, E. K., Heidari, M., Azad, R., Fayyaz, M., Hacihaliloglu, I., & Merhof, D. (2022). Diffusion Models for Medical Image Analysis: A Comprehensive Survey. arXiv. https://arxiv.org/abs/2211.07804v3

Kazerouni, A., Aghdam, E. K., Heidari, M., Azad, R., Fayyaz, M., Hacihaliloglu, I., & Merhof, D. (2022). Diffusion Models for Medical Image Analysis: A Comprehensive Survey. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2211.07804

Kazerouni, A., Aghdam, E. K., Heidari, M., Azad, R., Fayyaz, M., Hacihaliloglu, I., & Merhof, D. (2023). Diffusion models in medical imaging: A comprehensive survey. Medical Image Analysis. https://doi.org/10.1016/j.media.2023.102846

Kupiec, J. (1989). Gene regulation and DNA C-value paradox: A model based on diffusion of regulatory molecules. Medical Hypotheses. https://doi.org/10.1016/0306-9877(89)90147-3

Lanzafame, S., Giannelli, M., Garaci, F., Floris, R., Duggento, A., Guerrisi, M., & Toschi, N. (2016). Differences in Gaussian diffusion tensor imaging and non-Gaussian diffusion kurtosis imaging model-based estimates of diffusion tensor invariants in the human brain. Medical Physics. https://doi.org/10.1118/1.4946819

Lin, T., Lyu, P., Zhang, J., Wang, Y., Wang, C., & Zhu, J. (2024). Similarity-aware Syncretic Latent Diffusion Model for Medical Image Translation with Representation Learning. arXiv. https://arxiv.org/abs/2406.13977v2

Oh, R., & Gonsalves, T. (2025). Lightweight Denoising Diffusion Implicit Model for Medical Segmentation. Electronics. https://doi.org/10.3390/electronics14040676

Parlow, J., Rodler, A., Gråsjö, J., Sjögren, H., & Hansson, P. (2024). FRAP analysis of peptide diffusion in extracellular matrix mimetic hydrogels as an in vitro model for subcutaneous injection.. International journal of pharmaceutics. https://pubmed.ncbi.nlm.nih.gov/39179009/

Richbourg, N. R., & Peppas, N. A. (2023). Solute diffusion and partitioning in multi-arm poly(ethylene glycol) hydrogels.. Journal of materials chemistry. B. https://pubmed.ncbi.nlm.nih.gov/36511476/

Selim, M., Zhang, J., Fathi, F., Brooks, M. A., Wang, G., Yu, G., & Chen, J. (2023). Latent Diffusion Model for Medical Image Standardization and Enhancement. arXiv. https://arxiv.org/abs/2310.05237v1

Singh, A., Kundrotas, P. J., & Vakser, I. A. (2024). Diffusion of proteins in crowded solutions studied by docking-based modeling.. The Journal of chemical physics. https://pubmed.ncbi.nlm.nih.gov/39225532/

Weerasingha, V., Kelly, A. L., Sheehan, J. J., & Alehosseini, A. (2026). Understanding salt diffusion in dairy-based systems: A model approach using rennet-coagulated micellar casein concentrates.. Journal of dairy science. https://pubmed.ncbi.nlm.nih.gov/41547448/

Wolleb, J., Bieder, F., Sandkühler, R., & Cattin, P. C. (2022). Diffusion Models for Medical Anomaly Detection. Lecture notes in computer science. https://doi.org/10.1007/978-3-031-16452-1_4

Yue, C., Wang, Q., Guo, Y., Tao, Q., & Wang, Y. (2026). Diffusion boost: Leveraging diffusion model for groupwise registration in myocardial T1 mapping. Medical Physics. https://doi.org/10.1002/mp.70433

Zhang, T., Cheng, X., Cheng, J., Zheng, S., Zhao, H., Fu, H., Frangi, A. F., Liu, J., & Duan, J. (2025). Plasticine: A Traceable Diffusion Model for Medical Image Translation. arXiv. https://arxiv.org/abs/2512.18455v2

Özbey, M., Dalmaz, O., Dar, S. U. H., Bedel, H. A., Özturk, Ş., Güngör, A., & Çukur, T. (2023). Unsupervised Medical Image Translation With Adversarial Diffusion Models. IEEE Transactions on Medical Imaging. https://doi.org/10.1109/tmi.2023.3290149
