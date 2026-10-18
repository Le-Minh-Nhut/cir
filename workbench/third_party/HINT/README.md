<a id="top"></a>
<div align="center">
  <img src="./assets/logo.png" width="500"> 
  
  <h1>(ICASSP 2026) HINT: Composed Image Retrieval with Dual-Path Compositional Contextualized Network</h1>

  <div>
  <a target="_blank" href="https://zh-mingyu.github.io/">Mingyu&#160;Zhang</a><sup>1</sup>,
  <a target="_blank" href="https://lee-zixu.github.io/">Zixu&#160;Li</a><sup>1</sup>,
  <a target="_blank" href="https://zivchen-ty.github.io/">Zhiwei&#160;Chen</a><sup>1</sup>,
  <a target="_blank" href="https://zhihfu.github.io/">Zhiheng&#160;Fu</a><sup>1</sup>,
  Xiaowei&#160;Zhu</a><sup>1</sup>,
  Jiajia&#160;Nie</a><sup>1</sup>,
  <a target="_blank" href="https://faculty.sdu.edu.cn/weiyinwei1/zh_CN/index.htm">Yinwei&#160;Wei</a><sup>1</sup>
  <a target="_blank" href="https://faculty.sdu.edu.cn/huyupeng1/zh_CN/index.htm">Yupeng&#160;Hu</a><sup>1&#9993</sup>,
  </div>
  <sup>1</sup>School of Software, Shandong University &#160&#160&#160</span>
  <br />
  <sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
  <br/>
  
  <p>
      <a href="https://2026.ieeeicassp.org/"><img src="https://img.shields.io/badge/ICASSP-2026-blue.svg?style=flat-square" alt="ICASSP 2026"></a>
      <a href="https://arxiv.org/abs/2603.26341"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2603.26341-b31b1b.svg"></a>
      <a href="https://arxiv.org/pdf/2603.26341v1"><img alt='Paper' src="https://img.shields.io/badge/Paper-ICASSP-green.svg"></a>
    <a href="https://zh-mingyu.github.id/HINT.github.io"><img alt='page' src="https://img.shields.io/badge/Website-orange"></a>
    <a href="https://zh-mingyu.github.id"><img src="https://img.shields.io/badge/Author/Page-blue.svg" alt="Author Page"></a>
    <a href="https://pytorch.org/get-started/locally"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?&logo=pytorch&logoColor=white"></a>
    <img src="https://img.shields.io/badge/python-≥3.8-blue?style=flat-square" alt="Python">
    <a href="https://github.com/"><img alt='stars' src="https://img.shields.io/github/stars/zh-mingyu/HINT?style=social"></a>
  </p>

  <p>
    <b>Accepted by ICASSP 2026:</b> A novel contextualized network tackling the neglect of contextual information in Composed Image Retrieval (CIR) by amplifying similarity differences between matching and non-matching samples.
  </p>
</div>


## 📌 Introduction

**HINT** (dual-patH composItional coNtextualized neTwork) is our proposed framework for Composed Image Retrieval (CIR), accepted by ICASSP 2026. Although existing methods have made significant progress, they often neglect contextual information in discriminating matching samples. To address the implicit dependencies and the lack of a differential amplification mechanism, HINT systematically models contextual structure to improve the upper performance of CIR models in complex scenarios.

[⬆ Back to top](#top)

## 📢 News
* **[2026-03-26]** 🚀 Initial setup for the HINT repository. Source code is scheduled for release in April 2026.
* **[2026-01-18]** 🔥 Our paper *"HINT: COMPOSED IMAGE RETRIEVAL WITH DUAL-PATH COMPOSITIONAL CONTEXTUALIZED NETWORK"* has been accepted by **ICASSP 2026**!

[⬆ Back to top](#top)

## ✨ Key Features

  - 🧠 **Dual Context Extraction (DCE)**: Extracts both intra-modal context and cross-modal context, enhancing joint semantic representation by integrating multimodal contextual information.
  - 📏 **Quantification of Contextual Relevance (QCR)**: Evaluates the relevance between cross-modal contextual information and the target image semantics, enabling the quantification of implicit dependencies.
  - 🛡️ **Dual-Path Consistency Constraints (DPCC)**: Optimizes the training process by constraining the representation consistency between multimodal fusion features and the target, ensuring the stable enhancement of similarity for matching instances while lowering the similarity for non-matching instances.
  - 🏆 **Outstanding Performance**: Achieves competitive results on major metrics across two CIR benchmark datasets, FashionIQ and CIRR, demonstrating strong cross-domain generalization ability.

[⬆ Back to top](#top)

## 🏗️ Architecture

<p align="center">
  <img src="./assets/HINT.png" alt="HINT architecture" width="1000">
  <figcaption><strong>Figure 1.</strong> HINT framework consists of three modules: (a) Dual Context Extraction, (b) Quantification of Contextual Relevance, (c) Dual-Path Consistency Constraints. </figcaption>
</p>

[⬆ Back to top](#top)

## 🏃‍♂️ Experiment-Results

### CIR Task Performance

#### Experimental Results

<caption><strong>Table 1.</strong> Performance comparison on FashionIQ and CIRR datasets. HINT achieves a notable relative increase of approximately 9.74% in average R@10 on FashionIQ, and a 1.74% improvement in R@1 on the CIRR test set.</caption>

<p align="center">
  <img src="./assets/results.png" alt="Experimental Results on FashionIQ and CIRR">
</p>

[⬆ Back to top](#top)

---

## Table of Contents

- [📌 Introduction](#-introduction)
- [📢 News](#-news)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🏃‍♂️ Experiment-Results](#️-experiment-results)
  - [CIR Task Performance](#cir-task-performance)
    - [Experimental Results](#experimental-results)
- [Table of Contents](#table-of-contents)
- [📦 Install](#-install)
- [📂 Data Preparation](#-data-preparation)
- [🚀 Quick Start](#-quick-start)
  - [1. Training](#1-training)
  - [2. Testing](#2-testing)
- [📁 Project Structure](#-project-structure)
- [🤝 Acknowledgement](#-acknowledgement)
- [✉️ Contact](#️-contact)
- [🔗 Related Projects](#-related-projects)
- [📝⭐️ Citation](#️-citation)

---

## 📦 Install

**1. Clone the repository**

```bash
git clone https://github.com/zh-mingyu.github.io/HINT.git
cd HINT
```

**2. Setup Python Environment**

The code is evaluated on **Python 3.8.10** and **CUDA 12.6**. We recommend using Anaconda:

```bash
conda create -n habit python=3.8
conda activate habit

# Install PyTorch (The evaluated environment uses Torch 2.1.0 with CUDA 12.1 compatibility)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core dependencies
pip install open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16
```

[⬆ Back to top](#top)

-----

## 📂 Data Preparation

We evaluated our framework on two standard datasets: [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq) and [CIRR](https://github.com/Cuberick-Orion/CIRR). Please download the datasets first.

<details>
<summary><b>Click to expand: FashionIQ Dataset Directory Structure</b></summary>

Please follow the official instructions to download the FashionIQ dataset. Once downloaded, ensure the folder structure looks like this:

```text
├── FashionIQ
│   ├── captions
│   │   ├── cap.dress.[train | val | test].json
│   │   ├── cap.toptee.[train | val | test].json
│   │   ├── cap.shirt.[train | val | test].json
│   ├── image_splits
│   │   ├── split.dress.[train | val | test].json
│   │   ├── split.toptee.[train | val | test].json
│   │   ├── split.shirt.[train | val | test].json
│   ├── dress
│   │   ├── [B000ALGQSY.jpg | B000AY2892.jpg | B000AYI3L4.jpg |...]
│   ├── shirt
│   │   ├── [B00006M009.jpg | B00006M00B.jpg | B00006M6IH.jpg | ...]
│   ├── toptee
│   │   ├── [B0000DZQD6.jpg | B000A33FTU.jpg | B000AS2OVA.jpg | ...]
```

</details>

<details>
<summary><b>Click to expand: CIRR Dataset Directory Structure</b></summary>

Please follow the official instructions to download the CIRR dataset. Once downloaded, ensure the folder structure looks like this:

```text
├── CIRR
│   ├── train
│   │   ├── [0 | 1 | 2 | ...]
│   │   │   ├── [train-10108-0-img0.png | train-10108-0-img1.png | ...]
│   ├── dev
│   │   ├── [dev-0-0-img0.png | dev-0-0-img1.png | ...]
│   ├── test1
│   │   ├── [test1-0-0-img0.png | test1-0-0-img1.png | ...]
│   ├── cirr
│   ├── captions
│   │   ├── cap.rc2.[train | val | test1].json
│   ├── image_splits
│   │   ├── split.rc2.[train | val | test1].json
```

</details>

[⬆ Back to top](#top)

-----

## 🚀 Quick Start

### 1. Training

Our model is trained using the AdamW optimizer. The hyper-parameter $\lambda$ for the loss function is set to 0.2.

**Training on FashionIQ:**

```bash
python train.py \
    --dataset fashioniq \
    --fashioniq_path "/path/to/FashionIQ/" \
    --model_dir "./checkpoints/fashioniq_hint" \
    --batch_size 128 \
    --num_epochs 10 \
    --lr 2e-5
```

**Training on CIRR:**

```bash
python train.py \
    --dataset cirr \
    --cirr_path "/path/to/CIRR/" \
    --model_dir "./checkpoints/cirr_hint" \
    --batch_size 128 \
    --num_epochs 10 \
    --lr 2e-5
```

> **💡 Tips:** > - Our model is based on the powerful BLIP-2 architecture. It is highly recommended to run the training on GPUs with sufficient memory (e.g., NVIDIA A40 48G / V100 32G).
>
>   - The best model weights and evaluation metrics generated during training will be automatically saved in the `best_model.pt` and `metrics_best.json` files within your specified `--model_dir`.

### 2. Testing

To generate the prediction files on the CIRR dataset for submission to the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/), run the testing script:

```bash
python src/cirr_test_submission.py checkpoints/cirr_hint/
```
*(The corresponding script will automatically output `.json` based on the generated best checkpoints in the folder for online evaluation.)*

[⬆ Back to top](#top)

-----

## 📁 Project Structure

Our code is deeply customized based on the LAVIS framework. The core implementations are centralized in the following files:

```text
HINT/
├── lavis/
│   ├── models/
│   │   └── blip2_models/
│   │       └── HINT.py      # 🧠 Core model implementation: Includes DCE, QCR and DPCC modules
├── train.py                  # 🚀 Training entry point: Controls noise_ratio injection and training loops
├── datasets.py 
├── test.py 
├── utils.py 
├── data_utils.py 
├── cirr_test_submission.py   # Auxiliary scripts
├── datasets/                 # Dataset loading and processing logic
└── README.md
```

-----


## 🤝 Acknowledgement

The implementation of this project utilizes the pre-trained vision-language features from BLIP-2 and references the [LAVIS](https://github.com/salesforce/LAVIS) framework. We express our sincere gratitude to these open-source contributions!

[⬆ Back to top](#top)

## ✉️ Contact

For any questions, issues, or feedback, please open an [issue](https://github.com/zh-mingyu.github.io/HINT/issues) on GitHub or reach out to us at `mingyuzhang@mail.sdu.edu.cn`.

[⬆ Back to top](#top)

## 🔗 Related Projects

*Ecosystem & Other Works from our Team*
<table style="width:100%; border:none; text-align:center; background-color:transparent;">
<tr style="border:none;">
 <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/consep-logo.png" alt="ConeSep" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ConeSep (CVPR'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://lee-zixu.github.io/ConeSep.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/lee-zixu/ConeSep" target="_blank">Code</a> | 
        <!-- <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608" target="_blank">Paper</a> -->
      </span>
    </td>
     <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/airknow-logo.png" alt="Air-Know" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>Air-Know (CVPR'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zhihfu.github.io/Air-Know.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/zhihfu/Air-Know" target="_blank">Code</a> | 
        <!-- <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608" target="_blank">Paper</a> -->
      </span>
    </td>
     <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/retrack-logo.png" alt="ReTrack" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ReTrack (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://lee-zixu.github.io/ReTrack.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/ReTrack" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39507" target="_blank">Paper</a>
      </span>
    </td>
   </tr>
  <tr style="border:none;">
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/intent-logo.png" alt="INTENT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>INTENT (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/INTENT.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/INTENT" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/39181" target="_blank">Paper</a>
      </span>
    </td>  
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/hud-logo.png" alt="HUD" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>HUD (ACM MM'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/HUD.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/HUD" target="_blank">Code</a> | 
        <a href="https://dl.acm.org/doi/10.1145/3746027.3755445" target="_blank">Paper</a>
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/offset-logo.png" alt="OFFSET" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>OFFSET (ACM MM'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://zivchen-ty.github.io/OFFSET.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/ZivChen-Ty/OFFSET" target="_blank">Code</a> | 
        <a href="https://dl.acm.org/doi/10.1145/3746027.3755366" target="_blank">Paper</a>
      </span>
    </td>
     </tr>
  <tr style="border:none;">
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/encoder-logo.png" alt="ENCODER" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>ENCODER (AAAI'25)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://sdu-l.github.io/ENCODER.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/ENCODER" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32541" target="_blank">Paper</a>
      </span>
    </td>
    <td style="width:30%; border:none; vertical-align:top; padding-top:30px;">
      <img src="/assets/logos/habit-logo.png" alt="HABIT" style="height:65px; width:auto; border-radius:8px; margin-bottom:8px;"><br>
      <b>HABIT (AAAI'26)</b><br>
      <span style="font-size: 0.9em;">
        <a href="https://lee-zixu.github.io/HABIT.github.io/" target="_blank">Web</a> | 
        <a href="https://github.com/Lee-zixu/HABIT" target="_blank">Code</a> | 
        <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37608" target="_blank">Paper</a>
      </span>
    </td>
  </tr>
</table>

## 📝⭐️ Citation

If you find our work or this code useful in your research, please consider leaving a **Star**⭐️ or **Citing**📝 our paper 🥰. Your support is our greatest motivation!

```bibtex
@inproceedings{HINT2026,
  title={HINT: COMPOSED IMAGE RETRIEVAL WITH DUAL-PATH COMPOSITIONAL CONTEXTUALIZED NETWORK},
  author={Zhang, Mingyu and Li, Zixu and Chen, Zhiwei and Fu, Zhiheng and Zhu, Xiaowei and Nie, Jiajia and Wei, Yinwei and Hu, Yupeng},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

[⬆ Back to top](#top)

<div align="center">
  <br><br>

  <a href="https://github.com/zh-mingyu.github.io/HINT">
    <img src="https://img.shields.io/badge/⭐_Star_US-000000?style=for-the-badge&logo=github&logoColor=00D9FF" alt="Star">
  </a>
  <a href="https://github.com/zh-mingyu.github.io/HINT/issues">
    <img src="https://img.shields.io/badge/🐛_Report_Issues-000000?style=for-the-badge&logo=github&logoColor=FF6B6B" alt="Issues">
  </a>
  <a href="https://github.com/zh-mingyu/HINT/pulls">
    <img src="https://img.shields.io/badge/🧐_Pull_Requests-000000?style=for-the-badge&logo=github&logoColor=4ECDC4" alt="Pull Request">
  </a>

  <br><br>
<a href="zh-mingyu.github.io/HINT">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00D9FF&center=true&vCenter=true&width=500&lines=Thank+you+for+visiting+HINT!;Looking+forward+to+your+attention" alt="Typing SVG">
  </a>
</div>
