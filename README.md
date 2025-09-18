# 🤿 **DENSE VIDEO UNDERSTANDING WITH GATED RESIDUAL TOKENIZATION**
### **Dense Information Video Evaluation (DIVE) Benchmark**



<p align="center">
  <a href="https://arxiv.org/pdf/2509.14199">
    <img src="https://img.shields.io/badge/ArXiv-2509.14199-red?style=for-the-badge&logo=arxiv" alt="ArXiv"/>
  </a>
  <a href="https://zhanghaichao.xyz/DenseVideoUnderstand/">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=google-chrome" alt="Website"/>
  </a>
  <a href="https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-ffcc4d?style=for-the-badge&logo=huggingface" alt="HuggingFace Dataset"/>
  </a>
  <a href="https://github.com/hai-chao-zhang/DenseVideoUnderstand/">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>

The **first-ever benchmark** dedicated to **Dense Video Understanding**, focusing on **QA-driven high-frame-rate** comprehension where **answer-relevant information** appears **in nearly every frame**.

---

<p align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66393f5a1231260674ae798e/uOmH6pKW5yqk6PstJ4H8R.jpeg"
     alt="DIVE" width="1080">
</p>

---

## 👥 Authors
<p align="center">
  <a href="https://zhanghaichao.xyz"><b>Haichao Zhang<sup>1</sup></b></a> ·
  <a href="https://wenhaochai.com/"><b>Wenhao Chai<sup>2</sup></b></a> ·
  <a href="https://shwai-he.github.io/"><b>Shwai He<sup>3</sup></b></a> ·
  <a href="https://www.ang-li.com/"><b>Ang Li<sup>3</sup></b></a> ·
  <a href="https://www1.ece.neu.edu/~yunfu/"><b>Yun Fu<sup>1</sup></b></a>
</p>
<p align="center">
  <sub><b>1</b> Northeastern University &nbsp;|&nbsp; <b>2</b> Princeton University &nbsp;|&nbsp; <b>3</b> University of Maryland, College Park</sub>
</p>
<p align="center">
  <img src="https://brand.northeastern.edu/wp-content/uploads/2025/01/seal-yellow.svg" height="60" alt="NEU Seal"/>
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/Northeastern_University_wordmark.svg" height="30" alt="NEU Wordmark"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/Princeton_University_Shield.svg" height="60" alt="Princeton Shield"/>
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/Princeton_text_logo.svg" height="32" alt="Princeton Wordmark"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://prg.cs.umd.edu/img/logo/umd-logo-transparent.png" height="60" alt="UMD Logo"/>
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/University_of_Maryland_wordmark.svg" height="32" alt="UMD Wordmark"/>
</p>



---

## 📅 **Timeline**

| Date | Status | Description |
|------|--------|-------------|
| **2025/09/18** | ✅ | Release the **DIVE benchmark (test split only)** |
| *TBD* | ⭕ | Release **evaluation/test code** on [GitHub](https://github.com/hai-chao-zhang/DenseVideoUnderstand/) |
| *TBD* | ⭕ | Merge DIVE into **LMMS-EVAL** VLM test kit |
| *TBD* | ⭕ | Release **multi-FPS versions** of the dataset |
| *TBD* | ⭕ | Add **diverse dense video task categories** |
| *TBD* | ⭕ | **Release full GRT model and training/inference code** |
| *Future Ideas* | 💡 | Contact us with suggestions for new tasks or collaborations |
---

## 🔍 What is DIVE?
**DIVE (Dense Information Video Evaluation)** is a benchmark designed for scenarios where useful content is densely distributed across frames (e.g., educational/lecture videos, surgical procedures, sign language). Existing VLLM pipelines downsample aggressively to control token cost, which **drops critical temporal details**.


## 🔍 What is GRT?
### GRT in a Nutshell (method overview)
**Gated Residual Tokenization (GRT)** is our token-efficiency framework:
1. **Motion-Gated Tokenization (inter-tokenization):** detect static regions via motion cues and **skip** them during tokenization → **sub-linear token/time growth** w.r.t. FPS.
2. **Semantic Scene Token Merging (intra-tokenization):** **merge redundant tokens** within scenes while preserving dynamic semantics.

For details, see the paper: [arXiv:2509.14199](https://arxiv.org/html/2509.14199).

> ⚠️ **Note:** The **benchmark (test)** is released now. **GRT model/implementation** will be released later.

---

## 🧪 Tasks
- **Dense Video QA (DIVE)** – question answering that requires **frame-dense reasoning**  
  ↳ Dataset on 🤗: **https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation**

Minimal loading example:
```python
from datasets import load_dataset
ds = load_dataset("haichaozhang/DenseVideoEvaluation", split="test")
print(ds[0])
```

---

## ⚙️ Usage (Evaluation via LMMS-EVAL)
We are preparing a PR to integrate DIVE into **[LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval)**.

### Install LMMS-EVAL
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .
```

### Run (example with LLaVA-OneVision; customize as needed)
```bash
accelerate launch   --num_processes=1   -m lmms_eval   --model llava_onevision   --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen"   --tasks mme   --batch_size 1   --log_samples   --log_samples_suffix fps0.005   --output_path ./logs/   --verbosity=DEBUG
```

### Run (placeholder for our dense-video variant)
```bash
accelerate launch   --num_processes=1   -m lmms_eval   --model llava_ov_dense_video   --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen,use_gated_tok=True,use_vision_merge=False,profiling=False,dense_frame_fps=0.001"   --tasks mvbench   --batch_size 1   --log_samples   --output_path ./logs/   --verbosity=DEBUG
```

---

## 🗓️ Timeline
- ✅ **2025/09/18** – Release **DIVE benchmark (test split)**
- ⭕ Merge DIVE into **LMMS-EVAL** (PR in preparation)
- ⭕ Release **multi-FPS** variants of the dataset
- ⭕ Add **more dense-video task categories**
- ⭕ **Release full GRT model + training/inference code**
- 💡 Ideas or requests? Open an issue or reach out!

---

## 📎 Links
- 📄 Paper: [arXiv 2509.14199](https://arxiv.org/pdf/2509.14199)  
- 🤗 Dataset: [haichaozhang/DenseVideoEvaluation](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation)  
- 🌐 Project: [Website](https://zhanghaichao.xyz/DenseVideoUnderstand/)  
- 💻 Repo: [GitHub](https://github.com/hai-chao-zhang/DenseVideoUnderstand/)

---

## 📜 Citation
If you find DIVE/GRT useful, please cite:
```bibtex
@article{zhang2025dive,
  title={Dense Video Understanding with Gated Residual Tokenization},
  author={Haichao Zhang and Wenhao Chai and Shwai He and Ang Li and Yun Fu},
  journal={arXiv preprint arXiv:2509.14199},
  year={2025}
}
```

## ⚖️ License
- **Dataset (DIVE)**: OpenRAIL (see dataset card for terms)  
- **Code**: to be announced with the model release
