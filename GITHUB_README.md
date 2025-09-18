# 🤿 **DENSE VIDEO UNDERSTANDING WITH GATED RESIDUAL TOKENIZATION**

### **Dense Information Video Evaluation (DIVE) Benchmark**

The **first-ever benchmark** dedicated to **Dense Video Understanding**,  
focusing on **QA-driven high-frame-rate video comprehension**,  
where the **answer-relevant information** is present **in nearly every frame**.

---

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/66393f5a1231260674ae798e/uOmH6pKW5yqk6PstJ4H8R.jpeg"
       alt="DIVE" height="200">
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2509.14199">
    <img src="https://img.shields.io/badge/Arxiv-Paper-red?style=for-the-badge&logo=arxiv" alt="Arxiv Paper"/>
  </a>
  <a href="https://zhanghaichao.xyz/DenseVideoUnderstand/">
    <img src="https://img.shields.io/badge/Web-Page-blue?style=for-the-badge&logo=google-chrome" alt="Web Page"/>
  </a>
  <a href="https://github.com/hai-chao-zhang/DenseVideoUnderstand/">
    <img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github Code"/>
  </a>
  <a href="https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=for-the-badge&logo=huggingface" alt="HuggingFace Dataset"/>
  </a>
</p>

---

## 📖 **Introduction**

**Dense Video Understanding** is a new paradigm for video comprehension where answers rely on **dense temporal information**, meaning almost every frame matters.

Traditional video benchmarks and models use **low-FPS sampling** or sparse keyframes to avoid high computational costs, but this misses fine-grained temporal signals.  
To address this, we introduce:

- **DIVE Benchmark** — The first dataset explicitly designed for **dense, frame-level QA tasks**.
- **GRT (Gated Residual Tokenization)** — A two-stage token optimization framework:
  1. **Motion-Compensated Gating**: Skip static regions during tokenization to reduce redundant computation.
  2. **Residual Token Merging**: Merge similar tokens post-tokenization to further compress video representation.

For more details, check our [Arxiv Paper](https://arxiv.org/html/2509.14199).

> **Note:** Currently, we have **released only the benchmark test set**, not the full model.  
> Model weights and training code will be released in a future update.

---

## 👥 **Authors**

<p align="center">
  <a href="https://zhanghaichao.xyz"><b>Haichao Zhang<sup>1</sup></b></a> ·
  <a href="https://wenhaochai.com/"><b>Wenhao Chai<sup>2</sup></b></a> ·
  <a href="https://shwai-he.github.io/"><b>Shwai He<sup>3</sup></b></a> ·
  <a href="https://www.ang-li.com/"><b>Ang Li<sup>3</sup></b></a> ·
  <a href="https://www1.ece.neu.edu/~yunfu/"><b>Yun Fu<sup>1</sup></b></a>
</p>

<p align="center">
  <sub>
    <b>1</b> Northeastern University &nbsp;&nbsp; | &nbsp;&nbsp;
    <b>2</b> Princeton University &nbsp;&nbsp; | &nbsp;&nbsp;
    <b>3</b> University of Maryland, College Park
  </sub>
</p>

---

## 📅 **Timeline**

| Date | Status | Description |
|------|--------|-------------|
| **2025/09/18** | ✅ | Release the **DIVE benchmark test data** on [Hugging Face](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation) |
| *TBD* | ⭕ | Release **test code** on [GitHub](https://github.com/hai-chao-zhang/DenseVideoUnderstand/) |
| *TBD* | ⭕ | Merge into **LMMS-EVAL** VLM test kit |
| *TBD* | ⭕ | Release **multi-FPS versions** of the dataset |
| *TBD* | ⭕ | Add **diverse dense video task categories** |
| *TBD* | ⭕ | **Release full model weights and training code** |
| *Future Ideas* | 💡 | Contact us for suggestions and collaborations! |

---

## ⚙️ **Usage**

The DIVE benchmark is available on Hugging Face for easy access:  
🔗 [https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation)

DIVE is being integrated into [**LMMS-EVAL**](https://github.com/EvolvingLMMs-Lab/lmms-eval) for seamless benchmarking.  
*(Pull request in progress — coming soon!)*

---

### **Installation**

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .
```

---

### **Benchmark Evaluation Example**

Evaluate on `llava_onevision` using the DIVE dataset:

```bash
accelerate launch   --num_processes=1   -m lmms_eval     --model llava_onevision   --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen"   --tasks mme   --batch_size 1   --log_samples   --log_samples_suffix fps0.005   --output_path ./logs/   --verbosity=DEBUG >> log.txt 2>&1
```

---

### **Tips**
- `--tasks`:
  - `mvbench` for multi-video tasks
  - `mme` for general multi-modal evaluation
- Use `--log_samples` to debug and store predictions.
- Adjust `dense_frame_fps` to control temporal density.

---

## 🌐 **Links**

- 📄 [Arxiv Paper](https://arxiv.org/pdf/2509.14199)  
- 🌍 [Project Website](https://zhanghaichao.xyz/DenseVideoUnderstand/)  
- 💻 [GitHub Repository](https://github.com/hai-chao-zhang/DenseVideoUnderstand/)  
- 🤗 [Hugging Face Dataset](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation)

---

## ⭐ **Citation**

If you find DIVE or GRT useful, please cite our work:

```bibtex
@article{zhang2025dive,
  title={Dense Video Understanding with Gated Residual Tokenization},
  author={Haichao Zhang and Wenhao Chai and Shwai He and Ang Li and Yun Fu},
  journal={arXiv preprint arXiv:2509.14199},
  year={2025}
}
```
