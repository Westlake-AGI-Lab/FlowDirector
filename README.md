<p align="center">

  <h2 align="center">FlowDirector: Training-Free Flow Steering for Precise Text-to-Video Editing</h2>
  <p align="center">
    <a href="https://guangzhaoli.github.io/"><strong>Guangzhao Li</strong></a>
    ·
    <a href="https://ycgu.site/"><strong>Yanming Yang</strong></a>
    ·
    <a href="https://zhangjiewu.github.io/"><strong>Chenxi Song</strong></a>
    ·
    <a href="https://junhaozhang98.github.io//"><strong>Chi Zhang</strong></a>    <br>
    <br>
        <a href="https://arxiv.org/abs/2310.08465"><img src='https://img.shields.io/badge/arXiv-2310.08465-b31b1b.svg'></a>
        <a href='https://showlab.github.io/MotionDirector'><img src='https://img.shields.io/badge/Project_Page-MotionDirector-blue'></a>
        <a href='https://huggingface.co/spaces/ruizhaocv/MotionDirector'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
  </p>

<p align="center">
  <img src="assets/figure1_teaser.png" width="90%" alt="FlowDirector Teaser Results"/>
  <br>
  <em>FlowDirector edits videos based on text prompts, preserving unedited regions and maintaining temporal coherence.</em>
</p>


---

## 🌟 Key Features

*   🌊 **Inversion-Free Editing:** Directly evolves video in data space, bypassing noisy and error-prone inversion processes.
*   ⚙️ **ODE-Driven Transformation:** Smoothly transitions videos along their spatiotemporal manifold, preserving coherence and structural details.
*   🎨 **Spatially Attentive Flow Correction (SAFC):** An attention-guided masking mechanism precisely modulates the ODE velocity field, ensuring unedited regions remain unchanged both spatially and temporally.
*   🎯 **Differential Averaging Guidance (DAG):** A CFG-inspired strategy that leverages differential signals between multiple candidate flows to enhance semantic alignment with target prompts without compromising structural consistency.
*   🏆 **State-of-the-Art Performance:** Outperforms existing methods in instruction adherence, temporal consistency, and background preservation.

---

## 🔥 News

*   [Date] - News item 1...
*   [Date] - News item 2...

---

## 📑 ToDo

*   [ ] Task 1
*   [ ] Task 2

---

## 🚀 Getting Started

### Pre-trained Models

Download the **Wan2.1-T2V-1.3B** model checkpoints from their official sources (e.g., from the <a href="https://github.com/Wan-Video/Wan2.1">Wan2.1 GitHub</a> or <a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B">Hugging Face</a>). You will need to provide the path to the directory containing these checkpoints using the `--ckpt_dir` argument when running the editing script (see examples below).

For instance, if you download them to `/root/autodl-tmp/Wan2.1-T2V-1.3B`, you will use `--ckpt_dir /root/autodl-tmp/Wan2.1-T2V-1.3B`.


### Installation

1.  Clone the repository (replace `YOUR_USERNAME` with the actual path if forked, or use the main repo URL):
    ```bash
    git clone https://github.com/YOUR_USERNAME/FlowDirector.git
    cd FlowDirector
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ How to Use

You can edit a video using the `edit.py` script. Ensure you have a source video, corresponding source/target text prompts, and have downloaded the pre-trained models.

### Single-GPU Editing

Here's an example of how to run video editing on a single GPU:

```bash
bash script_edit_single_gpu.sh
```


### Multi-GPU Editing (using `torchrun`)

For multi-GPU editing (e.g., 4 GPUs):

```bash
bash script_edit_multi_gpu.sh
```

For detailed parameter explanations, please refer to the `edit.py` file.

---

## 🎬 Demos

FlowDirector achieves superior results across various editing tasks:

<p align="center">
  <img src="assets/figure3_qualitative.png" width="90%" alt="More Qualitative Results"/>
  <br>
  <em>Figure 3: Examples of object editing, texture transformation, and attribute modification. (Replace with your actual Figure 3 or a GIF)</em>
</p>

<p align="center">
  <img src="assets/figure4_comparison.png" width="90%" alt="Comparison with SOTA"/>
  <br>
  <em>Figure 4: Comparison with other state-of-the-art video editing methods. (Replace with your actual Figure 4 or a GIF)</em>
</p>

For detailed quantitative comparisons and more visual examples, please refer to our [paper](https://arxiv.org/abs/2310.08465) (Note: ArXiv ID 2310.08465 is already linked above, replace if this is a different paper) and its supplementary material.


---

## 📜 Citation

If you find FlowDirector useful for your research, please cite our paper:

```bibtex
@inproceedings{Li2023FlowDirector,
  title     = {FlowDirector: Training-Free Flow Steering for Precise Text-to-Video Editing},
  author    = {Guangzhao Li and Yanming Yang and Chenxi Song and Chi Zhang},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2310.08465}
}
```

---


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or inquiries, please contact Guangzhao Li at [gzhao.cs@gmail.com] or open an issue in this repository.
