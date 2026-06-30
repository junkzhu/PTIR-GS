# PTIR-GS: Path-Traced Inverse Rendering with Global Illumination in 3D Gaussian Fields

<p align="center">
  <img src="assets/teaser.jpg" width="100%" style="background-color: white;">
</p>

### [[Project]](https://junkzhu.github.io/project_pages/PTIR/) [[Paper]](https://arxiv.org/abs/2606.09606) 
          
> [Junke Zhu](https://github.com/junkzhu), [Hao Zhang](https://github.com/0xrabbyte), [Yutian Zhu](https://rubatotree.github.io/academic/), [Ang Li](https://github.com/Alan-sp), [Chenxiao Hu](https://github.com/Hineven), [Meng Gai](https://cs.pku.edu.cn/info/1160/3584.htm), Fei Zhu, [Zhangjin Huang](http://staff.ustc.edu.cn/~zhuang/), [Sheng Li](https://lishengpku.github.io/)

## Framework
<p align="center">
  <img src="assets/pipeline.jpg" width="100%">
</p>

## Dependencies and Installation
Clone the repository with submodules first:

```bash
git clone --recursive https://github.com/junkzhu/PTIR.git
cd PTIR
```

<details open>
<summary><strong>Linux</strong></summary>

```bash
chmod +x install_env.sh
./install_env.sh ptir
conda activate ptir
```

For CUDA 12.8.1:

```bash
CUDA_VERSION=12.8.1 ./install_env.sh ptir
conda activate ptir
```

If your default GCC version is higher than 11:

```bash
# CUDA 11.8.0
./install_env.sh ptir WITH_GCC11
conda activate ptir

# CUDA 12.8.1
CUDA_VERSION=12.8.1 ./install_env.sh ptir WITH_GCC11
conda activate ptir
```

</details>

<details open>
<summary><strong>Windows</strong></summary>

```powershell
.\install_env.ps1 -CondaEnv ptir
conda activate ptir
```

</details>

## Dataset
Download the following datasets:
- TensoIR_Material: [LINK](https://drive.google.com/file/d/1OM5LcHIHD2oBPhuSeU8vRtf3ILuI47Tj/view?usp=sharing)
- Synthetic4Relight: [LINK](https://drive.google.com/file/d/1Lr4Ola4XA0yqs2UAUWDdRs1Ww4gwOY2S/view?usp=sharing)

Put the datasets under the `data` folder as below:
```
data/
    TensoIR_Material/
    Synthetic4Relight/
```

## Training and Evaluation
Run the benchmark scripts for training and evaluation:
```bash
bash benchmark/tensoir.sh --cuda_device 0,1,2,3
bash benchmark/synthetic4relight.sh --cuda_device 0,1,2,3
```

Run a single scene:
```bash
bash benchmark/tensoir.sh --cuda_device 0 --scenes "ficus"
bash benchmark/synthetic4relight.sh --cuda_device 0 --scenes "hotdog"
```

Run arbitrary light relighting:
```bash
CUDA_VISIBLE_DEVICES=0 python render.py \
    --checkpoint path/to/ckpt_last.pt \
    --path path/to/data \
    --out-dir path/to/output \
    --environment-path path/to/envmap.hdr \
    --lights-relight
```

The light setup is configured in `render.py` under `renderer.model.lights`.

<p align="center">
  <img src="assets/garden.gif" width="49%">
  <img src="assets/kitchen.gif" width="49%">
</p>

## Mitsuba Implementation
A Mitsuba-based implementation is available at [PTIR-Mitsuba](https://github.com/junkzhu/PTIR-Mitsuba).
Before running the Mitsuba pipeline, please first refine the GS model separately, then use the refined GS model as the input to PTIR-Mitsuba.

## Acknowledge
Our work is built upon the following works:
- [3DGRUT](https://github.com/nv-tlabs/3dgrut) (Baseline of This Framework)
- [RGB2X](https://github.com/zheng95z/rgbx)

We also refer to the implementations of the following works:
- [TensoIR](https://github.com/Haian-Jin/TensoIR)
- [GS-IR](https://github.com/lzhnb/gs-ir)
- [GS-ID](https://github.com/dukang/gs-id)
- [R3DG](https://github.com/NJU-3DV/Relightable3DGaussian)
- [IRGS](https://github.com/fudan-zvg/IRGS)
- [SVG-IR](https://github.com/learner-shx/SVG-IR)

Thanks for these great projects!

## BibTeX
If you find our code or paper helps, please consider citing:
```bibtex
@misc{ptir-gs,
      title={Path-Traced Inverse Rendering with Global Illumination in 3D Gaussian Fields}, 
      author={Junke Zhu and Hao Zhang and Yutian Zhu and Ang Li and Chenxiao Hu and Meng Gai and Fei Zhu and Zhangjin Huang and Sheng Li},
      year={2026},
      eprint={2606.09606},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2606.09606}, 
}
```
