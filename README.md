# **ViSOR – *View-interpolated Sparse Occlusion Refraction***


<p align="center">
  <img src="docs/demo.gif" alt="ViSOR demo" width="100%">
</p>

A research-grade *dual-billboard prism* renderer plus helper tools  
(MiDaS depth batching, SH baking, live viewer, etc.).

---

## ✨  Why “Dual-Billboard Prism”?

Unlike NeRFs that march along every ray, ViSOR collapses the scene into **two** textured sheets:

```text
┌────────────────────┐
│   front  sheet     │   ←  **occlusion layer**  (diffuse ✚ spec ✚ α)
└────────────────────┘
        ▲   tiny Δθ refractions
┌────────────────────┐
│    rear  sheet     │   ←  **refraction layer** (RGB₀,₁,₂ ✚ α)
└────────────────────┘
```



* **80× faster** rendering on commodity GPUs  
* Baked Real-Spherical-Harmonics keep soft lighting  
* Hash-grid latents ✕ camera embeddings generalise to new views

---

## 🔧  Installation (GPU build)

```bash
# 0 · fresh Conda/venv (Python ≥ 3.9)
conda create -n visor python=3.10 -y
conda activate visor

# 1 · pick Torch wheels that match your driver (CUDA 11.8 example)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# 2 · compile tiny-cuda-nn against *that* Torch
pip install ninja cmake
pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch"

# 3 · ViSOR itself (lightweight, pure-Python wheel)
git clone https://github.com/YOURNAME/ViSOR.git
cd ViSOR
pip install -e .
