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
```


### 🖥️ Interactive Viewer

Launch the viewer (after you have installed **ViSOR**, your CUDA-enabled
PyTorch build **and** *tiny-cuda-nn*, and downloaded the demo checkpoint):

```bash
visor-view                            # uses the paths hard-coded in viewer.py
# or, if you copied the checkpoint somewhere else:
VISOR_CKPT=/path/to/your.pt visor-view
```

> The window opens at **512 × 512** (optionally doubled if
> `SCALE = 2` inside `viewer.py`).  
> If `SHOW_3D_DEBUG = True` a right-hand pane shows nearby camera
> positions and lets you click-jump between them.

---

#### Keyboard / mouse controls

| Keys / action | Effect |
|---------------|--------|
| **W / S**     | Move camera **forward / back** *(local Z)* |
| **A / D**     | Strafe **left / right** *(local X)* |
| **E**         | Move **up** *(+Y)* |
| **Q**         | Move **down** *(-Y)* |
| **R**         | **Reset** to initial pose |
| *(close window)* | Quit |

**Mouse**  
*LMB + drag* (main viewport) – orbit camera  
*Wheel* – dolly in/out (zoom)

---

#### Debug-pane (right half)

| Action | Effect |
|--------|--------|
| *LMB + drag* | Orbit the debug camera |
| *Wheel*   | Zoom the debug camera |
| *Left-click a blue dot* | Teleport the **main** camera to that frame |

Blue dots = training cameras (loaded from `views.jsonl`).  
Red dot = your current view.

---

### Optional helper scripts

| Script | Purpose | Typical command |
|--------|---------|-----------------|
| `visor-depth` | Batch **MiDaS** depth maps (stored as *.npy* + preview PNG) | `visor-depth --views renders/views.jsonl` |
| `visor-bake`  | Bake **Spherical Harmonics**:<br>• `--mode preproc` gathers top-K images per pixel <br>• `--mode bake` fits SH coefficients | `visor-bake --mode preproc --topk 128`<br>`visor-bake --mode bake --shorder 7 --samples 256` |

Each script safely skips files that already exist, so you can interrupt
and resume.

---

### Rendering your own dataset (`render_views_random.py`)

```python
# in Blender’s *Text Editor* (select your mesh, then press Run Script)
exec(open("render_views_random.py").read(), {})
```

* Generates **NUM_IMAGES** PNGs in `renders_random/`
* Appends a matching line to `renders_random/views.jsonl`
* Resumable – rerun to extend the dataset

Key parameters:

```python
OUT_DIR     = Path("renders_random")
NUM_IMAGES  = 10_000        # total frames
RHO_MIN, RHO_MAX = 2.0, 3.5 # camera radius range
DELTA_HI_DEG = 6            # equatorial band ±6°
DELTA_LO_DEG = 20           # high/low-angle bands start here
```

After rendering, point `visor-train` at the folder and (optionally) the
baked **SH** to start training your own dual-billboard model:

```bash
visor-train --sh_file_front renders_random/sh_billboard_L7.pt \
            --sh_file_rear  ""                            # rear SH optional
```


