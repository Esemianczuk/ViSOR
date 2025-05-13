#!/usr/bin/env python3
"""
render_views_random.py – ­random poses in a three-band orbital envelope
with *randomised* radius ρ.

If rerun it resumes from where it left off:

▸ Existing PNGs stay untouched  
▸ New frames are numbered after the last one present  
▸ `views.jsonl` grows by appending new records
"""
import bpy, math, json, random, gc, traceback, sys
from pathlib import Path
from mathutils import Vector

# ───────── user knobs ───────── #
OUT_DIR       = Path("renders_random2")
IMG_SIZE      = 512
NUM_IMAGES    = 10_000            # total frames *after* the run
RHO_MIN       = 2.0               # ρ lower bound
RHO_MAX       = 3.5               # ρ upper bound  (ρ is chosen ∈ [RHO_MIN,RHO_MAX])
# 3-band envelope
DELTA_HI_DEG  =  6                # ±6° around the equator
DELTA_LO_DEG  = 20                # inner edge of “slightly up/down”
PURGE_EVERY   = 400
# ────────────────────────────── #

# ────────────────── sanity & scene setup ────────────────── #
if bpy.context.active_object is None:
    raise RuntimeError("Select an active mesh object first.")
obj   = bpy.context.active_object
scene = bpy.context.scene

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT_DIR / "views.jsonl"

try:
    with open(MANIFEST, "r") as f:
        start_idx = sum(1 for _ in f)
except FileNotFoundError:
    start_idx = 0
    MANIFEST.touch()

scene.render.engine                          = "CYCLES"
scene.render.image_settings.file_format      = "PNG"
scene.render.resolution_x = scene.render.resolution_y = IMG_SIZE
scene.render.resolution_percentage = 100
bpy.context.preferences.edit.use_global_undo = False

# ───────────────────── camera helpers ───────────────────── #
def ensure_cam() -> bpy.types.Object:
    cam = bpy.data.objects.get("RenderCam")
    if cam:
        return cam
    data = bpy.data.cameras.new("RenderCamData")
    data.clip_start = 0.01
    cam = bpy.data.objects.new("RenderCam", data)
    scene.collection.objects.link(cam)
    return cam

def look_at(cam: bpy.types.Object, target: Vector):
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()

cam = ensure_cam()
scene.camera = cam

# ───────────── equal-area θ sampler ───────────── #
def sample_theta(theta_min, theta_max):
    s0, s1 = math.sin(theta_min), math.sin(theta_max)
    s      = random.random() * (s1 - s0) + s0
    return math.asin(max(-1, min(1, s)))

# ───────────── build 3 bands ───────────── #
theta_hi = math.radians(DELTA_HI_DEG)
theta_lo = math.radians(DELTA_LO_DEG)
bands = [
    (-theta_lo, -theta_hi),          # below
    (-theta_hi, +theta_hi),          # equator
    (+theta_hi, +theta_lo)           # above
]
per_band = [NUM_IMAGES // 3] * 3
per_band[0] += NUM_IMAGES - sum(per_band)   # remainder to band-0

# ───────── stratified φ buckets ───────── #
BUCKETS  = int(math.sqrt(NUM_IMAGES))
bucket_w = 2 * math.pi / BUCKETS
def stratified_phi(i):
    col = i % BUCKETS
    return (col + random.random()) * bucket_w

# ───────────────────── render loop ───────────────────── #
frames_needed = NUM_IMAGES - start_idx
if frames_needed <= 0:
    print(f"Dataset already complete ({NUM_IMAGES} frames).")
    sys.exit(0)

idx = start_idx
band_counters = [0, 0, 0]
with open(MANIFEST, "a", buffering=1) as mf:
    for b, (tmin, tmax) in enumerate(bands):
        while band_counters[b] < per_band[b]:
            if idx >= NUM_IMAGES:
                break

            phi   = stratified_phi(idx)
            theta = sample_theta(tmin, tmax)
            rho   = random.uniform(RHO_MIN, RHO_MAX)        # ← NEW

            x = rho * math.cos(theta) * math.cos(phi)
            y = rho * math.cos(theta) * math.sin(phi)
            z = rho * math.sin(theta)
            cam.location = Vector((x, y, z))
            look_at(cam, obj.location)

            fname = f"{idx:05d}.png"
            scene.render.filepath = str(OUT_DIR / fname)

            if (OUT_DIR / fname).exists():                  # resume safety
                idx += 1
                band_counters[b] += 1
                continue

            try:
                bpy.ops.render.render(write_still=True)
                rec = {
                    "file":     fname,
                    "phi":      float(phi),
                    "theta":    float(theta),
                    "rho":      float(rho),                 # store actual ρ
                    "location": [float(x), float(y), float(z)]
                }
                mf.write(json.dumps(rec) + "\n")

                if "Render Result" in bpy.data.images:
                    bpy.data.images.remove(bpy.data.images["Render Result"])

                if idx and idx % PURGE_EVERY == 0:
                    bpy.ops.outliner.orphans_purge(
                        do_local_ids=True, do_linked_ids=True, do_recursive=True)
                    gc.collect()

            except Exception:
                traceback.print_exc()
                print(f"skip {idx}", file=sys.stderr)

            idx += 1
            band_counters[b] += 1
            print(f"{idx - start_idx}/{frames_needed} new frames", end="\r")

print(f"\nFinished. Dataset now has {idx} images in {OUT_DIR}.")
print(f"Manifest → {MANIFEST.resolve()}")
