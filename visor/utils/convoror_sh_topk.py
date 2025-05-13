#!/usr/bin/env python3
"""
convoror_sh_topk.py

Combines two steps in a single script:
 1) (Preproc) For each pixel, pick the K=64 cameras whose pinhole ray is
    "best" by some metric (we'll use cos(z) or dot with world-forward).
    Save:
      cams_idx.npy  shape (P,K) int
      cams_dir.npy  shape (P,K,3) float32 (the direction each camera sees this pixel)
      cams_col.npy  shape (P,K,3) float32 (the color from that camera)
 2) (Bake) On the GPU, sample Ns directions, find which subset of
    top-K cameras actually meet angle threshold cos>=cos_thr,
    average color, and fit real spherical harmonics.

Usage
-----
# Step 1: Preproc
python convoror_sh_topk.py --mode=preproc --views=views.jsonl --topk=64

# Step 2: Bake
python convoror_sh_topk.py --mode=bake --shorder=5 --samples=48 --chunk=4096 \
       --outfile=sh_billboard_L5.pt

By using "top-K cameras" per pixel, we skip scanning all 4396 cameras
in the final pass. That reduces the final time from O(Npixels * Ncams)
down to O(Npixels * K). Usually K=64–128 is enough.

Dependencies
------------
- NumPy
- PyTorch
- PIL for image I/O
- tqdm
- GPU for the `bake` mode if you like (the script checks CUDA).
"""

import argparse, json, math, sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from PIL import Image
from visor import PROJECT_ROOT, RENDERS_DIR
# --------------------------------------------------------------------
#  Global for resolution & camera projection
# --------------------------------------------------------------------
RES     = 512
FOV_DEG = 60.0
FOCAL   = RES / (2 * math.tan(math.radians(FOV_DEG / 2)))

COS_THR = math.cos(math.radians(5.0))  # ±0.25°


# ====================================================================
#   Step (A): Preprocessing => pick top-K cameras for each pixel
# ====================================================================
def preprocess_views(views_jsonl: Path, topk=64):
    """
    Reads the full list of cameras from views.jsonl, loads each image,
    builds direction fields, picks the top-K by "cosine" measure
    (here we just pick the top-K in 'z' or something simple).
    Then saves 3 arrays:
      cams_idx.npy (P,K)
      cams_dir.npy (P,K,3)
      cams_col.npy (P,K,3)
    in float32 (except idx is int32).
    """

    # 1) load cameras from views.jsonl
    with views_jsonl.open() as f:
        recs = [json.loads(ln) for ln in f]
    V = len(recs)
    print(f"[preproc] Found {V} cameras in {views_jsonl}")

    # 2) build the pinhole direction for each pixel once
    yy, xx = np.meshgrid(np.arange(RES), np.arange(RES), indexing="ij")
    dx = (xx + 0.5 - RES/2)/ FOCAL
    dy = (yy + 0.5 - RES/2)/ FOCAL
    d_cam = np.stack([dx, -dy, np.ones_like(dx)], axis=-1).reshape(-1,3).astype(np.float32)
    P = RES*RES

    # We'll store a "score" array for each camera or just keep track of the topK
    # approach: keep arrays "best_cos" shape (P,K), "best_idx" shape (P,K), "best_dir"(P,K,3), "best_col"(P,K,3).
    best_cos = np.full((P, topk), -1e6, np.float32)
    best_idx = np.full((P, topk), -1, np.int32)
    best_dir = np.zeros((P, topk,3), np.float32)
    best_col = np.zeros((P, topk,3), np.float32)

    # 3) for each camera, parse location => rotation => direction for all P => measure "score"
    up = np.array([0.,1.,0.], dtype=np.float32)
    for c_id, rec in enumerate(tqdm(recs, desc="[preproc] scanning cameras")):
        # parse location
        if "location" in rec:
            cam_xyz = np.array(rec["location"], dtype=np.float32)
        else:
            phi= rec["phi"]; theta= rec["theta"]; rho= rec["rho"]
            cam_xyz= np.array([
                rho*math.cos(theta)*math.cos(phi),
                rho*math.sin(theta),
                rho*math.cos(theta)*math.sin(phi)
            ], dtype=np.float32)
        # build R
        fwd  = -cam_xyz/ (np.linalg.norm(cam_xyz)+1e-14)
        right= np.cross(up, fwd); right/= (np.linalg.norm(right)+1e-14)
        up2  = np.cross(fwd, right)
        R    = np.stack([right, up2, fwd],0)  # shape(3,3)

        # camera directions => shape (P,3)
        world_dir = (R.T @ d_cam.T).T
        norm_     = np.linalg.norm(world_dir, axis=1, keepdims=True)+1e-14
        world_dir/= norm_

        # We'll define "score" = the dot with +Z or something that
        # helps pick "closest" cameras. For a more general approach,
        # you might do e.g. world_dir·(some vantage?), or just use
        # the 'z' component of the direction. We'll do:
        #   score = world_dir[:,2]
        # so we pick cameras whose forward is largest "z" for each pixel.
        # This is a heuristic. Another approach: we want the best alignment
        # with the camera "fwd"? That's basically fwd·world_dir (?), or we can
        # do the simpler z approach if your billboard is upright.
        # We'll do "score = fwd dot world_dir => row-wise"
        # shape => (P,)
        score = np.einsum("ij, j->i", world_dir, fwd)

        # load image => shape(RES,RES,3)
        # Might need to resize if not exactly 512
        from PIL import Image
        img_pil= Image.open((views_jsonl.parent/ rec["file"])).convert("RGB")
        if img_pil.size != (RES,RES):
            img_pil= img_pil.resize((RES,RES), resample=Image.LANCZOS)
        img_np = np.asarray(img_pil, dtype=np.float32)/255.0
        col_flat = img_np.reshape(-1,3)

        # now for each pixel in [0..P-1], if score[p] is better than best_cos[p].min => replace
        # we can do a quick approach: we find the worst in the topK => if score> that => replace
        for p in range(P):
            worst_i = np.argmin(best_cos[p])  # index in [0..K-1]
            if score[p] > best_cos[p,worst_i]:
                best_cos[p,worst_i] = score[p]
                best_idx[p,worst_i] = c_id
                best_dir[p,worst_i] = world_dir[p]
                best_col[p,worst_i] = col_flat[p]

    # 4) save .npy files
    np.save("cams_idx.npy", best_idx)
    np.save("cams_dir.npy", best_dir)
    np.save("cams_col.npy", best_col)
    print(f"[preproc] => wrote cams_idx.npy, cams_dir.npy, cams_col.npy with shape = {best_idx.shape}, {best_dir.shape}, {best_col.shape}")


# ====================================================================
#   Step (B): GPU Bake using top-K arrays
# ====================================================================

def _legendre_p_batch(L, x: torch.Tensor):
    """
    Return shape (L+1, L+1, N).
    """
    N= x.shape[0]
    out= torch.zeros((L+1, L+1, N), dtype=torch.float32, device=x.device)
    out[0,0] = 1.0
    if L==0:
        return out
    out[1,0]= x
    out[1,1]= torch.sqrt(torch.clamp(1 - x*x,1e-15))
    for l in range(2,L+1):
        out[l,l]= (2*l -1)* out[1,1]* out[l-1,l-1]
        for m in range(l-1,-1,-1):
            if l==1 and m==0: continue
            a= (2*l -1)* x* out[l-1,m]
            b= (l+m-1)* out[l-2,m] if (l-2)>=0 else 0.0
            out[l,m]= (a-b)/(l-m)
    return out

def real_sh(dirs: torch.Tensor, L: int):
    """
    dirs: (N,3)
    return: (N,(L+1)^2)
    """
    N= dirs.shape[0]
    r= dirs.norm(dim=-1).clamp_min(1e-14)
    x= dirs[:,0]
    y= dirs[:,1]
    z= dirs[:,2]
    theta= torch.acos((z/r).clamp(-1,1))
    phi  = torch.atan2(y,x)
    P= _legendre_p_batch(L, torch.cos(theta))
    from math import factorial, pi, sqrt
    # norm
    norm_lm= torch.zeros((L+1,L+1), dtype=torch.float32, device=dirs.device)
    for l in range(L+1):
        for m in range(l+1):
            num= factorial(l-m)
            den= factorial(l+m)
            norm_lm[l,m]= math.sqrt((2*l+1)/(4*math.pi)* num/den)

    out_dim= (L+1)*(L+1)
    out= torch.empty((N,out_dim), dtype=torch.float32, device=dirs.device)
    mgrid= torch.arange(L+1, device=dirs.device, dtype=torch.float32).view(-1,1)
    cφ= torch.cos(mgrid* phi)
    sφ= torch.sin(mgrid* phi)
    idx=0
    for l in range(L+1):
        for m in range(-l,l+1):
            if m==0:
                out[:, idx]= norm_lm[l,0]* P[l,0]
            elif m>0:
                sign= (-1)**m
                out[:, idx]= math.sqrt(2.)* sign* norm_lm[l,m]* P[l,m]* cφ[m]
            else:
                mp= -m
                sign= (-1)**m
                out[:, idx]= math.sqrt(2.)* sign* norm_lm[l,mp]* P[l,mp]* sφ[mp]
            idx+=1
    return out

def fit_sh_least_squares(L: int, sample_dirs: torch.Tensor, sample_cols: torch.Tensor):
    """
    sample_dirs: (Ns,3)
    sample_cols: (Ns,3)
    => returns => ( (L+1)^2, 3 ) float32
    """
    B= real_sh(sample_dirs, L)  # (Ns,basis_dim)
    lam= 1e-8
    BT= B.t()
    BTB= BT @ B
    regI= lam* torch.eye(BTB.shape[0], device=B.device)
    inv_ = torch.inverse(BTB+ regI)
    pseudo= inv_@BT
    c= pseudo @ sample_cols  # => (basis_dim, 3)
    return c


@torch.no_grad()
def bake_topk(L: int, Ns: int, chunk_px=4096, outfile="sh_billboard.pt"):
    """
    Loads the top-K .npy arrays from disk, then does the usual random directions
    (Ns), for each pixel chunk we do:
      - build cos_sim => (chunk_size, K, Ns)
      - mask => cos_sim>=cos_thr
      - average color
      - fit SH
    Write to `outfile`.
    """
    device= "cuda" if torch.cuda.is_available() else "cpu"
    # 1) load top-K arrays
    print("[bake] Loading .npy arrays cams_idx.npy, cams_dir.npy, cams_col.npy ...")
    best_idx= np.load("cams_idx.npy")  # shape (P,K)
    best_dir= np.load("cams_dir.npy")  # shape (P,K,3)
    best_col= np.load("cams_col.npy")  # shape (P,K,3)
    P, K= best_idx.shape
    assert best_dir.shape==(P,K,3)
    assert best_col.shape==(P,K,3)

    # 2) convert them to torch on CPU for chunking
    best_dir_t= torch.from_numpy(best_dir).to(device)  # (P,K,3)
    best_col_t= torch.from_numpy(best_col).to(device)  # (P,K,3)

    # 3) sample random directions => shape (Ns,3)
    rng= torch.Generator(device=device)
    # random
    dirs_samp= F.normalize(torch.randn(Ns,3,generator=rng,device=device), dim=-1)

    # We'll store final => shape (RES,RES,(L+1)^2,3) => do it in CPU
    out_sh= torch.zeros((RES, RES, (L+1)**2, 3), dtype=torch.float32)

    # flatten pixel indexing
    idx_all= torch.arange(P, device=device)
    cos_thr= torch.tensor(COS_THR, device=device)
    # We'll do chunking in pixel space
    basis_dim= (L+1)*(L+1)
    for start_px in range(0,P, chunk_px):
        end_px= min(start_px+ chunk_px, P)
        csz   = end_px- start_px
        sub_i = idx_all[start_px:end_px]  # (csz,)

        # shape => (csz,K,3)
        dir_ck= best_dir_t[sub_i]  # float32
        col_ck= best_col_t[sub_i]  # float32

        # shape => (csz,K,Ns)
        # We'll do a matmul approach:
        # (csz,K,3) dot (Ns,3) => we can reorder (csz*K,3) x (3,Ns) => (csz*K,Ns) => then reshape => (csz,K,Ns)
        dir_ck_v= dir_ck.view(-1,3)  # (csz*K,3)
        cos_sim = (dir_ck_v @ dirs_samp.t()).view(csz, K, Ns)  # (csz,K,Ns)

        mask= (cos_sim>= cos_thr)  # bool => (csz,K,Ns)

        # average color => sum => (csz,Ns,3), hits => (csz,Ns)
        sum_col= torch.zeros((csz, Ns, 3), dtype=torch.float32, device=device)
        hits   = torch.zeros((csz, Ns),    dtype=torch.int32,   device=device)

        # broadcast expansions
        # mask => (csz,K,Ns,1)
        mask_e= mask.unsqueeze(-1)
        # col_ck => (csz,K,3) => need => (csz,K,1,3)
        col_e= col_ck.unsqueeze(2)
        # sum => reduce over K dimension
        # (mask_e* col_e).sum(dim=1) => (csz,Ns,3)
        sum_col= (mask_e* col_e).sum(dim=1)
        hits   = mask.sum(dim=1)  # => (csz,Ns)

        # shape => (csz,Ns,3), (csz,Ns)
        good= (hits>0)
        sum_col[good] /= hits[good].unsqueeze(-1)

        # now sum_col => (csz,Ns,3) => we do a single SH solve for each pixel
        # But that means csz * separate solves => we can do a big batch approach if we like
        # or we can do it individually. We'll do individually for clarity:
        # => shape => (csz, basis_dim,3)
        # We'll do a single "design matrix" approach if we want. Or simpler:
        #   for i in [0..csz-1], fit_sh_least_squares => cost is csz*Nsample*(L+1)^2

        # Batching approach: We'll do:
        #   B = real_sh(dirs_samp, L) => (Ns,basis_dim)
        #   Then cSH = pinv @ sum_col => done
        # so we do one matrix factor for the entire chunk. But that matrix factor is big.
        # We'll do that approach once, then multiply for each pixel. We'll do:
        #   (Ns,basis_dim)
        #   sum_col => (csz,Ns,3) => reorder => (csz,3,Ns)
        # => then (basis_dim, Ns) x (Ns, csz*3)
        # We'll define a function. Let's do it inline:

        # 1) B => (Ns,basis_dim)
        B= real_sh(dirs_samp, L)    # (Ns, basis_dim)
        lam= 1e-8
        BT= B.t()   # (basis_dim,Ns)
        BTB= BT@B   # (basis_dim,basis_dim)
        regI= lam* torch.eye(BTB.shape[0], device=device)
        invM= torch.inverse(BTB+ regI)  # (basis_dim,basis_dim)
        pseudo= invM@ BT                # => (basis_dim,Ns)

        # 2) shape => sum_col => (csz,Ns,3) => rearr => (csz,3,Ns) => (3, csz, Ns)
        # but we want => (Ns, csz*3) for the multiply
        sc_perm= sum_col.permute(1,0,2)  # => (Ns, csz,3)
        sc_flat= sc_perm.reshape(Ns, csz*3)
        # => (basis_dim,Ns) @ (Ns, csz*3) => (basis_dim, csz*3)
        bigC= pseudo@ sc_flat
        # => shape => (basis_dim, csz,3)
        bigC= bigC.reshape(basis_dim, csz, 3).permute(1,0,2).contiguous()  # => (csz,basis_dim,3)

        # store => out_sh => shape => (P,basis_dim,3) => (RES,RES,basis_dim,3)
        out_sh_flat= out_sh.view(-1, basis_dim,3)
        out_sh_flat[start_px:end_px]= bigC.cpu()

        del dir_ck, col_ck, dir_ck_v, cos_sim, mask, sum_col, hits, B, BT, BTB, regI, invM, pseudo, bigC
        torch.cuda.empty_cache()

    # reshape => (RES,RES,basis_dim,3)
    # save
    torch.save({"sh": out_sh, "L": L}, outfile)
    print(f"[bake] => wrote {outfile} with shape={tuple(out_sh.shape)}")


# ====================================================================
#   MAIN
# ====================================================================
def main():
    ap= argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["preproc","bake"], required=True)
    # preproc
    
    ap.add_argument("--views", type=str, default=str(RENDERS_DIR / "views.jsonl"))
    ap.add_argument("--topk",  type=int, default=128)
    # bake
    ap.add_argument("--shorder", type=int, default=7)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--chunk",   type=int, default=int(4096))
    ap.add_argument("--outfile", type=str, default= str(PROJECT_ROOT / "sh_billboard_L7.pt"))
    args= ap.parse_args()

    if args.mode=="preproc":
        preprocess_views(Path(args.views), topk=args.topk)
    else:
        # run GPU bake using top-K arrays
        bake_topk(
            L= args.shorder,
            Ns= args.samples,
            chunk_px= args.chunk,
            outfile=args.outfile
        )

if __name__=="__main__":
    main()
