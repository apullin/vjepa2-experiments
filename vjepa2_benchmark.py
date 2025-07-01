#!/usr/bin/env python3
# vjepa2_benchmark.py
import os, subprocess, time, numpy as np, torch
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

HF_MODEL_NAME = "facebook/vjepa2-vitg-fpc64-384"
PT_WEIGHTS    = "/home/ubuntu/vitg-384.pt"          # <-- edit path
SAMPLE_MP4    = "sample_video.mp4"
N_RUNS        = 10

# ---------------------------------------------------------------- helpers
def download_clip():
    if not os.path.exists(SAMPLE_MP4):
        url = ("https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/"
               "val/bowling/-WH-lxmGJVY_000005_000015.mp4")
        subprocess.run(["wget", url, "-O", SAMPLE_MP4], check=True)
    else:
        print("Already have sample video")

# def get_video_shape():
#     vr = VideoReader(SAMPLE_MP4)
#     batch = vr.get_batch(np.arange(0, 128, 2))     # same as demo
#     return batch.shape                              # (T,H,W,C)

# def random_video(shape):
#     return np.random.randint(0, 256, size=shape, dtype=np.uint8)

def build_pt_video_transform(img_size):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

@torch.no_grad()
def run_timed(forward, x, n=3, label=""):
    x = x.cuda()
    torch.cuda.synchronize()

    # always do one warmup run:
    _ = forward(x)
    torch.cuda.synchronize()

    latencies = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(n):
        print(f"{label}{i+1}/{n}  ", end="")
        tic = time.perf_counter()
        _ = forward(x)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        latency = toc - tic
        print( f"{int(latency * 1e3)} ms" )
        latencies.append(toc - tic)

    torch.cuda.synchronize()
    peakmem = torch.cuda.max_memory_allocated() / 1024**2
    return latencies, peakmem

## Do benchmarking in separate functions, so the peak CUDA memory should not overlap ##

def bench_hf(x_hf, n):
    print("\n[HF hub] building model …")
    model = AutoModel.from_pretrained(HF_MODEL_NAME).cuda().eval()
    lat, mem = run_timed(lambda y: model.get_vision_features(y), x_hf, n, "HF ")
    del model; torch.cuda.empty_cache()
    return lat, mem

def bench_pt_eager(x_pt, n):
    """
    Build the Vision Transformer, load weights, time it in eager mode.
    x_pt: (1, C, T, H, W) CUDA *or* CPU tensor — we’ll move it to CUDA inside run_timed.
    """
    print("\n[local PT eager] building model …")
    _, _, T, H, W = x_pt.shape          # derive from the sample tensor
    model = vit_giant_xformers_rope(
        img_size=(H, W),                # square crop so H == W
        num_frames=T
    ).cuda().eval()
    sd = torch.load(PT_WEIGHTS, weights_only=True, map_location="cpu")["encoder"]
    model.load_state_dict({k.replace("module.","").replace("backbone.",""): v for k,v in sd.items()}, strict=False)
    lat, mem = run_timed(model, x_pt, n, "PT-eager ")
    del model; torch.cuda.empty_cache()
    return lat, mem

def bench_pt_compiled(x_pt, n):
    """
    Build + torch.compile the Vision Transformer, then benchmark.
    Shape of x_pt is (1, C, T, H, W) – we infer H (==W) and T from that.
    """
    print("\n[local PT compiled] building + compiling …")

    _, _, T, H, W = x_pt.shape                       # infer from sample tensor
    model = vit_giant_xformers_rope(
        img_size=(H, W),
        num_frames=T
    ).cuda().eval()

    sd = torch.load(PT_WEIGHTS, map_location="cpu", weights_only=True)["encoder"]
    model.load_state_dict(
        {k.replace("module.","").replace("backbone.",""): v for k, v in sd.items()},
        strict=False
    )

    model_c = torch.compile(
        model,
        backend="inductor",
        dynamic=False,
        mode="reduce-overhead"
    )
    with torch.no_grad():
        _ = model_c(x_pt.cuda())          # warm-up / trigger compile
        torch.cuda.synchronize()

    lat, mem = run_timed(model_c, x_pt, n, label="PT-comp ")
    del model_c, model
    torch.cuda.empty_cache()
    return lat, mem

def setup():
    download_clip()
    # Sample video
    vr        = VideoReader(SAMPLE_MP4)
    frame_idx = np.arange(0, 128, 2)                       # 64 frames
    np_vid    = vr.get_batch(frame_idx).asnumpy()          # (T,H,W,C) uint8
    T, H, W, C = np_vid.shape
    print("raw video shape:", (T, H, W, C))                # (64,270,480,3)

    ## Model input shapes are a little different, so prepare both
    # HF preprocessing
    proc_hf   = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    img_size  = proc_hf.crop_size["height"]                # e.g. 384
    x_hf      = proc_hf(np_vid, return_tensors="pt")["pixel_values_videos"]

    # PyTorch preprocessing
    pt_transform = build_pt_video_transform(img_size=img_size)
    x_pt = pt_transform(
        torch.from_numpy(np_vid).permute(0, 3, 1, 2)       # to (T,C,H,W)
    ).unsqueeze(0)                                         # → (1,C,T,H,W)

    return x_hf, x_pt

# ---------------------------------------------------------------- main
def main():

    x_hf, x_pt = setup()

    #### Benchmark models ####
    lat_hf,  mem_hf  = bench_hf(x_hf, N_RUNS)
    lat_pt,  mem_pt  = bench_pt_eager(x_pt, N_RUNS)
    lat_ptc, mem_ptc = bench_pt_compiled(x_pt, N_RUNS)

    ## Print results
    def stats(name, lats, peak):
        ms = np.array(lats) * 1e3
        print(f"{name:>22}: {ms.mean():7.1f} ± {ms.std():5.1f} ms  "
              f"(min {ms.min():6.1f}, max {ms.max():6.1f})  "
              f"peak mem {peak:8.1f} MB")

    print(f"\n=== V-JEPA2 encoder, N = {N_RUNS} synthetic runs ===")
    stats("HF hub",              lat_hf,  mem_hf)
    stats("local PT (eager)",    lat_pt,  mem_pt)
    stats("local PT (compiled)", lat_ptc, mem_ptc)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")   # optional TF32 speed path
    main()
