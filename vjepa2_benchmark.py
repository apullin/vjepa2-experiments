#!/usr/bin/env python3
# vjepa2_benchmark.py
import os, subprocess, time, numpy as np, torch
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor, AutoVideoProcessor, AutoModel

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

HF_MODEL_NAME = "facebook/vjepa2-vitg-fpc64-384"
PT_WEIGHTS    = "/home/ubuntu/vitg-384.pt"
SAMPLE_MP4    = "sample_video.mp4"
N_RUNS        = 10

def download_clip():
    if not os.path.exists(SAMPLE_MP4):
        url = ("https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/"
               "val/bowling/-WH-lxmGJVY_000005_000015.mp4")
        subprocess.run(["wget", url, "-O", SAMPLE_MP4], check=True)
    else:
        print("Already have sample video")

def get_video() -> np.ndarray :
    vr = VideoReader("sample_video.mp4")
    # choosing some frames here, you can define more complex sampling strategy
    frame_idx = np.arange(0, 128, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    return video

# Taken from vjepa2_demo.py
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

def bench_hf(repo_name, n):
    print(f"[HF hub] building model for {repo_name}…")

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(repo_name)
    img_size = hf_transform.crop_size["height"]  # E.g. 384, 256, etc.

    # Create inputs with HF transform
    video = get_video()
    inputs = hf_transform(video, return_tensors="pt")
    x_hf = inputs["pixel_values_videos"] # input, as a tensor
    assert type(x_hf) is torch.Tensor

    model = AutoModel.from_pretrained(repo_name).eval().cuda()
    # The HF model ships the encoder AND predictor, but we ONLY want the encoder.
    encoder = model.encoder

    lat, mem = run_timed(lambda y: encoder.forward(y), x_hf, n, f"HF {repo_name}, ")
    del model; torch.cuda.empty_cache() # cleanup
    return lat, mem

def pt_prepare_video():
    # Sample video
    vr        = VideoReader(SAMPLE_MP4)
    frame_idx = np.arange(0, 128, 2)                       # 64 frames, strided by 2
    np_vid    = vr.get_batch(frame_idx).asnumpy()          # (T,H,W,C) uint8

    # PyTorch preprocessing
    img_size = 384 # hardcoded for now
    pt_transform = build_pt_video_transform(img_size=img_size)
    x_pt = pt_transform(
        torch.from_numpy(np_vid).permute(0, 3, 1, 2)       # to (T,C,H,W)
    ).unsqueeze(0)                                         # → (1,C,T,H,W)
    return x_pt

def bench_pt_eager(n):
    """
    Build the Vision Transformer, load weights, time it in eager mode.
    """
    x_pt = pt_prepare_video()
    _, _, T, H, W = x_pt.shape          # derive from the sample tensor

    print("\n[local PT eager] building model …")
    model = vit_giant_xformers_rope(
        img_size=(H, W),                # square crop so H == W
        num_frames=T
    ).cuda().eval()

    # Load the state dict then population the model with weights
    # TODO we actually don't need to do this for just benchmarking ...
    sd = torch.load(PT_WEIGHTS, weights_only=True, map_location="cpu")["encoder"]
    # Then populate the model
    model.load_state_dict({k.replace("module.","").replace("backbone.",""): v for k,v in sd.items()}, strict=False)
    del sd # cleanup, for low-RAM machines ...

    lat, mem = run_timed(model, x_pt, n, "PT-eager ") # Timed inference
    del model; torch.cuda.empty_cache()
    return lat, mem

def bench_pt_prec(n, prec="fp32"):
    """
    prec ∈ {"fp32", "fp16", "bf16"}.
      • fp16  → half-precision (CUDA only)
      • bf16  → bfloat16   (Ampere/Hopper GPUs, ROCm 5.7+)
      • fp32  → baseline
    """
    assert prec in {"fp32", "fp16", "bf16"}

    x_pt = pt_prepare_video()
    _, _, T, H, W = x_pt.shape          # derive from the sample tensor

    # build & load on CPU first
    model = vit_giant_xformers_rope(
        img_size=(H, W),                # square crop so H == W
        num_frames=T
    ).cuda().eval()

    # Load the state dict then population the model with weights
    # TODO we actually don't need to do this for just benchmarking ...
    sd = torch.load(PT_WEIGHTS, weights_only=True, map_location="cpu")["encoder"]
    # Then populate the model
    model.load_state_dict({k.replace("module.","").replace("backbone.",""): v for k,v in sd.items()}, strict=False)
    del sd # cleanup, for low-RAM machines ...

    # cast + move
    dtype_map = {"fp32": torch.float32,
                 "fp16": torch.float16,
                 "bf16": torch.bfloat16}
    dtype = dtype_map[prec]

    model = model.to(dtype=dtype, device="cuda")
    x_cast = x_pt.to(dtype=dtype, device="cuda")
    del x_pt; torch.cuda.empty_cache()

    lat, mem = run_timed(model, x_cast, n, f"PT-{prec} ")
    del model; torch.cuda.empty_cache()
    return lat, mem

def bench_pt_compiled(n):
    """
    Build + torch.compile the Vision Transformer, then benchmark.
    """
    x_pt = pt_prepare_video()
    _, _, T, H, W = x_pt.shape          # derive from the sample tensor

    print("\n[local PT compiled] building + compiling …")
    model = vit_giant_xformers_rope(
        img_size=(H, W),
        num_frames=T
    ).cuda().eval()

    # Load the state dict then population the model with weights
    # TODO we actually don't need to do this for just benchmarking ...
    sd = torch.load(PT_WEIGHTS, weights_only=True, map_location="cpu")["encoder"]
    # Then populate the model
    model.load_state_dict({k.replace("module.","").replace("backbone.",""): v for k,v in sd.items()}, strict=False)
    del sd # cleanup, for low-RAM machines ...

    model_c = torch.compile(
        model,
        backend="inductor",
        dynamic=False,
        mode="reduce-overhead"
    )
    # warm-up / trigger compile
    print("Compiling model ... ", end="")
    tic = time.perf_counter()
    with torch.no_grad():
        _ = model_c(x_pt.cuda())
        torch.cuda.synchronize()
    toc = time.perf_counter()
    print(f"took {int((toc-tic) * 1e3)} ms")

    lat, mem = run_timed(model_c, x_pt, n, label="PT-comp ")
    del model_c, model
    torch.cuda.empty_cache()
    return lat, mem

def main():

    download_clip()

    #### Benchmark models ####
    lat_hf,  mem_hf  = bench_hf(HF_MODEL_NAME, N_RUNS)
    lat_pt,  mem_pt  = bench_pt_eager(N_RUNS)
    lat_ptc, mem_ptc = bench_pt_compiled(N_RUNS)

    lat_fp16, mem_fp16 = bench_pt_prec(N_RUNS, prec="fp16")
    lat_bf16, mem_bf16 = bench_pt_prec(N_RUNS, prec="bf16")

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
    stats("local PT (fp16)", lat_fp16, mem_fp16)
    stats("local PT (bf16)", lat_bf16, mem_bf16)


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")   # optional TF32 speed path
    main()
