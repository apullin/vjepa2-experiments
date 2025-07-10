# vjepa2-experiments

This repo documents various experiments with V-JEPA2, starting with some benchmarking.

## vjep2_benchmark.py

This script benchmarks the encoder portion, as published by Meta. Tests are run for the Huggingface version, the local Pytorch verison, and a compiled PyTorch version. The action classifier is not yet included.

Inference time and peak CUDA memory will be reported.

Only a single video is tested, the same one used in the base repo. But the network should be deterministic, so the input data should not matter (I think).

### Setup

Cloine & follow setup instructions from [the official repo](https://github.com/facebookresearch/vjepa2).

This script currently assumes that the weights will be available at `/home/ubuntu/vitg-384.pt`.

Put the VJEPA2 repo `src` path onto your python path:
```
export PYTHONPATH=/home/ubuntu/vjepa-work/vjepa2/:$PYTHONPATH
```

Then run:
```
python -m vjepa2_benchmark
```
### Results

My first results here were wrong! The Huggingface model also packages the predictor, so we have to pick out ONLY the encoder and run that.

Rented big iron is as-delivered by Lambda Labs for on-demand instances.

TODO: separealy benchmark the predictor (is that block important? e.g. if I wanted to do RL/MPC)

TODO: turn these all into a big table.

#### Jetson Orion Nano 8GB

Hugginface Models, in `torch.fp32`:

`=== V-JEPA2 encoder, N = 10 synthetic runs ===`
| Model Name                               | Mean ± Std (ms) | Peak Mem (MB) |
|------------------------------------------|-----------------|---------------|
| facebook/vjepa2-vitl-fpc64-256           | 7112.0 ± 2.4    | 1715.6        |
| facebook/vjepa2-vith-fpc64-256           | 14343.4 ± 2.5   | 3102.0        |
| facebook/vjepa2-vitg-fpc64-256           | 48777.3 ± 23404.4 | 4649.7      |
| facebook/vjepa2-vitl-fpc16-256-ssv2      | 14643.3 ± 15468.4 | 1717.5      |
| facebook/vjepa2-vitl-fpc32-256-diving48  | 7299.4 ± 263.7  | 1717.5        |

Notes:
- Won't load in fp32:
    - `facebook/vjepa2-vitg-fpc64-384`
    - `facebook/vjepa2-vitg-fpc64-384-ssv2`
    - `facebook/vjepa2-vitg-fpc32-384-diving48`

Huggingface Models, in `torch.fp16`:

`=== V-JEPA2 encoder, N = 10 synthetic runs ===`
| Model Name                               | Mean ± Std (ms) | Peak Mem (MB) |
|------------------------------------------|-----------------|---------------|
| facebook/vjepa2-vitl-fpc64-256           | 1661.1 ± 9.6    | 885.9         |
| facebook/vjepa2-vith-fpc64-256           | 3233.0 ± 3.1    | 1569.6        |
| facebook/vjepa2-vitg-fpc64-256           | 4357.9 ± 3.4    | 2341.4        |
| facebook/vjepa2-vitg-fpc64-384           | 14228.7 ± 10.3  | 2780.9        |
| facebook/vjepa2-vitl-fpc16-256-ssv2      | 1687.4 ± 2.3    | 887.8         |
| facebook/vjepa2-vitg-fpc32-384-diving48  | 14252.6 ± 11.7  | 2780.9        |
| facebook/vjepa2-vitl-fpc32-256-diving48  | 1692.9 ± 1.9    | 887.8         |

Interesting results. The source video is 10 seconds long, so in fp16 all models except `vitg-fpc64-384` run faster than realtime.

Notably:
- the `device_map="auto"` is doing heavily lifting here - unclear what is going to CPU, what to GPU
- some models have identical results - maybe the encoders are the same between those?
- `facebook/vjepa2-vitg-fpc64-384-ssv2` will not even load, due to OOM panic
- above benchmarks are made using the **default** Jetson config with ~3.7GB of zram/compressed swap
- high variation for `vitg-fpc64-256` in fp32, only config that shows this behavior - thermal? swapping?
- TODO: redo measurements w/ no zram and swap on ssd
- TODO: redo in bf16 (is it even supported?)

#### GH200 ("gpu_1x_gh200"), CUDA 12.8

All the models from HuggingFace (huge range in performance!):

`=== V-JEPA2 encoder, N = 10 synthetic runs ===`
| Model Name                                 | Mean ± Std (ms)|  Peak Mem (MB) |
|-------------------------------------------|-----------------|----------------|
| facebook/vjepa2-vitl-fpc64-256            | 232.5 ± 0.5     |  1741.2        |
| facebook/vjepa2-vith-fpc64-256            | 413.9 ± 0.5     |  3158.9        |
| facebook/vjepa2-vitg-fpc64-256            | 543.0 ± 0.4     |  4749.3        |
| facebook/vjepa2-vitg-fpc64-384            | 2421.1 ± 1.4    |  5568.3        |
| facebook/vjepa2-vitg-fpc64-384-ssv2       | 2424.3 ± 2.2    |  5568.3        |
| facebook/vjepa2-vitl-fpc16-256-ssv2       | 234.4 ± 0.7     |  1742.0        |
| facebook/vjepa2-vitg-fpc32-384-diving48   | 2424.8 ± 1.6    |  5570.0        |
| facebook/vjepa2-vitl-fpc32-256-diving48   | 234.8 ± 0.4     |  1740.2        |

#### Comparing with the locally-loaded PT model:

with `torch.set_float32_matmul_precision("high")`:
```
=== V-JEPA2 encoder, N = 10 synthetic runs ===
                HF hub:  2413.2 ±   2.1 ms  (min 2409.2, max 2415.6)  peak mem   5533.1 MB
      local PT (eager):  2437.3 ±   1.8 ms  (min 2433.4, max 2439.5)  peak mem   5441.1 MB
   local PT (compiled):  2393.8 ±   1.6 ms  (min 2391.3, max 2396.5)  peak mem   4147.1 MB
```

without `torch.set_float32_matmul_precision("high")`:
```
=== V-JEPA2 encoder, N = 10 synthetic runs ===
                HF hub:  3039.0 ±   2.0 ms  (min 3035.6, max 3042.3)  peak mem   5533.1 MB
      local PT (eager):  3071.8 ±   1.3 ms  (min 3070.1, max 3074.0)  peak mem   5441.1 MB
   local PT (compiled):  3006.5 ±   2.1 ms  (min 3001.4, max 3008.8)  peak mem   4147.1 MB
```

#### With reduction to FP16/BF16:
```
=== V-JEPA2 encoder, N = 30 synthetic runs ===
       local PT (fp16):   401.7 ±   1.4 ms  (min  399.9, max  405.2)  peak mem   2710.4 MB
       local PT (bf16):   398.9 ±   1.0 ms  (min  396.2, max  400.7)  peak mem   2708.5 MB

```
(not affected by `torch.set_float32_matmul_precision("high")`)


## TODOs

Too many to list.
- Benchmark other published encoder sizes (weights & vid sizes)
- Benchmark classifier portion
- How to test i8? (quick test w/ `bitsandbytes` seems to cause a fault on aarch64)
- Change CUDA to a generic `device`, try on MPS