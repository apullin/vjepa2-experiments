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

On GH200 , CUDA 12.8 :

```
=== V-JEPA2 encoder, N = 10 synthetic runs ===
                HF hub:  3766.5 ±  85.9 ms  (min 3638.0, max 3861.7)  peak mem   5533.1 MB
      local PT (eager):  2633.5 ±  49.2 ms  (min 2515.4, max 2689.2)  peak mem   5441.1 MB
   local PT (compiled):  2476.8 ±  44.0 ms  (min 2407.7, max 2523.3)  peak mem   4147.1 MB
```

There is a noticeable difference between the HF and PT models. The HF model must be including some other transforms that we are not accounting for. TODO have to make sure this is a fair comparison.


## TODOs

Too many to list.
- Benchmark other published encoder sizes (weights & vid sizes)
- Benchmark classifier portion
- Change CUDA to a generic `device`, try on MPS