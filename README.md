# OphthaGen

OphthaGen is a high-performance, optimized framework designed for high-fidelity medical image generation (e.g., ophthalmic image synthesis) using Diffusion Transformers. 

---

## Repository Structure

```bash
OphthaGen/
├── diffusion/               # Gaussian Diffusion scheduling & respacing utilities
│   ├── __init__.py
│   ├── diffusion_utils.py
│   ├── gaussian_diffusion.py
│   ├── respace.py
│   └── timestep_sampler.py
├── download.py              # Auto-downloader for pre-trained weights
├── environment.yml          # Conda environment definition file
├── extract_features.py      # Multi-GPU pre-extraction of dataset latents
├── models.py                # Model architecture configurations (XL, L, B, S)
├── sample.py                # Single-GPU image sampling script
├── sample_ddp.py            # Multi-GPU high-throughput image sampling script
├── train.py                 # Accelerate-based high-efficiency training loop
└── README.md                # Documentation
```

---

## Setup & Environment

We provide a pre-configured `environment.yml` file to build a clean Conda environment.

### Create Conda Environment
To create and activate the default environment:
```bash
conda env create -f environment.yml
conda activate OphthaGen
```

---

## Step-by-Step Workflow

### Step 1: Pre-Extract VAE Features
To maximize training speed and optimize memory, we first project raw images into the latent space using a pre-trained AutoencoderKL (VAE) and save the latents to disk.

Run the DDP-based feature extraction script (e.g., using 1 or more GPUs):
```bash
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py \
  --data-path /path/to/dataset/train \
  --features-path /path/to/store/features \
  --image-size 512
```
*Note: This script automatically groups latents by rank to avoid parallel file collisions.*

### Step 2: Model Training
Launch mixed-precision, multi-GPU training via Hugging Face `accelerate`. The script uses `CustomDataset` to load pre-extracted latents directly, yielding massive speedups.

#### Launch on a Single GPU:
```bash
accelerate launch --mixed_precision fp16 train.py \
  --model DiT-L/2 \
  --dataset MESSIDOR \
  --feature-path /path/to/store/features \
  --image-size 512 \
  --num-classes 2 \
  --epochs 140 \
  --global-batch-size 64
```

#### Launch on Multiple GPUs (DDP):
```bash
accelerate launch --multi_gpu --num_processes N --mixed_precision fp16 train.py \
  --model DiT-L/2 \
  --dataset MESSIDOR \
  --feature-path /path/to/store/features \
  --image-size 512 \
  --num-classes 2 \
  --epochs 140 \
  --global-batch-size 64
```

### Step 3: Sampling & Image Generation

#### Single-GPU/CPU Sampling
Generate images for a specific category using `sample.py`:
```bash
python sample.py \
  --model DiT-L/2 \
  --image-size 512 \
  --num-classes 2 \
  --category 1 \
  --cfg-scale 4.0 \
  --num-sampling-steps 250 \
  --ckpt /path/to/checkpoints/min_loss.pt \
  --save-dir gen_STDR/1 \
  --num-imgs 100
```

#### Multi-GPU DDP Sampling (High Throughput)
Generate large-scale datasets in parallel across `N` GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py \
  --model DiT-L/2 \
  --image-size 512 \
  --num-classes 2 \
  --category 1 \
  --cfg-scale 4.0 \
  --num-sampling-steps 250 \
  --ckpt /path/to/checkpoints/min_loss.pt \
  --save-dir gen_STDR/1 \
  --num-imgs 2574
```

---

## Acknowledgements

This project is built upon or inspired by the following open-source projects. We thank the authors for their great work.

*   [DiT (Scalable Diffusion Models with Transformers)](https://github.com/facebookresearch/DiT) - The official architecture implementation.
*   [fast-DiT](https://github.com/chuanyangjin/fast-DiT) - The improved high-performance PyTorch implementation of DiT.
*   [guided-diffusion](https://github.com/openai/guided-diffusion) - OpenAI's codebase for ADM and diffusion evaluation tools.
*   [improved-diffusion](https://github.com/openai/improved-diffusion) - OpenAI's codebase for IDDPM and learned variances.
*   [diffusers](https://github.com/huggingface/diffusers) - Hugging Face's library for state-of-the-art diffusion models (e.g. VAE).
*   [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models library for high-quality ViT blocks.
*   [accelerate](https://github.com/huggingface/accelerate) - Hugging Face's library for simplified multi-GPU DDP training.
