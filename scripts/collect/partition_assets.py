#!/usr/bin/env python3
"""
Partition BAMBOO 6148 papers into capacity-bounded asset bins.

Algorithm: greedy capacitated set cover.
- Each bin has CAP_BYTES asset budget (default 150 GB).
- Paper p assigned to bin B iff A(p) ⊆ B (strict full-reproduction).
- Greedy step picks the candidate paper whose missing-asset bundle, when
  added to the current bin, brings in the most newly-covered papers (within budget).
- Iterates bins until marginal gain falls below MIN_GAIN.
- A secondary lenient view (per-claim coverage) is computed post-hoc.
"""
import json
import glob
import re
import csv
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

BAMBOO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BAMBOO_ROOT.parent
ASSETS_DIR = BAMBOO_ROOT / "data" / "paper_assets"
OUT_SHARING = BAMBOO_ROOT / "sharing"
OUT_REPORT = WORKSPACE_ROOT / "docs" / "bamboo" / "reports"
OUT_SHARING.mkdir(parents=True, exist_ok=True)
OUT_REPORT.mkdir(parents=True, exist_ok=True)

GB = 1024 ** 3
MB = 1024 ** 2

CAP_BYTES = 150 * GB           # bin asset budget
MIN_GAIN = 2                   # stop adding bins when marginal gain < MIN_GAIN papers
HARD_BIN_LIMIT = 60            # safety cap on number of bins
COVERAGE_CHECKPOINTS = [5, 10, 15, 25, 40, 60]  # report cumulative coverage at these K values

# =============================================================================
# Step 1: alias table for the top heavy/shared assets
# =============================================================================
ALIASES = {
    # ImageNet family — DO NOT merge variants
    "imagenet": "imagenet-1k", "imagenet-1k": "imagenet-1k", "imagenet 1k": "imagenet-1k",
    "imagenet1k": "imagenet-1k", "ilsvrc2012": "imagenet-1k", "ilsvrc-2012": "imagenet-1k",
    "ilsvrc 2012": "imagenet-1k", "imagenet-1000": "imagenet-1k", "imagenet1000": "imagenet-1k",
    "imagenet-21k": "imagenet-21k", "imagenet 21k": "imagenet-21k", "imagenet21k": "imagenet-21k",
    "imagenet-22k": "imagenet-21k", "imagenet 22k": "imagenet-21k", "imagenet22k": "imagenet-21k",
    "imagenet-r": "imagenet-r", "imagenet r": "imagenet-r",
    "imagenet-c": "imagenet-c", "imagenet c": "imagenet-c",
    "imagenet-a": "imagenet-a", "imagenet a": "imagenet-a",
    "imagenet-sketch": "imagenet-sketch", "imagenet sketch": "imagenet-sketch",
    "imagenet-v2": "imagenet-v2", "imagenetv2": "imagenet-v2",
    "tiny imagenet": "tiny-imagenet", "tiny-imagenet": "tiny-imagenet", "tinyimagenet": "tiny-imagenet",
    # COCO
    "coco": "ms-coco-2017", "ms coco": "ms-coco-2017", "ms-coco": "ms-coco-2017",
    "mscoco": "ms-coco-2017", "coco-2017": "ms-coco-2017", "coco 2017": "ms-coco-2017",
    "ms-coco 2017": "ms-coco-2017", "ms coco 2017": "ms-coco-2017", "mscoco-2017": "ms-coco-2017",
    "ms-coco-2017": "ms-coco-2017",
    "coco-2014": "ms-coco-2014", "coco 2014": "ms-coco-2014", "mscoco-2014": "ms-coco-2014",
    "ms-coco-2014": "ms-coco-2014", "ms-coco 2014": "ms-coco-2014",
    "coco caption": "coco-captions", "coco captions": "coco-captions", "coco-captions": "coco-captions",
    # CIFAR
    "cifar 10": "cifar-10", "cifar-10": "cifar-10", "cifar10": "cifar-10",
    "cifar 100": "cifar-100", "cifar-100": "cifar-100", "cifar100": "cifar-100",
    # MNIST
    "mnist": "mnist", "fashion mnist": "fashion-mnist", "fashion-mnist": "fashion-mnist",
    "fashionmnist": "fashion-mnist", "kmnist": "kmnist", "qmnist": "qmnist",
    # NLP eval
    "gsm8k": "gsm8k", "math": "math-benchmark", "math benchmark": "math-benchmark",
    "math-benchmark": "math-benchmark",
    "mmlu": "mmlu", "mmlu-pro": "mmlu-pro", "mmlu pro": "mmlu-pro",
    "hellaswag": "hellaswag", "winogrande": "winogrande", "piqa": "piqa", "siqa": "siqa",
    "arc": "arc", "arc-easy": "arc-easy", "arc easy": "arc-easy",
    "arc-challenge": "arc-challenge", "arc challenge": "arc-challenge",
    "ai2 reasoning challenge": "arc",
    "boolq": "boolq", "openbookqa": "openbookqa", "obqa": "openbookqa",
    "humaneval": "humaneval", "mbpp": "mbpp", "truthfulqa": "truthfulqa",
    "drop": "drop", "race": "race", "squad": "squad", "squad v1.1": "squad",
    "squad-v1.1": "squad", "squad v2.0": "squad-v2", "squad-v2": "squad-v2",
    "lambada": "lambada", "wikitext-2": "wikitext-2", "wikitext-103": "wikitext-103",
    "agieval": "agieval", "bbh": "bbh", "bigbench": "bigbench", "big-bench": "bigbench",
    "big-bench-hard": "bbh", "big bench hard": "bbh",
    "ifeval": "ifeval", "if-eval": "ifeval", "alpaca-eval": "alpaca-eval",
    "alpacaeval": "alpaca-eval", "mt-bench": "mt-bench", "mtbench": "mt-bench",
    # VQA / VLM eval
    "vqa": "vqa-v2", "vqa v2": "vqa-v2", "vqa-v2": "vqa-v2", "vqav2": "vqa-v2",
    "vqa 2.0": "vqa-v2", "ok-vqa": "ok-vqa", "okvqa": "ok-vqa",
    "gqa": "gqa", "textvqa": "textvqa", "text-vqa": "textvqa",
    "mme": "mme", "mmbench": "mmbench", "mm-bench": "mmbench",
    "pope": "pope", "mathvista": "mathvista", "math-vista": "mathvista",
    "seed-bench": "seed-bench", "seedbench": "seed-bench",
    "scienceqa": "scienceqa", "science qa": "scienceqa", "science-qa": "scienceqa",
    # Vision fine-grained
    "ade20k": "ade20k", "ade 20k": "ade20k", "ade-20k": "ade20k",
    "svhn": "svhn", "celeba": "celeba", "celebahq": "celeba-hq",
    "ucf101": "ucf101", "ucf-101": "ucf101", "kinetics-400": "kinetics-400",
    "kinetics400": "kinetics-400", "kinetics 400": "kinetics-400",
    "stanford cars": "stanford-cars", "stanfordcars": "stanford-cars",
    "stanford-cars": "stanford-cars", "cars": "stanford-cars",
    "eurosat": "eurosat", "dtd": "dtd", "flowers102": "oxford-flowers-102",
    "oxford flowers": "oxford-flowers-102", "oxford-flowers-102": "oxford-flowers-102",
    "oxford-flowers": "oxford-flowers-102", "flowers-102": "oxford-flowers-102",
    "oxford pets": "oxford-pets", "oxford-pets": "oxford-pets", "pets": "oxford-pets",
    "caltech101": "caltech-101", "caltech-101": "caltech-101", "caltech 101": "caltech-101",
    "caltech256": "caltech-256", "caltech-256": "caltech-256",
    "fgvc-aircraft": "fgvc-aircraft", "fgvc aircraft": "fgvc-aircraft",
    "cub": "cub-200-2011", "cub-200": "cub-200-2011", "cub-200-2011": "cub-200-2011",
    "cub 200 2011": "cub-200-2011",
    "places365": "places-365", "places-365": "places-365", "places 365": "places-365",
    # Driving
    "nuscenes": "nuscenes", "nu-scenes": "nuscenes",
    "kitti": "kitti", "kitti-360": "kitti-360",
    "cityscapes": "cityscapes", "waymo": "waymo", "waymo open": "waymo",
    "waymo-open": "waymo",
    # Time series & graph
    "ett": "ett", "etth1": "ett-h1", "etth2": "ett-h2", "ettm1": "ett-m1", "ettm2": "ett-m2",
    "weather": "weather-ts", "electricity": "electricity-ts",
    "traffic": "traffic-ts", "exchange-rate": "exchange-rate",
    "cora": "cora", "citeseer": "citeseer", "pubmed": "pubmed-graph",
    "ogb": "ogb", "ogbn-arxiv": "ogbn-arxiv", "ogbn-products": "ogbn-products",
    # Pre-training corpora
    "c4": "c4", "c4 (colossal clean crawled corpus)": "c4",
    "common crawl": "common-crawl", "commoncrawl": "common-crawl",
    "the pile": "the-pile", "thepile": "the-pile", "redpajama": "redpajama",
    "slimpajama": "slimpajama", "wikipedia": "wikipedia-dump",
    "wikipedia dump": "wikipedia-dump", "wiki dump": "wikipedia-dump",
    "books3": "books3", "openwebtext": "openwebtext",
    # Speech / audio
    "librispeech": "librispeech", "common voice": "common-voice", "commonvoice": "common-voice",
    # Llama family — base/chat/instruct kept SEPARATE (not interchangeable for benchmarks),
    # but "-hf" suffix (HuggingFace name) merges with base.
    "llama-2-7b": "llama-2-7b", "llama-2-7b-hf": "llama-2-7b", "llama 2 7b": "llama-2-7b",
    "llama2-7b": "llama-2-7b", "llama-2 7b": "llama-2-7b",
    "llama-2-7b-chat": "llama-2-7b-chat", "llama-2-7b-chat-hf": "llama-2-7b-chat",
    "llama2-7b-chat": "llama-2-7b-chat", "llama-2 7b chat": "llama-2-7b-chat",
    "llama-2-13b": "llama-2-13b", "llama-2-13b-hf": "llama-2-13b",
    "llama-2-13b-chat": "llama-2-13b-chat", "llama-2-13b-chat-hf": "llama-2-13b-chat",
    "llama-2-70b": "llama-2-70b", "llama-2-70b-hf": "llama-2-70b",
    "llama-2-70b-chat": "llama-2-70b-chat", "llama-2-70b-chat-hf": "llama-2-70b-chat",
    "llama-3-8b": "llama-3-8b", "meta-llama-3-8b": "llama-3-8b",
    "llama3-8b": "llama-3-8b",
    "llama-3-8b-instruct": "llama-3-8b-instruct", "meta-llama-3-8b-instruct": "llama-3-8b-instruct",
    "llama3-8b-instruct": "llama-3-8b-instruct",
    "llama-3-70b": "llama-3-70b", "meta-llama-3-70b": "llama-3-70b",
    "llama-3-70b-instruct": "llama-3-70b-instruct",
    "meta-llama-3-70b-instruct": "llama-3-70b-instruct",
    "llama-3.1-8b": "llama-3.1-8b", "meta-llama-3.1-8b": "llama-3.1-8b",
    "llama3.1-8b": "llama-3.1-8b",
    "llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "meta-llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "llama3.1-8b-instruct": "llama-3.1-8b-instruct",
    "llama-3.1-70b": "llama-3.1-70b", "llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "llama-3.1-405b": "llama-3.1-405b", "llama-3.1-405b-instruct": "llama-3.1-405b-instruct",
    "llama-3.2-1b": "llama-3.2-1b", "llama-3.2-1b-instruct": "llama-3.2-1b-instruct",
    "llama-3.2-3b": "llama-3.2-3b", "llama-3.2-3b-instruct": "llama-3.2-3b-instruct",
    "llama-3.2-11b-vision": "llama-3.2-11b-vision",
    "llama-3.2-11b-vision-instruct": "llama-3.2-11b-vision-instruct",
    "llama-3.2-90b-vision-instruct": "llama-3.2-90b-vision-instruct",
    "llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "llama-1-7b": "llama-1-7b", "llama-7b": "llama-1-7b",
    "llama-1-13b": "llama-1-13b", "llama-13b": "llama-1-13b",
    "llama-1-30b": "llama-1-30b", "llama-30b": "llama-1-30b",
    "llama-1-65b": "llama-1-65b", "llama-65b": "llama-1-65b",
    "tinyllama-1.1b": "tinyllama-1.1b", "tinyllama 1.1b": "tinyllama-1.1b",
    "codellama-7b": "codellama-7b", "codellama-7b-hf": "codellama-7b",
    "codellama-7b-instruct": "codellama-7b-instruct",
    # Mistral / Qwen / Gemma — common variants
    "mistral-7b-v0.1": "mistral-7b-v0.1", "mistralai-mistral-7b-v0.1": "mistral-7b-v0.1",
    "mistral-7b-instruct": "mistral-7b-instruct",
    "mistral-7b-instruct-v0.1": "mistral-7b-instruct-v0.1",
    "mistral-7b-instruct-v0.2": "mistral-7b-instruct-v0.2",
    "mistral-7b-instruct-v0.3": "mistral-7b-instruct-v0.3",
    "qwen2-7b": "qwen2-7b", "qwen2-7b-instruct": "qwen2-7b-instruct",
    "qwen2.5-7b": "qwen2.5-7b", "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "qwen2.5-14b": "qwen2.5-14b", "qwen2.5-14b-instruct": "qwen2.5-14b-instruct",
    "qwen2.5-32b": "qwen2.5-32b", "qwen2.5-32b-instruct": "qwen2.5-32b-instruct",
    "qwen2.5-72b": "qwen2.5-72b", "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
    "qwen2.5-math-7b": "qwen2.5-math-7b", "qwen2.5-coder-7b": "qwen2.5-coder-7b",
    "gemma-2b": "gemma-2b", "gemma-7b": "gemma-7b",
    "gemma-2-2b": "gemma-2-2b", "gemma-2-9b": "gemma-2-9b", "gemma-2-27b": "gemma-2-27b",
    "phi-2": "phi-2", "phi-3-mini-4k-instruct": "phi-3-mini-4k-instruct",
    "phi-3-mini-128k-instruct": "phi-3-mini-128k-instruct",
    # CLIP family kept by patch size
    "clip-vit-base-patch16": "clip-vit-base-patch16",
    "openai-clip-vit-base-patch16": "clip-vit-base-patch16",
    "clip-vit-base-patch32": "clip-vit-base-patch32",
    "openai-clip-vit-base-patch32": "clip-vit-base-patch32",
    "clip-vit-large-patch14": "clip-vit-large-patch14",
    "openai-clip-vit-large-patch14": "clip-vit-large-patch14",
    "clip-vit-huge-patch14": "clip-vit-huge-patch14",
    # API-only models (Closed-source, not downloadable). Treated as 0 GB.
    "gpt-4": "API:gpt-4", "gpt-4-turbo": "API:gpt-4-turbo",
    "gpt-4o": "API:gpt-4o", "gpt-4o-mini": "API:gpt-4o-mini",
    "openai-gpt-4o": "API:gpt-4o", "openai-gpt-4o-mini": "API:gpt-4o-mini",
    "gpt-3.5-turbo": "API:gpt-3.5-turbo", "gpt-3.5": "API:gpt-3.5",
    "openai-gpt-3.5-turbo": "API:gpt-3.5-turbo",
    "gpt-4-1106-preview": "API:gpt-4-turbo", "gpt-4-0125-preview": "API:gpt-4-turbo",
    "claude-3-opus": "API:claude-3-opus", "claude-3.5-sonnet": "API:claude-3.5-sonnet",
    "claude-3-sonnet": "API:claude-3-sonnet", "claude-3-haiku": "API:claude-3-haiku",
    "claude-3.5-haiku": "API:claude-3.5-haiku",
    "claude-2": "API:claude-2", "claude-2.1": "API:claude-2.1",
    "anthropic-claude-3.5-sonnet": "API:claude-3.5-sonnet",
    "gemini-pro": "API:gemini-pro", "gemini-1.5-pro": "API:gemini-1.5-pro",
    "gemini-1.5-flash": "API:gemini-1.5-flash", "gemini-2.0-flash": "API:gemini-2.0-flash",
    "google-gemini-1.5-pro": "API:gemini-1.5-pro",
    "deepseek-chat": "API:deepseek-chat", "deepseek-r1": "API:deepseek-r1",
    "deepseek-v3": "API:deepseek-v3",
    # ImageNet sub-variants — keep separate (different actual sizes)
    "imagenet-1k-(ilsvrc2012)-validation-set": "imagenet-1k-val",
    "imagenet-1k-validation-set": "imagenet-1k-val",
    "imagenet-1k val": "imagenet-1k-val",
    "imagenet-1k-(ilsvrc2012)-training-set": "imagenet-1k",
    "imagenet-1k-training-set": "imagenet-1k",
    "imagenet-1k-(ilsvrc2012)-256x256": "imagenet-1k-256",
    "imagenette": "imagenette",
    "imagenet-100": "imagenet-100", "imagenet100": "imagenet-100",
    "miniimagenet": "mini-imagenet", "mini-imagenet": "mini-imagenet",
    "imagenet-o": "imagenet-o", "imagenet-s": "imagenet-s",
    "imagenet-lt": "imagenet-lt",
}

# Authoritative size table (when LLM-extracted size is missing or absurd)
KNOWN_SIZES_BYTES = {
    "imagenet-1k": int(155 * GB), "imagenet-21k": int(1300 * GB),
    "imagenet-r": int(2 * GB), "imagenet-c": int(50 * GB),
    "imagenet-a": int(0.7 * GB), "imagenet-sketch": int(7 * GB),
    "imagenet-v2": int(0.5 * GB), "tiny-imagenet": int(0.25 * GB),
    "ms-coco-2017": int(25 * GB), "ms-coco-2014": int(19 * GB),
    "coco-captions": int(1 * GB),
    "cifar-10": 170 * MB, "cifar-100": 170 * MB,
    "mnist": 20 * MB, "fashion-mnist": 50 * MB,
    "kmnist": 30 * MB, "qmnist": 30 * MB,
    "gsm8k": 50 * MB, "math-benchmark": 30 * MB,
    "mmlu": 200 * MB, "mmlu-pro": 50 * MB,
    "hellaswag": 60 * MB, "winogrande": 20 * MB, "piqa": 5 * MB, "siqa": 5 * MB,
    "arc": 10 * MB, "arc-easy": 2 * MB, "arc-challenge": 2 * MB,
    "boolq": 5 * MB, "openbookqa": 2 * MB, "humaneval": 2 * MB,
    "mbpp": 2 * MB, "truthfulqa": 2 * MB,
    "drop": 50 * MB, "race": 30 * MB, "squad": 50 * MB, "squad-v2": 50 * MB,
    "lambada": 5 * MB, "wikitext-2": 5 * MB, "wikitext-103": 200 * MB,
    "agieval": 50 * MB, "bbh": 30 * MB, "bigbench": 500 * MB,
    "ifeval": 5 * MB, "alpaca-eval": 10 * MB, "mt-bench": 5 * MB,
    "vqa-v2": int(25 * GB), "ok-vqa": int(13 * GB), "gqa": int(20 * GB),
    "textvqa": int(7 * GB), "mme": int(2 * GB), "mmbench": int(2 * GB),
    "pope": 200 * MB, "mathvista": int(1 * GB), "seed-bench": int(3 * GB),
    "scienceqa": int(2 * GB),
    "ade20k": int(4 * GB), "svhn": int(3.5 * GB), "celeba": int(1.4 * GB),
    "celeba-hq": int(20 * GB),
    "ucf101": int(6.5 * GB), "kinetics-400": int(450 * GB),
    "stanford-cars": int(5.6 * GB), "eurosat": 100 * MB, "dtd": 600 * MB,
    "oxford-flowers-102": 350 * MB, "oxford-pets": 800 * MB,
    "caltech-101": 200 * MB, "caltech-256": 1 * GB,
    "fgvc-aircraft": 3 * GB, "cub-200-2011": int(1.2 * GB),
    "places-365": int(105 * GB),
    "nuscenes": int(326 * GB), "kitti": int(180 * GB), "kitti-360": int(650 * GB),
    "cityscapes": int(25 * GB), "waymo": int(1500 * GB),
    "ett": 50 * MB, "ett-h1": 10 * MB, "ett-h2": 10 * MB,
    "ett-m1": 10 * MB, "ett-m2": 10 * MB,
    "weather-ts": 30 * MB, "electricity-ts": 100 * MB,
    "traffic-ts": 50 * MB, "exchange-rate": 5 * MB,
    "cora": 5 * MB, "citeseer": 5 * MB, "pubmed-graph": 30 * MB,
    "ogbn-arxiv": 200 * MB, "ogbn-products": 1 * GB,
    "c4": int(284 * GB), "common-crawl": int(50 * 1024 ** 3),
    "the-pile": int(800 * GB), "redpajama": int(3 * 1024 ** 4),  # ~3 TB
    "slimpajama": int(900 * GB), "wikipedia-dump": int(20 * GB),
    "books3": int(40 * GB), "openwebtext": int(40 * GB),
    "librispeech": int(60 * GB), "common-voice": int(70 * GB),
    # Llama family
    "llama-2-7b": int(12.6 * GB), "llama-2-7b-chat": int(12.6 * GB),
    "llama-2-13b": int(24.2 * GB), "llama-2-13b-chat": int(24.2 * GB),
    "llama-2-70b": int(128.5 * GB), "llama-2-70b-chat": int(128.5 * GB),
    "llama-3-8b": int(14.9 * GB), "llama-3-8b-instruct": int(14.9 * GB),
    "llama-3-70b": int(130.4 * GB), "llama-3-70b-instruct": int(130.4 * GB),
    "llama-3.1-8b": int(14.9 * GB), "llama-3.1-8b-instruct": int(14.9 * GB),
    "llama-3.1-70b": int(130.4 * GB), "llama-3.1-70b-instruct": int(130.4 * GB),
    "llama-3.1-405b": int(800 * GB), "llama-3.1-405b-instruct": int(800 * GB),
    "llama-3.2-1b": int(2.5 * GB), "llama-3.2-1b-instruct": int(2.5 * GB),
    "llama-3.2-3b": int(6.5 * GB), "llama-3.2-3b-instruct": int(6.5 * GB),
    "llama-3.2-11b-vision": int(22 * GB),
    "llama-3.2-11b-vision-instruct": int(22 * GB),
    "llama-3.2-90b-vision-instruct": int(180 * GB),
    "llama-3.3-70b-instruct": int(130.4 * GB),
    "llama-1-7b": int(12.6 * GB), "llama-1-13b": int(24.2 * GB),
    "llama-1-30b": int(60 * GB), "llama-1-65b": int(122 * GB),
    "tinyllama-1.1b": int(2.2 * GB), "codellama-7b": int(12.6 * GB),
    "codellama-7b-instruct": int(12.6 * GB),
    "mistral-7b-v0.1": int(13 * GB), "mistral-7b-instruct": int(13 * GB),
    "mistral-7b-instruct-v0.1": int(13 * GB), "mistral-7b-instruct-v0.2": int(13 * GB),
    "mistral-7b-instruct-v0.3": int(13 * GB),
    "qwen2-7b": int(15 * GB), "qwen2-7b-instruct": int(15 * GB),
    "qwen2.5-7b": int(14.2 * GB), "qwen2.5-7b-instruct": int(14.2 * GB),
    "qwen2.5-14b": int(27 * GB), "qwen2.5-14b-instruct": int(27 * GB),
    "qwen2.5-32b": int(60 * GB), "qwen2.5-32b-instruct": int(60 * GB),
    "qwen2.5-72b": int(135 * GB), "qwen2.5-72b-instruct": int(135 * GB),
    "qwen2.5-math-7b": int(14.2 * GB), "qwen2.5-coder-7b": int(14.2 * GB),
    "gemma-2b": int(5 * GB), "gemma-7b": int(16 * GB),
    "gemma-2-2b": int(5 * GB), "gemma-2-9b": int(18 * GB), "gemma-2-27b": int(52 * GB),
    "phi-2": int(5.5 * GB), "phi-3-mini-4k-instruct": int(7.5 * GB),
    "phi-3-mini-128k-instruct": int(7.5 * GB),
    "clip-vit-base-patch16": int(0.6 * GB), "clip-vit-base-patch32": int(0.6 * GB),
    "clip-vit-large-patch14": int(1.6 * GB), "clip-vit-huge-patch14": int(4 * GB),
    # ImageNet sub-variants
    "imagenet-1k-val": int(6.7 * GB),
    "imagenet-1k-256": int(50 * GB),
    "imagenette": int(2 * GB), "imagenet-100": int(8 * GB),
    "mini-imagenet": int(7 * GB),
    "imagenet-o": 200 * MB, "imagenet-s": int(2 * GB), "imagenet-lt": int(20 * GB),
}

# API-only model keys (canonical) → 0 GB
API_KEYS_PREFIX = "API:"

PARENS_RE = re.compile(r'^(.*?)\s*\(([^)]+)\)\s*$')
WS_RE = re.compile(r'[\s_]+')

def normalize_name(name):
    """Normalize an asset name to a canonical key. Returns None if name empty."""
    if not name:
        return None
    s = name.lower().strip()
    if not s:
        return None
    # Strip surrounding whitespace, collapse internal whitespace+underscores
    # Try alias lookup at multiple stages
    def lookup(t):
        return ALIASES.get(t)

    # Stage 1: try exact match on raw lowercase
    h = lookup(s)
    if h: return h

    # Stage 2: handle parenthesized alias "X (Y)" → try X, Y
    m = PARENS_RE.match(s)
    if m:
        primary = m.group(1).strip()
        alt = m.group(2).strip()
        for cand in (primary, alt):
            h = lookup(cand)
            if h: return h
            # also try ws-normalized
            cand_ws = WS_RE.sub(' ', cand).strip()
            h = lookup(cand_ws)
            if h: return h
        # Use primary as base going forward
        s = primary

    # Stage 3: strip HF org prefix "org/name"
    if "/" in s:
        s_after = s.split("/", 1)[1]
        h = lookup(s_after)
        if h: return h
        s = s_after

    # Stage 4: collapse whitespace/underscores → single space
    s = WS_RE.sub(' ', s).strip()
    h = lookup(s)
    if h: return h
    # Stage 5: hyphenate
    s_dashed = s.replace(' ', '-')
    h = lookup(s_dashed)
    if h: return h
    # Stage 6: collapse multiple dashes
    s_dashed = re.sub(r'-+', '-', s_dashed)
    h = lookup(s_dashed)
    if h: return h

    return s_dashed


# =============================================================================
# Step 2: load all paper_assets
# =============================================================================
def load_papers():
    files = sorted(glob.glob(str(ASSETS_DIR / "bamboo-*.json")))
    papers = []
    for f in files:
        with open(f) as fp:
            try:
                d = json.load(fp)
            except json.JSONDecodeError:
                continue
        papers.append(d)
    return papers


# =============================================================================
# Step 3: build asset index
# =============================================================================
def build_index(papers):
    # asset_key = (kind, canonical_name)
    asset_to_papers = defaultdict(set)
    paper_to_assets = defaultdict(set)
    asset_size_samples = defaultdict(list)
    asset_display_name = {}
    asset_dead_count = defaultdict(int)
    asset_total_count = defaultdict(int)
    asset_to_claim_required = defaultdict(lambda: defaultdict(set))  # asset → paper → claims requiring it
    paper_claims_total = defaultdict(set)  # all claim ids per paper
    raw_name_count = Counter()  # for diagnostics
    items_no_name = 0
    items_no_size = 0
    papers_no_items = 0

    for d in papers:
        pid = d.get("paper_id")
        if not pid:
            continue
        had_item = False
        for kind, items in [("dataset", d.get("datasets") or []),
                            ("model", d.get("model_weights") or []),
                            ("other", d.get("other_assets") or [])]:
            for it in items:
                if not isinstance(it, dict):
                    continue
                raw_name = it.get("name", "")
                canonical = normalize_name(raw_name)
                if not canonical:
                    items_no_name += 1
                    continue
                key = (kind, canonical)
                had_item = True
                asset_to_papers[key].add(pid)
                paper_to_assets[pid].add(key)
                asset_total_count[key] += 1
                if it.get("availability") == "dead_url":
                    asset_dead_count[key] += 1
                sz = it.get("size_bytes")
                if isinstance(sz, int) and sz > 0:
                    asset_size_samples[key].append(sz)
                else:
                    items_no_size += 1
                if key not in asset_display_name:
                    asset_display_name[key] = raw_name
                # claim attribution
                claims = it.get("required_for_claims") or []
                if isinstance(claims, list):
                    for c in claims:
                        asset_to_claim_required[key][pid].add(c)
                        paper_claims_total[pid].add(c)
                raw_name_count[(kind, raw_name.lower().strip())] += 1
        if not had_item:
            papers_no_items += 1

    return {
        "asset_to_papers": asset_to_papers,
        "paper_to_assets": paper_to_assets,
        "asset_size_samples": asset_size_samples,
        "asset_display_name": asset_display_name,
        "asset_dead_count": asset_dead_count,
        "asset_total_count": asset_total_count,
        "asset_to_claim_required": asset_to_claim_required,
        "paper_claims_total": paper_claims_total,
        "stats": {
            "papers_total": len(papers),
            "papers_no_items": papers_no_items,
            "items_no_name": items_no_name,
            "items_no_size_raw": items_no_size,
            "raw_unique_names": len(raw_name_count),
            "canonical_unique_keys": len(asset_to_papers),
        },
    }


# =============================================================================
# Step 4: pick canonical size per asset
# =============================================================================
def assign_canonical_sizes(idx):
    asset_size = {}
    size_source = {}
    # Per-kind defaults when LLM size is missing — empirically grounded medians
    KIND_DEFAULTS = {"dataset": int(0.8 * GB), "model": int(3 * GB), "other": int(0.1 * GB)}
    for key in idx["asset_to_papers"].keys():
        kind, canon = key
        # API-only models cost 0 GB on disk
        if canon.startswith(API_KEYS_PREFIX):
            asset_size[key] = 0
            size_source[key] = "api-zero"
            continue
        # Priority 1: known table
        if canon in KNOWN_SIZES_BYTES:
            asset_size[key] = KNOWN_SIZES_BYTES[canon]
            size_source[key] = "known-table"
            continue
        # Priority 2: median of LLM-extracted samples
        samples = idx["asset_size_samples"].get(key, [])
        if samples:
            samples_sorted = sorted(samples)
            asset_size[key] = samples_sorted[len(samples_sorted) // 2]
            size_source[key] = f"llm-median-of-{len(samples)}"
            continue
        # Priority 3: unknown — kind-specific default
        asset_size[key] = KIND_DEFAULTS.get(kind, int(0.5 * GB))
        size_source[key] = f"default-{kind}"
    return asset_size, size_source


# =============================================================================
# Step 5: greedy capacitated bin construction
# =============================================================================
def greedy_one_bin(remaining_papers, paper_to_assets, asset_size, cap_bytes, log=None):
    """
    Build one bin by greedy: at each step, find the addition X (a paper's residual
    set) that maximizes # of newly-covered papers within remaining budget.

    Optimization: dedup by missing-set frozenset. Distinct missing-sets are
    evaluated as candidates; subset check against other distinct missing-sets is
    weighted by group size.
    """
    bin_assets = set()
    bin_size = 0
    bin_papers = set()
    inner_steps = 0

    while True:
        unfit = remaining_papers - bin_papers
        if not unfit:
            break

        # Group unfit papers by their (frozen) missing set
        groups = defaultdict(list)
        affordable_X = {}  # frozenset → (cost)
        for p in unfit:
            missing = paper_to_assets[p] - bin_assets
            if not missing:
                bin_papers.add(p)
                continue
            cost = sum(asset_size.get(a, 0) for a in missing)
            f = frozenset(missing)
            groups[f].append(p)
            if bin_size + cost <= cap_bytes:
                affordable_X[f] = cost

        if not affordable_X:
            break

        # Sort affordable candidates ascending by size (cheap-first for tie-break),
        # but search all of them
        distinct = sorted(groups.keys(), key=lambda s: (len(s), tuple(sorted(s))))
        # Index by len for faster subset enumeration
        groups_by_len = defaultdict(list)
        for s in distinct:
            groups_by_len[len(s)].append(s)

        best_X = None
        best_gain = 0
        best_cost = 0
        for X, cost_X in affordable_X.items():
            # Count papers with residual D ⊆ X (sum |group(D)|)
            gain = 0
            for k in range(1, len(X) + 1):
                for D in groups_by_len.get(k, ()):
                    if D.issubset(X):
                        gain += len(groups[D])
            if gain > best_gain or (gain == best_gain and best_X is not None and cost_X < best_cost):
                best_gain = gain
                best_X = X
                best_cost = cost_X

        if best_X is None or best_gain == 0:
            break

        bin_assets |= best_X
        bin_size += best_cost
        # Refresh bin_papers
        for p in remaining_papers:
            if p in bin_papers:
                continue
            if paper_to_assets[p].issubset(bin_assets):
                bin_papers.add(p)
        inner_steps += 1
        if log:
            log(f"    inner step {inner_steps}: +{best_gain} papers, +{best_cost / GB:.2f} GB "
                f"(bin {bin_size / GB:.1f}/{cap_bytes / GB:.0f} GB, {len(bin_assets)} assets, "
                f"{len(bin_papers)} papers)")

    return bin_assets, bin_papers, bin_size


def partition(remaining_papers, paper_to_assets, asset_size, cap_bytes,
              min_gain=MIN_GAIN, hard_limit=HARD_BIN_LIMIT, log=print, verbose_inner=False):
    bins = []
    remaining = set(remaining_papers)
    while remaining and len(bins) < hard_limit:
        t0 = time.time()
        b_assets, b_papers, b_size = greedy_one_bin(
            remaining, paper_to_assets, asset_size, cap_bytes,
            log=(log if verbose_inner else None),
        )
        if not b_papers:
            break
        if len(b_papers) < min_gain:
            log(f"  [stop] marginal bin only {len(b_papers)} papers (< MIN_GAIN={min_gain})")
            break
        bins.append({
            "assets": sorted(b_assets),
            "papers": sorted(b_papers),
            "size_bytes": b_size,
        })
        remaining -= b_papers
        log(f"  bin {len(bins)}: {len(b_papers)} papers, {len(b_assets)} assets, "
            f"{b_size / GB:.1f} GB, remaining {len(remaining)}, t={time.time()-t0:.1f}s")
    return bins, remaining


# =============================================================================
# Step 6: lenient view — for unassigned papers, count claim coverage in best bin
# =============================================================================
def lenient_view(unassigned_papers, bins, paper_to_assets, asset_to_claim_required,
                 paper_claims_total, paper_to_assets_full):
    """
    For each unassigned paper, find the bin maximizing # claims of p whose
    required assets all fit in that bin. Returns: list of dicts.
    """
    rows = []
    for p in unassigned_papers:
        all_claims = paper_claims_total.get(p, set())
        if not all_claims:
            rows.append({"paper_id": p, "best_bin": None, "claims_total": 0, "claims_covered": 0, "ratio": 0.0})
            continue
        best_bin_idx = None
        best_covered = 0
        for i, b in enumerate(bins):
            bin_assets = set(b["assets"])
            covered_claims = set()
            # For each claim c of p, find assets required for c
            for c in all_claims:
                # collect assets required by p for claim c
                required = set()
                for a in paper_to_assets_full[p]:
                    claim_set = asset_to_claim_required.get(a, {}).get(p, set())
                    if c in claim_set:
                        required.add(a)
                # claim c is covered iff all those assets are in bin_assets
                if required and required.issubset(bin_assets):
                    covered_claims.add(c)
                elif not required:
                    # no asset attribution → can't verify; conservatively skip
                    pass
            if len(covered_claims) > best_covered:
                best_covered = len(covered_claims)
                best_bin_idx = i
        rows.append({
            "paper_id": p,
            "best_bin": best_bin_idx,
            "claims_total": len(all_claims),
            "claims_covered": best_covered,
            "ratio": best_covered / max(len(all_claims), 1),
        })
    return rows


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"[load] reading paper_assets from {ASSETS_DIR}")
    papers = load_papers()
    print(f"[load] {len(papers)} papers")

    print("[index] building asset index with normalization")
    idx = build_index(papers)
    print(f"[index] stats: {idx['stats']}")

    print("[size] assigning canonical sizes")
    asset_size, size_source = assign_canonical_sizes(idx)
    src_count = Counter(s.split("-of-")[0] if "-of-" in s else s for s in size_source.values())
    print(f"[size] sources: {dict(src_count)}")

    paper_to_assets = idx["paper_to_assets"]
    all_papers = sorted(set(p["paper_id"] for p in papers if p.get("paper_id")))

    # Per-paper floor (max single asset) and total
    per_paper_max = {}
    per_paper_total = {}
    for p in all_papers:
        assets = paper_to_assets.get(p, set())
        if not assets:
            per_paper_max[p] = 0
            per_paper_total[p] = 0
            continue
        sizes = [asset_size.get(a, 0) for a in assets]
        per_paper_max[p] = max(sizes) if sizes else 0
        per_paper_total[p] = sum(sizes)

    # Classification
    no_items = []
    oversize_floor = []        # max single asset > cap
    oversize_total = []        # total > cap (max ≤ cap)
    fittable = []
    for p in all_papers:
        n_assets = len(paper_to_assets.get(p, set()))
        if n_assets == 0:
            no_items.append(p)
        elif per_paper_max[p] > CAP_BYTES:
            oversize_floor.append(p)
        elif per_paper_total[p] > CAP_BYTES:
            oversize_total.append(p)
        else:
            fittable.append(p)
    print(f"[fit] cap={CAP_BYTES / GB:.0f} GB → "
          f"{len(no_items)} no-items, {len(fittable)} fittable, "
          f"{len(oversize_total)} oversize-total, {len(oversize_floor)} oversize-floor")

    # Greedy partition fittable + no_items
    candidates = list(fittable) + list(no_items)
    print(f"[partition] greedy bin construction over {len(candidates)} papers (cap={CAP_BYTES / GB:.0f} GB, "
          f"min_gain={MIN_GAIN}, hard_limit={HARD_BIN_LIMIT})")
    bins, leftover = partition(candidates, paper_to_assets, asset_size, CAP_BYTES)
    n_assigned = sum(len(b["papers"]) for b in bins)
    print(f"[partition] {len(bins)} bins, {n_assigned} papers assigned, {len(leftover)} leftover")

    # Salvage: for each leftover paper, check if A(p) ⊆ some bin (covers without budget impact)
    salvaged = {}
    bin_assets_sets = [set(b["assets"]) for b in bins]
    for p in list(leftover):
        for i, bset in enumerate(bin_assets_sets):
            if paper_to_assets[p].issubset(bset):
                salvaged[p] = i
                bins[i]["papers"].append(p)
                break
    if salvaged:
        for p in salvaged:
            leftover.discard(p)
        print(f"[salvage] +{len(salvaged)} leftover papers fit existing bins")

    # Lenient view: for each unassigned (leftover + oversize_total + oversize_floor),
    # find best bin by claim coverage
    unassigned = set(leftover) | set(oversize_total) | set(oversize_floor)
    print(f"[lenient] computing claim-coverage for {len(unassigned)} unassigned papers")
    lenient_rows = lenient_view(
        unassigned, bins, paper_to_assets,
        idx["asset_to_claim_required"], idx["paper_claims_total"], paper_to_assets,
    )

    # Per-paper status map for the CSV
    paper_status = {}
    paper_strict_bin = {}
    for p in no_items:
        paper_status[p] = "no-items"
    for p in oversize_floor:
        paper_status[p] = "oversize-floor"
    for p in oversize_total:
        paper_status[p] = "oversize-total"
    for p in leftover:
        paper_status[p] = "leftover"
    for i, b in enumerate(bins):
        for p in b["papers"]:
            paper_status[p] = "salvaged" if p in salvaged else "strict"
            paper_strict_bin[p] = i + 1

    # Coverage curve
    coverage_curve = []
    cum_papers = set()
    for i, b in enumerate(bins):
        cum_papers |= set(b["papers"])
        if (i + 1) in COVERAGE_CHECKPOINTS or (i + 1) == len(bins):
            coverage_curve.append({
                "K": i + 1,
                "papers_covered": len(cum_papers),
                "pct_of_all": round(100 * len(cum_papers) / len(all_papers), 1),
                "pct_of_fittable": round(100 * len(cum_papers) / max(len(candidates), 1), 1),
                "cum_size_gb": round(sum(b2["size_bytes"] for b2 in bins[: i + 1]) / GB, 1),
            })

    # =========================================================================
    # Outputs
    # =========================================================================
    # asset_index.json
    asset_index_out = []
    for key, papers_set in idx["asset_to_papers"].items():
        kind, canon = key
        asset_index_out.append({
            "kind": kind,
            "canonical_name": canon,
            "display_name": idx["asset_display_name"].get(key, canon),
            "size_bytes": asset_size.get(key, 0),
            "size_gb": round(asset_size.get(key, 0) / GB, 3),
            "size_source": size_source.get(key, "?"),
            "n_papers": len(papers_set),
            "dead_url_listings": idx["asset_dead_count"].get(key, 0),
            "total_listings": idx["asset_total_count"].get(key, 0),
        })
    asset_index_out.sort(key=lambda x: -x["n_papers"])
    with open(OUT_SHARING / "asset_index.json", "w") as fp:
        json.dump(asset_index_out, fp, indent=2)
    print(f"[out] asset_index.json with {len(asset_index_out)} entries")

    # bins-K{N}.json
    bins_out = []
    for i, b in enumerate(bins):
        bin_dict = {
            "bin_id": i + 1,
            "size_bytes": b["size_bytes"],
            "size_gb": round(b["size_bytes"] / GB, 2),
            "n_papers": len(b["papers"]),
            "n_assets": len(b["assets"]),
            "assets": [
                {
                    "kind": a[0],
                    "canonical_name": a[1],
                    "display_name": idx["asset_display_name"].get(a, a[1]),
                    "size_bytes": asset_size.get(a, 0),
                    "size_gb": round(asset_size.get(a, 0) / GB, 3),
                    "size_source": size_source.get(a, "?"),
                }
                for a in sorted(b["assets"], key=lambda x: -asset_size.get(x, 0))
            ],
            "papers": b["papers"],
        }
        bins_out.append(bin_dict)
    with open(OUT_SHARING / f"bins-K{len(bins)}.json", "w") as fp:
        json.dump({"cap_gb": CAP_BYTES / GB, "n_bins": len(bins), "bins": bins_out}, fp, indent=2)
    print(f"[out] bins-K{len(bins)}.json")

    # paper_to_bin.csv — full picture per paper
    lenient_by_paper = {r["paper_id"]: r for r in lenient_rows}
    with open(OUT_SHARING / "paper_to_bin.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["paper_id", "fit_status", "strict_bin", "lenient_best_bin",
                    "claims_total", "claims_covered", "claim_ratio",
                    "per_paper_total_gb", "per_paper_max_asset_gb"])
        for p in all_papers:
            status = paper_status.get(p, "?")
            sb = paper_strict_bin.get(p, "")
            lr = lenient_by_paper.get(p)
            if lr is not None:
                lb = (lr["best_bin"] + 1) if lr["best_bin"] is not None else ""
                ct = lr["claims_total"]
                cc = lr["claims_covered"]
                rr = round(lr["ratio"], 3)
            else:
                lb = ""
                ct = len(idx["paper_claims_total"].get(p, set()))
                cc = ct  # if strict-assigned, all claims covered
                rr = 1.0 if status in ("strict", "salvaged") else 0.0
            w.writerow([p, status, sb, lb, ct, cc, rr,
                        round(per_paper_total.get(p, 0) / GB, 3),
                        round(per_paper_max.get(p, 0) / GB, 3)])
    print(f"[out] paper_to_bin.csv")

    # Summary stats for the report
    n_strict = sum(1 for v in paper_status.values() if v == "strict")
    n_salvaged = sum(1 for v in paper_status.values() if v == "salvaged")
    n_leftover = sum(1 for v in paper_status.values() if v == "leftover")
    summary = {
        "cap_gb": CAP_BYTES / GB,
        "papers_total": len(all_papers),
        "papers_no_items": len(no_items),
        "papers_fittable": len(fittable),
        "papers_oversize_total": len(oversize_total),
        "papers_oversize_floor": len(oversize_floor),
        "papers_strict_assigned": n_strict,
        "papers_salvaged": n_salvaged,
        "papers_leftover": n_leftover,
        "papers_lenient_any_claim": sum(1 for r in lenient_rows if r["claims_covered"] > 0),
        "papers_lenient_50pct_or_more": sum(1 for r in lenient_rows if r["ratio"] >= 0.5),
        "papers_lenient_full": sum(1 for r in lenient_rows if r["ratio"] >= 0.999 and r["claims_total"] > 0),
        "n_bins": len(bins),
        "bins_summary": [
            {
                "bin_id": i + 1,
                "n_papers": len(b["papers"]),
                "size_gb": round(b["size_bytes"] / GB, 2),
                "n_assets": len(b["assets"]),
            }
            for i, b in enumerate(bins)
        ],
        "coverage_curve": coverage_curve,
        "size_sources": dict(src_count),
        "asset_stats": idx["stats"],
    }
    with open(OUT_SHARING / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[out] summary.json")
    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
