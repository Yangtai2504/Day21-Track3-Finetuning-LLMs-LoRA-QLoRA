# Lab 21 — Submission Links

**Học viên**: Nguyễn Thái Dương — AI20K-009

## GitHub Repository
https://github.com/Yangtai2504/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA

## HuggingFace Hub — LoRA Adapter (r=16)
https://huggingface.co/Dwg2504/qwen2.5-3b-vi-lab21-r16

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (200 samples)
- **Config**: r=16, alpha=32, target_modules=["q_proj","v_proj"], QLoRA 4-bit
- **Eval perplexity**: 4.55
