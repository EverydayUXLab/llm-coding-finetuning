# LLM Coding Fine-Tuning (QLoRA)

This project demonstrates how to fine-tune a **code-focused LLM** using **QLoRA** on a custom instruction dataset.

## Features
- Fine-tunes `DeepSeek-Coder-6.7B`
- Uses LoRA adapters (low-cost training)
- Instruction-based coding tasks
- GPU-friendly (4-bit quantization)

## Project Structure
```ascii
llm-training/
├── train.py
├── test.py
├── train.jsonl
├── .gitignore
└── README.md
```

## Requirements
- Python 3.10+
- NVIDIA GPU (24GB recommended)
- CUDA installed

## Training
```bash
source venv/Scripts/activate
accelerate launch train.py
```

## Notes
- Base model weights are downloaded at runtime
- Trained adapters are not committed to the repo

---

## STEP 4 — Check What Will Be Committed

Before committing, **always check**:

```bash
git status
```