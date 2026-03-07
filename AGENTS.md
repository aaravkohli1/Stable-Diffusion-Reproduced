# Repository Guidelines

## Project Structure & Module Organization
`data/` holds dataset loading and preprocessing utilities (`DiffusionDataset`, image/text transforms).  
`diffusion/` contains core diffusion logic: beta schedules, forward noising (`Diffuser`), and DDPM sampling helpers.  
`models/clip/` contains the CLIP text encoder reimplementation plus a Hugging Face parity script (`hf_clip_text.py`).  
`models/unet/` and `models/vae/` are currently stubs/placeholders; `models/vae/VAE.ipynb` is exploratory.  
`tests/` currently contains smoke coverage (`test_clip_smoke.py`).  
`configs/*.yaml` files exist but are empty placeholders at this stage.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local virtual environment.
- `pip install -r requirements.txt`: install base dependencies.
- `pip install transformers`: required for CLIP Hugging Face loading/parity checks.
- `pytest -q`: run all tests.
- `pytest tests/test_clip_smoke.py -q`: run only the CLIP smoke test.
- `python -m models.clip.hf_clip_text`: compare local CLIP outputs against Hugging Face reference outputs.

Some tasks download remote artifacts (CIFAR-10, HF weights), so network access may be required.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and PEP 8-aligned formatting.  
Use `snake_case` for modules/functions/variables, `CamelCase` for classes, and `UPPER_SNAKE_CASE` for constants.  
Match current patterns: module docstrings, type hints on public methods, and minimal comments only for non-obvious logic.  
Prefer importing from package exports (for example `from diffusion import Diffuser`) when available.

## Testing Guidelines
Use `pytest`; test files should be named `test_*.py` and test functions `test_*`.  
Follow the existing smoke-test style by asserting key tensor contracts (rank, batch size, sequence length).  
For new code, add at least one fast smoke test and one behavior test.  
No enforced coverage threshold exists yet; prioritize core math paths and tensor/device correctness.

## Commit & Pull Request Guidelines
Git history favors short, imperative commit messages (for example: `Complete DDPM sampler`, `implemented architecture`). Keep this style and scope each commit to one logical change.  
PRs should include: purpose, main files changed, test commands executed, and any data/model download assumptions.  
For model behavior changes, include concise evidence such as shape checks or reference diff stats.
