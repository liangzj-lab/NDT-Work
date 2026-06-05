# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NDT (Non-Destructive Testing) data processing project, focused on industrial radiographic testing (RT, 射线检测) imaging. Uses DICOM-format (`.dcm`) image data with PyTorch for planned ML/deep learning work.

## Working Directory

All development work is done under `ZJDKY/`. Always `cd ZJDKY` before running commands.

## Commands

```bash
# Environment check
python hello.py

# Batch rename .dcn to .dcm (edit input_dir in main() first)
python data_processing/rename_dcn_to_dcm.py
```

No build system, package manager, test runner, or linter is configured yet. Python and PyTorch are the only runtime dependencies detected.

## Directory Convention

| Directory | Purpose |
|-----------|---------|
| `ZJDKY/DL/` | Deep learning related code (models, training, inference) |
| `ZJDKY/data_processing/` | Data processing utilities (cleaning, format conversion, etc.) |
| `ZJDKY/html/` | Test HTML pages and frontend experiments |

## Architecture

- `ZJDKY/hello.py` — PyTorch environment sanity check
- `ZJDKY/data_processing/rename_dcn_to_dcm.py` — Utility: recursively renames `.dcn` → `.dcm` under a given directory tree. Uses `pathlib.Path.rglob()`. Edit the hardcoded `input_dir` in `main()` before running.
