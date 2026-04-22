# M4 Lab: The Diagnostician

**PUBH 5106 — AI in Health Applications**

A competitive lab where student teams compare three approaches to clinical reasoning under uncertainty: human intuition, a Bayesian network, and an LLM. Scored on **calibration** — not who is most confident, but who is most *accurately* confident.

## Quick Start — Choose Your Platform

### Option A: GitHub Codespaces (recommended)

1. Click the green **Code** button → **Create codespace on main**
2. Wait for setup (~5 min — installs Julia + packages)
3. Open `M4_lab.ipynb`

### Option B: Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/utsw-hdsb/pubh5106-m4-lab/main?urlpath=lab%2Ftree%2FM4_lab.ipynb)

Click the badge above. First build takes ~5-10 min (Julia package compilation). Subsequent launches are faster if the image is cached.

### Option C: Run Locally

See [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) for step-by-step instructions (Mac and Windows). No separate Python installation needed — Julia manages everything.

### Then

1. Set your group name and Groq API keys in the Setup section
2. Run cells in order — 3 scored rounds + reflection

## Rounds

| Round | Name | What You Do |
|-------|------|-------------|
| — | Warm-Up | Explore the Aronsky & Haug pneumonia network |
| 1 | Base Rate Trap | Estimate probabilities by intuition |
| 2 | Build the Network | Construct the ASIA Bayesian network |
| 3 | Head to Head | BN vs. LLM vs. you on neonatal diagnosis + stress test |

## File Layout

```
.
├── M4_lab.ipynb           # Lab notebook (Codespaces / Binder / Local)
├── M4_lab.jl              # Notebook source (jupytext)
├── lab_utils.jl           # Shared utilities (BN, Groq API, scoring)
├── Project.toml           # Julia dependencies
├── LOCAL_SETUP_GUIDE.md   # Local installation instructions
├── README.md              # This file
├── .devcontainer/
│   └── devcontainer.json  # Codespaces configuration
└── data/
    ├── vignettes.json         # Clinical cases
    ├── llm_precomputed.json   # Pre-computed LLM responses (fallback)
    ├── asia_network.svg       # ASIA network DAG (pre-rendered)
    └── child_network.svg      # CHILD network DAG (pre-rendered)
```

## Submission

Email your completed notebook to **brian.chapman@utsouthwestern.edu**.
