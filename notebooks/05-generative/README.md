# Diffusion Model Tutorial

A notebook-first tutorial repo for learning diffusion models and then applying them to crystal generation.

The teaching ladder is:

1. start with diffusion intuition and the main diffusion-model families,
2. build a small crystal diffusion model from scratch,
3. compare that workflow against modern crystal-generation toolkits.

Each notebook is runnable on its own, but they are designed to work best in sequence.

## What This Repo Covers

- diffusion intuition from stochastic motion, Langevin dynamics, and denoising,
- model families including DDPM, score-based models / NCSN, and DDIM,
- unconditional and conditional generation,
- crystal-specific representations, corruption processes, and sampling,
- practical SOTA workflows with MatterGen and Chemeleon-DNG.

The notebooks include:

- tables of contents,
- questions and exercises throughout,
- hidden answers,
- collapsed implementation-heavy cells where possible,
- Colab-friendly setup cells and badges.

## Notebook Ladder

### 1. Diffusion Fundamentals
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/camml-tutorials/blob/main/notebooks/05-generative/main/diffusion-fundamentals.ipynb)

Notebook: [`diffusion-fundamentals.ipynb`](./diffusion-fundamentals.ipynb)

Focus:
- build intuition for noisy dynamics and denoising,
- understand what DDPM, NCSN / score-based models, and DDIM actually learn,
- compare these model families on the same toy dataset,
- bridge from generic diffusion ideas to materials and crystals.

Good for:
- first contact with diffusion models,
- teaching or self-study,
- readers who want both intuition and runnable examples.

Runtime:
- fast plotting sections at the start,
- roughly 10-20 minutes on CPU for a fuller first pass through the training demos.

### 2. Crystal Diffusion From Scratch
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/camml-tutorials/blob/main/notebooks/05-generative/main/crystal-diffusion-from-scratch.ipynb)

Notebook: [`crystal-diffusion-from-scratch.ipynb`](./crystal-diffusion-from-scratch.ipynb)

Focus:
- pull a small crystal dataset from Materials Project,
- build a compact MatterGen-like representation,
- define forward corruption for coordinates, lattice, and atom types,
- train an unconditional model,
- add conditional control for density, band gap, composition, and space group,
- inspect generated crystals, trajectories, and lightweight validity checks.

Good for:
- readers who want to see how a crystal diffusion pipeline is assembled,
- small-scale experimentation,
- understanding the jump from toy diffusion examples to periodic materials.

Runtime:
- `quick` mode is the best first pass and is designed to stay notebook-scale,
- `full` mode is heavier and better once the workflow already runs on your machine.

Notes:
- if an MP API key is available, the notebook can build a small live dataset from Materials Project,
- otherwise it falls back to a bundled small real-MP teaching dataset.

### 3. Crystal Generation SOTA Comparison
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/camml-tutorials/blob/main/notebooks/05-generative/main/sota-crystals-comparison.ipynb)

Notebook: [`sota-crystals-comparison.ipynb`](./sota-crystals-comparison.ipynb)

Focus:
- run pretrained MatterGen workflows,
- compare unconditional and property-conditioned generation,
- run Chemeleon-DNG in DNG and CSP modes,
- inspect distributions, galleries, trajectories, and screening outputs,
- compare what each toolkit is good at in practice.

Good for:
- readers who want practical modern crystal-generation demos,
- users choosing between direct crystal diffusion and more task-oriented toolkits,
- classroom demos of pretrained systems.

Runtime:
- the heaviest notebook in the repo,
- first run is install- and download-heavy because it builds external environments and fetches checkpoints,
- repeat runs are much faster once caches exist.

## Suggested Reading Order

1. [`diffusion-fundamentals.ipynb`](./diffusion-fundamentals.ipynb)
2. [`crystal-diffusion-from-scratch.ipynb`](./crystal-diffusion-from-scratch.ipynb)
3. [`sota-crystals-comparison.ipynb`](./sota-crystals-comparison.ipynb)

If you already know diffusion models well, you can jump straight to the crystal notebook.  
If you mainly want pretrained crystal-generation workflows, you can go directly to the SOTA notebook.

## Data And External Tools

- Materials Project is used in the crystal notebook for live dataset construction.
- MatterGen is used in the SOTA notebook as the direct crystal-diffusion example.
- Chemeleon-DNG is used in the SOTA notebook as the second modern crystal-generation system.

The SOTA notebook pins the external model repos by commit so the workflow is more reproducible.

## Repo Goal

This repo is meant to be a compact teaching path, not a full benchmark suite or production package.

The emphasis is on:

- clarity,
- runnable examples,
- conceptual continuity across notebooks,
- enough realism to make the crystal-generation workflow feel authentic.
