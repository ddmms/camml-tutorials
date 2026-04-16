# Day 5: Generative Workshop

This folder is the final workshop arc for the course. The teaching goal is not just to show modern diffusion models, but to make them feel like a natural continuation of what students already practiced earlier in the repo:

- from `01-intro/tutorial.ipynb`: inspect the data before trusting the model,
- from the neural-network notebooks: read losses and ablations critically,
- from `04-cgcnn/graph-networks.ipynb`: think in variable-size crystal graphs rather than fixed-size vectors.

The generative section now has a four-notebook ladder:

1. learn the diffusion mechanics on simple toy systems,
2. build a compact crystal diffusion model from scratch,
3. run a pretrained MatterGen workflow,
4. run a pretrained Chemeleon-DNG workflow.

Each notebook is designed as a hands-on lesson with short prediction prompts, `Task for you` checkpoints, and end-of-notebook exercises.

## Notebook Ladder

### 1. Diffusion Fundamentals
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/camml-tutorials/blob/main/notebooks/05-generative/diffusion-fundamentals.ipynb)

Notebook: [`diffusion-fundamentals.ipynb`](./diffusion-fundamentals.ipynb)

Focus:
- build intuition for stochastic dynamics, Langevin sampling, and denoising,
- compare NCSN / score-based models, DDPM, and DDIM on one toy dataset,
- separate the ideas of forward corruption, reverse prediction, and sampling,
- prepare for crystals by making the diffusion mechanics feel familiar first.

Good for:
- first contact with diffusion models,
- students who want the mathematics tied back to chemistry language,
- a short conceptual warm-up before the crystal notebooks.

Runtime:
- mostly lightweight plotting and toy training,
- roughly 10-20 minutes on CPU for a full first pass.

### 2. Crystal Diffusion From Scratch
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/camml-tutorials/blob/main/notebooks/05-generative/crystal-diffusion-from-scratch.ipynb)

Notebook: [`crystal-diffusion-from-scratch.ipynb`](./crystal-diffusion-from-scratch.ipynb)

Focus:
- build a small teaching corpus from Materials Project,
- create a flattened crystal-batch representation,
- define separate corruption processes for coordinates, lattice, and atom types,
- train an unconditional denoiser and then add lightweight conditioning,
- inspect generated crystals, trajectories, and screening diagnostics.

Good for:
- the core capstone lesson of the workshop,
- students who want to see how a crystal diffusion pipeline is assembled end to end,
- connecting diffusion fundamentals to graph-based crystal learning.

Runtime:
- `quick` mode is the recommended classroom path,
- `full` mode is better once the notebook already runs on your machine.

### 3. MatterGen In Practice
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/camml-tutorials/blob/main/notebooks/05-generative/mattergen-crystals.ipynb)

Notebook: [`mattergen-crystals.ipynb`](./mattergen-crystals.ipynb)

Focus:
- run pretrained MatterGen checkpoints,
- compare unconditional generation with a low-vs-high conditioned sweep,
- parse the returned structures into summary tables and galleries,
- inspect one recorded reverse-diffusion trajectory,
- practice the distinction between conditioning and post-generation screening.

Good for:
- students who want the cleanest bridge from the scratch notebook to a real pretrained workflow,
- classroom demos of direct crystal diffusion at larger scale,
- discussing what a candidate pool looks like before downstream relaxation or DFT.

Runtime:
- first run is install- and checkpoint-heavy,
- repeat runs are much faster after caching.

### 4. Chemeleon-DNG In Practice
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddmms/camml-tutorials/blob/main/notebooks/05-generative/chemeleon-crystals.ipynb)

Notebook: [`chemeleon-crystals.ipynb`](./chemeleon-crystals.ipynb)

Focus:
- run Chemeleon-DNG in open-ended DNG mode,
- steer DNG with different atom-count schedules,
- run CSP for explicit formula-conditioned generation,
- compare task choice, conditioning, and screening against the MatterGen workflow.

Good for:
- students who want a task-oriented crystal-generation toolkit,
- comparing open-ended generation with formula-conditioned search,
- discussing which control knobs are available before and after sampling.

Runtime:
- also install- and checkpoint-heavy on the first pass,
- later runs reuse the local environment and generated CIFs.

## Suggested Reading Order

1. [`diffusion-fundamentals.ipynb`](./diffusion-fundamentals.ipynb)
2. [`crystal-diffusion-from-scratch.ipynb`](./crystal-diffusion-from-scratch.ipynb)
3. [`mattergen-crystals.ipynb`](./mattergen-crystals.ipynb)
4. [`chemeleon-crystals.ipynb`](./chemeleon-crystals.ipynb)

If you already know diffusion models well, you can start at the scratch crystal notebook.  
If you mainly want pretrained workflows, start with MatterGen and then Chemeleon-DNG.

## External Data And Tools

- Materials Project is used for the from-scratch notebook and as context for the pretrained workflows.
- MatterGen is used as the direct pretrained crystal-diffusion example.
- Chemeleon-DNG is used as the task-oriented DNG/CSP example.

The pretrained notebooks pin upstream commits so the walkthroughs stay as reproducible as possible in a teaching setting.
