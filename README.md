# eigenfold
Implementing Eigenfold paper (https://arxiv.org/pdf/2304.02198.pdf)

## Overview of training process
1. Download PDB data
2. Process PDB data
3. Generate OmegaFold Embeddings on processed pdb data
4. Train on those embeddings

## The Model (Harmonic Diffusion)
Structure graph G = (V, E) where G represents a protein with a specific sequence, V is the set of residues and E the edges connecting neighboring residues.
The model learns G-dependent probability distributions under a forward diffusion process:

dx = −1/2Hxdt + dw

x are the coordinates of the alpha carbons and H is chosen such that undesired, chemically implausible structures have high energy E(x):

<img width="268" alt="image" src="https://github.com/jasonchitla/eigenfold/assets/3611926/bbf836e5-6333-43e1-9b5d-aa40ca7baf21">
