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

To enforce a RMS distance of 3.8 Å between adjacent alpha carbons:

<img width="194" alt="image" src="https://github.com/jasonchitla/eigenfold/assets/3611926/da2ebb7c-a5e3-4bf8-be0b-95d7f602313b">

Now we have a SDE and can train a score model. The score model is a graph neural net with message passing layers between all residues. The network does not just have residue coordinates but also has featurized OmegaFold embeddings.
The message passing layers are from [Tensor field networks: Rotation- and translation-equivariant neural networks](https://arxiv.org/pdf/1802.08219.pdf) so the model is invariant to 3D rotations/translations!


