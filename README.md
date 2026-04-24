# CSE425 Neural Network Project

This repository contains the final project notebook `CSE425_FINAL.ipynb` for a neural music generation and evaluation pipeline. The project implements four main tasks using MIDI sequence modeling, generative autoencoders, a transformer-based sequence model, and reinforcement learning with human preference tuning.

## Project Overview

The notebook is structured around a series of experiments that build on the Maestro dataset and genre-specific MIDI data:

- **Task 1:** LSTM autoencoder for piano-roll MIDI sequences.
- **Task 2:** Music VAE with genre-based training and music quality metrics.
- **Task 3:** Transformer-based token sequence model for genre-conditioned music generation.
- **Task 4:** Reinforcement Learning from Human Feedback (RLHF) to tune genre-conditioned music generation.
- **Baseline comparison:** Random note generator and Markov chain music model evaluation.

## Data

The notebook expects MIDI data in the following structure, as used in the code:

- `maestro-v3.0.0/` — Maestro MIDI dataset.
- `task2_data/classical`, `task2_data/jazz`, `task2_data/pop`, `task2_data/rock` — genre-specific MIDI folders for Task 2.

In the notebook, the dataset is downloaded from the Maestro dataset URL and prepared with piano-roll conversion.

## Task Details

### Task 1: LSTM Autoencoder

- Loads and preprocesses MIDI to piano-roll representations.
- Builds an `LSTMAutoencoder` model with encoder/decoder LSTMs.
- Trains the model on reconstructed piano-roll sequences.
- Saves the best model and training loss curves.
- Generates sample MIDI reconstructions and new sequences from latent vectors.
- Evaluates generated samples using:
  - Pitch histogram distance
  - Rhythm diversity
  - Repetition ratio

### Task 2: Music VAE

- Prepares genre-labeled MIDI sequences from classical, jazz, pop, and rock folders.
- Uses shorter piano-roll sequence windows and genre-specific preprocessing.
- Trains a variational autoencoder to learn a latent music representation.
- Tracks reconstruction loss and KL divergence.
- Records genre-aware evaluation metrics for generated music.

### Task 3: Transformer Sequence Model

- Tokenizes musical events into a symbolic sequence representation.
- Builds a transformer-based autoregressive model for sequence generation.
- Trains with cross-entropy loss and reports perplexity.
- Saves the trained transformer and generated MIDI samples.

### Task 4: Reinforcement Learning Human Preference Tuning

- Uses the Task 3 transformer as the base model.
- Creates a policy model and a frozen reference model.
- Generates pre-tuning and post-tuning music examples.
- Collects human preference data through survey templates.
- Tunes the model using human reward signals and saves RL training artifacts.

## Baseline Models

The notebook also includes comparison against at least two baselines:

- Random note generator using piano-roll random activations.
- Markov chain music model with token transition probability sampling.

These baselines are used to compare against the main Task 1–Task 4 models on quantitative and qualitative metrics.

## Outputs

The notebook saves result artifacts under output directories such as:

- `/content/outputs_task1_fixed`
- `/content/outputs_task2_vae`
- `/content/outputs_task3_transformer`
- `/content/outputs_task4_rlhf`
- `/content/outputs_baselines`

Common saved files include:

- model checkpoints (`*.pt`)
- training history CSVs
- metrics CSVs
- generated MIDI files
- plots of loss and evaluation metrics

## Requirements

The notebook installs dependencies at the top of the file including:

- `pretty_midi`
- `music21`
- `numpy`
- `matplotlib`
- `tqdm`
- `pandas`
- `torch`
- `torchvision`
- `torchaudio`

The notebook is designed to run in a Colab-style environment, so paths like `/content/` are used for data and outputs.

## How to Use

1. Open `CSE425_FINAL.ipynb` in Jupyter or Colab.
2. Run the notebook cells sequentially.
3. Ensure data is downloaded and the expected output directories are available.
4. Review saved metrics, generated MIDI files, and plots in the output folders.

## Notes

- The notebook includes Google Colab-specific commands such as `!wget`, `!unzip`, `from google.colab import drive`, and `files.download`.
- Some paths are hard-coded to `/content/` and may need adjustment for local execution.
- The project uses a mixture of piano-roll reconstruction, latent-space sampling, transformer generation, and reward shaping with human feedback.
