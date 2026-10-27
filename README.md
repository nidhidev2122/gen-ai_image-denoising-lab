# Image Denoising and Compression Lab

VAE (784 -> 256 -> 128 -> 16 latent -> 784) trained on MNIST, with a Streamlit UI.

## Setup (macOS)
    python3 -m venv .venv
    source .venv/bin/activate
## Setup (Windows PowerShell)
    py -m venv .venv
    .venv\Scripts\Activate.ps1

## Install, train, run
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python train_model.py
    python -m streamlit run app.py

Open http://localhost:8501. Training needs internet once (MNIST download).

## Test cases
1. Latent compression: compare 16 vs 2 latent dimensions.
2. Information loss: compare original vs reconstruction and MSE.
3. Denoising: noise 0.05 vs 0.30.
4. Reconstruction quality: record both MSE values for different settings.

Denoising does not always lower MSE; report results honestly.
