# Image Denoising and Compression Lab

**Live app:** https://gen-aiimage-denoising-lab-nrnxbdcswoq2fpmaspzvyi.streamlit.app/

A Variational Autoencoder (VAE) trained on MNIST handwritten digits, with a Streamlit interface.
Images pass through a real encoder and decoder: 784 pixels -> 256 -> 128 -> 16 latent values -> decoder -> 784 pixels.

## Features
- Controls: image upload, Gaussian noise slider (0 to 0.5), latent dimensions (2, 4, 8, 16), sample digit picker
- Output: original, noisy and reconstructed images side by side
- Metrics: compression ratio (784 / latent dimensions), noisy-input MSE, reconstruction MSE

## How it works
1. The image is converted to 28x28 grayscale.
2. Gaussian noise is added (set by the slider).
3. The encoder compresses the image to 16 latent values.
4. Only the selected number of latent dimensions is kept; the rest are set to zero.
5. The decoder rebuilds the image, and MSE is measured against the original.

The compression ratio counts values only (784 / latent dimensions). It is not a file-size ratio.

## Files
- `train_model.py` - trains the VAE on MNIST and saves `vae_model.pth`
- `app.py` - Streamlit application
- `vae_model.pth` - trained weights (needed by the deployed app)
- `samples/` - sample test digits (needed by the deployed app)
- `training_results.png` - original vs noisy vs reconstructed digits
- `requirements.txt` - dependencies

## Run locally
macOS:

    python3 -m venv .venv
    source .venv/bin/activate

Windows PowerShell:

    py -m venv .venv
    .venv\Scripts\Activate.ps1

Then:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python train_model.py
    python -m streamlit run app.py

Open http://localhost:8501. Training needs internet once to download MNIST.

## Test cases
1. Latent compression: compare 16 vs 2 latent dimensions.
2. Information loss: compare original vs reconstruction and MSE (noise 0).
3. Denoising: compare noise 0.05 vs 0.30.
4. Reconstruction quality: record both MSE values for different settings.

Denoising does not always lower MSE, so results should be reported honestly. The model is trained on MNIST, so it works best on handwritten digits.
