"""Streamlit app: Image Denoising and Compression Lab (VAE on MNIST)."""
import os

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from train_model import MODEL_PATH, VAE

st.set_page_config(page_title="Image Denoising & Compression Lab")
st.title("Image Denoising and Compression Lab")
st.caption("Variational Autoencoder trained on MNIST handwritten digits.")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("vae_model.pth not found. Run `python train_model.py` first.")
        st.stop()
    model = VAE()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def preprocess(img):
    """Grayscale, crop/resize to 28x28, scale to 0-1, invert if background is light."""
    img = ImageOps.fit(img.convert("L"), (28, 28), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    inverted = False
    if arr.mean() > 0.5:  # MNIST digits are white on black
        arr = 1.0 - arr
        inverted = True
    return arr, inverted


def show(arr, caption):
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    st.image(img.resize((196, 196), Image.NEAREST), caption=caption)


model = load_model()

# ---- Controls ----
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
noise_level = st.sidebar.slider("Gaussian noise level", 0.0, 0.5, 0.2, 0.01)
latent_dims = st.sidebar.selectbox("Latent dimensions kept", [2, 4, 8, 16], index=3)

if uploaded is not None:
    source = Image.open(uploaded)
else:
    sample = st.sidebar.selectbox("...or pick a sample digit", list(range(10)))
    path = f"samples/digit_{sample}.png"
    if not os.path.exists(path):
        st.info("Upload an image to begin (sample digits are created by train_model.py).")
        st.stop()
    source = Image.open(path)

original, inverted = preprocess(source)

# Fixed seed so the noisy image changes smoothly with the slider
rng = np.random.default_rng(42)
noisy = np.clip(original + rng.normal(0, noise_level, original.shape), 0, 1).astype(np.float32)

# ---- Encode -> keep k latent dims -> decode ----
with torch.no_grad():
    x = torch.from_numpy(noisy).view(1, 784)
    mu, _ = model.encode(x)
    z = model.mask(mu, latent_dims)
    recon = model.decode(z).view(28, 28).numpy()

mse_noisy = float(np.mean((noisy - original) ** 2))
mse_recon = float(np.mean((recon - original) ** 2))

# ---- Results ----
c1, c2, c3 = st.columns(3)
with c1:
    show(original, "Preprocessed original (28x28)")
with c2:
    show(noisy, f"Noisy (sigma = {noise_level:.2f})")
with c3:
    show(recon, "VAE reconstruction")
if inverted:
    st.caption("Light background detected: image was inverted to match MNIST style.")

st.subheader("Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Latent dimensions kept", f"{latent_dims} of 16")
m2.metric("Compression ratio", f"{784 / latent_dims:.1f} : 1")
m3.metric("Values: 784 -> latent", f"{latent_dims}")
m4, m5 = st.columns(2)
m4.metric("Noisy-input MSE", f"{mse_noisy:.4f}")
m5.metric("Reconstruction MSE", f"{mse_recon:.4f}")

if mse_recon < mse_noisy:
    st.success("Reconstruction reduced the error compared with the noisy input.")
else:
    st.warning("Reconstruction did NOT reduce the error compared with the noisy input.")

st.caption("Compression ratio = 784 / retained latent dimensions (value count only, "
           "not file size). Unused latent dimensions are set to zero before decoding. "
           "MSE is measured against the preprocessed original.")
