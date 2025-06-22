import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Modelo igual al entrenado
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, z, label):
        onehot = torch.nn.functional.one_hot(label, 10).float()
        z_cat = torch.cat([z, onehot], dim=1)
        return self.decoder(z_cat)

# Cargar modelo
device = torch.device("cpu")
model = Autoencoder()
model.load_state_dict(torch.load("autoencoder_generator.pth", map_location=device))
model.eval()

# Interfaz Streamlit
st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit (0-9):", list(range(10)))
if st.button("Generate Images"):
    images = []
    for _ in range(5):
        z = torch.randn(1, 64)  # vector latente aleatorio
        label = torch.tensor([digit])
        img = model(z, label).detach().numpy()[0, 0]
        images.append(img)

    st.markdown(f"### Generated images of digit {digit}")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(images[i], width=100, caption=f"Sample {i+1}")
