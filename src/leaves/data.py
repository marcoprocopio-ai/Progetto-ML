"""Caricamento e parsing del dataset.

`parse_class` e `load_images` sono copiate dalle celle 10 e 19 del
notebook `models/leaves_classifier.ipynb`.
"""

import numpy as np
from PIL import Image

# Costante dal notebook (cella 6).
IMG_SIZE = 128


def parse_class(folder_name):
    """Dal nome cartella 'Pianta___Condizione' ricava (pianta, condizione, sana)."""
    plant, _, condition = folder_name.partition("___")
    plant = plant.split("_(")[0].replace("_", " ").strip()
    healthy = condition.lower() == "healthy"
    return plant, condition, healthy


def load_images(frame, size=IMG_SIZE):
    """Carica le immagini di un DataFrame come array (N, size, size, 3) in [0, 255]."""
    out = np.empty((len(frame), size, size, 3), dtype="float32")
    for i, path in enumerate(frame["path"].values):
        img = Image.open(path).convert("RGB").resize((size, size))
        out[i] = np.asarray(img, dtype="float32")
    return out
