"""Test del parsing dei nomi cartella e del caricamento immagini.

Nessun dataset reale: le immagini sono sintetiche, generate in `tmp_path`.
"""

import numpy as np
import pandas as pd
from PIL import Image

from leaves.data import parse_class, load_images


#  parse_class

def test_parse_class_healthy():
    assert parse_class("Tomato___healthy") == ("Tomato", "healthy", True)


def test_parse_class_malata():
    plant, condition, healthy = parse_class("Tomato___Early_blight")
    assert plant == "Tomato"
    assert condition == "Early_blight"
    assert healthy is False


def test_parse_class_pulisce_parentesi():
    # "Corn_(maize)" deve diventare la sola specie "Corn".
    plant, _, _ = parse_class("Corn_(maize)___Common_rust")
    assert plant == "Corn"


def test_parse_class_healthy_case_insensitive():
    # La condizione "Healthy" (maiuscola) deve comunque risultare sana.
    _, _, healthy = parse_class("Tomato___Healthy")
    assert healthy is True




#  load_images

def _make_image(path, size, color=(10, 20, 30)):
    """Crea un'immagine RGB solida su disco e ritorna il path."""
    Image.new("RGB", size, color).save(path)
    return str(path)


def test_load_images_forma_e_dtype(tmp_path):
    paths = [_make_image(tmp_path / f"img{i}.png", (64, 64)) for i in range(3)]
    frame = pd.DataFrame({"path": paths})

    out = load_images(frame, size=32)

    assert out.shape == (3, 32, 32, 3)
    assert out.dtype == np.float32


def test_load_images_valori_in_range_e_resize_da_non_quadrato(tmp_path):
    # Input rettangolare (40x20) -> deve uscire quadrato (size x size).
    path = _make_image(tmp_path / "rect.png", (40, 20), color=(0, 128, 255))
    frame = pd.DataFrame({"path": [path]})

    out = load_images(frame, size=16)

    assert out.shape == (1, 16, 16, 3)
    assert out.min() >= 0.0
    assert out.max() <= 255.0
