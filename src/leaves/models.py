"""Costruzione dei modelli (encoder / decoder / autoencoder).

I corpi di `build_encoder`, `build_decoder` e `build_autoencoder` riproducono
verbatim le celle 26, 27 e 28 del notebook `models/leaves_classifier.ipynb`.
Le uniche differenze rispetto al notebook sono l'incapsulamento in funzioni e la
parametrizzazione su `img_size` / `bottleneck_ch` (con default identici alle
costanti del notebook). I layer, i filtri e le attivazioni sono invariati.

Nota: il decoder usa `Input(shape=(8, 8, ...))` con lo spaziale **hardcoded a 8**,
esattamente come nella cella 27. Questo valore deriva da IMG_SIZE=128 ridotto da
4 MaxPooling (128 -> 8): cambiare `img_size` rompe l'accoppiamento encoder/decoder.
"""

import os

# Il notebook imposta il backend torch (cella 2) prima di importare keras.
os.environ.setdefault("KERAS_BACKEND", "torch")

from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Rescaling
from keras.activations import leaky_relu

# Costanti dal notebook (cella 6).
IMG_SIZE = 128
BOTTLENECK_CH = 32


def build_encoder(img_size=IMG_SIZE, bottleneck_ch=BOTTLENECK_CH):
    """Encoder convoluzionale: (img_size, img_size, 3) -> (8, 8, bottleneck_ch)."""
    return Sequential(
        [
            Input(shape=(img_size, img_size, 3)),
            Rescaling(1 / 255),
            Conv2D(32, 3, padding="same", activation=leaky_relu),  MaxPooling2D(),
            Conv2D(64, 3, padding="same", activation=leaky_relu),  MaxPooling2D(),
            Conv2D(128, 3, padding="same", activation=leaky_relu), MaxPooling2D(),
            Conv2D(128, 3, padding="same", activation=leaky_relu), MaxPooling2D(),
            Conv2D(bottleneck_ch, 3, padding="same", activation=leaky_relu),
        ],
        name="encoder",
    )


def build_decoder(bottleneck_ch=BOTTLENECK_CH):
    """Decoder convoluzionale: (8, 8, bottleneck_ch) -> (128, 128, 3)."""
    return Sequential(
        [
            Input(shape=(8, 8, bottleneck_ch)),
            Conv2D(128, 3, padding="same", activation=leaky_relu), UpSampling2D(),
            Conv2D(128, 3, padding="same", activation=leaky_relu), UpSampling2D(),
            Conv2D(64, 3, padding="same", activation=leaky_relu),  UpSampling2D(),
            Conv2D(32, 3, padding="same", activation=leaky_relu),  UpSampling2D(),
            Conv2D(3, 3, padding="same", activation=leaky_relu),
            Rescaling(255.0),
        ],
        name="decoder",
    )


def build_autoencoder(img_size=IMG_SIZE, bottleneck_ch=BOTTLENECK_CH):
    """Autoencoder = encoder + decoder in cascata (cella 28)."""
    encoder = build_encoder(img_size, bottleneck_ch)
    decoder = build_decoder(bottleneck_ch)
    return Sequential([encoder, decoder])
