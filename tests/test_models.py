"""Smoke test delle SOLE forme dei tensori (nessun addestramento).

Questi test proteggono l'invariante strutturale della pipeline: il decoder ha lo
spaziale del bottleneck **hardcoded a 8x8** (cella 27 del notebook), valore che
deriva da IMG_SIZE=128 ridotto da 4 MaxPooling (128 -> 8). Se IMG_SIZE cambia,
l'uscita dell'encoder non è più 8x8 e non combacia con l'ingresso del decoder:
questi test falliscono, segnalando la rottura dell'invariante.
"""

import numpy as np

from leaves.models import (
    build_encoder,
    build_decoder,
    build_autoencoder,
    IMG_SIZE,
    BOTTLENECK_CH,
)


def test_encoder_output_shape():
    x = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
    out = build_encoder()(x)
    assert tuple(out.shape) == (1, 8, 8, BOTTLENECK_CH)


def test_decoder_output_shape():
    z = np.zeros((1, 8, 8, BOTTLENECK_CH), dtype="float32")
    out = build_decoder()(z)
    assert tuple(out.shape) == (1, IMG_SIZE, IMG_SIZE, 3)


def test_autoencoder_round_trip_shape():
    # Input e output devono avere la stessa forma (ricostruzione).
    x = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
    out = build_autoencoder()(x)
    assert tuple(out.shape) == (1, IMG_SIZE, IMG_SIZE, 3)
