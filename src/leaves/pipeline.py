"""Logica di dominio della pipeline: selezione foglie sane per l'autoencoder,
campionamento bilanciato per il classificatore, codifica delle etichette.

Copiata verbatim dalle celle 21, 37 e 40 del notebook
`models/leaves_classifier.ipynb`, incapsulata in funzioni.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Costanti dal notebook.
TARGET_PLANT = "Tomato"   # cella 6
MAX_PER_CLASS = 250       # cella 37: min(250, len(group))
SEED = 42                 # cella 5


def select_healthy_training(df, plant=TARGET_PLANT):
    """Foglie SANE della specie target: set di addestramento dell'autoencoder (cella 21)."""
    return df[(df["plant"] == plant) & (df["healthy"])]


def balanced_sample(df, plant=TARGET_PLANT, max_per_class=MAX_PER_CLASS, random_state=SEED):
    """Campionamento bilanciato: min(max_per_class, len(group)) per condizione (cella 37)."""
    parts = []
    for condition, group in df[df["plant"] == plant].groupby("condition"):
        parts.append(group.sample(n=min(max_per_class, len(group)), random_state=random_state))
    return pd.concat(parts).reset_index(drop=True)


def encode_labels(conditions):
    """Codifica le condizioni in interi con LabelEncoder (cella 40).

    Ritorna (label_encoder, y) dove y sono le etichette codificate.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(conditions)
    return label_encoder, y
