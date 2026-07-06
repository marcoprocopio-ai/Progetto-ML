"""Invarianti di dominio della pipeline (nessun modello, nessun dataset reale).

Il DataFrame è sintetico e ricalca lo schema prodotto dal notebook:
colonne `plant`, `condition`, `healthy`, `path`.
"""

import pandas as pd

from leaves.pipeline import (
    select_healthy_training,
    balanced_sample,
    encode_labels,
    MAX_PER_CLASS,
)


def _synthetic_df():
    """Costruisce un df bilanciabile: Tomato con più condizioni, oltre a rumore
    di un'altra specie che non deve finire nei campioni."""
    rows = []
    # Tomato healthy: 300 esempi (> MAX_PER_CLASS per testare il cap).
    rows += [{"plant": "Tomato", "condition": "healthy", "healthy": True,
              "path": f"h{i}"} for i in range(300)]
    # Tomato malate: due condizioni, una sopra e una sotto il cap.
    rows += [{"plant": "Tomato", "condition": "Early_blight", "healthy": False,
              "path": f"e{i}"} for i in range(280)]
    rows += [{"plant": "Tomato", "condition": "Leaf_Mold", "healthy": False,
              "path": f"l{i}"} for i in range(50)]
    # Altra specie: non deve mai comparire nei risultati (filtro su plant).
    rows += [{"plant": "Corn", "condition": "Common_rust", "healthy": False,
              "path": f"c{i}"} for i in range(100)]
    return pd.DataFrame(rows)


def test_training_autoencoder_zero_foglie_malate():
    # Presupposto dell'approccio non supervisionato: si allena SOLO sulle sane.
    df = _synthetic_df()
    train = select_healthy_training(df, plant="Tomato")

    assert len(train) > 0
    assert train["healthy"].all()
    assert not train["healthy"].eq(False).any()
    # Solo la specie target.
    assert (train["plant"] == "Tomato").all()


def test_campionamento_non_supera_il_cap_per_classe():
    df = _synthetic_df()
    sample = balanced_sample(df, plant="Tomato", max_per_class=MAX_PER_CLASS)

    per_class = sample["condition"].value_counts()
    assert (per_class <= MAX_PER_CLASS).all()
    # La classe con 300 esempi viene tagliata esattamente a MAX_PER_CLASS.
    assert per_class["healthy"] == MAX_PER_CLASS
    # La classe con 50 esempi resta intera (min(cap, len)).
    assert per_class["Leaf_Mold"] == 50
    # Nessun esempio di altre specie.
    assert "Common_rust" not in per_class.index


def test_label_encoder_round_trip():
    conditions = ["Early_blight", "healthy", "Leaf_Mold", "healthy", "Early_blight"]
    label_encoder, y = encode_labels(conditions)

    # encode + inverse_transform riporta alle etichette originali.
    restored = label_encoder.inverse_transform(y)
    assert list(restored) == conditions
