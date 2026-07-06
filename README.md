# Classificatore malattia delle foglie

Progetto di machine learning basato su notebook Jupyter:

- **`leaves_classifier.ipynb`** — Classificatore di malattie delle foglie su immagini del dataset
  **PlantVillage** (foglie di piante), preso da Kaggle. Usa un Autoencoder per una semplice Anomaly Detection,
  passa poi le feature trovate nel decoder e le passa a un modello di classsificazione XGBoost,
  che con un tuning degli iperparametri tramite Optuna, trova i più funzionali. I pesi del modello vengono salvati in `ckpt/`.

## Struttura del progetto

```
Progetto ML/
├── README.md
├── .gitignore
├── leaves_classifier.ipynb                 # progetto PlantVillage (Keras)
└── ckpt/                                   # checkpoint/pesi dei modelli (non versionato)
```

> `ckpt/` è escluso dal versionamento (vedi `.gitignore`): va
> ricreato/popolato in locale, non è presente nel repository.

## Installazione

Richiede **Python 3**. Si consiglia un ambiente virtuale:

```bash
# creazione ambiente virtuale
python -m venv .venv

# attivazione (Windows / PowerShell)
.venv\Scripts\Activate.ps1

# attivazione (Linux / macOS)
source .venv/bin/activate
```

Dipendenze principali:

```bash

# per leaves_classifier.ipynb 
pip  install numpy pandas matplotlib scikit-learn pillow keras optuna xgboost scipy

```

## Dati

Il dataset non è incluso nel repository e va scaricato a parte:

- **PlantVillage**: posizionare l'archivio `plantvillage-dataset*.zip` nella cartella
  del notebook; verrà estratto automaticamente all'esecuzione.

## Esecuzione

Avviare Jupyter dalla cartella del progetto ed eseguire il notebook:

```bash
jupyter notebook
# oppure
jupyter lab
```

Aprire `leaves_classifier.ipynb` ed eseguire le
celle in ordine. I percorsi ai dati (`ckpt/`) sono relativi alla cartella
del progetto: eseguire il notebook da qui per non rompere i riferimenti.
