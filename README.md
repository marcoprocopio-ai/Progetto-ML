# Progetto ML

Raccolta di due progetti di machine learning basati su notebook Jupyter:

- **`ML_Project.ipynb`** — Classificazione/anomaly detection su immagini del dataset
  **PlantVillage** (foglie di piante). Usa un autoencoder in Keras, con feature
  passate a modelli classici (scikit-learn, XGBoost) e tuning degli iperparametri
  tramite Optuna. I pesi del modello vengono salvati in `ckpt/`.
- **`patchcore_invecchiamento_utkface.ipynb`** — Analisi dell'invecchiamento su volti
  del dataset **UTKFace** con l'approccio **PatchCore** (feature ResNet in PyTorch +
  memory bank per l'anomaly detection).

## Struttura del progetto

```
Progetto ML/
├── README.md
├── .gitignore
├── ML_Project.ipynb                        # progetto PlantVillage (Keras)
├── patchcore_invecchiamento_utkface.ipynb  # progetto UTKFace (PyTorch/PatchCore)
├── data_utk/                               # dataset UTKFace (non versionato)
└── ckpt/                                   # checkpoint/pesi dei modelli (non versionato)
```

> `data_utk/` e `ckpt/` sono esclusi dal versionamento (vedi `.gitignore`): vanno
> ricreati/popolati in locale, non sono presenti nel repository.

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
# comuni
pip install numpy pandas matplotlib scikit-learn pillow

# per ML_Project.ipynb (PlantVillage)
pip install keras optuna xgboost scipy

# per patchcore_invecchiamento_utkface.ipynb (UTKFace)
pip install torch torchvision tqdm
```

## Dati

I dataset non sono inclusi nel repository e vanno scaricati a parte:

- **UTKFace**: scaricare l'archivio da Kaggle ed estrarlo in `data_utk/`
  (il notebook cerca ricorsivamente la cartella `UTKFace/` al suo interno).
- **PlantVillage**: posizionare l'archivio `plantvillage-dataset*.zip` nella cartella
  del notebook; verrà estratto automaticamente all'esecuzione.

## Esecuzione

Avviare Jupyter dalla cartella del progetto ed eseguire il notebook desiderato:

```bash
jupyter notebook
# oppure
jupyter lab
```

Aprire `ML_Project.ipynb` o `patchcore_invecchiamento_utkface.ipynb` ed eseguire le
celle in ordine. I percorsi ai dati (`data_utk/`, `ckpt/`) sono relativi alla cartella
del progetto: eseguire i notebook da qui per non rompere i riferimenti.
