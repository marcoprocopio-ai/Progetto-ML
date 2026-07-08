# Classificatore malattie delle foglie — PlantVillage

Progetto di Machine Learning: rilevamento delle malattie nelle foglie di
**pomodoro** sul dataset **PlantVillage** preso da Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
 
Tramite una pipeline suddivisa in due parti nel
notebook **`models/leaves_classifier.ipynb`** con motore Keras con backend torch [docs/decisioni/0001](docs/decisioni/0001-keras-torch.md):

1. **Autoencoder convoluzionale** addestrato **solo su foglie sane**
   di pomodoro. Il suo **encoder** produce una rappresentazione compatta (bottleneck
   `8×8×32`).
2. **Classificatore XGBoost** che, a partire dalle feature dell'encoder,
   assegna una tra **10 condizioni** del pomodoro (9 malattie + `healthy`). Gli
   iperparametri sono identificati con Optuna. I pesi dell'autoencoder sono salvati in `ckpt/`.

> **Nota.** L'obiettivo iniziale prevedeva anche il rilevamento delle foglie malate
> tramite errore di ricostruzione dell'autoencoder: nel codice attuale **questa parte non
> è implementata** e l'encoder è usato solo come estrattore di feature. Dettagli in
> [docs/decisioni/0007](docs/decisioni/0007-encoder-feature-extractor.md).

E' in oltre importante specificare che **XGBoost** utilizza dati tabulari. Il dataset PlantVillage è composto da sole immagini, motivo per cui non è stato possibile applicare il solo modello di classificazione sin dall'inizio. E l'architettura utilizzata vedi [docs/architettura.md](docs/architettura.md), risolve questo limite utilizzando feature tabulari estratte direttamente dall'autoencoder.

## Documentazione

- [docs/architettura.md](docs/architettura.md) — Pipeline completa e diagramma del flusso.
- [docs/esperimenti.md](docs/esperimenti.md) — Risultati finali
- [docs/decisioni/](docs/decisioni/) — ADR sulle scelte progettuali principali.

## Struttura del progetto

```
Progetto ML/
├── README.md
├── .gitignore
├── requirements.txt                        # dipendenze runtime (notebook), versioni pinnate
├── docs/                                   # documentazione (architettura, esperimenti, ADR)
├── models/
│   └── leaves_classifier.ipynb             # notebook del modello
├── plantvillage/                           # dataset estratto (non versionato)
└── ckpt/                                   # pesi dell'autoencoder (non versionato)
```

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

Installare le dipendenze runtime (versioni pinnate in `requirements.txt`):

```bash
pip install -r requirements.txt
```

Il progetto usa Keras con backend **torch**: impostare la variabile d'ambiente
`KERAS_BACKEND=torch` prima di avviare Jupyter (PowerShell: `$env:KERAS_BACKEND="torch"`;
Linux/macOS: `export KERAS_BACKEND=torch`).

## Dati

Il dataset PlantVillage (~2,2 GB) **non è incluso nel repository** — è troppo grande per
GitHub — e va scaricato **una sola volta** da Kaggle:
[`abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

Non serve rinominare né spostare nulla a mano: basta lasciare lo zip scaricato in una di
queste posizioni e il notebook lo trova ed estrae in automatico (cella di caricamento dati):

- la cartella da cui si avvia il notebook, oppure la root del progetto;
- la cartella `~/Downloads`.

Sono riconosciuti i nomi più comuni dello zip (es. `archive*.zip`, `plantvillage*.zip`).
Alla prima esecuzione viene estratto in `plantvillage/`; nelle esecuzioni successive, se la
cartella esiste già, l'estrazione viene saltata. Il notebook usa il sottoinsieme `segmented`[docs/decisioni/0004](docs/decisioni/0004-sottoinsieme-segmented.md).

## Esecuzione

Avviare Jupyter ed eseguire il notebook:

```bash
jupyter notebook
# oppure
jupyter lab
```

Aprire `models/leaves_classifier.ipynb` ed eseguire le celle in ordine.

I percorsi ai dati e ai checkpoint (`plantvillage/`, `ckpt/`) sono **relativi alla cartella di
esecuzione**: avviare Jupyter dalla cartella del progetto per non rompere i riferimenti.
