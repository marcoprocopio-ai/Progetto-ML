# Classificatore malattie delle foglie — PlantVillage

Progetto d'esame di Machine Learning: rilevamento delle malattie nelle foglie di
**pomodoro** sul dataset **PlantVillage** preso da Kaggle.
 
Tramite una pipeline suddivisa in due parti nel
notebook **`models/leaves_classifier.ipynb`** con motore Keras con backend torch:

1. **Autoencoder convoluzionale** addestrato **solo su foglie sane**
   di pomodoro. Il suo **encoder** produce una rappresentazione compatta (bottleneck
   `8×8×32`).
2. **Classificatore XGBoost** che, a partire dalle feature dell'encoder,
   assegna una tra **10 condizioni** del pomodoro (9 malattie + `healthy`). Gli
   iperparametri sono identificati con Optuna. I pesi dell'autoencoder sono salvati in `ckpt/`.

> **Nota.** L'obiettivo iniziale prevedeva anche il rilevamento delle foglie malate
> tramite errore di ricostruzione dell'autoencoder: nel codice attuale **questa parte non
> è implementata** e l'encoder è usato solo come estrattore di feature. Dettagli in
> [docs/decisioni/0004](docs/decisioni/0004-encoder-feature-extractor.md).

## Documentazione

- [docs/architettura.md](docs/architettura.md) — pipeline completa e diagramma del flusso.
- [docs/esperimenti.md](docs/esperimenti.md) — risultati reali (accuratezza test,
  F1 per classe, limiti).
- [docs/test.md](docs/test.md) — strategia di test (cosa si testa e perché).
- [docs/decisioni/](docs/decisioni/) — ADR sulle scelte progettuali principali.

## Struttura del progetto

```
Progetto ML/
├── README.md
├── .gitignore
├── pyproject.toml                          # configurazione pytest + coverage
├── docs/                                   # documentazione
├── models/
│   └── leaves_classifier.ipynb             # notebook modello
│  
├── src/leaves/                             # logica deterministica estratta dal notebook
│   ├── data.py                             # `parse_class`, `load_images`
│   ├── models.py                           # build_encoder/decoder/autoencoder
│   └── pipeline.py                         # filtro sane, campionamento, LabelEncoder
├── tests/                                  # suite pytest (`test_data`, `test_model`, `test_pipeline`)
├── leaves_tests.ipynb                      # notebook dimostrativo: esegue la suite inline
├── plantvillage/                           # dataset estratto (non versionato)
└── ckpt/                                   # pesi dell'autoencoder (non versionato)
```

> La logica in `src/leaves/` è copiata dal notebook per renderla
> testabile: vedi [docs/decisioni/0006](docs/decisioni/0006-logica-estratta-src-testabilita.md).

> `plantvillage/`, `ckpt/` e gli archivi `*.zip` sono esclusi dal versionamento
> (vedi `.gitignore`) e vanno ricreati/popolati in locale.

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
pip install numpy pandas matplotlib scikit-learn pillow scipy keras torch optuna xgboost
```

## Dati

Il dataset non è incluso nel repository e va scaricato a parte:

- **PlantVillage**: scaricare l'archivio da Kaggle (`abdallahalidev/plantvillage-dataset`)
  e posizionare lo zip `plantvillage-dataset*.zip` nella cartella da cui si esegue il
  notebook; verrà estratto automaticamente in `plantvillage/`. Il notebook usa il
  sottoinsieme `segmented`.

## Esecuzione

Avviare Jupyter ed eseguire il notebook:

```bash
jupyter notebook
# oppure
jupyter lab
```

Aprire `models/leaves_classifier.ipynb` ed eseguire le celle in ordine. I percorsi ai dati
e ai checkpoint (`plantvillage/`, `ckpt/`) sono **relativi alla cartella di esecuzione**:
per non rompere i riferimenti, avviare Jupyter dalla stessa cartella in cui si trovano lo
zip del dataset e la cartella `ckpt/`.

## Come lanciare i test

I test verificano la logica deterministica estratta in `src/leaves/` (parsing, forme dei
tensori, invarianti di dominio). Non addestrano la rete, non leggono il dataset e
non asseriscono metriche stocastiche (accuratezza/AUROC/F1). Dettagli in [docs/test.md](docs/test.md).

Dipendenze di test:

```bash
pip install pytest pytest-cov pandas scikit-learn
```

Esecuzione dalla root del progetto:

```bash
pytest        # esegue la suite con coverage su src/leaves
pytest -v     # output dettagliato per singolo test
```

La configurazione è in `pyproject.toml` (aggiunge `src/` al path e abilita la coverage).
La tabella finale di `pytest` mostra la coverage per file: la colonna `Cover` indica la
percentuale di righe eseguite e `Missing` le righe non coperte. La coverage su `src/leaves`
è ~100% perché in `src/` è stata estratta solo logica deterministica, tutta testata; le parti
escluse (training, IO, plotting, Optuna) restano nel notebook e fuori dalla misura.

In alternativa, il notebook `leaves_tests.ipynb` esegue la stessa suite inline mostrandone
l'esito.

Mentre `ci.yml` crea un workflow che ad ogni push riesegue i test e ne verifica il successo o il fallimento.
