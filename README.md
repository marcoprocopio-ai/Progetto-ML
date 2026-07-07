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
├── requirements.txt                        # dipendenze runtime (notebook), versioni pinnate
├── requirements-dev.txt                    # dipendenze per i test
├── pyproject.toml                          # configurazione pytest + coverage
├── .github/workflows/ci.yml                # CI: esegue i test a ogni push
├── docs/                                   # documentazione
├── models/
│   ├── leaves_classifier.ipynb             # notebook modello (originale)
│   └── leaves_classifier_tuned.ipynb       # variante con iperparametri ottimizzati
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

Installare le dipendenze runtime (versioni pinnate in `requirements.txt`):

```bash
pip install -r requirements.txt
```

Il progetto usa Keras con backend **torch**: impostare la variabile d'ambiente
`KERAS_BACKEND=torch` prima di avviare Jupyter (PowerShell: `$env:KERAS_BACKEND="torch"`;
Linux/macOS: `export KERAS_BACKEND=torch`). Senza questa variabile Keras tenterebbe di
caricare TensorFlow.

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
cartella esiste già, l'estrazione viene saltata. Il notebook usa il sottoinsieme `segmented`.

## Esecuzione

Avviare Jupyter ed eseguire il notebook:

```bash
jupyter notebook
# oppure
jupyter lab
```

Aprire un notebook ed eseguire le celle in ordine:

- `models/leaves_classifier.ipynb` — versione originale del modello;
- `models/leaves_classifier_tuned.ipynb` — stessa struttura con iperparametri ottimizzati
  (cap del campionamento 250→500, Optuna 20→45 trial con sampler seminato, `batch_size`
  32→64, `EarlyStopping patience` 8→12). Il protocollo di valutazione (seed, split, metriche)
  è identico all'originale per un confronto equo.

I percorsi ai dati e ai checkpoint (`plantvillage/`, `ckpt/`) sono **relativi alla cartella di
esecuzione**: avviare Jupyter dalla cartella del progetto per non rompere i riferimenti.

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
