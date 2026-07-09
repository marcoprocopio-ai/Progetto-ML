# Classificatore di malattie delle foglie — PlantVillage

Progetto di Machine Learning di classificazione delle malattie nelle foglie di
**pomodoro** sul dataset **PlantVillage** preso da Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset. 


## Setup

### Requisiti di sistema
- **Python 3.13** *consigliato* (garantisce i wheel di `torch` su Linux/macOS/Windows).
- ~3 GB liberi per il dataset estratto.


### Ambiente e dipendenze 
versioni pinnate in [`requirements.txt`](requirements.txt):

```bash
# 1. ambiente virtuale
python -m venv .venv
# Windows / PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 2. dipendenze runtime e Jupyter per aprire il notebook
pip install -r requirements.txt
pip install jupyterlab          

# 3. backend di Keras
   PowerShell:  $env:KERAS_BACKEND="torch"
   Linux/macOS: export KERAS_BACKEND=torch
```


### Esecuzione

Avviare Jupyter ed eseguire il notebook:

```bash
jupyter notebook
# oppure
jupyter lab
```

Aprire `models/leaves_classifier.ipynb` ed eseguire le celle in ordine.

I percorsi ai dati e ai checkpoint (`plantvillage/`, `ckpt/`) sono **relativi alla cartella di esecuzione**: avviare Jupyter dalla cartella del progetto per non rompere i riferimenti.

## Spiegazione del progetto

Il progetto affronta la **diagnosi visiva di malattie fogliari**, data l'immagine di una
foglia di pomodoro, riconoscerne una tra le 10 condizioni (9 malattie + `healthy`), seguita da una 
**classificazione multi-classe** su immagini.

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
> è implementata** e l'encoder è usato solo come estrattore di feature.
> per i motivi documentati negli ADR:
> [0002](docs/decisioni/0002-autoencoder-vs-patchcore.md) e
> [0007](docs/decisioni/0007-encoder-feature-extractor.md).

E' in oltre importante specificare che **XGBoost** utilizza dati tabulari. Il dataset PlantVillage è composto da sole immagini, motivo per cui non è stato possibile applicare il solo modello di classificazione sin dall'inizio. E l'architettura utilizzata vedi [docs/architettura.md](docs/architettura.md), risolve questo limite utilizzando feature tabulari estratte direttamente dall'autoencoder.

Maggiori informazioni e diagramma di flusso completo è documentato in [`docs/architettura.md`](docs/architettura.md) e i
risultati in [`docs/esperimenti.md`](docs/esperimenti.md).

## Dati
- **Fonte:** [PlantVillage — Kaggle `abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) ~2,2 GB, **non incluso nel repo** per via della sua dimensione.
Va scaricato **una sola volta**; non serve rinominare né spostare nulla a mano: basta lasciare lo zip scaricato in una di queste posizioni e il notebook lo trova ed estrae in automatico:
  - la cartella da cui si avvia il notebook, oppure la root del progetto;
  - la cartella `~/Downloads`.

Sono riconosciuti i nomi più comuni dello zip (es. `archive*.zip`, `plantvillage*.zip`).
Alla prima esecuzione viene estratto in `plantvillage/`; nelle esecuzioni successive, se la
cartella esiste già, l'estrazione viene saltata.

- **Panoramica generale:** **54.306 immagini**, **14 specie**, **21 condizioni** complessive. Le immagini sono organizzate in cartelle `Pianta___Condizione`

- **Focus sulla pianta di pomodoro:** si usa la sola specie `Tomato`, la più ricca con **10 condizioni** (9 malattie + `healthy`). Vedi [ADR 0003](docs/decisioni/0003-scelta-foglia-pomodoro.md).

- **`segmented` vs `color`:** si usa il sottoinsieme **`segmented`** (foglia su sfondo nero). Vedi [ADR 0004](docs/decisioni/0004-sottoinsieme-segmented.md).

**Gestione dati mancanti o mal formattati:**
- **Parsing** (`parse_class`): dal nome cartella ricava `(specie, condizione, sana)`;
  ripulisce le parentesi della specie (`Corn_(maize)` → `Corn`) e determina  `healthy` in modo *case-insensitive*.

- **Filtri:** vengono considerati solo i file con estensione `.jpg/.jpeg/.png` e solo le cartelle che contengono `___`; il resto è ignorato.

- Non è presente imputazione di dati mancanti né gestione esplicita di file corrotti: il dataset PlantVillage è già pulito e strutturato.

## Ciclo di vita ML

Mappatura delle fasi del **ciclo di vita ML** su questo progetto:

| Fase | Come applicarli |
|---|---|
| **Raccolta dati** | Download una-tantum di PlantVillage da Kaggle; estrazione automatica; sottoinsieme `segmented`, specie `Tomato`. |
| **Preparazione / feature** | Parsing cartelle, resize 128×128, estrazione feature tramite l'encoder (bottleneck `8×8×32` → 2.048 dim). |
| **Training** | Autoencoder sulle foglie sane; XGBoost supervisionato sulle feature, con tuning iperparametri via Optuna. |
| **Validazione** | Split **stratificato 60/20/20**; valutazione su test (accuratezza, F1 per classe, matrice di confusione). In ottica *testing ML* non si verificano output esatti ma che **il modello abbia senso**. |
| **Deploy** | Impacchettando `encoder + XGBoost` dietro un servizio di inferenza, che riceve un'immagine e restituisce la condizione. |
| **Monitoring** |   |

## MLOps

Se il modello andasse in produzione, gli aspetti da presidiare sono *monitoraggio e osservabilità*, *Continuous Training*:

**Cosa monitorare**
- **Data drift**  → sulle immagini in ingresso: nuove inquadrature, illuminazione, sfondi reali diversi da `segmented` di training → le feature dell'encoder possono diventare poco affidabili.
- **Concept drift** → comparsa di **nuove specie o malattie** non tra le 10 classi note; il modello, per costruzione, può classificare solo condizioni viste in training.
- **Calo di performance** →  accuratezza/F1 complessivi e **per classe** (specialmente le classi già deboli).

**Quando fare re-training**
- Superamento di una soglia di degrado di accuratezza/F1 su un set di controllo etichettato.
- Drift rilevato sulle distribuzioni delle immagini o delle feature.
- Ingresso di nuove classi  → richiede ampliare le etichette e riaddestrare.


## Rischi, assunzioni e limiti

Il notebook, dai dati alle metriche risulta pienamente funzionante (end-to-end) , produce un classificatore con **accuratezza sul test = 0.702** (risultati in [`docs/esperimenti.md`](docs/esperimenti.md)).


### Limite strutturale 
L'idea iniziale di rilevare le malate tramite **errore di ricostruzione** dell'autoencoder ha un limite pixel-based →  l'errore tende a concentrarsi sulle **strutture ad alta frequenza** (venature, bordi) più che sulle
**lesioni** della malattia  →  di fatto misura quanto è difficile ricostruire le venature e non quanto è malata la foglia. 

Da qui la scelta di usare l'encoder come estrattore di feature ([ADR 0007](docs/decisioni 0007-encoder-feature-extractor.md)).

**Floor di performance.** Le feature vengono da un encoder ottimizzato per **ricostruire
foglie sane**, non per **riconoscere malattie** → questo pone un tetto alla resa di XGBoost, ma è stato proprio il punto focale dell'esperimento.
Le confusioni maggiori sono tra malattie visivamente simili (`Early_blight` / `Target_Spot` /
`Late_blight`). .


### Assunzioni
- Campionamento **bilanciato** a 250 immagini per classe →  non riflette la distribuzione reale, ma è rientra in una decisione presa in corso d'opera che ha posto anche un forte limite alla resa finale del modello di classificazione, vedi ([ADR 0008](docs/decisioni/0008-optuna-xgboost-cpu.md)).

- Solo pomodoro → il modello non classifica altre specie, vedi ([ADR 0003](docs/decisioni/0003-scelta-foglia-pomodoro.md)).

- Nessuna gestione di input fuori distribuzione → il modello assegna sempre una delle 10 classi note Un'immagine non-foglia o di un'altra specie viene comunque classificata, senza opzione di reject.


### Come ampliarlo
- Usare più dati per classe e più specie.
- Mettere in opera tutte le fasi del ciclo di vita ML, realizzando il deploy e monitoring.
- Rivedere l'approccio di anomaly detection con metodi non pixel-based, se si volesse recuperare il rilevamento sano/malato non supervisionato.
- Testare il modello con dataset diversi cosi da testare la funzionalità del modello che potrebbe rivelare ottimi risultati.

## Ulteriori informazioni

### Struttura del repository
```
Progetto ML/
├── README.md
├── requirements.txt        # dipendenze runtime (Notebook)
├── .gitignore
├── docs/                   # documentazione
│   ├── architettura.md     # pipeline completa e diagramma di flusso
│   ├── esperimenti.md      # metriche reali (accuratezza, F1, limiti)
│   └── decisioni/          # ADR 0001–0008
└── models/
    └── leaves_classifier.ipynb   # notebook del modello
```
Cartelle non versionate (in `.gitignore`): `.venv/`, `plantvillage/` (dataset), `ckpt/`
(pesi dell'autoencoder), `*.zip`.

### Documentazione

- [docs/architettura.md](docs/architettura.md) — Pipeline completa e diagramma del flusso.
- [docs/esperimenti.md](docs/esperimenti.md) — Risultati finali
- [docs/decisioni/](docs/decisioni/) — ADR sulle scelte progettuali principali.


| **#** | **Architettura ADR:** |
|---|---|
| [0001](docs/decisioni/0001-keras-torch.md) | Keras con backend torch |
| [0002](docs/decisioni/0002-autoencoder-vs-patchcore.md) | Autoencoder vs PatchCore |
| [0003](docs/decisioni/0003-scelta-foglia-pomodoro.md) | Focus sulle foglie di pomodoro |
| [0004](docs/decisioni/0004-sottoinsieme-segmented.md) | Sottoinsieme `segmented` vs `color` |
| [0005](docs/decisioni/0005-bottleneck-convoluzionale.md) | Bottleneck convoluzionale |
| [0006](docs/decisioni/0006-attivazione-output-leaky-relu.md) | Attivazione `leaky_relu` |
| [0007](docs/decisioni/0007-encoder-feature-extractor.md) | Encoder come estrattore di feature |
| [0008](docs/decisioni/0008-optuna-xgboost-cpu.md) | Tuning Optuna + XGBoost su CPU |

### Sviluppi Futuri
- **Test e CI:** una suite `pytest` e un workflow GitHub Actions.
