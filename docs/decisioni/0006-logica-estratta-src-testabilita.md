# ADR 0006 — Logica estratta in `src/` per testabilità, notebook originale preservato

**Stato:** accettata · **Ambito:** struttura del progetto e testing

## Contesto

Tutta la logica del progetto viveva dentro `models/leaves_classifier.ipynb`. Un notebook
non è direttamente importabile né testabile con `pytest`: le funzioni sono legate allo stato
delle celle e all'ordine di esecuzione. Serviva una suite di test automatica, ma il notebook
è un **artefatto d'esame da preservare intatto** (deve restare identico byte per byte).

## Decisione

**Copiare** (non spostare) la sola logica deterministica in un package importabile
`src/leaves/`:

- `data.py` — `parse_class`, `load_images` (celle 10, 19);
- `models.py` — `build_encoder`, `build_decoder`, `build_autoencoder` (celle 26–28),
  parametrizzati su `IMG_SIZE`/`BOTTLENECK_CH` ma con layer identici all'originale;
- `pipeline.py` — filtro foglie sane, campionamento bilanciato, `LabelEncoder`
  (celle 21, 37, 40).

Il notebook originale **non viene modificato** e continua a funzionare in autonomia. La suite
`pytest` in `tests/` importa da `src/`; `pyproject.toml` aggiunge `src/` al path e configura
la coverage.

## Conseguenze

- **Positive:**
  - Logica verificabile automaticamente, con invarianti protette da regressioni (es. la
    forma 8×8 hardcoded del decoder).
  - Il notebook resta l'artefatto d'esame originale, non contaminato da codice di test.
  - Separazione netta tra codice riusabile (`src/`), test (`tests/`) e narrazione/valutazione
    (notebook).
- **Negative / vincoli:**
  - **Duplicazione**: la logica esiste in due punti (notebook e `src/`). Una modifica al
    notebook non si propaga automaticamente ai moduli e viceversa: vanno mantenuti allineati
    a mano. Il confronto affiancato in fase di estrazione documenta l'identità iniziale.
  - Solo la logica deterministica è stata copiata: training, IO, plotting e Optuna restano
    nel notebook e fuori dalla coverage (vedi [../test.md](../test.md)).

> Nota: l'identità byte-per-byte del notebook è stata verificata con hash SHA-256 prima e
> dopo l'estrazione.
