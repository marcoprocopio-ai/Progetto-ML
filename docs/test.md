# Strategia di test

La logica deterministica del notebook `models/leaves_classifier.ipynb` è stata **copiata**
in `src/leaves/` per renderla importabile e testabile. Su questa logica è costruita una suite `pytest` in `tests/`. Vedi anche
[decisioni/0006](decisioni/0006-logica-estratta-src-testabilita.md).

## Principio: test ≠ valutazione

- Un **test** verifica un'**invariante deterministica**: dato un input, l'output è sempre
  quello atteso (forma di un tensore, parsing di una stringa, una regola di dominio).
- La **valutazione** misura **metriche stocastiche** del modello (accuratezza, AUROC, F1),
  che dipendono da inizializzazione, addestramento e split. Sono l'oggetto del notebook e
  di [esperimenti.md](esperimenti.md), **non** dei test.

Di conseguenza la suite **non asserisce mai** accuratezza/AUROC/F1: sarebbe fragile e
concettualmente sbagliato. Nessun test addestra la rete o legge il dataset reale.

## Cosa si testa (`tests/`)

| File | Verifica |
|---|---|
| `test_data.py` | `parse_class`: estrazione `(pianta, condizione, sana)`, pulizia parentesi della specie (`Corn_(maize)` → `Corn`), riconoscimento `healthy` case-insensitive. `load_images`: forma `(N,size,size,3)`, dtype `float32`, valori in `[0,255]`, resize a quadrato da input non quadrato. Immagini **sintetiche** in `tmp_path`. |
| `test_models.py` | Smoke test delle **sole forme**: encoder `(1,128,128,3)→(1,8,8,BOTTLENECK_CH)`, decoder `(1,8,8,BOTTLENECK_CH)→(1,128,128,3)`, autoencoder round-trip. Nessun addestramento. |
| `test_pipeline.py` | Invarianti di dominio: il training dell'autoencoder contiene **zero foglie malate**; il campionamento bilanciato non supera mai **250** esempi per classe; `LabelEncoder` è invertibile (`encode`→`inverse_transform`). |

### Invariante protetta dai test sulle forme

Il decoder ha lo spaziale del bottleneck **hardcoded a 8×8** (cella 27 del notebook), valore
che deriva da `IMG_SIZE=128` ridotto da 4 MaxPooling (128 → 8). Se `IMG_SIZE` cambia,
l'uscita dell'encoder non è più 8×8 e non combacia con l'ingresso del decoder: i test di
`test_models.py` falliscono, segnalando esplicitamente la rottura dell'accoppiamento.

## Cosa NON si testa (escluso deliberatamente)

- **Addestramento** dell'autoencoder e di XGBoost: lento, stocastico, non un'invariante.
- **IO del dataset reale** e download da Kaggle.
- **Plotting** (grafici, matrici di confusione, ricostruzioni).
- **Tuning con Optuna**: ricerca stocastica, appartiene alla valutazione.

Queste parti **non sono state copiate in `src/`**: restano solo nel notebook, quindi non
figurano nella misura di coverage.

## Coverage

La configurazione (`pyproject.toml`) misura la coverage su `src/leaves`. Poiché in `src/`
è stata estratta **solo** logica deterministica — tutta esercitata dai test — la coverage
risulta **~100% per costruzione**. Non è un numero gonfiato: è alto perché il codice non
testabile (training/IO/plot/Optuna) è stato lasciato fuori da `src/`, non perché siano stati
aggiunti test finti. La coverage va letta come *"tutto ciò che è stato estratto è coperto"*,
non come una misura della copertura dell'intero notebook.

## Come eseguire

```bash
pytest                 # esecuzione con coverage (config in pyproject.toml)
pytest -v              # output dettagliato per singolo test
```

Il backend Keras (`torch`) è impostato automaticamente da `src/leaves/models.py`
(`os.environ.setdefault("KERAS_BACKEND", "torch")`), come nella cella 2 del notebook.
L'artefatto dimostrativo `leaves_tests.ipynb` esegue la stessa suite inline.
