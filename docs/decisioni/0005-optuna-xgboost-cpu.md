# ADR 0005 — Tuning con Optuna e XGBoost forzato su CPU

**Stato:** accettata · **Ambito:** classificatore supervisionato

## Contesto

Il classificatore XGBoost ha diversi iperparametri sensibili (profondità, learning rate,
sottocampionamenti, regolarizzazione). Serviva una ricerca efficiente e riproducibile,
oltre a una scelta del dispositivo di calcolo.

## Decisione

- **Ricerca iperparametri con Optuna** (ottimizzazione bayesiana), 20 trial, obiettivo:
  massimizzare l'**accuratezza sul validation set**. Spazio di ricerca: `max_depth [3,8]`,
  `learning_rate [0.01,0.3]` (log), `subsample [0.6,1.0]`, `colsample_bytree [0.6,1.0]`,
  `reg_lambda [1e-3,10]` (log).
- **XGBoost su CPU**: `XGBClassifier(..., tree_method="hist", device="cpu", n_jobs=-1)`,
  `n_estimators=600` con `early_stopping_rounds=20` sul validation.

## Conseguenze

- **Positive:**
  - Optuna esplora lo spazio in modo più mirato del grid search; l'early stopping limita
    il numero effettivo di alberi ed evita overfitting.
  - Configurazione trovata: `max_depth=7, learning_rate≈0.0274, subsample≈0.793,
    colsample_bytree≈0.809, reg_lambda≈0.0309`.
  - `device="cpu"` con `tree_method="hist"` rende l'esecuzione **riproducibile e portabile**,
    senza dipendere dalla presenza/versione della GPU — utile su ambienti eterogenei
    (es. valutazione d'esame su macchine diverse).
- **Negative / vincoli:**
  - Su feature da 2.048 dim, la CPU è più lenta di una GPU; 20 trial sono un compromesso
    tra qualità della ricerca e tempo.
  - L'obiettivo di Optuna è la sola accuratezza sul validation: non ottimizza
    direttamente F1 macro o il comportamento sulle classi deboli.

> Nota: il notebook imposta esplicitamente `device="cpu"`. La motivazione (portabilità /
> riproducibilità) è ricostruita dal contesto del progetto e non da un commento nel codice.
