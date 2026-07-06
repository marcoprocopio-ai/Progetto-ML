# ADR 0001 — Uso del sottoinsieme `segmented` invece di `color`

**Stato:** accettata · **Ambito:** preprocessing dataset

## Contesto

PlantVillage fornisce le stesse foglie in tre varianti (`color`, `grayscale`,
`segmented`). Le immagini `segmented` hanno lo **sfondo rimosso** (foglia su sfondo nero),
mentre `color` conserva lo sfondo originale. La scelta è controllata dal flag
`USE_SEGMENTED = True`, che seleziona la sottocartella `segmented`.

## Decisione

Usare il sottoinsieme **`segmented`** come input sia per l'autoencoder sia per
l'estrazione di feature.

## Conseguenze

- **Positive:** eliminando lo sfondo, il modello si concentra sulla foglia; l'autoencoder
  non spreca capacità del bottleneck a ricostruire contesto irrilevante, e le feature
  dell'encoder dipendono meno da variazioni di sfondo.
- **Negative / vincoli:** lo sfondo nero domina l'immagine ed è composto da pixel a
  valore ~0; questo interagisce con la scelta dell'attivazione di output
  (vedi [0003](0003-attivazione-output-leaky-relu.md)). Inoltre il modello non è esposto
  a foglie su sfondo reale, quindi la generalizzazione a immagini "in campo" non è
  garantita.

> Nota: il notebook imposta esplicitamente `USE_SEGMENTED = True` ma non contiene un
> confronto quantitativo `segmented` vs `color`. La motivazione qui è ricostruita dalla
> logica del codice, non da un esperimento comparativo documentato.
