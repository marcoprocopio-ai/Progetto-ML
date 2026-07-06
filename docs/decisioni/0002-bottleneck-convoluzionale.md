# ADR 0002 — Bottleneck convoluzionale invece di denso

**Stato:** accettata · **Ambito:** architettura autoencoder

## Contesto

Un autoencoder può comprimere l'immagine in un vettore latente **denso** (appiattendo e
usando layer `Dense`) oppure mantenere un bottleneck **convoluzionale** che conserva la
struttura spaziale. Nel progetto il bottleneck è l'ultimo layer dell'encoder:
`Conv2D(32, 3, padding="same", activation=leaky_relu)`, con uscita **8×8×32**.

## Decisione

Adottare un **bottleneck convoluzionale** (8×8×32 = 2.048 valori) senza layer densi.

## Conseguenze

- **Positive:**
  - Conserva l'informazione spaziale (griglia 8×8): forma, colore e venature restano
    ricostruibili, come osservato nella verifica qualitativa delle ricostruzioni.
  - Meno parametri e training più stabile rispetto a un collo di bottiglia denso
    equivalente su immagini 128×128.
  - Il tensore 8×8×32 è direttamente riutilizzabile: appiattito diventa il vettore di
    feature da 2.048 dim per XGBoost (vedi
    [0004](0004-encoder-feature-extractor.md)).
- **Negative / vincoli:**
  - La rappresentazione resta **locale**: non c'è un layer che forzi una sintesi globale
    dell'immagine, quindi le feature sono ancora legate alla posizione dei pixel.
  - 2.048 dimensioni sono relativamente alte come input per un classificatore su soli
    2.500 esempi.

> Nota: il notebook implementa il bottleneck convoluzionale e ne cita il beneficio nelle
> celle markdown, ma non riporta un confronto diretto con una variante densa.
