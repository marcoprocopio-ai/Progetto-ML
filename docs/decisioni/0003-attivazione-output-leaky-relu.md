# ADR 0003 — Attivazione di output `leaky_relu`

**Stato:** accettata · **Ambito:** architettura autoencoder

## Contesto

L'ultimo layer del decoder deve produrre 3 canali che, dopo `Rescaling(255.0)`,
ricostruiscono pixel in `[0, 255]`. Le immagini `segmented` hanno **ampie aree di sfondo
nero** (pixel ~0). La scelta dell'attivazione di output influenza sia la capacità di
riprodurre questi valori sia la stabilità del gradiente.

## Decisione

Usare **`leaky_relu`** come attivazione di output (e in tutti i layer convoluzionali),
seguita da `Rescaling(255.0)`.

## Conseguenze

- **Positive:**
  - Gestisce bene lo sfondo scuro delle immagini segmentate.
  - Mantiene sempre un **piccolo gradiente** anche per valori negativi, evitando neuroni
    "morti" tipici della `relu` pura.
- **Negative / vincoli:**
  - `leaky_relu` non vincola l'uscita a un intervallo limitato: la ricostruzione può
    produrre valori fuori `[0, 255]`, gestiti a valle con `clip(0, 1)` in fase di
    visualizzazione.

## Alternative scartate

- **`sigmoid`** (uscita naturale in `[0, 1]`) e **`relu`** — secondo l'esperienza dello
  sviluppatore avrebbero portato a **collassi della ricostruzione** su questo dataset.

> **Da confermare:** il notebook adotta `leaky_relu` e ne motiva l'uso per lo sfondo scuro
> e il gradiente, ma **non documenta** gli esperimenti di collasso con `sigmoid`/`relu`.
> Questa motivazione proviene dall'autore e non è dimostrata da output nel codice.
