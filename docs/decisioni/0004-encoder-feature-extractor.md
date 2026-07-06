# ADR 0004 — Encoder come estrattore di feature invece di rilevamento anomalie pixel-based

**Stato:** accettata (di fatto, nel codice attuale) · **Ambito:** strategia della pipeline

## Contesto

L'idea iniziale del progetto era **non supervisionata**: addestrare un autoencoder solo su
foglie sane e rilevare le malate tramite l'**errore di ricostruzione** (una foglia malata,
mai vista in training, dovrebbe ricostruirsi peggio → punteggio di anomalia elevato,
valutabile con AUROC). A questo scopo il notebook predispone `TOPK_FRAC = 0.02`
(frazione di pixel peggiori come score), `gaussian_filter` e `roc_auc_score`.

Un limite strutturale di questo approccio pixel-based: l'errore di ricostruzione tende a
concentrarsi sulle **strutture ad alta frequenza** della foglia (venature, bordi) più che
sulle **lesioni** della malattia. Lo score di anomalia finisce per misurare "quanto è
difficile ricostruire le venature" invece di "quanto è malata la foglia".

## Decisione

Nel codice attuale il rilevamento anomalie via errore di ricostruzione **non è
implementato**: `TOPK_FRAC`, `gaussian_filter` e `roc_auc_score` restano inutilizzati.
L'autoencoder, addestrato sulle sane, viene sfruttato **solo come estrattore di feature**:
il suo encoder produce il bottleneck 8×8×32, appiattito a 2.048 dim, che alimenta un
classificatore **XGBoost supervisionato** sul tipo di malattia.

## Conseguenze

- **Positive:**
  - Si aggira il limite pixel-based: la classificazione supervisionata impara
    direttamente a distinguere le condizioni, ottenendo un'accuratezza test di **0.702**
    (vedi [../esperimenti.md](../esperimenti.md)).
  - Le feature dell'encoder sono compatte e riutilizzabili.
- **Negative / vincoli:**
  - Si **perde la natura non supervisionata**: il rilevamento delle malate ora richiede
    etichette. Non esiste più un rilevatore di anomalie funzionante né una metrica AUROC.
  - Le feature vengono da un encoder ottimizzato per **ricostruire foglie sane**, non per
    discriminare malattie: sono sub-ottimali per la classificazione e pongono un tetto
    alla resa.
  - Rimane codice "morto" (import e costanti dell'anomaly detection) che segnala
    l'intento originale ma non è attivo.

> **Parzialmente da confermare:** l'assenza dell'anomaly detection e il codice inutilizzato
> sono **fatti verificati** nel notebook. La spiegazione del limite pixel-based
> (errore che segue le venature invece delle lesioni) è la motivazione progettuale
> dell'autore e non è dimostrata da un esperimento nel codice.
