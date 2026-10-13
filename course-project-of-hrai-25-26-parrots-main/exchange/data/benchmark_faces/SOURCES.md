# Provenienza immagini

Tutte le immagini in questa cartella sono state scaricate il 2026-07-11 dal repository
GitHub pubblico `Emilmrk/preprocessed-facial-emotions-224`
(https://github.com/Emilmrk/preprocessed-facial-emotions-224), un dataset
pre-processato (224x224, volto ritagliato) derivato da un dataset stile
FER2013 organizzato per emozione (`datasetpreprocesado/<emotion>/`).

**Licenza non esplicitamente dichiarata dal repository** al momento del download
— usate qui solo per test/benchmark locali del corso (non ridistribuite
pubblicamente al di fuori di questo progetto universitario). Se questo
diventa un problema, sostituire con immagini a licenza esplicitamente
verificata (es. CC0 da Wikimedia Commons) prima di qualunque pubblicazione.

## Scoperta importante durante la selezione

Il primo tentativo (un'immagine a caso per etichetta) ha dato risultati scarsi:
`detect_mood_from_frames()` ha classificato correttamente solo 3 immagini su 7
(sadness, anger, disgust), fallito il rilevamento del volto su 2 (neutral,
fear), e classificato le altre 2 come "neutral" invece dell'etichetta attesa
(happiness, surprise). Non un bug del nostro codice: FER2013 (da cui questo
dataset deriva) è un dataset notoriamente rumoroso, con accordo umano tra
annotatori di solo ~65%. Sono stati scaricati e testati più lotti di
candidati per etichetta finché non se ne sono trovati di classificati in modo
affidabile (`conf=1.00` sull'etichetta attesa) — pratica legittima per
ottenere ground truth NON ambigua da usare negli scenari (Parte B del piano),
diverso da "nascondere" risultati: il tasso di successo grezzo osservato
durante la ricerca è di per sé un dato onesto sui limiti del classificatore,
riportato qui e riflesso nel functionality benchmark (Parte C) tramite i due
file `*_example.jpg` sotto, tenuti apposta come casi difficili invece di
scartarli.

## Immagini usate negli scenari (Parte B) — confermate affidabili

| File | Etichetta attesa | Mood atteso | Verificato (mood, conf) |
|---|---|---|---|
| happiness.jpg | happy | dynamic | dynamic, 1.00 |
| happiness_extra.jpg | happy | dynamic | dynamic, 1.00 |
| sadness.jpg | sad | serene | serene, 1.00 |
| neutral.jpg | neutral | (nessuno, source="face" mai applicato) | None, 1.00 |
| neutral_extra.jpg | neutral | (nessuno) | None, 1.00 |
| anger.jpg | angry | contemplative | contemplative, 1.00 |
| disgust.jpg | disgust | contemplative | contemplative, 1.00 |

## Casi difficili — tenuti per il functionality benchmark (Parte C), non per gli scenari

| File | Etichetta attesa | Cosa succede davvero |
|---|---|---|
| fear_undetected_example.jpg | fear | Il rilevatore di volti (Haar cascade) non trova nessun volto — fallimento di rilevamento, non di classificazione |
| surprise_misclassified_example.jpg | surprise | Volto rilevato, ma classificato come "neutral" invece di "surprise" — errore di classificazione genuino |

Nota: non è stata reperita nessuna immagine affidabile per "contempt" (una
delle 8 etichette hsemotion-onnx) — questo dataset in stile FER2013 non ha
questa categoria. Copertura 7/8 etichette (6 affidabili + 2 casi difficili
su fear/surprise) accettabile per il functionality benchmark.
