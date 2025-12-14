# Manuale dell'Applicazione: Generatore di Schede Giornalistiche per Degustazioni

## 📋 Indice

1. [Panoramica](#panoramica)
2. [Architettura](#architettura)
3. [Componenti del Sistema](#componenti-del-sistema)
4. [Installazione e Setup](#installazione-e-setup)
5. [Utilizzo](#utilizzo)
6. [Visualizzazione del Training](#visualizzazione-del-training)
7. [Struttura dei File](#struttura-dei-file)
8. [Formato dei Dati](#formato-dei-dati)
9. [Configurazione](#configurazione)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Panoramica

Questa applicazione è un sistema di **fine-tuning** basato su Large Language Models (LLM) per la generazione automatica di **schede giornalistiche** e **abbinamenti culinari** a partire da dati strutturati di degustazioni di vino.

### Cosa fa l'applicazione?

L'applicazione:
- **Legge** dati strutturati di degustazioni (JSON) contenenti informazioni analitiche e descrittive
- **Converte** questi dati in formato narrativo
- **Addestra** un modello linguistico (Microsoft Phi-2) usando la tecnica LoRA (Low-Rank Adaptation)
- **Genera** automaticamente schede giornalistiche professionali e suggerimenti di abbinamento culinario

### Tecnologie utilizzate

- **Modello base**: Microsoft Phi-2 (modello generativo di linguaggio)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) per addestramento efficiente
- **Framework**: Hugging Face Transformers, PEFT, PyTorch
- **Ambiente consigliato**: Google Colab (per accesso a GPU)

---

## 🏗️ Architettura

Il sistema è organizzato in una **pipeline modulare** composta da 4 fasi principali:

```
┌─────────────────┐
│  Dati JSON      │
│  (degustazioni) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. Preparazione │  ← pipeline_dataset.py
│    Dataset      │     fine_tuning.py
└────────┬────────┘     pipeline_prompts.py
         │
         ▼
┌─────────────────┐
│ 2. Training     │  ← pipeline_train_lora.py
│    LoRA         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Inferenza    │  ← pipeline_infer.py
│    Generazione  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Output       │
│    Schede JSON  │
└─────────────────┘
```

---

## 🔧 Componenti del Sistema

### 1. `fine_tuning.py`
**Scopo**: Conversione dati strutturati → testo narrativo

**Funzione principale**: `json_to_narrative()`
- Converte i dati JSON strutturati di una degustazione in un testo narrativo leggibile
- Estrae e formatta:
  - Informazioni base (nome, produttore, denominazione, anno, colore, vitigni)
  - Esame visivo (limpidezza, colore, consistenza)
  - Esame olfattivo (intensità, complessità, descrittori)
  - Esame gusto-olfattivo (dolcezza, acidità, struttura, equilibrio, persistenza)
  - Considerazioni finali

### 2. `pipeline_prompts.py`
**Scopo**: Costruzione dei prompt per training e inferenza

**Funzioni**:
- `build_prompt()`: Crea il prompt con istruzioni per il modello
- `build_target_json()`: Formatta l'output target come JSON
- `escape_json_str()`: Gestisce l'escape dei caratteri speciali nel JSON

**Formato prompt**:
```
### INPUT
[testo narrativo della degustazione]

### ISTRUZIONI
- Scrivi una scheda giornalistica in italiano, 2-4 frasi, tono professionale.
- Suggerisci un abbinamento (un piatto o una tipologia di piatto), massimo 10 parole.
- Non inventare fatti non presenti nell'input.
- Rispondi SOLO con un JSON valido.

### OUTPUT (JSON)
```

### 3. `pipeline_dataset.py`
**Scopo**: Preparazione del dataset per il training

**Funzioni principali**:
- `load_degustazioni()`: Carica i dati JSON
- `make_training_examples()`: Crea coppie (prompt, target) per il training
- `tokenize_and_mask()`: Tokenizza e maschera i token del prompt (la loss viene calcolata solo sul target)
- `build_hf_dataset()`: Costruisce un dataset Hugging Face

**Strategia di masking**:
- I token del prompt hanno `label = -100` (ignorati nel calcolo della loss)
- Solo i token del target contribuiscono alla loss
- Questo insegna al modello a generare solo la risposta, non a ripetere il prompt

### 4. `pipeline_train_lora.py`
**Scopo**: Addestramento del modello con LoRA

**Caratteristiche**:
- **Modello base**: Microsoft Phi-2
- **Tecnica**: LoRA (Low-Rank Adaptation)
  - `r=16`: Rank della matrice di decomposizione
  - `lora_alpha=32`: Scaling factor
  - `lora_dropout=0.05`: Dropout per regolarizzazione
  - `target_modules="all-linear"`: Applica LoRA a tutti i layer lineari

**Parametri di training**:
- Batch size: 1 (con gradient accumulation = 8, batch effettivo = 8)
- Learning rate: 2e-4
- Epochs: 5
- Max length: 768 token
- Precision: FP16 (se GPU disponibile)

**Output**: Salva il modello LoRA in `./fine_tuned_lora`

### 5. `pipeline_train_lora.py` (Callback MetricsCallback)
**Scopo**: Salvataggio automatico delle metriche durante il training

**Classe**: `MetricsCallback`
- Salva automaticamente le metriche in `training_metrics.json` durante il training
- Traccia: loss, learning rate, step, epoch
- Salvataggio in tempo reale ad ogni log step

### 6. `visualize_training.py`
**Scopo**: Visualizzazione grafica delle metriche di training

**Funzioni principali**:
- `load_metrics()`: Carica le metriche dal file JSON
- `plot_training_curves()`: Crea 4 grafici completi (loss, epoch, LR, statistiche)
- `plot_loss_only()`: Crea un grafico semplice solo della loss

**Caratteristiche**:
- Grafici ad alta risoluzione (300 DPI)
- Annotazioni con valori finali
- Statistiche riassuntive
- Supporto per visualizzazione in notebook

### 7. `pipeline_infer.py`
**Scopo**: Generazione di schede giornalistiche per nuove degustazioni

**Funzioni principali**:
- `load_model()`: Carica il modello base e la LoRA addestrata
- `generate_for_degustazione()`: Genera scheda e abbinamento per una degustazione
- `extract_json()`: Estrae il JSON dall'output del modello
- `main()`: Esegue l'inferenza su un file JSON completo

**Parametri di generazione**:
- `temperature=0.25`: Bassa temperatura per output più deterministici
- `top_p=0.7`: Nucleus sampling
- `top_k=40`: Top-k sampling
- `repetition_penalty=1.2`: Penalità per ripetizioni
- `max_new_tokens=220`: Lunghezza massima dell'output

---

## 💻 Installazione e Setup

### Prerequisiti

- Python 3.9+
- Google Colab (consigliato) o ambiente con GPU
- Account Hugging Face (per scaricare i modelli)

### Installazione su Google Colab

1. **Apri un nuovo notebook su Colab**

2. **Installa le dipendenze** (copia il contenuto di `install_dependencies_colab.txt`):
```python
!pip install -q --upgrade pip && \
pip install -q --upgrade --force-reinstall numpy>=2.0 && \
pip install -q --upgrade --force-reinstall pandas && \
pip install -q --upgrade \
  torch \
  transformers==4.44.2 \
  peft==0.12.0 \
  accelerate==0.33.0 \
  datasets \
  bitsandbytes \
  sentencepiece \
  fsspec
```

3. **Carica i file del progetto** su Colab:
   - `fine_tuning.py`
   - `pipeline_prompts.py`
   - `pipeline_dataset.py`
   - `pipeline_train_lora.py`
   - `pipeline_infer.py`
   - `degustazioni.json` (file di training)
   - `input.json` (file per inferenza)

### Installazione locale

```bash
pip install -r requirements.txt
```

---

## 🚀 Utilizzo

### Fase 1: Preparazione del Dataset

Assicurati di avere un file `degustazioni.json` con il formato corretto (vedi sezione [Formato dei Dati](#formato-dei-dati)).

### Fase 2: Training

```python
# Su Colab o in un ambiente Python
from pipeline_train_lora import main

# Modifica i parametri in main() se necessario
main()
```

Oppure esegui direttamente:
```bash
python pipeline_train_lora.py
```

**Tempo stimato**: Dipende dalla GPU e dal numero di esempi. Su Colab T4: ~30-60 minuti per 100 esempi.

**Output**: Il modello LoRA viene salvato in `./fine_tuned_lora/`

### Fase 3: Inferenza

```python
# Su Colab o in un ambiente Python
from pipeline_infer import main

# Modifica i parametri in main() se necessario
main()
```

Oppure esegui direttamente:
```bash
python pipeline_infer.py
```

**Input**: `input.json` (file con degustazioni da processare)
**Output**: `output_infer.json` (file con schede generate)

### Fase 4: Visualizzazione del Training

Dopo aver completato il training, puoi visualizzare le metriche per monitorare l'andamento dell'addestramento.

#### Visualizzazione Completa (4 grafici)

```python
from visualize_training import load_metrics, plot_training_curves

# Carica le metriche
metrics = load_metrics("./fine_tuned_lora/training_metrics.json")

# Crea i grafici
plot_training_curves(metrics, output_file="training_curves.png", show_plot=True)
```

Oppure dalla riga di comando:
```bash
python visualize_training.py
```

**Output**: Crea un file `training_curves.png` con 4 grafici:
1. **Loss per Step**: Mostra l'andamento della loss durante il training
2. **Loss Media per Epoch**: Media della loss per ogni epoca
3. **Learning Rate**: Andamento del learning rate (scala logaritmica)
4. **Statistiche Riassuntive**: Riepilogo numerico delle metriche

#### Visualizzazione Semplice (solo Loss)

```python
from visualize_training import load_metrics, plot_loss_only

metrics = load_metrics("./fine_tuned_lora/training_metrics.json")
plot_loss_only(metrics, output_file="loss_curve.png", show_plot=True)
```

Oppure dalla riga di comando:
```bash
python visualize_training.py --simple
```

**Opzioni disponibili**:
- `--metrics-file PATH`: Specifica il percorso del file metriche (default: `./fine_tuned_lora/training_metrics.json`)
- `--output FILE`: Nome del file di output (default: `training_curves.png`)
- `--simple`: Crea solo il grafico della loss
- `--show`: Mostra il grafico (utile in notebook)

#### Esempio su Google Colab

```python
# Dopo il training
from visualize_training import load_metrics, plot_training_curves
from IPython.display import Image, display

metrics = load_metrics("./fine_tuned_lora/training_metrics.json")
plot_training_curves(metrics, output_file="training_curves.png", show_plot=True)

# Visualizza nel notebook
display(Image("training_curves.png"))
```

**Nota**: Le metriche vengono salvate automaticamente durante il training in `./fine_tuned_lora/training_metrics.json`. Se il file non esiste, assicurati di aver eseguito il training con la versione aggiornata di `pipeline_train_lora.py`.

### Esempio di utilizzo programmatico

```python
from pipeline_infer import load_model, generate_for_degustazione
import json

# Carica il modello
tokenizer, model = load_model(
    base_model_name="microsoft/phi-2",
    lora_path="./fine_tuned_lora"
)

# Carica una degustazione
with open("input.json", "r") as f:
    degustazioni = json.load(f)

# Genera la scheda
scheda, abbinamento, prompt = generate_for_degustazione(
    degustazioni[0], 
    tokenizer, 
    model
)

print(f"Scheda: {scheda}")
print(f"Abbinamento: {abbinamento}")
```

---

## 📁 Struttura dei File

```
.
├── fine_tuning.py              # Conversione JSON → testo narrativo
├── pipeline_prompts.py        # Costruzione prompt e target
├── pipeline_dataset.py         # Preparazione dataset Hugging Face
├── pipeline_train_lora.py     # Addestramento modello LoRA
├── pipeline_infer.py          # Inferenza e generazione
├── visualize_training.py      # Visualizzazione metriche training
├── example_visualize_colab.py # Esempio utilizzo visualizzazione su Colab
├── degustazioni.json          # Dataset di training
├── input.json                 # Input per inferenza
├── output_infer.json          # Output generato (creato dopo inferenza)
├── fine_tuned_lora/           # Modello LoRA addestrato (creato dopo training)
│   └── training_metrics.json  # Metriche del training (creato durante training)
├── requirements.txt           # Dipendenze Python
├── install_dependencies_colab.txt  # Comandi installazione per Colab
├── install_dependencies_colab.sh   # Script installazione per Colab
└── MANUAL.md                  # Questo manuale
```

---

## 📊 Formato dei Dati

### Formato JSON per Training (`degustazioni.json`)

```json
[
  {
    "id": "1",
    "data": "2024-05-07",
    "informazioni_base": {
      "nome_vino": "Chablis 1er Cru Fourchaume",
      "produttore": "Domaine Laroche",
      "tipologia": "vino fermo",
      "Denominazione": "Francia, Borgogna, Chablis AOC",
      "colore": "bianco",
      "anno": "2021",
      "vitigni": [
        {"nome": "Chardonnay", "percentuale": 100}
      ]
    },
    "scheda_analitico_descrittiva": {
      "esame_visivo": {
        "limpidezza": "cristallino",
        "colore_bianchi": "paglierino con riflessi verdolini",
        "consistenza": "consistente"
      },
      "esame_olfattivo": {
        "intensita_olfattiva": "intenso",
        "complessita_olfattiva": "complesso",
        "qualita_olfattiva": "ottimo",
        "descrittori": {
          "fruttato": ["scorza lime", "mango", "ananas"],
          "varietale": ["timo"],
          "altro": ["ardesia", "roccia bagnata"]
        }
      },
      "esame_gusto_olfattivo": {
        "dolcezza": "secco",
        "acidita": "fresco",
        "alcolicita": "caldo",
        "rotondita": "moderatamente_morbido",
        "sapidita": "sapido",
        "intensita_gusto": "intenso",
        "struttura": "di corpo pieno",
        "equilibrio": "equilibrato",
        "persistenza": "persistente",
        "qualita_gusto_olfattiva": "ottimo"
      },
      "considerazioni_finali": {
        "stato_evolutivo": "maturo",
        "armonia": "armonico",
        "qualita_complessiva": "ottimo"
      }
    },
    "scheda_giornalistica": "Colore che incarna ancora la gioventù...",
    "abbinamento": "linguine allo scoglio"
  }
]
```

### Formato JSON per Inferenza (`input.json`)

Stesso formato del training, ma **senza** i campi `scheda_giornalistica` e `abbinamento` (che verranno generati).

### Formato Output (`output_infer.json`)

```json
[
  {
    "id": "1",
    "scheda_giornalistica": "Colore paglierino con riflessi verdolini...",
    "abbinamento": "risotto ai frutti di mare"
  }
]
```

---

## ⚙️ Configurazione

### Parametri di Training (`pipeline_train_lora.py`)

Modifica nella funzione `main()`:

```python
MODEL_NAME = "microsoft/phi-2"      # Modello base
JSON_PATH = "degustazioni.json"     # File dataset
OUTPUT_DIR = "./fine_tuned_lora"    # Directory output
MAX_LENGTH = 768                    # Lunghezza massima token
BATCH_SIZE = 1                      # Batch size
EPOCHS = 5                          # Numero epoche
LR = 2e-4                           # Learning rate
```

### Parametri LoRA

Modifica nella funzione `main()`:

```python
lora_cfg = LoraConfig(
    r=16,                    # Rank (aumenta per più capacità)
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,       # Dropout
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
```

### Parametri di Inferenza (`pipeline_infer.py`)

Modifica nella funzione `main()`:

```python
BASE_MODEL_NAME = "microsoft/phi-2"
LORA_PATH = "./fine_tuned_lora"
INPUT_JSON = "input.json"
OUTPUT_JSON = "output_infer.json"
DEBUG = True  # Attiva debug per vedere output grezzo
```

Modifica nella funzione `generate_for_degustazione()`:

```python
max_input_tokens=800,    # Token massimi input
max_new_tokens=220,      # Token massimi output
temperature=0.25,        # Bassa = più deterministico
top_p=0.7,               # Nucleus sampling
top_k=40,                # Top-k sampling
```

---

## 🔍 Troubleshooting

### Errore: "ModuleNotFoundError: No module named 'fine_tuning'"

**Soluzione**: Assicurati che tutti i file Python siano nella stessa directory.

### Errore: "Scheda troppo corta o vuota"

**Possibili cause**:
1. Il modello non è stato addestrato correttamente
2. Il modello ha bisogno di più epoche di training
3. I dati di training non sono sufficienti

**Soluzioni**:
- Aumenta il numero di epoche nel training
- Verifica che il dataset di training sia completo
- Attiva `DEBUG=True` in `pipeline_infer.py` per vedere cosa genera il modello
- Aumenta `max_new_tokens` nella generazione

### Errore: "Nessun JSON trovato nell'output"

**Causa**: Il modello non genera output in formato JSON valido.

**Soluzioni**:
- Verifica che il modello sia stato addestrato correttamente
- Controlla il prompt (deve chiedere esplicitamente JSON)
- Aumenta `max_new_tokens` per dare più spazio al modello
- Attiva `DEBUG=True` per vedere l'output grezzo

### Errore: Conflitti di dipendenze (numpy, pandas, fsspec)

**Soluzione**: Usa i comandi in `install_dependencies_colab.txt` che gestiscono automaticamente i conflitti.

### Il modello genera output non coerenti

**Soluzioni**:
- Riduci `temperature` (es. 0.1-0.2) per output più deterministici
- Aumenta `repetition_penalty` (es. 1.3-1.5)
- Verifica la qualità del dataset di training
- Aumenta il numero di esempi nel dataset

### Training troppo lento

**Soluzioni**:
- Usa una GPU (Colab Pro o GPU locale)
- Riduci `MAX_LENGTH` se possibile
- Riduci `EPOCHS` e valuta se il modello migliora
- Usa `fp16=True` (già attivo se GPU disponibile)

### Out of Memory (OOM)

**Soluzioni**:
- Riduci `BATCH_SIZE` (già a 1, ma puoi aumentare `gradient_accumulation_steps`)
- Riduci `MAX_LENGTH`
- Usa `bitsandbytes` per quantizzazione 8-bit (richiede modifiche al codice)

---

## 📝 Note Aggiuntive

### Best Practices

1. **Dataset di training**: Almeno 50-100 esempi per risultati decenti, 200+ per risultati migliori
2. **Validazione**: Considera di dividere il dataset in train/validation (80/20)
3. **Backup**: Salva regolarmente i checkpoint durante il training
4. **Monitoraggio**: Controlla la loss durante il training per evitare overfitting

### Limitazioni

- Il modello genera testo in italiano (basato sul dataset di training)
- La qualità dipende fortemente dalla qualità e quantità del dataset
- Richiede GPU per training efficiente
- Il modello può "allucinare" informazioni non presenti nell'input

### Estensioni Future

- Aggiungere validazione del dataset
- Implementare early stopping
- Aggiungere metriche di valutazione (BLEU, ROUGE)
- Supporto per altri modelli base
- Interfaccia web per inferenza

---

## 📞 Supporto

Per problemi o domande:
1. Controlla la sezione [Troubleshooting](#troubleshooting)
2. Attiva `DEBUG=True` per vedere output dettagliati
3. Verifica che tutti i file siano presenti e nella directory corretta

---

**Versione**: 1.0  
**Ultimo aggiornamento**: 2024

