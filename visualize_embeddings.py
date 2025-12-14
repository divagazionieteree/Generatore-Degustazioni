# visualize_embeddings.py
# Visualizza i chunk (sequenze) come punti nel spazio vettoriale

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly non disponibile. Installalo con: pip install plotly")

# Configurazione per backend non-interattivo
matplotlib.use('Agg')


def extract_embeddings(model, tokenizer, texts: List[str], max_length: int = 512, 
                       device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    """
    Estrae le embeddings per una lista di testi.
    
    Args:
        model: Modello addestrato
        tokenizer: Tokenizer
        texts: Lista di testi da processare
        max_length: Lunghezza massima
        device: Device (cuda/cpu)
    
    Returns:
        Array numpy con le embeddings (n_samples, embedding_dim)
    """
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenizza
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            
            # Ottieni gli output del modello
            try:
                outputs = model(**enc, output_hidden_states=True)
                
                # Estrai le hidden states dall'ultimo layer
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # Per modelli con hidden_states esposti
                    last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                elif hasattr(outputs, 'last_hidden_state'):
                    # Alcuni modelli espongono direttamente last_hidden_state
                    last_hidden = outputs.last_hidden_state
                else:
                    # Per modelli PEFT, accedi al modello base
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                        base_outputs = model.base_model.model(**enc, output_hidden_states=True)
                        if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states is not None:
                            last_hidden = base_outputs.hidden_states[-1]
                        elif hasattr(base_outputs, 'last_hidden_state'):
                            last_hidden = base_outputs.last_hidden_state
                        else:
                            raise ValueError("Impossibile estrarre hidden states")
                    else:
                        raise ValueError("Impossibile estrarre hidden states")
            except Exception as e:
                # Fallback: usa i logits (meno ideale ma funziona)
                print(f"⚠️ Warning: Usando logits come fallback per embeddings: {e}")
                logits = model(**enc).logits
                # Usa la media dei logits come rappresentazione
                last_hidden = logits.mean(dim=1, keepdim=True).expand(-1, enc['input_ids'].shape[1], -1)
            
            # Mean pooling: media su tutti i token (escludendo padding)
            attention_mask = enc['attention_mask']
            # last_hidden shape: (batch, seq_len, hidden_dim)
            # attention_mask shape: (batch, seq_len)
            
            # Calcola la media pesata per l'attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_embeddings / sum_mask  # (batch, hidden_dim)
            
            embeddings_list.append(embedding.cpu().numpy())
    
    return np.vstack(embeddings_list)


def reduce_dimensions(embeddings: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
    """
    Riduce le dimensioni delle embeddings per visualizzazione.
    
    Args:
        embeddings: Array (n_samples, embedding_dim)
        method: "pca" o "tsne"
        n_components: Numero di dimensioni finali (2 o 3)
    
    Returns:
        Array ridotto (n_samples, n_components)
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings)
        print(f"✓ PCA: Varianza spiegata: {reducer.explained_variance_ratio_.sum():.2%}")
        return reduced
    elif method == "tsne":
        print("⚠️ t-SNE può richiedere tempo per dataset grandi...")
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced = reducer.fit_transform(embeddings)
        print("✓ t-SNE completato")
        return reduced
    else:
        raise ValueError(f"Metodo non supportato: {method}. Usa 'pca' o 'tsne'")


def plot_embeddings_2d(embeddings_2d: np.ndarray, labels: List[str], 
                       colors: Optional[List[str]] = None,
                       title: str = "Visualizzazione Embeddings",
                       output_file: str = "embeddings_plot.png",
                       show_plot: bool = False):
    """
    Crea un grafico scatter 2D delle embeddings.
    
    Args:
        embeddings_2d: Array (n_samples, 2)
        labels: Lista di etichette per ogni punto
        colors: Lista di colori (opzionale)
        title: Titolo del grafico
        output_file: File di output
        show_plot: Se True, mostra il plot
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colori di default
    if colors is None:
        colors = ['blue'] * len(embeddings_2d)
    
    # Crea il grafico scatter
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Aggiungi etichette (solo per i primi punti per non sovraccaricare)
    max_labels = min(20, len(labels))
    for i in range(max_labels):
        ax.annotate(
            labels[i][:30] + "..." if len(labels[i]) > 30 else labels[i],
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7
        )
    
    ax.set_xlabel('Dimensione 1', fontsize=12)
    ax.set_ylabel('Dimensione 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Grafico salvato in: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def extract_embeddings_from_training_data(model, tokenizer, json_path: str,
                                         max_samples: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Estrae embeddings dai dati di training.
    
    Args:
        model: Modello addestrato
        tokenizer: Tokenizer
        json_path: Percorso al file JSON di training
        max_samples: Numero massimo di campioni (None = tutti)
    
    Returns:
        Tuple (embeddings, labels)
    """
    from pipeline_dataset import load_degustazioni
    from fine_tuning import json_to_narrative
    from pipeline_prompts import build_prompt
    
    print(f"Caricamento dati da: {json_path}")
    degustazioni = load_degustazioni(json_path)
    
    if max_samples:
        degustazioni = degustazioni[:max_samples]
    
    texts = []
    labels = []
    
    for deg in degustazioni:
        # Filtra campioni senza ID valido
        deg_id = deg.get("id", None)
        if deg_id is None or deg_id == "Unknown" or deg_id == "":
            print(f"⚠️ Saltato campione senza ID valido: {deg.get('informazioni_base', {}).get('nome_vino', 'Unknown')}")
            continue
        
        input_data = dict(deg)
        input_data.pop("scheda_giornalistica", None)
        input_data.pop("abbinamento", None)
        
        input_text = json_to_narrative(input_data, include_output=False)
        prompt = build_prompt(input_text)
        
        texts.append(prompt)
        # Label: ID + nome vino
        nome = deg.get("informazioni_base", {}).get("nome_vino", "Unknown")
        label = f"ID: {deg_id} - {nome}"
        labels.append(label)
    
    print(f"Estrazione embeddings per {len(texts)} campioni...")
    embeddings = extract_embeddings(model, tokenizer, texts)
    print(f"✓ Embeddings estratte: shape {embeddings.shape}")
    
    return embeddings, labels


def extract_embeddings_from_inference(model, tokenizer, json_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Estrae embeddings dai dati di inferenza.
    
    Args:
        model: Modello addestrato
        tokenizer: Tokenizer
        json_path: Percorso al file JSON di input
    
    Returns:
        Tuple (embeddings, labels)
    """
    from fine_tuning import json_to_narrative
    from pipeline_prompts import build_prompt
    
    print(f"Caricamento dati inferenza da: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        degustazioni = json.load(f)
    
    texts = []
    labels = []
    
    for deg in degustazioni:
        # Filtra campioni senza ID valido
        deg_id = deg.get("id", None)
        if deg_id is None or deg_id == "Unknown" or deg_id == "":
            print(f"⚠️ Saltato campione inferenza senza ID valido: {deg.get('informazioni_base', {}).get('nome_vino', 'Unknown')}")
            continue
        
        input_data = dict(deg)
        input_data.pop("scheda_giornalistica", None)
        input_data.pop("abbinamento", None)
        
        input_text = json_to_narrative(input_data, include_output=False)
        prompt = build_prompt(input_text)
        
        texts.append(prompt)
        nome = deg.get("informazioni_base", {}).get("nome_vino", "Unknown")
        label = f"Inferenza - ID: {deg_id} - {nome}"
        labels.append(label)
    
    print(f"Estrazione embeddings per {len(texts)} campioni di inferenza...")
    embeddings = extract_embeddings(model, tokenizer, texts)
    print(f"✓ Embeddings estratte: shape {embeddings.shape}")
    
    return embeddings, labels


def plot_3d_interactive(train_embeddings: np.ndarray, train_labels: List[str],
                       infer_embeddings: Optional[np.ndarray] = None,
                       infer_labels: Optional[List[str]] = None,
                       method: str = "pca",
                       output_file: str = "embeddings_3d.html",
                       show_plot: bool = False):
    """
    Crea un grafico 3D interattivo con Plotly (zoom, rotazione, pan).
    
    Args:
        train_embeddings: Embeddings di training
        train_labels: Labels di training
        infer_embeddings: Embeddings di inferenza (opzionale)
        infer_labels: Labels di inferenza (opzionale)
        method: Metodo di riduzione dimensionale
        output_file: File HTML di output
        show_plot: Se True, mostra il plot
    """
    if not PLOTLY_AVAILABLE:
        print("✗ Plotly non disponibile. Installa con: pip install plotly")
        print("  Usando visualizzazione 2D come fallback...")
        plot_combined_embeddings(train_embeddings, train_labels, infer_embeddings, 
                                infer_labels, method, output_file.replace(".html", ".png"), show_plot)
        return
    
    # Combina tutte le embeddings
    all_embeddings = train_embeddings
    all_labels = train_labels
    all_colors = ['blue'] * len(train_embeddings)
    all_types = ['Training'] * len(train_embeddings)
    
    if infer_embeddings is not None:
        all_embeddings = np.vstack([train_embeddings, infer_embeddings])
        all_labels = train_labels + (infer_labels or [])
        all_colors = ['blue'] * len(train_embeddings) + ['red'] * len(infer_embeddings)
        all_types = ['Training'] * len(train_embeddings) + ['Inferenza'] * len(infer_embeddings)
    
    print(f"Riduzione dimensionale 3D con {method}...")
    embeddings_3d = reduce_dimensions(all_embeddings, method=method, n_components=3)
    
    # Separa training e inferenza
    train_3d = embeddings_3d[:len(train_embeddings)]
    infer_3d = embeddings_3d[len(train_embeddings):] if infer_embeddings is not None else None
    
    # Crea il grafico 3D interattivo
    fig = go.Figure()
    
    # Filtra punti senza ID valido (controlla che l'etichetta inizi con "ID: " e non sia "ID: Unknown")
    valid_train_indices = []
    valid_train_labels = []
    valid_train_3d = []
    
    for i, label in enumerate(train_labels):
        # Verifica che l'etichetta abbia un ID valido
        if label.startswith("ID: ") and not label.startswith("ID: Unknown"):
            valid_train_indices.append(i)
            valid_train_labels.append(label)
            valid_train_3d.append(train_3d[i])
    
    if len(valid_train_3d) == 0:
        print("⚠️ Nessun punto di training valido trovato!")
        return
    
    valid_train_3d = np.array(valid_train_3d)
    print(f"✓ Filtro applicato: {len(train_labels)} -> {len(valid_train_labels)} punti validi")
    
    # Aggiungi punti di training validi
    # Crea etichette più corte per il testo (solo ID per performance)
    train_text_labels = [label.split(" - ")[0] if " - " in label else label[:20] for label in valid_train_labels]
    
    fig.add_trace(go.Scatter3d(
        x=valid_train_3d[:, 0],
        y=valid_train_3d[:, 1],
        z=valid_train_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.7,
            line=dict(width=1, color='darkblue')
        ),
        text=train_text_labels[:50],  # Limita etichette per performance
        textposition="middle center",
        name='Training',
        hovertemplate='<b>%{customdata}</b><br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<br>' +
                      'Z: %{z:.3f}<extra></extra>',
        customdata=valid_train_labels  # Salva label completo (ID + nome) per hover
    ))
    
    # Aggiungi punti di inferenza se presenti (filtrati)
    if infer_3d is not None:
        # Filtra punti inferenza senza ID valido
        valid_infer_indices = []
        valid_infer_labels = []
        valid_infer_3d = []
        
        for i, label in enumerate(infer_labels or []):
            # Verifica che l'etichetta abbia un ID valido
            if label.startswith("Inferenza - ID: ") and not "ID: Unknown" in label:
                valid_infer_indices.append(i)
                valid_infer_labels.append(label)
                valid_infer_3d.append(infer_3d[i])
        
        if len(valid_infer_3d) > 0:
            valid_infer_3d = np.array(valid_infer_3d)
            print(f"✓ Filtro inferenza: {len(infer_labels or [])} -> {len(valid_infer_labels)} punti validi")
            
            infer_text_labels = [label.split(" - ")[1] if " - " in label else label[:20] for label in valid_infer_labels]
            
            fig.add_trace(go.Scatter3d(
                x=valid_infer_3d[:, 0],
                y=valid_infer_3d[:, 1],
                z=valid_infer_3d[:, 2],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color='red',
                    opacity=0.9,
                    symbol='diamond',
                    line=dict(width=2, color='darkred')
                ),
                text=infer_text_labels,
                textposition="middle center",
                name='Inferenza',
                hovertemplate='<b>%{customdata}</b><br>' +
                              'X: %{x:.3f}<br>' +
                              'Y: %{y:.3f}<br>' +
                              'Z: %{z:.3f}<extra></extra>',
                customdata=valid_infer_labels  # Salva label completo (ID + nome) per hover
            ))
        else:
            print("⚠️ Nessun punto di inferenza valido trovato dopo il filtro")
    
    # Configura layout
    fig.update_layout(
        title=dict(
            text='Visualizzazione 3D Embeddings: Training vs Inferenza',
            font=dict(size=18, color='black')
        ),
        scene=dict(
            xaxis_title='Dimensione 1',
            yaxis_title='Dimensione 2',
            zaxis_title='Dimensione 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor='white',
            aspectmode='cube'
        ),
        width=1200,
        height=900,
        hovermode='closest',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Salva come HTML interattivo
    fig.write_html(output_file)
    print(f"✓ Grafico 3D interattivo salvato in: {output_file}")
    print(f"  Apri il file nel browser per interagire (zoom, rotazione, pan)")
    
    if show_plot:
        fig.show()


def plot_combined_embeddings(train_embeddings: np.ndarray, train_labels: List[str],
                             infer_embeddings: Optional[np.ndarray] = None,
                             infer_labels: Optional[List[str]] = None,
                             method: str = "pca",
                             output_file: str = "embeddings_combined.png",
                             show_plot: bool = False):
    """
    Visualizza embeddings di training e inferenza insieme.
    
    Args:
        train_embeddings: Embeddings di training
        train_labels: Labels di training
        infer_embeddings: Embeddings di inferenza (opzionale)
        infer_labels: Labels di inferenza (opzionale)
        method: Metodo di riduzione dimensionale
        output_file: File di output
        show_plot: Se True, mostra il plot
    """
    # Combina tutte le embeddings
    all_embeddings = train_embeddings
    all_labels = train_labels
    all_colors = ['blue'] * len(train_embeddings)
    
    if infer_embeddings is not None:
        all_embeddings = np.vstack([train_embeddings, infer_embeddings])
        all_labels = train_labels + (infer_labels or [])
        all_colors = ['blue'] * len(train_embeddings) + ['red'] * len(infer_embeddings)
    
    print(f"Riduzione dimensionale con {method}...")
    embeddings_2d = reduce_dimensions(all_embeddings, method=method, n_components=2)
    
    # Separa training e inferenza
    train_2d = embeddings_2d[:len(train_embeddings)]
    infer_2d = embeddings_2d[len(train_embeddings):] if infer_embeddings is not None else None
    
    # Filtra punti senza ID valido
    valid_train_indices = [i for i, label in enumerate(train_labels) 
                          if label.startswith("ID: ") and not label.startswith("ID: Unknown")]
    valid_train_2d = train_2d[valid_train_indices]
    valid_train_labels = [train_labels[i] for i in valid_train_indices]
    
    if len(valid_train_2d) == 0:
        print("⚠️ Nessun punto di training valido trovato!")
        return
    
    print(f"✓ Filtro 2D training: {len(train_labels)} -> {len(valid_train_labels)} punti validi")
    
    # Crea il grafico
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot training (solo punti validi)
    ax.scatter(
        valid_train_2d[:, 0],
        valid_train_2d[:, 1],
        c='blue',
        alpha=0.6,
        s=100,
        edgecolors='darkblue',
        linewidths=0.5,
        label='Training',
        marker='o'
    )
    
    # Plot inferenza se presente (filtrata)
    if infer_2d is not None:
        valid_infer_indices = [i for i, label in enumerate(infer_labels or []) 
                              if label.startswith("Inferenza - ID: ") and "ID: Unknown" not in label]
        
        if len(valid_infer_indices) > 0:
            valid_infer_2d = infer_2d[valid_infer_indices]
            valid_infer_labels = [(infer_labels or [])[i] for i in valid_infer_indices]
            print(f"✓ Filtro 2D inferenza: {len(infer_labels or [])} -> {len(valid_infer_labels)} punti validi")
            
            ax.scatter(
                valid_infer_2d[:, 0],
                valid_infer_2d[:, 1],
                c='red',
                alpha=0.8,
                s=150,
                edgecolors='darkred',
                linewidths=1,
                label='Inferenza',
                marker='*'
            )
            
            # Aggiungi etichette inferenza
            for i in range(len(valid_infer_labels)):
                ax.annotate(
                    valid_infer_labels[i][:25] + "..." if len(valid_infer_labels[i]) > 25 else valid_infer_labels[i],
                    (valid_infer_2d[i, 0], valid_infer_2d[i, 1]),
                    fontsize=8,
                    alpha=0.8,
                    color='red',
                    fontweight='bold'
                )
    
    # Aggiungi etichette per alcuni punti training
    for i in range(min(15, len(valid_train_labels))):
        ax.annotate(
            valid_train_labels[i][:25] + "..." if len(valid_train_labels[i]) > 25 else valid_train_labels[i],
            (valid_train_2d[i, 0], valid_train_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
            color='blue'
        )
    
    ax.set_xlabel('Dimensione 1', fontsize=13)
    ax.set_ylabel('Dimensione 2', fontsize=13)
    ax.set_title('Visualizzazione Embeddings: Training vs Inferenza', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Grafico combinato salvato in: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Visualizza embeddings dei chunk come punti su un grafico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Solo training
  python visualize_embeddings.py --train-data degustazioni.json --model-path ./fine_tuned_lora
  
  # Training + Inferenza
  python visualize_embeddings.py --train-data degustazioni.json --infer-data input.json --model-path ./fine_tuned_lora
  
  # Grafico 3D interattivo
  python visualize_embeddings.py --train-data degustazioni.json --infer-data input.json --3d --model-path ./fine_tuned_lora
  
  # Con t-SNE invece di PCA (3D)
  python visualize_embeddings.py --train-data degustazioni.json --method tsne --3d --model-path ./fine_tuned_lora
        """
    )
    
    parser.add_argument(
        "--train-data",
        type=str,
        default="degustazioni.json",
        help="File JSON con dati di training"
    )
    
    parser.add_argument(
        "--infer-data",
        type=str,
        default=None,
        help="File JSON con dati di inferenza (opzionale)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="./fine_tuned_lora",
        help="Percorso al modello LoRA addestrato"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/phi-2",
        help="Nome del modello base"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "tsne"],
        help="Metodo di riduzione dimensionale (default: pca)"
    )
    
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="three_d",
        help="Crea un grafico 3D interattivo (richiede plotly)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Numero massimo di campioni da processare (None = tutti)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings_combined.png",
        help="File di output"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra il grafico (utile in notebook)"
    )
    
    args = parser.parse_args()
    
    # Carica modello
    print(f"Caricamento modello base: {args.base_model}")
    print(f"Caricamento LoRA da: {args.model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base, args.model_path)
        model.eval()
        print("✓ Modello caricato")
    except Exception as e:
        print(f"✗ Errore nel caricamento del modello: {e}")
        return
    
    # Estrai embeddings di training
    try:
        train_embeddings, train_labels = extract_embeddings_from_training_data(
            model, tokenizer, args.train_data, max_samples=args.max_samples
        )
    except Exception as e:
        print(f"✗ Errore nell'estrazione embeddings training: {e}")
        return
    
    # Estrai embeddings di inferenza se fornito
    infer_embeddings = None
    infer_labels = None
    if args.infer_data and os.path.exists(args.infer_data):
        try:
            infer_embeddings, infer_labels = extract_embeddings_from_inference(
                model, tokenizer, args.infer_data
            )
        except Exception as e:
            print(f"⚠️ Errore nell'estrazione embeddings inferenza: {e}")
            print("  Continuo solo con i dati di training...")
    
    # Visualizza
    if args.three_d:
        # Output HTML per grafico 3D
        output_file = args.output if args.output.endswith('.html') else args.output.replace('.png', '.html')
        plot_3d_interactive(
            train_embeddings,
            train_labels,
            infer_embeddings,
            infer_labels,
            method=args.method,
            output_file=output_file,
            show_plot=args.show
        )
    else:
        # Output PNG per grafico 2D
        plot_combined_embeddings(
            train_embeddings,
            train_labels,
            infer_embeddings,
            infer_labels,
            method=args.method,
            output_file=args.output,
            show_plot=args.show
        )
    
    print("\n✓ Visualizzazione completata!")


if __name__ == "__main__":
    main()

