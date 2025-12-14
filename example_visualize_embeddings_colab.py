# example_visualize_embeddings_colab.py
# Esempio di utilizzo della visualizzazione embeddings su Google Colab

from visualize_embeddings import (
    extract_embeddings_from_training_data,
    extract_embeddings_from_inference,
    plot_combined_embeddings
)
from pipeline_infer import load_model
from IPython.display import Image, display

# Carica il modello
BASE_MODEL = "microsoft/phi-2"
LORA_PATH = "./fine_tuned_lora"

tokenizer, model = load_model(BASE_MODEL, LORA_PATH)

# Estrai embeddings dai dati di training
train_embeddings, train_labels = extract_embeddings_from_training_data(
    model, tokenizer, "degustazioni.json", max_samples=50  # Limita a 50 per velocità
)

# Estrai embeddings dai dati di inferenza (opzionale)
infer_embeddings, infer_labels = extract_embeddings_from_inference(
    model, tokenizer, "input.json"
)

# Opzione 1: Grafico 3D interattivo (consigliato!)
try:
    from visualize_embeddings import plot_3d_interactive
    
    plot_3d_interactive(
        train_embeddings,
        train_labels,
        infer_embeddings,
        infer_labels,
        method="pca",  # o "tsne" per risultati migliori ma più lento
        output_file="embeddings_3d.html",
        show_plot=True
    )
except ImportError:
    print("⚠️ Funzione plot_3d_interactive non trovata. Usando codice inline...")
    # Codice inline come fallback
    import numpy as np
    import plotly.graph_objects as go
    from visualize_embeddings import reduce_dimensions
    
    # Combina embeddings
    all_embeddings = np.vstack([train_embeddings, infer_embeddings]) if infer_embeddings is not None else train_embeddings
    all_labels = train_labels + (infer_labels or []) if infer_embeddings is not None else train_labels
    
    # Riduzione dimensionale 3D
    print("Riduzione dimensionale 3D...")
    embeddings_3d = reduce_dimensions(all_embeddings, method="pca", n_components=3)
    
    # Separa
    train_3d = embeddings_3d[:len(train_embeddings)]
    infer_3d = embeddings_3d[len(train_embeddings):] if infer_embeddings is not None else None
    
    # Crea grafico
    fig = go.Figure()
    
    # Filtra punti training senza ID valido
    valid_train_indices = [i for i, label in enumerate(train_labels) 
                          if label.startswith("ID: ") and not label.startswith("ID: Unknown")]
    if len(valid_train_indices) == 0:
        print("⚠️ Nessun punto di training valido trovato!")
    else:
        valid_train_3d = train_3d[valid_train_indices]
        valid_train_labels = [train_labels[i] for i in valid_train_indices]
        print(f"✓ Filtro training: {len(train_labels)} -> {len(valid_train_labels)} punti validi")
    
        # Training - mostra solo ID nel testo, ID+nome nell'hover
        train_text_labels = [label.split(" - ")[0] if " - " in label else label[:20] for label in valid_train_labels]
        fig.add_trace(go.Scatter3d(
            x=valid_train_3d[:, 0], y=valid_train_3d[:, 1], z=valid_train_3d[:, 2],
            mode='markers+text',
            marker=dict(size=8, color='blue', opacity=0.7, line=dict(width=1, color='darkblue')),
            text=train_text_labels[:50],  # Limita per performance
            textposition="middle center",
            name='Training',
            hovertemplate='<b>%{customdata}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
            customdata=valid_train_labels  # ID + nome completo per hover
        ))
    
        # Inferenza - filtra punti senza ID valido
        if infer_3d is not None:
            valid_infer_indices = [i for i, label in enumerate(infer_labels or []) 
                                  if label.startswith("Inferenza - ID: ") and "ID: Unknown" not in label]
            
            if len(valid_infer_indices) > 0:
                valid_infer_3d = infer_3d[valid_infer_indices]
                valid_infer_labels = [(infer_labels or [])[i] for i in valid_infer_indices]
                print(f"✓ Filtro inferenza: {len(infer_labels or [])} -> {len(valid_infer_labels)} punti validi")
                
                infer_text_labels = [label.split(" - ")[1] if " - " in label else label[:20] for label in valid_infer_labels]
                fig.add_trace(go.Scatter3d(
                    x=valid_infer_3d[:, 0], y=valid_infer_3d[:, 1], z=valid_infer_3d[:, 2],
                    mode='markers+text',
                    marker=dict(size=12, color='red', opacity=0.9, symbol='diamond', line=dict(width=2, color='darkred')),
                    text=infer_text_labels,
                    textposition="middle center",
                    name='Inferenza',
                    hovertemplate='<b>%{customdata}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
                    customdata=valid_infer_labels  # ID + nome completo per hover
                ))
            else:
                print("⚠️ Nessun punto di inferenza valido trovato dopo il filtro")
    
    fig.update_layout(
        title='Visualizzazione 3D Embeddings: Training vs Inferenza',
        scene=dict(
            xaxis_title='Dimensione 1',
            yaxis_title='Dimensione 2',
            zaxis_title='Dimensione 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1200,
        height=900
    )
    
    fig.write_html("embeddings_3d.html")
    fig.show()
    print("✓ Grafico 3D salvato in embeddings_3d.html")

# Il grafico HTML si aprirà automaticamente nel browser
# Puoi zoomare, ruotare e fare pan interagendo con il mouse

# Opzione 2: Grafico 2D statico (alternativa)
# from visualize_embeddings import plot_combined_embeddings
# plot_combined_embeddings(
#     train_embeddings,
#     train_labels,
#     infer_embeddings,
#     infer_labels,
#     method="pca",
#     output_file="embeddings_combined.png",
#     show_plot=True
# )
# display(Image("embeddings_combined.png"))

