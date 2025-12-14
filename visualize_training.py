# visualize_training.py

import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Configurazione per backend non-interattivo (utile su Colab)
matplotlib.use('Agg')  # Usa backend non-interattivo


def load_metrics(metrics_file: str) -> dict:
    """Carica le metriche dal file JSON."""
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(
            f"File metriche non trovato: {metrics_file}\n"
            f"Assicurati di aver eseguito il training prima di visualizzare le metriche."
        )
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    return metrics


def plot_training_curves(metrics: dict, output_file: str = "training_curves.png", show_plot: bool = False):
    """
    Crea grafici delle curve di training.
    
    Args:
        metrics: Dizionario con le metriche
        output_file: Nome del file di output
        show_plot: Se True, mostra il plot (utile in notebook)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Metriche del Fine-Tuning', fontsize=16, fontweight='bold')
    
    steps = metrics.get("step", [])
    epochs = metrics.get("epoch", [])
    train_loss = metrics.get("train_loss", [])
    learning_rate = metrics.get("learning_rate", [])
    
    if not steps or not train_loss:
        print("⚠️ Nessuna metrica disponibile per la visualizzazione.")
        return
    
    # Grafico 1: Loss per Step
    ax1 = axes[0, 0]
    ax1.plot(steps, train_loss, 'b-', linewidth=2, marker='o', markersize=3, label='Train Loss')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss durante il Training (per Step)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Aggiungi annotazione con ultimo valore
    if train_loss:
        last_loss = train_loss[-1]
        last_step = steps[-1]
        ax1.annotate(
            f'Finale: {last_loss:.4f}',
            xy=(last_step, last_loss),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    
    # Grafico 2: Loss per Epoch
    ax2 = axes[0, 1]
    if epochs and len(epochs) == len(train_loss):
        # Raggruppa per epoch e calcola media
        epoch_dict = {}
        for epoch, loss in zip(epochs, train_loss):
            if epoch not in epoch_dict:
                epoch_dict[epoch] = []
            epoch_dict[epoch].append(loss)
        
        unique_epochs = sorted(epoch_dict.keys())
        epoch_avg_loss = [np.mean(epoch_dict[e]) for e in unique_epochs]
        
        ax2.plot(unique_epochs, epoch_avg_loss, 'g-', linewidth=2, marker='s', markersize=5, label='Avg Loss per Epoch')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Average Loss', fontsize=12)
        ax2.set_title('Loss Media per Epoch', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Aggiungi valore finale
        if epoch_avg_loss:
            ax2.annotate(
                f'Finale: {epoch_avg_loss[-1]:.4f}',
                xy=(unique_epochs[-1], epoch_avg_loss[-1]),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
    
    # Grafico 3: Learning Rate
    ax3 = axes[1, 0]
    if learning_rate and len(learning_rate) == len(steps):
        ax3.plot(steps, learning_rate, 'r-', linewidth=2, marker='^', markersize=3, label='Learning Rate')
        ax3.set_xlabel('Step', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate durante il Training', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')  # Scala logaritmica per LR
    
    # Grafico 4: Statistiche riassuntive
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calcola statistiche
    stats_text = "📊 STATISTICHE RIASSUNTIVE\n" + "="*40 + "\n\n"
    
    if train_loss:
        stats_text += f"Loss iniziale: {train_loss[0]:.4f}\n"
        stats_text += f"Loss finale: {train_loss[-1]:.4f}\n"
        stats_text += f"Riduzione loss: {((train_loss[0] - train_loss[-1]) / train_loss[0] * 100):.2f}%\n"
        stats_text += f"Loss minima: {min(train_loss):.4f}\n"
        stats_text += f"Loss media: {np.mean(train_loss):.4f}\n\n"
    
    if steps:
        stats_text += f"Step totali: {steps[-1]}\n"
        stats_text += f"Punti di log: {len(steps)}\n\n"
    
    if epochs:
        stats_text += f"Epoche: {max(epochs):.1f}\n"
        stats_text += f"Step per epoca: ~{steps[-1] / max(epochs):.0f}\n\n"
    
    if learning_rate:
        stats_text += f"LR iniziale: {learning_rate[0]:.2e}\n"
        stats_text += f"LR finale: {learning_rate[-1]:.2e}\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Grafici salvati in: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_loss_only(metrics: dict, output_file: str = "loss_curve.png", show_plot: bool = False):
    """
    Crea un grafico semplice solo della loss.
    
    Args:
        metrics: Dizionario con le metriche
        output_file: Nome del file di output
        show_plot: Se True, mostra il plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = metrics.get("step", [])
    train_loss = metrics.get("train_loss", [])
    
    if not steps or not train_loss:
        print("⚠️ Nessuna metrica disponibile per la visualizzazione.")
        return
    
    ax.plot(steps, train_loss, 'b-', linewidth=2.5, marker='o', markersize=4, label='Train Loss', alpha=0.8)
    ax.fill_between(steps, train_loss, alpha=0.3)  # Area sotto la curva
    
    ax.set_xlabel('Step', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Curva di Loss durante il Fine-Tuning', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    
    # Aggiungi valore finale
    if train_loss:
        last_loss = train_loss[-1]
        last_step = steps[-1]
        ax.annotate(
            f'Loss finale: {last_loss:.4f}',
            xy=(last_step, last_loss),
            xytext=(20, 20),
            textcoords='offset points',
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2)
        )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Grafico salvato in: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Visualizza le metriche del fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python visualize_training.py
  python visualize_training.py --metrics-file ./fine_tuned_lora/training_metrics.json
  python visualize_training.py --simple
  python visualize_training.py --show
        """
    )
    
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="./fine_tuned_lora/training_metrics.json",
        help="Percorso al file delle metriche (default: ./fine_tuned_lora/training_metrics.json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="training_curves.png",
        help="Nome del file di output (default: training_curves.png)"
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Crea solo il grafico della loss (più semplice)"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra il grafico (utile in notebook)"
    )
    
    args = parser.parse_args()
    
    print(f"Caricamento metriche da: {args.metrics_file}")
    try:
        metrics = load_metrics(args.metrics_file)
        print(f"✓ Metriche caricate: {len(metrics.get('step', []))} punti")
    except FileNotFoundError as e:
        print(f"✗ Errore: {e}")
        return
    
    if args.simple:
        output_file = args.output.replace(".png", "_simple.png") if args.output.endswith(".png") else args.output + "_simple.png"
        plot_loss_only(metrics, output_file=output_file, show_plot=args.show)
    else:
        plot_training_curves(metrics, output_file=args.output, show_plot=args.show)
    
    print("\n✓ Visualizzazione completata!")


if __name__ == "__main__":
    main()

