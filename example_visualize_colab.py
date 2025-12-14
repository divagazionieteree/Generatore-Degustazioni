# example_visualize_colab.py
# Esempio di utilizzo della visualizzazione su Google Colab

# Dopo aver eseguito il training, esegui questo codice per visualizzare le metriche

from visualize_training import load_metrics, plot_training_curves, plot_loss_only

# Carica le metriche
metrics_file = "./fine_tuned_lora/training_metrics.json"
metrics = load_metrics(metrics_file)

# Opzione 1: Grafico completo con 4 pannelli
plot_training_curves(metrics, output_file="training_curves.png", show_plot=True)

# Opzione 2: Grafico semplice solo della loss
# plot_loss_only(metrics, output_file="loss_curve.png", show_plot=True)

# Visualizza le metriche nel notebook
from IPython.display import Image, display
display(Image("training_curves.png"))

