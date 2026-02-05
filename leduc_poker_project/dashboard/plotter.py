import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def generate_run_plots(run_name):
    """
    Legge il file .jsonl di una run e genera un'immagine PNG con i grafici delle metriche.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(project_root, "logs", f"{run_name}.jsonl")
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(log_file):
        print(f"File log {log_file} non trovato.")
        return

    data = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue

    metrics = [d for d in data if d.get("type") == "metrics"]
    if not metrics:
        return

    df = pd.DataFrame(metrics)
    
    # Determina la colonna delle X (episode o iteration)
    x_col = "episode" if "episode" in df.columns else "iteration"
    
    plt.figure(figsize=(12, 5), facecolor='#0a0b10')
    plt.style.use('dark_background')

    # Grafico NashConv
    if "nash_conv" in df.columns:
        plt.subplot(1, 2, 1)
        plt.plot(df[x_col], df["nash_conv"], color='#00d2ff', linewidth=2)
        plt.title(f"NashConv Convergence - {run_name}", color='white', fontsize=12)
        plt.xlabel(x_col.capitalize(), color='gray')
        plt.ylabel("NashConv", color='gray')
        plt.grid(True, linestyle='--', alpha=0.2)

    # Grafico Exploitability
    if "exploitability" in df.columns:
        plt.subplot(1, 2, 2)
        plt.plot(df[x_col], df["exploitability"], color='#ff0055', linewidth=2)
        plt.title(f"Exploitability - {run_name}", color='white', fontsize=12)
        plt.xlabel(x_col.capitalize(), color='gray')
        plt.ylabel("Exploitability", color='gray')
        plt.grid(True, linestyle='--', alpha=0.2)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"{run_name}.png")
    plt.savefig(plot_path, facecolor='#0a0b10')
    plt.close()
    print(f"Grafico salvato in: {plot_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_run_plots(sys.argv[1])
