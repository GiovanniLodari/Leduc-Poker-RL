import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from open_spiel.python import rl_environment

# Importiamo le utility che hai già definito in tournament_study
from tournament_study import load_agent

def plot_strategy_heatmap(agent_path, algo, game_name="leduc_poker"):
    env = rl_environment.Environment(game_name)
    agent = load_agent(agent_path, algo, env)
    
    # Definiamo gli stati che vogliamo analizzare (Pre-flop: J, Q, K)
    actions = ["Fold", "Call", "Raise"]
    cards = ["Jack", "Queen", "King"]
    
    # In Leduc le azioni di gioco sono sempre 3
    num_actions = 3
    strategy_matrix = np.zeros((len(cards), num_actions))
    
    print(f"Generazione Strategy Map 3x3 per {algo}...")
    
    for i in range(len(cards)):
        state = env.game.new_initial_state()
        # Distribuiamo le carte: P0 riceve la carta i, P1 riceve un'altra carta
        state.apply_action(i * 2) 
        state.apply_action(1 if i*2 == 0 else 0)
        
        # Prendiamo le probabilità dell'agente (già normalizzate da load_agent)
        probs_dict = agent.action_probabilities(state)
        for a_idx in range(num_actions):
            strategy_matrix[i, a_idx] = probs_dict.get(a_idx, 0.0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(strategy_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=actions, yticklabels=cards, vmin=0, vmax=1)
    
    plt.title(f"Strategy Heatmap - {algo.upper()}", fontsize=14, fontweight='bold')
    plt.xlabel("Azione")
    plt.ylabel("Carta in Mano")
    
    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/strategy_{algo.lower()}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Salvataggio completato: {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Percorso del checkpoint (es: leduc_poker_project/checkpoints/.../params.pkl)")
    parser.add_argument("--algo", type=str, choices=["dqn", "nfsp", "cfr", "deep_cfr"])
    args = parser.parse_args()
    
    if args.path and args.algo:
        plot_strategy_heatmap(args.path, args.algo)
    else:
        print("Esempio d'uso: python visualize_strategy.py --path [percorso] --algo [algoritmo]")
