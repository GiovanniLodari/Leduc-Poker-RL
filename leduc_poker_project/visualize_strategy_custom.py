import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import jax
import sys
from open_spiel.python import rl_environment
from open_spiel.python.jax import nfsp
from open_spiel.python import policy

from deepmind_nsfp_attention import AttentionNFSP, AttentionDQN, setup_gpu as setup_gpu_attention

def setup_jax_gpu():
    """Configura l'ambiente per JAX su WSL."""
    venv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nvidia_base = os.path.join(venv_base, ".venv", "lib", "python3.10", "site-packages", "nvidia")
    lib_paths = ["/usr/lib/wsl/lib"]
    if os.path.exists(nvidia_base):
        for root, dirs, files in os.walk(nvidia_base):
             if "lib" in dirs:
                lib_paths.append(os.path.join(root, "lib"))
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + current_ld if current_ld else "")
    os.environ["GPU_SETUP_DONE"] = "1"

from tournament_study import load_agent, AgentWrapper

def plot_custom_strategy(agent_path, algo, game_name="leduc_poker"):
    setup_jax_gpu()
    env = rl_environment.Environment(game_name)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    print(f"Caricamento agente {algo} da {agent_path}...")
    
    agent = None
    already_loaded = False

    if "attention" in agent_path.lower() or "attention" in algo.lower():
        if "nfsp" in algo.lower():
            agent = AttentionNFSP(0, info_state_size, num_actions, [128], 
                                reservoir_buffer_capacity=1000, 
                                replay_buffer_capacity=1000,
                                anticipatory_param=0.1)
        else:
            agent = AttentionDQN(0, info_state_size, num_actions, [128])
    elif "mlp_2_layers" in agent_path.lower() or "128x128" in agent_path.lower():
        agent = nfsp.NFSP(0, info_state_size, num_actions, [128, 128],
                         reservoir_buffer_capacity=1000,
                         replay_buffer_capacity=1000,
                         anticipatory_param=0.1)
    else:

        agent = load_agent(agent_path, algo, env)
        already_loaded = True

    if not already_loaded and agent is not None:
        with open(agent_path, "rb") as f:
            state = pickle.load(f)
        
        if "nfsp" in algo.lower():
            agent.params_avg_network = state["avg_params"]
            agent._rl_agent.params_q_network = state["q_params"]
        else:
            agent.params_q_network = state["q_params"]
        
        agent = AgentWrapper(agent, env)

    actions = ["Fold", "Call", "Raise"]
    cards = ["Jack", "Queen", "King"]
    strategy_matrix = np.zeros((len(cards), 3))
    
    print(f"Calcolo delle probabilità d'azione...")
    for i in range(len(cards)):
        state = env.game.new_initial_state()
        state.apply_action(i * 2) 
        state.apply_action(1 if i*2 == 0 else 0)
        
        probs_dict = agent.action_probabilities(state)
        for a_idx in range(3):
            strategy_matrix[i, a_idx] = probs_dict.get(a_idx, 0.0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(strategy_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=actions, yticklabels=cards, vmin=0, vmax=1)
    
    plt.title(f"Strategy Heatmap - {algo.upper()}", fontsize=14, fontweight='bold')
    plt.xlabel("Azione")
    plt.ylabel("Carta in Mano")
    
    os.makedirs("plots", exist_ok=True)
    out_path = f"plots/strategy_custom_{algo.lower()}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"File salvato in: {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "nfsp", "cfr", "deep_cfr"])
    args = parser.parse_args()
    plot_custom_strategy(args.path, args.algo)
