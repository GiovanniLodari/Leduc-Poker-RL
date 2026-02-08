import os
import argparse
import numpy as np
import pickle
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python import policy
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import visualization
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patheffects as PathEffects

def _improved_draw_network(self):
    """Versione migliorata del disegno della rete per Alpha-Rank."""
    import networkx as nx
    self.fig.clf()
    ax = self.fig.add_subplot(111)
    
    node_sizes = 5000 if self.num_populations == 1 else 15000
    vmin, vmax = 0, np.max(self.pi) + 0.1
    
    nx.draw_networkx_nodes(
        self.g, self.pos, ax=ax, node_size=node_sizes, node_color=self.node_colors,
        edgecolors="k", cmap=plt.cm.Blues, vmin=vmin, vmax=vmax, linewidths=1.5)
    
    nx.draw_networkx_edges(
        self.g, self.pos, ax=ax, node_size=node_sizes, arrowstyle="->", arrowsize=15,
        edge_color=self.edge_colors, width=2, alpha=0.5,
        connectionstyle='arc3,rad=0.2')
    
    nx.draw_networkx_edge_labels(self.g, self.pos, edge_labels=self.edge_labels, 
                                 ax=ax, font_size=7, label_pos=0.3)
    
    for i in self.g.nodes:
        x, y = self.pos[i]
        label = self.state_labels[i]

        node_text = f"{label}\n{self.pi[i]:.2f}"
        txt = ax.text(x, y, node_text, ha="center", va="center", 
                     fontsize=9, fontweight='bold', color='black')

        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="white", alpha=0.9)])
    
    ax.set_axis_off()

alpharank_visualizer.NetworkPlot._draw_network = _improved_draw_network

def setup_gpu():
    """Configura l'ambiente per usare la GPU correttamente in WSL/Linux."""
    if os.environ.get("GPU_SETUP_DONE") == "1":
        return

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    import sys
    import glob

    venv_site_packages = next((p for p in sys.path if 'site-packages' in p and '.venv' in p), None)
    
    lib_paths = ["/usr/lib/wsl/lib"]
    if venv_site_packages:

        nvidia_libs = glob.glob(os.path.join(venv_site_packages, "nvidia/*/lib"))
        lib_paths.extend(nvidia_libs)
    
    if lib_paths:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + current_ld if current_ld else "")
        os.environ["GPU_SETUP_DONE"] = "1"

        os.execv(sys.executable, [sys.executable] + sys.argv)

setup_gpu()

try:
    import jax
    print(f"JAX Device: {jax.devices()[0]}")
except Exception as e:
    print(f"Nota: JAX non ha rilevato GPU (usa CPU): {e}")

from open_spiel.python.jax import dqn
from open_spiel.python.jax import nfsp
from open_spiel.python.jax import deep_cfr

class AgentWrapper(policy.Policy):
    """Wrapper to make RL agents compatible with OpenSpiel Policy interface."""
    def __init__(self, agent, env):
        super().__init__(env.game, [0, 1])
        self.agent = agent
        self.env = env

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        obs = {"info_state": [None, None], "legal_actions": [None, None], "current_player": cur_player}
        obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        obs["legal_actions"][cur_player] = legal_actions
        time_step = rl_environment.TimeStep(observations=obs, rewards=None, discounts=None, step_type=None)
        
        original_player_id = getattr(self.agent, "player_id", None)
        if original_player_id is not None:
            self.agent.player_id = cur_player
            
        if hasattr(self.agent, "temp_mode_as"):
             with self.agent.temp_mode_as(nfsp.MODE.average_policy):
                p = self.agent.step(time_step, is_evaluation=True).probs
        else:
            p = self.agent.step(time_step, is_evaluation=True).probs
        
        if original_player_id is not None:
            self.agent.player_id = original_player_id
        
        legals_p = np.array([p[a] for a in legal_actions], dtype=np.float64)
        legals_p = np.clip(legals_p, 0, 1)
        sum_p = np.sum(legals_p)
        if sum_p > 0:
            legals_p /= sum_p
        else:
            legals_p = np.ones(len(legal_actions)) / len(legal_actions)
            
        return {a: float(prob) for a, prob in zip(legal_actions, legals_p)}

def load_agent(path, algo, env):
    """Utility to load an agent from a path."""
    print(f"Caricamento agente {algo} da {path}...")
    num_actions = env.action_spec()["num_actions"]
    info_state_size = env.observation_spec()["info_state"][0]
    
    if algo == "dqn":
        agent = dqn.DQN(0, info_state_size, num_actions, [128])

        with open(path, "rb") as f:
            state = pickle.load(f)
            try:
                agent.params_q_network = state["q_params"]
            except KeyError:
                raise ValueError(f"Errore: Il file {path} non sembra contenere un modello DQN (chiave 'q_params' mancante).")
    elif algo == "nfsp":
        agent = nfsp.NFSP(0, info_state_size, num_actions, [128], reservoir_buffer_capacity=100, anticipatory_param=0.1)
        with open(path, "rb") as f:
            state = pickle.load(f)
            try:
                agent.params_avg_network = state["avg_params"]
                agent._rl_agent.params_q_network = state["q_params"]
            except KeyError:
                raise ValueError(f"Errore: Il file {path} non sembra contenere un modello NFSP (chiavi 'avg_params' o 'q_params' mancanti). Forse è un modello DQN?")
    elif algo == "deep_cfr":

        agent = deep_cfr.DeepCFRSolver(
            env.game,
            policy_network_layers=[128, 128],
            advantage_network_layers=[128, 128],
            num_iterations=1,
            num_traversals=1
        )
        with open(path, "rb") as f:
            state = pickle.load(f)
            agent._params_adv_network = state["adv_params"]
            agent._params_policy_network = state["policy_params"]
        return agent
    elif algo == "cfr":
        with open(path, "rb") as f:
            solver = pickle.load(f)

            return solver.average_policy()
    else:
        raise ValueError(f"Algoritmo {algo} non supportato")
    
    return AgentWrapper(agent, env)

def play_matchup(game, policy0, policy1, num_episodes=100, report_every=500):
    """Gioca due politiche l'una contro l'altra e restituisce la storia dei payoff medi."""
    rewards = []
    history = []
    
    for ep in range(1, num_episodes + 1):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                cur_player = state.current_player()
                pol = policy0 if cur_player == 0 else policy1
                probs_dict = pol.action_probabilities(state)
                actions = list(probs_dict.keys())
                probs = np.array(list(probs_dict.values()), dtype=np.float64)

                sum_p = np.sum(probs)
                if sum_p > 0:
                    probs /= sum_p
                else:
                    probs = np.ones(len(actions)) / len(actions)
                action = np.random.choice(actions, p=probs)
                state.apply_action(action)
        
        rewards.append(state.returns()[0])
        
        if ep % report_every == 0 or ep == num_episodes:
            history.append((ep, np.mean(rewards)))
            
    return np.mean(rewards), history

def run_tournament(args):
    env = rl_environment.Environment(args.game)
    
    if not args.agents:
        print("Nessun agente specificato. Esempio: --agents path1:dqn:DQN1 path2:nfsp:NFSP1")
        return

    agent_configs = []
    for a_str in args.agents:
        parts = a_str.split(":")
        if len(parts) >= 3:
            path, algo, label = parts[0], parts[1], parts[2]
            nconv = float(parts[3]) if len(parts) > 3 else 0.0
            agent_configs.append((path, algo, label, nconv))
        else:
            print(f"Formato agente non valido: {a_str}. Usa path:algo:label[:nashconv]")

    num_agents = len(agent_configs)
    raw_payoff_matrix = np.zeros((num_agents, num_agents))
    payoff_evolution = {}
    
    if not agent_configs:
        print("\n\033[91mErrore: Nessun agente caricato correttamente. Verifica il formato 'path:algo:label'.\033[0m")
        return

    print(f"Inizio Torneo tra {num_agents} agenti (Matchup a posizioni invertite)...")
    report_interval = 500
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            p_i = load_agent(agent_configs[i][0], agent_configs[i][1], env)
            p_j = load_agent(agent_configs[j][0], agent_configs[j][1], env)
            
            rew_i0, hist_i0 = play_matchup(env.game, p_i, p_j, num_episodes=args.episodes // 2, report_every=report_interval)

            rew_j0, hist_j0 = play_matchup(env.game, p_j, p_i, num_episodes=args.episodes // 2, report_every=report_interval)
            
            raw_payoff_matrix[i, j] = rew_i0
            raw_payoff_matrix[j, i] = rew_j0
            
            net_hist = []
            for (ep_i, val_i), (ep_j, val_j) in zip(hist_i0, hist_j0):
                net_hist.append((ep_i + ep_j, (val_i - val_j) / 2))
            payoff_evolution[(i, j)] = net_hist

            net_gain = (rew_i0 - rew_j0) / 2
            print(f"Matchup {agent_configs[i][2]} vs {agent_configs[j][2]}: Net Gain {net_gain:+.4f}")

    payoff_matrix = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                payoff_matrix[i, j] = (raw_payoff_matrix[i, j] - raw_payoff_matrix[j, i]) / 2
    
    alphas_to_study = [0.25, 0.5, 0.75, 1.0]
    multi_alpha_results = {}
    
    print("\nCalcolo Alpha-Rank per molteplici valori di Alpha...")
    payoff_tables = [payoff_matrix]
    for a in alphas_to_study:
        _, _, pi_a, _, _ = alpharank.compute(payoff_tables, alpha=a)
        multi_alpha_results[a] = pi_a
    
    _, _, pi, _, _ = alpharank.compute(payoff_tables, alpha=args.alpha)
    print("\nClassifica Finale Alpha-Rank (alpha={}):".format(args.alpha))
    labels = [c[2] for c in agent_configs]
    nconvs = [c[3] for c in agent_configs]
    sorted_indices = np.argsort(pi)[::-1]
    for idx in sorted_indices:
        print(f"{labels[idx]:15s}: {pi[idx]:.4f} (NashConv: {nconvs[idx]:.4f})")

    print("\nLeaderboard per Payoff Medio (Chips guadagnate per mano contro tutti):")
    avg_payoffs = np.mean(payoff_matrix, axis=1)
    leaderboard_indices = np.argsort(avg_payoffs)[::-1]
    for idx in leaderboard_indices:
        print(f"{labels[idx]:15s}: {avg_payoffs[idx]:+.4f} chips/mano")

    if args.plot:
        print("Generazione visualizzazioni avanzate (OpenSpiel EGT Framework)...")
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(labels))
        width = 0.25
        for idx, a in enumerate(alphas_to_study):
            plt.bar(x + (idx-1)*width, multi_alpha_results[a], width, label=f'Alpha={a}')
        
        new_labels = [f"{l}\n(NC: {nc:.3f})" for l, nc in zip(labels, nconvs)]
        
        plt.xlabel('Strategie degli Agenti (con NashConv)')
        plt.ylabel('Probabilità Stazionaria (Alpha-Rank)')
        plt.title('Alpha-Rank vs Intensità di Selezione (Fisica Evolutiva)', fontsize=14, fontweight='bold')
        plt.xticks(x, new_labels, rotation=0)
        plt.legend(title="Parametro Alpha")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.savefig("plots/multi_alpha_analysis.png", bbox_inches='tight', dpi=150)
        plt.close()

        print("Generazione grafici di transizione Markov per Alpha-Rank...")
        for a in alphas_to_study:
            rhos, rho_m, pi_a, _, _ = alpharank.compute(payoff_tables, alpha=a)
            
            network_plot = alpharank_visualizer.NetworkPlot(
                payoff_tables, rhos, rho_m, pi_a, labels)
            
            network_plot.first_run = False 
            
            fig = plt.figure(figsize=(12, 12))
            network_plot.fig = fig
            network_plot.compute_and_draw_network()
            
            fig.suptitle(f"Analisi Evolutiva della Dominanza (Alpha={a})", fontsize=16, fontweight='bold', y=0.92)
            fig.savefig(f"plots/markov_transition_alpha_{a}.png", bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"- markov_transition_alpha_{a}.png")

        print("Generazione evoluzione temporale del payoff...")
        plt.figure(figsize=(12, 7))
        for (i, j), hist in payoff_evolution.items():
            eps, vals = zip(*hist)
            plt.plot(eps, vals, label=f'{labels[i]} vs {labels[j]}', linewidth=2, alpha=0.8)
        
        plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
        plt.xlabel('Episodi (Hands)', fontsize=12)
        plt.ylabel('Payoff Medio Netto (M_ij)', fontsize=12)
        plt.title('Stabilità e Convergenza della Classifica Payoff', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.15)
        plt.savefig("plots/payoff_history_evolution.png", bbox_inches='tight', dpi=150)
        plt.close()

        if num_agents == 3:
            print("Generazione Phase Portrait (Replicator Dynamics Simplesso)...")
            payoff_matrix_egt = payoff_matrix
            dyn = dynamics.SinglePopulationDynamics(payoff_matrix_egt, dynamics.replicator)
            
            fig = plt.figure(figsize=(10, 8))

            ax = fig.add_subplot(111, projection='3x3')
            ax.streamplot(dyn)
            ax.set_labels(labels)
            plt.title("Mappa delle Traiettorie Evolutive (3-Strategy Simplex)", fontsize=14, fontweight='bold')
            plt.savefig("plots/egt_phase_portrait_advanced.png", bbox_inches='tight', dpi=150)
            plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(payoff_matrix, annot=True, fmt=".3f", cmap="RdYlGn", 
                    xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Net Gain'})
        plt.title("Matrice Finale dei Payoff (Symmetric Zero-Sum)", fontsize=14, fontweight='bold')
        plt.savefig("plots/final_payoff_matrix.png", bbox_inches='tight', dpi=150)
        plt.close()

        print("\nAnalisi avanzata completata. Report salvati in 'plots/'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--episodes", type=int, default=10000, help="Numero di mani per ogni matchup")
    parser.add_argument("--alpha", type=float, default=1.0, help="Selection intensity for Alpha-Rank")
    parser.add_argument("--agents", nargs="+", help="Lista di agenti: path:algo:label")
    parser.add_argument("--plot", action="store_true", help="Genera grafici avanzati")
    args = parser.parse_args()
    
    os.makedirs("plots", exist_ok=True)
    run_tournament(args)

