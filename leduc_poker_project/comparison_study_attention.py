import os
import sys
import argparse
import subprocess
import time
import socket
import signal
import pickle

def setup_gpu():
    """Configura l'ambiente per abilitare la GPU su WSL/Linux prima degli import pesanti."""
    if os.environ.get("GPU_SETUP_DONE") == "1":
        return
        
    venv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nvidia_base = os.path.join(venv_base, ".venv", "lib", "python3.10", "site-packages", "nvidia")
    
    if os.path.exists(nvidia_base):
        lib_paths = []
        for root, dirs, files in os.walk(nvidia_base):
            if "lib" in dirs:
                lib_paths.append(os.path.join(root, "lib"))
        
        if lib_paths:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + current_ld if current_ld else "")
            os.environ["GPU_SETUP_DONE"] = "1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
            os.execv(sys.executable, [sys.executable] + sys.argv)

# Esegui setup immediato
setup_gpu()

import numpy as np
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, cfr
from open_spiel.python.jax import dqn
from open_spiel.python.jax import nfsp 
from open_spiel.python.jax import deep_cfr
import jax
import jax.numpy as jnp
import haiku as hk

# Silenzia avvisi tecnici XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dashboard.logger import LocalLogger
from dashboard.plotter import generate_run_plots

class AttentionBlock(hk.Module):
    """Modulo di Self-Attention per vettori di informazione piatti."""
    def __init__(self, model_size=128, num_heads=4, name=None):
        super().__init__(name=name)
        self.model_size = model_size
        self.num_heads = num_heads

    def __call__(self, x):
        x = hk.Linear(self.model_size)(x)
        x_seq = x[:, None, :]
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_size=self.model_size // self.num_heads, 
            model_size=self.model_size,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        )
        x_attn = attn(x_seq, x_seq, x_seq)
        x = x + x_attn.squeeze(1)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        return x

class AttentionDQN(dqn.DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def network(x):
            x = AttentionBlock()(x)
            mlp = hk.nets.MLP(self._layer_sizes + [self._num_actions])
            return mlp(x)
        self.hk_network = hk.without_apply_rng(hk.transform(network))
        self.hk_network_apply = jax.jit(self.hk_network.apply)
        rng = jax.random.PRNGKey(42)
        self._create_networks(rng, args[1] if len(args)>1 else kwargs['state_representation_size'])

class AttentionNFSP(nfsp.NFSP):
    def __init__(self, *args, **kwargs):
        original_dqn = dqn.DQN
        dqn.DQN = AttentionDQN
        try:
            super().__init__(*args, **kwargs)
        finally:
            dqn.DQN = original_dqn
        def network(x):
            x = AttentionBlock()(x)
            mlp = hk.nets.MLP(self._layer_sizes + [self._num_actions])
            return mlp(x)
        self.hk_avg_network = hk.without_apply_rng(hk.transform(network))
        def avg_network_policy(param, info_state):
            action_values = self.hk_avg_network.apply(param, info_state)
            action_probs = jax.nn.softmax(action_values, axis=1)
            return action_values, action_probs
        self._avg_network_policy = jax.jit(avg_network_policy)
        rng = jax.random.PRNGKey(42)
        x = jnp.ones([1, args[1] if len(args)>1 else kwargs['state_representation_size']])
        self.params_avg_network = self.hk_avg_network.init(rng, x)

class AttentionDeepCFR(deep_cfr.DeepCFRSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def base_network(x, layers):
            x = AttentionBlock()(x)
            x = hk.nets.MLP(layers[:-1], activate_final=True)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = hk.Linear(layers[-1])(x)
            x = jax.nn.relu(x)
            x = hk.Linear(self._num_actions)(x)
            return x
        def adv_network(x, mask):
            x = base_network(x, self._advantage_network_layers)
            x = mask * x
            return x
        def policy_network(x, mask):
            x = base_network(x, self._policy_network_layers)
            x = jnp.where(mask == 1, x, -10e20)
            x = jax.nn.softmax(x)
            return x
        self._hk_adv_network = hk.without_apply_rng(hk.transform(adv_network))
        self._hk_policy_network = hk.without_apply_rng(hk.transform(policy_network))
        x, mask = (jnp.ones([1, self._embedding_size]), jnp.ones([1, self._num_actions]))
        self._params_adv_network = [self._hk_adv_network.init(self._next_rng_key(), x, mask) for _ in range(self._num_players)]
        self._params_policy_network = self._hk_policy_network.init(self._next_rng_key(), x, mask)
        self._opt_adv_state = [self._opt_adv_init(params) for params in self._params_adv_network]
        self._opt_policy_state = self._opt_policy_init(self._params_policy_network)

def handle_exit(signum, frame, run_name):
    print(f"\nStudio interrotto. Salvataggio finale e generazione grafici...")
    generate_run_plots(run_name)
    sys.exit(0)

class OS_Policy_Wrapper(policy.Policy):
    """Wrapper ottimizzato con cache per calcolare NashConv."""
    def __init__(self, env, agents, mode=None):
        super().__init__(env.game, [0, 1])
        self._agents = agents
        self._mode = mode
        self._cache = {}

    def action_probabilities(self, state, player_id=None):
        infostate_key = state.information_state_string()
        if infostate_key in self._cache:
            return self._cache[infostate_key]

        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        obs = {"info_state": [None, None], "legal_actions": [None, None], "current_player": cur_player}
        obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        obs["legal_actions"][cur_player] = legal_actions
        time_step = rl_environment.TimeStep(observations=obs, rewards=None, discounts=None, step_type=None)
        if self._mode is not None:
            with self._agents[cur_player].temp_mode_as(self._mode):
                p = self._agents[cur_player].step(time_step, is_evaluation=True).probs
        else:
            p = self._agents[cur_player].step(time_step, is_evaluation=True).probs
        
        # --- FIX: Robust Normalization ---
        # Prendiamo solo le azioni legali e assicuriamoci che siano >= 0
        legals_p = np.array([p[a] for a in legal_actions], dtype=np.float64)
        legals_p = np.clip(legals_p, 0, 1)
        
        sum_p = np.sum(legals_p)
        if sum_p > 0:
            legals_p /= sum_p
        else:
            # Fallback se tutto è zero (molto raro): uniforme sulle legali
            legals_p = np.ones(len(legal_actions)) / len(legal_actions)
            
        probs_dict = {a: float(prob) for a, prob in zip(legal_actions, legals_p)}
        self._cache[infostate_key] = probs_dict
        return probs_dict

    def clear_cache(self):
        self._cache = {}


def start_dashboard():
    """Avvia il server dashboard in background se non è attivo."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', 8000)) == 0:
            print(f"\033[94mDashboard attiva: http://localhost:8000\033[0m")
            return 
            
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(base_dir, "dashboard", "server.py")
    python_exec = sys.executable

    print(f"Avvio Dashboard Locale su http://localhost:8000...")
    subprocess.Popen([python_exec, server_script], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
    print(f"\033[94mDashboard attiva: http://localhost:8000\033[0m")

def run_comparison(args):
    print(f"JAX Device: {jax.devices()[0]}")
    start_dashboard()
    local_logger = LocalLogger(f"{args.algo}_attention")
    run_name = local_logger.run_name
    print(f"Inizio run: \033[92m{run_name}\033[0m")
    env = rl_environment.Environment(args.game)
    checkpoint_dir = os.path.join("leduc_poker_project/checkpoints", "research", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Configura segnale di uscita
    signal.signal(signal.SIGINT, lambda s, f: handle_exit(s, f, run_name))
    
    # --- ALGORITMI TABULARI (CFR) ---
    if args.algo == "cfr":
        print("--- CFR COMPARISON STUDY ---")
        solver = cfr.CFRSolver(env.game)
        for i in range(args.iterations):
            solver.evaluate_and_update_policy()
            if i % args.eval_every == 0:
                conv = exploitability.exploitability(env.game, solver.average_policy())
                print(f"Iter {i:7d} | Exploitability: {conv:.6f} | NashConv: {conv*2:.6f}")
                local_logger.log({"iteration": i, "exploitability": float(conv), "nash_conv": float(conv * 2)})
            
            if i > 0 and i % args.save_every == 0:
                path = os.path.join(checkpoint_dir, f"cfr_solver_{i}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(solver, f)
                print(f"Salvato checkpoint CFR: {path}")

    # --- DEEP CFR (Iterative Approach) ---
    elif args.algo == "deep_cfr":
        print("--- DEEP CFR ATTENTION STUDY ---")
        solver = AttentionDeepCFR(
            env.game,
            policy_network_layers=[args.hidden, args.hidden],
            advantage_network_layers=[args.hidden, args.hidden],
            num_iterations=args.iterations,
            num_traversals=args.traversals,
            learning_rate=args.lr,
            batch_size_advantage=args.batch_size,
            batch_size_strategy=args.batch_size,
            memory_capacity=args.reservoir
        )

        for i in range(args.iterations):
            # Esegui una singola iterazione di Deep CFR manually per monitorare la NashConv
            for p in range(env.game.num_players()):
                for _ in range(args.traversals):
                    solver._traverse_game_tree(solver._root_node, p)
                solver._reinitialize_advantage_network(p)
                solver._learn_advantage_network(p)
            solver._iteration += 1

            if i % args.eval_every == 0:
                # Per Deep CFR, dobbiamo addestrare brevemente la policy network per ottenere la politica media
                solver._learn_strategy_network()
                conv = exploitability.exploitability(env.game, solver)
                print(f"Iter {i:7d} | Exploitability: {conv:.6f} | NashConv: {conv*2:.6f}")
                local_logger.log({"iteration": i, "exploitability": float(conv), "nash_conv": float(conv * 2)})

            if i > 0 and i % args.save_every == 0:
                path = os.path.join(checkpoint_dir, f"deep_cfr_{i}")
                os.makedirs(path, exist_ok=True)
                state = {
                    "adv_params": solver._params_adv_network,
                    "policy_params": solver._params_policy_network,
                }
                with open(os.path.join(path, "params.pkl"), "wb") as f:
                    pickle.dump(state, f)
                print(f"Salvato checkpoint Deep CFR (params): {path}")


    # --- ALGORITMI DEEP RL (NFSP, DQN) ---
    else:
        print(f"--- {args.algo.upper()} COMPARISON STUDY ---")
        num_actions = env.action_spec()["num_actions"]
        info_state_size = env.observation_spec()["info_state"][0]
        
        if args.algo == "nfsp":
            agents = [AttentionNFSP(idx, info_state_size, num_actions, [args.hidden], 
                               reservoir_buffer_capacity=args.reservoir,
                               anticipatory_param=args.anticipatory) for idx in range(2)]
            eval_policy = OS_Policy_Wrapper(env, agents, nfsp.MODE.average_policy)
        else: # dqn
            agents = [AttentionDQN(idx, info_state_size, num_actions, [args.hidden]) for idx in range(2)]
            eval_policy = OS_Policy_Wrapper(env, agents)

        for ep in range(args.iterations):
            # Mappa dei ruoli: agents[0] ha sempre player_id=0, agents[1] ha sempre player_id=1.
            if np.random.rand() < 0.5:
                env_to_agent = {0: agents[0], 1: agents[1]}
            else:
                env_to_agent = {0: agents[1], 1: agents[0]}

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                acting_agent = env_to_agent[player_id]
                
                if acting_agent.player_id != player_id:
                    obs = time_step.observations.copy()
                    obs["info_state"] = [obs["info_state"][1], obs["info_state"][0]]
                    obs["legal_actions"] = [obs["legal_actions"][1], obs["legal_actions"][0]]
                    obs["current_player"] = acting_agent.player_id
                    
                    rev_rewards = [time_step.rewards[1], time_step.rewards[0]] if time_step.rewards is not None else None
                    rev_discounts = [time_step.discounts[1], time_step.discounts[0]] if time_step.discounts is not None else None
                    
                    ts_for_agent = time_step._replace(observations=obs, rewards=rev_rewards, discounts=rev_discounts)
                    action_output = acting_agent.step(ts_for_agent)
                else:
                    action_output = acting_agent.step(time_step)
                    
                time_step = env.step([action_output.action])
            
            # Step finale per l'apprendimento
            for role, agent in env_to_agent.items():
                if agent.player_id != role:
                    obs = time_step.observations.copy()
                    obs["info_state"] = [obs["info_state"][1], obs["info_state"][0]]
                    obs["legal_actions"] = [obs["legal_actions"][1], obs["legal_actions"][0]]
                    
                    rev_rewards = [time_step.rewards[1], time_step.rewards[0]] if time_step.rewards is not None else None
                    rev_discounts = [time_step.discounts[1], time_step.discounts[0]] if time_step.discounts is not None else None
                    
                    ts_for_agent = time_step._replace(observations=obs, rewards=rev_rewards, discounts=rev_discounts)
                    agent.step(ts_for_agent)
                else:
                    agent.step(time_step)

            if ep % args.eval_every == 0:
                eval_policy.clear_cache()
                conv = exploitability.exploitability(env.game, eval_policy)
                print(f"Episode {ep:7d} | Exploitability: {conv:.6f} | NashConv: {conv*2:.6f}")
                local_logger.log({"iteration": ep, "exploitability": float(conv), "nash_conv": float(conv * 2)})

            if ep > 0 and ep % args.save_every == 0:
                print(f"Salvataggio checkpoint episodio {ep}...")
                for idx, agent in enumerate(agents):
                    agent_path = os.path.join(checkpoint_dir, f"agent_{idx}_{ep}")
                    if args.algo == "nfsp":
                        os.makedirs(agent_path, exist_ok=True)
                        state = {
                            "avg_params": agent.params_avg_network,
                            "q_params": agent._rl_agent.params_q_network,
                        }
                        with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                            pickle.dump(state, f)
                    else:
                        # DQN JAX
                        state = {"q_params": agent.params_q_network}
                        with open(agent_path + ".pkl", "wb") as f:
                            pickle.dump(state, f)

    print(f"\n--- ESECUZIONE VALUTAZIONE FINALE ({args.iterations} iterazioni) ---")
    final_path = os.path.join(checkpoint_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)

    if args.algo == "cfr":
        # Valutazione finale per CFR (Tabulare)
        conv = exploitability.exploitability(env.game, solver.average_policy())
        print(f"FINALE CFR | NashConv: {conv * 2:.6f}")
        local_logger.log({"iteration": args.iterations, "nash_conv": float(conv * 2)})
        
        # Salvataggio oggetto solver CFR
        with open(os.path.join(final_path, "cfr_solver_final.pkl"), "wb") as f:
            pickle.dump(solver, f)
    elif args.algo == "deep_cfr":
        # Valutazione finale per Deep CFR
        solver._learn_strategy_network()
        conv = exploitability.exploitability(env.game, solver)
        print(f"FINALE DEEP CFR | Exploitability: {conv:.6f} | NashConv: {conv * 2:.6f}")
        local_logger.log({"iteration": args.iterations, "exploitability": float(conv), "nash_conv": float(conv * 2)})
        
        state = {
            "adv_params": solver._params_adv_network,
            "policy_params": solver._params_policy_network,
        }
        with open(os.path.join(final_path, "params.pkl"), "wb") as f:
            pickle.dump(state, f)
    else:
        # Valutazione finale per NFSP/DQN (Deep RL)
        eval_policy.clear_cache()
        conv = exploitability.exploitability(env.game, eval_policy)
        print(f"FINALE {args.algo.upper()} | NashConv: {conv * 2:.6f}")
        local_logger.log({"iteration": args.iterations, "nash_conv": float(conv * 2), "exploitability": float(conv)})
        
        # Salvataggio pesi JAX per ogni agente
        for idx, agent in enumerate(agents):
            agent_dir = os.path.join(final_path, f"agent_{idx}")
            os.makedirs(agent_dir, exist_ok=True)
            if args.algo == "nfsp":
                state = {"avg_params": agent.params_avg_network, "q_params": agent._rl_agent.params_q_network}
            else: # dqn
                state = {"q_params": agent.params_q_network}
            with open(os.path.join(agent_dir, "params.pkl"), "wb") as f:
                pickle.dump(state, f)

    print("Studio completato. Generazione report finale...")
    generate_run_plots(run_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="nfsp", 
                        choices=["cfr", "nfsp", "dqn", "deep_cfr"])
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--anticipatory", type=float, default=0.1)
    parser.add_argument("--reservoir", type=int, default=2000000)
    parser.add_argument("--traversals", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    run_comparison(args)
