import os
import argparse
import numpy as np
import subprocess
import time
import socket
import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import nfsp
from open_spiel.python.jax import dqn
import sys
import pickle
import signal
import jax
import jax.numpy as jnp
import haiku as hk


# Silenzia avvisi tecnici XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dashboard.logger import LocalLogger

class AttentionBlock(hk.Module):
    """Modulo di Self-Attention per vettori di informazione piatti."""
    def __init__(self, model_size=128, num_heads=4, name=None):
        super().__init__(name=name)
        self.model_size = model_size
        self.num_heads = num_heads

    def __call__(self, x):
        # x shape: [Batch, Dim]
        # Proiettiamo in uno spazio di embedding e aggiungiamo dimensione sequenza [Batch, 1, model_size]
        # Nota: In Leduc Poker, un'attenzione su un singolo 'token' funge da gating complesso.
        x = hk.Linear(self.model_size)(x)
        x_seq = x[:, None, :]
        
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_size=self.model_size // self.num_heads, 
            model_size=self.model_size,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        )
        x_attn = attn(x_seq, x_seq, x_seq)
        
        # Residuo e LayerNorm
        x = x + x_attn.squeeze(1)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        return x

class AttentionDQN(dqn.DQN):
    """Versione di DQN con Attention Block iniettato nell'architettura."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sovrascriviamo la rete hk_network definita nel super().__init__
        def network(x):
            x = AttentionBlock()(x)
            mlp = hk.nets.MLP(self._layer_sizes + [self._num_actions])
            return mlp(x)
        
        self.hk_network = hk.without_apply_rng(hk.transform(network))
        self.hk_network_apply = jax.jit(self.hk_network.apply)
        # Re-inizializziamo i parametri con la nuova rete
        rng = jax.random.PRNGKey(42)
        self._create_networks(rng, args[1] if len(args)>1 else kwargs['state_representation_size'])

class AttentionNFSP(nfsp.NFSP):
    """Versione di NFSP con Attention Block iniettato sia nella policy media che nel DQN interno."""
    def __init__(self, *args, **kwargs):
        # NFSP crea internamente un DQN. Dobbiamo assicurarci che usi AttentionDQN.
        # Patchiamo temporaneamente dqn.DQN prima dell'init
        original_dqn = dqn.DQN
        dqn.DQN = AttentionDQN
        try:
            super().__init__(*args, **kwargs)
        finally:
            dqn.DQN = original_dqn
            
        # Ora sovrascriviamo la average policy network di NFSP
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
from dashboard.plotter import generate_run_plots


def handle_exit(signum, frame, run_name):
    print(f"\nTraining interrotto. Salvataggio finale e generazione grafici...")
    generate_run_plots(run_name)
    sys.exit(0)


class OS_Policy_Wrapper(policy.Policy):
    """
    Wrapper ottimizzato con cache per calcolare la NashConv.

    """
    def __init__(self, env, agents, mode):
        super().__init__(env.game, [0, 1])
        self._agents = agents
        self._mode = mode
        self._cache = {}


    def action_probabilities(self, state, player_id=None):
        # Leduc Poker ha molti stati che condividono la stessa information state
        # Usiamo una cache locale per non invocare JAX inutilmente
        # (questo accelera la valutazione di 5-10x)
        infostate_key = state.information_state_string()
        if infostate_key in self._cache:
            return self._cache[infostate_key]

        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        obs = {
            "info_state": [None, None],
            "legal_actions": [None, None],
            "current_player": cur_player
        }
        obs["info_state"][cur_player] = state.information_state_tensor(cur_player)
        obs["legal_actions"][cur_player] = legal_actions
        time_step = rl_environment.TimeStep(observations=obs, rewards=None, discounts=None, step_type=None)

        if self._mode is not None:
            with self._agents[cur_player].temp_mode_as(self._mode):
                p = self._agents[cur_player].step(time_step, is_evaluation=True).probs
        else:
            p = self._agents[cur_player].step(time_step, is_evaluation=True).probs

        probs_dict = {a: p[a] for a in legal_actions}
        self._cache[infostate_key] = probs_dict
        return probs_dict

    def clear_cache(self):
        self._cache = {}



def setup_gpu():
    """Configura l'ambiente per abilitare la GPU su WSL/Linux se necessario."""
    if os.environ.get("GPU_SETUP_DONE") == "1":
        return

    print("Configurazione GPU/CUDA per WSL...")
    
    # 1. Percorsi fondamentali per WSL
    lib_paths = ["/usr/lib/wsl/lib"]
    
    # 2. Percorsi librerie NVIDIA nel venv
    venv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nvidia_base = os.path.join(venv_base, ".venv", "lib", "python3.10", "site-packages", "nvidia")
    
    if os.path.exists(nvidia_base):
        for root, dirs, files in os.walk(nvidia_base):
            if "lib" in dirs:
                lib_paths.append(os.path.join(root, "lib"))

    # 3. Fix specifico per cuSPARSE e altre lib se mancano i link .so generici
    # JAX spesso cerca libcusparse.so.12 o .11. Se abbiamo .so.12 ma cerca .so, creiamo link.
    for p in lib_paths:
        if "cusparse" in p:
            for f in os.listdir(p):
                if f.startswith("libcusparse.so.") and not os.path.exists(os.path.join(p, "libcusparse.so")):
                    try:
                        os.symlink(f, os.path.join(p, "libcusparse.so"))
                    except: pass
        if "cublas" in p:
             for f in os.listdir(p):
                if f.startswith("libcublas.so.") and not os.path.exists(os.path.join(p, "libcublas.so")):
                    try:
                        os.symlink(f, os.path.join(p, "libcublas.so"))
                    except: pass

    # 4. Applica LD_LIBRARY_PATH
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + current_ld if current_ld else "")
    os.environ["GPU_SETUP_DONE"] = "1"
    
    # 5. Silenzia logging e forza XLA a vedere le librerie
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    print(f"Riavvio script con LD_LIBRARY_PATH ottimizzato...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


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
    time.sleep(2)
    print(f"\033[94mDashboard attiva: http://localhost:8000\033[0m")

def train(args):
    setup_gpu()
    start_dashboard()
    local_logger = LocalLogger("nsfp_attention")
    run_name = local_logger.run_name

    print(f"Inizio run: \033[92m{run_name}\033[0m")
    # Configura segnale di uscita per salvare i grafici
    signal.signal(signal.SIGINT, lambda s, f: handle_exit(s, f, run_name))

    # 1. Setup Ambiente OpenSpiel
    env = rl_environment.Environment(args.game)
    num_actions = env.action_spec()["num_actions"]
    info_state_size = env.observation_spec()["info_state"][0]

    # 2. Setup Agenti
    if args.algo == "nfsp":
        agents = [AttentionNFSP(idx, info_state_size, num_actions, [args.hidden], 
                           reservoir_buffer_capacity=args.reservoir,
                           anticipatory_param=args.anticipatory) for idx in range(2)]
        eval_policy = OS_Policy_Wrapper(env, agents, nfsp.MODE.average_policy)
    else: # dqn
        agents = [AttentionDQN(idx, info_state_size, num_actions, [args.hidden]) for idx in range(2)]
        eval_policy = OS_Policy_Wrapper(env, agents, None)

    print(f"--- REPLICA GOOGLE DEEPMIND: {args.algo.upper()} ---")

    # 3. Training Loop
    checkpoint_dir = os.path.join("leduc_poker_project/checkpoints", "replica", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for ep in range(args.episodes):
        # Mappa dei ruoli: agents[0] ha sempre _player_id=0, agents[1] ha sempre _player_id=1.
        # Decidiamo chi interpreta quale ruolo nell'ambiente per questo episodio.
        if np.random.rand() < 0.5:
            env_to_agent = {0: agents[0], 1: agents[1]}
        else:
            env_to_agent = {0: agents[1], 1: agents[0]}

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            acting_agent = env_to_agent[player_id]
            
            # Se l'agente sta interpretando il ruolo opposto al suo ID, 
            # invertiamo le osservazioni, i reward e i discount affinché trovi i suoi dati al giusto indice (0 o 1).
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
       
        # Step finale per l'apprendimento (necessario per aggiornare i buffer terminali)
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

        # 4. Valutazione e Salvataggio Periodico
        if ep % args.eval_every == 0:
            # Svuota la cache per calcolare la politica corrente dell'agente
            eval_policy.clear_cache()
            expl = exploitability.exploitability(env.game, eval_policy)
            nash_conv = expl * 2
            print(f"Ep {ep:7d} | Exploitability: {expl:.6f} | NashConv: {nash_conv:.6f}")
            local_logger.log({"episode": ep, "exploitability": expl, "nash_conv": nash_conv})

        if ep > 0 and ep % args.save_every == 0:
            print(f"Salvataggio checkpoint episodio {ep}...")
            for idx, agent in enumerate(agents):
                agent_path = os.path.join(checkpoint_dir, f"agent_{idx}")
                os.makedirs(agent_path, exist_ok=True)
               
                if args.algo == "nfsp":
                    # Salvataggio manuale parametri JAX (Haiku)
                    state = {
                        "avg_params": agent.params_avg_network,
                        "q_params": agent._rl_agent.params_q_network,
                    }
                    with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                        pickle.dump(state, f)
                else:
                    # DQN JAX
                    state = {"q_params": agent.params_q_network}
                    with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                        pickle.dump(state, f)

    print(f"\n--- VALUTAZIONE E SALVATAGGIO FINALE (Episodio {args.episodes}) ---")

    eval_policy.clear_cache()
    expl = exploitability.exploitability(env.game, eval_policy)
    nash_conv = expl * 2

    print(f"FINALE | Exploitability: {expl:.6f} | NashConv: {nash_conv:.6f}")
    local_logger.log({"episode": args.episodes, "exploitability": expl, "nash_conv": nash_conv})
    final_checkpoint_dir = os.path.join(checkpoint_dir, "final_model")
    os.makedirs(final_checkpoint_dir, exist_ok=True)

    for idx, agent in enumerate(agents):
        agent_path = os.path.join(final_checkpoint_dir, f"agent_{idx}")
        os.makedirs(agent_path, exist_ok=True)

        if args.algo == "nfsp":
            # Salvataggio manuale parametri JAX (Haiku)
            state = {
                "avg_params": agent.params_avg_network,
                "q_params": agent._rl_agent.params_q_network,
            }
            with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                pickle.dump(state, f)
        else:
            # DQN JAX
            state = {"q_params": agent.params_q_network}
            with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                pickle.dump(state, f)

    print("Training Completato. Generazione report finale...")
    generate_run_plots(run_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="nfsp", choices=["nfsp", "dqn"])
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--episodes", type=int, default=10000000, help="10M for full convergence")
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=100000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--anticipatory", type=float, default=0.1)
    parser.add_argument("--reservoir", type=int, default=2000000)
    args = parser.parse_args()
    train(args)