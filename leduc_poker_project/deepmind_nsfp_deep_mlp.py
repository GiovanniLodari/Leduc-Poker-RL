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
from open_spiel.python.jax import dqn as dqn_jax
import sys
import pickle
import signal
import jax

# Silenzia avvisi tecnici XLA/JAX
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dashboard.logger import LocalLogger
from dashboard.plotter import generate_run_plots


def handle_exit(signum, frame, run_name):
    print(f"\nTraining interrotto. Salvataggio finale e generazione grafici...")
    generate_run_plots(run_name)
    sys.exit(0)


class OS_Policy_Wrapper(policy.Policy):
    def __init__(self, env, agents, mode):
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
    
    # Percorsi NVIDIA nel venv
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
    os.execv(sys.executable, [sys.executable] + sys.argv)


def start_dashboard():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', 8000)) == 0:
            print(f"\033[94mDashboard attiva: http://localhost:8000\033[0m")
            return
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(base_dir, "dashboard", "server.py")
    subprocess.Popen([sys.executable, server_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def train(args):
    setup_gpu()
    start_dashboard()
    
    # Architettura definita dall'utente (es. 128,128)
    hidden_layers = [int(h) for h in args.hidden_list.split(",")]
    tag = f"mlp_2_layers_{args.hidden_list.replace(',', 'x')}"
    
    local_logger = LocalLogger(f"nsfp_{tag}")
    run_name = local_logger.run_name
    checkpoint_dir = os.path.join("leduc_poker_project/checkpoints", "research", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Inizio run \033[92m{run_name}\033[0m con architettura: {hidden_layers}")
    signal.signal(signal.SIGINT, lambda s, f: handle_exit(s, f, run_name))

    env = rl_environment.Environment(args.game)
    num_actions = env.action_spec()["num_actions"]
    info_state_size = env.observation_spec()["info_state"][0]

    # Setup Agenti con architettura profonda
    agents = [
        nfsp.NFSP(idx, info_state_size, num_actions, hidden_layers,
                  reservoir_buffer_capacity=args.reservoir,
                  replay_buffer_capacity=int(args.reservoir / 10),
                  anticipatory_param=0.1)
        for idx in range(2)
    ]
    eval_policy = OS_Policy_Wrapper(env, agents, nfsp.MODE.average_policy)

    for ep in range(args.episodes):
        if np.random.rand() < 0.5:
            env_to_agent = {0: agents[0], 1: agents[1]}
        else:
            env_to_agent = {0: agents[1], 1: agents[0]}

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            acting_agent = env_to_agent[player_id]
            
            # Swapping LOGIC con fix dei reward (fondamentale!)
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
            expl = exploitability.exploitability(env.game, eval_policy)
            print(f"Ep {ep:7d} | NashConv: {expl*2:.6f}")
            local_logger.log({"episode": ep, "exploitability": expl, "nash_conv": expl*2})

        if ep > 0 and ep % args.save_every == 0:
            final_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{ep}")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            for idx, agent in enumerate(agents):
                agent_path = os.path.join(final_checkpoint_dir, f"agent_{idx}")
                os.makedirs(agent_path, exist_ok=True)
                state = {"avg_params": agent.params_avg_network, "q_params": agent._rl_agent.params_q_network}
                with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                    pickle.dump(state, f)

    generate_run_plots(run_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--episodes", type=int, default=10000000)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=500000)
    parser.add_argument("--hidden_list", type=str, default="128,128", help="Strati separati da virgola")
    parser.add_argument("--reservoir", type=int, default=2000000, help="Capacità del reservoir buffer")
    args = parser.parse_args()
    train(args)
