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
from open_spiel.python.pytorch import dqn as dqn_pt
from open_spiel.python.jax import nfsp
from open_spiel.python.jax import dqn as dqn_jax
import sys
import pickle
import signal
import jax

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dashboard.logger import LocalLogger
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
    try:
        import jax
        if jax.devices()[0].platform == 'gpu':
            return
    except:
        pass

    venv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nvidia_base = os.path.join(venv_base, ".venv", "lib", "python3.10", "site-packages", "nvidia")

    if os.path.exists(nvidia_base):
        lib_paths = []
        for root, dirs, files in os.walk(nvidia_base):
            if "lib" in dirs:
                lib_paths.append(os.path.join(root, "lib"))

        if lib_paths:
            print("Configurazione GPU in corso...")
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + current_ld if current_ld else "")

            os.execv(sys.executable, [sys.executable] + sys.argv)

def start_dashboard():
    """Avvia il server dashboard in background se non è attivo."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', 8000)) == 0:
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
    local_logger = LocalLogger(f"{args.algo}_replica")
    run_name = local_logger.run_name

    print(f"Inizio run: \033[92m{run_name}\033[0m")

    signal.signal(signal.SIGINT, lambda s, f: handle_exit(s, f, run_name))

    env = rl_environment.Environment(args.game)
    num_actions = env.action_spec()["num_actions"]
    info_state_size = env.observation_spec()["info_state"][0]

    if args.algo == "nfsp":
        agents = [
            nfsp.NFSP(idx, info_state_size, num_actions, [128],
                      reservoir_buffer_capacity=int(2e6),
                      replay_buffer_capacity=int(2e5),
                      anticipatory_param=0.1,
                      epsilon_start=0.06, epsilon_end=0.001)
            for idx in range(2)
        ]
        eval_policy = OS_Policy_Wrapper(env, agents, nfsp.MODE.average_policy)

    elif args.algo == "dqn":
        agents = [
            dqn_jax.DQN(idx, info_state_size, num_actions, [128])
            for idx in range(2)
        ]
        eval_policy = OS_Policy_Wrapper(env, agents, None)

    print(f"--- REPLICA GOOGLE DEEPMIND: {args.algo.upper()} ---")

    checkpoint_dir = os.path.join("leduc_poker_project/checkpoints", "replica", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for ep in range(args.episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            action_output = agents[player_id].step(time_step)
            time_step = env.step([action_output.action])
       
        for agent in agents:
            agent.step(time_step)

        if ep % args.eval_every == 0:

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

                    state = {
                        "avg_params": agent.params_avg_network,
                        "q_params": agent._rl_agent.params_q_network,
                    }
                    with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                        pickle.dump(state, f)
                else:

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

            state = {
                "avg_params": agent.params_avg_network,
                "q_params": agent._rl_agent.params_q_network,
            }
            with open(os.path.join(agent_path, "params.pkl"), "wb") as f:
                pickle.dump(state, f)
        else:

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
    args = parser.parse_args()
    train(args)
