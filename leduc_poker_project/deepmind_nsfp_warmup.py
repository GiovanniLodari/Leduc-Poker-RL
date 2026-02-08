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
import jax.numpy as jnp
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
    if os.environ.get("GPU_SETUP_DONE") == "1": return
    venv_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nvidia_base = os.path.join(venv_base, ".venv", "lib", "python3.10", "site-packages", "nvidia")
    lib_paths = ["/usr/lib/wsl/lib"]
    if os.path.exists(nvidia_base):
        for root, dirs, files in os.walk(nvidia_base):
            if "lib" in dirs: lib_paths.append(os.path.join(root, "lib"))
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (":" + current_ld if current_ld else "")
    os.environ["GPU_SETUP_DONE"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

def train(args):
    setup_gpu()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', 8000)) != 0:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            subprocess.Popen([sys.executable, os.path.join(base_dir, "dashboard", "server.py")], stdout=subprocess.DEVNULL)
    
    hidden_layers = [int(h) for h in args.hidden_list.split(",")]
    local_logger = LocalLogger(f"nsfp_warmup_{args.hidden_list.replace(',', 'x')}")
    run_name = local_logger.run_name
    checkpoint_dir = os.path.join("leduc_poker_project/checkpoints", "research", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Inizio run \033[92m{run_name}\033[0m")
    print(f"Fase Warm-up: {args.warmup} episodi (Buffer Seeding)")
    signal.signal(signal.SIGINT, lambda s, f: handle_exit(s, f, run_name))

    env = rl_environment.Environment(args.game)
    num_actions = env.action_spec()["num_actions"]
    info_state_size = env.observation_spec()["info_state"][0]

    agents = [
        nfsp.NFSP(idx, info_state_size, num_actions, hidden_layers,
                  reservoir_buffer_capacity=args.reservoir,
                  replay_buffer_capacity=int(args.reservoir / 10),
                  batch_size=1024,
                  anticipatory_param=0.1,
                  learning_rate=1e-3,
                  sl_learning_rate=0.01)
        for idx in range(2)
    ]
    eval_policy = OS_Policy_Wrapper(env, agents, nfsp.MODE.average_policy)

    original_learn_dqn = [a._rl_agent.learn for a in agents]
    original_learn_avg = [a._learn for a in agents]

    def dummy_learn(*args, **kwargs): return None

    for ep in range(args.episodes + args.warmup):
        is_warmup = (ep < args.warmup)
        
        if is_warmup:
            for a in agents:
                a._rl_agent.learn = dummy_learn
                a._learn = dummy_learn

                a._rl_agent._epsilon = 1.0 
        else:

            if ep == args.warmup:
                print(f"\n--- FINE WARMUP: Inizio addestramento reale ---")
                for i, a in enumerate(agents):
                    a._rl_agent.learn = original_learn_dqn[i]
                    a._learn = original_learn_avg[i]
            
            decay_duration = 500000
            epsilon_start = 0.2
            epsilon_end = 0.06
            
            if ep < args.warmup + decay_duration:
                fraction = (ep - args.warmup) / decay_duration
                current_eps = epsilon_start + fraction * (epsilon_end - epsilon_start)
            else:
                current_eps = epsilon_end
                
            for a in agents:
                a._rl_agent._epsilon = current_eps

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
            else:
                ts_for_agent = time_step
            
            action_output = acting_agent.step(ts_for_agent)
            time_step = env.step([action_output.action])

        for role, agent in env_to_agent.items():
            if agent.player_id != role:
                obs = time_step.observations.copy()
                obs["info_state"] = [obs["info_state"][1], obs["info_state"][0]]
                obs["legal_actions"] = [obs["legal_actions"][1], obs["legal_actions"][0]]
                rev_rewards = [time_step.rewards[1], time_step.rewards[0]] if time_step.rewards is not None else None
                rev_discounts = [time_step.discounts[1], time_step.discounts[0]] if time_step.discounts is not None else None
                ts_for_agent = time_step._replace(observations=obs, rewards=rev_rewards, discounts=rev_discounts)
            else:
                ts_for_agent = time_step
            
            agent.step(ts_for_agent)

        if ep % args.eval_every == 0:
            eval_policy.clear_cache()
            expl = exploitability.exploitability(env.game, eval_policy)
            status = "WARMUP" if is_warmup else "TRAIN"
            print(f"[{status}] Ep {ep:7d} | NashConv: {expl*2:.6f}")
            local_logger.log({"episode": ep, "exploitability": expl, "nash_conv": expl*2, "phase": status})

    generate_run_plots(run_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--episodes", type=int, default=10000000)
    parser.add_argument("--warmup", type=int, default=20000, help="Episodi di gioco casuale senza training")
    parser.add_argument("--reservoir", type=int, default=200000)
    parser.add_argument("--hidden_list", type=str, default="128,128")
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=500000)
    args = parser.parse_args()
    train(args)
