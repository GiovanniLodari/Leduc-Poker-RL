# Leduc Poker RL

Esperimenti di reinforcement learning su **Leduc Poker**, un piccolo poker a informazione imperfetta, usando [OpenSpiel](https://github.com/google-deepmind/open_spiel). Il progetto confronta **NFSP** (*Neural Fictitious Self-Play*) con **DQN**, **CFR** e **Deep CFR**, misura quanto le strategie apprese sono sfruttabili (*exploitability*) e prova varianti della rete di NFSP.

## Contenuto

| Percorso | Contenuto |
|---|---|
| `leduc_poker_project/replica_deepmind.py` | Addestramento di NFSP e DQN (JAX) su Leduc Poker |
| `leduc_poker_project/deepmind_nsfp_*.py` | Varianti di NFSP: rete con attention, MLP piccolo o profondo, warm-up, ecc. |
| `leduc_poker_project/comparison_study*.py` | Confronto tra algoritmi (`nfsp`, `dqn`, `cfr`, `deep_cfr`) |
| `leduc_poker_project/tournament_study.py` | Torneo tra agenti con classifica Alpha-Rank |
| `leduc_poker_project/visualize_strategy*.py` | Visualizzazione delle strategie apprese |
| `leduc_poker_project/run_experiments.py` | Lancia una serie di esperimenti in sequenza |
| `leduc_poker_project/dashboard/` | Monitor locale (FastAPI) per seguire l'addestramento |
| `leduc_poker_project/logs/`, `plots/` | Log delle run (JSONL) e grafici prodotti |
| `archive/` | Grafici di run precedenti |

Durante l'addestramento le metriche vengono scritte in `leduc_poker_project/logs/` e i grafici in `leduc_poker_project/plots/`. I checkpoint vanno in `leduc_poker_project/checkpoints/`, che non è nel repository perché pesante.

## Installazione

Serve un ambiente in cui OpenSpiel sia installabile (Linux, macOS o WSL).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install open_spiel            # oppure build da sorgente, vedi la guida di OpenSpiel
```

## Uso

Gli script vanno lanciati dalla radice del repository.

```bash
# Addestramento NFSP su Leduc Poker
python leduc_poker_project/replica_deepmind.py --algo nfsp --episodes 1000000

# Confronto con altri algoritmi
python leduc_poker_project/comparison_study.py --algo cfr --iterations 100000

# Torneo Alpha-Rank tra checkpoint (path:algo:label)
python leduc_poker_project/tournament_study.py --agents <path>:nfsp:NFSP_16 <path>:dqn:DQN --plot

# Strategia appresa da un checkpoint
python leduc_poker_project/visualize_strategy.py --path <checkpoint>/params.pkl --algo nfsp
```

Gli script di addestramento avviano anche il monitor su <http://localhost:8000>, che mostra le run in `logs/`.
