"Per avviare esperimanti anche di notte"


import subprocess
import sys
import os

def run_experiment(neurons, logger_prefix):
    print(f"\n" + "="*50)
    print(f"AVVIO ESPERIMENTO: {neurons} neuroni | Prefisso: {logger_prefix}")
    print("="*50 + "\n")
    
    cmd = [
        sys.executable, 
        "leduc_poker_project/deepmind_nsfp_last_try.py",
        "--neurons", str(neurons),
        "--logger_prefix", logger_prefix,
        "--episodes", "10000000"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[OK] Esperimento con {neurons} neuroni completato con successo.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRORE] L'esperimento con {neurons} neuroni è fallito con codice {e.returncode}.")

if __name__ == "__main__":
    experiments = [
        (32, "nfsp_swapped_32")
    ]
    
    print(f"Orchestratore avviato. Esecuzione di {len(experiments)} esperimenti in sequenza.")
    
    for neurons, prefix in experiments:
        run_experiment(neurons, prefix)
        
    print("\n" + "!"*50)
    print("TUTTI GLI ESPERIMENTI SONO STATI COMPLETATI.")
    print("!"*50 + "\n")
