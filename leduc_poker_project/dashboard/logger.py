import json
import os
from datetime import datetime

class LocalLogger:
    """
    Gestore dei log per il dashboard locale.
    Salva le metriche in formato JSONL nella cartella 'logs/'.
    """
    def __init__(self, run_name):
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(dashboard_dir)
        self.log_dir = os.path.join(project_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        now = datetime.now()
        timestamp_suffix = now.strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{run_name}_{timestamp_suffix}"
        self.log_file = os.path.join(self.log_dir, f"{self.run_name}.jsonl")
        self.start_time = now.isoformat()
        
        with open(self.log_file, "w") as f:
            f.write(json.dumps({
                "type": "info", 
                "run_name": self.run_name, 
                "start_time": self.start_time
            }) + "\n")

    def log(self, metrics):
        """Aggiunge nuove metriche al file log."""
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["type"] = "metrics"
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
