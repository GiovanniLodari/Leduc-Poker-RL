import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Leduc Poker Local Monitor")

UI_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(UI_DIR)
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

@app.get("/api/runs")
async def get_runs():
    files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".jsonl")]
    runs_info = []
    for f in files:
        path = os.path.join(LOGS_DIR, f)
        try:
            with open(path, "r") as log_file:
                first_line = log_file.readline()
                info = json.loads(first_line)
                if info.get("type") == "info":

                    run_id = f.replace(".jsonl", "")
                    runs_info.append({
                        "name": run_id, 
                        "display_name": info.get("run_name"),
                        "start_time": info.get("start_time"),
                        "file": f
                    })
        except:
            continue
    
    runs_info.sort(key=lambda x: x["start_time"] if x["start_time"] else "", reverse=True)
    return runs_info

@app.get("/api/metrics/{run_name}")
async def get_metrics(run_name: str):
    """Read metrics for a specific run."""
    file_path = os.path.join(LOGS_DIR, f"{run_name}.jsonl")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Run not found")
    
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    index_path = os.path.join(UI_DIR, "index.html")
    if not os.path.exists(index_path):
        return "<h1>Dashboard UI not found</h1>"
    with open(index_path, "r") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
