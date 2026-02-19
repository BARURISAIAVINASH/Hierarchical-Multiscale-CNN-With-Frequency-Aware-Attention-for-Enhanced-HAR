from pathlib import Path
import time

class SimpleLogger:
    def __init__(self, log_dir="results/logs", name="run"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(log_dir) / f"{name}_{int(time.time())}.log"

    def log(self, msg: str):
        print(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
