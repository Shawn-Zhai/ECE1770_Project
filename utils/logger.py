import os
import json
import uuid
from datetime import datetime
from typing import Any, List

class RunLogger:
    """
    Creates a unique run directory and provides helpers to log
    text, JSON, and images for that run.
    """

    def __init__(self, root_dir: str = "runs", mode: str = "agent"):
        os.makedirs(root_dir, exist_ok=True)

        now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        self.run_id = f"run-{mode}-{now_str}-{short_id}"
        self.run_dir = os.path.join(root_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        print(f"[RunLogger] Run ID: {self.run_id}")
        print(f"[RunLogger] Logging to: {self.run_dir}")


    def log_text(self, name: str, text: str) -> None:
        path = os.path.join(self.run_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    
    def log_json(self, name: str, obj: Any) -> None:
        path = os.path.join(self.run_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    
    