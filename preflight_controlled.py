"""Verify W&B access before requesting the experiment GPU."""
import json
from pathlib import Path

import wandb

cfg = json.loads(Path("controlled.json").read_text())
directory = Path("preflight")
directory.mkdir(exist_ok=True)
try:
    if not wandb.login(prompt=False):
        raise RuntimeError("No configured W&B login")
    with wandb.init(entity=cfg["wandb_entity"], project=cfg["wandb_project"],
                    name="controlled-a100-protocol", job_type="protocol", config=cfg,
                    dir=str(directory), settings=wandb.Settings(init_timeout=45)) as run:
        run.summary["phase"] = "prepared; GPU experiment not yet run"
        artifact = wandb.Artifact("controlled-a100-protocol", type="protocol")
        for filename in ["CONTROLLED_PROTOCOL.md", "controlled.json", "run_controlled.py", "controlled_routes.py"]:
            artifact.add_file(filename)
        run.log_artifact(artifact)
        Path("wandb_preflight.json").write_text(json.dumps({"mode": "online", "url": run.url}, indent=2))
    Path("wandb_mode.txt").write_text("online")
except Exception as error:
    Path("wandb_mode.txt").write_text("offline")
    Path("wandb_preflight.json").write_text(json.dumps({"mode": "offline", "error_type": type(error).__name__}))
    print("W&B will record offline; cloud synchronization still required.", flush=True)
