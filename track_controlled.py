"""Keep W&B network and file uploads outside every measured inference call."""
import json
import os
from pathlib import Path

import wandb


class Tracker:
    def __init__(self, directory, name, cfg, group, job_type):
        mode = os.environ.get("WANDB_MODE", "offline")
        self.run = wandb.init(entity=cfg["wandb_entity"], project=cfg["wandb_project"],
                              group=group, name=name, job_type=job_type, config=cfg,
                              dir=str(directory), mode=mode,
                              settings=wandb.Settings(disable_git=True))

    def epoch(self, row):
        self.run.log({"epoch": row["epoch"],
                      "train/loss": row["loss"],
                      "validation/macro_f1": row["validation_macro_f1"]})

    def finish(self, directory, summary=None):
        if summary:
            self.run.summary.update(summary)
        artifact = wandb.Artifact(f"{self.run.name}-{self.run.id}", type="experiment-results")
        for path in Path(directory).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv"} and "wandb" not in path.parts:
                artifact.add_file(str(path), name=str(path.relative_to(directory)))
        self.run.log_artifact(artifact)
        Path(directory, "wandb_run.json").write_text(json.dumps({
            "id": self.run.id, "url": self.run.url, "mode": self.run.settings.mode,
            "entity": self.run.entity, "project": self.run.project}, indent=2))
        self.run.finish()
