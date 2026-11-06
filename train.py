"""Train only the eight-layer student; keep the original teacher unchanged."""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from data import Headlines, loader, logits_for, macro_f1, save_json, seed_everything
from models import load_model, make_student


def distillation_loss(student, teacher, labels, cfg):
    """Use gold labels alone when the teacher is wrong; otherwise also imitate it."""
    temperature = cfg["distillation_temperature"]
    gold = F.cross_entropy(student, labels, reduction="none")
    imitation = F.kl_div((student / temperature).log_softmax(-1),
                         (teacher.detach() / temperature).softmax(-1),
                         reduction="none").sum(-1) * temperature**2
    weight = (1 - cfg["gold_weight"]) * teacher.argmax(-1).eq(labels)
    return ((1 - weight) * gold + weight * imitation).mean()


def train_student(teacher, tokenizer, splits, cfg, directory, device):
    # Compute teacher targets once, on training data only, in the original row order.
    targets = logits_for(teacher, loader(splits["train"], tokenizer, cfg), device)
    dataset = Headlines(splits["train"], tokenizer, cfg["max_length"])
    padding = DataCollatorWithPadding(tokenizer)

    def collate(indices):
        return padding([dataset[index] for index in indices]), targets[indices]

    seed_everything(cfg["seed"])
    student = make_student(teacher, cfg["kept_layers"]).requires_grad_(True)
    batches = DataLoader(range(len(dataset)), batch_size=cfg["batch_size"], shuffle=True,
                         generator=torch.Generator().manual_seed(cfg["seed"]), collate_fn=collate)
    validation = loader(splits["validation"], tokenizer, cfg)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * cfg["warmup_fraction"]), steps)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -1.0
    history = []
    checkpoint = directory / "student"
    for epoch in range(cfg["epochs"] + 1):
        # Epoch zero preserves the initial model as a valid checkpoint candidate.
        total_loss = 0.0
        if epoch:
            student.train()
            for batch, teacher_scores in batches:
                labels = batch.pop("labels").to(device)
                tokens = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = distillation_loss(student(**tokens).logits.float(),
                                             teacher_scores.to(device), labels, cfg)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite student training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                total_loss += float(loss.detach()) * len(labels)
        scores = logits_for(student, validation, device)
        score = macro_f1(splits["validation"].label, scores.argmax(-1))
        history.append({"epoch": epoch, "loss": total_loss / len(dataset) if epoch else None,
                        "validation_macro_f1": score})
        if score > best:
            best, best_epoch = score, epoch
            student.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
        save_json(directory / "history.json", history)
        print(f"Epoch {epoch}: validation macro-F1 {score:.6f}", flush=True)
    save_json(directory / "selection.json", {"best_epoch": best_epoch, "validation_macro_f1": best,
              "test_used_for_selection": False, "kept_layers": cfg["kept_layers"]})
    del optimizer, scheduler, scaler, student
    return load_model(checkpoint, device)
