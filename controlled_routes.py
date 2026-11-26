"""Batch routing and controlled overhead variants with a frozen teacher."""
import numpy as np
import torch


@torch.inference_mode()
def fixed_batch(model, tokens):
    labels = model(**tokens).logits.argmax(-1).cpu().numpy()
    return labels, np.full(len(labels), len(model.bert.encoder.layer))


class RoutedBatch:
    """Dynamic mode really skips blocks; other modes always execute all blocks.

    loop: no intermediate computation; head: score on GPU only;
    check: score plus host transfer and threshold comparison, but never exit.
    Only max probability and label are computed; entropy is unused by this policy.
    """
    def __init__(self, teacher, heads, temperatures, policy, mode="dynamic"):
        if mode not in {"dynamic", "loop", "head", "check"}:
            raise ValueError(mode)
        self.teacher, self.heads = teacher, heads
        self.temperatures, self.policy, self.mode = temperatures, dict(policy), mode

    @torch.inference_mode()
    def __call__(self, tokens):
        bert = self.teacher.bert
        mask = bert.get_extended_attention_mask(tokens["attention_mask"], tokens["input_ids"].shape)
        hidden = bert.embeddings(input_ids=tokens["input_ids"], token_type_ids=tokens.get("token_type_ids"))
        active = np.arange(len(hidden))
        labels, depths = np.empty(len(active), dtype=int), np.full(len(active), len(bert.encoder.layer))
        for layer, block in enumerate(bert.encoder.layer, 1):
            hidden = block(hidden, attention_mask=mask)[0]
            if layer not in self.policy or self.mode == "loop":
                continue
            scores = self.heads[str(layer)](hidden) / self.temperatures[layer]
            confidence, predicted = scores.softmax(-1).max(-1)
            packed = torch.stack((confidence, predicted.to(confidence.dtype)), -1)
            if self.mode == "head":
                continue
            values = packed.cpu().numpy()
            exiting = values[:, 0] >= self.policy[layer]
            if self.mode == "check":
                continue
            labels[active[exiting]] = values[exiting, 1].astype(int)
            depths[active[exiting]] = layer
            active = active[~exiting]
            if not len(active):
                return labels, depths
            if exiting.any():
                keep = torch.as_tensor(np.flatnonzero(~exiting), device=hidden.device)
                hidden, mask = hidden.index_select(0, keep), mask.index_select(0, keep)
        final = self.teacher.classifier(self.teacher.dropout(bert.pooler(hidden)))
        labels[active] = final.argmax(-1).cpu().numpy()
        return labels, depths
