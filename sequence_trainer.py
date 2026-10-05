"""Training loop for the sequence models with early stopping on validation AUC."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


@torch.no_grad()
def predict_proba(model: nn.Module, steps: torch.Tensor, static: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    chunks = [torch.sigmoid(model(steps[i:i + batch_size], static[i:i + batch_size])) for i in range(0, len(steps), batch_size)]
    return torch.cat(chunks).cpu().numpy()


def train_sequence_model(model: nn.Module, train: tuple, val: tuple, params: dict, batch_size: int, max_epochs: int,
                         patience: int, weight_decay: float, seed: int, on_epoch=None) -> tuple[nn.Module, list[dict]]:
    """Fit a model and return the weights from its best validation epoch plus the learning history."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    steps, static, target = train
    optimiser = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, stale, history, started = -1.0, None, 0, [], time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(steps), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            loss = loss_fn(model(steps[index], static[index]), target[index])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            total += float(loss.detach()) * len(index)
        val_auc = float(roc_auc_score(val[2].numpy(), predict_proba(model, val[0], val[1])))
        scheduler.step(val_auc)
        row = {"epoch": epoch, "train_loss": total / len(steps), "val_auc": val_auc, "seconds": round(time.time() - started, 1)}
        history.append(row)
        if on_epoch:
            on_epoch(row)
        if val_auc > best_auc + 1e-4:
            best_auc, best_state, stale = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history
