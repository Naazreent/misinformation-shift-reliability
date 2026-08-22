"""Deterministic PyTorch text architectures for the neural reconstruction.

The FNN, CNN, and LSTM preserve the intent of the legacy notebook while using
one train-only vocabulary and one controlled preprocessing path. The BERT path
records and pins the exact pretrained checkpoint revision.
"""

from __future__ import annotations

import copy
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"


def tokenize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return TOKEN_PATTERN.findall(normalized)


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_size: int,
        min_frequency: int,
    ) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize_words(text))
        ordered = sorted(
            (
                (token, count)
                for token, count in counts.items()
                if count >= min_frequency
            ),
            key=lambda item: (-item[1], item[0]),
        )
        tokens = [PAD_TOKEN, UNKNOWN_TOKEN]
        tokens.extend(token for token, _ in ordered[: max(0, max_size - 2)])
        return cls({token: index for index, token in enumerate(tokens)})

    def encode(self, text: str, max_length: int) -> list[int]:
        unknown = self.token_to_id[UNKNOWN_TOKEN]
        encoded = [self.token_to_id.get(token, unknown) for token in tokenize_words(text)]
        encoded = encoded[:max_length]
        if not encoded:
            encoded = [unknown]
        return encoded + [0] * (max_length - len(encoded))

    def encode_many(self, texts: Iterable[str], max_length: int) -> torch.Tensor:
        return torch.tensor(
            [self.encode(text, max_length) for text in texts], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.token_to_id)


class FNNTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(0).unsqueeze(-1)
        embedded = self.embedding(input_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


class CNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filters: int,
        kernel_sizes: list[int],
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters, kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters * len(kernel_sizes), 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = [torch.relu(layer(embedded)).amax(dim=2) for layer in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class LSTMTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = input_ids.ne(0).sum(dim=1).clamp_min(1).cpu()
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(self.dropout(hidden[-1]))


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training labels must contain both classes")
    return torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float32)


def _loader(
    inputs: torch.Tensor,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(inputs, torch.tensor(labels, dtype=torch.long))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def predict_word_model(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = _loader(inputs, labels, batch_size, False, 0)
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            logits = model(batch_inputs.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(batch_labels)
    logits = torch.cat(all_logits)
    observed = torch.cat(all_labels).numpy()
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    predictions = logits.argmax(dim=1).numpy()
    macro_f1 = float(f1_score(observed, predictions, average="macro", zero_division=0))
    return predictions, probabilities, macro_f1


def train_word_model(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_labels: np.ndarray,
    validation_inputs: torch.Tensor,
    validation_labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    batch_size = int(config["batch_size"])
    loader = _loader(train_inputs, train_labels, batch_size, True, seed)
    model.to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights(train_labels).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
            examples += len(batch_labels)
        _, _, validation_f1 = predict_word_model(
            model,
            validation_inputs,
            validation_labels,
            batch_size,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(examples, 1),
                "validation_macro_f1": validation_f1,
            }
        )
        if validation_f1 > best_f1 + 1e-8:
            best_f1 = validation_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "fit_seconds": time.perf_counter() - started,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def build_word_model(name: str, vocab_size: int, config: dict[str, Any]) -> nn.Module:
    embedding_dim = int(config["embedding_dim"])
    if name == "fnn":
        return FNNTextClassifier(
            vocab_size,
            embedding_dim,
            int(config["fnn_hidden_dim"]),
            float(config["fnn_dropout"]),
        )
    if name == "cnn":
        return CNNTextClassifier(
            vocab_size,
            embedding_dim,
            int(config["cnn_filters"]),
            [int(value) for value in config["cnn_kernel_sizes"]],
            float(config["cnn_dropout"]),
        )
    if name == "lstm":
        return LSTMTextClassifier(
            vocab_size,
            embedding_dim,
            int(config["lstm_hidden_dim"]),
            int(config["lstm_layers"]),
            float(config["lstm_dropout"]),
        )
    raise ValueError(f"Unknown word-level neural model: {name}")


def _bert_loader(
    encodings: dict[str, torch.Tensor],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def predict_bert_model(
    model: nn.Module,
    encodings: dict[str, torch.Tensor],
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = _bert_loader(encodings, labels, batch_size, False, 0)
    logits_parts: list[torch.Tensor] = []
    observed_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            logits_parts.append(output.logits.cpu())
            observed_parts.append(batch_labels)
    logits = torch.cat(logits_parts)
    observed = torch.cat(observed_parts).numpy()
    probability = torch.softmax(logits, dim=1)[:, 1].numpy()
    prediction = logits.argmax(dim=1).numpy()
    macro_f1 = float(f1_score(observed, prediction, average="macro", zero_division=0))
    return prediction, probability, macro_f1


def train_bert_model(
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
    validation_labels: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, Any, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    model_id = str(config["bert_model_id"])
    revision = str(config["bert_revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    train_encodings = tokenizer(
        train_texts,
        padding="max_length",
        truncation=True,
        max_length=int(config["bert_max_length"]),
        return_tensors="pt",
    )
    validation_encodings = tokenizer(
        validation_texts,
        padding="max_length",
        truncation=True,
        max_length=int(config["bert_max_length"]),
        return_tensors="pt",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=2,
    ).to(device)
    batch_size = int(config["bert_batch_size"])
    loader = _bert_loader(train_encodings, train_labels, batch_size, True, seed)
    loss_function = nn.CrossEntropyLoss(weight=class_weights(train_labels).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["bert_learning_rate"]),
        weight_decay=float(config["bert_weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["bert_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for input_ids, attention_mask, batch_labels in loader:
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            loss = loss_function(output.logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
            examples += len(batch_labels)
        _, _, validation_f1 = predict_bert_model(
            model,
            validation_encodings,
            validation_labels,
            batch_size,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(examples, 1),
                "validation_macro_f1": validation_f1,
            }
        )
        if validation_f1 > best_f1 + 1e-8:
            best_f1 = validation_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    return model, tokenizer, train_encodings, validation_encodings, {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "fit_seconds": time.perf_counter() - started,
        "history": history,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pretrained_model_id": model_id,
        "pretrained_revision": revision,
    }
