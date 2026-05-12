"""
DL-эксперименты для классификации уязвимостей смарт-контрактов.

Четыре маленькие модели на одном и том же 5-fold CV что и baseline_comparison.py:
1. MLP-small — табличные признаки, 1 скрытый слой
2. MLP-deep - табличные признаки, 2 скрытых слоя + Dropout
3. MLP-bn — табличные признаки, 2 скрытых слоя + BatchNorm + Dropout
4. CharCNN-tiny — сырой исходник контракта (truncated 2k bytes), 1D CNN

Запуск:
conda run -n dev_env python dl_models.py
# или быстрее, для проверки:
conda run -n dev_env python dl_models.py --quick

Результат: dl_results.json рядом с этим файлом.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from baseline_comparison import (
    REPO_ROOT,
    build_feature_matrix,
    load_dataset,
    read_source,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")
RESULTS_PATH = Path(__file__).parent / "dl_results.json"

CHAR_MAX_LEN = 2048
CHAR_VOCAB = 128  # ASCII


#  модели 

class MLPSmall(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPDeep(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLPBN(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CharCNNTiny(nn.Module):
    def __init__(self, n_classes, vocab=CHAR_VOCAB, embed_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        # x: (B, L) long
        e = self.embed(x).transpose(1, 2)  # (B, E, L)
        h = torch.relu(self.conv1(e))
        h = self.pool(h)
        h = torch.relu(self.conv2(h))
        h = h.mean(dim=-1)  # global avg pool
        return self.fc(h)


#  данные 

def encode_source_to_chars(src):
    arr = np.zeros(CHAR_MAX_LEN, dtype=np.int64)
    if not src:
        return arr
    b = src.encode("utf-8", errors="ignore")[:CHAR_MAX_LEN]
    for i, ch in enumerate(b):
        arr[i] = ch if ch < CHAR_VOCAB else 0
    return arr


def build_char_matrix(df):
    rows = [encode_source_to_chars(read_source(p)) for p in df["path"].tolist()]
    return np.stack(rows, axis=0)


#  обучение 

def train_one_fold(model_factory, X_train, y_train, X_val, y_val,
                   n_classes, epochs, batch=16, lr=1e-3, long_input=False):
    model = model_factory(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dtype = torch.long if long_input else torch.float32
    Xtr = torch.tensor(X_train, dtype=dtype, device=DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    Xva = torch.tensor(X_val, dtype=dtype, device=DEVICE)

    n = len(Xtr)

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], ytr[idx]
            if xb.size(0) < 2 and isinstance(model, MLPBN):
                continue  # BatchNorm нужен >=2
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        logits = model(Xva)
        preds = logits.argmax(dim=1).cpu().numpy()

    return preds, elapsed


def run_cv(name, model_factory, X, y, n_classes, epochs, batch=16, long_input=False,
           arch_str="", hp_str=""):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_all_true, y_all_pred = [], []
    total_time = 0.0

    for tr, va in cv.split(X, y):
        X_tr, X_va = X[tr], X[va]
        if not long_input:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        preds, elapsed = train_one_fold(
            model_factory, X_tr, y[tr], X_va, y[va],
            n_classes=n_classes, epochs=epochs, batch=batch,
            long_input=long_input,
        )
        total_time += elapsed
        y_all_true.extend(y[va].tolist())
        y_all_pred.extend(preds.tolist())

    acc = accuracy_score(y_all_true, y_all_pred)
    f1w = f1_score(y_all_true, y_all_pred, average="weighted", zero_division=0)
    f1m = f1_score(y_all_true, y_all_pred, average="macro", zero_division=0)

    print(f"{name:14s}  acc {acc:.3f}  f1w {f1w:.3f}  f1m {f1m:.3f}  time {total_time:.2f}s")

    return {
        "method": name,
        "architecture": arch_str,
        "hyperparameters": hp_str,
        "accuracy": round(float(acc), 3),
        "f1_weighted": round(float(f1w), 3),
        "f1_macro": round(float(f1m), 3),
        "train_time_s": round(total_time, 2),
        "epochs": epochs,
        "batch": batch,
    }


#  main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="меньше эпох для быстрой проверки")
    args = parser.parse_args()

    epochs_mlp = 10 if args.quick else 30
    epochs_cnn = 5 if args.quick else 15

    df, raw = load_dataset()
    print(f"loaded {len(df)} contracts, {df['target'].nunique()} classes")

    X_tab = build_feature_matrix(df, raw).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["target"].values)
    n_classes = len(le.classes_)
    in_dim = X_tab.shape[1]

    print(f"tabular X: {X_tab.shape}, classes: {n_classes}")
    print("building char matrix ...")
    X_char = build_char_matrix(df)
    print(f"char X: {X_char.shape}")
    print()

    hp_common = f"Adam lr=1e-3, batch=16, epochs={epochs_mlp}, seed={SEED}"
    hp_cnn = f"Adam lr=1e-3, batch=16, epochs={epochs_cnn}, seed={SEED}"

    results = []

    results.append(run_cv(
        "MLP-small",
        lambda nc: MLPSmall(in_dim, nc),
        X_tab, y, n_classes, epochs=epochs_mlp,
        arch_str=f"Linear({in_dim}->32)->ReLU->Linear(32->{n_classes})",
        hp_str=hp_common,
    ))
    results.append(run_cv(
        "MLP-deep",
        lambda nc: MLPDeep(in_dim, nc),
        X_tab, y, n_classes, epochs=epochs_mlp,
        arch_str=f"{in_dim}->64->32->{n_classes}, ReLU, Dropout 0.3",
        hp_str=hp_common,
    ))
    results.append(run_cv(
        "MLP-bn",
        lambda nc: MLPBN(in_dim, nc),
        X_tab, y, n_classes, epochs=epochs_mlp,
        arch_str=f"{in_dim}->64->32->{n_classes}, BatchNorm, ReLU, Dropout 0.3",
        hp_str=hp_common,
    ))
    results.append(run_cv(
        "CharCNN-tiny",
        lambda nc: CharCNNTiny(nc),
        X_char, y, n_classes, epochs=epochs_cnn, long_input=True,
        arch_str="Embed(128,8)->Conv1d(8->16,k=5)->Pool->Conv1d(16->16,k=5)->GAP->Linear(16->C)",
        hp_str=hp_cnn,
    ))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
