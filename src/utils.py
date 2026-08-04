"""
Shared stuff for the MCQ answer predictor.
Project T22026 - 24f2005363

Everything that both train.py and inference.py need lives here so the two
scripts stay short and cannot drift apart (same split, same text join,
same model classes).
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# the 5 answer options
CHOICES = ['A', 'B', 'C', 'D', 'E']

# defaults, same values the notebook used
SPLIT_SEED = 7
VAL_SIZE = 0.15
EMB_NAME_B = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
EMB_NAME_C = 'all-mpnet-base-v2'
WB_PROJECT = '24f2005363-t22026'
HF_REPO_SUFFIX = 'mcq-answer-predictor-t22026'
KAGGLE_PATH = '/kaggle/input/competitions/smart-mcq-solver-challenge'


# ---------------------------------------------------------------- basics

def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_data_dir(user_path=None):
    """Pick the folder that has train.csv in it."""
    tries = []
    if user_path:
        tries.append(user_path)
    tries += [KAGGLE_PATH, 'data', './data', '.', '/kaggle/input/smart-mcq-solver-challenge']
    for p in tries:
        if os.path.exists(os.path.join(p, 'train.csv')):
            return p
    raise FileNotFoundError('could not find train.csv, tried: ' + str(tries))


# ---------------------------------------------------------------- data

def load_data(data_dir=None):
    d = find_data_dir(data_dir)
    df_train = pd.read_csv(os.path.join(d, 'train.csv'))
    df_test = pd.read_csv(os.path.join(d, 'test.csv'))
    sub_path = os.path.join(d, 'sample_submission.csv')
    if os.path.exists(sub_path):
        df_sub = pd.read_csv(sub_path)
    else:
        # fall back so the scripts still run if the sample file is missing
        id_col = 'id' if 'id' in df_test.columns else df_test.columns[0]
        df_sub = pd.DataFrame({'id': df_test[id_col], 'Prediction': ''})
    print('train shape', df_train.shape, '| test shape', df_test.shape)
    return df_train, df_test, df_sub


def join_row(r):
    """Question text plus all 5 options glued into one string."""
    line = str(r['prompt'])
    for c in CHOICES:
        line = line + ' ' + c + ') ' + str(r[c])
    return line


def add_text(df):
    df = df.copy()
    df['text'] = df.apply(join_row, axis=1)
    return df


def make_split(df_train, seed=SPLIT_SEED, val_size=VAL_SIZE):
    """
    Group split on the prompt so the same question never sits in both
    train and val. Returns the two frames, the labels and the encoder.
    """
    enc = LabelEncoder()
    y_full = enc.fit_transform(df_train['answer'])

    sp = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_i, va_i = next(sp.split(df_train, y_full, groups=df_train['prompt']))

    df_tr = df_train.iloc[tr_i].reset_index(drop=True)
    df_va = df_train.iloc[va_i].reset_index(drop=True)
    y_tr = y_full[tr_i]
    y_va = y_full[va_i]

    overlap = set(df_tr['prompt']) & set(df_va['prompt'])
    print('train rows', len(df_tr), 'val rows', len(df_va), '| prompt overlap (want 0):', len(overlap))
    return df_tr, df_va, y_tr, y_va, enc


class SimpleData(Dataset):
    """Plain feature/label dataset. Labels are optional for test time."""

    def __init__(self, feats, labels=None):
        self.feats = torch.tensor(np.asarray(feats), dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.feats[idx], self.labels[idx]
        return self.feats[idx]


# ---------------------------------------------------------------- metrics

def get_map3(true_y, prob_arr):
    """Mean average precision at 3, the competition metric."""
    true_y = np.asarray(true_y)
    total = 0.0
    for i in range(len(true_y)):
        order = prob_arr[i].argsort()[::-1][:3]
        s = 0.0
        for rank in range(len(order)):
            if order[rank] == true_y[i]:
                s = 1.0 / (rank + 1)
        total = total + s
    return total / len(true_y)


def score_all(y_true, prob):
    pred = prob.argmax(axis=1)
    return {
        'acc': accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred, average='macro'),
        'map3': get_map3(y_true, prob),
    }


def write_submission(prob_arr, df_sub, fname):
    """Top 3 letters per row, space separated, in the Prediction column."""
    lines = []
    for i in range(len(prob_arr)):
        order = prob_arr[i].argsort()[::-1][:3]
        lines.append(' '.join([CHOICES[j] for j in order]))
    out = df_sub.copy()
    if len(out) != len(lines):
        raise ValueError('row count mismatch: sub has %d, preds have %d' % (len(out), len(lines)))
    out['Prediction'] = lines
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    out.to_csv(fname, index=False)
    print('wrote', fname)
    return out


# ---------------------------------------------------------------- models

class NetA(nn.Module):
    """Model A - plain net on top of tfidf counts."""

    def __init__(self, in_size, out_size=5):
        super().__init__()
        self.l1 = nn.Linear(in_size, 400)
        self.l2 = nn.Linear(400, 128)
        self.l3 = nn.Linear(128, out_size)
        self.drop = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.drop(x)
        x = self.act(self.l2(x))
        x = self.l3(x)
        return x


class NetB(nn.Module):
    """Model B - small head on MiniLM sentence embeddings."""

    def __init__(self, in_size, out_size=5):
        super().__init__()
        self.l1 = nn.Linear(in_size, 200)
        self.l2 = nn.Linear(200, out_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.drop(x)
        x = self.l2(x)
        return x


class NetC(nn.Module):
    """Model C - wider head on MPNet embeddings."""

    def __init__(self, in_size=768, out_size=5):
        super().__init__()
        self.l1 = nn.Linear(in_size, 384)
        self.l2 = nn.Linear(384, 96)
        self.l3 = nn.Linear(96, out_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.drop(x)
        x = self.act(self.l2(x))
        x = self.l3(x)
        return x


# ---------------------------------------------------------------- train loop

def run_epoch(model, loader, opt, loss_fn, train_mode, dev):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_pred, all_true, all_prob = [], [], []

    torch.set_grad_enabled(train_mode)
    for xb, yb in loader:
        xb = xb.to(dev)
        yb = yb.to(dev)
        if train_mode:
            opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        if train_mode:
            loss.backward()
            opt.step()
        total_loss += loss.item()
        p = torch.softmax(out, dim=1).detach().cpu().numpy()
        all_prob.append(p)
        all_pred.extend(p.argmax(axis=1))
        all_true.extend(yb.cpu().numpy())
    torch.set_grad_enabled(True)

    return total_loss / max(len(loader), 1), np.array(all_pred), np.array(all_true), np.vstack(all_prob)


def predict_probs(model, feats, dev, batch_size=64):
    """Softmax probs for a feature matrix, no labels needed."""
    model.eval()
    out_all = []
    with torch.no_grad():
        for xb in DataLoader(SimpleData(feats), batch_size=batch_size):
            out = model(xb.to(dev))
            out_all.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.vstack(out_all)


# ---------------------------------------------------------------- embeddings

def embed_texts(model_name, texts_list, dev, batch_size=32):
    """
    Encode a list of text lists with one sentence transformer, then drop the
    encoder so the gpu memory comes back.
    """
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model_name, device=str(dev))
    out = []
    for txt in texts_list:
        out.append(enc.encode(txt, batch_size=batch_size, show_progress_bar=True))
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- logins

def get_secret(name):
    """Kaggle secrets first, then plain env vars for running anywhere else."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)


def hf_login():
    """Log into the hub and return the username, or None if no token."""
    tok = get_secret('HF_TOKEN')
    if not tok:
        print('no HF_TOKEN found, skipping huggingface login')
        return None
    from huggingface_hub import HfApi, login
    login(token=tok)
    user = HfApi().whoami()['name']
    print('logged into huggingface as:', user)
    return user


class Run:
    """
    Thin wrapper over a wandb run. If wandb is off or the key is missing it
    turns into a no-op so the script still finishes.
    """

    def __init__(self, enabled=True, **kw):
        self.run = None
        if not enabled:
            return
        key = get_secret('WANDB_API_KEY')
        if not key:
            print('no WANDB_API_KEY found, running without wandb')
            return
        try:
            import wandb
            wandb.login(key=key)
            self.run = wandb.init(**kw)
        except Exception as e:
            print('wandb init failed, carrying on without it:', e)
            self.run = None

    def log(self, d):
        if self.run is not None:
            self.run.log(d)

    def log_image(self, key, path):
        if self.run is not None:
            import wandb
            self.run.log({key: wandb.Image(path)})

    def set_summary(self, k, v):
        if self.run is not None:
            self.run.summary[k] = v

    def finish(self):
        if self.run is not None:
            self.run.finish()


# ---------------------------------------------------------------- small io

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
