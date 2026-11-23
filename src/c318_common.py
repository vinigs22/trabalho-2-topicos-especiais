"""Funções compartilhadas do Trabalho 2.

Este módulo concentra o que precisa ser IDÊNTICO em todos os notebooks
(pipeline manual, TPOT, AutoGluon e consolidação):

* leitura do CSV e normalização dos textos;
* filtros do Trabalho 1 (5 classes sem suporte e 8 registros implausíveis);
* partição 60/20/20 estratificada (semente 42), salva em data/splits/;
* asserções que garantem que todos usam exatamente os mesmos índices;
* métricas (F1 macro primária + secundárias) e medição de tempo/memória.

Uso nos outros notebooks:

    from c318_common import *
    ROOT = find_repo_root()
    train_df, val_df, test_df = load_split(ROOT)   # já confere os índices
"""
from __future__ import annotations

import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
#Constantes do protocolo (enunciado, Seções 4.1 e 5.1)
# --------------------------------------------------------------------------
SEED = 42
SEEDS = [42, 101, 223, 347, 409]
TARGET = "Diagnosis"
ID_COLS = ["Sl No"]                 # identificadora: descartada em todos os sistemas
DATA_REL = Path("data") / "HPLC_data.csv"
SPLIT_REL = Path("data") / "splits" / "split_indices.csv"

#Filtros do Trabalho 1 (Seção 2.10 / 3.1 do notebook do T1)
CLASSES_REMOVIDAS = [
    "Beta trait with high foetal haemoglobin",   # 6 exemplos: inviabiliza CV de 5 folds
    "Delta beta carrier",                        # 1
    "HPFH trait",                                # 2
    "HbD disease",                               # 1
    "HbS-Beta Thalassaemia",                     # 2
]
N_LINHAS_CSV = 13031
N_LINHAS_MODELO = 13011
N_CLASSES = 8
TAMANHOS_PARTICAO = {"train": 7806, "val": 2602, "test": 2603}

#Atributo derivado do T1 (engenharia de atributos, por registro, sem parâmetros)
DERIVED_COLS = ["HbS_peak_presente", "HbD_peak_presente"]

CAT_COLS = ["Gender", "Religion", "Present District", "Weekness", "Jaundice"]


# --------------------------------------------------------------------------
#Caminhos
# --------------------------------------------------------------------------
def find_repo_root(start: str | Path | None = None) -> Path:
    """Sobe a partir da pasta atual até achar data/HPLC_data.csv.

    Funciona tanto rodando de notebooks/ quanto da raiz do repositório,
    localmente (JupyterLab) ou no Colab depois de `%cd` para o repositório.
    Não usa Google Drive nem caminhos fixos em /content.
    """
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / DATA_REL).exists():
            return p
    raise FileNotFoundError(
        "Não encontrei data/HPLC_data.csv. Abra o notebook de dentro do "
        "repositório clonado (pasta notebooks/ ou raiz)."
    )


# --------------------------------------------------------------------------
#Leitura e filtros
# --------------------------------------------------------------------------
def load_raw(root: Path) -> pd.DataFrame:
    """Lê o CSV e remove espaços sobrando nos textos ("Normal  " -> "Normal")."""
    df = pd.read_csv(root / DATA_REL)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith(("str", "string")):
            df[c] = df[c].str.strip()
    assert df.shape == (N_LINHAS_CSV, 19), df.shape
    return df


def outlier_mask(df: pd.DataFrame) -> pd.Series:
    """Limites laboratoriais fixos do T1 (regra por registro, sem estatística da base)."""
    return (df["MCHC"] > 50) | (df["HB"] > 22) | (df["MCH"] > 45)


def apply_t1_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove as 5 classes sem suporte e os 8 registros implausíveis (antes da partição)."""
    counts = df[TARGET].value_counts()
    regra = set(counts[counts < 3].index) | {"Beta trait with high foetal haemoglobin"}
    assert regra == set(CLASSES_REMOVIDAS), "Regra de remoção diferente da do T1"
    out = df[~df[TARGET].isin(CLASSES_REMOVIDAS) & ~outlier_mask(df)].copy()
    assert len(out) == N_LINHAS_MODELO and out[TARGET].nunique() == N_CLASSES
    assert out.index.is_unique
    return out


def parse_age(value) -> float:
    """'14 Yrs 11 month' -> 14.9167 (anos decimais). Ausente -> NaN."""
    if pd.isna(value):
        return np.nan
    m = re.match(r"(\d+)\s*Yrs\s*(\d+)\s*month", str(value))
    return int(m.group(1)) + int(m.group(2)) / 12 if m else np.nan


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Age_years"] = out["Age"].apply(parse_age)
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Atributo derivado do T1: presença de pico na janela S (HbS) e em Unknown (HbD).

    É calculado registro a registro, sem estimar parâmetros, então não há
    risco de vazamento ao aplicá-lo em treino, validação e teste.
    """
    out = df.copy()
    out["HbS_peak_presente"] = (out["S-Window"] > 0).astype(int)
    out["HbD_peak_presente"] = (out["Unknown"] > 0).astype(int)
    return out


# --------------------------------------------------------------------------
#Partição 60/20/20
# --------------------------------------------------------------------------
def make_split(df_model: pd.DataFrame):
    """Mesma chamada do T1: 60/20/20 estratificado, random_state=42."""
    train_df, temp_df = train_test_split(
        df_model, test_size=0.4, stratify=df_model[TARGET], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[TARGET], random_state=SEED)
    return train_df, val_df, test_df


def split_table(train_df, val_df, test_df) -> pd.DataFrame:
    rows = []
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        rows.append(pd.DataFrame({
            "row_index": part.index.astype(int),
            "sl_no": part["Sl No"].astype(int).to_numpy(),
            "subset": name,
            "diagnosis": part[TARGET].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def save_split(root: Path, train_df, val_df, test_df) -> Path:
    path = root / SPLIT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    split_table(train_df, val_df, test_df).to_csv(path, index=False)
    return path


def assert_split(train_df, val_df, test_df, df_model: pd.DataFrame | None = None):
    """Asserções explícitas exigidas pelo enunciado (Seção 4.1)."""
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        assert len(part) == TAMANHOS_PARTICAO[name], (name, len(part))
        assert part[TARGET].nunique() == N_CLASSES, f"classe ausente em {name}"
    a, b, c = (set(p.index) for p in (train_df, val_df, test_df))
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
    if df_model is not None:
        assert a | b | c == set(df_model.index)


def load_split(root: Path, with_age_years: bool = False, with_derived: bool = False):
    """Recria a partição e CONFERE com os índices salvos no repositório.

    Retorna (train_df, val_df, test_df) com todas as colunas originais.
    Se os índices recalculados não baterem com data/splits/split_indices.csv,
    o notebook para com erro (protege contra versões diferentes de bibliotecas).
    """
    df_model = apply_t1_filters(load_raw(root))
    saved = pd.read_csv(root / SPLIT_REL)
    parts = []
    for name in ("train", "val", "test"):
        idx = saved.loc[saved["subset"] == name, "row_index"].to_numpy()
        part = df_model.loc[idx]
        assert (part["Sl No"].to_numpy() ==
                saved.loc[saved["subset"] == name, "sl_no"].to_numpy()).all()
        parts.append(part)
    recomputed = make_split(df_model)
    for p_saved, p_new in zip(parts, recomputed):
        assert p_saved.index.equals(p_new.index), "Partição diferente da salva!"
    assert_split(*parts, df_model=df_model)
    if with_age_years:
        parts = [add_age_years(p) for p in parts]
    if with_derived:
        parts = [add_derived_features(p) for p in parts]
    return tuple(parts)


# --------------------------------------------------------------------------
#Métricas (primária: F1 macro)
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, labels=None) -> dict:
    kw = dict(labels=labels, zero_division=0)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", **kw),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", **kw),
        "recall_macro": recall_score(y_true, y_pred, average="macro", **kw),
    }


def fmt_max_mean_std(values) -> str:
    """Formato do artigo para F1: 'máx (média ± dp)'."""
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.max():.4f} ({v.mean():.4f} ± {sd:.4f})"


def fmt_min_mean_std(values) -> str:
    """Formato do artigo para tempo: 'mín (média ± dp)'."""
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.min():.2f} ({v.mean():.2f} ± {sd:.2f})"


# --------------------------------------------------------------------------
#Tempo de parede e pico de memória (processo + subprocessos)
# --------------------------------------------------------------------------
@contextmanager
def measure(interval: float = 0.05):
    """Mede tempo de parede e pico de RSS somando o processo e seus filhos.

    Método declarado: amostragem do RSS via psutil a cada `interval` segundos,
    em uma thread paralela; o pico é o maior valor de
    RSS(processo) + soma(RSS dos subprocessos, recursivo) observado.
    Uso:
        with measure() as m:
            modelo.fit(X, y)
        m["wall_s"], m["peak_mb"]
    """
    import psutil

    proc = psutil.Process()
    result = {"wall_s": None, "peak_mb": None, "baseline_mb": None}
    stop = threading.Event()
    peak = [0]

    def total_rss():
        rss = proc.memory_info().rss
        for ch in proc.children(recursive=True):
            try:
                rss += ch.memory_info().rss
            except psutil.Error:
                pass
        return rss

    def sampler():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], total_rss())
            except psutil.Error:
                pass
            stop.wait(interval)

    result["baseline_mb"] = total_rss() / 2**20
    th = threading.Thread(target=sampler, daemon=True)
    t0 = time.perf_counter()
    th.start()
    try:
        yield result
    finally:
        result["wall_s"] = time.perf_counter() - t0
        stop.set()
        th.join()
        peak[0] = max(peak[0], total_rss())
        result["peak_mb"] = peak[0] / 2**20


def hardware_info() -> dict:
    import os
    import platform
    import sys

    import psutil
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus_logicas": os.cpu_count(),
        "memoria_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "colab": "google.colab" in sys.modules,
    }
