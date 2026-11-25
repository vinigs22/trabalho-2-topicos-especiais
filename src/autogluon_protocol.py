"""Protocolo oficial do AutoGluon para o Trabalho 2.

O módulo mantém validação e teste fora do ``fit``. Cada execução cria seu
próprio holdout estratificado usando somente o treino oficial e salva o
``TabularPredictor`` em uma pasta exclusiva. As funções não consultam caminhos
globais de resultados: o notebook escolhe explicitamente todos os destinos.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from c318_common import (
    CAT_COLS,
    DERIVED_COLS,
    N_CLASSES,
    SEEDS,
    TARGET,
    add_age_years,
    add_derived_features,
    compute_metrics,
    measure,
)

PRESET = "medium_quality"
HOLDOUT_FRAC = 0.20
LABEL_COUNT_THRESHOLD = 1

BASE_FEATURES = [
    "Age_years",
    "HbA2",
    "HbF",
    "S-Window",
    "Unknown",
    "RBC",
    "HB",
    "MCV",
    "MCHC",
    "RDWcv",
    *CAT_COLS,
]
AUTOGLOUON_FEATURES = [*BASE_FEATURES, *DERIVED_COLS]

OFFICIAL_RUNS = [
    (5, 42),
    (5, 101),
    (5, 223),
    (5, 347),
    (5, 409),
    (30, 42),
    (60, 42),
]


def run_id(budget_min: int, seed: int, with_derived: bool = True) -> str:
    """Identificador estável de uma execução oficial ou de ablação."""
    prefix = "budget" if with_derived else "ablacao_sem_derivado"
    return f"{prefix}_{int(budget_min):02d}m_seed_{int(seed)}"


def prepare_autogluon_data(part: pd.DataFrame, with_derived: bool = True) -> pd.DataFrame:
    """Prepara as 17 (ou 15) entradas e o alvo sem aprender parâmetros.

    AutoGluon recebe categorias e ausências em sua representação original para
    que imputação/codificação sejam ajustadas apenas dentro da ferramenta.
    ``Age_years`` e os dois indicadores de pico são regras por registro.
    """
    work = add_age_years(part)
    if with_derived:
        work = add_derived_features(work)

    features = AUTOGLOUON_FEATURES if with_derived else BASE_FEATURES
    out = work[[*features, TARGET]].copy()

    assert out.columns.tolist() == [*features, TARGET]
    assert out.index.equals(part.index)
    assert out[TARGET].notna().all()
    assert out[TARGET].nunique() == N_CLASSES
    assert "Sl No" not in out.columns and "Age" not in out.columns
    assert "HbA0" not in out.columns and "MCH" not in out.columns
    return out


def make_inner_holdout(
    train_data: pd.DataFrame,
    seed: int,
    holdout_frac: float = HOLDOUT_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide apenas o treino oficial em ajuste/tuning, estratificado por seed."""
    if TARGET not in train_data:
        raise ValueError(f"Coluna alvo ausente: {TARGET}")
    if seed not in SEEDS:
        raise ValueError(f"Seed não declarada no protocolo: {seed}")
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout_frac deve estar entre 0 e 1")

    fit_data, tuning_data = train_test_split(
        train_data,
        test_size=holdout_frac,
        stratify=train_data[TARGET],
        random_state=seed,
    )
    fit_idx, tuning_idx = set(fit_data.index), set(tuning_data.index)
    assert fit_idx.isdisjoint(tuning_idx)
    assert fit_idx | tuning_idx == set(train_data.index)
    assert fit_data[TARGET].nunique() == N_CLASSES
    assert tuning_data[TARGET].nunique() == N_CLASSES
    return fit_data, tuning_data


def set_global_seed(seed: int) -> None:
    """Configura geradores globais; a seed também determina o holdout interno."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def index_hash(values: Iterable) -> str:
    """Hash ordenado dos índices para auditoria sem duplicar a tabela de split."""
    payload = ",".join(str(v) for v in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def predictor_artifact_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and any(path.iterdir())


def assert_new_predictor_path(path: str | Path) -> Path:
    """Impede sobrescrever inclusive uma execução parcial do AutoGluon."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"A pasta do predictor já existe: {path}. Preserve-a para auditoria "
            "ou remova-a manualmente somente se a execução for descartada."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_predictor(path: str | Path):
    """Cria o predictor oficial, preservando todas as oito classes raras."""
    from autogluon.tabular import TabularPredictor

    path = assert_new_predictor_path(path)
    return TabularPredictor(
        label=TARGET,
        problem_type="multiclass",
        eval_metric="f1_macro",
        path=str(path),
        verbosity=3,
        log_to_file=True,
        log_file_path="auto",
        learner_kwargs={"label_count_threshold": LABEL_COUNT_THRESHOLD},
    )


def predictor_classes(predictor) -> list:
    classes = list(predictor.classes_)
    if len(classes) != N_CLASSES:
        raise AssertionError(
            f"AutoGluon preservou {len(classes)} classes, esperado: {N_CLASSES}"
        )
    return classes


def predictions_frame(predictor, eval_data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prediz com ``model_best`` e devolve probabilidades e métricas comuns."""
    features = eval_data.drop(columns=[TARGET])
    with measure() as pred_measure:
        pred = predictor.predict(features)
        proba = predictor.predict_proba(features)

    pred = pd.Series(pred, index=eval_data.index, name="y_pred")
    y_true = eval_data[TARGET]
    labels = sorted(predictor_classes(predictor))
    metrics = {
        **compute_metrics(y_true, pred, labels=labels),
        "tempo_inferencia_s": pred_measure["wall_s"],
        "pico_memoria_inferencia_mb": pred_measure["peak_mb"],
    }

    predictions = pd.DataFrame({"y_true": y_true, "y_pred": pred}, index=eval_data.index)
    if not isinstance(proba, pd.DataFrame):
        proba = pd.DataFrame(proba, index=eval_data.index, columns=predictor_classes(predictor))
    proba = proba.reindex(eval_data.index)
    for label in proba.columns:
        predictions[f"proba_{label}"] = proba[label]
    assert predictions.index.equals(eval_data.index)
    return predictions, metrics


def fit_and_evaluate(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    model_path: str | Path,
    budget_min: int,
    seed: int,
):
    """Ajusta só no treino/holdout interno e avalia no conjunto externo informado."""
    set_global_seed(seed)
    fit_data, tuning_data = make_inner_holdout(train_data, seed)
    predictor = build_predictor(model_path)

    with measure() as fit_measure:
        predictor.fit(
            train_data=fit_data,
            tuning_data=tuning_data,
            time_limit=int(budget_min * 60),
            presets=PRESET,
            num_gpus=0,
            fit_strategy="sequential",
        )

    classes = predictor_classes(predictor)
    if set(classes) != set(train_data[TARGET].unique()):
        raise AssertionError("As classes do predictor diferem das classes do treino oficial")

    predictions, metrics = predictions_frame(predictor, eval_data)
    metrics.update(
        {
            "tempo_ajuste_s": fit_measure["wall_s"],
            "pico_memoria_ajuste_mb": fit_measure["peak_mb"],
            "pico_memoria_mb": max(
                fit_measure["peak_mb"], metrics["pico_memoria_inferencia_mb"]
            ),
        }
    )
    split_meta = {
        "n_fit_interno": len(fit_data),
        "n_tuning_interno": len(tuning_data),
        "hash_indices_fit_interno": index_hash(fit_data.index),
        "hash_indices_tuning_interno": index_hash(tuning_data.index),
        "classes_fit_interno": fit_data[TARGET].value_counts().sort_index().to_dict(),
        "classes_tuning_interno": tuning_data[TARGET].value_counts().sort_index().to_dict(),
    }
    return predictor, metrics, predictions, split_meta


def load_predictor(path: str | Path):
    from autogluon.tabular import TabularPredictor

    if not predictor_artifact_exists(path):
        raise FileNotFoundError(f"Predictor ausente ou vazio: {path}")
    predictor = TabularPredictor.load(str(path))
    predictor_classes(predictor)
    return predictor


def expected_run_pairs() -> set[tuple[int, int]]:
    return set(OFFICIAL_RUNS)


def validate_run_matrix(validation_runs: pd.DataFrame) -> None:
    required = {"run_id", "budget_min", "seed", "f1_macro"}
    if not required <= set(validation_runs.columns):
        raise ValueError(f"Colunas ausentes: {sorted(required - set(validation_runs.columns))}")
    if validation_runs["run_id"].duplicated().any():
        raise ValueError("Há run_id duplicado na curva de validação")
    pairs = list(zip(validation_runs["budget_min"].astype(int), validation_runs["seed"].astype(int)))
    if len(pairs) != len(set(pairs)):
        raise ValueError("Há par budget/seed duplicado na curva de validação")
    if set(pairs) != expected_run_pairs():
        missing = sorted(expected_run_pairs() - set(pairs))
        extra = sorted(set(pairs) - expected_run_pairs())
        raise ValueError(f"Matriz de runs incompleta ou inválida; ausentes={missing}, extras={extra}")


def select_budget(validation_runs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Seleciona pela média de F1-macro externa; empate favorece menor budget."""
    validate_run_matrix(validation_runs)
    summary = (
        validation_runs.groupby("budget_min", as_index=False)
        .agg(
            f1_macro_media=("f1_macro", "mean"),
            f1_macro_dp=("f1_macro", "std"),
            f1_weighted_media=("f1_weighted", "mean"),
            tempo_ajuste_s_medio=("tempo_ajuste_s", "mean"),
            pico_memoria_mb_max=("pico_memoria_mb", "max"),
            repeticoes=("seed", "nunique"),
        )
        .fillna({"f1_macro_dp": 0.0})
        .sort_values(["f1_macro_media", "budget_min"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary, int(summary.iloc[0]["budget_min"])


def append_unique_csv(path: str | Path, row: dict, key: str = "run_id") -> pd.DataFrame:
    """Acrescenta uma linha com chave única e substituição atômica do CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    if key in current and row[key] in set(current[key]):
        raise ValueError(f"{key} já registrado em {path}: {row[key]}")
    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    updated.to_csv(tmp, index=False)
    tmp.replace(path)
    return updated


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def package_versions() -> dict[str, str | None]:
    packages = [
        "autogluon.tabular",
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "xgboost",
        "xgboost-cpu",
        "fastai",
        "psutil",
    ]
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
