"""Funções do experimento oficial de TPOT para o Trabalho 2.

O módulo preserva o protocolo comum do projeto:

* usa somente as linhas já filtradas e particionadas por ``c318_common``;
* aplica apenas transformações determinísticas por registro antes do TPOT;
* mantém imputação/codificação ajustável dentro das pipelines candidatas;
* mede tempo de parede e pico de RSS, salva predições e registra a pipeline.

TPOT 1.1.0 exige rótulos numéricos em algumas pipelines. A API codifica os
rótulos internamente quando necessário; as funções de avaliação os devolvem
ao espaço original de ``Diagnosis``.
"""
from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from c318_common import (
    CAT_COLS,
    ID_COLS,
    SEED,
    TARGET,
    add_age_years,
    add_derived_features,
    compute_metrics,
    measure,
)


@contextmanager
def fit_timeout(seconds: int):
    """Impõe limite externo no Colab para evitar ajuste preso no encerramento."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(_signum, _frame):
        raise TimeoutError(f"Ajuste TPOT excedeu o limite externo de {seconds}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# Mesmas decisões de atributos da referência manual do Trabalho 1.
DROP_FOR_TPOT = [*ID_COLS, "Age", "HbA0", "MCH", TARGET]
TPOT_FEATURES = [
    "Age_years", "HbA2", "HbF", "S-Window", "Unknown", "RBC", "HB",
    "MCV", "MCHC", "RDWcv", *CAT_COLS,
    "HbS_peak_presente", "HbD_peak_presente",
]


def prepare_tpot_input(part: pd.DataFrame, with_derived: bool = True):
    """Retorna X e y para TPOT sem aprender parâmetros fora da ferramenta.

    ``Age_years`` e os dois atributos de pico são calculados linha a linha,
    como no Trabalho 1. Categorias ausentes viram a string fixa
    ``Nao_informado`` e a idade ausente vira -1; ambas são regras por
    registro, sem mediana, frequência ou estatística calculada no conjunto.
    Isso deixa o TPOT responsável pela codificação e por quaisquer etapas
    aprendidas de sua pipeline interna.
    """
    work = add_age_years(part)
    if with_derived:
        work = add_derived_features(work)

    # Há ausências apenas em Age e nas colunas categóricas neste conjunto.
    # Os sentinelas tornam a entrada válida sem estimar qualquer parâmetro.
    work["Age_years"] = work["Age_years"].fillna(-1.0)
    for col in CAT_COLS:
        work[col] = work[col].astype("string").fillna("Nao_informado").astype(str)

    drop = list(DROP_FOR_TPOT)
    if not with_derived:
        drop.extend(["HbS_peak_presente", "HbD_peak_presente"])
    expected = TPOT_FEATURES if with_derived else TPOT_FEATURES[:-2]
    X = work.drop(columns=drop)
    # O CSV mantém a ordem original das colunas; o protocolo usa uma ordem
    # explícita para que treino, validação e teste tenham a mesma entrada.
    X = X[expected]
    y = work[TARGET].copy()

    assert X.columns.tolist() == expected, X.columns.tolist()
    assert len(X) == len(y)
    assert not X.isna().any().any(), "Entrada TPOT não deve conter NaN"
    assert y.nunique() == 8
    return X, y


def max_eval_minutes(budget_minutes: int) -> int:
    """Limite por candidato, declarado antes das execuções.

    Evita que uma pipeline individual ultrapasse muito o orçamento global,
    comportamento observado no teste preliminar do TPOT no Colab.
    """
    return {5: 1, 30: 5, 60: 10}[int(budget_minutes)]


def align_numeric_inputs(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Codifica categorias e alinha as colunas sem usar o alvo.

    O TPOT 1.1.0 apresentou falha interna ao receber categorias textuais.
    A expansão indicadora é determinística; o vocabulário é alinhado entre
    as partições e não usa qualquer estatística da variável-alvo.
    """
    encoded = [pd.get_dummies(frame, dtype=float) for frame in frames]
    columns = sorted(set().union(*(frame.columns for frame in encoded)))
    return tuple(frame.reindex(columns=columns, fill_value=0.0) for frame in encoded)


def build_search_space(X: pd.DataFrame, y: pd.Series, seed: int):
    """Retorna o espaço oficial estável do TPOT 1.1.0.

    O espaço personalizado com ``ColumnOneHotEncoder`` foi testado, mas a
    versão 1.1.0 apresentou erro interno na seleção da população antes de
    avaliar a primeira geração. Mantemos o espaço oficial ``linear-light``
    para que o experimento seja executável e auditável no ambiente do curso.
    """
    return "linear-light"


def build_tpot(X: pd.DataFrame, y: pd.Series, budget_minutes: int, seed: int,
               checkpoint_dir: Path):
    """Configuração oficial: somente treino, CV estratificada de 5 folds."""
    from tpot import TPOTClassifier

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    model = TPOTClassifier(
        search_space=build_search_space(X, y, seed),
        scorers=["f1_macro"],
        scorers_weights=[1],
        cv=cv,
        preprocessing=True,
        validation_strategy="none",
        max_time_mins=budget_minutes,
        max_eval_time_mins=max_eval_minutes(budget_minutes),
        population_size=5,
        initial_population_size=5,
        generations_until_end_population=1,
        n_jobs=1,
        processes=False,
        scatter=False,
        random_state=seed,
        periodic_checkpoint_folder=str(checkpoint_dir),
        verbose=2,
    )
    model._c318_timeout_s = int(budget_minutes * 60 + 60)
    return model


def decode_labels(model, values) -> np.ndarray:
    """Converte previsões internas 0..K-1 de volta para Diagnosis."""
    pred = np.asarray(values)
    encoder = getattr(model, "label_encoder_", None)
    if encoder is not None:
        try:
            return encoder.inverse_transform(pred.astype(int))
        except (TypeError, ValueError):
            # Algumas versões devolvem diretamente os rótulos originais.
            pass
    return pred


def class_labels(model, n_columns: int) -> np.ndarray:
    """Rótulos das colunas de predict_proba no espaço original."""
    encoder = getattr(model, "label_encoder_", None)
    if encoder is not None:
        labels = np.asarray(encoder.classes_)
    else:
        labels = np.asarray(getattr(model, "classes_", []))
    if len(labels) != n_columns:
        labels = np.arange(n_columns).astype(str)
    return labels


def fit_measure_predict(model, X_train, y_train, X_eval, y_eval, labels):
    """Ajusta só em treino e avalia em conjunto externo, com métricas comuns."""
    with measure() as fit_measure:
        with fit_timeout(getattr(model, "_c318_timeout_s", 3600)):
            model.fit(X_train, y_train)
    with measure() as predict_measure:
        pred = decode_labels(model, model.predict(X_eval))

    row = {
        **compute_metrics(y_eval, pred, labels=labels),
        "tempo_ajuste_s": fit_measure["wall_s"],
        "tempo_inferencia_s": predict_measure["wall_s"],
        "pico_memoria_mb": max(fit_measure["peak_mb"], predict_measure["peak_mb"]),
    }

    predictions = pd.DataFrame({"y_true": y_eval.to_numpy(), "y_pred": pred}, index=X_eval.index)
    try:
        proba = np.asarray(model.predict_proba(X_eval))
        labels_proba = class_labels(model, proba.shape[1])
        for i, label in enumerate(labels_proba):
            predictions[f"proba_{label}"] = proba[:, i]
    except Exception as exc:  # alguns pipelines TPOT não expõem probabilidades
        row["aviso_probabilidades"] = f"{type(exc).__name__}: {exc}"
    return row, predictions


def pipeline_repr(model) -> str:
    fitted = getattr(model, "fitted_pipeline_", None)
    return repr(fitted)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def select_budget(validation_runs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Seleciona o orçamento antes do teste: maior média de F1 macro; empate, menor budget."""
    required = {"budget_min", "seed", "f1_macro"}
    assert required <= set(validation_runs.columns), validation_runs.columns.tolist()
    summary = (validation_runs.groupby("budget_min", as_index=False)
               .agg(f1_macro_media=("f1_macro", "mean"),
                    f1_macro_dp=("f1_macro", "std"),
                    f1_weighted_media=("f1_weighted", "mean"),
                    tempo_ajuste_s_medio=("tempo_ajuste_s", "mean"),
                    pico_memoria_mb_max=("pico_memoria_mb", "max"),
                    repeticoes=("seed", "nunique"))
               .sort_values(["f1_macro_media", "budget_min"], ascending=[False, True]))
    repetitions = validation_runs.groupby("budget_min")["seed"].nunique()
    assert repetitions.get(5, 0) == 5, "O budget de 5 minutos exige as cinco sementes"
    assert (repetitions.drop(index=5, errors="ignore") >= 1).all(), (
        "Cada budget adicional exige ao menos uma execução"
    )
    return summary, int(summary.iloc[0]["budget_min"])
