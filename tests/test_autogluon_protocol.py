from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autogluon_protocol import (  # noqa: E402
    AUTOGLOUON_FEATURES,
    BASE_FEATURES,
    OFFICIAL_RUNS,
    index_hash,
    make_inner_holdout,
    prepare_autogluon_data,
    select_budget,
)
from c318_common import TARGET, load_split  # noqa: E402


@pytest.fixture(scope="module")
def official_train():
    train_df, _, _ = load_split(ROOT)
    return train_df


def test_prepare_input_with_and_without_derived(official_train):
    with_derived = prepare_autogluon_data(official_train, with_derived=True)
    without_derived = prepare_autogluon_data(official_train, with_derived=False)

    assert with_derived.columns.tolist() == [*AUTOGLOUON_FEATURES, TARGET]
    assert without_derived.columns.tolist() == [*BASE_FEATURES, TARGET]
    assert with_derived.shape == (7806, 18)
    assert without_derived.shape == (7806, 16)
    assert {"Sl No", "Age", "HbA0", "MCH"}.isdisjoint(with_derived.columns)
    pd.testing.assert_frame_equal(
        with_derived[[*BASE_FEATURES, TARGET]], without_derived
    )


def test_inner_holdout_is_deterministic_disjoint_and_complete(official_train):
    data = prepare_autogluon_data(official_train)
    fit_a, tune_a = make_inner_holdout(data, 42)
    fit_b, tune_b = make_inner_holdout(data, 42)

    assert fit_a.index.equals(fit_b.index)
    assert tune_a.index.equals(tune_b.index)
    assert len(fit_a) == 6244
    assert len(tune_a) == 1562
    assert set(fit_a.index).isdisjoint(tune_a.index)
    assert set(fit_a.index) | set(tune_a.index) == set(data.index)
    assert fit_a[TARGET].nunique() == tune_a[TARGET].nunique() == 8


def test_holdout_seeds_are_distinct(official_train):
    data = prepare_autogluon_data(official_train)
    hashes = {
        index_hash(make_inner_holdout(data, seed)[1].index)
        for seed in [42, 101, 223, 347, 409]
    }
    assert len(hashes) == 5


def validation_table(f1_by_budget=None):
    f1_by_budget = f1_by_budget or {5: 0.6, 30: 0.8, 60: 0.9}
    rows = []
    for budget, seed in OFFICIAL_RUNS:
        rows.append(
            {
                "run_id": f"budget_{budget:02d}m_seed_{seed}",
                "budget_min": budget,
                "seed": seed,
                "f1_macro": f1_by_budget[budget],
                "f1_weighted": f1_by_budget[budget],
                "tempo_ajuste_s": budget * 60,
                "pico_memoria_mb": 1000,
            }
        )
    return pd.DataFrame(rows)


def test_select_budget_requires_exact_matrix_and_selects_best():
    summary, chosen = select_budget(validation_table())
    assert chosen == 60
    assert summary.set_index("budget_min").loc[5, "repeticoes"] == 5
    assert summary.set_index("budget_min").loc[30, "repeticoes"] == 1


def test_select_budget_tie_uses_smallest_budget():
    _, chosen = select_budget(validation_table({5: 0.9, 30: 0.9, 60: 0.9}))
    assert chosen == 5


def test_select_budget_rejects_incomplete_and_duplicate_runs():
    table = validation_table()
    with pytest.raises(ValueError, match="Matriz de runs"):
        select_budget(table.iloc[:-1])

    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicado"):
        select_budget(duplicate)
