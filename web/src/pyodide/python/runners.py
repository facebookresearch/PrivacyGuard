"""
runners.py — Python entry-points callable from JS via the Pyodide bridge.

Each function accepts string inputs and returns a JSON string so that
results cross the JS/Python boundary cleanly.
"""

import json
from io import StringIO

import pandas as pd


# ---------------------------------------------------------------------------
# MIA: FDP (analytical — no data upload needed)
# ---------------------------------------------------------------------------


def run_fdp(
    m: int,
    c: int,
    c_cap: int,
    target_noise: float = 0.001,
    threshold: float = 0.05,
    k: int = 2,
    delta: float = 1e-6,
) -> str:
    from privacy_guard.analysis.mia.fdp_analysis_node import FDPAnalysisNode

    node = FDPAnalysisNode(
        m=m,
        c=c,
        c_cap=c_cap,
        target_noise=target_noise,
        threshold=threshold,
        k=k,
        delta=delta,
    )
    output = node.run_analysis()
    return json.dumps(output.to_dict())


# ---------------------------------------------------------------------------
# MIA: LiRA attack
# ---------------------------------------------------------------------------


def run_lira(csv_string: str, params_json: str) -> str:
    from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
    from privacy_guard.attacks.lira_attack import LiraAttack

    params = json.loads(params_json)
    df = pd.read_csv(StringIO(csv_string))
    df_train = df[df["is_member"] == 1].copy()
    df_test = df[df["is_member"] == 0].copy()

    attack = LiraAttack(
        df_train_merge=df_train,
        df_test_merge=df_test,
        row_aggregation=AggregationType(params.get("row_aggregation", "none")),
        std_dev_type=params.get("std_dev_type", "global"),
        online_attack=params.get("online_attack", False),
    )
    analysis_input = attack.run_attack()

    result = {
        "train_scores": analysis_input.df_train_user["score"].describe().to_dict(),
        "test_scores": analysis_input.df_test_user["score"].describe().to_dict(),
        "n_train": len(analysis_input.df_train_user),
        "n_test": len(analysis_input.df_test_user),
        "per_sample": pd.concat(
            [
                analysis_input.df_train_user.assign(split="train"),
                analysis_input.df_test_user.assign(split="test"),
            ]
        ).to_dict(orient="records"),
    }
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# MIA: RMIA attack
# ---------------------------------------------------------------------------


def run_rmia(csv_string: str, population_csv_string: str, params_json: str) -> str:
    from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
    from privacy_guard.attacks.rmia_attack import RmiaAttack

    params = json.loads(params_json)
    df = pd.read_csv(StringIO(csv_string))
    df_population = pd.read_csv(StringIO(population_csv_string))
    df_train = df[df["is_member"] == 1].copy()
    df_test = df[df["is_member"] == 0].copy()

    attack = RmiaAttack(
        df_train_merge=df_train,
        df_test_merge=df_test,
        df_population=df_population,
        row_aggregation=AggregationType(params.get("row_aggregation", "none")),
        num_reference_models=params.get("num_reference_models"),
        alpha_coefficient=params.get("alpha_coefficient", 0.3),
    )
    analysis_input = attack.run_attack()

    result = {
        "train_scores": analysis_input.df_train_user["score"].describe().to_dict(),
        "test_scores": analysis_input.df_test_user["score"].describe().to_dict(),
        "n_train": len(analysis_input.df_train_user),
        "n_test": len(analysis_input.df_test_user),
        "per_sample": pd.concat(
            [
                analysis_input.df_train_user.assign(split="train"),
                analysis_input.df_test_user.assign(split="test"),
            ]
        ).to_dict(orient="records"),
    }
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# MIA: Calibration attack
# ---------------------------------------------------------------------------


def run_calib(csv_string: str, calib_csv_string: str, params_json: str) -> str:
    from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
    from privacy_guard.attacks.calib_attack import CalibAttack, CalibScoreType

    params = json.loads(params_json)
    df = pd.read_csv(StringIO(csv_string))
    df_calib = pd.read_csv(StringIO(calib_csv_string))
    df_train = df[df["is_member"] == 1].copy()
    df_test = df[df["is_member"] == 0].copy()
    df_train_calib = df_calib[df_calib["is_member"] == 1].copy()
    df_test_calib = df_calib[df_calib["is_member"] == 0].copy()

    attack = CalibAttack(
        df_hold_out_train=df_train,
        df_hold_out_test=df_test,
        df_hold_out_train_calib=df_train_calib,
        df_hold_out_test_calib=df_test_calib,
        row_aggregation=AggregationType(params.get("row_aggregation", "none")),
        should_calibrate_scores=params.get("should_calibrate_scores", True),
        score_type=CalibScoreType(params.get("score_type", "loss")),
    )
    analysis_input = attack.run_attack()

    result = {
        "train_scores": analysis_input.df_train_user["score"].describe().to_dict(),
        "test_scores": analysis_input.df_test_user["score"].describe().to_dict(),
        "n_train": len(analysis_input.df_train_user),
        "n_test": len(analysis_input.df_test_user),
        "per_sample": pd.concat(
            [
                analysis_input.df_train_user.assign(split="train"),
                analysis_input.df_test_user.assign(split="test"),
            ]
        ).to_dict(orient="records"),
    }
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# LIA: Label Inference Attack
# ---------------------------------------------------------------------------


def run_lia(csv_string: str, calib_csv_string: str, params_json: str) -> str:
    from privacy_guard.analysis.mia.aggregate_analysis_input import AggregationType
    from privacy_guard.attacks.lia_attack import LIAAttack, LIAAttackInput

    params = json.loads(params_json)
    df = pd.read_csv(StringIO(csv_string))
    df_calib = pd.read_csv(StringIO(calib_csv_string))
    df_train = df[df["is_member"] == 1].copy()
    df_train_calib = df_calib[df_calib["is_member"] == 1].copy()

    attack_input_builder = LIAAttackInput(
        df_hold_out_train=df_train,
        df_hold_out_train_calib=df_train_calib,
        row_aggregation=AggregationType(params.get("row_aggregation", "none")),
    )
    prepared_input = attack_input_builder.prepare_attack_input()

    attack = LIAAttack(
        attack_input=prepared_input,
        row_aggregation=AggregationType(params.get("row_aggregation", "none")),
        y1_generation=params.get("y1_generation", "calibration"),
        num_resampling_times=params.get("num_resampling_times", 100),
    )
    analysis_input = attack.run_attack()

    result = {
        "train_scores": analysis_input.df_train_user["score"].describe().to_dict(),
        "test_scores": analysis_input.df_test_user["score"].describe().to_dict(),
        "n_train": len(analysis_input.df_train_user),
        "n_test": len(analysis_input.df_test_user),
        "per_sample": pd.concat(
            [
                analysis_input.df_train_user.assign(split="train"),
                analysis_input.df_test_user.assign(split="test"),
            ]
        ).to_dict(orient="records"),
    }
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Extraction: Text Inclusion + Edit Similarity
# ---------------------------------------------------------------------------


def run_text_inclusion(jsonl_string: str, params_json: str) -> str:
    from privacy_guard.analysis.extraction.edit_similarity_node import (
        EditSimilarityNode,
    )
    from privacy_guard.analysis.extraction.text_inclusion_analysis_node import (
        TextInclusionAnalysisNode,
    )
    from privacy_guard.attacks.text_inclusion_attack import TextInclusionAttack

    rows = [
        json.loads(line) for line in jsonl_string.strip().split("\n") if line.strip()
    ]
    df = pd.DataFrame(rows)

    attack = TextInclusionAttack(data=df)
    analysis_input = attack.run_attack()

    results = {}

    ti_node = TextInclusionAnalysisNode(analysis_input=analysis_input)
    ti_output = ti_node.run_analysis()
    results["text_inclusion"] = ti_output.to_dict()

    es_node = EditSimilarityNode(analysis_input=analysis_input)
    es_output = es_node.run_analysis()
    results["edit_similarity"] = es_output.to_dict()

    return json.dumps(results, default=str)


# ---------------------------------------------------------------------------
# Extraction: Probabilistic Memorization
# ---------------------------------------------------------------------------


def run_prob_memorization(csv_string: str, params_json: str) -> str:
    from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_input import (
        ProbabilisticMemorizationAnalysisInput,
    )
    from privacy_guard.analysis.extraction.probabilistic_memorization_analysis_node import (
        ProbabilisticMemorizationAnalysisNode,
    )

    params = json.loads(params_json)
    df = pd.read_csv(StringIO(csv_string))

    if df.empty:
        return json.dumps({"error": "No data rows found in CSV"})

    # Handle serialised JSON lists in the prediction_logprobs column
    if isinstance(df["prediction_logprobs"].iloc[0], str):
        df["prediction_logprobs"] = df["prediction_logprobs"].apply(json.loads)

    analysis_input = ProbabilisticMemorizationAnalysisInput(
        generation_df=df,
        prob_threshold=params.get("prob_threshold", 0.5),
        n_values=params.get("n_values"),
    )
    node = ProbabilisticMemorizationAnalysisNode(analysis_input=analysis_input)
    output = node.run_analysis()
    return json.dumps(output.to_dict(), default=str)
