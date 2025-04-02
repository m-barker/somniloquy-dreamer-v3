from typing import Dict
import numpy as np

import evaluation


def test_bleu_score():
    """Tests the BLEU score calculation, and that it correctly
    ignores punctuation and casing.
    """

    predicted_str = "The Cat sat; on the: mat."
    target_str = "The cat sAt oN the mat."

    bleu_score = evaluation.bleu_metric_from_strings(predicted_str, target_str)
    assert bleu_score == 1.0, f"Expected BLEU score of 1.0, got {bleu_score}"


def test_bleu_score_with_stopwords():
    """Tests that the bleu score works with stopwords."""

    predicted_str = "The large round and lazy cat sat on the mat."
    target_str = "The lazy cat sat on the mat."
    bleu_score = evaluation.bleu_metric_from_strings(
        predicted_str, target_str, words_to_remove=["large", "round", "and"]
    )
    assert bleu_score == 1.0, f"Expected BLEU score of 1.0, got {bleu_score}"


def test_word_error_rate():
    """Tests the word error rate computation."""

    translated_string = "the cat sat on the mat"
    true_string = "the cat sat on the mat"
    word_error_rate = evaluation.calculate_wer(translated_string, true_string)
    assert float(word_error_rate) == 0.0, f"Expected WER of 0.0, got {word_error_rate}"

    translated_string = "the fast dog sat on top of the mat"
    word_error_rate = evaluation.calculate_wer(translated_string, true_string)
    assert float(word_error_rate) == (
        2.0 / 3.0
    ), f"Expected WER of 0.666667, got {word_error_rate}"


def test_rouge_score():
    """Tests the ROUGE score calculation."""

    predicted_str = "the cat sat on the mat"
    target_str = "the cat sat on the mat"

    rouge_score = evaluation.compute_rouge_score(predicted_str, target_str)
    assert (
        rouge_score["rouge1_precision"] == 1.0
    ), f"Expected ROUGE-1 precision of 1.0, got {rouge_score['rouge1_precision']}"
    assert (
        rouge_score["rouge1_recall"] == 1.0
    ), f"Expected ROUGE-1 recall of 1.0, got {rouge_score['rouge1_recall']}"
    assert (
        rouge_score["rouge1_fmeasure"] == 1.0
    ), f"Expected ROUGE-1 F1 of 1.0, got {rouge_score['rouge1_f1']}"
    assert (
        rouge_score["rouge2_precision"] == 1.0
    ), f"Expected ROUGE-2 precision of 1.0, got {rouge_score['rouge2_precision']}"
    assert (
        rouge_score["rouge2_recall"] == 1.0
    ), f"Expected ROUGE-2 recall of 1.0, got {rouge_score['rouge2_recall']}"
    assert (
        rouge_score["rouge2_fmeasure"] == 1.0
    ), f"Expected ROUGE-2 F1 of 1.0, got {rouge_score['rouge2_f1']}"
    assert (
        rouge_score["rougeL_precision"] == 1.0
    ), f"Expected ROUGE-L precision of 1.0, got {rouge_score['rougeL_precision']}"
    assert (
        rouge_score["rougeL_recall"] == 1.0
    ), f"Expected ROUGE-L recall of 1.0, got {rouge_score['rougeL_recall']}"
    assert (
        rouge_score["rougeL_fmeasure"] == 1.0
    ), f"Expected ROUGE-L F1 of 1.0, got {rouge_score['rougeL_f1']}"

    predicted_str = "the large cat sat on the very thin mat"
    rouge_score = evaluation.compute_rouge_score(predicted_str, target_str)

    assert (
        round(rouge_score["rouge1_precision"], 2) == 0.67
    ), f"Expected ROUGE-1 precision of 0.67, got {rouge_score['rouge1_precision']}"
    assert (
        rouge_score["rouge1_recall"] == 1.0
    ), f"Expected ROUGE-1 recall of 1.0, got {rouge_score['rouge1_recall']}"
    assert (
        round(rouge_score["rouge2_precision"], 2) == 0.38
    ), f"Expected ROUGE-1 precision of 0.67, got {rouge_score['rouge2_precision']}"
    assert (
        round(rouge_score["rouge2_recall"], 2) == 0.6
    ), f"Expected ROUGE-1 recall of 0.4, got {rouge_score['rouge2_recall']}"
    assert round(rouge_score["rougeL_precision"], 2) == (
        0.67
    ), f"Expected ROUGE-L precision of 0.67, got {rouge_score['rougeL_precision']}"
    assert (
        rouge_score["rougeL_recall"] == 1.0
    ), f"Expected ROUGE-L recall of 1.0, got {rouge_score['rougeL_recall']}"


def test_ter():
    """Tests the calculation of the Translation Edit Rate (TER)."""

    predicted_str = "the cat sat on the mat"
    target_str = "the cat sat on the mat"

    ter_score = float(evaluation.compute_ter(predicted_str, target_str))
    assert ter_score == 0.0, f"Expected TER of 0.0, got {ter_score}"

    # Just need to move "the cat"
    predicted_str = "sat on the mat the cat"
    ter_score = float(evaluation.compute_ter(predicted_str, target_str))
    assert round(ter_score, 2) == round(
        (1 / 6), 2
    ), f"Expected TER of 1/6, got {ter_score}"

    # A shift is constrained such that it must exacty match the target
    # string, so the TER is 3/6 not 2/6, as shifting "the dog" would not
    # exactly match "the cat"
    predicted_str = "sat on the mat the dog"
    ter_score = float(evaluation.compute_ter(predicted_str, target_str))
    assert round(ter_score, 2) == round(
        (3 / 6), 2
    ), f"Expected TER of 2/6, got {ter_score}"


def test_chrf():
    """Tests the calculation of the chrf++ metrics"""

    predicted_str = "the cat sat on the mat"
    target_str = "the cat sat on the mat"

    chrf_score = evaluation.compute_chrf(predicted_str, target_str)
    assert chrf_score == 1.0, f"Expected chrF++ score of 1.0, got {chrf_score}"


def test_meteor_score():
    """Tests the calculation of the METEOR score, which uses
    the nltk library.
    """

    predicted_str = "the cat sat on the mat"
    target_str = "the cat sat on the mat"

    meteor_score = evaluation.compute_meteor_score(predicted_str, target_str)
    assert meteor_score == 1.0, f"Expected METEOR score of 1.0, got {meteor_score}"


def create_mock_metrics() -> Dict[str, float]:
    """Creates a mock dictionary of translation evlauation metrics
    used for testing"""
    return {
        "bleu_score": 0.0,
        "bleu_score_no_stopwords": 0.0,
        "wer": 0.0,
        "wer_no_stopwords": 0.0,
        "ter": 0.0,
        "ter_no_stopwords": 0.0,
        "chrf": 0.0,
        "chrf_no_stopwords": 0.0,
        "meteor": 0.0,
        "meteor_no_stopwords": 0.0,
        "rouge1_fmeasure": 0.0,
        "rouge1_precision": 0.0,
        "rouge1_recall": 0.0,
        "rouge2_fmeasure": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "rougeL_fmeasure": 0.0,
        "rougeL_precision": 0.0,
        "rougeL_recall": 0.0,
        "rougeLsum_fmeasure": 0.0,
        "rougeLsum_precision": 0.0,
        "rougeLsum_recall": 0.0,
        "rouge1_fmeasure_no_stopwords": 0.0,
        "rouge1_precision_no_stopwords": 0.0,
        "rouge1_recall_no_stopwords": 0.0,
        "rouge2_fmeasure_no_stopwords": 0.0,
        "rouge2_precision_no_stopwords": 0.0,
        "rouge2_recall_no_stopwords": 0.0,
        "rougeL_fmeasure_no_stopwords": 0.0,
        "rougeL_precision_no_stopwords": 0.0,
        "rougeL_recall_no_stopwords": 0.0,
        "rougeLsum_fmeasure_no_stopwords": 0.0,
        "rougeLsum_precision_no_stopwords": 0.0,
        "rougeLsum_recall_no_stopwords": 0.0,
    }


def test_update_running_metrics():
    """Tests the function that is used to update the dictionary storing
    the translation evaluation metrics for all evaluation episodes
    and plan translations
    """

    mock_imagined_metrics = create_mock_metrics()
    mock_reconstruction_metrics = create_mock_metrics()

    running_eval_metrics = {}
    evaluation.update_running_metrics(
        mock_imagined_metrics,
        mock_reconstruction_metrics,
        running_eval_metrics,
        episode_number=0,
        total_episodes=10,
        max_plans_per_episode=20,
        current_plan_index=0,
    )

    for k, v in running_eval_metrics.items():
        assert v.shape == (10, 20), f"Expected shape of (10, 20), got {v.shape}"
        assert v[0, 0] == 0.0, f"Expected value of 0.0, got {v[0, 0]}"
        # Check all other values are set to -1.0.
        assert np.all(
            np.delete(v, (0, 0)) == -1.0
        ), f"Expected all values to be -1.0, got {np.delete(v, (0, 0))}"

    for k, v in mock_imagined_metrics.items():
        mock_imagined_metrics[k] = 1.0
    for k, v in mock_reconstruction_metrics.items():
        mock_reconstruction_metrics[k] = 1.0
    evaluation.update_running_metrics(
        mock_imagined_metrics,
        mock_reconstruction_metrics,
        running_eval_metrics,
        episode_number=0,
        total_episodes=10,
        max_plans_per_episode=20,
        current_plan_index=1,
    )

    for k, v in running_eval_metrics.items():
        assert v[0, 1] == 1.0, f"Expected value of 1.0, got {v[0, 1]}"

    for k, v in mock_imagined_metrics.items():
        mock_imagined_metrics[k] = 0.5
    for k, v in mock_reconstruction_metrics.items():
        mock_reconstruction_metrics[k] = 0.5

    evaluation.update_running_metrics(
        mock_imagined_metrics,
        mock_reconstruction_metrics,
        running_eval_metrics,
        episode_number=5,
        total_episodes=10,
        max_plans_per_episode=20,
        current_plan_index=5,
    )

    for k, v in running_eval_metrics.items():
        assert v[5, 5] == 0.5, f"Expected value of 1.0, got {v[0, 1]}"


def test_metric_statistics():
    """Tests the function that is use compute the statistics of the evalution
    metrics for all evaluation episodes and plan translations
    """

    mock_imagined_metrics = create_mock_metrics()
    mock_reconstruction_metrics = create_mock_metrics()

    running_eval_metrics = {}
    evaluation.update_running_metrics(
        mock_imagined_metrics,
        mock_reconstruction_metrics,
        running_eval_metrics,
        episode_number=0,
        total_episodes=10,
        max_plans_per_episode=20,
        current_plan_index=0,
    )

    for k, v in mock_imagined_metrics.items():
        mock_imagined_metrics[k] = 1.0
    for k, v in mock_reconstruction_metrics.items():
        mock_reconstruction_metrics[k] = 2.0
    evaluation.update_running_metrics(
        mock_imagined_metrics,
        mock_reconstruction_metrics,
        running_eval_metrics,
        episode_number=0,
        total_episodes=10,
        max_plans_per_episode=20,
        current_plan_index=1,
    )

    imagined_metrics = {
        k: v for k, v in running_eval_metrics.items() if "imagined" in k
    }
    print(imagined_metrics["imagined_bleu_score"])
    reconstruction_metrics = {
        k: v for k, v in running_eval_metrics.items() if "reconstructed_" in k
    }
    imagined_metrics = evaluation.compute_evaluation_statistics(imagined_metrics)
    reconstruction_metrics = evaluation.compute_evaluation_statistics(
        reconstruction_metrics
    )
    print(imagined_metrics)
    assert (
        imagined_metrics["mean_imagined_bleu_score"] == 0.5
    ), f"Expected mean bleu score of 0.5, got {imagined_metrics['mean_imagined_bleu_score']}"
    assert (
        imagined_metrics["var_imagined_bleu_score"] == 0.25
    ), f"Expected var bleu score of 0.25, got {imagined_metrics['var_imagined_bleu_score']}"
    assert (
        imagined_metrics["max_imagined_bleu_score"] == 1.0
    ), f"Expected max bleu score of 1.0, got {imagined_metrics['max_imagined_bleu_score']}"
    assert (
        imagined_metrics["min_imagined_bleu_score"] == 0.0
    ), f"Expected min bleu score of 0.0, got {imagined_metrics['min_imagined_bleu_score']}"

    for k, v in mock_imagined_metrics.items():
        mock_imagined_metrics[k] = 10.0
    for k, v in mock_reconstruction_metrics.items():
        mock_reconstruction_metrics[k] = 20.0
    evaluation.update_running_metrics(
        mock_imagined_metrics,
        mock_reconstruction_metrics,
        running_eval_metrics,
        episode_number=1,
        total_episodes=10,
        max_plans_per_episode=20,
        current_plan_index=1,
    )

    imagined_metrics = {
        k: v for k, v in running_eval_metrics.items() if "imagined" in k
    }
    print(imagined_metrics["imagined_bleu_score"])
    reconstruction_metrics = {
        k: v for k, v in running_eval_metrics.items() if "reconstructed_" in k
    }
    imagined_metrics = evaluation.compute_evaluation_statistics(imagined_metrics)
    reconstruction_metrics = evaluation.compute_evaluation_statistics(
        reconstruction_metrics
    )
    print(reconstruction_metrics)
    assert imagined_metrics["mean_imagined_bleu_score"] == (
        11 / 3
    ), f"Expected mean bleu score of 11/3, got {imagined_metrics['mean_imagined_bleu_score']}"
    assert reconstruction_metrics["mean_reconstructed_ter"] == (
        22 / 3
    ), f"Expected mean ter score of 22/3, got {reconstruction_metrics['mean_reconstructed_ter']}"
