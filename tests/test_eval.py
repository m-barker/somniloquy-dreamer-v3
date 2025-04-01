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
