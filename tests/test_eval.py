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
