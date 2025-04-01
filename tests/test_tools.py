import tools
import torch
import numpy as np
import pytest


def test_batchify_translator_input():
    """Tests the batchify_translator_input function
    to check if the latent sequences are properly created,
    the padding mask is correctly set up, and all the shapes
    of the tensors returned are correct.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_latent_states = torch.randn(2, 8, 512).to(device)
    is_first_indices = torch.tensor(
        [[1, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 1]]
    )
    seq_len = 6

    latent_sequences, padding_mask = tools.batchify_translator_input(
        input_latent_states,
        is_first_indices,
        seq_len,
        device,
    )

    target_padding_mask = torch.zeros(size=(5, 6), dtype=torch.bool).to(device)
    target_padding_mask[0, 4:] = True
    target_padding_mask[1, 4:] = True
    target_padding_mask[2, 2:] = True
    target_padding_mask[3, 5:] = True
    target_padding_mask[4, 1:] = True

    assert latent_sequences.shape == (5, 6, 512)
    assert torch.equal(padding_mask, target_padding_mask)
    # As they are padded with zeros
    assert latent_sequences[padding_mask].sum() == 0


def test_word_tokenise_text():
    """Tests the function that is used to convert a list of string
    sentences into a list of tokenised sentences. The function should
    add the <BOS> and <EOS> tokens, as well as ensuring each sentence
    is padded to the same length.

    It should also strip punctuation and lowercase the text.

    """

    mock_vocab = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "the": 3,
        "cat": 4,
        "sat": 5,
        "on": 6,
        "mat": 7,
        "lazy": 8,
    }

    input_sentences = ["The Lazy Cat Sat On the mat", "The cat.", "On, the, Mat."]

    expected_output = np.array(
        [
            [1, 3, 8, 4, 5, 6, 3, 7, 2, 0],
            [1, 3, 4, 2, 0, 0, 0, 0, 0, 0],
            [1, 6, 3, 7, 2, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    tokenised_sentences = tools.word_tokenise_text(
        input_sentences,
        mock_vocab,
        required_length=10,
    )
    assert np.array_equal(tokenised_sentences, expected_output)


def test_task_stopwords():
    """Tests that the task stopwords are correctly retrieved"""

    task = "crafter"
    expected_stopwords = [
        "i",
        "will",
        "see",
        "harvest",
        "craft",
        "and",
        "health",
        "hunger",
        "thirst",
    ]
    stopwords = tools.get_task_stopwords(task)
    assert (
        stopwords == expected_stopwords
    ), f"Expected {expected_stopwords}, got {stopwords}"


def test_every():
    """Tests the original implemention of the Every function which
    is used for defining how often to train the model.
    """

    every = tools.Every(5)
    ratio = tools.Ratio(0.2)

    for i in range(20):
        if i % 5 == 0:
            assert every(i) == True
            assert ratio(i) == True
