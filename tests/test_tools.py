import tools
import torch
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
