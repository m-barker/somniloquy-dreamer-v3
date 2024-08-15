import torch


if __name__ == "__main__":

    batch = torch.randn(16, 64, 1536)
    print(batch.shape)

    narration_sequences = []
    for b in batch:
        for i in range(0, 64, 16):
            narration_sequences.append(b[i : i + 16])

    narration_sequences = torch.stack(narration_sequences)
    print(narration_sequences.shape)

    assert torch.equal(narration_sequences, batch.reshape(64, 16, -1))
