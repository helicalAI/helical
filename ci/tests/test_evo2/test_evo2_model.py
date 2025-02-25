import pytest

import csv
from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F

try:
    from helical.models.evo_2 import Evo2, Evo2Config

    evo2_unavailable = False  # only run tests if able to import the package
except:
    evo2_unavailable = True

torch.manual_seed(1)
torch.cuda.manual_seed(1)


@pytest.mark.skipif(evo2_unavailable, reason="No Evo 2 module present")
@pytest.fixture
def read_prompts() -> Union[List[List[str]]]:
    """Read prompts from input file."""
    promptseqs: List[str] = []

    with open("ci/tests/data/prompts.csv", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            promptseqs.append(row[0])

    return promptseqs


@pytest.mark.skipif(evo2_unavailable, reason="No Evo 2 module present")
@pytest.fixture
def evo2_model():
    """Initialize Evo2 model."""
    model_config = Evo2Config(model_name="evo2-7b", batch_size=1)
    model = Evo2(model_config)
    return model


@pytest.mark.skipif(evo2_unavailable, reason="No Evo 2 module present")
@pytest.fixture
def test_forward_pass(evo2_model, read_prompts):
    """Test model forward pass accuracy on sequences."""
    sequences = read_prompts
    losses = []
    accuracies = []

    for seq in sequences:
        # Convert sequence to model input format
        input_ids = torch.tensor(evo2_model.tokenizer.tokenize(seq), dtype=int).to(
            "cuda:0"
        )

        with torch.inference_mode():
            # Forward pass
            logits, _ = evo2_model.model.forward(input_ids.unsqueeze(0))

            # Calculate loss and accuracy
            target_ids = input_ids[1:]  # Shift right for next token prediction
            pred_logits = logits[0, :-1, :]

            # Cross entropy loss
            loss = F.cross_entropy(pred_logits, target_ids.long())

            # Get predictions
            pred_tokens = torch.argmax(pred_logits, dim=-1)

            # Calculate accuracy
            accuracy = (target_ids == pred_tokens).float().mean().item()

            losses.append(loss.item())
            accuracies.append(accuracy)

    # Print sequence results
    print("\nSequence Results:")
    for i, (loss, acc) in enumerate(zip(losses, accuracies)):
        print(f"Sequence {i+1}: Loss = {loss:.3f}, Accuracy = {acc:.2%}")
        if acc < 0.5:
            print(
                "WARNING: Forward pass accuracy is below 50% on test sequence. Model may be broken, trained models should have >80% accuracy."
            )

    return accuracies, losses


@pytest.mark.skipif(evo2_unavailable, reason="No Evo 2 module present")
def test_evo2(test_forward_pass, evo2_model):
    """
    Test sequence prediction accuracy using Evo2 models.
    Expected results for forward pass:
    - Evo 2 40B 1m: Loss ~0.216, Accuracy ~91.67%
    - Evo 2 7B 1m: Loss ~0.348, Accuracy ~86.35%
    - Evo 2 1B base: Loss ~0.502, Accuracy ~79.56%
    """
    accuracies, losses = test_forward_pass
    # Set random seeds

    # Calculate and validate results
    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies) * 100
    print(f"\nMean Loss: {mean_loss:.3f}")
    print(f"Mean Accuracy: {mean_accuracy:.3f}%")

    # Validate against expected scores
    eps = 1e-3  # epsilon for float comparison
    expected_metrics = {
        # "evo2_40b": {"loss": 0.2159424, "acc": 91.673},
        "evo2_7b": {"loss": 0.3476563, "acc": 86.346},
        # "evo2_40b_base": {"loss": 0.2149658, "acc": 91.741},
        # "evo2_7b_base": {"loss": 0.3520508, "acc": 85.921},
        # "evo2_1b_base": {"loss": 0.501953125, "acc": 79.556},
    }

    expected = expected_metrics[evo2_model.config["model_map"]["model_name"]]
    if abs(mean_loss - expected["loss"]) < eps:
        print(f"\nTest Passed! Loss matches expected {expected['loss']:.3f}")
    else:
        print(
            f"\nTest Failed: Expected loss {expected['loss']:.3f}, got {mean_loss:.3f}"
        )
