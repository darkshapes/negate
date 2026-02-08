from pathlib import Path
import numpy as np
from negate.datasets import generate_dataset
from negate.wavelet import WaveletAnalyzer


def _compute_similarities(path: str, label: int) -> list[float]:
    """Compute similarity scores for images in a directory.\n
    :param path: Directory containing images.
    :param label: Label to assign (unused but kept for API compatibility).
    :returns: List of WaRPAD(x) scores per image (average of HFwav over patches).
    """
    dataset = generate_dataset(Path(path), 0)
    analyzer = WaveletAnalyzer()
    result = analyzer.decompose(dataset)
    return list(result["sensitivity"])


if __name__ == "__main__":
    genuine_sims = _compute_similarities("/Users/e6d64/Downloads/real_train", 0)
    synthetic_sims = _compute_similarities(
        "/Users/e6d64/Downloads/phantomDiffusionS3FinalImages",
        1,
    )

    avg_genuine_sim = float(np.mean(genuine_sims))
    avg_synthetic_sim = float(np.mean(synthetic_sims))

    print(f"Average similarity (genuine): {avg_genuine_sim:.4f}")
    print(f"Average similarity (synthetic): {avg_synthetic_sim:.4f}")

    all_sims = genuine_sims + synthetic_sims
    overall_avg = float(np.mean(all_sims))
    print(f"Overall average cosine similarity: {overall_avg:.4f}")
