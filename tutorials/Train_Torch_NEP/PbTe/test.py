from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wizard.torchNEP.evaluate import compare_artifacts, evaluate_artifact, format_summary
from wizard.torchNEP.parser import load_train_config


if __name__ == "__main__":
    config = load_train_config(THIS_DIR)
    print(format_summary(compare_artifacts(config, left="exports/nep.txt", right="checkpoints/best.pt")))
    print(format_summary(evaluate_artifact(config, artifact="exports/nep.txt")))
