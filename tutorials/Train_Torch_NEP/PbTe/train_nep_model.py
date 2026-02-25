from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wizard.torchNEP.train import train_nep


if __name__ == "__main__":
    train_nep(
        train_xyz=THIS_DIR / "train.xyz",
        save_path=THIS_DIR / "nep_model.pt",
        elements=["Te", "Pb"],  # Explicit order decides type id: Te->0, Pb->1
        optimizer_name="adam",  # or "snes"
        epochs=500,
        generations=1000,
    )
