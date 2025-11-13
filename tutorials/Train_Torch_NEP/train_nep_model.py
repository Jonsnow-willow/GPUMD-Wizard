from wizard.torchNEP.dataset import StructureDataset, collate_fn
from wizard.torchNEP.model import NEP
from wizard.torchNEP.optimizer import Optimizer
from torch.utils.data import DataLoader
from wizard.io import read_xyz


frames = read_xyz("../Repository/PbTe/train.xyz")

train_dataset = StructureDataset(frames=frames, para=para)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)

model = NEP(para)
training = Optimizer(
    model=model,
    training_set=train_loader,
    save_path="../Repository/PbTe/nep_model.pt",
    use_wandb= True
)
training.fit(epochs=500)