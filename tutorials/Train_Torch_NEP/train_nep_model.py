from wizard.torchNEP.dataset import StructureDataset, collate_fn
from wizard.torchNEP.model import NEP
from wizard.torchNEP.optimizer import Optimizer
from torch.utils.data import DataLoader
from wizard.io import read_xyz


ELEMENT_CONFIG = {
    "elements": ['Te', 'Pb'], 
}

CUTOFF_CONFIG = {
    "rcut_radial": 8.0,   
    "rcut_angular": 4.0,  
}

DESCRIPTOR_CONFIG = {
    "n_desc_radial": 5,    
    "n_desc_angular": 5,  
    "k_max_radial": 9,      
    "k_max_angular": 9,    
    "l_max": 4,           
}

DATASET_CONFIG = {
    "NN_radial": 100,       
    "NN_angular": 30,      
}

MODEL_CONFIG = {
    "hidden_dims": [30]
}

TRAIN_CONFIG = {
    "save_path": "nep_model.pt",
    "early_stopping_patience": 10,
    "device": "cuda",  
    "optimizer": {"type": "Adam", "lr": 1e-3},
    "loss_fn": "MAELoss",  
    "batch_size": 16,  
}

def get_nep_config():
    config = {}
    config.update(ELEMENT_CONFIG)
    config.update(CUTOFF_CONFIG)
    config.update(DESCRIPTOR_CONFIG)
    config.update(DATASET_CONFIG)
    config.update(MODEL_CONFIG)
    config.update(TRAIN_CONFIG)
    
    config["n_types"] = len(config["elements"])
    
    return config

def main():
    para = get_nep_config()
    frames = read_xyz("PbTe/train.xyz")

    train_dataset = StructureDataset(frames=frames, para=para)
    train_loader = DataLoader(
        train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn
    )

    model = NEP(para)
    training = Optimizer(
        model=model,
        training_set=train_loader,
        save_path="PbTe/nep_model.pt",
        use_wandb=True
    )
    training.fit(epochs=500)

if __name__ == "__main__":
    main()