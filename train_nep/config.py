ELEMENT_CONFIG = {
    "elements": ['Pb', 'Te'], 
}

CUTOFF_CONFIG = {
    "rcut_radial": 6.0,   
    "rcut_angular": 4.0,  
}

DESCRIPTOR_CONFIG = {
    "n_desc_radial": 8,    
    "k_max_radial": 8,     
    "n_desc_angular": 8,   
    "k_max_angular": 8,    
    "l_max": 4,           
}

DATASET_CONFIG = {
    "NN_radial": 50,       
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

def print_config():
    print("=" * 50)
    print("NEP模型参数配置")
    print("=" * 50)
    
    config = get_nep_config()
    
    print(f"\n 元素信息:")
    print(f"  elements: {config['elements']}")
    print(f"  n_types: {config['n_types']}")
    
    print("\n 截断半径:")
    for key in CUTOFF_CONFIG:
        print(f"  {key}: {config[key]}")
    
    print("\n 描述符配置:")
    for key in DESCRIPTOR_CONFIG:
        print(f"  {key}: {config[key]}")
    
    print("\n 近邻列表最大值:")
    for key in DATASET_CONFIG:
        print(f"  {key}: {config[key]}")
    
    print("\n 模型配置:")
    for key in MODEL_CONFIG:
        print(f"  {key}: {config[key]}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_config()
