from wizard.torchNEP.model import NEP

nep = NEP.from_checkpoint("../Repository/PbTe/nep_model.pt")
nep.print_model_info()
nep.save_to_nep_format("../Repository/PbTe/nep_torch.txt")