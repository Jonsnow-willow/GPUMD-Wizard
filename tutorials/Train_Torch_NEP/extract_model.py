from wizard.torchNEP.model import NEP

nep = NEP.from_checkpoint("PbTe/nep_model.pt")
nep.print_model_info()
nep.save_to_nep_format("PbTe/nep.txt")