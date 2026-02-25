from wizard.torchNEP.model import NEP

nep = NEP.from_checkpoint("nep_model.pt")
nep.print_model_info()
nep.save_to_nep_format("nep.txt")