def __getattr__(name):
    if name == "Generator":
        from .generator import Generator
        return Generator
    elif name == "Discriminator":
        from .discriminator import Discriminator
        return Discriminator
    elif name == "CGAN":
        from .cgan import CGAN
        return CGAN
    elif name == "DPSGDEngine":
        from .dp_sgd import DPSGDEngine
        return DPSGDEngine
    elif name == "CGANTrainer":
        from .training import CGANTrainer
        return CGANTrainer
    raise AttributeError(f"module 'src.phase2' has no attribute {name}")
