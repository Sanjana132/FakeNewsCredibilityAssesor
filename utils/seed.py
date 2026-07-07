"""
Global reproducibility seeding. Call set_seed(42) at the top of every
entrypoint before any random operations to guarantee reproducible runs.
"""

def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        from langdetect import DetectorFactory
        DetectorFactory.seed = seed
    except ImportError:
        pass
