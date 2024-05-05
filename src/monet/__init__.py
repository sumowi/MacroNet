from monet.base import MoNetInitial

try:
    import torch.nn as nn
    from monet.torch_ddf import torch_dict
    m = MoNetInitial(nn)
    m.ddf(torch_dict)
except Exception:
    m = MoNetInitial()
    print("monet loaded without torch")
finally:
    m.nn = m.__funcspace__
    nn = m.nn