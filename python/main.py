import torch
import numpy as np
from model import DiscardModel

from safetensors.torch import load_file

discard_weights = load_file("./tensors/discard_sl.safetensors")
discard_model = DiscardModel()
discard_model.load_state_dict(discard_weights)
f = np.load("./tensors/test.npy")
f = torch.Tensor(f)
f = f.unsqueeze(0)
mask = [0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0       
        ]
output = discard_model(f)
output = output.squeeze(0).detach().numpy()
output = np.exp(output/10.0) * mask
print(output)

#[0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.67308891 0.9181205  0.9181205
# 1.10084546 1.13431489 1.15319681 1.15319681 1.42058337 0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.        ]

#[0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.63550222 0.9251641  0.86671942
# 1.05282056 1.12088335 1.11960053 1.14039636 1.3941747  0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.      