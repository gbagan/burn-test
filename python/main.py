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
print(f.shape)
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

#torch.save(discard_model, "test2.pt")

print(output)

#[0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.67308891 0.9181205  0.9181205
# 1.10084546 1.13431489 1.15319681 1.15319681 1.42058337 0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.
# 0.         0.         0.         0.         0.         0.        ]
  