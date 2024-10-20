import torch
import numpy as np
from model import DiscardModel
from safetensors.torch import load_file

discard_weights = load_file("./tensors/discard_sl.safetensors")
discard_model = DiscardModel()
discard_model.load_state_dict(discard_weights)
discard_model.eval()

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