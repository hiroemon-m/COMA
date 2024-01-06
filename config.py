# Third Party Library
import torch

select_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# select_device = "mps"
# select_device = "cpu"
