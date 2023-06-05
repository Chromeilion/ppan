import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len = 15
