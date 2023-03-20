import torch

def init_device():
  if torch.cuda.is_available():
    dev_str = "cuda"
  else:
    dev_str = "cpu"
  dev = torch.device(dev_str)
  return dev