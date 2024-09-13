import torch

class minimal_implementation_dataset(torch.utils.data.Dataset):
  def __init__(self, x, y, device):
    self.x = torch.tensor(x).type(torch.float).to(device)
    self.y = torch.tensor(y).type(torch.float).to(device)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
  


class minimal_implementation_dataset_classification(torch.utils.data.Dataset):
  def __init__(self, x, y, device, dtype=torch.float):
    self.x = torch.tensor(x).type(dtype).to(device)
    self.y = torch.tensor(y).type(torch.long).to(device)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]