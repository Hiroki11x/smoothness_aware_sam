import numpy as np
import torch

def np2torch(
    np_X:np.ndarray,
    np_y:np.ndarray
    ) -> [torch.Tensor, torch.Tensor]:

    torch_X = torch.stack([
        torch.from_numpy(
            np.array(i).transpose(2,0,1).astype(np.float32)
        ) for i in np_X]
    )
    
    torch_y = torch.stack([
        torch.from_numpy(
            np.array(i).astype(np.int64)
        ) for i in np_y]
    )
    
    return [torch_X, torch_y]

def create_dataloader(
    torch_X:torch.Tensor,
    torch_y:torch.Tensor
    ) -> torch.utils.data.DataLoader:
    
    dataset = torch.utils.data.TensorDataset(torch_X, torch_y)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=500,
        shuffle=False,
        pin_memory=True
    )
    
    return dataloader
