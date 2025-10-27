import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For the ith element in tensor, set indices beyond lengths[i] to 0
def mask_tensor(tensor, lengths):
    assert tensor.shape[0] == lengths.shape[0]
    masked_tensor = torch.zeros(tensor.shape).to(device)
    for i in range(tensor.shape[0]):
        length = lengths[i]
        masked_tensor[i, :length] = tensor[i, :length]
    return masked_tensor

def dataloader_generator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def put_batch_on_device(batch):
    if isinstance(batch, dict):
        for key in batch.keys():
            batch[key] = put_batch_on_device(batch[key])
        return batch
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch