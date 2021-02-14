import torch as T

def get_device():
    device = T.device('cpu')
    if T.cuda.is_available():
        device = T.device('cuda')
    return device