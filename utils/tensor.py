def get_bytes(tensor):
    return tensor.numel() * tensor.element_size()