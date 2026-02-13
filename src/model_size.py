import io, torch


def get_model_size(model: torch.nn.Module, location: str = "RAM") -> float:
    """
    Measure model size in MB.

    :param model: the model to measure
    :type model: torch.nn.Module
    :param location: whether to measure the size of the model in RAM or on disk
    :type location: str
    """
    if location == "RAM":
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size  # in bytes
        return total_size / (1024**2)  # Convert to MB
    elif location == "DISK":
        bio = io.BytesIO()
        torch.save(model.state_dict(), bio)
        return bio.tell() / (1024**2)
    else:
        raise ValueError("location must be either 'RAM' or 'DISK'")


def get_model_params(model: torch.nn.Module):
    """
    Count the number of parameters in the model.

    :param model: the model to measure
    :type model: torch.nn.Module
    """
    return sum(p.numel() for p in model.parameters())
