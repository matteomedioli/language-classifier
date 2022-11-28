import torch

def cuda_setup():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def save_state_dict(save_path, model, optimizer, train, valid, test):
    if save_path == None:
        return
    state_dict = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "data": {"train": train, "valid": valid, "test": test}
                  }
    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def load_state_dict(model, optimizer, load_path="model.pt", device=torch.device("cpu")):
    if load_path==None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f"Model loaded from <== {load_path}")
    model.load_state_dict(state_dict["model_state_dict"])
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    data = state_dict["data"]
    return model, optimizer, data