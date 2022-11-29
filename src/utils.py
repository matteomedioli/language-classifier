import torch
import logging
import os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def cuda_setup():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("There are %d GPU(s) available." % torch.cuda.device_count())
        logger.info("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        logger.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def save_state_dict(save_path, model, optimizer, train, valid, test):
    if save_path == None:
        return
    state_dict = {"model": model.state_dict(),
                  "kwargs": model.kwargs,
                  "optimizer": optimizer.state_dict(),
                  "data": {"train": train, "valid": valid, "test": test}
                  }
    logger.info("\tSaving model checkpoint...")
    torch.save(state_dict, save_path)
    logger.info(f"\tModel saved to: {save_path}")

def rollback_training():
    logger.info(f"\tERROR: clean vocab.pt and model.pt")
    if os.path.exists("vocab.pt"):
        os.system("del vocab.pt")
    if os.path.exists("model.pt"):
        os.system("del model.pt")
