import importlib


def get_model(cfg, datamodule, phase="train"):
    modeltype = cfg.model.model_type
    if modeltype == "convofusion":
        return get_module(cfg, datamodule)
    else:
        raise ValueError(f"Invalid model type {modeltype}.")


def get_module(cfg, datamodule):
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(
        f".modeltype.{cfg.model.model_type}", package="convofusion.models")
    Model = model_module.__getattribute__(f"{modeltype.title()}") # changed from MLD (uppercase) to Convofusion (camelcase)
    return Model(cfg=cfg, datamodule=datamodule)
