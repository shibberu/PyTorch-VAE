import pytorch_lightning as pl

## Utils to handle newer PyTorch Lightning changes
## (In PL 2.0 the "pl.data_loader" decorator is removed.)

def data_loader(fn):
    # Simply return the function as is.
    return fn

