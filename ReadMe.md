This repository is a minimal example of how a VAE and a cVAE work.
I have used the dataset: https://susanqq.github.io/UTKFace/

I have also included my trained weights. You can finetune further by loading those checkpoints.

To generate a bunch of faces:
```commandline
python test_vae.py
```

To generate a bunch of faces with parameters:

Edit the `test_cvae.py` file to adjust the parameters. Then,

```commandline
python test_cvae.py
```

