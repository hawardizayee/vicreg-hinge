# vicreg-hinge

VICReg on CIFAR-10 / CIFAR-100 with a ResNet-18, where the variance term can be a
hinge on the **standard deviation** (the VICReg default) or on the **variance**.
Everything else is held fixed, so the two hinges can be compared at the same weight.

All code is in [examples/vicreg_hinge](examples/vicreg_hinge).

## Installation

We use [uv](https://docs.astral.sh/uv/guides/projects/) for package management
(Python 3.12, PyTorch 2.8.0 with CUDA 12.8).

```bash
uv sync
source .venv/bin/activate
```

Set where the datasets and checkpoints go. CIFAR is downloaded on the first run.

```bash
# Required: dataset directory
export EBJEPA_DSETS=/path/to/datasets
# Optional: checkpoint directory (default: ./checkpoints)
export EBJEPA_CKPTS=/path/to/checkpoints
```

Runs log to [Weights & Biases](https://wandb.ai). Run `wandb login` once, or turn
logging off with `export WANDB_DISABLED=true`.

## Training

Run from inside the example folder:

```bash
cd examples/vicreg_hinge

# Standard-deviation hinge, weight 1.0
python main.py --loss.hinge=std --loss.hinge_coeff=1.0 --meta.seed=42

# Variance hinge, weight 1.0
python main.py --loss.hinge=var --loss.hinge_coeff=1.0 --meta.seed=42
```

Any value in [cfgs/default.yaml](examples/vicreg_hinge/cfgs/default.yaml) can be
overridden the same way. The ones that matter here:

| Option | Values | Meaning |
|--------|--------|---------|
| `loss.hinge` | `std`, `var` | Hinge on the standard deviation or on the variance |
| `loss.hinge_coeff` | float | Weight of the variance term |
| `loss.cov_coeff` | float | Weight of the covariance term (default `100.0`) |
| `data.dataset` | `cifar10`, `cifar100` | Dataset |
| `meta.seed` | int | Random seed |

Each run reports the accuracy of an online linear probe on the validation set
(`val_acc`) and writes its checkpoints to
`{EBJEPA_CKPTS}/image_jepa/dev_{date}/{exp_name}_seed{seed}/`. The run name encodes
the setting, e.g. `resnet_vicreg_proj_bs256_ep300_ph1024_po1024_var1.0_cov100.0`.

With `logging.save_embeddings=true` (the default) the run also saves the encoder and
projector embeddings of a fixed batch at every epoch to `embedding_history.pt`
(about 0.9 GB per run).

## Acknowledgements

This code is derived from the `image_jepa` example of
[EB-JEPA](https://github.com/facebookresearch/eb_jepa) (Apache 2.0). We added the
`loss.hinge` option and removed the parts of the library that this experiment does
not use.

```bibtex
@misc{terver2026lightweightlibraryenergybasedjointembedding,
      title={A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures},
      author={Basile Terver and Randall Balestriero and Megi Dervishi and David Fan and Quentin Garrido and Tushar Nagarajan and Koustuv Sinha and Wancong Zhang and Mike Rabbat and Yann LeCun and Amir Bar},
      year={2026},
      eprint={2602.03604},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03604},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE.md).
