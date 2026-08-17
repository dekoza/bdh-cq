
## BDH-CQ (wip)

Implementation of <a href="https://arxiv.org/abs/2608.09888">BDH-CQ: In-Context Learning with Recurrent Latent Reasoning</a>, proposed by Pathway Research

## Install

```bash
$ pip install bdh-cq
```

## Usage

```python
import torch
from bdh_cq import BDH

model = BDH(
    dim = 512,
    num_tokens = 20_000
)

ids = torch.randint(0, 20_000, (2, 1024))

logits = model(ids) # (2, 1024, 20_000)
```

## Citations

```bibtex
@misc{engdahl2026bdhcq,
    title   = {BDH-CQ: In-Context Learning with Recurrent Latent Reasoning},
    author  = {Björn Engdahl and Adrian Kosowski and Jan Chorowski and Zuzanna Stamirowska and Przemysław Uznański and Junlin Jiang and Rohan Phadke and Remigiusz Kinas and Richard Zhong},
    year    = {2026},
    eprint  = {2608.09888},
    archivePrefix = {arXiv},
    primaryClass = {cs.NE},
    url     = {https://arxiv.org/abs/2608.09888}
}
```
