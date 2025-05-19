# BOLT-loss

> **Implementation & reproducibility companion for the ICASSP 2025 paper  
> “Universal Training of Neural Networks to Achieve Bayes Optimal Classification Accuracy.”**

This repository provides a ** PyTorch reference implementation** of the **B**ayes‑optimal **O**ptimal **L**earning **T**hreshold (BOLT) loss together with
Jupyter notebooks and scripts that reproduce the experiments reported in the paper.

---

## ✨ What is BOLT?

BOLT minimises a novel, sample‑level upper bound on the Bayes error rate.  
In practice this gives **equal or better accuracy** than cross‑entropy while being **hyper‑parameter free**.

```
Cross‑entropy  → minimises −log p(label | x)
BOLT           → minimises an upper bound on Bayes error
```

---

## 🗂 Repository layout

```
BOLT-loss/
├── bolt_loss.py              ← stand‑alone PyTorch implementation
├── notebooks/                ← demo & reproduction notebooks
│   ├── toy_example.ipynb
│   ├── MNIST.ipynb
│   ├── Fashion_MNIST_BOLT_vs_CE.ipynb
│   └── Cifar10_BOLT_loss.ipynb
└── README.md                 ← you are here
```

---

## 🔧 Installation

```bash
git clone https://github.com/MohammadrezaTavasoli/BOLT-loss.git
cd BOLT-loss
conda create -n bolt python=3.10 -y     # or use venv
conda activate bolt
pip install -r requirements.txt         # torch ≥ 2.2, torchvision, numpy, …
```

> **macOS / Apple Silicon** — code auto‑detects `torch.device("mps")`.  
> GPU acceleration on NVIDIA works the usual way with CUDA.

---

## 🚀 Quick start (script)

```bash
python examples/train_cifar10.py --epochs 100 --loss bolt
```

*Expected accuracy:* ≈ **93.3 %** on CIFAR‑10 with ResNet‑18.

---

## 💡 BOLT loss in 15 lines

```python
import torch
import torch.nn.functional as F  # for softmax

def BOLT_loss(logits: torch.Tensor,
              targets: torch.Tensor,
              norm: str = "l2") -> torch.Tensor:
    """Batch‑averaged BOLT loss.

    logits : (B, K)  raw network outputs for K≥2 classes
    targets: (B,)    ground‑truth labels in 0…K−1
    norm   : "l1" or "l2" — absolute or squared error variant
    """
    probs = F.softmax(logits, dim=1)[:, 1:]   # drop class‑0, shape (B, K−1)
    B, C  = probs.size()

    class_mask = torch.arange(C, device=targets.device).expand(B, C)
    tgt = targets.unsqueeze(1).expand_as(class_mask)

    loss_mat  = (class_mask >= tgt).float() * probs
    loss_mat += (class_mask == (tgt - 1)).float() * (1.0 - probs)

    if norm.lower() == "l2":
        return loss_mat.pow(2).sum() / B
    if norm.lower() == "l1":
        return loss_mat.abs().sum() / B
    raise ValueError("norm must be 'l1' or 'l2'")
```

---

## 📊 Reproducing the paper

Open any notebook in `notebooks/` and run it end‑to‑end; results, plots and
LaTeX‑ready tables will be generated automatically.

| Dataset   | Model      | BOLT | Cross‑entropy |
|-----------|------------|------|---------------|
| CIFAR‑10  | ResNet‑18  | **93.29 %** | 91.95 % |
| IMDb      | BERT‑base  | **94.56 %** | 93.51 % |
| MNIST     | 4‑layer CNN | 99.29 % | 99.29 % |

---

## 📝 Citation

```bibtex
@inproceedings{tavasoli2025bolt,
  title     = {Universal Training of Neural Networks to Achieve Bayes Optimal Classification Accuracy},
  author    = {Mohammadreza Tavasoli Naeini and Ali Bereyhi and Morteza Noshad and Ben Liang and Alfred O. Hero III},
  booktitle = {Proc. ICASSP},
  year      = {2025}
}
```

---

## 📄 License

Released under the MIT license – see `LICENSE` for details.

---

Happy training! 🚀
