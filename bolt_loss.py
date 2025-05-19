
import torch
import torch.nn.functional as F

def BOLT_loss(logits: torch.Tensor,
              targets: torch.Tensor,
              norm: str = "l2") -> torch.Tensor:
    """Compute the BOLT loss with class‑0 probability removed.

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs of shape (batch_size, num_classes).
    targets : torch.Tensor
        Integer class labels of shape (batch_size,).
    norm : str
        Either 'l1' or 'l2' to choose the aggregation norm.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the batch‑averaged loss.
    """
    # Convert logits to probabilities and discard class‑0
    probs = F.softmax(logits, dim=1)[:, 1:]
    B, C = probs.size()

    # Build comparison mask
    class_mask = torch.arange(C, device=targets.device).expand(B, C)
    tgt = targets.unsqueeze(1).expand_as(class_mask)

    loss_mat  = (class_mask >= tgt).float() * probs
    loss_mat += (class_mask == (tgt - 1)).float() * (1.0 - probs)

    if norm.lower() == "l2":
        return loss_mat.pow(2).sum() / B
    elif norm.lower() == "l1":
        return loss_mat.abs().sum() / B
    else:
        raise ValueError("norm must be 'l1' or 'l2'")
