import torch

from deepspec.utils.metrics import add_metric


def compute_mtp_loss(*, model, batch):
    """Teacher-forced MTP training loss.

    ``batch["target_last_hidden_states"][:, i]`` is the base model's final
    hidden state after seeing tokens[0..i] (from an offline target cache built
    by ``scripts/data/prepare_target_cache.py``). MTP pairs that with the
    embedding of token i+1 to predict token i+2, so everything here is shifted
    relative to the cached, unshifted arrays.
    """
    input_ids = batch["input_ids"].long()
    attention_mask = batch["attention_mask"].long()
    loss_mask = batch["loss_mask"]
    target_last_hidden_states = batch["target_last_hidden_states"]

    seq_len = int(input_ids.shape[1])
    assert seq_len >= 3, (
        f"MTP predicts token i+2 from hidden state i and embedding i+1; "
        f"need seq_len >= 3, got {seq_len}."
    )

    hidden_states = target_last_hidden_states[:, :-1, :]
    next_token_ids = input_ids[:, 1:]
    position_ids = torch.arange(
        1, seq_len, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0).expand(input_ids.shape[0], -1)
    head_attention_mask = attention_mask[:, 1:]

    logits = model(
        target_last_hidden_states=hidden_states,
        input_ids=next_token_ids,
        position_ids=position_ids,
        attention_mask=head_attention_mask,
    )

    # logits[:, j] sits at sequence position j+1 and predicts token j+2.
    pred_logits = logits[:, :-1, :]
    targets = input_ids[:, 2:]
    mask = loss_mask[:, 2:].to(torch.float32)

    loss_per_token = torch.nn.functional.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    valid_tokens = mask.sum().clamp_min(1.0)
    loss = (loss_per_token * mask).sum() / valid_tokens

    with torch.no_grad():
        correct = (pred_logits.argmax(-1) == targets).to(torch.float32) * mask
        add_metric("accuracy", correct.sum(), den=mask.sum(), tag="train")
    add_metric("loss", loss.detach(), reduction="dp_mean", tag="train")
    return loss


__all__ = ["compute_mtp_loss"]
