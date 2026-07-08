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
        # Greedy speculative-decoding acceptance rate: a drafted token is
        # accepted iff it exactly matches what the target model would have
        # produced, i.e. this top-1 match rate.
        correct = (pred_logits.argmax(-1) == targets).to(torch.float32) * mask
        add_metric("accept_rate_greedy", correct.sum(), den=mask.sum(), tag="train")

        # Soft/probabilistic acceptance rate: 1 - 0.5*L1(draft_probs, target_probs),
        # matching the eagle3 accept_rate@i convention. target_last_hidden_states is
        # the target model's own final hidden state, and model.lm_head is a frozen
        # copy of the target's lm_head, so applying it directly (bypassing the draft
        # head) at position j+1 reconstructs the target's own next-token distribution
        # for the same token pred_logits[:, j] is trying to predict.
        # T=1 matches the untempered target distribution reconstructed above;
        # only valid as a rejection-sampling proxy when target sampling is
        # itself T=1 (rescale both logits by T before softmax otherwise).
        temperature = 1.0
        target_logits = model.lm_head(target_last_hidden_states[:, 1:-1, :])
        draft_probs = torch.softmax(pred_logits.float() / temperature, dim=-1)
        target_probs = torch.softmax(target_logits.float() / temperature, dim=-1)
        accept_rate_soft = (1.0 - 0.5 * (draft_probs - target_probs).abs().sum(-1)) * mask
        add_metric("accept_rate_soft", accept_rate_soft.sum(), den=mask.sum(), tag="train")
    add_metric("loss", loss.detach(), reduction="dp_mean", tag="train")
    return loss


__all__ = ["compute_mtp_loss"]
