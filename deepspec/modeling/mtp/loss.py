import torch
import torch.nn.functional as F

from deepspec.utils.metrics import add_metric


def compute_mtp_loss(*, model, batch, ce_loss_alpha=1.0, kl_loss_alpha=0.0):
    """Teacher-forced MTP training loss, optionally distilled from the target.

    ``batch["target_last_hidden_states"][:, i]`` is the base model's final
    hidden state after seeing tokens[0..i] (from an offline target cache built
    by ``scripts/data/prepare_target_cache.py``). MTP pairs that with the
    embedding of token i+1 to predict token i+2, so everything here is shifted
    relative to the cached, unshifted arrays.

    ``ce_loss_alpha``/``kl_loss_alpha`` weight the two loss terms below (same
    ``*_alpha`` convention as ``deepspec/modeling/dspark/loss.py``). Default
    to CE-only (``kl_loss_alpha=0.0``) so existing MTP recipes that don't set
    these keys train identically to before.
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
    valid_tokens = mask.sum().clamp_min(1.0)

    ce_loss_per_token = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    ce_loss = (ce_loss_per_token * mask).sum() / valid_tokens

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
        draft_probs = torch.softmax(pred_logits / temperature, dim=-1)
        target_probs = torch.softmax(target_logits / temperature, dim=-1)
        accept_rate_soft = (1.0 - 0.5 * (draft_probs - target_probs).abs().sum(-1)) * mask
        add_metric("accept_rate_soft", accept_rate_soft.sum(), den=mask.sum(), tag="train")
        # target_logits/draft_probs are full (batch, seq, vocab) tensors with
        # no further use past this point -- at seq_len=12288 and vocab~152k
        # each is several GiB, so drop the references now rather than let
        # them sit alive (still bound to these names) through the KL term
        # below, which needs its own set of same-sized tensors.
        del target_logits, draft_probs

    # KL(target_probs ‖ draft_probs) distillation term: unlike accept_rate_soft
    # above (purely diagnostic, no_grad), this backprops into pred_logits so the
    # draft's full next-token distribution is actually pulled toward the target's
    # own -- the thing missing from CE-only training, which only ever sees the
    # single ground-truth corpus token and gives accept_rate_soft no gradient
    # signal. target_probs came out of the no_grad block above, so it's already
    # detached; the target model (frozen) receives no gradient from this term.
    # Kept in bf16 (not upcast to fp32 like the CE path) -- these are the same
    # (batch, seq, vocab) shape as above, and softmax/log_softmax stability
    # depends on exponent range (unaffected by bf16's reduced mantissa), not
    # on fp32. Only the post-reduction per-token value (vocab dim already
    # summed out, so tiny) is cast up for loss-accumulation precision.
    draft_log_probs = torch.log_softmax(pred_logits / temperature, dim=-1)
    kl_loss_per_token = F.kl_div(
        draft_log_probs, target_probs, reduction="none"
    ).sum(-1).float()
    kl_loss = (kl_loss_per_token * mask).sum() / valid_tokens

    loss = ce_loss_alpha * ce_loss + kl_loss_alpha * kl_loss

    add_metric("ce_loss", (ce_loss_per_token.detach() * mask).sum(), den=mask.sum(), tag="train")
    add_metric("kl_loss", (kl_loss_per_token.detach() * mask).sum(), den=mask.sum(), tag="train")
    add_metric("loss", loss.detach(), reduction="dp_mean", tag="train")
    return loss


__all__ = ["compute_mtp_loss"]
