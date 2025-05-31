import torch
import itertools

# Import config from the same directory
import config


def si_snr_loss(
    estimated_sources,
    target_sources,
    n_sources=config.N_SOURCES,
    pit=True,
    reduction="mean",
):
    """
    Calculates the Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss.
    Optionally uses Permutation Invariant Training (PIT).

    Args:
        estimated_sources (torch.Tensor): Estimated sources (batch, n_sources, time).
        target_sources (torch.Tensor): Target sources (batch, n_sources, time).
        n_sources (int): Number of sources.
        pit (bool): Whether to use Permutation Invariant Training.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.

    Returns:
        torch.Tensor: The calculated SI-SNR loss.
        torch.Tensor: The SI-SNR value (positive, for monitoring).
    """
    batch_size = estimated_sources.shape[0]
    max_snr_list = []

    for b_idx in range(batch_size):
        est_s = estimated_sources[b_idx]  # (n_sources, time)
        tgt_s = target_sources[b_idx]  # (n_sources, time)

        if pit:
            perms = list(itertools.permutations(range(n_sources)))
            snr_for_perms = []
            for p in perms:
                tgt_s_permuted = tgt_s[list(p), :]

                # s_target = <s_hat, s>s / ||s||^2
                s_target_num = torch.sum(est_s * tgt_s_permuted, dim=1, keepdim=True)
                s_target_den = torch.sum(tgt_s_permuted**2, dim=1, keepdim=True)
                s_target = (s_target_num / (s_target_den + config.EPS)) * tgt_s_permuted

                # e_noise = s_hat - s_target
                e_noise = est_s - s_target

                # SNR = 10 * log10 (||s_target||^2 / ||e_noise||^2)
                snr = 10 * torch.log10(
                    torch.sum(s_target**2, dim=1)
                    / (torch.sum(e_noise**2, dim=1) + config.EPS)
                    + config.EPS
                )
                snr_for_perms.append(
                    torch.mean(snr)
                )  # Average SNR over sources for this permutation

            max_snr_for_batch_item = torch.max(torch.stack(snr_for_perms))
            max_snr_list.append(max_snr_for_batch_item)
        else:  # No PIT, direct calculation
            s_target_num = torch.sum(est_s * tgt_s, dim=1, keepdim=True)
            s_target_den = torch.sum(tgt_s**2, dim=1, keepdim=True)
            s_target = (s_target_num / (s_target_den + config.EPS)) * tgt_s
            e_noise = est_s - s_target
            snr = 10 * torch.log10(
                torch.sum(s_target**2, dim=1)
                / (torch.sum(e_noise**2, dim=1) + config.EPS)
                + config.EPS
            )
            max_snr_list.append(torch.mean(snr))

    final_snr = torch.stack(max_snr_list)

    if reduction == "mean":
        loss = -torch.mean(final_snr)  # Negative because we want to maximize SI-SNR
        return loss, -loss  # Return loss and actual SI-SNR value
    elif reduction == "sum":
        loss = -torch.sum(final_snr)
        return loss, -loss / batch_size  # Return sum loss and avg SI-SNR
    else:  # 'none'
        return -final_snr, final_snr


def match_length(estimated, target):
    """Truncates estimated and target tensors to the minimum of their lengths."""
    min_len = min(estimated.shape[-1], target.shape[-1])
    return estimated[..., :min_len], target[..., :min_len]
