import torch
import torch.nn.functional as F


def polarity_contrastive_loss(
    senti_repr,
    senti_types,
    senti_sample_masks,
    temperature=0.07,
):
    """
    Polarity-aware contrastive learning for sentiment triplets.

    Args:
        senti_repr: (B, N, D) sentiment pair representations before classifier
        senti_types: (B, N, C) one-hot sentiment labels
        senti_sample_masks: (B, N) valid sentiment masks
        temperature: InfoNCE temperature

    Returns:
        Contrastive loss scalar
    """
    batch_size, num_pairs, hidden_dim = senti_repr.shape

    # Get polarity labels from one-hot encoding
    # senti_types: (B, N, C) where C is number of sentiment classes
    # Assume class 0 is 'None', classes 1+ are polarities
    polarity_labels = senti_types.argmax(dim=-1)  # (B, N)

    # Mask out invalid and 'None' samples
    valid_mask = senti_sample_masks.bool() & (polarity_labels > 0)
    if valid_mask.sum() < 2:
        # Need at least 2 valid samples for contrastive loss
        return senti_repr.sum() * 0.0

    # Flatten batch dimension
    flat_repr = senti_repr.view(batch_size * num_pairs, hidden_dim)
    flat_labels = polarity_labels.view(batch_size * num_pairs)
    flat_mask = valid_mask.view(batch_size * num_pairs)

    # Extract valid samples
    valid_repr = flat_repr[flat_mask]  # (N_valid, D)
    valid_labels = flat_labels[flat_mask]  # (N_valid,)

    if valid_repr.shape[0] < 2:
        return senti_repr.sum() * 0.0

    # Normalize representations
    valid_repr = F.normalize(valid_repr, dim=-1)

    # Compute pairwise similarity matrix
    sim_matrix = torch.matmul(valid_repr, valid_repr.t()) / temperature  # (N_valid, N_valid)

    # Create label mask: same polarity = positive, different = negative
    label_matrix = valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)  # (N_valid, N_valid)

    # Mask out diagonal (self-similarity)
    mask_eye = torch.eye(valid_repr.shape[0], device=valid_repr.device, dtype=torch.bool)
    label_matrix = label_matrix & (~mask_eye)

    # For numerical stability
    sim_matrix_exp = torch.exp(sim_matrix)

    # Detach negative samples to prevent model collapse
    neg_mask = ~label_matrix & (~mask_eye)
    sim_matrix_exp_detached = sim_matrix_exp.clone()
    sim_matrix_exp_detached[neg_mask] = sim_matrix_exp[neg_mask].detach()

    # Compute InfoNCE loss for each anchor
    losses = []
    for i in range(valid_repr.shape[0]):
        # Positive samples for anchor i
        pos_mask = label_matrix[i]
        if pos_mask.sum() == 0:
            continue

        # Numerator: sum of positive similarities
        pos_sim = sim_matrix_exp[i][pos_mask].sum()

        # Denominator: sum of all similarities (excluding self)
        all_sim = sim_matrix_exp_detached[i][~mask_eye[i]].sum()

        # InfoNCE loss
        loss_i = -torch.log(pos_sim / (all_sim + 1e-8))
        losses.append(loss_i)

    if len(losses) == 0:
        return senti_repr.sum() * 0.0

    return torch.stack(losses).mean()
