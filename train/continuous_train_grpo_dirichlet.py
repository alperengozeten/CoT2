import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Dirichlet
from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
import numpy as np
import random
import argparse
import copy

from data.generate_data import (
    generate_vocab,
    dataset_random_split,
    permutation_train_val_split_continuous,
)
from utils import plot_losses, plot_accuracies
import itertools

# seed everything for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FourDigitsSoftDataset(Dataset):
    """
    For each 4-digit sequence in [1..5], we:
      1) Start from partial_sum=0.
      2) For i in [1..4], we expand partial sums by +/- seq[i],
         resulting in 2^i partial sums, each with uniform probability 1/(2^i).
         Build a dist_vec for step i, setting dist_vec[S{ps}] = 1/(2^i) if "S{ps}" in vocab.
      3) Among those 16 final sums, pick the min non-negative sum as final_hard_label = S{best_sum}.
      4) If no non-negative sum is found, skip the sequence.
    """
    def __init__(self, token2id, digit_range=range(1, 6), seq_length=4):
        super().__init__()
        self.token2id = token2id
        self.vocab_size = len(token2id)
        self.examples = []

        for seq in itertools.product(digit_range, repeat=seq_length):
            # We'll accumulate partial sums at each step,
            # always starting from 0 for step 0.
            partial_sums_at_step = []
            current_partial_sums = [0]  # step 0
            partial_sums_at_step.append(current_partial_sums)

            # Build dist_steps (4 steps, each distribution over S{ps})
            dist_steps = []

            for i in range(seq_length):
                # Expand to 2^i+1 partial sums
                new_sums = []
                digit = seq[i]
                for ps in current_partial_sums:
                    new_sums.append(ps + digit)
                    new_sums.append(ps - digit)
                current_partial_sums = new_sums
                partial_sums_at_step.append(current_partial_sums)

                # Now build a distribution vector: each sum has probability 1/2^(i+1)
                dist_vec = torch.zeros(self.vocab_size)
                prob = 1.0 / len(current_partial_sums)  # = 1/(2^(i+1))
                for ps_val in current_partial_sums:
                    key = f"S{ps_val}"
                    if key in self.token2id:
                        dist_vec[self.token2id[key]] += prob
                # dist_steps.append(torch.sqrt(dist_vec))
                dist_steps.append(dist_vec)

            # current_partial_sums now has 16 final sums (2^4)
            # pick smallest non-negative final sum
            final_sums = current_partial_sums
            best_sum = None
            for candidate in sorted(final_sums):
                if candidate >= 0:
                    best_sum = candidate
                    break

            if best_sum is None:
                # skip if no non-negative sum
                continue

            # final label => S{best_sum}
            final_label_str = f"S{best_sum}"
            if final_label_str not in self.token2id:
                # skip if not in vocab
                continue
            final_label_id = self.token2id[final_label_str]

            # Build the prompt: <BOS> + digits
            #prompt_tokens = ["<BOS>"] + [f"D{d}" for d in seq] + ["->"]
            prompt_tokens = ["<BOS>"] + [f"D{d}" for d in seq] 
            prompt_ids = []
            for pt in prompt_tokens:
                if pt in self.token2id:
                    prompt_ids.append(self.token2id[pt])

            ex = {
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "dist_steps": dist_steps,  # 4 distributions
                "final_hard_label": final_label_id
            }
            self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    max_len = max(len(ex["prompt_ids"]) for ex in batch)
    prompt_list = []
    attn_list = []
    dist_steps_list = []
    final_labels_list = []

    for ex in batch:
        p = ex["prompt_ids"]
        pad_len = max_len - len(p)
        padded = torch.cat([p, torch.full((pad_len,), 0, dtype=torch.long)])
        attn = torch.cat([torch.ones(len(p)), torch.zeros(pad_len)])

        prompt_list.append(padded.unsqueeze(0))
        attn_list.append(attn.unsqueeze(0))
        dist_steps_list.append(ex["dist_steps"])
        final_labels_list.append(ex["final_hard_label"])

    prompt_ids = torch.cat(prompt_list, dim=0)     # (B, max_len)
    attention_mask = torch.cat(attn_list, dim=0)  # (B, max_len)

    return {
        "prompt_ids": prompt_ids,
        "attention_mask": attention_mask,
        "dist_steps": dist_steps_list,      # list of lists
        "final_labels": final_labels_list
    }


def cross_entropy_distribution(logits, target_dist):
    """
    logits: shape (vocab_size,)
    target_dist: shape (vocab_size,)
    Returns scalar: cross-entropy = - sum_{k} p(k) log softmax(logits)[k]
    """
    EPS = 10**(-8)
    log_probs = F.log_softmax(logits, dim=-1)
    return - (target_dist * log_probs).sum() + (target_dist*torch.log(target_dist + EPS)).sum()


# GRPO ROLLOUT (K sampling)
def rollout_grpo_batch(
    model,
    old_model,
    prompt_ids,         # (B, seq_len)
    attention_mask,     # (B, seq_len)
    final_labels,       # (B,)
    seq_length=4,             # => we do 3 "soft" steps, then 1 final step
    K=3,
    num_rollouts=5,
    clip_epsilon=0.1,
    kl_coeff=0.0,
    ref_model=None,
    sampling_normalization=True
):
    """
    For each example in the batch (size B), generate 'num_rollouts' separate
    trajectories. Each trajectory has:
      - (T-1) "soft" steps, each sampling K=3 tokens from the old_model's distribution,
        then feeding back the average embedding
      - 1 final "hard" token

    We explicitly compute:
      - prob_theta( tau ) = product of stepwise probabilities under model 'model'
      - prob_old  ( tau ) = product of stepwise probabilities under 'old_model'

    Returns:
      ratio:    (B, num_rollouts) => ratio[i,m] = prob_theta(tau_i,m) / prob_old(tau_i,m)
      rewards:  (B, num_rollouts) => 0 or 1
      final_tokens: (B, num_rollouts) => the final chosen token
    """
    old_model.eval()

    B = prompt_ids.size(0)
    device = prompt_ids.device

    rollout_loss = torch.zeros((), device=device, requires_grad=True)
    kl_loss = torch.zeros((), device=device, requires_grad=True)

    # Loop over each example i in the batch
    for i in range(B):
        pi = prompt_ids[i].unsqueeze(0)
        am = attention_mask[i].unsqueeze(0)
        correct_label = final_labels[i].item()

        # encode prompt with grads
        out = model(pi, attention_mask=am, use_cache=True, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :]
        pkv = out.past_key_values

        with torch.no_grad():
            old_out = old_model(pi, attention_mask=am, use_cache=True, output_hidden_states=True)
            old_last_hidden = old_out.hidden_states[-1][:, -1, :]
            old_pkv = old_out.past_key_values

        if ref_model is not None:
            with torch.no_grad():
                ref_out = ref_model(pi, attention_mask=am, use_cache=True, output_hidden_states=True)
                ref_last_hidden = ref_out.hidden_states[-1][:, -1, :]
                ref_pkv = ref_out.past_key_values
        else:
            ref_last_hidden, ref_pkv = None, None

        partial_ratios_list = []
        rewards_list = []
        # For each rollout
        for m in range(num_rollouts):
            # Make local copies so each rollout is independent
            local_pkv = [tuple(t.clone() for t in layer) for layer in pkv]
            local_hidden = last_hidden.clone()

            local_old_pkv = [tuple(t.clone() for t in layer) for layer in old_pkv]
            local_old_hidden = old_last_hidden.clone()

            if ref_model is not None:
                local_ref_pkv = [tuple(t.clone() for t in layer) for layer in ref_pkv]
                local_ref_hidden = ref_last_hidden.clone()
            else:
                local_ref_pkv, local_ref_hidden = None, None

            # We'll store partial-step (ratio) terms in lists,
            # then finalize the advantage after final token
            partial_ratios = []

            # --- (T-1) "soft" steps ---
            for step_idx in range(seq_length - 1):
                # Current model's distribution
                logits = model.lm_head(local_hidden)  # (1, vocab_size)
                probs = F.softmax(logits, dim=-1).squeeze(0)

                with torch.no_grad():
                    old_logits = old_model.lm_head(local_old_hidden)
                    old_probs = F.softmax(old_logits, dim=-1).squeeze(0)

                if ref_model is not None:
                    with torch.no_grad():
                        ref_logits = ref_model.lm_head(local_ref_hidden)
                        ref_probs = F.softmax(ref_logits, dim=-1).squeeze(0)
                else:
                    ref_probs = None

                MIN_PROB = 1e-3
                DIRICHLET_SCALE = 1

                # 1) Build a union_mask across old_probs, new_probs, ref_probs
                union_mask = (probs >= MIN_PROB) | (old_probs >= MIN_PROB)
                if ref_probs is not None:
                    union_mask = union_mask | (ref_probs >= MIN_PROB)

                # 2) Extract just the union dimension for each distribution
                probs_subset = probs[union_mask].clone()
                old_probs_subset = old_probs[union_mask].clone()
                if ref_probs is not None:
                    ref_probs_subset = ref_probs[union_mask].clone()
                else:
                    ref_probs_subset = None

                # 3) Clamp everything below MIN_PROB up to MIN_PROB, so Dirichlet won't choke
                probs_subset.clamp_(min=MIN_PROB)
                old_probs_subset.clamp_(min=MIN_PROB)
                if ref_probs_subset is not None:
                    ref_probs_subset.clamp_(min=MIN_PROB)

                # 4) Renormalize each subset
                probs_subset /= probs_subset.sum()
                old_probs_subset /= old_probs_subset.sum()
                if ref_probs_subset is not None:
                    ref_probs_subset /= ref_probs_subset.sum()

                # 5) Build Dirichlet from these smaller “subset” vectors
                dirichlet_new = Dirichlet(probs_subset * DIRICHLET_SCALE)
                dirichlet_old = Dirichlet(old_probs_subset * DIRICHLET_SCALE)
                if ref_probs_subset is not None:
                    dirichlet_ref = Dirichlet(ref_probs_subset * DIRICHLET_SCALE)

                # 6) Sample in the reduced dimension
                alpha_old_subset = dirichlet_old.sample()  # shape = [#union]

                # 7) Expand alpha_old_subset back to the full dimension
                #    by placing its values into alpha_old[union_mask].
                alpha_old = torch.zeros_like(probs)  # shape = [vocab_size]
                alpha_old[union_mask] = alpha_old_subset

                # 8) Then log-prob must also be computed in the subset dimension
                log_old = dirichlet_old.log_prob(alpha_old_subset)
                log_new = dirichlet_new.log_prob(alpha_old_subset)
                ratio_t_old = torch.exp(log_new - log_old)
                partial_ratios.append(ratio_t_old)

                # Use Schulman approximator to compute KL Loss
                if ref_model is not None and kl_coeff > 1e-9:
                    log_ref = dirichlet_ref.log_prob(alpha_old)
                    ratio_t_ref = log_ref / log_new
                    kl_approx = ratio_t_ref - torch.log(ratio_t_ref) - 1.0
                    kl_loss = kl_loss + kl_coeff * kl_approx

                # feed average embedding
                embedding_weights = alpha_old.unsqueeze(0)  # shape (1, vocab_size)
                wte = model.transformer.wte.weight  # shape (vocab_size, emb_dim)
                avg_emb = torch.matmul(embedding_weights, wte)  # shape (1, emb_dim)

                out2 = model(
                    inputs_embeds=avg_emb.unsqueeze(0),
                    past_key_values=local_pkv,
                    use_cache=True,
                    output_hidden_states=True
                )
                local_hidden = out2.hidden_states[-1][:, -1, :]
                local_pkv = out2.past_key_values

                # old model => step prob
                with torch.no_grad():
                    old_out2 = old_model(
                        inputs_embeds=avg_emb.unsqueeze(0),
                        past_key_values=local_old_pkv,
                        use_cache=True,
                        output_hidden_states=True
                    )
                    local_old_hidden = old_out2.hidden_states[-1][:, -1, :]
                    local_old_pkv = old_out2.past_key_values

                if ref_model is not None:
                    with torch.no_grad():
                        ref_out2 = ref_model(
                            inputs_embeds=avg_emb.unsqueeze(0),
                            past_key_values=local_ref_pkv,
                            use_cache=True,
                            output_hidden_states=True
                        )
                        local_ref_hidden = ref_out2.hidden_states[-1][:, -1, :]
                        local_ref_pkv = ref_out2.past_key_values

            # final "hard" step => single token
            final_logits = model.lm_head(local_hidden)
            final_probs = F.softmax(final_logits, dim=-1).squeeze(0)

            with torch.no_grad():
                # old model probability of that final token
                old_final_logits = old_model.lm_head(local_old_hidden)
                old_final_probs = F.softmax(old_final_logits, dim=-1).squeeze(0)

            ftoken = torch.multinomial(old_final_probs, 1).item()
            alpha_old = old_final_probs[ftoken]
            alpha_new = final_probs[ftoken]

            # do the kl again using Schulman approximator
            if ref_model is not None and kl_coeff > 1e-9:
                with torch.no_grad():
                    ref_final_logits = ref_model.lm_head(local_ref_hidden)
                    ref_final_probs = F.softmax(ref_final_logits, dim=-1).squeeze(0)
                    alpha_ref = ref_final_probs[ftoken]
                    ratio_final_ref = alpha_ref / (alpha_new + 1e-30)
                    kl_approx = ratio_final_ref - torch.log(ratio_final_ref) - 1.0
                    kl_loss = kl_loss + (kl_coeff * kl_approx)

            ratio_final = alpha_new / (alpha_old + 1e-30)
            partial_ratios.append(ratio_final)

            # reward
            reward_val = float(ftoken == correct_label)
            partial_ratios_list.append(partial_ratios)
            rewards_list.append(reward_val)

        # Now we compute baseline and std of rewards
        if len(rewards_list) > 0:
            baseline = sum(rewards_list) / len(rewards_list)
            rewards_std = float(np.std(rewards_list))
            if rewards_std < 1e-12:
                rewards_std = 1e-12
        else:
            baseline = 0.0
            rewards_std = 1e-12  # to avoid div-by-zero

        # For each rollout, apply advantage = (reward - baseline)
        for m in range(num_rollouts):
            advantage = (rewards_list[m] - baseline) / rewards_std
            # partial_ratios_list[m] => partial ratios for that rollout
            for ratio_t in partial_ratios_list[m]:
                unclipped = ratio_t * advantage
                clipped = ratio_t.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                obj = torch.min(unclipped, clipped)
                # negative => we want to MINimize
                rollout_loss = rollout_loss - obj

    rollout_loss = rollout_loss + kl_loss
    # For GRPO objective, divide by the total number of generated tokens
    rollout_loss = rollout_loss / float(num_rollouts * seq_length)
    return rollout_loss


# TRAIN LOOP
def train_grpo(
    model,
    data_loader,
    optimizer,
    scheduler,
    device,
    seq_length,
    num_rollouts=5,
    kl_coeff=0.0,
    K=3,
    clip_epsilon=0.1,
    ref_model=None,
    sampling_normalization=True
):
    """
    - For each batch, we do multi-rollout sampling.
    - The ratio = pi_theta(tau) / pi_old(tau) is computed explicitly (product of stepwise probabilities).
    - The advantage is simply reward[i,m] (since we have a sparse final reward).
    - We do a second pass to compute log p_theta( final_token ), weighted by ratio*advantage.
    - Optionally add KL with old_model.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0

    if ref_model is None:
        ref_model = copy.deepcopy(model).eval()

    for batch in data_loader:
        old_model = copy.deepcopy(model).eval()
        prompt_ids = batch["prompt_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        final_labels = torch.tensor(batch["final_labels"], device=device)
        B = prompt_ids.size(0)

        # Single pass rollout => direct PPO objective
        rollout_loss = rollout_grpo_batch(
            model=model,
            old_model=old_model,
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            final_labels=final_labels,
            seq_length=seq_length,
            K=K,
            num_rollouts=num_rollouts,
            clip_epsilon=clip_epsilon,
            kl_coeff=kl_coeff,
            ref_model=ref_model,
            sampling_normalization=sampling_normalization
        )

        # average over batch
        batch_loss = rollout_loss / float(B)

        # Optimize
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += batch_loss.item()
        total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


def main(args=None):
    # set hyperparameters
    MAX_SEQ_LEN = args.max_seq_len
    EMBEDDING_DIM = args.embedding_dim
    DIGIT_RANGE = args.digit_range
    BATCH_SIZE = args.batch_size
    SEQ_LENGTH = args.seq_length
    SPLIT_METHOD = args.split_method
    NUM_EPOCHS = args.num_epochs
    OUTPUT_DIR = args.output_dir
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    KL_COEFF = args.kl_coeff
    NUM_ROLLOUTS = args.num_rollouts
    K = args.K
    CLIP_EPSILON = args.clip_epsilon
    SAMPLING_NORMALIZATION = args.sampling_normalization

    # Build vocab & model
    vocab, token2id, id2token = generate_vocab()
    vocab_size = len(vocab)
    print("Vocab size =", vocab_size)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_SEQ_LEN,
        n_embd=EMBEDDING_DIM,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
    )
    model = GPT2LMHeadModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory_name = f"continuous_model_digit_range_{DIGIT_RANGE[0]}_{DIGIT_RANGE[-1] + 1}_Seq_Length_{SEQ_LENGTH}_Layer_{NUM_LAYERS}_Head_{NUM_HEADS}"
    #directory_name = f"discrete_model_for_grpo_digit_range_{DIGIT_RANGE[0]}_{DIGIT_RANGE[-1] + 1}_Seq_Length_{SEQ_LENGTH}_Layer_{NUM_LAYERS}_Head_{NUM_HEADS}"
    model_path = f"models/{directory_name}/continuous_model_Digit{DIGIT_RANGE[0]}-{DIGIT_RANGE[-1] + 1}_seq{SEQ_LENGTH}_emb{EMBEDDING_DIM}_steps{''.join(map(str, range(SEQ_LENGTH-1)))}h_supervisionhard_teacher_split{SPLIT_METHOD}_batch{BATCH_SIZE}_epochs2000.pt"
    #model_path = f"models/{directory_name}/discrete_model_Digit{DIGIT_RANGE[0]}-{DIGIT_RANGE[-1] + 1}_Mod1_Seq{SEQ_LENGTH}_Emb{EMBEDDING_DIM}_Split{SPLIT_METHOD}_Batch{BATCH_SIZE}_Epochs1000.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Create the dataset
    dataset = FourDigitsSoftDataset(
        token2id=token2id, digit_range=DIGIT_RANGE, seq_length=SEQ_LENGTH
    )

    # Split train/val
    train_size = int(0.8 * len(dataset))
    if SPLIT_METHOD == "sequential":
        train_data = dataset[:train_size]
        val_data = dataset[train_size:]
    elif SPLIT_METHOD == "random":
        train_data, val_data = dataset_random_split(dataset, train_ratio=0.8)
    else:  # random_permutation
        train_data, val_data = permutation_train_val_split_continuous(
            dataset=dataset,
            id2token=id2token,
            seq_length=SEQ_LENGTH,
            train_ratio=0.8
        )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    num_training_steps = 410 * 100
    num_warmup_steps = int(0.1 * num_training_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    #scheduler = get_cosine_schedule_with_warmup(
    #    optimizer=optimizer,
    #    num_training_steps=num_training_steps
    #    num_warmup_steps=num_warmup_steps,
    #)
    scheduler = None  

    train_losses = []
    val_losses = []
    val_accuracies = []

    ref_model = copy.deepcopy(model).eval()

    for epoch in range(NUM_EPOCHS):
        train_loss = train_grpo(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            seq_length=SEQ_LENGTH,
            num_rollouts=NUM_ROLLOUTS,
            kl_coeff=KL_COEFF,
            K=K,
            clip_epsilon=CLIP_EPSILON,
            ref_model=ref_model,
            sampling_normalization=SAMPLING_NORMALIZATION
        )

        # Evaluate
        val_loss, val_acc, val_entropies = grpo_eval_soft_steps_acc(model, val_loader, device, SEQ_LENGTH, K, sampling_normalization=SAMPLING_NORMALIZATION)
        train_label_loss, train_acc, train_entropies = grpo_eval_soft_steps_acc(model, train_loader, device, SEQ_LENGTH, K, sampling_normalization=SAMPLING_NORMALIZATION)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train RL Loss: {train_loss:.4f} | "
              f"Train Label Loss: {train_label_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | "
              f"Val Label Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2%} | "
              f"Val Entropies: {[f'{e:.4f}' for e in val_entropies]} | "
              f"Train Entropies: {[f'{e:.4f}' for e in train_entropies]}")

    # Plot
    epochs_range = range(1, NUM_EPOCHS + 1)
    plot_losses(epochs_range, train_losses, val_losses, model_type="grpo_multirollout", output_dir=OUTPUT_DIR)
    plot_accuracies(epochs_range, val_accuracies, model_type="grpo_multirollout", output_dir=OUTPUT_DIR)

    # Save
    torch.save(model.state_dict(), f"models/{OUTPUT_DIR}grpo_multirollout_model_Digit{DIGIT_RANGE[0]}-{DIGIT_RANGE[-1] + 1}_seq{SEQ_LENGTH}_emb{EMBEDDING_DIM}_K_{K}_split{SPLIT_METHOD}_batch{BATCH_SIZE}_epochs{NUM_EPOCHS}.pt")


@torch.no_grad()
def grpo_eval_soft_steps_acc(model, data_loader, device, seq_length, K, sampling_normalization=True):
    """
    Evaluates the model on the 4-step distribution tasks + final step,
    computing:
      - total loss (distribution steps + final CE)
      - accuracy of the final predicted token vs. final_hard_label
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    steps = 0
    step_entropy_sums = [0.0] * seq_length

    ce_loss_fn = nn.CrossEntropyLoss()

    def get_last_logits(o):
        return o.logits[:, -1, :]

    for batch in data_loader:
        prompt_ids = batch["prompt_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        final_labels = batch["final_labels"]
        batch_size = prompt_ids.size(0)

        batch_loss = 0.0  # python float is fine; each step_loss will track grad separately

        for i in range(batch_size):
            # Forward the prompt in discrete form
            pi = prompt_ids[i].unsqueeze(0)
            am = attention_mask[i].unsqueeze(0)

            outputs = model(pi, attention_mask=am, use_cache=True)
            pkv = outputs.past_key_values

            # Distribution steps (1..4)
            for count in range(seq_length - 1):
                last_logits = get_last_logits(outputs).squeeze(0)  # shape (vocab_size,)
                if sampling_normalization:
                    last_logits = last_logits / K
                probs = F.softmax(last_logits, dim=-1)

                # calculate the entropy for this step
                entropy = -(probs * (probs + 1e-12).log()).sum()
                step_entropy_sums[count] += entropy.item()

                # continuous token by averaging K sampled discrete tokens
                sampled_tokens = torch.multinomial(probs, K, replacement=True)
                emb = model.transformer.wte(sampled_tokens)
                avg_emb = emb.mean(dim=0, keepdim=True)

                out2 = model(
                    inputs_embeds=avg_emb.unsqueeze(0),  # (1,1,n_embd)
                    past_key_values=pkv,
                    use_cache=True
                )
                pkv = out2.past_key_values
                outputs = out2

            # final step => measure CE to final_label, also do discrete "prediction" for accuracy
            last_logits = get_last_logits(outputs).squeeze(0)  # (vocab_size,)
            final_label_id = final_labels[i]

            # calculate the entropy for the final step
            final_probs = F.softmax(last_logits, dim=-1)
            entropy_final = -(final_probs * (final_probs + 1e-12).log()).sum()
            step_entropy_sums[seq_length - 1] += entropy_final.item()

            # we can do cross-entropy with the final label
            final_loss = ce_loss_fn(last_logits.unsqueeze(0), torch.tensor([final_label_id], device=device))
            batch_loss += final_loss.item()

            # for accuracy, pick argmax
            predicted_id = last_logits.argmax(dim=-1).item()
            if predicted_id == final_label_id:
                total_correct += 1

        # average over batch
        batch_loss /= batch_size
        total_loss += batch_loss
        total_samples += batch_size
        steps += 1
    
    avg_entropies = [esum / float(total_samples) for esum in step_entropy_sums]
    avg_loss = total_loss / steps
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, avg_entropies


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--digit_range", type=int, nargs=2, default=[1, 10])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--split_method", type=str, default="random_permutation")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the model.")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads in the model.")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Number of rollouts for GRPO.")
    parser.add_argument("--kl_coeff", type=float, default=0.1, help="KL coefficient for GRPO.")
    parser.add_argument("--K", type=int, default=3, help="Number of samples for GRPO.")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="Clipping epsilon for GRPO.")
    parser.add_argument("--sampling_normalization", choices=['true', 'false'], default='true',
        help='Normalize logits during sampling (true/false).'
    )
    args = parser.parse_args()
    args.digit_range = range(args.digit_range[0], args.digit_range[1])
    args.sampling_normalization = (args.sampling_normalization == 'true')

    main(args)
