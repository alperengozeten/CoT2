import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from data.generate_data import (
    generate_vocab,
    dataset_random_split,
    permutation_train_val_split_continuous,
)
from utils import plot_losses, plot_accuracies
import numpy as np
import random
import itertools
import argparse


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FourDigitsSoftDataset(Dataset):
    """
    dist_strategy = "uniform" ("beam" with num_beams=all):
      At step i, assign 1/2^(i+1) to each occurrence of a partial sum among all paths (duplicates accumulate).
    dist_strategy = "beam":
      1) Enumerate all 2^L sign sequences; keep those with final sum >= 0.
      2) Sort by final sum ascending (tie-break by sign tuple for determinism), take top-K.
      3) For each step t, count states visited across those K beams; normalize to a distribution.

    We still emit len(dist_steps) == seq_length. Training will roll out only the first (seq_length-1) soft steps
    and then supervise the final hard token (h).
    """
    def __init__(self,
                 token2id,
                 digit_range=range(1, 10),
                 seq_length=4,
                 dist_strategy="uniform",
                 num_beams=8):
        super().__init__()
        self.token2id = token2id
        self.vocab_size = len(token2id)
        self.examples = []
        self.seq_length = seq_length
        self.dist_strategy = dist_strategy
        self.num_beams = num_beams

        for seq in itertools.product(digit_range, repeat=seq_length):
            if self.dist_strategy == "beam_minabs":
                # per-step distributions from top-K by |final|
                dist_steps, topk = self._build_beam_minabs_dists(seq, return_topk=True)

                # final label: min non-negative among the kept K traces
                cand_nonneg = [final for (final, _, _) in topk if final >= 0]
                if not cand_nonneg:
                    continue  # skip if none of the kept traces end non-negative
                best_sum = min(cand_nonneg)
                final_label_str = f"S{best_sum}"
                if final_label_str not in self.token2id:
                    continue

            else:  # uniform (num_beams=all)
                # final label: smallest non-negative among ALL traces
                final_sums = []
                for signs in itertools.product((1, -1), repeat=seq_length):
                    s = 0
                    for d, sg in zip(seq, signs):
                        s += sg * d
                    final_sums.append(s)
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
                    continue

                # per-step distributions
                if self.dist_strategy == "uniform":
                    dist_steps = self._build_uniform_dists(seq)
                else:  # "beam"
                    dist_steps = self._build_beam_dists(seq)
                    if dist_steps is None:
                        continue

            # prompt ids
            prompt_tokens = ["<BOS>"] + [f"D{d}" for d in seq]
            prompt_ids = [self.token2id[pt] for pt in prompt_tokens if pt in self.token2id]

            self.examples.append({
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "dist_steps": dist_steps,           # list length == seq_length
                "final_hard_label": self.token2id[final_label_str],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def _build_uniform_dists(self, seq):
        dist_steps = []
        current_partial_sums = [0]
        for i, digit in enumerate(seq):
            new_sums = []
            for ps in current_partial_sums:
                new_sums.append(ps + digit)
                new_sums.append(ps - digit)
            current_partial_sums = new_sums

            # mass uniform over paths (duplicates accumulate)
            vec = torch.zeros(self.vocab_size)
            prob = 1.0 / len(current_partial_sums)
            for ps_val in current_partial_sums:
                key = f"S{ps_val}"
                if key in self.token2id:
                    vec[self.token2id[key]] += prob
            # vec sums to 1.0 with your default vocab
            dist_steps.append(vec)
        return dist_steps

    def _build_beam_dists(self, seq):
        L = len(seq)
        paths = []  # (final_sum, signs_tuple, partials_list)
        for signs in itertools.product((1, -1), repeat=L):
            partials = []
            s = 0
            for d, sg in zip(seq, signs):
                s += sg * d
                partials.append(s)
            paths.append((partials[-1], signs, partials))

        valid = [p for p in paths if p[0] >= 0]
        if not valid:
            return None

        # sort by final sum asc, then by sign tuple to be deterministic
        valid.sort(key=lambda x: (x[0], x[1]))
        K = min(self.num_beams, len(valid))
        topk = valid[:K]

        dist_steps = []
        for t in range(L):
            vec = torch.zeros(self.vocab_size)
            for _, _, partials in topk:
                val = partials[t]
                key = f"S{val}"
                if key in self.token2id:
                    vec[self.token2id[key]] += 1.0  # count occurrences across beams
            total = vec.sum().item()
            if total > 0:
                vec /= total  # normalize to a distribution
            dist_steps.append(vec)
        return dist_steps

    def _build_beam_minabs_dists(self, seq, return_topk=False):
        L = len(seq)
        paths = []  # (final_sum, signs_tuple, partials_list)
        for signs in itertools.product((1, -1), repeat=L):
            partials = []
            s = 0
            for d, sg in zip(seq, signs):
                s += sg * d
                partials.append(s)
            paths.append((s, signs, partials))

        # Sort by |final| ascending. Tie-break deterministically by
        # prefer non-negative when |final| ties,
        # then by final value, then by the sign tuple.
        paths.sort(key=lambda x: (abs(x[0]), 0 if x[0] >= 0 else 1, x[0], x[1]))
        K = min(self.num_beams, len(paths))
        topk = paths[:K]

        dist_steps = []
        for t in range(L):
            vec = torch.zeros(self.vocab_size)
            for final_sum, _, partials in topk:
                val = partials[t]
                key = f"S{val}"
                if key in self.token2id:
                    vec[self.token2id[key]] += 1.0  # count across retained traces
            total = vec.sum().item()
            if total > 0:
                vec /= total  # normalize
            dist_steps.append(vec)

        if return_topk:
            return dist_steps, topk
        return dist_steps

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


def cross_entropy_distribution_batch(logits, target_dist):
    """
    Batch version
    logits:  (B, vocab_size)
    dist:    (B, vocab_size)  (already on same device as logits)
    Returns (B,) => one scalar CE loss per example, same formula as above.
    """
    EPS = 1e-8
    log_probs = F.log_softmax(logits, dim=-1)  # (B, vocab_size)
    return -(target_dist * log_probs).sum(dim=-1) + (target_dist*torch.log(target_dist + EPS)).sum(dim=-1)


def train_soft_steps_batch(
    model,
    data_loader,
    optimizer,
    device,
    loss_steps_to_include,
    supervision_type,
    seq_length,
    num_soft_steps
):
    model.train()
    total_loss = 0.0
    steps = 0

    ce_loss_fn = nn.CrossEntropyLoss()

    # keep track of partial losses
    train_loss_tokens = np.zeros(seq_length, dtype=np.float32)
    embedding_matrix = model.transformer.wte.weight  # (vocab_size, n_embd)

    for batch in data_loader:
        # Move prompt and attention mask to device
        prompt_ids = batch["prompt_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device) 
        final_labels = torch.tensor(batch["final_labels"], device=device)

        all_dist_steps = []
        for ex_dists in batch["dist_steps"]:
            # ex_dists is e.g. [Tensor(vocab_size), Tensor(vocab_size), ...] or lists
            ex_tensors = []
            for d in ex_dists:
                # If it's already a Tensor, just .to(device); if it's a list, convert
                if isinstance(d, torch.Tensor):
                    ex_tensors.append(d.to(device))
                else:
                    ex_tensors.append(torch.tensor(d, dtype=torch.float, device=device))
            # Stack them: shape => (T, vocab_size)
            stacked = torch.stack(ex_tensors, dim=0)
            all_dist_steps.append(stacked)

        dist_steps = torch.stack(all_dist_steps, dim=0)  # (B, T, vocab_size)

        batch_size = prompt_ids.size(0)
        batch_loss = torch.zeros((), device=device)

        # One forward pass for the entire batch to get "past_key_values"
        outputs = model(input_ids=prompt_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values

        # We'll accumulate partial losses in a small tensor
        partial_losses = torch.zeros(num_soft_steps + 1, device=device)

        # Loop over each soft step in parallel for the entire batch
        for step_idx in range(num_soft_steps):
            # Get last logits for the batch
            last_logits = outputs.logits[:, -1, :]  # (B, vocab_size)

            # Cross entropy wrt. teacher distribution at this step
            dist_vec = dist_steps[:, step_idx, :]    # (B, vocab_size)
            step_ce_vals = cross_entropy_distribution_batch(last_logits, dist_vec)
            step_loss = step_ce_vals.mean()
            partial_losses[step_idx] = step_loss.detach()

            if str(step_idx) in loss_steps_to_include:
                batch_loss += step_loss

            # Build soft embedding for the entire batch
            if supervision_type == "soft_teacher":
                token_dist_vec = F.softmax(last_logits, dim=-1)  # (B, vocab_size)
                e_soft = token_dist_vec @ embedding_matrix       # (B, n_embd)
            else:
                # "hard_teacher"
                e_soft = dist_vec @ embedding_matrix             # (B, n_embd)

            # Feed that embedding as the next token for all B examples
            out2 = model(
                inputs_embeds=e_soft.unsqueeze(1),  # (B, 1, n_embd)
                past_key_values=past_key_values,
                use_cache=True
            )
            outputs = out2
            past_key_values = out2.past_key_values

        # Final hard step => CE with final_label
        last_logits = outputs.logits[:, -1, :]  # (B, vocab_size)
        final_loss = ce_loss_fn(last_logits, final_labels)
        partial_losses[-1] = final_loss.detach()

        if "h" in loss_steps_to_include:
            batch_loss += final_loss

        # Backprop once for the entire batch
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        steps += 1

    avg_loss = total_loss / max(1, steps)
    return avg_loss, train_loss_tokens / max(1, steps)


def train_soft_steps(model, data_loader, optimizer, device, loss_steps_to_include, supervision_type, seq_length, curriculum_epoch, current_epoch):
    # function to get the last logits from outputs "the last token's logits."
    def get_last_logits(outputs):
        # shape of outputs.logits => (1, seq_len, vocab_size)
        return outputs.logits[:, -1, :]  # (1, vocab_size)

    model.train()
    total_loss = 0.0
    steps = 0

    ce_loss_fn = nn.CrossEntropyLoss()
    embedding_matrix = model.transformer.wte.weight  # shape (vocab_size, n_embd)
    train_loss_tokens = np.zeros((seq_length,))
    
    for batch in data_loader:
        prompt_ids = batch["prompt_ids"].to(device)      # (B, seq_len)
        attention_mask = batch["attention_mask"].to(device)
        dist_steps_list = batch["dist_steps"]            # list of length B, each is 4 distributions
        final_labels = batch["final_labels"]             # list of length B

        batch_size = prompt_ids.size(0)
        batch_loss = torch.tensor(0.0, device=device)
        train_loss_tokens_batch = np.zeros((seq_length,))

        for i in range(batch_size):
            # Feed the entire prompt for sample i
            pi = prompt_ids[i].unsqueeze(0)
            am = attention_mask[i].unsqueeze(0)

            outputs = model(input_ids=pi, attention_mask=am, use_cache=True)
            past_key_values = outputs.past_key_values

            item_loss = torch.tensor(0.0, device=device)

            # For steps 1..4 => distribution-based CE, then feed "soft" embedding
            for count, dist_vec in enumerate(dist_steps_list[i]):
                if count == len(dist_steps_list[i]) - 1:
                    break
                # get last logits (the model's guess for next token)
                last_logits = get_last_logits(outputs).squeeze(0)  # shape (vocab_size,)
                # measure dist-based CE
                step_loss = cross_entropy_distribution(last_logits, dist_vec.to(device))
                train_loss_tokens_batch[count] += step_loss.item()
                
                if str(count) in loss_steps_to_include:  # Select the steps that we want to backprop the loss
                    item_loss += step_loss

                # build "soft" embedding => e_soft = sum_v dist_vec[v]*embedding_matrix[v]
                token_dist_vec = F.softmax(last_logits, dim=-1)
                if supervision_type == "soft_teacher":
                    e_soft = torch.matmul(token_dist_vec, embedding_matrix) # shape (n_embd,)
                elif supervision_type == "hard_teacher":
                    e_soft = torch.matmul(dist_vec.to(device), embedding_matrix) # shape (n_embd,)
                elif supervision_type == "hard_soft_curriculum":
                    if current_epoch < curriculum_epoch:
                        e_soft = torch.matmul(dist_vec.to(device), embedding_matrix) # shape (n_embd,)
                    else:
                        e_soft = torch.matmul(token_dist_vec, embedding_matrix) # shape (n_embd,)

                # feed that embedding as the next token
                out2 = model(
                    inputs_embeds=e_soft.unsqueeze(0).unsqueeze(1),  # (batch=1, seq=1, emb)
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = out2.past_key_values
                outputs = out2  # so next iteration sees the new "last_logits"

            # Final step => Hard sample
            # We'll do actual sampling from the last logits & measure no distribution-based loss.
            last_logits = get_last_logits(outputs).squeeze(0)
            # Hard sample
            probs = F.softmax(last_logits, dim=-1)
            sampled_id = torch.multinomial(probs, num_samples=1).item()

            # If we also have a known label => measure CE
            target_token = torch.tensor([final_labels[i]], dtype=torch.long, device=device)
            final_step_loss = ce_loss_fn(last_logits.unsqueeze(0), target_token)
            train_loss_tokens_batch[-1] += final_step_loss.item()

            if "h" in loss_steps_to_include:  # Include hard (answer) token to loss
                item_loss += final_step_loss

            # Accumulate
            batch_loss += item_loss

        batch_loss = batch_loss / batch_size
        train_loss_tokens_batch = train_loss_tokens_batch / batch_size
        train_loss_tokens += train_loss_tokens_batch

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        steps += 1

    return total_loss / steps, train_loss_tokens / steps


def main(args=None):
    # set the hyperparameters
    MAX_SEQ_LEN = args.max_seq_len
    EMBEDDING_DIM = args.embedding_dim
    DIGIT_RANGE = args.digit_range
    BATCH_SIZE = args.batch_size
    SEQ_LENGTH = args.seq_length
    LOSS_STEPS = args.loss_steps
    SPLIT_METHOD = args.split_method
    NUM_EPOCHS = args.num_epochs
    OUTPUT_DIR = args.output_dir
    SUPERVISION = args.supervision
    CURRICULUM_EPOCH = args.curriculum_epoch
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    DIST_STRATEGY = args.dist_strategy
    NUM_BEAMS = args.num_beams

    print(f"Max Seq Len: {MAX_SEQ_LEN}, "
          f"Embedding Dim: {EMBEDDING_DIM}, "
          f"Digit Range: {DIGIT_RANGE}, "
          f"Batch Size: {BATCH_SIZE}, "
          f"Seq Length: {SEQ_LENGTH}, "
          f"Loss Steps: {LOSS_STEPS}, "
          f"Split Method: {SPLIT_METHOD}, "
          f"Num Epochs: {NUM_EPOCHS}, "
          f"Output Dir: {OUTPUT_DIR}, "
          f"Supervision: {SUPERVISION}, "
          f"Curriculum Epoch: {CURRICULUM_EPOCH}, "
          f"Num Layers: {NUM_LAYERS}, "
          f"Num Heads: {NUM_HEADS}",
          f"Dist Strategy: {DIST_STRATEGY}, "
          f"Num Beams: {NUM_BEAMS}")
    
    # Build vocab & model
    vocab, token2id, id2token = generate_vocab()
    vocab_size = len(vocab)
    print("Vocab size =", vocab_size)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_SEQ_LEN,
        n_embd=EMBEDDING_DIM,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS
    )
    model = GPT2LMHeadModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print device
    print("Using device:", device)
    model.to(device)

    # Create the dataset
    dataset = FourDigitsSoftDataset(token2id=token2id, digit_range=DIGIT_RANGE, seq_length=SEQ_LENGTH, dist_strategy=DIST_STRATEGY, num_beams=NUM_BEAMS)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if SPLIT_METHOD == "sequential":
        train_data = dataset[:train_size]
        val_data = dataset[train_size:]
    elif SPLIT_METHOD == "random":
        train_data, val_data = dataset_random_split(dataset, train_ratio=0.8)
    else:  # "random_permutation"
        train_data, val_data = permutation_train_val_split_continuous(
            dataset=dataset,
            id2token=id2token,
            seq_length=SEQ_LENGTH,
            train_ratio=0.8
        )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_losses_tokens = []
    val_losses_tokens = []
    for epoch in range(NUM_EPOCHS):
        if SUPERVISION == "hard_soft_curriculum" and epoch == CURRICULUM_EPOCH:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # reset optimizer for curriculum learning
        #train_loss, train_loss_tokens = train_soft_steps(model, train_loader, optimizer, device, LOSS_STEPS, SUPERVISION, SEQ_LENGTH, CURRICULUM_EPOCH, epoch)
        train_loss, train_loss_tokens = train_soft_steps_batch(
            model,
            train_loader,
            optimizer,
            device,
            LOSS_STEPS,
            SUPERVISION,
            SEQ_LENGTH,
            num_soft_steps=SEQ_LENGTH - 1
        )
        val_loss, val_acc, val_loss_tokens = eval_soft_steps_acc(model, val_loader, device, SEQ_LENGTH)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        train_losses_tokens.append(train_loss_tokens)
        val_losses_tokens.append(val_loss_tokens)
        print(f"Epoch {epoch + 1} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_acc:.2%} | "
              f"Val Loss Tokens: " + "-".join([f"{val_loss_tokens[i]:.4f}" for i in range(len(val_loss_tokens))]) + " | "
              f"Train Loss Tokens: " + "-".join([f"{train_loss_tokens[i]:.4f}" for i in range(len(train_loss_tokens))]) + " | ")

    # Plot losses & accuracies
    epochs_range = range(1, NUM_EPOCHS + 1)
    file_name = (
        f"Digit{DIGIT_RANGE.start}-{DIGIT_RANGE.stop}" + "_seq" + str(SEQ_LENGTH) + "_emb" +
        str(EMBEDDING_DIM) + "_steps" + "".join(LOSS_STEPS) + "_supervision" + SUPERVISION +
        "_split" + SPLIT_METHOD + "_batch" + str(BATCH_SIZE) + "_epochs" + str(NUM_EPOCHS)
    )
    plot_losses(epochs_range, train_losses, val_losses, model_type=f"continuous_{file_name}", output_dir = OUTPUT_DIR)
    plot_accuracies(epochs_range, val_accuracies, model_type=f"continuous_{file_name}", output_dir = OUTPUT_DIR)
    # for i, loss in enumerate(train_losses_tokens):
        # plot_losses(epochs_range, loss, val_losses_tokens[i], model_type=f"continuous_{file_name}_tokens_{i}", output_dir = OUTPUT_DIR)

    # save the model
    torch.save(model.state_dict(), f"models/{OUTPUT_DIR}continuous_model_{file_name}.pt")

    # load the model
    #model.load_state_dict(torch.load('countdown_continuous_model.pt'))
    #model.to(device)

    # Evaluate on the validation set
    #val_loss, val_acc, _ = eval_soft_steps_acc(model, val_loader, device, LOSS_STEPS)
    #print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")


@torch.no_grad()
def eval_soft_steps_acc(model, data_loader, device, seq_length):
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

    ce_loss_fn = nn.CrossEntropyLoss()
    embedding_matrix = model.transformer.wte.weight
    val_loss_tokens = np.zeros((seq_length,))

    def get_last_logits(o):
        return o.logits[:, -1, :]

    for batch in data_loader:
        prompt_ids = batch["prompt_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        dist_steps_list = batch["dist_steps"]
        final_labels = batch["final_labels"]
        batch_size = prompt_ids.size(0)

        batch_loss = 0.0  # python float is fine; each step_loss will track grad separately
        val_loss_tokens_batch = np.zeros((seq_length,))

        for i in range(batch_size):
            # Forward the prompt in discrete form
            pi = prompt_ids[i].unsqueeze(0)
            am = attention_mask[i].unsqueeze(0)

            outputs = model(pi, attention_mask=am, use_cache=True)
            pkv = outputs.past_key_values

            item_loss = 0.0

            # Distribution steps (1..4)
            for count, dist_vec in enumerate(dist_steps_list[i]):
                if count == len(dist_steps_list[i]) - 1:
                    break
                last_logits = get_last_logits(outputs).squeeze(0)  # shape (vocab_size,)
                step_loss = cross_entropy_distribution(last_logits, dist_vec.to(device))
                item_loss += step_loss.item()
                val_loss_tokens_batch[count] += step_loss.item()

                # build "soft" embedding => e_soft = sum_v dist_vec[v]*embedding_matrix[v]
                token_dist_vec = F.softmax(last_logits, dim=-1)
                e_soft = torch.matmul(token_dist_vec, embedding_matrix)  # shape (n_embd,)
                out2 = model(
                    inputs_embeds=e_soft.unsqueeze(0).unsqueeze(1),  # (1,1,n_embd)
                    past_key_values=pkv,
                    use_cache=True
                )
                pkv = out2.past_key_values
                outputs = out2

            # final step => measure CE to final_label, also do discrete "prediction" for accuracy
            last_logits = get_last_logits(outputs).squeeze(0)  # (vocab_size,)
            final_label_id = final_labels[i]

            # we can do cross-entropy with the final label
            final_loss = ce_loss_fn(last_logits.unsqueeze(0), torch.tensor([final_label_id], device=device))
            item_loss += final_loss.item()
            val_loss_tokens_batch[-1] += final_loss.item()

            # for accuracy, pick argmax
            predicted_id = last_logits.argmax(dim=-1).item()
            if predicted_id == final_label_id:
                total_correct += 1

            batch_loss += item_loss

        # average over batch
        batch_loss /= batch_size
        val_loss_tokens_batch /= batch_size
        val_loss_tokens += val_loss_tokens_batch
        total_loss += batch_loss
        total_samples += batch_size
        steps += 1

    avg_loss = total_loss / steps
    val_loss_tokens /= steps
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, val_loss_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a continuous generation model.")
    parser.add_argument("--max_seq_len", type=int, default=13, help="Maximum sequence length.")
    parser.add_argument("--embedding_dim", type=int, default=24, help="Embedding dimension.")
    parser.add_argument("--digit_range", type=int, nargs=2, default=[1, 10], help="Range of digits (start, end).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--seq_length", type=int, default=4, help="Length of digit sequences.")
    parser.add_argument("--loss_steps", type=str, nargs="+", default=["0", "1", "2", "3", "h"],
                        help="Steps to backpropagate loss.")
    parser.add_argument("--split_method", type=str, default="random_permutation",
                        choices=["random", "sequential", "random_permutation"],
                        help="Method to split dataset.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for saving models and plots.")
    parser.add_argument("--supervision", type=str, default="hard_teacher",
                        choices=["hard_teacher", "soft_teacher", "hard_soft_curriculum"],
                        help="Supervision type.")
    parser.add_argument("--curriculum_epoch", type=int, default=0,
                        help="curriculum epoch for the hard_soft_curriculum supervision type.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads in the model.")
    parser.add_argument("--dist_strategy", type=str, default="uniform",
                    choices=["uniform", "beam", "beam_minabs"],
                    help="How to build per-step target distributions.")
    parser.add_argument("--num_beams", type=int, default=8,
                        help="K beams used when dist_strategy=beam.")
    parser.add_argument("--seed", type=int, default=42, help="Random-seed for reproducibility.")
    
    args = parser.parse_args()
    args.digit_range = range(args.digit_range[0], args.digit_range[1])  # Convert to range object

    set_seed(args.seed)
    main(args)