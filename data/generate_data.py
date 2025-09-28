import itertools
import random
from collections import defaultdict, Counter
from torch.utils.data import Subset
import matplotlib.pyplot as plt


def generate_vocab():
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "->", "+", "-"]

    # Digits 0..9
    digit_tokens = [f"D{i}" for i in range(10)]

    # Partial sums from -36..36 (optional range—change as needed)
    sum_tokens = [f"S{i}" for i in range(-36, 37)]

    # Combine them into a single list (vocab)
    vocab = special_tokens + digit_tokens + sum_tokens

    # Create mapping from token to ID and back
    token2id = {token: idx for idx, token in enumerate(vocab)}
    id2token = {idx: token for token, idx in token2id.items()}

    return vocab, token2id, id2token


###############################################################################
# 2) DATA GENERATION FOR THE "MINIMUM NON-NEGATIVE SUM" GAME
###############################################################################
def generate_text_dataset(digit_range=range(1, 6), seq_length=4, mod=1):
    """
    For each 4-digit sequence in [1..5], we consider sign patterns of length 4,
    starting from 0. That is:
       partial_sum = 0 (+/-) seq[0] (+/-) seq[1] (+/-) seq[2] (+/-) seq[3].

    Among all 16 possibilities, we pick the sign pattern whose final sum is
    the smallest non-negative integer. If no sign pattern yields a non-negative
    result, we skip that sequence.

    We then record the digits and the *4* resulting partial sums in a line like:
       <BOS> D1 D1 D1 D4 -> S-1 S-2 S-3 S1 <EOS>
    (omitting the initial S0 in the text).
    """
    dataset_text = []

    for seq in itertools.product(digit_range, repeat=seq_length):
        best_final_sum = None
        best_partial_sums = None

        # Try all sign patterns for the 4 digits (2^4 = 16),
        # starting partial_sum = 0, then apply +/- for each digit in seq.
        for signs in itertools.product(["+", "-"], repeat=seq_length):
            partial_sum = 0
            partial_sums = [partial_sum]  # [0, x, y, ...]

            for i in range(seq_length):
                if signs[i] == "+":
                    partial_sum += seq[i]
                else:
                    partial_sum -= seq[i]
                partial_sums.append(partial_sum)

            # Check if final sum is >= 0
            if partial_sum >= 0:
                # If this is the first non-negative final sum found
                # or if it's smaller than our current best
                if best_final_sum is None or partial_sum < best_final_sum:
                    best_final_sum = partial_sum
                    best_partial_sums = partial_sums

        # If we found at least one sign pattern that yields a non-negative sum,
        # record the best partial sums in textual form.
        if best_final_sum is not None:
            digit_seq_tokens = [f"D{d}" for d in seq]
            # omit the initial partial sum (index 0) so only 4 sums remain
            # best_partial_sums has length 5 => skip best_partial_sums[0]
            sum_seq_tokens = [f"S{ps}" for ps in best_partial_sums[1:]]

            line_tokens = ["<BOS>"] + digit_seq_tokens + ["->"] + sum_seq_tokens + ["<EOS>"]
            line_text = " ".join(line_tokens)
            dataset_text.append(line_text)
    
    new_dataset_text = []
    for example in dataset_text:
        tokens = example.split()
        arrow_idx = tokens.index("->")
        new_tokens = []
        for i in range(arrow_idx):
            new_tokens.append(tokens[i])
        new_tokens.append("->")
        for i in range(arrow_idx + 1, len(tokens) - 1):
            # Include tokens if they match the mod pattern
            # and always include the last partial-sum token.
            if i % mod == ((len(tokens) - 2) % mod):
                new_tokens.append(tokens[i])
        new_tokens.append("<EOS>")
        new_example = " ".join(new_tokens)
        new_dataset_text.append(new_example)
    dataset_text = new_dataset_text

    return dataset_text

def generate_text_dataset_one_shot(digit_range=range(1, 6), seq_length=4):
    """
    Produces lines such as
        <BOS> D5 D3 D2 D4 -> -> -> -> S1 <EOS>
    i.e. *seq_length* arrows followed by ONE answer token.
    """
    dataset_text = []

    for seq in itertools.product(digit_range, repeat=seq_length):
        best_final = None
        for signs in itertools.product(["+", "-"], repeat=seq_length):
            s = 0
            for d, sign in zip(seq, signs):
                s = s + d if sign == "+" else s - d
            if s >= 0 and (best_final is None or s < best_final):
                best_final = s

        if best_final is None:          # no non‑negative path
            continue

        # ---------- build the textual example -------------------------
        digit_tokens  = [f"D{d}" for d in seq]
        arrows        = ["->"] * seq_length          # N arrows
        answer_token  = f"S{best_final}"

        line = " ".join(["<BOS>"] + digit_tokens + arrows +
                        [answer_token, "<EOS>"])
        dataset_text.append(line)

    return dataset_text


def dataset_random_split(dataset, train_ratio=0.8):
    """
    Splits 'dataset' into train and val subsets using Python's random module.
    This respects 'random.seed(...)' rather than 'torch.manual_seed(...)'.
    """
    # Build a list of all indices
    indices = list(range(len(dataset)))
    # Shuffle in place using Python's random
    random.shuffle(indices)

    # Compute split point
    train_size = int(train_ratio * len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Wrap them into Subset objects
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    return train_data, val_data


def permutation_train_val_split(dataset_text, train_ratio=0.8):
    """
    This function takes a dataset of text lines, groups them by the sorted digits
    contained in each line, shuffles the groups, and splits them into training
    and validation sets based on a specified ratio.
    """
    # Group lines by sorted digits
    groups = defaultdict(list)
    for line in dataset_text:
        tokens = line.split()
        arrow_idx = tokens.index("->")
        digit_tokens = tokens[1:arrow_idx]  # ignoring <BOS>
        # parse digits from lines like "D5"
        digits = tuple(sorted(int(dt[1:]) for dt in digit_tokens))
        groups[digits].append(line)

    # Shuffle the group keys
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    # Split group keys 80/20
    split_idx = int(train_ratio * len(group_keys))
    train_keys = group_keys[:split_idx]
    val_keys = group_keys[split_idx:]

    # Gather lines
    train_lines = []
    val_lines = []
    for k in train_keys:
        train_lines.extend(groups[k])
    for k in val_keys:
        val_lines.extend(groups[k])

    return train_lines, val_lines


def permutation_train_val_split_continuous(dataset, id2token, seq_length, train_ratio=0.8):
    """
    Ensures that all permutations of the same digit sequence go entirely into
    train or val. We do this by grouping examples based on sorted digits in
    their prompt, then performing a group-level random split.

    Args:
      dataset: A PyTorch Dataset whose __getitem__ returns a dict like:
               {
                 "prompt_ids": Tensor,  # shape [prompt_len]
                 "dist_steps": ...,
                 "final_hard_label": ...
               }
      id2token: mapping from token ID to string token, e.g. "D5"
      seq_length: how many 'D' tokens in each prompt (e.g. 4)
      train_ratio: fraction of groups to go to train (e.g. 0.8 => 80% train)

    Returns:
      train_data, val_data: Subset objects pointing to the train/val samples.
    """
    # Build groups: canonical sorted digits -> list of example indices
    groups = defaultdict(list)

    for idx in range(len(dataset)):
        ex = dataset[idx]     # e.g. {"prompt_ids": ..., "dist_steps":..., "final_hard_label":...}
        prompt_ids = ex["prompt_ids"]

        # The digits are typically in the tokens from index 1..(1+seq_length) ignoring <BOS>.
        digit_ids = prompt_ids[1: 1 + seq_length].tolist()
        digit_strs = [id2token[d] for d in digit_ids]  # e.g. ["D5", "D1", ...]
        digits = [int(s[1:]) for s in digit_strs]      # strip off the "D", e.g. [5, 1, ...]
        canon_digits = tuple(sorted(digits))           # canonical form, e.g. (1,5,5,4)

        groups[canon_digits].append(idx)

    # Shuffle group keys
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    # Split group keys based on train_ratio
    split_idx = int(train_ratio * len(group_keys))
    train_keys = group_keys[:split_idx]
    val_keys = group_keys[split_idx:]

    # Gather indices
    train_indices = []
    val_indices = []
    for k in train_keys:
        train_indices.extend(groups[k])
    for k in val_keys:
        val_indices.extend(groups[k])

    # Create Subset objects for train and val
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


def compute_max_abs_value(dataset_text):
    """
    Return the largest |n| that appears in any token of any example line.
    Scans both digit tokens (D#) and partial‑sum tokens (S#).

    Returns:
        Maximum absolute integer encountered.
    """
    max_abs_val = 0
    for line in dataset_text:
        for tok in line.split():
            if tok.startswith(("D", "S")):
                try:
                    val = abs(int(tok[1:]))      # strip leading 'D' or 'S'
                    max_abs_val = max(max_abs_val, val)
                except ValueError:
                    pass                         # skip malformed tokens
    return max_abs_val


def plot_final_label_distribution(
        dataset_text,
        digit_range,
        seq_length,
        title="Distribution of Final Labels"
):
    """Bar plot of how often each final‐sum token (e.g. S0, S1, …) appears."""
    labels = [int(line.split()[-2][1:])   # grab the ‘S#’ token before <EOS>
              for line in dataset_text]
    counts = Counter(labels)
    xs, ys = sorted(counts), [counts[x] for x in sorted(counts)]
    title = title + f", {digit_range.start}-{digit_range.stop - 1}, Seq Length {seq_length}"

    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys)
    plt.xlabel("Final label (sum)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"figures/{title}.pdf", format='pdf', bbox_inches='tight')
    plt.show()
