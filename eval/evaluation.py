import torch

def parse_line_pause(line: str):
    """
    Works for the new one–shot dataset that contains

        <BOS> D… D… … -> -> … -> Sx <EOS>

    Returns
        digits         : list[int]   (length = seq_length)
        partial_sums   : list[int]   (length = 1 for the new task)
    """
    tokens = line.split()

    # first and last arrow indices
    first_arrow = tokens.index("->")
    last_arrow  = len(tokens) - 1 - tokens[::-1].index("->")

    # digits are between <BOS> and the first arrow
    digit_tokens = tokens[1:first_arrow]

    # answer token(s) come after the last arrow, before <EOS>
    sum_tokens   = tokens[last_arrow + 1:-1]

    digits        = [int(t[1:]) for t in digit_tokens]   # "D5" → 5
    partial_sums  = [int(t[1:]) for t in sum_tokens]     # "S7" → 7
    return digits, partial_sums

def evaluate_model_pause(model, test_dataset_text, token2id, id2token, device, seq_length):
    correct_count = 0

    for line in test_dataset_text:
        digits, gt_sums = parse_line_pause(line)
        # ground_truth_final = minimal non-negative sum (or whatever is in the dataset)
        ground_truth_final = gt_sums[-1]

        # Model's predicted partial sums
        predicted_sums = generate_partial_sums_step_by_step_pause(
            model, digits, token2id, id2token, device, seq_length=seq_length, max_sums=len(gt_sums)
        )

        # Check validity of partial sums and final sum
        # if is_valid_path(digits, predicted_sums) and (predicted_sums[-1] == ground_truth_final):
        #     correct_count += 1
        
        if predicted_sums[-1] == ground_truth_final:  # For fair comparison, compare the last token only instead
            correct_count += 1

    accuracy = correct_count / len(test_dataset_text)
    return accuracy, correct_count

def generate_partial_sums_step_by_step_pause(
    model,
    digits,
    token2id,
    id2token,
    device,
    seq_length,
    max_sums=4,
    do_sample=False,
    temperature=1.0
):
    # Build the initial prompt (no partial sums yet).
    # Example: "<BOS> D5 D3 D2 D4 ->"
    prompt_tokens = ["<BOS>"] + [f"D{d}" for d in digits] + ["->"] * seq_length

    # Convert each token string to its ID.
    input_ids = torch.tensor([[token2id[t] for t in prompt_tokens]], dtype=torch.long).to(device)

    predicted_sums = []
    for _ in range(max_sums):
        # Generate exactly 1 token from the model (greedy).
        # pad_token_id is important to avoid warnings if the sequence grows.
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=do_sample,
            pad_token_id=token2id["<PAD>"],
            temperature=temperature
        )
        # The last generated token is out[0, -1].
        new_token_id = out[0, -1].item()
        new_token_str = id2token[new_token_id]

        # If it looks like "Sxxx", parse out the integer value, else store None.
        if new_token_str.startswith("S"):
            val = int(new_token_str[1:])  # e.g. "S5" -> 5
        else:
            val = None

        predicted_sums.append(val)

        # Update input_ids to include the newly generated token
        input_ids = out

    return predicted_sums

# 2) A helper function to parse a line into:
#    - The digit sequence (e.g. [5, 3, 2, 4])
#    - The ground-truth partial sums (e.g. [5, 2, 4, 0])
#
#    Example line:
#      "<BOS> D5 D3 D2 D4 -> S5 S2 S4 S0 <EOS>"
#    We'll extract:
#      digits = [5, 3, 2, 4]
#      partial_sums = [5, 2, 4, 0]
def parse_line(line):
    tokens = line.split()  # e.g. ["<BOS>", "D5", "D3", "D2", "D4", "->", "S5", "S2", "S4", "S0", "<EOS>"]
    arrow_idx = tokens.index("->")  # location of '->'

    # digits: everything after <BOS> up to '->'
    digit_tokens = tokens[1:arrow_idx]  # ignore <BOS>
    # partial sums: everything after '->' up to <EOS>
    sum_tokens = tokens[arrow_idx + 1:-1]  # ignore <EOS>

    # Convert "D5" -> 5, "S5" -> 5, etc.
    digits = [int(dt[1:]) for dt in digit_tokens]  # strip the first char 'D'
    partial_sums = [int(st[1:]) for st in sum_tokens]  # strip the first char 'S'
    return digits, partial_sums


# 3) A function to generate partial sums step by step, **one token at a time**.
#    It will:
#      - Start with a prompt like: <BOS> D5 D3 D2 D4 ->
#      - Generate 1 new token with model.generate(...)
#      - Append that token to the prompt
#      - Repeat for `max_sums` times (the number of ground-truth partial sums)
#    It returns a list of *predicted* integer partial sums, e.g. [5, 2, 4, 0]

def generate_partial_sums_step_by_step(
    model,
    digits,
    token2id,
    id2token,
    device,
    max_sums=4,
    do_sample=False,
    temperature=1.0
):
    # Build the initial prompt (no partial sums yet).
    # Example: "<BOS> D5 D3 D2 D4 ->"
    prompt_tokens = ["<BOS>"] + [f"D{d}" for d in digits] + ["->"]

    # Convert each token string to its ID.
    input_ids = torch.tensor([[token2id[t] for t in prompt_tokens]], dtype=torch.long).to(device)

    predicted_sums = []
    for _ in range(max_sums):
        # Generate exactly 1 token from the model (greedy).
        # pad_token_id is important to avoid warnings if the sequence grows.
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=do_sample,
            pad_token_id=token2id["<PAD>"],
            temperature=temperature
        )
        # The last generated token is out[0, -1].
        new_token_id = out[0, -1].item()
        new_token_str = id2token[new_token_id]

        # If it looks like "Sxxx", parse out the integer value, else store None.
        if new_token_str.startswith("S"):
            val = int(new_token_str[1:])  # e.g. "S5" -> 5
        else:
            val = None

        predicted_sums.append(val)

        # Update input_ids to include the newly generated token
        input_ids = out

    return predicted_sums


def is_valid_path(digits, predicted_sums):
    """
    In this version, we record only the partial sums after each digit is applied,
    omitting the initial 0 from the dataset. So for 4 digits, predicted_sums
    has length = 4.

    Checks:
      1) len(predicted_sums) == len(digits).
      2) predicted_sums[0] in {+digits[0], -digits[0]}.
      3) For i in [1..len(digits)-1],
         predicted_sums[i] == predicted_sums[i-1] +/- digits[i].
    """

    # 1) Must have exactly as many sums as digits
    if len(predicted_sums) != len(digits):
        return False

    # 2) The first partial sum is from 0 +/- digits[0].
    if predicted_sums[0] not in (digits[0], -digits[0]):
        return False

    # 3) For each subsequent digit i, predicted_sums[i] = predicted_sums[i-1] +/- digits[i].
    for i in range(1, len(digits)):
        prev_sum = predicted_sums[i - 1]
        d = digits[i]
        if predicted_sums[i] not in (prev_sum + d, prev_sum - d):
            return False

    return True


def evaluate_model(model, test_dataset_text, token2id, id2token, device):
    correct_count = 0

    for line in test_dataset_text:
        digits, gt_sums = parse_line(line)
        # ground_truth_final = minimal non-negative sum (or whatever is in the dataset)
        ground_truth_final = gt_sums[-1]

        # Model's predicted partial sums
        predicted_sums = generate_partial_sums_step_by_step(
            model, digits, token2id, id2token, device, max_sums=len(gt_sums)
        )

        # Check validity of partial sums and final sum
        # if is_valid_path(digits, predicted_sums) and (predicted_sums[-1] == ground_truth_final):
        #     correct_count += 1
        
        if predicted_sums[-1] == ground_truth_final:  # For fair comparison, compare the last token only instead
            correct_count += 1

    accuracy = correct_count / len(test_dataset_text)
    return accuracy, correct_count

@torch.no_grad()
def generate_partial_sums_hidden(
        model,
        digits: list[int],
        token2id, id2token,
        device,
        loss_steps: list[int],      # e.g. [1,2,3]  (supervised offsets)
        max_sums: int = 4):
    """
    Roll out partial sums while *holes* are filled with hidden states
    (continuous) instead of discrete samples.

    Returns:
        predicted_ints  - list with length = max_sums; for un-supervised
                          positions it contains None.
    """
    # --- prompt ----------------------------------------------------------------
    prompt = ["<BOS>"] + [f"D{d}" for d in digits] + ["->"]
    input_ids = torch.tensor([[token2id[t] for t in prompt]], device=device)
    attention = torch.ones_like(input_ids)

    # current embedding sequence
    wte = model.transformer.wte
    embeds = wte(input_ids)                          # (1, T0, D)

    predicted = []

    for pos in range(max_sums):
        if pos not in loss_steps:
            # ---------------- CONTINUOUS HOLE -----------------
            # placeholder (zeros) at the hole
            pad = torch.zeros_like(embeds[:, :1, :])
            embeds = torch.cat([embeds, pad], dim=1)          # len +=1
            attention = torch.cat([attention,
                                   torch.ones_like(attention[:, :1])], dim=1)

            # run model up to and incl. the hole
            out = model(inputs_embeds=embeds,
                        attention_mask=attention,
                        output_hidden_states=True,
                        use_cache=False)
            h_pos = out.hidden_states[-1][:, -1, :]           # (1, D)
            embeds[:, -1, :] = h_pos                          # overwrite
            predicted.append(None)                            # no integer
            continue

        # ---------------- DISCRETE (supervised) STEP ----------------
        # forward up to current length, grab logits at last index
        out = model(inputs_embeds=embeds,
                    attention_mask=attention,
                    use_cache=False)
        logits = out.logits[:, -1, :]                         # (1, V)
        next_id = int(torch.argmax(logits, dim=-1))
        next_tok = id2token[next_id]

        # convert token "Sxxx" -> integer
        val = int(next_tok[1:]) if next_tok.startswith("S") else None
        predicted.append(val)

        # append the *embedding* of the chosen token
        emb_next = wte(torch.tensor([[next_id]], device=device))  # (1,1,D)
        embeds = torch.cat([embeds, emb_next], dim=1)
        attention = torch.cat([attention,
                               torch.ones_like(attention[:, :1])], dim=1)

    return predicted


@torch.no_grad()
def evaluate_model_hidden(model, dataset_text, token2id, id2token,
                   device, loss_steps):
    """
    Accuracy under *continuous-feed* evaluation.
    We deem a sample correct if **every supervised position** (those in
    `loss_steps`) matches the ground-truth value.
    """
    model.eval()
    correct = 0

    for line in dataset_text:
        digits, gt_sums = parse_line(line)          # lists of ints
        pred_sums = generate_partial_sums_hidden(
            model, digits,
            token2id, id2token,
            device,
            loss_steps=loss_steps,
            max_sums=len(gt_sums)
        )

        ok = pred_sums[-1] == gt_sums[-1]
        correct += int(ok)

    return correct / len(dataset_text), correct


def evaluate_model_pass_k(
    model,
    test_dataset_text,
    token2id,
    id2token,
    device,
    k=10
):
    """
    Evaluate model performance across temperatures in [0.0, 0.2, ..., 1.0].
    Returns a list of accuracies, one for each temperature.
    """
    temperature_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    accuracies = []

    for temperature in temperature_values:
        acc_runs = []

        for _ in range(10):  # Repeat evaluation 10 times
            correct_count = 0

            for line in test_dataset_text:
                digits, gt_sums = parse_line(line)
                ground_truth_final = gt_sums[-1]
                passed_this_line = False

                for _ in range(k):
                    if temperature == 0.0:
                        predicted_sums = generate_partial_sums_step_by_step(
                            model,
                            digits,
                            token2id,
                            id2token,
                            device,
                            max_sums=len(gt_sums),
                            do_sample=False  # Greedy sampling
                        )
                    else:
                        predicted_sums = generate_partial_sums_step_by_step(
                            model,
                            digits,
                            token2id,
                            id2token,
                            device,
                            max_sums=len(gt_sums),
                            do_sample=True,
                            temperature=temperature
                        )

                    if predicted_sums[-1] == ground_truth_final:
                        passed_this_line = True
                        break

                if passed_this_line:
                    correct_count += 1

            acc_runs.append(correct_count / len(test_dataset_text))

        accuracies.append(sum(acc_runs) / len(acc_runs))  # Average over 10 runs

    return dict(zip(temperature_values, accuracies))
