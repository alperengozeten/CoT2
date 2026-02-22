import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from data.generate_data import generate_vocab, generate_text_dataset, permutation_train_val_split
from eval.evaluation import evaluate_model_hidden, evaluate_model_pass_k 
from utils import plot_losses, plot_accuracies
import argparse
import torch.nn as nn
import numpy as np

# seed everything for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenization
def tokenize_line(line, token2id_map):
    tokens = line.strip().split()
    return [token2id_map[t] for t in tokens]


class MathExpressionDataset(Dataset):
    def __init__(self, tokenized_samples, max_len, token2id):
        self.samples = tokenized_samples
        self.max_len = max_len
        self.token2id = token2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids = self.samples[idx]
        # Truncate if longer than max_len
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]

        # Create attention mask
        attention_mask = [1] * len(token_ids)

        # Pad if shorter
        while len(token_ids) < self.max_len:
            token_ids.append(self.token2id["<PAD>"])
            attention_mask.append(0)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # For a causal LM, labels are the same as input_ids (the shift happens internally)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


def get_token_loss(outputs, labels, seq_length):
    # Minimal additional code to get token-level losses:
    logits = outputs.logits
    # Shift the logits and labels by one for causal LM:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1)
    )
    # Reshape into (batch_size, sequence_length - 1) for easy interpretation
    per_token_loss = per_token_loss.view(shifted_labels.size())
    # the output length with mod is ceil of seq_length / mod
    return per_token_loss[:, seq_length+1:2*seq_length+1].sum(dim=0).cpu().detach().numpy()


def mask_labels_for_loss(
    input_ids: torch.Tensor,  # (B, S)
    attention_mask: torch.Tensor,  # (B, S)
    arrow_id: int,
    output_offsets: list[int] | None = None,
) -> torch.Tensor:
    """
    Return a label tensor where ONLY tokens after the first "->" (arrow)
    are kept. All earlier tokens and all padding positions are replaced
    with -100 so they are ignored by CrossEntropyLoss.

    If `output_offsets` is given, keep only those relative positions
    (0-based) in the output.  Example: offsets=[0,2] keeps the 1st and 3rd
    token after the arrow.
    """
    labels = input_ids.clone()
    B, S = labels.shape
    device = labels.device

    # find the arrow index for every example
    arrow_pos = (input_ids == arrow_id).int().argmax(dim=1)  # (B,)

    # build a mask of tokens we want to keep
    pos_ids = torch.arange(S, device=device).expand(B, S)  # (B, S)
    keep = pos_ids > arrow_pos.unsqueeze(1)  # after arrow
    keep &= attention_mask.bool()  # ignore padding

    if output_offsets is not None:
        # turn offsets (same for every sample) into absolute indices
        offsets = torch.tensor(output_offsets, device=device)  # (K,)
        abs_pos = arrow_pos.unsqueeze(1) + 1 + offsets  # (B, K)
        keep_specific = torch.zeros_like(keep)
        keep_specific.scatter_(1, abs_pos, True)
        keep = keep_specific & attention_mask.bool()

    # mask everything else with -100
    labels[~keep] = -100
    return labels


@torch.no_grad()
def autoreg_fill(model, input_ids, mask_offsets,
                 pad_id, max_step_len=1, do_sample=False, temperature=1.0):
    """
    Given a batch of prompts `input_ids` (B, S) that already contain the arrow,
    generate tokens at absolute positions `mask_offsets` (same offsets for
    every sample in the batch) and splice them back in. Returns a new tensor
    of shape (B, S) with the filled-in tokens.
    """
    filled = input_ids.clone()

    # We assume mask_offsets are in ascending order.
    for pos in mask_offsets:
        # Build left context up to <pos> and let generate() produce ONE token.
        # pad_token_id suppresses warnings when sequence grows.
        out = model.generate(
            filled[:, :pos],             # prompt up to the hole
            max_new_tokens=max_step_len,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pad_id
        )
        new_tok = out[:, -1]            # (B,)
        filled[:, pos] = new_tok        # in‑place splice
    return filled


def soft_autoreg_fill(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,               # (B, S) 
    attention_mask: torch.Tensor,          # (B, S)
    hole_offsets: list[int],               # absolute positions to fill
) -> torch.Tensor:
    """
    Returns an `inputs_embeds` tensor (B, S, D) where every hole position
    contains the hidden output. Earlier holes are already filled
    when computing later ones, so information can flow forward.
    """
    wte = model.transformer.wte           # token‑>embed matrix (V, D)
    emb_pad = wte(input_ids)                 # (B, S, D)  ordinary embeddings
    embeds = emb_pad.clone()                # will be edited in‑place

    for pos in hole_offsets:
        dummy = torch.zeros_like(embeds[:, :1, :])
        test_embeds = torch.cat([embeds[:, :pos], dummy], dim=1)

        out = model(inputs_embeds=test_embeds,
                    attention_mask=attention_mask[:, :pos+1],
                    use_cache=False,
                    output_hidden_states=True)

        embeds[:, pos, :] = out.hidden_states[-1][:, pos, :]   # hidden for the hole itself
    return embeds


def main(args=None):
    # set the hyperparameters
    MAX_SEQ_LEN = args.max_seq_len
    EMBEDDING_DIM = args.embedding_dim
    DIGIT_RANGE = args.digit_range
    BATCH_SIZE = args.batch_size
    SEQ_LENGTH = args.seq_length
    MOD = 1
    SPLIT_METHOD = args.split_method
    NUM_EPOCHS = args.num_epochs
    OUTPUT_DIR = args.output_dir
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    CURRICULUM = [list(range(i, SEQ_LENGTH)) for i in range(SEQ_LENGTH)]

    print(f"Max Seq Len: {MAX_SEQ_LEN}, "
          f"Embedding Dim: {EMBEDDING_DIM}, "
          f"Digit Range: {DIGIT_RANGE}, "
          f"Batch Size: {BATCH_SIZE}, "
          f"Seq Length: {SEQ_LENGTH}, "
          f"MOD: {MOD}, "
          f"Split Method: {SPLIT_METHOD}, "
          f"Num Epochs: {NUM_EPOCHS}, "
          f"Output Dir: {OUTPUT_DIR}, "
          f"Num Layers: {NUM_LAYERS}, "
          f"Num Heads: {NUM_HEADS}",
          f"Curriculum: {CURRICULUM}")

    # Generate vocab, dataset
    vocab, token2id, id2token = generate_vocab()
    dataset_text = generate_text_dataset(DIGIT_RANGE, SEQ_LENGTH, MOD)
    vocab_size = len(vocab)

    print(f"Vocab size = {vocab_size}")
    print(f"Number of valid sequences in dataset: {len(dataset_text)}")
    print("Sample line:", dataset_text[0])

    # Form the train/val based on "sequential", "random", or "random_permutation"
    if SPLIT_METHOD == "random":  # if random split, shuffle
        random.shuffle(dataset_text)
        tokenized_dataset = [tokenize_line(line, token2id) for line in dataset_text]
        split_idx = int(0.8 * len(tokenized_dataset))
        train_data = tokenized_dataset[:split_idx]
        val_data = tokenized_dataset[split_idx:]
        val_lines = dataset_text[split_idx:]

    elif SPLIT_METHOD == "random_permutation":
        train_lines, val_lines = permutation_train_val_split(dataset_text, train_ratio=0.8)
        train_data = [tokenize_line(line, token2id) for line in train_lines]
        val_data = [tokenize_line(line, token2id) for line in val_lines]
        random.shuffle(train_data)
        random.shuffle(val_data)

    else:  # sequential split
        tokenized_dataset = [tokenize_line(line, token2id) for line in dataset_text]
        split_idx = int(0.8 * len(tokenized_dataset))
        train_data = tokenized_dataset[:split_idx]
        val_data = tokenized_dataset[split_idx:]
        val_lines = dataset_text[split_idx:]

    # Create the model configuration
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_SEQ_LEN,
        n_embd=EMBEDDING_DIM,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS
    )

    # Instantiate the model (random init)
    model = GPT2LMHeadModel(config)

    # Create PyTorch datasets and loaders
    train_dataset = MathExpressionDataset(train_data, max_len=MAX_SEQ_LEN, token2id=token2id)
    val_dataset = MathExpressionDataset(val_data, max_len=MAX_SEQ_LEN, token2id=token2id)

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the model
    #dir_name = f"discrete_model_for_grpo_digit_range_{DIGIT_RANGE.start}_{DIGIT_RANGE.stop}_Seq_Length_{SEQ_LENGTH}_Layer_{NUM_LAYERS}_Head_{NUM_HEADS}"
    #model_path = f"models/{dir_name}/discrete_model_Digit{DIGIT_RANGE.start}-{DIGIT_RANGE.stop}_Mod1_Seq{SEQ_LENGTH}_Emb{EMBEDDING_DIM}_Split{SPLIT_METHOD}_Batch{BATCH_SIZE}_Epochs1000.pt"
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # curriculum on the loss steps over output tokens
    for stage_idx, LOSS_STEPS in enumerate(CURRICULUM, start=1):
        print("\n" + "=" * 70)
        print(f"CURRICULUM STAGE {stage_idx}/{len(CURRICULUM)} - supervising tokens {LOSS_STEPS}")
        print("=" * 70)

        train_losses = []
        train_losses_tokens = []
        val_losses = []
        val_losses_tokens = []
        val_accuracies = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.
            train_loss_tokens = np.zeros((SEQ_LENGTH, ))  # ceil of seq_length / mod
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # self‑generation on unsupervised positions
                arrow_pos = (input_ids == token2id["->"]).int().argmax(dim=1)  # (B,)
                all_offsets = arrow_pos.unsqueeze(1) + 1 + torch.arange(SEQ_LENGTH, device=device)
                keep_mask = torch.zeros_like(all_offsets, dtype=torch.bool)
                keep_mask[:, LOSS_STEPS] = True
                holes = all_offsets[~keep_mask].unique().tolist()  # e.g. [fixed S0 position]

                # generate predictions for the holes
                #if holes:
                #    input_ids = autoreg_fill(model, input_ids, holes, pad_id=token2id["<PAD>"])
                if holes:
                    embeds = soft_autoreg_fill(
                        model,
                        input_ids.clone(),
                        attention_mask,
                        hole_offsets=holes
                    )
                else:
                    embeds = model.transformer.wte(input_ids)

                labels_masked = mask_labels_for_loss(
                    input_ids, attention_mask,
                    arrow_id=token2id["->"],
                    output_offsets=LOSS_STEPS
                )
                outputs = model(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    labels=labels_masked
                )
                loss = outputs.loss

                #Calculate token-level loss
                train_loss_tokens += get_token_loss(outputs, labels, SEQ_LENGTH)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_loss_tokens /= len(train_loader)
            train_loss_tokens /= BATCH_SIZE
            train_losses.append(avg_train_loss)
            train_losses_tokens.append(train_loss_tokens.tolist())

            # Validation
            model.eval()
            val_loss = 0.0
            val_loss_tokens = np.zeros((SEQ_LENGTH, ))  # ceil of seq_length / mod
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    # self‑generation on unsupervised positions
                    arrow_pos = (input_ids == token2id["->"]).int().argmax(dim=1)  # (B,)
                    all_offsets = arrow_pos.unsqueeze(1) + 1 + torch.arange(SEQ_LENGTH, device=device)
                    keep_mask = torch.zeros_like(all_offsets, dtype=torch.bool)
                    keep_mask[:, LOSS_STEPS] = True
                    holes = all_offsets[~keep_mask].unique().tolist()  # e.g. [fixed S0 position]

                    # generate predictions for the holes
                    #if holes:
                    #    input_ids = autoreg_fill(model, input_ids, holes, pad_id=token2id["<PAD>"])
                    if holes:
                        embeds = soft_autoreg_fill(
                            model,
                            input_ids.clone(),
                            attention_mask,
                            hole_offsets=holes
                        )
                    else:
                        embeds = model.transformer.wte(input_ids)

                    labels_masked = mask_labels_for_loss(
                        input_ids, attention_mask,
                        arrow_id=token2id["->"],
                        output_offsets=LOSS_STEPS
                    )
                    outputs = model(
                        inputs_embeds=embeds,
                        attention_mask=attention_mask,
                        labels=labels_masked
                    )
                    val_loss += outputs.loss.item()

                    # Calculate token-level validation loss
                    val_loss_tokens += get_token_loss(outputs, labels, SEQ_LENGTH)

            avg_val_loss = val_loss / len(val_loader)
            val_loss_tokens /= len(val_loader)
            val_loss_tokens /= BATCH_SIZE
            val_losses_tokens.append(val_loss_tokens.tolist())
            val_losses.append(avg_val_loss)

            val_accuracy, _ = evaluate_model_hidden(model, val_lines, token2id, id2token, device, LOSS_STEPS)
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Accuracy: {val_accuracy:.2%} | "
                  f"Val Loss Tokens: " + "-".join([f"{val_loss_tokens[i]:.4f}" for i in range(len(val_loss_tokens))]) + " | "
                  f"Train Loss Tokens: " + "-".join([f"{train_loss_tokens[i]:.4f}" for i in range(len(train_loss_tokens))]) + " | ")

        # Plot losses and accuracies
        epochs_range = range(1, NUM_EPOCHS+1)
        file_name = (
            f"Digit{DIGIT_RANGE.start}-{DIGIT_RANGE.stop}_Mod{MOD}"
            f"_Seq{SEQ_LENGTH}_Emb{EMBEDDING_DIM}_Split{SPLIT_METHOD}"
            f"_Batch{BATCH_SIZE}_Epochs{NUM_EPOCHS}_Stage{stage_idx}"
        )
        plot_losses(epochs_range, train_losses, val_losses, model_type=f"discrete_{file_name}")
        plot_accuracies(epochs_range, val_accuracies, model_type=f"discrete_{file_name}")

        # save the model to disk
        torch.save(model.state_dict(), f"models/{OUTPUT_DIR}discrete_model_{file_name}.pt")

    # Inference
    model.eval()

    # Calculate final accuracy on the validation set
    accuracy, _ = evaluate_model_hidden(model, val_lines, token2id, id2token, device, LOSS_STEPS)
    print(f"Validation step-by-step accuracy: {accuracy:.2%}")
    """k_values = [i for i in range(1, 15)]
    for k in k_values:
        acc_dict = evaluate_model_pass_k(model, val_lines, token2id, id2token, device, k=k)
        print(f"\nPass-{k} accuracies by temperature:")
        for temp, acc in acc_dict.items():
            print(f"\tTemperature {temp:.1f}: {acc:.2%}")"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model for the countdown game.")
    parser.add_argument("--max_seq_len", type=int, default=11, help="Maximum sequence length.")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension.")
    parser.add_argument("--digit_range", type=int, nargs=2, default=[1, 10], help="Range of digits (start, end).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--seq_length", type=int, default=4, help="Length of digit sequences.")
    parser.add_argument("--split_method", type=str, default="random_permutation", choices=["random", "sequential", "random_permutation"],
                        help="Method to split dataset.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for saving models and plots.")
    parser.add_argument("--mod", type=int, default=1, help="Mod value for the dataset.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads in the model.")

    args = parser.parse_args()
    args.digit_range = range(args.digit_range[0], args.digit_range[1])

    main(args)
