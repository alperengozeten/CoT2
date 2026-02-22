import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
from data.generate_data import generate_vocab, generate_text_dataset, permutation_train_val_split
from eval.evaluation import evaluate_model, evaluate_model_pass_k
from utils import plot_losses, plot_accuracies
import argparse
import torch.nn as nn
import numpy as np


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
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


def encode_prompt(digit_seq, token2id):
    # e.g. digit_seq = [5, 3, 2, 4]
    # We'll build a prompt like: <BOS> D5 D3 D2 D4 ->
    tokens = ["<BOS>"] + [f"D{d}" for d in digit_seq] + ["->"]
    return [token2id[t] for t in tokens]


def decode_tokens(token_ids, id2token):
    return [id2token[i] for i in token_ids]


def get_token_loss(outputs, labels, seq_length, mod):
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
    return per_token_loss[:, seq_length+1:seq_length+1+((seq_length + mod - 1) // mod)].sum(dim=0).cpu().detach().numpy()


def main(args=None):
    # set the hyperparameters
    MAX_SEQ_LEN = args.max_seq_len
    EMBEDDING_DIM = args.embedding_dim
    DIGIT_RANGE = args.digit_range
    BATCH_SIZE = args.batch_size
    SEQ_LENGTH = args.seq_length
    MOD = args.mod
    SPLIT_METHOD = args.split_method
    NUM_EPOCHS = args.num_epochs
    OUTPUT_DIR = args.output_dir
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads

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
          f"Num Heads: {NUM_HEADS}")

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

    train_losses = []
    train_losses_tokens = []
    val_losses = []
    val_losses_tokens = []
    val_accuracies = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.
        train_loss_tokens = np.zeros((((SEQ_LENGTH + MOD - 1) // MOD), ))  # ceil of seq_length / mod
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            #Calculate token-level loss
            train_loss_tokens += get_token_loss(outputs, labels, SEQ_LENGTH, MOD)
            
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
        val_loss_tokens = np.zeros((((SEQ_LENGTH + MOD - 1) // MOD), ))  # ceil of seq_length / mod
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

                # Calculate token-level validation loss
                val_loss_tokens += get_token_loss(outputs, labels, SEQ_LENGTH, MOD)

        avg_val_loss = val_loss / len(val_loader)
        val_loss_tokens /= len(val_loader)
        val_loss_tokens /= BATCH_SIZE
        val_losses_tokens.append(val_loss_tokens.tolist())
        val_losses.append(avg_val_loss)

        val_accuracy, _ = evaluate_model(model, val_lines, token2id, id2token, device)
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
        f"_Batch{BATCH_SIZE}_Epochs{NUM_EPOCHS}"
    )
    plot_losses(epochs_range, train_losses, val_losses, model_type=f"discrete_{file_name}")
    plot_accuracies(epochs_range, val_accuracies, model_type=f"discrete_{file_name}")

    # Inference
    model.eval()

    # Calculate final accuracy on the validation set
    accuracy, _ = evaluate_model(model, val_lines, token2id, id2token, device)
    print(f"Validation step-by-step accuracy: {accuracy:.2%}")
    """k_values = [i for i in range(1, 15)]
    for k in k_values:
        acc_dict = evaluate_model_pass_k(model, val_lines, token2id, id2token, device, k=k)
        print(f"\nPass-{k} accuracies by temperature:")
        for temp, acc in acc_dict.items():
            print(f"\tTemperature {temp:.1f}: {acc:.2%}")"""

    # save the model to disk
    torch.save(model.state_dict(), f"models/{OUTPUT_DIR}discrete_model_{file_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT-2 model for the countdown game.")
    parser.add_argument("--max_seq_len", type=int, default=11, help="Maximum sequence length.")
    parser.add_argument("--embedding_dim", type=int, default=24, help="Embedding dimension.")
    parser.add_argument("--digit_range", type=int, nargs=2, default=[1, 10], help="Range of digits (start, end).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--seq_length", type=int, default=4, help="Length of digit sequences.")
    parser.add_argument("--split_method", type=str, default="random_permutation", choices=["random", "sequential", "random_permutation"],
                        help="Method to split dataset.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for saving models and plots.")
    parser.add_argument("--mod", type=int, default=1, help="Mod value for the dataset.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the model.")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads in the model.")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random-seed for reproducibility.")

    args = parser.parse_args()
    args.digit_range = range(args.digit_range[0], args.digit_range[1])

    set_seed(args.seed)
    main(args)
