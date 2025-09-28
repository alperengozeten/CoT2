import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import re


def parse_val_accuracy(file_path):
    """
    Parse Val Accuracy from lines of the form:
    Epoch 1 | Train Loss: 12.2003 | Val Loss: 10.8905 | Val Accuracy: 19.27% | ...
    
    Returns a pandas DataFrame with columns ['epoch', 'val_accuracy'].
    """
    # Regex: 
    #   1) Capture epoch number: "Epoch (\d+)"
    #   2) Capture Val Accuracy: "Val Accuracy: ([\d\.]+)%"
    pattern = re.compile(r"Epoch\s+(\d+)\s+.*Val Accuracy:\s+([\d\.]+)%")

    data = []
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                val_acc = float(match.group(2))  # Convert "19.27" to float
                data.append({"epoch": epoch, "val_accuracy": val_acc})

    df = pd.DataFrame(data)
    return df


def bar_plot():
    plt.rcParams['font.family'] = 'serif'
    mnns_accs = {
        "CoT2": 98.94,
        "COCONUT": 92.58,
        "Discrete CoT": 84.92,
        "Discrete no-CoT": 68.35
    }

    prosqa_accs = {
        "CoT2": 93.37,
        "COCONUT": 90.03,
        "Discrete CoT": 68.50,
        "Discrete no-CoT": 54.91,
    }

    prontoqa_accs = {
        "CoT2": 98.01,
        "COCONUT": 96.94,
        "Discrete CoT": 82.47,
        "Discrete no-CoT": 73.65,
    }

    #'lightseagreen', 'tomato', 'goldenrod', 'cornflowerblue', 'mediumorchid', 'darkcyan'
    colors = {
        "CoT2": "lightseagreen",
        "COCONUT": "tomato",
        "Discrete CoT": "goldenrod",
        "Discrete no-CoT": "yellowgreen"
    }

    labels = list(mnns_accs.keys())
    x = np.arange(len(labels))   # bar centers
    width = 0.25                 # half-spacing between paired bars
    mnns_values = [mnns_accs[l] for l in labels]  # keep label order
    prosqa_values = [prosqa_accs[l] for l in labels]  # keep label order
    prontoqa_values = [prontoqa_accs[l] for l in labels]  # keep label order
    bar_colors = [colors[l] for l in labels]    # colour per label

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(
        x - width, mnns_values, width,
        color="lightseagreen", alpha=0.8, label='MNNS'
    )
    bars3 = plt.bar(
        x, prontoqa_values, width,
        color="goldenrod", alpha=0.8, label='ProntoQA'
    )
    bars2 = plt.bar(
        x + width, prosqa_values, width,
        color="tomato", alpha=0.8, label='ProsQA'
    )

    for bar, value in zip(bars1, mnns_values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 1,
                f"{value:.2f}", ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    for bar, value in zip(bars3, prontoqa_values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 1,
                 f"{value:.2f}", ha='center', va='bottom',
                 fontsize=12, fontweight='bold')
    for bar, value in zip(bars2, prosqa_values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 1,
                f"{value:.2f}", ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    ax = plt.gca()
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    plt.ylabel("Accuracy (%)", fontweight="bold", fontsize=17.5)
    plt.ylim(0, 100)
    plt.xticks(x, labels, fontweight='bold', fontsize=17.5)
    plt.yticks(fontweight="bold", fontsize=17.5)
    plt.legend(frameon=False, fontsize=17.5)
    plt.tight_layout()
    plt.show()
    plt.savefig("/path/to/logs/latent_thinking/latent-thinking/figures/04222025MeetingFigures/Model_Accuracies.pdf", bbox_inches='tight', format='pdf')

def plot_pass_k():
    plt.rcParams['font.family'] = 'serif'

    discrete_accs = {
        0: [39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76, 39.76],
        #0.2: [38.75, 45.89, 49.50, 51.82, 54.07, 55.50, 56.83, 57.76, 59.20, 59.67, 60.86, 61.60, 61.52, 62.22],
        0.4: [37.49, 50.15, 57.27, 61.99, 65.38, 67.82, 69.53, 71.14, 72.94, 74.12, 75.70, 76.19, 77.04, 77.85],
        #0.6: [36.31, 52.52, 61.19, 67.48, 71.69, 74.60, 78.03, 79.72, 81.33, 82.68, 84.08, 85.20, 86.02, 87.29],
        0.8: [33.28, 50.91, 61.80, 70.27, 75.17, 78.48, 81.48, 83.46, 85.91, 87.49, 88.42, 89.44, 90.43, 91.61],
        1.0: [30.28, 49.11, 61.67, 69.72, 75.45, 79.11, 82.21, 85.30, 87.32, 88.98, 90.41, 91.54, 92.35, 93.11]
    }

    k_values = [i for i in range(1, 15)]
    continuous_acc = 88.61

    emb_dim = 24
    num_layers = 1
    num_heads = 1

    plt.rcParams.update({'lines.linewidth': 4})
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.5
    })

    colors = {
        0.0: "goldenrod",
        0.2: "plum",
        0.4: "mediumseagreen",
        0.6: "skyblue",
        0.8: "lightsalmon",
        1.0: "tomato" 
    }

    plt.figure(figsize=(10, 6))
    for temp, accs in discrete_accs.items():
        plt.plot(k_values, accs, marker='o', label=f'Discrete CoT (Temp={temp:.1f})', color=colors[temp])
    plt.axhline(y=continuous_acc, color='lightseagreen', linestyle='--', label='CoT2')
    plt.xlabel("k (number of samples)", fontsize=20, fontweight='bold')
    plt.ylabel("Pass@k accuracy (%)", fontsize=20, fontweight='bold')
    plt.xlim(0, 15)
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.legend(loc='lower right', fontsize=20)
    plt.grid(True, alpha=0.5)
    plt.savefig(f"/path/to/logs/latent_thinking/latent-thinking/figures/04222025MeetingFigures/test_Pass_k_Embed_{emb_dim}_Layers_{num_layers}_Heads_{num_heads}.pdf", bbox_inches='tight', format='pdf')


def generate_discrete_comparison_plot():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'lines.linewidth': 4,
    })

    dr = 25  ## downsample rate
    digit_range_1 = 5
    digit_range_2 = 14
    embed_dim = 32
    num_layers = 2
    num_heads = 2
    seq_length = 5
    split_method = 'random_permutation'
    out_file_path_list = [
        f"/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_{digit_range_1}_{digit_range_2}_Seq_Length_{seq_length}_Layer_{num_layers}_Head_{num_heads}/Embed{embed_dim}_random_permutation_hard_teacher_64.out",
        f"/path/to/logs/latent_thinking/latent-thinking/logs/discrete_model_digit_range_{digit_range_1}_{digit_range_2}_Seq_Length_{seq_length}_Layer_{num_layers}_Head_{num_heads}/Embed{embed_dim}_random_permutation_1_64.out",
        f"/path/to/logs/latent_thinking/latent-thinking/logs/discrete_model_digit_range_{digit_range_1}_{digit_range_2}_Seq_Length_{seq_length}_Layer_{num_layers}_Head_{num_heads}/Embed{embed_dim}_random_permutation_2_64.out",
        f"/path/to/logs/latent_thinking/latent-thinking/logs/discrete_model_digit_range_{digit_range_1}_{digit_range_2}_Seq_Length_{seq_length}_Layer_{num_layers}_Head_{num_heads}/Embed{embed_dim}_random_permutation_5_64.out"
    ]

    color_list = ['lightseagreen', 'tomato', 'goldenrod', 'cornflowerblue']

    plt.figure(figsize=(10, 6))
    for out_file_path, color_fig in zip(out_file_path_list, color_list):
        df_val_acc = parse_val_accuracy(out_file_path)

        if len(df_val_acc) > 750:
            df_val_acc = df_val_acc.iloc[:750]

        # Downsample the data to reduce the number of markers
        downsample_rate = max(1, len(df_val_acc) // dr)  # Adjust to have at most ~200 markers
        df_val_acc_downsampled = df_val_acc.iloc[::downsample_rate]
        plt.plot(df_val_acc_downsampled['epoch'], df_val_acc_downsampled['val_accuracy'], marker='o', markersize=4, linestyle='--', color=color_fig)

    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.legend(['Continuous', 'Discrete, 5 tokens', 'Discrete, 3 tokens', 'Discrete, 1 token'], loc='lower right')
    plt.savefig(f"/path/to/logs/latent_thinking/latent-thinking/figures/04222025MeetingFigures/Seq_Length_{seq_length}_Layer_{num_layers}_Head_{num_heads}_Embed_{embed_dim}_Digit_Range_{digit_range_1}_{digit_range_2}_{split_method}.pdf", bbox_inches='tight', format='pdf')

def hard_soft_teacher_plot():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'lines.linewidth': 2.3,
    })
    # mpl.rcParams['font.serif'] = ['Computer Modern']
    dr = 25 ## downsample rate
    digit_range_1 = 1
    digit_range_2 = 10
    out_file_path_list = [
        "/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_1_10/Embed16_random_permutation_hard_teacher_16.out",
        "/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_1_10/Embed16_random_permutation_soft_teacher_16.out",
        "/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_1_10/Embed24_random_permutation_hard_teacher_16.out",
        "/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_1_10/Embed24_random_permutation_soft_teacher_16.out",
        "/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_1_10/Embed32_random_permutation_hard_teacher_16.out",
        "/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_1_10/Embed32_random_permutation_soft_teacher_16.out"]
    
    color_list = ['lightseagreen', 'tomato', 'goldenrod', 'cornflowerblue', 'mediumorchid', 'darkcyan']

    plt.figure(figsize=(10, 6))
    count = -1
    lines = []
    labels = []
    for out_file_path in out_file_path_list:
        count += 1
        df_val_acc = parse_val_accuracy(out_file_path)
        downsample_rate = max(1, len(df_val_acc) // dr)
        df_val_acc_downsampled = df_val_acc.iloc[::downsample_rate]
        
        emb_dim = [16, 24, 32][count // 2]  # Integer division to get embedding dim
        teacher_type = 'Hard' if count % 2 == 0 else 'Soft'
        label = f'{teacher_type} Teacher, Emb {emb_dim}'

        linestyle = '-' if count % 2 == 0 else '--'
        marker = 'o' if count % 2 == 0 else 's'
        color = color_list[count // 2]

        line, = plt.plot(
            df_val_acc_downsampled['epoch'], df_val_acc_downsampled['val_accuracy'],
            marker=marker, markersize=6, linestyle=linestyle, color=color
        )
        lines.append(line)
        labels.append(label)
        
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.legend(lines, labels, loc='lower right')
    plt.savefig(f"/path/to/logs/latent_thinking/latent-thinking/figures/04222025MeetingFigures/hard_soft_teacher_digit_range_{digit_range_1}_{digit_range_2}.pdf", bbox_inches='tight', format='pdf')


def generate_discrete_continuous_comparison_plot():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'lines.linewidth': 4,
    })

    dr = 25  ## downsample rate
    digit_range_1 = 1
    digit_range_2 = 10
    embed_dims = [16, 24, 32]
    num_layers = 2
    num_heads = 2
    seq_length = 4
    split_method = 'random_permutation'
    out_file_path_list = []
    for embed_dim in embed_dims:
        out_file_path_list.append((
            embed_dim,
            f"/path/to/logs/latent_thinking/latent-thinking/logs/continuous_model_digit_range_{digit_range_1}_{digit_range_2}_Seq_Length_{seq_length}_Layer_{num_layers}_Head_{num_heads}/Embed{embed_dim}_{split_method}_hard_teacher_16.out",
            f"/path/to/logs/latent_thinking/latent-thinking/logs/discrete_model_for_grpo_digit_range_{digit_range_1}_{digit_range_2}_Seq_Length_{seq_length}_Layer_{num_layers}_Head_{num_heads}/Embed{embed_dim}_{split_method}_1_16.out"
        ))
    color_list = ['lightseagreen', 'tomato', 'goldenrod']

    plt.figure(figsize=(10, 6))
    lines = []
    labels = []

    for i, (embed_dim, cont_path, disc_path) in enumerate(out_file_path_list):
        color = color_list[i]

        # Continuous
        df_val_acc_cont = parse_val_accuracy(cont_path)

        if len(df_val_acc_cont) > 1000:
            df_val_acc_cont = df_val_acc_cont.iloc[:1000]

        downsample_rate = max(1, len(df_val_acc_cont) // dr)
        df_val_acc_cont = df_val_acc_cont.iloc[::downsample_rate]
        line1, = plt.plot(
            df_val_acc_cont['epoch'], df_val_acc_cont['val_accuracy'],
            linestyle='-', color=color, marker='o', markersize=5, linewidth=2
        )
        lines.append(line1)
        labels.append(f"Continuous SFT, Emb {embed_dim}")

        # Discrete
        df_val_acc_disc = parse_val_accuracy(disc_path)
        downsample_rate = max(1, len(df_val_acc_disc) // dr)
        df_val_acc_disc = df_val_acc_disc.iloc[::downsample_rate]
        line2, = plt.plot(
            df_val_acc_disc['epoch'], df_val_acc_disc['val_accuracy'],
            linestyle='--', color=color, marker='s', markersize=6, linewidth=2
        )
        lines.append(line2)
        labels.append(f"Discrete SFT, Emb {embed_dim}")

    plt.xlabel('Epoch', fontweight='bold', fontsize=20)
    plt.ylabel('Validation Accuracy (%)', fontweight='bold', fontsize=20)
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.grid(True)
    plt.legend(lines, labels, loc='lower right', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"/path/to/logs/latent_thinking/latent-thinking/figures/04222025MeetingFigures/compare_cont_disc_digit_range_{digit_range_1}_{digit_range_2}_layers{num_layers}_heads{num_heads}.pdf", bbox_inches='tight', format='pdf')

def generate_discrete_continuous_comparison_multi_run_plot():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'legend.fontsize': 14,
        'lines.linewidth': 4,
    })

    dr = 10  ## downsample rate
    digit_range_1 = 1
    digit_range_2 = 10
    embed_dims = [16, 24, 32]
    num_layers = 2
    num_heads = 2
    seq_length = 4
    split_method = 'random_permutation'
    out_file_path_list = []
    seeds = [42, 43, 44, 45, 46]
    base_colors = ["lightseagreen", "tomato", "goldenrod"]

    cont_tmpl = ("/path/to/logs/latent_thinking/latent-thinking/logs/"
                 "continuous_model_multi_run_digit_range_{d1}_{d2}"
                 "_Seq_Length_{sl}_Layer_{nl}_Head_{nh}/"
                 "Embed{ed}_{sm}_hard_teacher_16_Seed{sd}.out")
    disc_tmpl = ("/path/to/logs/latent_thinking/latent-thinking/logs/"
                 "discrete_model_multi_run_digit_range_{d1}_{d2}"
                 "_Seq_Length_{sl}_Layer_{nl}_Head_{nh}/"
                 "Embed{ed}_{sm}_1_16_Seed{sd}.out")

    paths_by_dim: dict[int, dict[str, list[str]]] = {
        ed: {"cont": [], "disc": []} for ed in embed_dims
    }
    for ed in embed_dims:
        for sd in seeds:
            paths_by_dim[ed]["cont"].append(
                cont_tmpl.format(d1=digit_range_1, d2=digit_range_2,
                                 sl=seq_length, nl=num_layers, nh=num_heads,
                                 ed=ed, sm=split_method, sd=sd))
            paths_by_dim[ed]["disc"].append(
                disc_tmpl.format(d1=digit_range_1, d2=digit_range_2,
                                 sl=seq_length, nl=num_layers, nh=num_heads,
                                 ed=ed, sm=split_method, sd=sd))

    plt.figure(figsize=(10, 6))
    legend_lines, legend_labels = [], []

    for ed, color in zip(embed_dims, base_colors):
        dfs_c = [parse_val_accuracy(p) for p in paths_by_dim[ed]["cont"]
                 if os.path.isfile(p)]
        if not dfs_c:
            print(f"[warning] no continuous logs for Emb {ed}")
            continue

        df_c = (pd.concat(dfs_c)
                  .groupby("epoch", as_index=False)["val_accuracy"]
                  .agg(["mean", "std"])
                  .reset_index())  # columns: epoch, mean, std

        df_c = df_c[df_c["epoch"] < 1000]
        step_c = max(1, len(df_c) // dr)  # down‑sample
        df_c   = df_c.iloc[::step_c]

        # plot mean ± std
        l_cont, = plt.plot(df_c["epoch"], df_c["mean"],
                           "-", color=color, marker="o",
                           markersize=5, linewidth=2)
        plt.fill_between(df_c["epoch"],
                         df_c["mean"] - df_c["std"],
                         df_c["mean"] + df_c["std"],
                         color=color, alpha=0.20)
        legend_lines.append(l_cont)
        legend_labels.append(f"CoT2, Emb {ed}")

        # Discrete
        dfs_d = [parse_val_accuracy(p) for p in paths_by_dim[ed]["disc"]
                 if os.path.isfile(p)]
        if not dfs_d:
            print(f"[warning] no discrete logs for Emb {ed}")
            continue

        df_d = (pd.concat(dfs_d)
                  .groupby("epoch", as_index=False)["val_accuracy"]
                  .agg(["mean", "std"])
                  .reset_index())

        df_d = df_d[df_d["epoch"] < 1000]
        step_d = max(1, len(df_d) // dr)
        df_d   = df_d.iloc[::step_d]

        l_disc, = plt.plot(df_d["epoch"], df_d["mean"],
                           "--", color=color, marker="s",
                           markersize=6, linewidth=2)
        plt.fill_between(df_d["epoch"],
                         df_d["mean"] - df_d["std"],
                         df_d["mean"] + df_d["std"],
                         color=color, alpha=0.20)
        legend_lines.append(l_disc)
        legend_labels.append(f"Discrete CoT, Emb {ed}")

    plt.xlabel("Epoch", fontweight="bold", fontsize=20)
    plt.ylabel("Validation Accuracy (%)", fontweight="bold", fontsize=20)
    plt.xticks(fontweight='bold', fontsize=18)
    plt.yticks(fontweight='bold', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(legend_lines, legend_labels, loc="lower right", fontsize=15)
    plt.tight_layout()

    out_path = ("/path/to/logs/latent_thinking/latent-thinking/figures/"
                "04222025MeetingFigures/"
                f"compare_cont_disc_multi_run_digit_range_{digit_range_1}_{digit_range_2}"
                f"_layers{num_layers}_heads{num_heads}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[saved] {out_path}")

def bar_plot_beam_vs_embed():
    """Grouped bar plot: x-axis = embedding dims; bars = beam sizes."""
    plt.rcParams['font.family'] = 'serif'

    data = {
        1:  {16: 55.48, 24: 70.43, 32: 84.92},
        4:  {16: 64.20, 24: 88.31, 32: 99.04},
        16: {16: 55.27, 24: 96.56, 32: 98.94},
    }

    beam_colors = {
        1:  "tomato",
        4:  "lightseagreen",
        #8:  "cornflowerblue",
        16: "goldenrod",
    }

    embeds = [16, 24, 32]
    beams  = [1, 4, 16]

    x = np.arange(len(embeds))
    width = 0.24
    offsets = (np.arange(len(beams)) - (len(beams) - 1) / 2.0) * width

    plt.figure(figsize=(10, 6))

    bars_by_beam = []
    for off, b in zip(offsets, beams):
        vals = [data[b][e] for e in embeds]
        bars = plt.bar(x + off, vals, width,
                       label=f'B={b}',
                       color=beam_colors[b], alpha=0.85)
        bars_by_beam.append((bars, vals))

    for bars, vals in bars_by_beam:
        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                     f"{val:.2f}",
                     ha='center', va='bottom',
                     fontsize=12, fontweight='bold')

    ax = plt.gca()
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    plt.ylabel("Accuracy (%)", fontweight="bold", fontsize=17.5)
    plt.ylim(0, 100)
    plt.xticks(x, [str(e) for e in embeds], fontweight='bold', fontsize=17.5)
    plt.yticks(fontweight="bold", fontsize=17.5)
    plt.xlabel("Embedding dimension", fontweight="bold", fontsize=18)
    plt.legend(frameon=False, fontsize=17.5, ncols=1)
    plt.tight_layout()

    out_path = "/path/to/logs/latent_thinking/latent-thinking/figures/04222025MeetingFigures/beam_vs_embed_bar.pdf"
    plt.savefig(out_path, bbox_inches='tight', format='pdf')
    print(f"[saved] {out_path}")

def compute_avg_final_accuracy_multi_run():
    digit_range_1 = 1
    digit_range_2 = 10
    embed_dims = [16, 24, 32]
    num_layers = 1
    num_heads = 1
    seq_length = 4
    split_method = 'random_permutation'
    num_beams = [4, 8] # budgets
    dist_strategy = 'beam_minabs'
    seeds = [42, 43, 44, 45, 46]

    cont_tmpl = ("/path/to/logs/latent_thinking/latent-thinking/logs/"
                 "continuous_model_multi_run_digit_range_{d1}_{d2}"
                 "_Seq_Length_{sl}_Layer_{nl}_Head_{nh}/"
                 "Embed{ed}_{sm}_hard_teacher_16_Seed{sd}.out")
    cont_beam_tmpl = ("/path/to/logs/latent_thinking/latent-thinking/logs/"
                      "continuous_model_multi_run_digit_range_{d1}_{d2}"
                      "_Seq_Length_{sl}_Layer_{nl}_Head_{nh}_Dist_Strategy_{ds}_Num_Beams_{nb}/"
                      "Embed{ed}_{sm}_hard_teacher_16_Seed{sd}.out")
    disc_tmpl = ("/path/to/logs/latent_thinking/latent-thinking/logs/"
                 "discrete_model_multi_run_digit_range_{d1}_{d2}"
                 "_Seq_Length_{sl}_Layer_{nl}_Head_{nh}/"
                 "Embed{ed}_{sm}_1_16_Seed{sd}.out")

    rows = []
    for ed in embed_dims:
        # CoT2 baseline with full budget
        cont_finals = []
        for sd in seeds:
            p = cont_tmpl.format(d1=digit_range_1, d2=digit_range_2,
                                 sl=seq_length, nl=num_layers, nh=num_heads,
                                 ed=ed, sm=split_method, sd=sd)
            if not os.path.isfile(p):
                continue
            df = parse_val_accuracy(p)
            if len(df) == 0:
                continue
            cont_finals.append(float(df["val_accuracy"].iloc[-1]))

        if cont_finals:
            rows.append({
                "model": "CoT2",
                "embed_dim": ed,
                "n_runs": len(cont_finals),
                "mean_final_acc": sum(cont_finals)/len(cont_finals),
                "std_final_acc": float(pd.Series(cont_finals).std(ddof=1)) if len(cont_finals) > 1 else 0.0,
            })

        # CoT2 with different budgets
        for nb in num_beams:
            beam_finals = []
            for sd in seeds:
                p = cont_beam_tmpl.format(d1=digit_range_1, d2=digit_range_2,
                                          sl=seq_length, nl=num_layers, nh=num_heads,
                                          ds=dist_strategy, nb=nb,
                                          ed=ed, sm=split_method, sd=sd)
                if not os.path.isfile(p):
                    continue
                df = parse_val_accuracy(p)
                if len(df) == 0:
                    continue
                beam_finals.append(float(df["val_accuracy"].iloc[-1]))

            if beam_finals:
                rows.append({
                    "model": f"CoT2 (Beam {nb})",
                    "embed_dim": ed,
                    "n_runs": len(beam_finals),
                    "mean_final_acc": sum(beam_finals)/len(beam_finals),
                    "std_final_acc": float(pd.Series(beam_finals).std(ddof=1)) if len(beam_finals) > 1 else 0.0,
                })

        # Discrete CoT 
        disc_finals = []
        for sd in seeds:
            p = disc_tmpl.format(d1=digit_range_1, d2=digit_range_2,
                                 sl=seq_length, nl=num_layers, nh=num_heads,
                                 ed=ed, sm=split_method, sd=sd)
            if not os.path.isfile(p):
                continue
            df = parse_val_accuracy(p)
            if len(df) == 0:
                continue
            disc_finals.append(float(df["val_accuracy"].iloc[-1]))

        if disc_finals:
            rows.append({
                "model": "Discrete CoT",
                "embed_dim": ed,
                "n_runs": len(disc_finals),
                "mean_final_acc": sum(disc_finals)/len(disc_finals),
                "std_final_acc": float(pd.Series(disc_finals).std(ddof=1)) if len(disc_finals) > 1 else 0.0,
            })

    summary = pd.DataFrame(rows, columns=["model","embed_dim","n_runs","mean_final_acc","std_final_acc"])

    # Pretty print
    if not summary.empty:
        print("\nAverage FINAL Validation Accuracy (multi-run):")
        for (m, ed), sub in summary.groupby(["model","embed_dim"]):
            r = sub.iloc[0]
            print(f"  {m:16s} | Emb {ed:2d} | runs={int(r['n_runs'])} | "
                  f"mean={r['mean_final_acc']:.2f}% | std={r['std_final_acc']:.2f}")
    else:
        print("[info] No runs found for the specified setting.")

    return summary


if __name__ == "__main__":
    #plot_pass_k()
    #generate_discrete_comparison_plot()
    #hard_soft_teacher_plot()
    #generate_discrete_continuous_comparison_plot()
    #bar_plot()
    #generate_discrete_continuous_comparison_multi_run_plot()
    #calc_averages()
    #compute_avg_final_accuracy_multi_run()
    bar_plot_beam_vs_embed()