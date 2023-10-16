import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../results/model_fit_results.csv")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: "Strategies across trials"
strategies = ['strong', 'weak1', 'weak2', 'prototype']
colors = plt.cm.Paired(np.linspace(0, 1, len(strategies)))

for i, strategy in enumerate(strategies):
    # Count strategy occurrences for each trial
    trial_counts = df[df['best_strategy'] == strategy].groupby('trial').size()

    # Fill missing trial indices with 0
    trial_counts = trial_counts.reindex(np.arange(1, 11), fill_value=0)

    ax[0].plot(trial_counts.index, trial_counts.values, color=colors[i], label=strategy)

ax[0].set_title("Strategies across trials")
ax[0].set_xlabel("Trial")
ax[0].set_ylabel("Count")
ax[0].set_xticks(np.arange(1, 11))
ax[0].legend()
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot 2: "Total strategy incidence"
total_strategy_counts = df['best_strategy'].value_counts()
total_strategy_counts.reindex(strategies).plot(kind='bar', ax=ax[1], color=colors)
ax[1].set_title("Total strategy incidence")
ax[1].set_xlabel("Strategy")
ax[1].set_ylabel("Count")
ax[1].grid(True, axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
