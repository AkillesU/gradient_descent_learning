import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

df = pd.read_csv("../likelihood_model/likel_results/model_fit_results.csv")

df_exclude = pd.read_csv("../data/exclusions.csv")

# Exclude participants
df = df[~df['id'].isin(df_exclude['id'])]

n_participants = int(len(df)/10)

"""
For the "best_strategies" column we need to convert list-like strings into actual lists and include the contents
in the count for strategies/trial. 
"""

# Function to safely convert stringified lists to actual lists
def parse_strategies(strategy_entry):
    try:
        # This will handle both string entries and stringified lists
        strategies = ast.literal_eval(strategy_entry)
        if isinstance(strategies, list):
            return strategies
        else:
            return [strategies]
    except ValueError:
        # Direct string entries will cause a ValueError, so we return them as a list
        return [strategy_entry]

# Apply the function to the 'best_strategies' column
df['parsed_strategies'] = df['best_strategies'].apply(parse_strategies)

# Count occurrences
strategy_counts_per_trial = pd.DataFrame(index=np.arange(1, 11), columns=["strong", "weak1", "weak2", "prototype"]).fillna(0)

# Iterate through the DataFrame and count occurrences
for index, row in df.iterrows():
    trial = row['trial']
    strategies = row['parsed_strategies']
    for strategy in strategies:
        if strategy in strategy_counts_per_trial.columns:
            strategy_counts_per_trial.loc[trial, strategy] += 1


fig, ax = plt.subplots(2, 2, figsize=(16, 14)) # Initialise figure

# Plot 1: "Strategies across trials"
strategies = ['strong', 'weak1', 'weak2', 'prototype', 'guessing']
colors = plt.cm.Paired(np.linspace(0, 1, len(strategies)))

for i, strategy in enumerate(strategies):
    # Count strategy occurrences for each trial
    trial_counts = df[df['best_strategy_guess'] == strategy].groupby('trial').size()

    # Fill missing trial indices with 0
    trial_counts = trial_counts.reindex(np.arange(1, 11), fill_value=0)

    ax[0,0].plot(trial_counts.index, trial_counts.values/n_participants, color=colors[i], label=strategy)

ax[0,0].set_title(f"Strategies across trials (n = {n_participants})")
ax[0,0].set_xlabel("Trial")
ax[0,0].set_ylabel("Count")
ax[0,0].set_xticks(np.arange(1, 11))
ax[0,0].legend()
ax[0,0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot 2: "Total strategy incidence"
total_strategy_counts = df['best_strategy_guess'].value_counts()
total_strategy_counts.reindex(strategies).plot(kind='bar', ax=ax[0,1], color=colors)
ax[0,1].set_title(f"Total strategy incidence (n = {n_participants})")
ax[0,1].set_xlabel("Strategy")
ax[0,1].set_ylabel("Count")
ax[0,1].grid(True, axis='y', linestyle='--', linewidth=0.5)

# Plot 3: "Strategies across trials" for 'best_strategies'
# Colors for each strategy
colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_counts_per_trial)))
strategy_counts_per_trial = strategy_counts_per_trial.reindex(np.arange(1, 11), fill_value=0)

# Plot each strategy with a different color
for (strategy, counts), color in zip(strategy_counts_per_trial.items(), colors):
    ax[1,0].plot(range(1,11), counts/n_participants, label=strategy, color=color)

ax[1, 0].set_title(f"Strategies across trials for 'best_strategies' (n = {n_participants})")
ax[1, 0].set_xlabel("Trial")
ax[1, 0].set_ylabel("Count")
ax[1, 0].set_xticks(np.arange(1, 11))
ax[1, 0].legend()
ax[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot 4: "Total strategy incidence" for 'best_strategies'
total_strategy_counts = strategy_counts_per_trial.sum(axis=0)
total_strategy_counts.reindex(strategies).plot(kind='bar', ax=ax[1, 1], color=colors)
ax[1, 1].set_title(f"Total strategy incidence for 'best_strategies' (n = {n_participants})")
ax[1, 1].set_xlabel("Strategy")
ax[1, 1].set_ylabel("Count")
ax[1, 1].grid(True, axis='y', linestyle='--', linewidth=0.5)


plt.tight_layout()
plt.savefig(f"images/strategy_plot_part{n_participants}")
plt.show()
