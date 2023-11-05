import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("../likelihood_model/likel_results/model_fit_results.csv")

df_exclude = pd.read_csv("../data/exclusions.csv")

# Exclude participants
df = df[~df['id'].isin(df_exclude['id'])]

n_participants = int(len(df)/10) # Set n_participants for versioning

grouped_means = df.groupby('trial').mean()
grouped_stds = df.groupby('trial').std()
grouped_counts = df.groupby('trial').count()
grouped_se = grouped_stds / np.sqrt(grouped_counts) # Grouped Standard errors

plt.figure(figsize=(10, 6)) # Initialise figure

columns = ['strong_alpha', 'weak1_alpha', 'weak2_alpha', 'proto_alpha']

for column in columns:
    plt.plot(grouped_means.index, grouped_means[column], label=column)
    plt.fill_between(grouped_means.index,
                     grouped_means[column] - grouped_se[column],
                     grouped_means[column] + grouped_se[column],
                     alpha=0.2)  # Shading for 1 standard error

plt.title('Mean Alpha Across Trials (Shade = SE)')
plt.xlabel('Trial')
plt.ylabel('Value')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f"images/alpha_plot_part{n_participants}")

plt.show()

