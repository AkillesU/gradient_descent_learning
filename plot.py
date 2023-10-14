import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read results data
data = pd.read_csv("results/model_fit_results.csv")

# Data into pd Dataframe
df = pd.DataFrame(data)

# Group by "trial" and "strategy" and then count the occurrences
agg_df = df.groupby(['trial', 'strategy']).size().reset_index(name='count')

# Fill in missing combinations
all_trials = range(1, 11)  # assuming trial numbers go from 1 to 10
all_categories = df['strategy'].unique()
index = pd.MultiIndex.from_product([all_trials, all_categories], names=['trial', 'strategy'])
agg_df = agg_df.set_index(['trial', 'strategy']).reindex(index).reset_index().fillna(0)


# Plot using seaborn
plt.figure(figsize=(10,6))
sns.lineplot(data=agg_df, x='trial', y='count', hue='strategy', marker='o')
plt.title('Number of Categories per Trial')
plt.show()