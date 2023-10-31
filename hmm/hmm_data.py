import ast
import pandas as pd

initial_data = pd.read_csv("../likelihood_model/likel_results/model_fit_results.csv")

# Creating list of methods to deal with equally likely strategies
methods = ["best_strategies", "best_strategy_rand", "best_strategy_guess" ]

"""
best_strategy_rand: A random strategy is selected from equally likely strategies for a given trial

best_strategy_guess: When two or more strategies are equally as likely for a given trial
                     the strategy is set as "guessing"

best_strategies: Outputs all equally likely strategies in a list []
"""
# For each method, create a .csv file with columns for all strategies.
# If a strategy is being used on a trial, the value is set to 1
# if not, the value is set to 0.
for method in methods:

    data = initial_data[method]

    # Creating a dictionary to map strategies to an index on the df
    strategy_to_idx = {
        'strong': 0,
        'weak1': 1,
        'weak2': 2,
        'prototype': 3,
        'guessing': 4
    }
    train_data = pd.DataFrame(columns=["strong", "weak1", "weak2", "prototype", "guessing"])
    for row in range(0, len(data)):
        empty_row = [0, 0, 0, 0, 0]  # Create an empty list

        # Check if the current row's data is a string representation of a list or just a single string
        item = data[row]
        if "[" in item and "]" in item:  # Check for brackets to identify lists
            strategies = ast.literal_eval(item) # Convert string list "[]" into a list []
        else:
            strategies = [item]

        for strategy in strategies:
            idx = strategy_to_idx[strategy]  # Get strategy index from dict
            empty_row[idx] = 1  # Place "1" to the correct index in the list

        # Convert list to a DataFrame for concatenation
        empty_df_row = pd.DataFrame([empty_row], columns=["strong", "weak1", "weak2", "prototype", "guessing"])
        train_data = pd.concat([train_data, empty_df_row])  # Concat the new row to the train_data

    # Drop the guessing column if it's empty
    if method != "best_strategy_guess":
        train_data.drop(train_data.columns[-1], axis=1, inplace=True)

    print(train_data)
    train_data.to_csv(f"likel_results/hmm_data_{method}.csv", index=False)