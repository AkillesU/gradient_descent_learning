import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.models import clone_model
from tqdm import tqdm
from scipy import optimize
import os
import ast
from scipy.optimize import differential_evolution
from bayes_opt import BayesianOptimization, UtilityFunction
import random
import time

tqdm.pandas()  # This enables `progress_apply` functionality

"""
This script is for behavioural modelling of human data in the GD project.
The modelling will be done in the following way:
    For each participant
1. Construct a 3 unit 1 output ANN
2. Train the model on the first sequence of 14 bugs in the order they were
   presented. No batching. Use a "random" learning rate
3. Do a forward pass with the 8 bug permutations and apply a softmax function.
4. Remove the most likely choice and run forward pass again with 7 bugs. 
5. Repeat process until 4 choices are made. 
6. Calculate error against human choices. 
7. Run again with new learning rate and temperature parameter (softmax).
8. Optimise this process until negative log likelihood is maximised.
9. Save results from the block (LR, weights, and Temp). 
10. Continue on to the next learning and test block with the saved model.
11. Repeat for all 10 learning and test blocks

Repeat process for all participants
"""


"""
Let's define it such that the first feature is the strong one, second is weak1, third is weak2

strong, weak1, weak2

1 will be predictive of bug 1 and 0 will be predictive of 0

This setup will be kept across participants to ease interpretation.
"""
def strong_code(Spreadsheet, bug, code_file):
    # Dictionary for getting bug_id for each dim_value (1,2)
    dim_val_to_bug_id = {
        '1': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['feature_type'] == 'strong') &  # Match feature_type
            (code_file['dim_value'] == 1)  # Match dim_value
            ]['bug_id'].iloc[0],  # Get bug_id for dim_value == 1
        '2': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['feature_type'] == 'strong') &  # Match feature_type
            (code_file['dim_value'] == 2)  # Match dim_value
            ]['bug_id'].iloc[0]  # Get bug_id for dim_value == 2
    }

    strong_dimension = code_file[
                           (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
                           (code_file['feature_type'] == 'strong')  # Match feature_type == 'strong'
                           ]['dimension'].iloc[0] - 1  # Get dimension (- 1 to get indexer)

    # ensure bug is string
    bug = str(bug)

    # Get strong value
    strong_value = bug[strong_dimension]

    # Get whether it predicts 1 or 0 (label)
    strong_predict = dim_val_to_bug_id[strong_value]

    return strong_predict


def weak1_code(Spreadsheet, bug, code_file):
    # Filter the code_file to contain only rows with Spreadsheet and 'weak' features
    filtered_data = code_file[
        (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
        (code_file['feature_type'] == 'weak')  # Match feature_type
        ]

    # Weak1 is the weak feature with the SMALLER dimension (out of 1,3,4)
    smallest_dimension = filtered_data['dimension'].min()

    # Dictionary for getting bug_id for each dim_value (1,2)
    dim_val_to_bug_id = {
        '1': filtered_data[(
                (filtered_data['dimension'] == smallest_dimension) &  # Select Weak features with lower dimension
                (filtered_data['dim_value'] == 1))  # Match dim_value
        ]['bug_id'].iloc[0],  # Get bug_id for dim_value == 1

        '2': filtered_data[(
                (filtered_data['dimension'] == smallest_dimension) &  # Select Weak features with lower dimension
                (filtered_data['dim_value'] == 2))  # Match dim_value
        ]['bug_id'].iloc[0]  # Get bug_id for dim_value == 2
    }

    # Setting weak1_dimension based on the smaller dimension (- 1 to get indexer)
    weak1_dimension = smallest_dimension - 1

    # ensure bug is string
    bug = str(bug)

    # Get strong value
    weak1_value = bug[weak1_dimension]

    # Get whether it predicts 1 or 0 (label)
    weak1_predict = dim_val_to_bug_id[weak1_value]

    return weak1_predict


def weak2_code(Spreadsheet, bug, code_file):
    # Filter the code_file to contain only rows with Spreadsheet and 'weak' features
    filtered_data = code_file[
        (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
        (code_file['feature_type'] == 'weak')  # Match feature_type
        ]

    # Weak2 is the weak feature with the LARGER dimension (out of 1,3,4)
    largest_dimension = filtered_data['dimension'].max()

    # Dictionary for getting bug_id for each dim_value (1,2)
    dim_val_to_bug_id = {
        '1': filtered_data[(
                (filtered_data['dimension'] == largest_dimension) &  # Select Weak features with larger dimension
                (filtered_data['dim_value'] == 1))  # Match dim_value
        ]['bug_id'].iloc[0],  # Get bug_id for dim_value == 1

        '2': filtered_data[(
                (filtered_data['dimension'] == largest_dimension) &  # Select Weak features with larger dimension
                (filtered_data['dim_value'] == 2))  # Match dim_value
        ]['bug_id'].iloc[0]  # Get bug_id for dim_value == 2
    }

    # Setting weak2_dimension based on the larger dimension (- 1 to get indexer)
    weak2_dimension = largest_dimension - 1

    # ensure bug is string
    bug = str(bug)

    # Get strong value
    weak2_value = bug[weak2_dimension]

    # Get whether it predicts 1 or 0 (label)
    weak2_predict = dim_val_to_bug_id[weak2_value]

    return weak2_predict


# 21211
def decode_features(Spreadsheet, bug, code_file):

    strong_value = int(strong_code(Spreadsheet, bug, code_file))
    weak1_value = int(weak1_code(Spreadsheet, bug, code_file))
    weak2_value = int(weak2_code(Spreadsheet, bug, code_file))

    # Make into tuple to allow for use as dict keys later on. Can also be converted to numpy array
    cleaned_bug = (strong_value, weak1_value, weak2_value)

    return cleaned_bug


def create_model(learning_rate=0.01):
    # Define the model
    initializer = Zeros()  # Zero initialisation
    # build Sequential model with three inputs and one output
    model = keras.Sequential([
        layers.Dense(1, input_shape=(3,), kernel_initializer=initializer, bias_initializer=Zeros(),
                     activation="sigmoid") # Sigmoid activation for prob output. Otherwise getting label=0 probabilities is difficult
    ]) # NOTE: sigmoid might not be the correct answer

    # Configure the optimizer
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


# Softmax function
def softmax(scores_dict, temperature, test_label):
    # Extract values and apply transformation if test_label is 0
    scores = np.array(list(scores_dict.values()))
    if test_label == 0:
        scores = 1 - scores  # Invert scores if test_label is 0

    # Stability improvement: subtract the max score to prevent overflow
    c = np.max(scores / temperature)
    exp_scores = np.exp((scores / temperature) - c)
    exp_scores = np.clip(exp_scores, 1e-10, 1e10)  # Clipping both lower and upper bounds

    softmax_probs = exp_scores / np.sum(exp_scores)
    # Create a new dictionary with the same keys but updated values
    softmax_dict = {key: prob for key, prob in zip(scores_dict.keys(), softmax_probs)}
    return softmax_dict


def optimiser(method, initial_lr, initial_temp, train_input, train_labels, test_label, targets, model, permutations,
              callback, weights, list_manager):
    start_time = time.time()
    if method == 'Nelder-Mead':
        res = optimize.minimize(
            objective_function,
            np.array([initial_lr, initial_temp]),
            args=(train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager),
            method="Nelder-Mead",
            options={"maxiter": 100},
            bounds=[(0.001, 5.0), (0.01, 10)]
        )
    elif method == 'SLSQP':
        res = optimize.minimize(
            objective_function,
            np.array([initial_lr, initial_temp]),
            args=(train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager),
            method="SLSQP",
            options={"maxiter": 100},
            bounds=[(0.001, 5.0), (0.01, 10)]
        )

    elif method == 'Differential Evolution':
        bounds = [(0.001, 5.0), (0.01, 10)]
        res = differential_evolution(
            objective_function,
            bounds,
            args=(train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager),
            strategy='rand1bin',
            maxiter=200,
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1.2),
            recombination=0.7,
            seed=None,
            callback=None,
            disp=False,
            polish=True,
            init='latinhypercube',
            #updating='deferred',  # Use 'deferred' for better parallel performance
            #workers=2  # Use all CPU cores
        )
    elif method == 'Basin-hopping':
        res = optimize.basinhopping(
            objective_function,
            np.array([initial_lr, initial_temp]),
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'args': (train_input, train_labels, targets, test_label, model, permutations, method, weights),
                'bounds': [(0.001, 5.0), (0.01, 10)]
            }
        )

    elapsed_time = time.time() - start_time





    best_lr, best_temp = res.x
    best_nlgl = res.fun

    # Train model with best settings and callback activated to save weights on each batch (item)
    # Update the learning rate of the model's existing optimizer
    tf.keras.backend.set_value(model.optimizer.learning_rate, best_lr)
    model.set_weights(weights)
    # Train new model
    model.fit(train_input, train_labels, shuffle=False, batch_size=1, verbose=2,
              callbacks=[callback]
              )

    # Save weights to csv
    callback.save_weights_to_csv(filepath="data/model_weights.csv")

    return best_nlgl, best_lr, best_temp, model, elapsed_time


class WeightSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, participant_id, block_num):
        super(WeightSaveCallback, self).__init__()
        self.participant_id = participant_id
        self.block_num = block_num
        self.weights_data = []  # List to store weight data

    def on_epoch_begin(self, epoch, logs=None):
        # Clear the weights data at the start of each epoch
        self.weights_data = []

    def set_participant_block(self, participant_id, block_num):
        self.participant_id = participant_id
        self.block_num = block_num

    def on_batch_end(self, batch, logs=None):
        # Extract weights and flatten them
        weights = self.model.get_weights()
        # Create a single row DataFrame from weights
        weights_row = {
            'Strong': weights[0][0][0],
            'Weak1': weights[0][1][0],
            'Weak2': weights[0][2][0],
            'Bias': weights[1][0],
            'Participant_ID': self.participant_id,
            'Block': self.block_num,
            'Batch': batch
        }
        # Append the row dictionary to the weights_data list
        self.weights_data.append(weights_row)

    def save_weights_to_csv(self, filepath):
        if not self.weights_data:
            print("No data to save. Check model training.")
            return

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(self.weights_data)

        # Check the DataFrame before saving
        print("DataFrame to be saved:\n", df.head())

        # Check if the file exists
        if os.path.exists(filepath):
            # File exists, so load it and concatenate new data
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(filepath, index=False)
            print(f"Data appended and saved to {filepath}")
        else:
            # File does not exist, save new data with header
            df.to_csv(filepath, index=False)
            print(f"New file created and data saved to {filepath}")


# Class to manage list of probabilities between functions
class ListManager:
    def __init__(self):
        self._stored_list = []  # Initialize with an empty list

    def set_list(self, new_list):
        # Set or update the internal list
        self._stored_list = new_list

    def get_list(self):
        # Return a copy of the list to prevent external modifications
        return self._stored_list.copy()


def objective_function(params, train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager):
    # Get learning rate and temperature
    learning_rate, temperature = params

    # Update the learning rate of the model's existing optimizer
    tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    model.set_weights(weights)
    # Train new model
    model.fit(train_input, train_labels, shuffle=False, batch_size=1, verbose=0
              )

    # Get utilities for all permutations from model
    utilities = model.predict(permutations).flatten()
    # Create dictionary with tuple keys and values from the arrays
    utilities_dict = {tuple(key): value for key, value in zip(permutations, utilities)}

    # initialise log likelihood
    total_log_likelihood = 0

    # initialise probabilities list
    probabilities = []

    # Process each choice in a sequence
    for target in targets:
        # calculate softmax probabilities for the current set of choices
        probabilities_dict = softmax(utilities_dict, temperature, test_label)

        # Get the probability of the participant choice
        target_prob = probabilities_dict[target]

        probabilities.append(target_prob)

        # remove choice from utilities
        del utilities_dict[target]

        # Ensure target_prob is not zero or near zero
        target_prob = max(target_prob, 1e-10)

        # Update log likelihood with the log likelihood of the chosen target
        total_log_likelihood += np.log(target_prob)

    # Update probabilities list in ListManager
    list_manager.set_list(new_list=probabilities)

    print(temperature, learning_rate, total_log_likelihood)
    print(method)
    # Return negative log likelihood
    return -total_log_likelihood


def main():
    # First we need to load in the participant data.
    test_data = pd.read_csv("data/test_data.csv")
    train_data = pd.read_csv("data/train_data.csv")

    # Exclude participants
    exclude_ids = pd.read_csv("../data/exclusions.csv")

    # Check Rows where id doesn't match with exlusion
    test_mask = ~test_data['id'].isin(exclude_ids['id'])
    train_mask = ~train_data['id'].isin(exclude_ids['id'])

    # Keep only rows where participant was not excluded
    test_data = test_data[test_mask]
    train_data = train_data[train_mask]

    # Read code file
    code_file = pd.read_csv("../data/fullcode.csv", delimiter=";")

    # Change labels to 0 and 1 instead of 1 and 2
    code_file["bug_id"] = code_file["bug_id"].astype("int") - 1

    train_data_modified_path = "data/train_data_modified.csv"
    if not os.path.exists(train_data_modified_path):
        # Change train_data to decoded format [strong, weak1, weak2]
        train_data['input'] = train_data.progress_apply(lambda row: decode_features(row['Spreadsheet'], row['input'], code_file), axis=1)
        train_data.to_csv(train_data_modified_path, index=False)
    else:
        train_data = pd.read_csv(train_data_modified_path)
        # Make string representations of tuples back into tuples
        train_data['input'] = train_data['input'].apply(ast.literal_eval)

    # Create list of columns to decode
    test_cols = ["1", "2", "3", "4"]
    # Decode each column
    for col in test_cols:
        test_data[col] = test_data.progress_apply(lambda row: decode_features(row["Spreadsheet"], row[col], code_file), axis=1)

    # Create new column "targets" with a list of participant selections for each block
    test_data['targets'] = test_data.apply(lambda row: [row['1'], row['2'], row['3'], row['4']], axis=1)


    # Get participant IDs in a list
    part_ids = train_data['id'].drop_duplicates().tolist()

    # Define all bug permutations for forward pass and softmax
    permutations = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
    callback = WeightSaveCallback(part_ids[0], 0)
    data_df = pd.DataFrame(columns=["id", "block", "neg_logl", "learning_rate", "temperature", "strong", "weak1", "weak2"])
    for id in part_ids:
        print(id)
        # First get the training data as a numpy array
        id_train_data = train_data[train_data["id"] == id]

        # Group by block and collect all inputs
        grouped_input = id_train_data.groupby('block')['input'].apply(list).tolist()
        grouped_label = id_train_data.groupby('block')['label'].apply(list).tolist()

        # Convert the list of lists (for each block) into a numpy array
        input_array = [np.array(block) for block in grouped_input]
        label_array = [np.array(block) for block in grouped_label]

        # Get the test data for a participant
        id_test_data = test_data[test_data["id"] == id]
        target_array = id_test_data["targets"].tolist()
        test_labels = id_test_data["label"].tolist()
        # Create model for participant
        model = create_model()
        #learning_rate = 0.9 # Maybe randomly initialise learning rate and temp?
        #temperature = 10
        for block in range(0,10):

            if block == 0:
                # Get zero initialised weights if first block
                best_weights = model.get_weights()

            inputs = input_array[block]
            labels = label_array[block]
            targets = target_array[block]
            test_label = test_labels[block]

            callback.set_participant_block(id, block)
            list_manager = ListManager()


            # Nelder-Mead seems to go for 0 learning rate after a few blocks so let's try initialising learning rate
            # for each block
            learning_rate = random.uniform(0.001, 3.0)  # Maybe randomly initialise learning rate and temp?
            temperature = random.uniform(0.01, 10)

            methods = ['Differential Evolution'] #'Nelder-Mead',
            results = []

            for method in methods:
                best_nlgl, best_lr, best_temp, best_model, comp_time = optimiser(
                    method,
                    learning_rate,
                    temperature,
                    inputs,
                    labels,
                    test_label,
                    targets,
                    model,
                    permutations,
                    callback,
                    best_weights,
                    list_manager
                )
                results.append((method, best_nlgl, best_lr, best_temp, comp_time))

            print(results)

            best_weights = best_model.get_weights()

            # Get human choice softmax probabilties
            probabilities = list_manager.get_list()

            new_data = pd.DataFrame({
                'id': id,
                'block': block,
                "neg_logl": best_nlgl,
                "learning_rate": best_lr,
                "temperature": best_temp,
                "strong": best_weights[0][0],
                "weak1": best_weights[0][1],
                "weak2": best_weights[0][2],
                "choice_prob1": probabilities[0],
                "choice_prob2": probabilities[1],
                "choice_prob3": probabilities[2],
                "choice_prob4": probabilities[3],
            }, index=[0])

            data_df = pd.concat([data_df, new_data], ignore_index=True)

            # Update model
            model = best_model
            print(data_df)
        print(data_df)

        data_df.to_csv("data/model_fit.csv")



    data_df.to_csv("data/model_fit.csv")


if __name__ == '__main__':
    main()  # Your main function that launches multiprocessing tasks





