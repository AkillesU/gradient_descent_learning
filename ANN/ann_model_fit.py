import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Zeros
from tqdm import tqdm
from scipy import optimize
import os
import ast
from scipy.optimize import differential_evolution, dual_annealing
import random
import time
from pympler import tracker
import gc
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt
tqdm.pandas()  # This enables `progress_apply` functionality
# Initialize the memory tracker
tr = tracker.SummaryTracker()

"""
This script is for behavioural modelling of human data in the GD learning project.
The modelling will be done in the following way:
    For each participant
1. Construct a 3 unit 1 output ANN
2. Train the model on the first sequence of 14 bugs in the order they were
   presented. No batching. Use a random learning rate within [0.001,5]
3. Do a forward pass with the 8 bug permutations and apply a softmax function with a temperature parameter.
4. Remove the first human choice probability and redo softmax with 7 bugs. 
5. Repeat process until all 4 choices are processed. 
6. Calculate total log likelihood for all 4 choices (sum individual lgl). 
7. Run again with new learning rate and temperature parameter (softmax).
8. Optimise this process until negative log likelihood is minimised (as close to 0 as possible).
9. Save results from the block (LR, weights, and Temp). Additionally save weights after each training item. 
10. Continue on to the next learning and test block with the saved model 
    (each optimisation iteration starts with weights from previous block)
11. Repeat for all 10 learning and test blocks

Repeat process for all participants
"""


"""
Let"s define it such that the first feature is the strong one, second is weak1, third is weak2

strong, weak1, weak2

1 will be predictive of bug 1 and 0 will be predictive of 0

This setup will be kept across participants to ease interpretation.
"""

# Function to check which label does the strong feature correspond to
def strong_code(Spreadsheet, bug, code_file):
    # Dictionary for getting bug_id for each dim_value (1,2)
    dim_val_to_bug_id = {
        "1": code_file[
            (code_file["Spreadsheet"] == Spreadsheet) &  # Match Spreadsheet
            (code_file["feature_type"] == "strong") &  # Match feature_type
            (code_file["dim_value"] == 1)  # Match dim_value
            ]["bug_id"].iloc[0],  # Get bug_id for dim_value == 1
        "2": code_file[
            (code_file["Spreadsheet"] == Spreadsheet) &  # Match Spreadsheet
            (code_file["feature_type"] == "strong") &  # Match feature_type
            (code_file["dim_value"] == 2)  # Match dim_value
            ]["bug_id"].iloc[0]  # Get bug_id for dim_value == 2
    }

    strong_dimension = code_file[
                           (code_file["Spreadsheet"] == Spreadsheet) &  # Match Spreadsheet
                           (code_file["feature_type"] == "strong")  # Match feature_type == "strong"
                           ]["dimension"].iloc[0] - 1  # Get dimension (- 1 to get indexer)

    # ensure bug is string
    bug = str(bug)

    # Get strong value
    strong_value = bug[strong_dimension]

    # Get whether it predicts 1 or 0 (label)
    strong_predict = dim_val_to_bug_id[strong_value]

    return strong_predict


# Function to check which label does the weak1 feature correspond to
def weak1_code(Spreadsheet, bug, code_file):
    # Filter the code_file to contain only rows with Spreadsheet and "weak" features
    filtered_data = code_file[
        (code_file["Spreadsheet"] == Spreadsheet) &  # Match Spreadsheet
        (code_file["feature_type"] == "weak")  # Match feature_type
        ]

    # Weak1 is the weak feature with the SMALLER dimension (out of 1,3,4)
    smallest_dimension = filtered_data["dimension"].min()

    # Dictionary for getting bug_id for each dim_value (1,2)
    dim_val_to_bug_id = {
        "1": filtered_data[(
                (filtered_data["dimension"] == smallest_dimension) &  # Select Weak features with lower dimension
                (filtered_data["dim_value"] == 1))  # Match dim_value
        ]["bug_id"].iloc[0],  # Get bug_id for dim_value == 1

        "2": filtered_data[(
                (filtered_data["dimension"] == smallest_dimension) &  # Select Weak features with lower dimension
                (filtered_data["dim_value"] == 2))  # Match dim_value
        ]["bug_id"].iloc[0]  # Get bug_id for dim_value == 2
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

# Function to check which label does the weak2 feature correspond to
def weak2_code(Spreadsheet, bug, code_file):
    # Filter the code_file to contain only rows with Spreadsheet and "weak" features
    filtered_data = code_file[
        (code_file["Spreadsheet"] == Spreadsheet) &  # Match Spreadsheet
        (code_file["feature_type"] == "weak")  # Match feature_type
        ]

    # Weak2 is the weak feature with the LARGER dimension (out of 1,3,4)
    largest_dimension = filtered_data["dimension"].max()

    # Dictionary for getting bug_id for each dim_value (1,2)
    dim_val_to_bug_id = {
        "1": filtered_data[(
                (filtered_data["dimension"] == largest_dimension) &  # Select Weak features with larger dimension
                (filtered_data["dim_value"] == 1))  # Match dim_value
        ]["bug_id"].iloc[0],  # Get bug_id for dim_value == 1

        "2": filtered_data[(
                (filtered_data["dimension"] == largest_dimension) &  # Select Weak features with larger dimension
                (filtered_data["dim_value"] == 2))  # Match dim_value
        ]["bug_id"].iloc[0]  # Get bug_id for dim_value == 2
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


# Function to decode bug representation to standardised format (e.g., 21121 --> (1,0,0))
# The feature values correspond to the bug label they"re most predictive of: 1 --> 1, 0 --> 0
# Prototype item for label == 0 would be (0,0,0) and correspondingly for label == 1 (1,1,1)
def decode_features(Spreadsheet, bug, code_file):

    # Get which label each feature value is predictive of and subtract 0.5 (to get -0.5 and 0.5 feature values)
    strong_value = int(strong_code(Spreadsheet, bug, code_file)) /2
    weak1_value = int(weak1_code(Spreadsheet, bug, code_file)) /2
    weak2_value = int(weak2_code(Spreadsheet, bug, code_file)) /2

    # Make into tuple to allow for use as dict keys later on. Can also be converted to numpy array
    cleaned_bug = (strong_value, weak1_value, weak2_value)

    return cleaned_bug


# Create model with zero weight and bias initialisation. 3 units 1 output
# Mean squared error to correspond to linear regression. Stochastic GD optimiser
def create_model(learning_rate=0.01):
    # Define the model
    initializer = Zeros()  # Zero initialisation
    # build Sequential model with three inputs and one output
    model = keras.Sequential([
        layers.Dense(1, input_shape=(3,), kernel_initializer=initializer, bias_initializer=Zeros())
    ]) # NOTE: sigmoid might not be the correct answer (Now using linear activation!)

    # Configure the optimizer
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    return model


# Softmax function. Input dictionary of utilities (scores) for each bug permutation, temperature, and label.
# Output dictionary of softmax probabilities for each bug permutation. Permutations are keys in the dict.
def softmax(scores_dict, temperature, test_label):
    # Extract values and apply transformation if test_label is -1
    scores = np.array(list(scores_dict.values()))

    if test_label == -1:
        scores = - scores  # Invert scores if test_label is -1. This gives score for permutation being label == -1

    # Get exponent for scores divided by temperature
    # Stability improvement: subtract the max score to prevent overflow
    c = np.max(scores / temperature)
    exp_scores = np.exp((scores / temperature) - c)
    exp_scores = np.clip(exp_scores, 1e-10, 1e10)  # Clipping both lower and upper bounds

    # Get softmax probabilities
    softmax_probs = exp_scores / np.sum(exp_scores)
    # Create a new dictionary with the same keys but updated values
    softmax_dict = {key: prob for key, prob in zip(scores_dict.keys(), softmax_probs)}
    return softmax_dict


losses = []


# This function minimises the objective function (neg log likel) with a specified method.
# It then saves the best values and reruns the best model setup to get weights for each batch and saves these to
# a csv.
def optimiser(method, initial_lr, initial_temp, train_input, train_labels, test_label, targets, model, permutations,
              callback, weights, list_manager):
    def loss_callback(xk, convergence=None):
        current_loss = objective_function(xk, train_input, train_labels, targets, test_label, model, permutations,
                                          method, weights, list_manager)
        losses.append(current_loss)  # Append the current loss
    start_time = time.time()
    if method == "Nelder-Mead":
        res = optimize.minimize(
            objective_function,
            np.array([initial_lr, initial_temp]),
            args=(train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager),
            method="Nelder-Mead",
            options={"maxiter": 100},
            bounds=[(0.001, 5.0), (0.01, 10)]
        )
    elif method == "SLSQP":
        res = optimize.minimize(
            objective_function,
            np.array([initial_lr, initial_temp]),
            args=(train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager),
            method="SLSQP",
            options={"maxiter": 100},
            bounds=[(0.001, 5.0), (0.01, 10)]
        )

    elif method == "Differential Evolution":
        bounds = [(0.001, 2), (0.01, 10)]
        res = differential_evolution(
            objective_function,
            bounds,
            args=(train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager),
            strategy="best1bin",
            maxiter=100,
            popsize=50,
            tol=0.001,
            mutation=(0.6, 1.99),
            recombination=0.8,
            seed=None,
            callback=loss_callback,
            disp=False,
            polish=True,
            init="sobol"
        )
    elif method == "Dual Annealing":
        bounds = [(0.001, 5.0), (0.01, 10)]
        res = dual_annealing(
            objective_function,
            bounds,
            args=(train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager),
            no_local_search=False,
            maxfun=1000,
            maxiter=300,
            #visit= 2.2,
            #restart_temp_ratio= 0.01,
            #accept= -100
        )
    elif method == "Basin-hopping":
        res = optimize.basinhopping(
            objective_function,
            np.array([initial_lr, initial_temp]),
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "args": (train_input, train_labels, targets, test_label, model, permutations, method, weights),
                "bounds": [(0.001, 5.0), (0.01, 10)]
            }
        )


    # Get elapsed time for optimisation process
    elapsed_time = time.time() - start_time

    # Get best parameters and lowest neg log likelihood
    best_lr, best_temp = res.x
    best_nlgl = res.fun

    # Train model with best settings and callback activated to save weights on each batch (item)
    # Update the learning rate of the model"s existing optimizer
    tf.keras.backend.set_value(model.optimizer.learning_rate, best_lr)
    model.set_weights(weights)
    # Train model
    model.fit(train_input, train_labels, shuffle=False, batch_size=1, verbose=2,
              callbacks=[callback])

    # Save weights to csv
    callback.save_weights_to_csv(filepath="data/model_weights.csv")

    return best_nlgl, best_lr, best_temp, model, elapsed_time


# Keras callback class to save weights after each batch
class WeightSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, participant_id, block_num):
        super(WeightSaveCallback, self).__init__()
        self.participant_id = participant_id # Define participant id at initialisation
        self.block_num = block_num # Define block number at initialisation
        self.weights_data = []  # List to store weight data

    # Clear weight data at beginning of epoch. (Useful if using callback during optimisation process)
    def on_epoch_begin(self, epoch, logs=None):
        # Clear the weights data at the start of each epoch
        self.weights_data = []

    # Set participant id and block number
    def set_participant_block(self, participant_id, block_num):
        self.participant_id = participant_id
        self.block_num = block_num

    # On batch end fetch model weights and trial number. Append to weights_data
    def on_batch_end(self, batch, logs=None):
        # Extract weights and flatten them
        weights = self.model.get_weights()
        # Create a single row DataFrame from weights
        weights_row = {
            "Strong": weights[0][0][0], # First weight
            "Weak1": weights[0][1][0], # Second weight
            "Weak2": weights[0][2][0], # Third weight
            "Bias": weights[1][0], # Bias
            "Participant_ID": self.participant_id,
            "Block": self.block_num,
            "Batch": batch
        }

        # Append the row dictionary to the weights_data list
        self.weights_data.append(weights_row)

    # Save model weights to a csv file.
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
        # Set the internal list
        self._stored_list = new_list

    def get_list(self):
        # Return a copy of the list to prevent external modifications
        return self._stored_list.copy()


# Objective function to optimise. The model from the end of the previous block is trained on the next training sequence
# with a specified learning rate. This model is then used to pass through all 8 bug permutations. The outputs are
# processed with a softmax function and the participants first choice and probability are saved. Then the remaining 7
# permutations are processed with the softmax and the participants 2 choice and probability are saved etc. until all 4
# choice probabilities are acquired. The natural logarithms of these probabilities are summed together to get the total
# test trial log likelihood.
# The function returns the total negative log likelihood given the learning rate and temperature.
def objective_function(params, train_input, train_labels, targets, test_label, model, permutations, method, weights, list_manager):
    # Get learning rate and temperature
    learning_rate, temperature = params
    # Update the learning rate of the model"s existing optimizer
    tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    # Set model weights to the end of the previous block
    model.set_weights(weights)
    # Train model without shuffling
    model.fit(train_input, train_labels, shuffle=False, batch_size=1, verbose=0)

    # Get utilities for all 8 permutations from model
    utilities = model.predict(permutations).flatten()
    # Create dictionary with tuple keys and values from the arrays. Format e.g.: (0.5,-0.5,0.5): 0.67315,
    utilities_dict = {tuple(key): value for key, value in zip(permutations, utilities)}

    # initialise total log likelihood
    total_log_likelihood = 0

    # initialise probabilities list
    probabilities = []

    # Process each choice in a sequence
    for target in targets:
        # calculate softmax probabilities for the current set of choices
        probabilities_dict = softmax(utilities_dict, temperature, test_label)
        # Get the probability of the participant choice
        target_prob = probabilities_dict[target]

        # Append this choice probability to a list of probabilities.
        probabilities.append(target_prob)

        # Remove choice from utilities
        del utilities_dict[target]

        # Ensure target_prob is not zero or near zero. (remove -inf values from next line log function)
        target_prob = max(target_prob, 1e-10)

        # Update log likelihood with the log likelihood of the chosen target
        total_log_likelihood += np.log(target_prob)

    # Update probabilities list in ListManager
    list_manager.set_list(new_list=probabilities)

    print(temperature, learning_rate, total_log_likelihood)
    print(method)

    # Return negative log likelihood
    return -total_log_likelihood


def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', markersize=5, linestyle='-', color='b')
    plt.title('Optimization Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


# Main function
def main():
    # First we need to load in the participant data.
    test_data = pd.read_csv("data/test_data.csv")
    train_data = pd.read_csv("data/train_data.csv")

    # Exclude participants
    exclude_ids = pd.read_csv("../data/exclusions.csv")

    # Check Rows where id doesn't match with exclusion
    test_mask = ~test_data["id"].isin(exclude_ids["id"])
    train_mask = ~train_data["id"].isin(exclude_ids["id"])

    # Keep only rows where participant was not excluded
    test_data = test_data[test_mask]
    train_data = train_data[train_mask]

    # Read code file
    code_file = pd.read_csv("../data/fullcode.csv", delimiter=";")

    # Change labels from 1 and 2 to -0.5 and 0.5
    code_file["bug_id"] = code_file["bug_id"].astype("int") - 1.5

    # Decode original bug representations to standardised format. (e.g., 21211 --> (0.5,-0.5,0.5))
    train_data_modified_path = "data/train_data_standardised.csv"
    # Check if file (path) already exists
    # If not
    if not os.path.exists(train_data_modified_path):
        # Change train_data to decoded format [strong, weak1, weak2]
        train_data["input"] = train_data.progress_apply(lambda row: decode_features(row["Spreadsheet"], row["input"], code_file), axis=1)
        # Save results to csv
        train_data.to_csv(train_data_modified_path, index=False)
    # If yes
    else:
        # Read standardised train data file
        train_data = pd.read_csv(train_data_modified_path)
        # Make string representations of tuples back into tuples
        train_data["input"] = train_data["input"].apply(ast.literal_eval)

    # Create list of columns to decode
    test_cols = ["1", "2", "3", "4"]

    test_data_modified_path = "data/test_data_standardised.csv"
    # Check if file (path) already exists
    # If not
    if not os.path.exists(test_data_modified_path):
        # Decode each column
        for col in test_cols:
            test_data[col] = test_data.progress_apply(
                lambda row: decode_features(row["Spreadsheet"], row[col], code_file), axis=1)
        # Save results to csv
        test_data.to_csv(test_data_modified_path, index=False)
    # If yes
    else:
        # Read standardised test data file
        test_data = pd.read_csv(test_data_modified_path)
        # Make string representations of tuples back into tuples
        for col in test_cols:
            test_data[col] = test_data[col].apply(ast.literal_eval)


    # Create new column "targets" with a list of participant selections for each block
    test_data["targets"] = test_data.apply(lambda row: [row["1"], row["2"], row["3"], row["4"]], axis=1)


    # Get participant IDs in a list
    part_ids = train_data["id"].drop_duplicates().tolist()

    # Define all bug permutations for forward pass and softmax
    permutations = np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5]
])
    # Initialise Keras callback to save weights after each batch
    callback = WeightSaveCallback(part_ids[0], 0)

    # Initialise per block output dataframe (individual trial weights in separate object & file)
    data_df = pd.DataFrame(columns=["id", "block", "neg_logl", "learning_rate", "temperature", "strong", "weak1", "weak2"])

    # For each participant
    for id in part_ids:
        print(id)
        # First get the training data as a numpy array for given participant
        id_train_data = train_data[train_data["id"] == id]

        # Group by block and collect all inputs
        grouped_input = id_train_data.groupby("block")["input"].apply(list).tolist()
        grouped_label = id_train_data.groupby("block")["label"].apply(list).tolist()

        # Convert the list of lists (for each block) into a numpy array
        input_array = [np.array(block) for block in grouped_input]
        label_array = [np.array(block) for block in grouped_label]
        # Recoding 0 values to -1 in each array
        label_array = [np.where(block == 0, -1, block) for block in label_array]

        # Get the test data for a given participant
        id_test_data = test_data[test_data["id"] == id]
        target_array = id_test_data["targets"].tolist()
        test_labels = id_test_data["label"].tolist()
        # Recoding 0 values to -1 in the list
        test_labels = [-1 if label == 0 else label for label in test_labels]

        # Create model for participant
        model = create_model()

        # Process each training block for participant
        for block in range(0,10):
            tr.print_diff()
            if block == 0:
                # Get zero initialised weights if first block
                best_weights = model.get_weights()

            # Define training inputs and labels, targets (participant test trial choices), and the test label (0 or 1)
            inputs = input_array[block]
            labels = label_array[block]
            targets = target_array[block]
            test_label = test_labels[block]
            global losses
            losses = []

            # Set participant ID and block for weight save callback
            callback.set_participant_block(id, block)

            # Initialise ListManager class for handling list of choice probabilities between functions
            list_manager = ListManager()

            # E.g., Nelder-Mead seems to go for 0 learning rate after a few blocks so let"s try randomly initialising
            # learning rate for each block. Same for temperature
            learning_rate = random.uniform(0.001, 2)
            temperature = random.uniform(0.01, 10)

            # Define method/methods to be used in optimising objective function. Results are stored for the last
            # method in the list. Including multiple methods is mainly for comparing compute times and effectiveness
            # across methods.
            methods = ["Differential Evolution"] # "Nelder-Mead" quicker but less reliable

            # Define empty list for storing optimisation results
            results = []

            # Run optimisation process for each method. Output best parameters, nlgl, best model, and computation time
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
                # Append optimisation results to empty list
                results.append((method, best_nlgl, best_lr, best_temp, comp_time))
            print(results)
            #plot_loss_curve(losses)
            # Get best weights from model. Used to initialise models for next block.
            best_weights = best_model.get_weights()

            # Get human choice softmax probabilties from ListManager class
            probabilities = list_manager.get_list()

            # Define data row for the current participant and block
            new_data = pd.DataFrame({
                "id": id,
                "block": block,
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

            # Concat row to per-block dataframe
            data_df = pd.concat([data_df, new_data], ignore_index=True)

            # Update model (redundant?)
            model = best_model
            print(data_df)

        print(data_df)

        gc.collect()

        # Save model fit results to csv after each participant has been processed
        data_df.to_csv("data/model_fit.csv")


if __name__ == "__main__":
    main()
