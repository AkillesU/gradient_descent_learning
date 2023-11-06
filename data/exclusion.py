import pandas as pd

"""
This script creates a list of participant IDs to be excluded based on two criteria:

1. The alpha value for the most likely strategy is 0 on trial 10
2. The participant did not select the prototype bug on trial 10
"""


# Read participant data
part_data = pd.read_csv("cleaned_pilotdata.csv", delimiter=",")
# Read code file
code_file = pd.read_csv("fullcode.csv", delimiter=";")


# This function returns if a participant selection was a prototype bug for bug_id 1 or bug_id 2
# If not, it returns None
def use_prototype(Spreadsheet, stim):
    # Dictionary for getting bug_id for each dim_value in the first dimension
    first_dim = {
        '1': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['dimension'] == 1) &  # Match dimension
            (code_file['dim_value'] == 1)  # Match dim_value
            ]['bug_id'].iloc[0],  # Get bug_id for dimension == 1, dim_value == 1

        '2': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['dimension'] == 1) &  # Match dimension
            (code_file['dim_value'] == 2)  # Match dim_value
            ]['bug_id'].iloc[0]  # Get bug_id for dimension == 1, dim_value == 2
    }

    # Dictionary for getting bug_id for each dim_value in the third dimension
    third_dim = {
        '1': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['dimension'] == 3) &  # Match dimension
            (code_file['dim_value'] == 1)  # Match dim_value
            ]['bug_id'].iloc[0],  # Get bug_id for dimension == 3, dim_value == 1

        '2': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['dimension'] == 3) &  # Match dimension
            (code_file['dim_value'] == 2)  # Match dim_value
            ]['bug_id'].iloc[0]  # Get bug_id for dimension == 3, dim_value == 2
    }

    # Dictionary for getting bug_id for each dim_value in the fourth dimension
    fourth_dim = {
        '1': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['dimension'] == 4) &  # Match dimension
            (code_file['dim_value'] == 1)  # Match dim_value
            ]['bug_id'].iloc[0],  # Get bug_id for dimension == 4, dim_value == 1

        '2': code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
            (code_file['dimension'] == 4) &  # Match dimension
            (code_file['dim_value'] == 2)  # Match dim_value
            ]['bug_id'].iloc[0]  # Get bug_id for dimension == 4, dim_value == 2
    }

    # Making sure stimulus is a string
    stim = str(stim)
    # Initialising votes list
    votes = []
    votes.append(first_dim[stim[0]])  # Get bug_id from first dimension
    votes.append(third_dim[stim[2]])  # Get bug_id from third dimension
    votes.append(fourth_dim[stim[3]])  # Get bug_id from fourth dimension

    n_bug_id_1 = 0
    n_bug_id_2 = 0

    for n in votes: # Count how many features point towards each bug_id
        if n == 1:
            n_bug_id_1 += 1
        elif n == 2:
            n_bug_id_2 += 1

    if n_bug_id_1 == 3: # If all three features point towards bug_id 1
        true_label = 1
    elif n_bug_id_2 == 3: # If all three features point towards bug_id 2
        true_label = 2
    else:
        true_label = None

    return true_label

# This function checks whether a participant selected the prototype bug for each trial
def check_match():

    # Define dataframe to store values in
    proto_df = pd.DataFrame(columns=["id","trial", "prototype"])
    proto_df["id"] = part_data["id"].unique()  # Get all participant IDs
    proto_df["trial"] = 10  # Set trial as 10
    proto_df["prototype"] = 0
    print(proto_df)

    for row in range(9,int(len(part_data)), 10):  # Check every 10th row, starting from 10.

        # Print the current participant ID and trial number
        print(f"Participant {part_data['id'][row]} trial {part_data['trial'][row]}")

        # Setting participant spreadsheet to allow for decoding of dimensions
        Spreadsheet = part_data["Spreadsheet"][row]

        # Setting the bug_id the participant was choosing for
        part_label = int(part_data["bug_id"][row])

        # Iterate over all participant choices in trial
        for stim in ["1", "2", "3", "4"]:

            # Set stimulus
            stimulus = part_data[stim][row]

            # Check whether participant selected a prototype bug for bug_id == 1 or bug_id == 2
            true_label = use_prototype(Spreadsheet, stimulus) # Can take values bug_id = (1, 2, None)

            # If participant label matches true_label, add 1 to prototype column for current trial
            if part_label == true_label:
                participant = part_data["id"][row] # Get participant id

                # Set prototype column value as 1, if stimulus is prototype bug
                proto_df.loc[proto_df['id'] == participant, "prototype"] += 1

                print("MATCH")

        print("Results: ", proto_df)

    return proto_df

# Create exclusion dataframe. Prototype == 1 (Participant selected prototype bug). Prototype == 0 (no prototype)
exclusions1 = check_match()

# Filter out participants who chose prototype bug on trial 10
exclusions1 = exclusions1[exclusions1['prototype'] != 1]

print(exclusions1)

"""
Now checking for alpha values in trial 10. 
If alpha = 0 (guessing) in trial 10, participant is added to the exclusion list 
"""

# Read likelihood model output
model_output = pd.read_csv("../likelihood_model/likel_results/model_fit_results.csv", delimiter=",")

# Filter trial 10
model_output = model_output[model_output["trial"] == 10]
print(model_output)

# Check if "best_alpha" was 0 on trial 10
exclusions2 = model_output[model_output["best_alpha"] == 0]

print(exclusions2)

# Combine exclusions from both criteria
exclusions_total = pd.concat([exclusions2["id"], exclusions1["id"]])

# Delete duplicates and set column name to "id"
exclusions_total = pd.DataFrame(exclusions_total.unique(), columns=["id"])

print(exclusions_total)

# Save exclusions as a .csv
exclusions_total.to_csv("exclusions.csv", index=False)