import pandas as pd
import matplotlib.pyplot as plt

"""
This script creates a plot showing what proportion of participants selected the 
"prototype" bug across trials. The prototype bug refers to the bug permutation
with all features predicting the queried bug category.
"""

# Read participant data
part_data = pd.read_csv("../data/cleaned_data.csv", delimiter=",")
# Read code file
code_file = pd.read_csv("../data/fullcode.csv", delimiter=";")

df_exclude = pd.read_csv("../data/exclusions.csv")

# Exclude participants
part_data = part_data[~part_data['id'].isin(df_exclude['id'])].reset_index()

n_participants = int(len(part_data) / 10)


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
    stim = str(stim)
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
    n_participants = len(part_data) / 10 # Calculate number of participants

    # Define dataframe to store values in
    proto_df = pd.DataFrame(columns=["trial", "prototype"])
    proto_df["trial"] = range(1, 11)
    proto_df["prototype"] = 0
    print(proto_df)

    for row in range(0,int(len(part_data))):

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
                trial = int(part_data["trial"][row])  # Get trial number
                proto_df["prototype"][trial-1] += 1  # Add 1 to trial count

                print("MATCH")

        print("Results: ", proto_df)
    proto_df["prototype"] = proto_df["prototype"]/n_participants

    return proto_df
results = check_match()

print(results)

plt.figure(figsize=(10, 5))  # Set the figure size as desired
plt.plot(results['trial'], results['prototype'], marker='o')  # 'o' for circle markers

# Set the title and labels
plt.title('Proportion of Prototype Bug Choices Across Trials')
plt.xlabel('Trial')
plt.ylabel('Prototype Accuracy')

# Save figure
plt.savefig(f"images/participants_{n_participants}/prototype_plot_part{n_participants}.png")

# Show the plot
plt.show()
