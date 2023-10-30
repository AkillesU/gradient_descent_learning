import ast
import pandas as pd
import os


def process_dataframe(df):
    # Filter for rows where "Display" is "test"
    df_test = df[df["Display"] == "test"].copy()

    # Find the last occurrence for each "Trial Number" from 1-10
    relevant_rows = []
    for i in range(1, 11):  # For trial numbers 1 to 10
        last_occurrence = df_test[df_test["Trial Number"] == i].tail(1)
        relevant_rows.append(last_occurrence)

    # Concatenate the rows into a single DataFrame
    df_filtered = pd.concat(relevant_rows)

    # Initialize an empty list to collect rows for the new DataFrame
    new_rows = []

    for idx, row in df_filtered.iterrows():
        participant_id = row['Participant Private ID']
        trial_num = row['Trial Number']
        spreadsheet = row['Spreadsheet']
        bug_id = row['Spreadsheet: bug_id']

        # Convert the string representation of list to actual list
        file_list = ast.literal_eval(row['Response'])

        # Extract the number sequences from each filename
        number_sequences = [file.split('.')[0] for file in file_list]

        # Pad the list with None values if it has less than 4 entries
        number_sequences += [None] * (4 - len(number_sequences))

        # Create a new row with the participant ID, trial number, the number sequences,
        # and values for Spreadsheet and bug_id
        new_row = [participant_id, trial_num] + number_sequences + [spreadsheet, bug_id]
        new_rows.append(new_row)

    # Create a new DataFrame from the collected rows
    new_df = pd.DataFrame(new_rows, columns=['id', 'trial', '1', '2', '3', '4', 'Spreadsheet', 'bug_id'])

    return new_df


# Main code
folder_path = 'pilotdata/'
all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

master_df = pd.DataFrame()

for file in all_files:
    df = pd.read_csv(file)
    processed_df = process_dataframe(df)
    master_df = pd.concat([master_df, processed_df], ignore_index=True)

print(master_df)

master_df.to_csv('cleaned_pilotdata.csv', index=False)
