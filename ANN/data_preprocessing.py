import os
import pandas as pd
import ast


# Process the study test trials
def process_test_df(df):
    # Filter for rows where "Display" column is "test"
    df_test = df[df["Display"] == "test"].copy()
    # Filter for test rows where the participant reacted with the continue button
    df_test = df_test[df["Object Name"] == "Continue Button"]
    # Filter out responses where the reaction didn"t advance the screen
    df_test = df_test[df["Response"] != "(Continue)"]
    # Find the occurrence for each "Trial Number" from 1-10
    relevant_rows = []
    for i in range(1, 11):  # For trial numbers 1 to 10
        occurrences = df_test[df_test["Trial Number"] == i]
        relevant_rows.append(occurrences)
    # Concatenate the rows into a single DataFrame
    df_filtered = pd.concat(relevant_rows)

    # Initialize an empty list to collect rows for the new DataFrame
    new_rows = []

    # Process each row
    for idx, row in df_filtered.iterrows():
        participant_id = row["Participant Private ID"]
        trial_num = row["Trial Number"]
        spreadsheet = row["Spreadsheet"]
        bug_id = int(row["Spreadsheet: bug_id"]) -1 # labels as 0 and 1 instead of 1 and 2

        # Convert the string representation of list to actual list
        file_list = ast.literal_eval(row["Response"])

        # Extract the number sequences from each filename (e.g., "21221.jpg")
        number_sequences = [file.split(".")[0] for file in file_list]

        # Pad the list with None values if it has less than 4 entries
        number_sequences += [None] * (4 - len(number_sequences))

        # Create a new row with the participant ID, trial number, the number sequences,
        # and values for Spreadsheet and bug_id
        new_row = [participant_id, trial_num] + number_sequences + [spreadsheet, bug_id]
        new_rows.append(new_row)

    # Create a new DataFrame from the collected rows
    new_df = pd.DataFrame(new_rows, columns=["id", "block", "1", "2", "3", "4", "Spreadsheet", "label"])
    return new_df


# Process training data
def process_train_df(df):
    # Filter to contain only training trials
    df_train = df[df["Display"] == "bug_images"]

    # Keep only one copy of each training trial
    df_train = df_train.drop_duplicates(subset=["Trial Number",
                                                "Participant Private ID"],
                                        ignore_index=True)

    # Extract the input and label for each training trial into new columns
    df_train["label"] = df_train["Spreadsheet: bug_id"].str[-1]
    df_train["label"] = df_train["label"].astype("int") -1 # making labels 0 and 1
    df_train["input"] = df_train["Spreadsheet: bug_id"].str[:-2]

    # Keeping only relevant columns
    cols_to_keep = ["Participant Private ID", "Trial Number", "input", "label", "Spreadsheet"]
    df_final = df_train[cols_to_keep]

    # Renaming columns
    rename_dict = {"Participant Private ID": "id",
                   "Trial Number": "trial"}
    df_final.rename(columns=rename_dict, inplace=True)

    # Creating new column "Block" to increment every 14 trials
    df_final["block"] = ((df_final["trial"] - 1) // 14) + 1

    return df_final


def main():
    # Folder path for all participant data
    folder_path = "data/experiment_data"
    # Get all file paths
    all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

    # initialise empty dataframe
    test_df = pd.DataFrame()
    # Process each file for test data
    for file in all_files:
        print(file)
        df = pd.read_csv(file)
        processed_df = process_test_df(df)
        test_df = pd.concat([test_df, processed_df], ignore_index=True)

    # Initialise empty dataframe
    train_df = pd.DataFrame()
    # Process each file for training data
    for file in all_files:
        print(file)
        df = pd.read_csv(file)
        processed_df = process_train_df(df)
        train_df = pd.concat([train_df, processed_df], ignore_index=True)


    # Order based on id and trial
    test_df = test_df.sort_values(by=["id", "block"], ascending=[True, True], ignore_index=True)
    print(test_df)

    # For this experiment run it seems like 10937464.0 got 17 trials somehow. Will delete participant
    test_df = test_df[test_df["id"] != 10937464.0]
    print(test_df)

    # Save training and test data to csv
    train_df.to_csv("data/train_data.csv", index=False)
    test_df.to_csv("data/test_data.csv", index=False)


if __name__ == "__main__":
    main()
