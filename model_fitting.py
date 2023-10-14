import numpy as np
import pandas as pd
from scipy import optimize

# Read participant data
part_data = pd.read_csv("data/cleaned_pilotdata.csv", delimiter=";")
# Read code file
code_file = pd.read_csv("data/fullcode.csv", delimiter=";")

# Creating list of participant IDs. Order is maintained.
part_ids = part_data['id'].drop_duplicates().tolist()



"""
This function is correct. Use this as a template for the other use_"Strategy" functions
"""
def use_strong(Spreadsheet, stim):
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
            ]['bug_id'].iloc[0] # Get bug_id for dim_value == 1
    }

    strong_dimension = code_file[
        (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
        (code_file['feature_type'] == 'strong') # Match feature_type == 'strong'
        ]['dimension'].iloc[0] -1 # Get dimension (- 1 to get indexer)
    stim = str(stim)
    # Get bug_id from participant stim, given strategy == 'strong'
    true_label = dim_val_to_bug_id[stim[(strong_dimension)]]

    return true_label


def use_weak1(Spreadsheet, stim):

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
                (filtered_data['dimension'] == smallest_dimension) & # Select Weak features with lower dimension
                (filtered_data['dim_value'] == 1)) # Match dim_value
                ]['bug_id'].iloc[0],  # Get bug_id for dim_value == 1

        '2': filtered_data[(
                (filtered_data['dimension'] == smallest_dimension) & # Select Weak features with lower dimension
                (filtered_data['dim_value'] == 2)) # Match dim_value
                ]['bug_id'].iloc[0] # Get bug_id for dim_value == 2
    }

    # Setting weak1_dimension based on the smaller dimension (- 1 to get indexer)
    weak1_dimension = smallest_dimension - 1
    stim = str(stim)
    # Get bug_id for participant stim based on strategy 'weak1'
    true_label = dim_val_to_bug_id[stim[(weak1_dimension)]]

    return true_label


def use_weak2(Spreadsheet, stim):
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
                (filtered_data['dimension'] == largest_dimension) & # Select Weak features with larger dimension
                (filtered_data['dim_value'] == 1)) # Match dim_value
                ]['bug_id'].iloc[0], # Get bug_id for dim_value == 1

        '2': filtered_data[(
                (filtered_data['dimension'] == largest_dimension) & # Select Weak features with larger dimension
                (filtered_data['dim_value'] == 2)) # Match dim_value
                ]['bug_id'].iloc[0] # Get bug_id for dim_value == 2
    }

    # Setting weak2_dimension based on the larger dimension (- 1 to get indexer)
    weak2_dimension = largest_dimension -1
    stim = str(stim)
    # Get bug_id for participant stim based on strategy 'weak2'
    true_label = dim_val_to_bug_id[stim[(weak2_dimension)]]

    return true_label

"""
For the prototype strategy we check which label each dimension gives out.
If two or more dimensions point towards a bug_id, the true_label will be that bug_id.
"""
def use_prototype(Spreadsheet, stim):

    # Dicitionary for getting bug_id for each dim_value in the first dimension
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

    # Dicitionary for getting bug_id for each dim_value in the third dimension
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

    # Dicitionary for getting bug_id for each dim_value in the fourth dimension
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
    votes.append(first_dim[stim[0]]) # Get bug_id from first dimension
    votes.append(third_dim[stim[2]]) # Get bug_id from third dimension
    votes.append(fourth_dim[stim[3]]) # Get bug_id from fourht dimension

    n_bug_id_1 = 0
    n_bug_id_2 = 0
    for n in votes:
        if n == 1:
            n_bug_id_1 += 1
        elif n == 2:
            n_bug_id_2 += 1

    if n_bug_id_1 > n_bug_id_2:
        true_label = 1
    else:
        true_label = 2

    return true_label


# Read which bug_id a participant was choosing bugs for on a given trial
def read_participant_choice(pid, trial):

    part_label = part_data[((part_data['id'] == pid) & # Match participant ID
                            (part_data['trial'] == trial)) # Match trial number
                            ]['bug_id'].iloc[0] # Get bug_id of that test trial
    return part_label


# Funtion to extract Spreadsheet based on participant ID.
def map_participant_to_spreadsheet(pid):
    # Find the first row with the correct pid
    row = part_data[part_data['id'] == pid].iloc[0]

    # Get the corresponding value from Spreadsheet column
    Spreadsheet = row['Spreadsheet']

    return Spreadsheet



def main():
    n_trials = 11  # Change to 10
    strategies = ['strong', 'weak1', 'weak2', 'prototype']
    # Creating empty results dataframe
    columns = ['id',
               'trial',
               'best_strategy',
               'best_alpha',
               'best_nlogl',
               'strong_alpha',
               'strong_nlogl',
               'weak1_alpha',
               'weak1_nlogl',
               'weak2_alpha',
               'weak2_nlogl',
               'proto_alpha',
               'proto_nlogl'
               ]
    results = pd.DataFrame(columns=columns)

    for pid in part_ids:
        print("Participant", pid)
        Spreadsheet = map_participant_to_spreadsheet(pid)

        best_strategy_across_trials = []
        best_strategy_alphas_across_trials = []
        best_strategy_neg_log_likelihood_across_trials = []

        for trial in range(1,n_trials):
            print("Trial:", trial)
            best_strategy, best_alpha, best_neg_log_likelihood = optimizer(strategies, Spreadsheet, trial, pid)

            best_strategy_across_trials.append(best_strategy)
            best_strategy_alphas_across_trials.append(best_alpha)
            best_strategy_neg_log_likelihood_across_trials.append(best_neg_log_likelihood)
            print(best_strategy_across_trials)

            # Save results.
            new_data = pd.DataFrame({ # TODO: Add each strategies likelihood and alpha per trial
                'id': [pid],
                'trial': [trial],
                'strategy': [best_strategy],
                'alpha': [round(float(best_alpha),4)],
                'nlog_likel': [best_neg_log_likelihood]
            })
            # Concat new row to results dataframe
            results = pd.concat([results,new_data], ignore_index=True)
            print(results)

    results.to_csv('results/model_fit_results.csv', index=False)

def optimizer(strategies, Spreadsheet, trial, pid):
    best_strategy = ''
    best_alpha = []
    min_neg_log_likelihood = np.inf
    for strategy in strategies:
        alpha = 0.5 # initial guess
        # minimize negative log likelihood
        res = optimize.minimize(
            joint_likelihood,
            alpha,
            args=(strategy, Spreadsheet, trial, pid), # I think "alpha" is not included here
            method="Nelder-Mead",
            options= {'maxiter': 100}
        )
        if res.fun < min_neg_log_likelihood:
            best_strategy = strategy
            best_alpha = res.x
            min_neg_log_likelihood = res.fun


    print(
        f'Best strategy: {best_strategy}, \n'
        f'Best alpha: {best_alpha[0]:.2f} \n'
        f'Best neg log likelihood: {min_neg_log_likelihood:.2f}'
        )
    return best_strategy, best_alpha, min_neg_log_likelihood

"""
This function is used to check whether participants used the prototype
strategy. If 4 or more characters match in the participants selected stimulus
and the prototype label, this means that at least 2/3 of the dimensions match
the prototype label. (e.g. 21221 and 11221 (2/3 matching))
"""


def joint_likelihood(alpha, strategy, Spreadsheet, trial, pid):
    """
    For a strategy, an alpha, compute the joint likelihood of
    the entire set of stimuli within a trial.
    """
    p = 1  # initialize joint likelihood

    # Full set of bug permutations: 8 bugs
    entire_set = ['11121', '21111', '21221', '21121',
                  '11221', '11111', '21211', '11211']

    """
    Make a list of the four selections based on participant ID and trial number.
    ['1', '2', '3', '4'] denote the four choices made.
    """
    stimuli_selected = part_data[(part_data['id'] == pid) &
                                 (part_data['trial'] == trial)
                                ].iloc[0][['1', '2', '3', '4']].tolist()

    for stim in stimuli_selected:  # 4 bugs [21111, 21222, 12222, 2...]
        if strategy == 'strong':
            true_label = use_strong(Spreadsheet, stim)
        elif strategy == 'weak1':
            true_label = use_weak1(Spreadsheet, stim)
        elif strategy == 'weak2':
            true_label = use_weak2(Spreadsheet, stim)
        elif strategy == 'prototype':
            true_label = use_prototype(Spreadsheet, stim)
        else:
            print("Strategy not set")
        part_label = read_participant_choice(pid, trial)

        if part_label == true_label:
            likelihood = alpha + (1 - alpha) * 0.5
        elif part_label != true_label:
            likelihood = (1 - alpha) * 0.5

        p *= likelihood

    # Switch part_label (bug_id) for unselected stimuli
    if part_label == 1:
        part_label = 2
    else:
        part_label = 1

    stimuli_unselected = [item for item in entire_set if item not in stimuli_selected]
    for stim in stimuli_unselected:  # 4 bugs [12222, ]
        if strategy == 'strong':
            true_label = use_strong(Spreadsheet, stim)
        elif strategy == 'weak1':
            true_label = use_weak1(Spreadsheet, stim)
        elif strategy == 'weak2':
            true_label = use_weak2(Spreadsheet, stim)
        elif strategy == 'prototype':
            true_label = use_prototype(Spreadsheet, stim)
        else:
            print("Strategy not set")

        if part_label == true_label:
            likelihood = alpha + (1 - alpha) * 0.5
        elif part_label != true_label:
            likelihood = (1 - alpha) * 0.5


        p *= likelihood

    # return negative log likelihood
    return -np.log(p)

if __name__ == '__main__':
    main()

