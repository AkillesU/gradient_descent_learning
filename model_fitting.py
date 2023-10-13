import numpy as np
import pandas as pd
from scipy import optimize

# Read participant data
part_data = pd.read_csv("/data/cleaned_pilotdata.csv", delimiter=";")
# Read code file
code_file = pd.read_csv("/data/fullcode.csv", delimiter = ";")


# Creating list of participant IDs. Order is maintained.
part_ids = part_data['id'].drop_duplicates().tolist()


# Read participant choice for given strategy.
def read_participant_choice(Spreadsheet, stim, strategy):

    # Get the dimension (1,3,4) based on Spreadsheet and feature_type: "strong"
    if strategy == 'strong':
        dimension = code_file[(code_file['Spreadsheet'] == Spreadsheet) &
                              (code_file['feature_type'] == 'strong')][
            'dimension'].iloc[0]
        dimension -= 1 # Correct for indexing

    # Get the dimension (1,3,4) based on Spreadsheet and feature_type: "weak"
    elif strategy == 'weak1':
        dimension = code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &
            (code_file['feature_type'] == 'weak')
            ]['dimension'].iloc[0] # dimension of FIRST "weak" feature
        dimension -= 1 # Correct for indexing

    # Get the dimension (1,3,4) based on Spreadsheet and feature_type: "weak"
    elif strategy == 'weak2':
        dimension = code_file[
            (code_file['Spreadsheet'] == Spreadsheet) &
            (code_file['feature_type'] == 'weak')
            ]['dimension'].iloc[1] # dimension of SECOND "weak" feature
        dimension -= 1 # Correct for indexing

    # Returns the entire string if strategy is prototype
    elif strategy == 'prototype':
        dimension = slice(None)

    # Get part_label from correct dimension (index) based on strategy
    part_label = stim[dimension]
    return part_label


# Funtion to extract Spreadsheet based on participant ID.
def map_participant_to_spreadsheet(pid):
    # Find the first row with the correct pid
    row = part_data[part_data['id'] == pid].iloc[0]

    # Get the corresponding value from Spreadsheet column
    Spreadsheet = row['Spreadsheet']

    return Spreadsheet

part_ids = part_ids[:1]
def main():
  n_trials = 1 # Change to 10
  strategies = ['strong', 'weak1', 'weak2', 'prototype']
  for pid in part_ids:
    Spreadsheet = map_participant_to_spreadsheet(pid)

    best_strategy_across_trials = []
    best_strategy_alphas_across_trials = []
    best_strategy_neg_log_likelihood_across_trials = []

    for trial in range(n_trials):
      best_strategy, best_alpha, best_neg_log_likelihood = optimizer(strategies, Spreadsheet, trial,pid)

      best_strategy_across_trials.append(best_strategy)
      best_strategy_alphas_across_trials.append(best_alpha)
      best_strategy_neg_log_likelihood_across_trials.append(best_neg_log_likelihood)

    # Save results.


def optimizer(strategies, Spreadsheet, trial, pid):
  best_strategy = ''
  best_alpha = 0
  min_neg_log_likelihood = np.inf
  for strategy in strategies:
    alpha = 0.5  # initial guess

    # minimize negative log likelihood
    res = optimize.minimize(
        joint_likelihood,
        alpha,
        args=(strategy, alpha, Spreadsheet, trial, pid),
        method="Nelder-Mead"
    )
    if res.fun < min_neg_log_likelihood:
        best_strategy = strategy
        best_alpha = res.x
        min_neg_log_likelihood = res.fun

  print(
      f'Best strategy: {best_strategy}, \n'\
      f'Best alpha: {best_alpha[0]:.2f} \n'\
      f'Best neg log likelihood: {min_neg_log_likelihood:.2f}'
  )
  return best_strategy, best_alpha, min_neg_log_likelihood


"""
This function is used to check whether participants used the prototype
strategy. If 4 or more characters match in the participants selected stimulus
and the prototype label, this means that at least 2/3 of the dimensions match
the prototype label. (e.g. 21221 and 11221 (2/3 matching))
"""
def four_or_more_match(s1, s2):

    count = 0
    for i in range(5):
        if s1[i] == s2[i]:
            count += 1

    return count >= 4


def joint_likelihood(strategy, alpha, Spreadsheet, trial, bug_id, pid):
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
      true_label = use_strong(Spreadsheet, bug_id)
    elif strategy == 'weak1':
      true_label = use_weak1(Spreadsheet, bug_id)
    elif strategy == 'weak2':
      true_label = use_weak2(Spreadsheet, bug_id)
    elif strategy == 'prototype':
      true_label = use_prototype(Spreadsheet, bug_id)

    part_label = read_participant_choice(Spreadsheet, stim, strategy)

    """
    In the following section, for strategies (strong, weak1, weak2) if the
    participant label does not match the true label for those strategies
    the likelihood is: (1-alpha) * 0.5 (guessing), if it matches:
    alpha + (1-alpha) * 0.5 (use strategy + guessing). For the PROTOTYPE strategy
    if 2 or more dimensions match between the participant choice and the
    prototype label likelihood = (use strategy + guessing). If less than 2/3
    dimensions match, likelihood = (guessing).
    """
    if strategy != 'prototype' & part_label == true_label:
      likelihood = alpha + (1-alpha) * 0.5
    elif strategy != 'prototype' & part_label != true_label:
      likelihood = (1-alpha) * 0.5
    elif strategy == 'prototype' & four_or_more_match(part_label, true_label):
      likelihood = alpha + (1-alpha) * 0.5
    elif strategy == 'prototype' and not four_or_more_match(part_label, true_label):
      likelihood = (1-alpha) * 0.5

    p *= likelihood

  # Switch bug_id for unselected stimuli
  if bug_id == 1:
    bug_id = 2
  else:
    bug_id = 1


  stimuli_unselected = entire_set[~stimuli_selected] # TODO: ken check correct syntax
  for stim in stimuli_unselected: # 4 bugs [12222, ]
    if strategy == 'strong':
      true_label = use_strong(Spreadsheet, bug_id)
    elif strategy == 'weak1':
      true_label = use_weak1(Spreadsheet, bug_id)
    elif strategy == 'weak2':
      true_label = use_weak2(Spreadsheet, bug_id)
    elif strategy == 'prototype':
      true_label = use_prototype(Spreadsheet, bug_id)

    part_label = read_participant_choice(Spreadsheet, stim, strategy)

    if strategy != 'prototype' & part_label == true_label:
      likelihood = alpha + (1-alpha) * 0.5
    elif strategy != 'prototype' & part_label != true_label:
      likelihood = (1-alpha) * 0.5
    elif strategy == 'prototype' & four_or_more_match(part_label, true_label):
      likelihood = alpha + (1-alpha) * 0.5
    elif strategy == 'prototype' and not four_or_more_match(part_label, true_label):
      likelihood = (1-alpha) * 0.5

    p *= likelihood

  # return negative log likelihood
  return -np.log(p)


# Get true_label based on Spreadsheet and bug_id
def use_strong(Spreadsheet, bug_id):

  # Find dim_value for strong bug_id
  true_label = code_file[
      (code_file['Spreadsheet'] == Spreadsheet) & # Match Spreadsheet
      (code_file['feature_type'] == "strong") & # Match dimension.
      (code_file['bug_id'] == int(bug_id)) # Match bug_id
      ]['dim_value'].iloc[0] # Get dim_value. iloc gets the first element in the series of len == 1

  return true_label




"""
Weak1 is defined as the first matching value (iloc[0])
from two matches in the use_weak functions. There are
two matches because the two weak strategies are not
differentiated in the code_file.
Weak2 is the second matching value.
"""
# Get the true label for weak1 strategy by Spreadsheet and bug_id
def use_weak1(Spreadsheet, bug_id):

  # Find dim_value for strong bug_id
  true_label = code_file[
      (code_file['Spreadsheet'] == Spreadsheet) & # Match Spreadsheet
      (code_file['feature_type'] == "weak") & # Match dimension.
      (code_file['bug_id'] == int(bug_id)) # Match bug_id
      ]['dim_value'].iloc[0] # Get dim_value. FIRST element in the series.

  return true_label


# Get the true label for weak2 strategy by Spreadsheet and bug_id
def use_weak2(Spreadsheet, bug_id):

  # Find dim_value for strong bug_id
  true_label = code_file[
      (code_file['Spreadsheet'] == Spreadsheet) & # Match Spreadsheet
      (code_file['feature_type'] == "weak") & # Match dimension.
      (code_file['bug_id'] == int(bug_id)) # Match bug_id
      ]['dim_value'].iloc[1] # Get dim_value. SECOND element in the series.

  return true_label

# Outputs the prototype bug (e.g. true_label == "11111")
def use_prototype(Spreadsheet, bug_id):

    # Get the correct label for dimension == 1
    true_first_dim_label = code_file[
        (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
        (code_file['dimension'] == 1) &  # Match dimension.
        (code_file['bug_id'] == int(bug_id))  # Match bug_id
        ]['dim_value'].iloc[0]  # Get dim_value.

    # Get the correct label for dimension == 3
    true_third_dim_label = code_file[
        (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
        (code_file['dimension'] == 3) &  # Match dimension.
        (code_file['bug_id'] == int(bug_id))  # Match bug_id
        ]['dim_value'].iloc[0]  # Get dim_value.

    # Get the correct label for dimension == 4
    true_fourth_dim_label = code_file[
        (code_file['Spreadsheet'] == Spreadsheet) &  # Match Spreadsheet
        (code_file['dimension'] == 4) &  # Match dimension.
        (code_file['bug_id'] == int(bug_id))  # Match bug_id
        ]['dim_value'].iloc[0]  # Get dim_value.

    true_label = (str(true_first_dim_label) +
                  '1' +
                  str(true_third_dim_label) +
                  str(true_fourth_dim_label) +
                  '1')

    return true_label


if __name__ == '__main__':
  main()