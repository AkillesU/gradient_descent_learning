# gradient_descent_learning

This is the code for analysing the data from a rerun of https://osf.io/preprints/psyarxiv/75e4t/ .

The code contains:
- Initial data wrangling
- A maximum likelihood model that outputs the strategy a participant was using for a trial
- A Hidden Markov Model that infers hidden states based on the output of the maximum likelihood model

## Data details `/data` 

This directory contains the initial data and preprocessing pipeline.

`datawrangle.py`: Collects all participant responses into a single csv file

`fullcode.csv`: Code file for decoding randomisation

`cleaned_pilotdata.csv`: output from `datawrangle.py`. Contains participant trials along with randomisation coding.

`/pilotdata`: contains initial raw data from 10 participants.

---

## Likelihood Model `/likelihood_model`

`model_fitting.py`

This model finds the optimal strategy for a given participant trial.

Each participant has four observations per trial (e.g., [21211, 21121, 21221, 11211]).

The script first decodes the strong and weak features and then based on the trial observations finds the strategy that is most likely to result in the given observations.

The model **outputs** a csv file `/likel_results/model_fit_results.csv`:

This contains the likelihoods and guessing parameter (alpha) values for each strategy. 

It also contains three columns with alternative ways of deriving the *best_strategy*:

- *best_strategy_guess*: If multiple strategies have the same likelihood, the label "guessing" is set as the value for that trial
- *best_strategy_rand*: If multiple strategies have the same likelihood, a strategy is selected at random from the equally likely strategies
- *best_strategies*: If multiple strategies have the same likelihood, a list containing all the equally likely strategies is set as the value for that trial
---

## Hidden Markov Model `/hmm`

### Data preprocessing

`hmm_data.py`: Reads in `model_fit_results.csv`





