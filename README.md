# gradient_descent_learning

This is the code for analysing the data from a rerun of https://osf.io/preprints/psyarxiv/75e4t/ .

The code contains:
- Initial data wrangling
- A maximum likelihood model that outputs the strategy a participant was using for a trial
- A Hidden Markov Model that infers hidden states based on the output of the maximum likelihood model

## Data details

`/data` contains the data and preprocessing pipeline.

`datawrangle.py`: Collects all participant responses into a single csv file

`fullcode.csv`: Code file for decoding randomisation

`cleaned_pilotdata.csv`: output from `datawrangle.py`. Contains participant trials along with randomisation coding.

`/pilotdata`: contains initial raw data from 10 participants.

---



