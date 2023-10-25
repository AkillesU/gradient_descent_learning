import numpy as np
import pandas as pd
from hmmlearn import hmm
"""
Sequence contains a 4 column array with the the column names
corresponding to the following strategies:
0: Strong
1: Weak1
2: Weak2
3: Prototype

The maximum likelihood model determined by "model_fitting.py" determines a trial value:
e.g., [1,0,0,0] -> Strong strategy trial
      [0,0,1,0] -> Weak2 strategy trial
"""

# Load datafile generated by hmm_data.py
sequence = pd.read_csv("results/hmm_data_best_strategies.csv")

# Sum of all strategies used per trial. Used to confirm the count matches in the hmm.fit algorithm
trials = sequence.sum(axis=1)

for q in range(1000):
    print(q)
    model = hmm.MultinomialHMM(n_components=3,
                               n_iter=1000,
                               params='tse',
                               init_params="tse",# Including letter ignores user parameter initialisation
                               n_trials=trials,
                               verbose=True) #

    # Set 5 features: Strong, Weak1, Weak2, Prototype, Guessing. NOTE: Has to be set to 4 with random method
    model.n_features = 4
    lengths = [10]*10
    model.startprob_ = [0.25,0.25,0.25,0.25]
    model.fit(sequence)
    log_likelihood=model.monitor_.history[-1]
    print(model.monitor_.history[-1])
    if q==0:
        best_ll=log_likelihood
        best_model=model
    else:
        if log_likelihood>best_ll:
            best_ll=log_likelihood
            best_model=model

    print(best_model.monitor_.history[-1])

# save model results
np.savetxt("emission_probs.csv",best_model.emissionprob_)
np.savetxt("start_prob.csv",best_model.startprob_)
np.savetxt("trans_mat.csv",best_model.transmat_)

print(f"Emission_probabilities: {np.around(best_model.emissionprob_,3)}\n",
      f"starting probabilities: {np.around(best_model.startprob_,3)}\n",
      f"Transition Matrix: {np.around(best_model.transmat_,3)}")


import pickle

with open("results/best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

results = best_model.predict(sequence, lengths=lengths)
print(results)
result_data = pd.DataFrame(sequence)
result_data["State"] = results

# Save results into the "results folder"
result_data.to_csv("results/hmm_hidden_states.csv", index=False)
