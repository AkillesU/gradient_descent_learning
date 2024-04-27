from hmmlearn import hmm
import matplotlib.pyplot as plt
import pandas as pd

data_path = "hmm_data_best_strategies_part156.csv"
sequence = pd.read_csv(f"hmm_data/{data_path}")

def file_num(input_string):
    if "strategies" in input_string:
        return 1
    elif "guess" in input_string:
        return 2
    elif "rand" in input_string:
        return 3

print(sequence.shape)
trials = sequence.sum(axis=1) # Set trials variable for "n_trials" arg in MultinomialHMM
lengths = [10]*int(len(sequence)/10) # Set lengths variable for "lengths" arg in model.fit

bic = []  # Initialise list of BIC values
aic = []  # Initialise list of AIC values
lls = []  # Initialise list of log likelihood values
ns = [2, 3, 4, 5, 6, 7, 8]  # Set list containing number of states

# This function fits 100 randomly initialised MultinomialHMM models with the data and selects the best model.
# The process is repeated for each n_states and the results (AIC,BIC,LogLikel) are saved and plotted.

def select_best_model(X):

    for n_states in ns:
        best_ll = None
        best_model = None
        for q in range(100):
            print(q)
            # Train the HMM
            model = hmm.MultinomialHMM(n_components=n_states,
                                       n_iter=1000,
                                       n_trials=trials)
            # Set number of features. (Should be 5 if using best_strategy_guess)
            if "guess" in data_path:
                model.n_features = 5
            else:
                model.n_features = 4
            model.fit(X, lengths=lengths)

            score = model.score(X)  # Get log likelihood for model fit
            if not best_ll or best_ll < best_ll:
                best_ll = score
                best_model = model
        aic.append(best_model.aic(X))
        bic.append(best_model.bic(X))
        lls.append(best_model.score(X))

    fig, ax = plt.subplots()
    ax.plot(ns, aic, label="AIC", color="blue", marker="o")
    ax.plot(ns, bic, label="BIC", color="green", marker="o")
    ax2 = ax.twinx()
    ax2.plot(ns, lls, label="LL", color="orange", marker="o")

    ax.legend(handles=ax.lines + ax2.lines)
    ax.set_title(data_path)
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    fig.tight_layout()

    plt.savefig(f"hmm_results/hmm_search_part{int(len(trials) / 10)}_{file_num(data_path)}")
    plt.show()


select_best_model(sequence)