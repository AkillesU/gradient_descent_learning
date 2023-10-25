from hmmlearn import hmm
import matplotlib.pyplot as plt
import pandas as pd

sequence = pd.read_csv("results/hmm_data_best_strategies.csv")

print(sequence.shape)
trials = sequence.sum(axis=1)
lengths = [10]*10

bic = []
aic = []
lls = []
ns = [1,2,3,4,5,6,7,8]
def select_best_model(X):
    best_score = 0
    best_num_states = None

    for n_states in ns:
        best_ll = None
        best_model = None
        print(best_ll)
        for q in range(10):
            print(q)
            # Train the HMM
            model = hmm.MultinomialHMM(n_components=n_states,
                                       n_iter=1000,
                                       n_trials=trials)
            model.n_features = 4
            model.fit(X, lengths=lengths)

            score = model.score(X)
            if not best_ll or best_ll < best_ll:
                best_ll = score
                best_model = model
        aic.append(best_model.aic(X))
        bic.append(best_model.bic(X))
        lls.append(best_model.score(X))

    fig, ax = plt.subplots()
    ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
    ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
    ax2 = ax.twinx()
    ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")

    ax.legend(handles=ax.lines + ax2.lines)
    ax.set_title("Using AIC/BIC for Model Selection")
    ax.set_ylabel("Criterion Value (lower is better)")
    ax2.set_ylabel("LL (higher is better)")
    ax.set_xlabel("Number of HMM Components")
    fig.tight_layout()

    plt.show()


best_num_states = select_best_model(sequence)
print(f"The best number of states based on aic is: {best_num_states}")
