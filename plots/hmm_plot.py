import graphviz
import pickle

# Load the model from the pickle file
with open('../results/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

print(model.emissionprob_.shape)


def generate_dot(model, state_names, observation_labels):
    dot = graphviz.Digraph('HMM', format='png')

    # Add states
    for i, state in enumerate(state_names):
        label = "{}\n".format(state)
        for j, obs in enumerate(observation_labels):
            label += "P({})={:.2f} ".format(obs, model.emissionprob_[i][j])
        dot.node(str(i), label)

    # Add transitions
    for i, origin_state in enumerate(state_names):
        for j, destination_state in enumerate(state_names):
            if model.transmat_[i][j] > 0:
                dot.edge(str(i), str(j), label="{:.2f}".format(model.transmat_[i][j]))

    # Add start probabilities
    for i, state in enumerate(state_names):
        dot.edge('start', str(i), label="{:.2f}".format(model.startprob_[i]))

    return dot

# Example usage
state_names = ['State1', 'State2', 'State3', 'State4']
observation_labels = ['strong', 'weak1', 'weak2', 'prototype']

dot = generate_dot(model, state_names, observation_labels)
dot.render('hmm_graph', view=True)
