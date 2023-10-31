import matplotlib.pyplot as plt
import networkx as nx
import pickle
import numpy as np

with open("../hmm/hmm_results/best_model_states_4.pkl", "rb") as file:
    model = pickle.load(file)

n_states = model.n_components
transition_probabilities = np.around(model.transmat_,3)
emission_probabilities = np.around(model.emissionprob_,3)
starting_probabilities = np.around(model.startprob_,3)

print(transition_probabilities,
      emission_probabilities,
      starting_probabilities,
      n_states)

n_participants = int(len(model.n_trials)/10)

# Create graph
G = nx.DiGraph()

# Add nodes with starting probabilities
for i, prob in enumerate(starting_probabilities):
    G.add_node(i, starting_prob=prob, emissions=emission_probabilities[i])

# Add edges with transition probabilities
for i, row in enumerate(transition_probabilities):
    for j, prob in enumerate(row):
        if prob > 0:
            G.add_edge(i, j, weight=prob)

# Draw graph
pos = nx.circular_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=False, node_size=3000, node_color='skyblue', font_weight='bold', arrowsize=20)

# Draw starting probabilities inside nodes
for i, prob in enumerate(starting_probabilities):
    plt.text(*pos[i], f'{prob:.2f}', fontsize=12, ha='center', va='center')

# Draw transition probabilities on edges
edge_labels = {(i, j): f'{w["weight"]:.2f}' for i, j, w in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

for (i, j), label in edge_labels.items():
    if i == j:  # If it's a self loop
        x, y = pos[i]
        plt.text(x-0.1, y+0.2, label, fontsize=10, ha='center', va='center')

# Draw emission probabilities next to nodes
for i, emissions in enumerate(emission_probabilities):
    plt.text(pos[i][0], pos[i][1]-0.15, f'{emissions}', fontsize=10, ha='center', va='center')

plt.title('Hidden Markov Model Visualization')
plt.show()

plt.savefig(f"images/hmm_graph_s{n_states}_part{n_participants}.png")


