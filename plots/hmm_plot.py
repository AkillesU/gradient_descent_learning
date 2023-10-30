import matplotlib.pyplot as plt
import networkx as nx
import pickle

with open("../results/best_model.pkl", "rb") as file:
    model = pickle.load(file)

n_states = model.n_states
transition_probabilities = model.transition_prob_
emission_probabilities = model.emission_prob_
starting_probabilities = model.start_prob_

# Create graph
G = nx.DiGraph()

# Add nodes
for s in range(n_states):
    G.add_node(s)

# Add edges for transitions
for i in range(n_states):
    for j in range(n_states):
        G.add_edge(i, j, weight=transition_probabilities[i, j])

# Draw graph
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=3000)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
labels = {(i, j): '{:.2f}'.format(transition_probabilities[i][j]) for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Display emission probabilities
for i, (x, y) in pos.items():
    plt.text(x, y - 0.15, f'E: {emission_probabilities[i]}', horizontalalignment='center', verticalalignment='center')

# Display starting probabilities (optional)
for i, (x, y) in pos.items():
    plt.text(x, y + 0.15, f'S: {starting_probabilities[i]:.2f}', horizontalalignment='center', verticalalignment='center')

plt.title('Hidden Markov Model Visualization')
plt.show()
