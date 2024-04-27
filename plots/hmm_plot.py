import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import pickle
import numpy as np


""")
This code creates a HMM graph. 
Colors denote nodes and transition probabilities FROM that node.
"""
# Load a model
with open("../hmm/hmm_results/best_model_states_2_part156.pkl", "rb") as file:
    model = pickle.load(file)

# Set variables with trained model properties
n_states = model.n_components  # Number of hidden states
transition_probabilities = np.around(model.transmat_,3)  # Trained transition matrix
emission_probabilities = np.around(model.emissionprob_,3)  # Trained emission prob matrix
starting_probabilities = np.around(model.startprob_,3)  # Trained starting probabilities for each hidden state

print(transition_probabilities,
      emission_probabilities,
      starting_probabilities,
      n_states)

n_participants = int(len(model.n_trials)/10)  # Set number of participants for versioning

# Define a color map for the nodes and edges, dynamically creating enough colors for the number of states
colors = cm.rainbow(np.linspace(0, 1, n_states))

# Create graph
G = nx.DiGraph()

# Add edges to the graph based on the transition probabilities matrix
for i in range(n_states):
    for j in range(n_states):
        # Add an edge only if the transition probability is greater than a threshold, for instance, 0.001
        if transition_probabilities[i][j] > 0.001:
            G.add_edge(i, j, weight=transition_probabilities[i][j])

# Function to determine position for edge labels to be on the right side of the transition
def get_label_pos(source, target, pos, offset=0.1):
    # Vector from source to target
    dx, dy = pos[target][0] - pos[source][0], pos[target][1] - pos[source][1]
    # Normalize vector
    norm = np.sqrt(dx**2 + dy**2)
    # Perpendicular vector to the right
    perp_dx, perp_dy = dy/norm, -dx/norm
    # Offset position by the perpendicular vector
    return pos[source][0] + dx/2 + perp_dx*offset, pos[source][1] + dy/2 + perp_dy*offset

# Draw graph
if n_states > 2:
    pos = nx.circular_layout(G)  # Set graph shape
else:
    pos = nx.spring_layout(G)
fig, ax = plt.subplots(figsize=(10, 8))  # Initialise figure

# Draw the graph with node-specific colors
for i, prob in enumerate(starting_probabilities):
    nx.draw_networkx_nodes(G, pos, nodelist=[i], node_size=3000, node_color=[colors[i]], ax=ax)


# Draw starting probabilities inside nodes
for i, prob in enumerate(starting_probabilities):
   plt.text(*pos[i], f'{prob:.2f}', fontsize=12, ha='center', va='center')

# Function to calculate label offset
def edge_label_offset(source_pos, target_pos, xoffset=0.1, yoffset=0.1):
    # Adjust the offset based on the position of the source and target
    x_offset = xoffset * (1 if source_pos[0] < target_pos[0] else -1)
    y_offset = yoffset * (1 if source_pos[1] < target_pos[1] else -1)

    # Calculate mid-point for label
    mid_x = (source_pos[0] + target_pos[0]) / 2
    mid_y = (source_pos[1] + target_pos[1]) / 2

    # Adjust the mid-point based on the calculated offset
    label_x = mid_x + y_offset  # Notice that we use y_offset for x coordinate
    label_y = mid_y + x_offset  # Notice that we use x_offset for y coordinate

    return label_x, label_y

# Function to calculate label offset
def calculate_offset(source_pos, target_pos, xoffset=0.1, yoffset=0.1):
    # Adjust the offset based on the position of the source and target
    x_offset = xoffset * (1 if source_pos[0] < target_pos[0] else -1)
    y_offset = yoffset * (1 if source_pos[1] < target_pos[1] else -1)

    # Calculate mid-point for label
    mid_x = (source_pos[0] + target_pos[0]) / 2
    mid_y = (source_pos[1] + target_pos[1]) / 2

    # Adjust the mid-point based on the calculated offset
    label_x = mid_x + y_offset  # Notice that we use y_offset for x coordinate
    label_y = mid_y + x_offset  # Notice that we use x_offset for y coordinate

    return label_x, label_y


# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='black', ax=ax, arrowsize=20, node_size=3000)

# Add edge labels with source node color
for i, j, data in G.edges(data=True):
    weight = data['weight']
    source_pos = pos[i]
    target_pos = pos[j]
    label_color = colors[i]  # Color label the same as the source node

    if i != j:
        edge_label_pos = calculate_offset(source_pos, target_pos)
        plt.text(*edge_label_pos, f"{weight:.2f}", fontsize=10, ha='center', va='center', color=label_color, fontweight="bold")
    else:
        label_pos = (source_pos[0], source_pos[1] + 0.25)  # Offset for self-loop
        plt.text(*label_pos, f"{weight:.2f}", fontsize=10, ha='center', va='center', color=label_color, fontweight="bold")

# Draw emission probabilities next to nodes
for i, emissions in enumerate(emission_probabilities):
    plt.text(pos[i][0], pos[i][1]-0.2, f'{emissions}', fontsize=10, ha='center', va='center')

ax.set_title(f'Hidden Markov Model Visualization ({n_states} states)')
plt.axis('off')  # Turn off the axis

plt.savefig(f"images/participants_{n_participants}/hmm_graph_s{n_states}_part{n_participants}.png")
plt.show()