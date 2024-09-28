import sys
import string
import itertools
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image



VARIABLE_COUNT = 4 # x,y,z,w are allowed, the rest not, use equations_4.txt


def tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] in '()+':
            tokens.append(s[i])
            i += 1
        else:
            # Variable name
            start = i
            while i < len(s) and s[i].isalnum():
                i += 1
            tokens.append(s[start:i])
    return tokens


def parse_expression(tokens):
    def parse_expression():
        node = parse_term()
        while tokens and tokens[0] == '+':
            op = tokens.pop(0)  # Consume '+'
            right = parse_term()
            node = {'type': 'op', 'op': op, 'left': node, 'right': right}
        return node

    def parse_term():
        token = tokens.pop(0)
        if token == '(':
            node = parse_expression()
            if not tokens or tokens.pop(0) != ')':
                raise ValueError("Expected ')'")
            return node
        else:
            # Assume token is a variable
            return {'type': 'var', 'value': token}

    return parse_expression()


# this gpt code is incorrect, but i leave it as a reminder that a single einsum could solve the whole task.
def generate_einsum(ast, M, index_counter, indices_set):
    if ast['type'] == 'var':
        var_name = ast['value']
        indices_set.add(var_name)
        return [], [], var_name
    elif ast['type'] == 'op':
        # Process left and right operands
        left_tensors, left_subscripts, left_index = generate_einsum(
            ast['left'], M, index_counter, indices_set)
        right_tensors, right_subscripts, right_index = generate_einsum(
            ast['right'], M, index_counter, indices_set)
        # Create a new index for the result
        result_index = f"k{index_counter[0]}"
        index_counter[0] += 1
        # Subscript for M: concatenate indices
        m_subscript = f"{left_index}{right_index}{result_index}"
        tensors = left_tensors + right_tensors + [M]
        subscripts = left_subscripts + right_subscripts + [m_subscript]
        return tensors, subscripts, result_index
def assemble_einsum(subscripts, indices_set, result_index):
    # Collect all indices used
    all_indices = set()
    for s in subscripts:
        all_indices.update(s)
    all_indices.update(indices_set)
    all_indices.add(result_index)

    # Create a mapping from indices to single letters
    available_letters = list(string.ascii_letters)
    index_to_letter = {}
    for idx in sorted(all_indices):
        if available_letters:
            letter = available_letters.pop(0)
            index_to_letter[idx] = letter
        else:
            raise ValueError("Ran out of letters for indexing.")

    # Map indices in subscripts
    mapped_subscripts = []
    for s in subscripts:
        mapped_s = ''.join(index_to_letter[idx] for idx in s)
        mapped_subscripts.append(mapped_s)

    # Map output indices
    mapped_output_indices = ''.join(index_to_letter[idx] for idx in sorted(indices_set)) + index_to_letter[result_index]

    # Assemble the einsum subscript
    input_subscripts = ','.join(mapped_subscripts)
    einsum_subscript = f"{input_subscripts}->{mapped_output_indices}"
    return einsum_subscript, index_to_letter


def combine_tables(P1, P2, Ms):
    """
    Combine two multiplication tables P1 and P2 using multiple binary operation tables Ms.

    Parameters:
        P1: numpy array of shape (N, ..., n), where '...' represents variable dimensions.
        P2: numpy array of shape (N, ..., n), same '...' dimensions as P1.
        Ms: numpy array of shape (N, n, n, n), representing N binary operations.

    Returns:
        P: numpy array of shape (N, ..., n)
    """
    # P1: (N, ..., a)
    # P2: (N, ..., b)
    # Ms: (N, n, n, n)
    # Compute P[N, ..., c] = sum_{a,b} P1[N, ..., a] * P2[N, ..., b] * Ms[N, a, b, c]
    P = np.einsum('N...a,N...b,Nabc->N...c', P1, P2, Ms, optimize=True)
    return P


def compute_table(Ms, ast):
    """
    Compute the multiplication table P for the formula represented by the ast, for multiple binary operation tables Ms.
    Returns a 6D array P[N, x, y, z, w, k], where k is the result of the formula given x, y, z, w, for each Ms[N].

    Parameters:
        Ms: numpy array of shape (N, n, n, n), representing N binary operations.
        ast: Abstract syntax tree representing the formula.

    Returns:
        P: numpy array of shape (N, n, n, n, n, n), where P[N, x, y, z, w, k] = 1 if the formula evaluates to k for Ms[N].
    """
    N, n, _, _ = Ms.shape

    assert VARIABLE_COUNT <= 4, "the following code block assumes less than 5 variables"

    if ast['type'] == 'var':
        var_name = ast['value']
        # Initialize P_var without the N dimension
        P_var = np.zeros(tuple([n] * (VARIABLE_COUNT + 1)), dtype=np.uint8)
        if var_name == 'x':
            for x in range(n):
                P_var[x, :, :, :, x] = 1  # For all y, z, w
        elif var_name == 'y':
            for y in range(n):
                P_var[:, y, :, :, y] = 1
        elif var_name == 'z':
            for z in range(n):
                P_var[:, :, z, :, z] = 1
        elif var_name == 'w':
            for w in range(n):
                P_var[:, :, :, w, w] = 1
        else:
            raise ValueError(f"Unknown variable '{var_name}'")
        # Expand P_var to include the N dimension via broadcasting
        P_var = np.broadcast_to(P_var, (N,) + P_var.shape)
        return P_var
    elif ast['type'] == 'op' and ast['op'] == '+':
        # Recursively compute left and right multiplication tables
        P_left = compute_table(Ms, ast['left'])  # Shape: (N, n, n, n, n, n)
        P_right = compute_table(Ms, ast['right'])  # Shape: (N, n, n, n, n, n)
        # Combine using combine_tables
        P = combine_tables(P_left, P_right, Ms)
        return P
    else:
        raise ValueError(f"Unsupported AST node {ast}")



def test_given_magmas():
    expression = "((x + y) + z) + x"
    tokens = tokenize(expression)
    ast = parse_expression(tokens)
    print(ast)

    # Example binary operation table M for modular addition
    n = 3

    # Create N different binary operation tables Ms
    N = 5  # Number of different Ms
    Ms = np.zeros((N, n, n, n))
    for i in range(N):
        # For demonstration, let's create Ms that represent addition modulo n, but shifted by i
        # M[i, a, b, c] = 1 if c = (a + b + i) % n
        for a in range(n):
            for b in range(n):
                c = (a + b + i) % n
                Ms[i, a, b, c] = 1

    P = compute_table(Ms, ast)

    # Verify the shape of P
    assert P.shape == tuple([N] + [n] * (VARIABLE_COUNT + 1))

    # Test the correctness of P for each Ms
    for idx in range(N):
        M = Ms[idx]
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    for w in range(n):
                        # Expected result using the shifted addition
                        expected_k = (x + y + z + x + idx * 3) % n  # idx * 3 accounts for the shifts in Ms
                        k = np.argmax(P[idx, x, y, z, w])
                        assert k == expected_k, f"Mismatch at N={idx}, x={x}, y={y}, z={z}, w={w}: expected {expected_k}, got {k}"
    print("all tests passed!")


# n is the size of the domain.
def collect_magmas(n):
    magmas = np.array(list(itertools.product(range(n), repeat=n*n))).reshape(-1, n, n)
    N = magmas.shape[0]  # Total number of magmas


    Ms = np.equal(
        magmas[:, :, :, np.newaxis],     # Shape: (N, n, n, 1)
        np.arange(n)[np.newaxis, np.newaxis, np.newaxis, :]  # Shape: (1, 1, 1, n)
    ).astype(np.uint8)

    return Ms


def compute_logical_implication(S):
    # Compute the logical NOT of S
    neg_S = np.logical_not(S)
    
    # Use broadcasting to compute the logical OR across all pairs
    # neg_S[:, None, :] has shape (E, 1, N)
    # S[None, :, :] has shape (1, E, N)
    implication = np.logical_or(neg_S[:, None, :], S[None, :, :])  # Shape: (E, E, N)
    
    # Check if the implication holds true for all elements in the last axis
    implication_matrix = np.all(implication, axis=-1)  # Shape: (E, E)
    
    return implication_matrix


def test_single_expression_all_magmas():
    # Define your expression
    expression = "((x + y) + z) + x"
    tokens = tokenize(expression)
    ast = parse_expression(tokens)
    print(ast)

    Ms = collect_magmas(3)
    print("considering", len(Ms), "magmas")

    P = compute_table(Ms, ast)
    print("calculated all tables")



def read_all_equations(lines):
    formulas = set()
    equations = []
    for l in lines:
        lhs, rhs = l.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()
        formulas.add(lhs)
        formulas.add(rhs)
        equations.append((lhs, rhs))
    print(f"{len(equations)} equations from {len(formulas)} formulas.")

    ast_dict = {}
    for formula in formulas:
        tokens = tokenize(formula)
        ast = parse_expression(tokens)
        ast_dict[formula] = ast

    return equations, ast_dict


# does not use caching of formulas (ast_dict)
def is_satisfied(equation, Ms):
    lhs, rhs = equation
    ast_lhs = parse_expression(tokenize(lhs))
    ast_rhs = parse_expression(tokenize(rhs))
    P_lhs = compute_table(Ms, ast_lhs)
    P_rhs = compute_table(Ms, ast_rhs)
    # for each magma Ms[i], passing[i] tells if it satisfies equation or not:
    passing = np.all(np.isclose(P_lhs, P_rhs), axis=tuple(range(1, VARIABLE_COUNT + 2)))
    return passing, P_lhs, P_rhs


def get_all_satisfied(equations, ast_dict, Ms):
    n = Ms.shape[-1]

    P_dict = {}
    print("building multiplication tables for each formula")
    log_step_size = 10 if n >= 3 else 100
    for i, (formula, ast) in enumerate(ast_dict.items()):
        if i % log_step_size == 0:
            print(i, "/", len(ast_dict))
            sys.stdout.flush()
        P = compute_table(Ms, ast)
        P_dict[formula] = P
    with open('P_dict.pkl', 'wb') as f:
        pickle.dump(P_dict, f)

    S = []
    print("evaluating the truth value of each equation for all magmas")
    for i, equation in enumerate(equations):
        if i % 100 == 0:
            print(i, "/", len(equations))
            sys.stdout.flush()
        lhs, rhs = equation
        P_lhs = P_dict[lhs]
        P_rhs = P_dict[rhs]
        # for each magma Ms[i], passing[i] tells if it satisfies equation or not:
        passing = np.all(np.isclose(P_lhs, P_rhs), axis=tuple(range(1, VARIABLE_COUNT + 2)))

        # print(i, lhs, "=", rhs, "satified by", passing.sum(), "magmas")
        S.append(passing)

    S = np.array(S)

    print("collected S matrix about which magma satisfies which equation")
    print(f"{S.shape} matrix with {S.sum()} true elements")

    return S

'''
    implications = compute_logical_implication(S)
    print(f"{implications.sum()} logical implications across the {implications.shape} equation pairs.")
    return implications
'''


def pp_magma(magma):
    return np.argmax(magma, axis=-1)


def pp_eq(equation):
    return f"{equation[0]} = {equation[1]}"


def smallest_diff_index(a, b):
    # Compute the element-wise difference
    diff = a != b
    # Use np.where to find indices where they differ
    indices = np.where(diff)[0]
    # Return the smallest index if they differ, else -1
    return indices[0] if indices.size > 0 else -1


# TODO only works for x,y formulas
def condense_P(P, n):
    assert P.shape[0] == 1
    mult = pp_magma(P[0])
    for z in range(n):
        for w in range(n):
            assert np.allclose(mult[:, :, z, w], mult[:, :, 0, 0])
    mult = mult[:, :, 0, 0]
    return mult


# Function to use networkx for transitive reduction
def transitive_reduction(M):
    """
    Perform transitive reduction on a graph using NetworkX's transitive_reduction function.

    :param M: 2D numpy array representing the adjacency matrix of the DAG
    :return: 2D numpy array representing the transitive reduction of the DAG
    """

    M = M.copy()
    np.fill_diagonal(M, False)

    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)

    # Get the transitive reduction using NetworkX
    G_reduced = nx.transitive_reduction(G)

    # Convert the reduced graph back to an adjacency matrix
    reduced_M_nx = nx.to_numpy_array(G_reduced, dtype=int)

    return np.array(reduced_M_nx)


def merge_equivalent_nodes_and_transitive_reduction(M):
    """
    Identify equivalent nodes (i.e., nodes that imply each other), collect them,
    keep only one representative node, and perform transitive reduction.
    
    :param M: 2D numpy array representing the adjacency matrix of the DAG
    :return: 2D numpy array representing the transitive reduction of the DAG with merged equivalent nodes
    """
    # Remove diagonal elements (self-loops)
    np.fill_diagonal(M, 0)
    
    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(M, create_using=nx.DiGraph)
    
    # Find equivalent nodes (bidirectional edges mean nodes imply each other)
    equivalent_sets = []
    visited = set()

    for u in G.nodes():
        if u not in visited:
            # Find all nodes equivalent to u (nodes with bidirectional edges to u)
            equivalent_nodes = {u}
            for v in G.nodes():
                if u != v and G.has_edge(u, v) and G.has_edge(v, u):
                    equivalent_nodes.add(v)
            
            if len(equivalent_nodes) > 1:
                equivalent_sets.append(equivalent_nodes)
            visited.update(equivalent_nodes)
    
    # Merge equivalent nodes by keeping one representative from each set
    for eq_set in equivalent_sets:
        eq_list = list(eq_set)
        representative = eq_list[0]  # Pick the first node as the representative
        for node in eq_list[1:]:  # Remove other equivalent nodes
            G = nx.contracted_nodes(G, representative, node, self_loops=False)

    # Perform transitive reduction on the simplified graph
    G_reduced = nx.transitive_reduction(G)

    return G_reduced, equivalent_sets


def visualize_graph_with_labels_using_dot(G, equations, output_filename='reduced_graph.png'):
    """
    Visualize the graph G using Graphviz with custom labels from 'equations' and save it to an output file.
    
    :param G: NetworkX DiGraph object (graph to visualize)
    :param equations: List of labels (strings) for each node in the matrix
    :param output_filename: Filename for saving the rendered graph (default: 'reduced_graph.png')
    """
    # Create a Graphviz AGraph object for visualization
    A = nx.drawing.nx_agraph.to_agraph(G)
    
    # Apply custom labels from the 'equations' list
    for i, equation in enumerate(equations):
        if i in G.nodes():
            A.get_node(i).attr['label'] = equation
    
    # Render the graph to a file (e.g., PNG) and visualize it
    A.layout(prog='dot')  # Use 'dot' for layout
    A.draw(output_filename)  # Save as PNG
    
    # Optionally display the image using matplotlib
    img = plt.imread(output_filename)
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels
    plt.show()


def visualize_graph_with_labels_using_plotly(G, equations):
    # Use Graphviz layout for a hierarchical layout (like DOT)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Extract node positions and labels
    x_nodes = [pos[node][0] for node in G.nodes()]  # X-coordinates of nodes
    y_nodes = [pos[node][1] for node in G.nodes()]  # Y-coordinates of nodes
    node_labels = list(G.nodes())  # Node labels

    # Create edge traces for lines between nodes
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # Add None to separate lines
        edge_y.extend([y0, y1, None])

    # Create edge trace (scatter for the edges)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='black'),
        hoverinfo='none',
        mode='lines'
    )

    # Create scatter trace for hover points on edges
    edge_hovertexts = [f'({edge[0] + 1}) -> ({edge[1] + 1})' for edge in G.edges()]
    edge_hover_trace = go.Scatter(
        x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],  # Midpoint x
        y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],  # Midpoint y
        text=edge_hovertexts,  # Hovertext for each edge
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'),  # Transparent markers
        hoverinfo='text'
    )

    # Create node trace with hover functionality
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(size=20, color='lightblue', line=dict(width=2)),
        text=[equations[node] + f" ({node + 1})" for node in G.nodes()],  # Text for nodes
        hoverinfo='text',
        textposition="bottom center"
    )

    # Add arrow annotations to represent directed edges
    annotations = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Create an arrow annotation for each edge
        annotations.append(
            dict(
                ax=x0, ay=y0,  # Start of the arrow (source node)
                x=x1, y=y1,    # End of the arrow (target node)
                xref='x', yref='y',  # Reference to the scatter plot coordinates
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=3,  # Type of arrowhead
                arrowsize=2,  # Size of the arrowhead
                arrowwidth=1.5,
                arrowcolor='black'
            )
        )

    # Create figure with arrows as annotations
    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        annotations=annotations  # Add the arrows
                    ))

    # Show the figure
    fig.update_layout(title_text="Hasse diagram", font_size=12)
    fig.show()

    fig.write_html("hasse.html")


def implications_to_image(implications):
    # Define the colors
    UNKNOWN_COLOR = (0, 0, 0)
    KNOWN_IMPLIES_COLOR = (0, 255, 0)
    KNOWN_NOT_IMPLIES_COLOR = (255, 0, 0)

    # Create a 3D array for the image where each pixel has 3 values (R, G, B)
    image_shape = implications.shape
    pixel_array = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Apply colors using numpy indexing
    pixel_array[implications] = UNKNOWN_COLOR  # Where True
    pixel_array[~implications] = KNOWN_NOT_IMPLIES_COLOR  # Where False

    # Convert the pixel array to an image
    img = Image.fromarray(pixel_array, "RGB")
    
    return img


def main_hasse_diagram():
    n = 3

    equations, ast_dict = read_all_equations(sys.stdin.readlines())

    Ms = collect_magmas(n)
    S = get_all_satisfied(equations, ast_dict, Ms)
    np.save("S.npy", S)
    implications = compute_logical_implication(S)
    print(f"number of true implications for {n}-magmas", implications.sum(), "out of", implications.size)
    print(implications.astype(int))

    img = implications_to_image(implications)
    img.save("implications.png")

    # reduced_implications = transitive_reduction(implications)
    G_reduced_implications, equivalent_sets = merge_equivalent_nodes_and_transitive_reduction(implications)
    print(f"number of primitive implications for {n}-magmas", G_reduced_implications.number_of_edges(),
        "out of", implications.size, "but now just on", G_reduced_implications.number_of_nodes(), "nodes, not", len(implications))
    for equivalent_set in equivalent_sets:
        print("  <=>  ".join(pp_eq(equations[eq_index]) for eq_index in sorted(equivalent_set)))

    visualize_graph_with_labels_using_plotly(G_reduced_implications, [pp_eq(equation) for equation in equations])


def main():
    equations, ast_dict = read_all_equations(sys.stdin.readlines())

    Ms_2 = collect_magmas(2)
    S_2 = get_all_satisfied(equations, ast_dict, Ms_2)
    implications_2 = compute_logical_implication(S_2)
    print("number of true implications for 2-magmas", implications_2.sum(), "out of", implications_2.size)

    Ms_3 = collect_magmas(3)
    S_3 = get_all_satisfied(equations, ast_dict, Ms_3)
    implications_3 = compute_logical_implication(S_3)
    print("number of true implications for 3-magmas", implications_3.sum(), "out of", implications_3.size)

    E = len(equations)
    for i in range(E):
        for j in range(E):
            if implications_2[i, j] != implications_3[i, j]:
                assert implications_2[i, j] > implications_3[i, j], "3-magmas are supposed to have fewer true implications"
                print("-------")
                print(pp_eq(equations[i]), "implies", pp_eq(equations[j]), "for 2-magmas, but not for 3-magmas.")

                print("counterexample:")
                assert np.all(S_2[i] <= S_2[j])
                passings_i = S_3[i]
                passings_j = S_3[j]
                magma_index = smallest_diff_index(passings_i, passings_j)
                assert magma_index != -1
                print(pp_magma(Ms_3[magma_index]))
                sys.stdout.flush()


# test_given_magmas() ; exit()

# test_single_expression_all_magmas() ; exit()

main_hasse_diagram() ; exit()

main()
