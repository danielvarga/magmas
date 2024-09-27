import numpy as np
import string
import itertools


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

    if ast['type'] == 'var':
        var_name = ast['value']
        # Initialize P_var without the N dimension
        P_var = np.zeros((n, n, n, n, n))
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
    assert P.shape == (N, n, n, n, n, n)

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
    ).astype(int)

    return Ms


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


test_given_magmas()

test_single_expression_all_magmas()
