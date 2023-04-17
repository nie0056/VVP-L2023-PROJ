from numpy import ndarray

def _diagonal_solve(diagonal_matrix: ndarray, rhs: ndarray) -> ndarray:
    from numpy import diag
    
    return rhs/diag(diagonal_matrix)

def _backward_substitution_solve(upper_triangular_matrix: ndarray, rhs: ndarray) -> ndarray:
    from numpy import zeros, dot

    n = upper_triangular_matrix.shape[0]

    x = zeros(shape = upper_triangular_matrix.shape[0], dtype = float)

    x[n - 1] = rhs[n - 1]/upper_triangular_matrix[n - 1, n - 1]

    for i in range(n - 2, -1, -1):

        x[i] = (rhs[i] - dot(upper_triangular_matrix[i, i:n], x[i:n]))/upper_triangular_matrix[i, i]

    return x

def _forward_substitution_solve(lower_triangular_matrix: ndarray, rhs: ndarray) -> ndarray:
    from numpy import zeros, dot

    n = lower_triangular_matrix.shape[0]

    x = zeros(shape = n, dtype = float)

    x[0] = rhs[0]/lower_triangular_matrix[0, 0]

    for i in range(1, n):
        
        x[i] = (rhs[i] - dot(lower_triangular_matrix[i, 0:i], x[0:i]))/lower_triangular_matrix[i, i]

    return x