from numpy import ndarray

def _check_matrix_symmetry(matrix: ndarray) -> bool:
    from numpy import all

    return all(matrix == matrix.T)

def _check_symmetric_matrix_positive_definitness(symmetric_matrix: ndarray) -> bool:
    from numpy import all, diag
    
    return(all(diag(symmetric_matrix) > 0))