from numpy import ndarray

from . import _auxiliary_routines as _aux

class grad_preconditioner(object):

    def __call__(self, matrix: ndarray, r: ndarray) -> ndarray:

        raise NotImplementedError()
    
class diagonal_preconditioner(grad_preconditioner):
    """
    A class representing diagonal preconditioner.
    """

    def __call__(self, matrix: ndarray, r: ndarray) -> ndarray:
        
        return _aux._diagonal_solve(matrix, r)
    
class gauss_seidel_DLDLT_preconditioner(grad_preconditioner):
    """
    A class representing Gauss-Seidel DLDLT preconditioner.
    """

    def __call__(self, matrix: ndarray, r: ndarray) -> ndarray:
        from numpy import tril, diagflat, diag
        
        tril_matrix = tril(matrix, -1)
        diag_matrix = diagflat(diag(matrix))

        y = (diag_matrix - (tril_matrix.T/2)) @ r
        z = _aux._forward_substitution_solve(diag_matrix + (tril_matrix/2), y)

        return z

class gauss_seidel_LDDLD_preconditioner(grad_preconditioner):
    """
    A class representing Gauss-Seidel LDDLD preconditioner.
    """

    def __call__(self, matrix: ndarray, r: ndarray) -> ndarray:
        from numpy import tril, diagflat, diag

        tril_matrix = tril(matrix, -1)
        diag_matrix = diagflat(diag(matrix))

        x = _aux._backward_substitution_solve((diag_matrix + tril_matrix).T, r)
        y = diag_matrix @ x
        z = _aux._forward_substitution_solve(diag_matrix + tril_matrix, y)

        return z