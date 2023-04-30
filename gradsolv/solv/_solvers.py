from numpy import ndarray

from .. import error_handle
from ..precond import grad_preconditioner

from . import _auxiliary_routines as _aux

class grad_solver_result(object):

    def __init__(self, x: ndarray, iterations: int, relres_vec: ndarray):

        self.x = x
        self.iterations = iterations
        self.relres_vec = relres_vec

class grad_solver(object):

    def __init__(self):

        self._preconditioner = None

    def set_preconditioner(self, preconditioner: grad_preconditioner):

        self._preconditioner = preconditioner

    def _solve(self, matrix: ndarray, rhs: ndarray, err_handle: error_handle, tol: float, max_it: int, include_relres_vec=False) -> grad_solver_result:

        raise NotImplementedError()

    def solve(self, matrix: ndarray, rhs: ndarray, err_handle: error_handle, tol: float, max_it: int, include_relres_vec=False) -> grad_solver_result:

        error_occured = False

        if len(matrix.shape) != 2:
            
            err_handle._set_error("Value of argument matrix must be 2-dimensional!")
            error_occured = True

        elif matrix.shape[0] != matrix.shape[1]:

            err_handle._set_error("Matrix must be square!")
            error_occured = True

        elif len(rhs.shape) != 1:
            
            err_handle._set_error("Value of argument rhs must be 1-dimensional!")

        elif rhs.shape[0] != matrix.shape[0]:
            
            err_handle._set_error("")
            error_occured = True

        elif not _aux._check_matrix_symmetry(matrix):
            
            err_handle._set_error("Matrix must be symmetric!")
            error_occured = True

        elif not _aux._check_symmetric_matrix_positive_definitness(matrix):
            
            err_handle._set_error("Symmetric matrix must be positive definite!")
            error_occured = True

        if error_occured:

            return grad_solver_result(None, 0, None)
        
        else:

            return self._solve(matrix, rhs, err_handle, tol, max_it, include_relres_vec)
    
class steepest_descend_solver(grad_solver):

    def __init__(self):

        super().__init__()

    def _solve(self, matrix: ndarray, rhs: ndarray, err_handle: error_handle, tol: float, max_it: int, include_relres_vec=False) -> grad_solver_result:
        from numpy import array, zeros, dot
        from numpy.linalg import norm

        n_it = 0

        n = matrix.shape[0]

        x = zeros(shape = n)
        r = rhs - (matrix @ x)
        z = self._preconditioner(matrix, r) if self._preconditioner else r

        norm_r0 = norm(r)
        norm_r = norm_r0

        relres_vec = [] if include_relres_vec else None

        while (n_it < max_it) and ((norm_r/norm_r0) > tol):

            alpha = dot(r, z)/dot(z, matrix @ z)

            x = x + alpha*z

            r = r - alpha*(matrix @ z)

            z = self._preconditioner(matrix, r) if self._preconditioner else r

            norm_r = norm(r)
            
            if include_relres_vec:
                
                relres_vec.append(norm_r/norm_r0)

            n_it += 1

        if (n_it == max_it) and ((norm_r/norm_r0) > tol):

            err_handle._set_error("Max number of iterations was reached, but solver did not converge!")

        relres_vec = array(relres_vec) if include_relres_vec else None

        return grad_solver_result(x, n_it, relres_vec)

class conjugate_gradient_solver(grad_solver):

    def __init__(self):

        super().__init__()

    def _solve(self, matrix: ndarray, rhs: ndarray, err_handle: error_handle, tol: float, max_it: int, include_relres_vec=False) -> grad_solver_result:
        from numpy import array, zeros, dot
        from numpy.linalg import norm

        n_it = 0

        n = matrix.shape[0]

        x = zeros(shape = n)
        r = rhs - (matrix @ x)
        z = self._preconditioner(matrix, r) if self._preconditioner else r
        p = z.copy()

        norm_r0 = norm(r)
        norm_r = norm_r0

        relres_vec = [] if include_relres_vec else None

        if relres_vec:

            relres_vec.append(norm_r/norm_r0)

        while (n_it < max_it) and ((norm_r/norm_r0) > tol):

            alpha = dot(r, z)/dot(p, matrix @ p)

            x = x + alpha*p

            r_prev = r.copy()
            r = r - alpha*(matrix @ p)

            z_prev = z.copy()
            z = self._preconditioner(matrix, r) if self._preconditioner else r

            beta = dot(r, z)/dot(r_prev, z_prev)

            p = z + beta*p

            norm_r = norm(r)
            
            if include_relres_vec:

                relres_vec.append(norm_r/norm_r0)

            n_it += 1

        if (n_it == max_it) and ((norm_r/norm_r0) > tol):

            err_handle._set_error("Max number of iterations was reached, but solver did not converge!")

        relres_vec = array(relres_vec) if include_relres_vec else None

        return grad_solver_result(x, n_it, relres_vec)