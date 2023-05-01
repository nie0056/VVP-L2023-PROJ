class error_handle(object):
    """
    A class used for handling various errors that can occur when using gradsolv solvers.

    If an error occurs during solving a system of linear equations using one of the gradsolv solvers, the error is stored in the given instance of this class.
    """

    def __init__(self):

        self.__error_occured = False
        self.__error_message = None

    def _set_error(self, error_message: str):

        self.__error_occured = True
        self.__error_message = error_message

    def _error_has_occured(self) -> bool:

        return self.__error_occured

    def _get_error_message(self) -> str:

        return self.__error_message
    
class GradSolvError(Exception):
    """
    A class representing an exception used in gradsolv package.
    """

    def __init__(self, error_message: str):

        super().__init__(error_message)

def chkerr(error_handle: error_handle):
    """
    A function used for error checking.

    It checks whether there is an error, stored in the given instance of the class error_handle.

    If there is an error, stored in the given instance of the class error_handle, it raises the GradSolvError with an appropriate error message.
    """

    if error_handle._error_has_occured():

        raise GradSolvError(error_handle._get_error_message())