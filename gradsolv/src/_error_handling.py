class error_handle(object):

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

    def __init__(self, error_message: str):

        super().__init__(error_message)

def chkerr(error_handle: error_handle):

    if error_handle._error_has_occured():

        raise GradSolvError(error_handle._get_error_message())