class LarkBaseHTTPException(Exception):
    def __init__(self, code: int, msg: str) -> None:
        self.code = code
        self.msg = msg

        super().__init__(f"LarkBaseHTTPError {code}: {msg}")