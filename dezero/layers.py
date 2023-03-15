from dezero.core import Parameter


class Layer:
    def __init__(self) -> None:
        self._params = set()

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Parameter):
            self._params.add(__name)
        super().__setattr__(__name, __value)
