from typing import Tuple
from typing import Optional
from typing import Mapping

from typeguard import check_argument_types, check_return_type


class ClassChoices:
    """Helper class to manage the options for variable objects and its configuration.
    Example:

    >>> class A:
    ...     def __init__(self, foo=3):  pass
    >>> class B:
    ...     def __init__(self, bar="aaaa"):  pass
    >>> choices = ClassChoices("var", dict(a=A, b=B), default="a")
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> _ = parser.add_argument("--var", default='a')
    >>> args = parser.parse_args(["--var", "a"])
    >>> class_obj = choices.get_class(args.var)
    """

    def __init__(
        self,
        name: str,
        classes: Mapping[str, type],
        type_check: type = None,
        default: str = None,
        optional: bool = False,
    ):
        assert check_argument_types()
        self.name = name
        self.base_type = type_check
        self.classes = {k.lower(): v for k, v in classes.items()}
        if "none" in self.classes or "nil" in self.classes or "null" in self.classes:
            raise ValueError('"none", "nil", and "null" are reserved.')
        if type_check is not None:
            for v in self.classes.values():
                if not issubclass(v, type_check):
                    raise ValueError(f"must be {type_check.__name__}, but got {v}")

        self.optional = optional
        self.default = default
        if default is None:
            self.optional = True

    def choices(self) -> Tuple[Optional[str], ...]:
        retval = tuple(self.classes)

        # None type 허용 여부
        if self.optional:
            return retval + (None,)
        else:
            return retval

    def get_class(self, name: Optional[str]) -> Optional[type]:
        assert check_argument_types()
        if name is None or (self.optional and name.lower() == ("none", "null", "nil")):
            retval = None
        elif name.lower() in self.classes:
            class_obj = self.classes[name]
            assert check_return_type(class_obj)
            retval = class_obj
        else:
            raise ValueError(
                f"--{self.name} must be one of {self.choices()}: "
                f"--{self.name} {name.lower()}"
            )

        return retval


if __name__ == "__main__":
    print("??")