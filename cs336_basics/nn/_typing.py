from jaxtyping import AbstractDtype, Bool, Float


class MaskBias(AbstractDtype):
    dtypes = Bool.dtypes + Float.dtypes  # ty:ignore[unresolved-attribute]
    # dtypes = [ # static checker friendly version of the above
    #     "bool",
    #     "bool_",
    #     regex.compile(r"^(?:bfloat|float)\d+(?:_[A-Za-z0-9]+)*$"),
    # ]
