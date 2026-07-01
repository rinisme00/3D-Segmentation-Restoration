class Dict2Obj(dict):
    """A dictionary that supports dot-notation access to its items, recursively.
    It inherits from dict so it can be passed to functions expecting dict(kwargs)."""
    
    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is not None:
            for k, v in d.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Dict2Obj):
            value = Dict2Obj(value)
        super().__setitem__(key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'Dict2Obj' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value
