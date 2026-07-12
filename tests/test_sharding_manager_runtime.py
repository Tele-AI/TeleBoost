from teleboost.engines.fsdp.sharding.runtime import run_with_sharding_managers


class _Manager:
    def __init__(self, name):
        self.name = name
        self.events = []

    def __enter__(self):
        self.events.append(f"{self.name}:enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.events.append(f"{self.name}:exit")
        return False

    def preprocess_data(self, data=None, **_kwargs):
        self.events.append(f"{self.name}:pre")
        return {self.name: data}

    def postprocess_data(self, data=None, **_kwargs):
        self.events.append(f"{self.name}:post")
        return {f"{self.name}_out": data}


def test_run_with_sharding_managers_separates_context_and_data_order():
    outer = _Manager("outer")
    inner = _Manager("inner")

    out = run_with_sharding_managers(
        "x",
        context_managers=(outer, inner),
        preprocess_managers=(inner, outer),
        postprocess_managers=(inner,),
        run=lambda data: {"seen": data},
    )

    assert out == {"inner_out": {"seen": {"outer": {"inner": "x"}}}}
    assert outer.events == ["outer:enter", "outer:pre", "outer:exit"]
    assert inner.events == ["inner:enter", "inner:pre", "inner:post", "inner:exit"]
