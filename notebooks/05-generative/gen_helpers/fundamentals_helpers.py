import io

try:
    import ipywidgets as widgets
except Exception:
    widgets = None

try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

from IPython.display import display

def bind_widget_state(controls, apply_fn):
    state_holder = {"has_rendered": False, "last": None}

    def refresh(change=None):
        state = {name: control.value for name, control in controls.items()}
        state_key = tuple((name, repr(value)) for name, value in state.items())
        if state_holder["has_rendered"] and state_holder["last"] == state_key:
            return
        state_holder["has_rendered"] = True
        state_holder["last"] = state_key
        apply_fn(**state)

    refresh()
    for control in controls.values():
        control.observe(refresh, names="value")
    return refresh


def figure_to_png_bytes(fig, *, dpi: int = 160) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


__all__ = [
    "IN_COLAB",
    "bind_widget_state",
    "display",
    "figure_to_png_bytes",
    "widgets",
]
