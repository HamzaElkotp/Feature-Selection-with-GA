import tkinter as tk
import ttkbootstrap as ttk


def _validate_float_spinbox(proposed_value, spinbox_widget, var, min_, max_):
    """
    Validates and updates the background color of the spinbox.
    - proposed_value: the text currently in the widget
    - spinbox_widget: reference to the Spinbox itself
    - var: tkinter variable linked to Spinbox
    """
    if proposed_value.strip() == "":
        # temporarily allow empty input while typing
        spinbox_widget.configure(bootstyle="danger")  # red highlight for empty
        return True

    try:
        val = float(proposed_value)
    except ValueError:
        # invalid number
        spinbox_widget.configure(bootstyle="danger")
        return False

    # check bounds
    if val < min_ or val > max_:
        spinbox_widget.configure(bootstyle="danger")  # red highlight
        return True  # allow editing but show warning color
    else:
        spinbox_widget.configure(bootstyle="default")  # restore normal
        var.set(round(val, 2))
        return True


def numeric_spinbox(parent, var, min_=0.0, max_=100.0, step=1.0,
                    width=8, bootstyle=None, label_text=None):
    """
    Reusable helper to create a labeled numeric Spinbox with validation & highlight.
    """
    frame = ttk.Frame(parent)

    if label_text:
        ttk.Label(frame, text=label_text).pack(side="left", padx=(0, 5))

    spin = ttk.Spinbox(
        frame,
        from_=min_,
        to=max_,
        increment=step,
        textvariable=var,
        width=width,
        format="%.2f",
        wrap=False,
        bootstyle=bootstyle or "default"
    )

    # register validation command (pass the spinbox instance)
    vcmd = (frame.register(lambda p: _validate_float_spinbox(p, spin, var, min_, max_)), "%P")
    spin.configure(validate="key", validatecommand=vcmd)
    spin.pack(side="left")

    return frame