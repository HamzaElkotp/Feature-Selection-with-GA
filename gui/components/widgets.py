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


def dropdown_combobox(parent, var, values, label_text=None, width=15, bootstyle=None, readonly=True):
    """
    Reusable helper to create a labeled Combobox (dropdown) with optional styling.

    Args:
        parent: parent widget (frame or window)
        var: a tk.StringVar (or ttk.Variable) to bind the selection
        values: list of options for the dropdown
        label_text: optional label on the left
        width: width of the combobox entry box
        bootstyle: ttkbootstrap style string (e.g., "info", "darkly")
        readonly: if True, use "readonly" state (prevents typing custom values)
    """
    frame = ttk.Frame(parent)

    if label_text:
        ttk.Label(frame, text=label_text).pack(side="left", padx=(0, 5))

    state = "readonly" if readonly else "normal"

    combo = ttk.Combobox(
        frame,
        textvariable=var,
        values=values,
        width=width,
        state=state,
        bootstyle=bootstyle or "default"
    )

    combo.pack(side="left", fill="x", expand=True)

    # Optional default: select first item if var is empty
    if not var.get() and values:
        var.set(values[0])

    return frame


def text_entry(parent, var, label_text=None, width=20,
               bootstyle=None, placeholder=None, validate_func=None):
    """
    Reusable helper to create a labeled text Entry.

    Args:
        parent: parent widget
        var: tk.StringVar linked to the Entry
        label_text: optional label text
        width: width of the entry
        bootstyle: ttkbootstrap style (e.g., "default", "info", "danger")
        placeholder: optional placeholder text
        validate_func: optional validation callback accepting proposed text
    """
    frame = ttk.Frame(parent)

    if label_text:
        ttk.Label(frame, text=label_text).pack(side="left", padx=(0, 5))

    entry = ttk.Entry(
        frame,
        textvariable=var,
        width=width,
        bootstyle=bootstyle or "default"
    )

    # Optional placeholder behavior
    if placeholder:
        entry.insert(0, placeholder)
        entry.configure(foreground="grey")

        def _clear_placeholder(event):
            if entry.get() == placeholder:
                entry.delete(0, "end")
                entry.configure(foreground="white")

        def _restore_placeholder(event):
            if not entry.get():
                entry.insert(0, placeholder)
                entry.configure(foreground="grey")

        entry.bind("<FocusIn>", _clear_placeholder)
        entry.bind("<FocusOut>", _restore_placeholder)

    # Optional validation (Tk native validation)
    if validate_func:
        vcmd = (frame.register(validate_func), "%P")
        entry.configure(validate="key", validatecommand=vcmd)

    entry.pack(side="left", fill="x", expand=True)
    return frame
