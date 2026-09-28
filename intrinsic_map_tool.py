#!/usr/bin/env python3
"""
intrinsic_map_tool.py

Intrinsic command-map and form interface for relational_compiler.py.

The compiler remains the authority.

This tool imports relational_compiler.py, calls build_parser(), and derives:

    COMMANDS
    ARGUMENTS
    REQUIRED / OPTIONAL STATUS
    TYPES
    CHOICES
    DEFAULTS
    EXPECTED INPUT FORM
    FILLABLE FORM

directly from the compiler's argparse definitions.

Therefore:

    compiler command changes
            |
            v
    intrinsic map changes automatically

No second command definition is maintained here.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import tkinter as tk

from pathlib import Path
from tkinter import filedialog, messagebox, ttk


HERE = Path(__file__).resolve().parent
COMPILER_PATH = HERE / "relational_compiler.py"


INPUT_FORMAT_DOCS = {

    "registry": """
REGISTRY NPZ

Expected arrays:

ascii_code
    shape: (95,)
    dtype: integer
    required value:
        32, 33, ... 126

dimension_name
    shape: (437,)
    dtype: string

participation
    shape: (95, 437)
    dtype: bool / 0-1

Meaning:

The registry specifies which relational dimensions are available
for measurement for each printable ASCII character.

It does NOT specify embedding coordinates.
""",

    "evidence": """
EVIDENCE NPZ

Expected arrays:

y
    shape: (N,)
    observed correct printable ASCII code for each occurrence

g
    shape: (N, 437)
    relational measurements g[i,k]

    finite value:
        valid relational measurement

    NaN:
        measurement unavailable for that occurrence

w
    shape: (N,)
    nonnegative evidence weight w[i]

source_id
    shape: (N,)
    optional traceable occurrence identifier

Example occurrence:

    y[0] = 65

    character = 'A'

    w[0] = 1.0

    source_id[0] = "corpus.txt:1042"

    g[0] =
        [g_0, g_1, ... g_436]
""",

    "center_weight_npz": """
CENTER-WEIGHT NPZ

Expected array:

w_ck
    shape: (95, 437)
    dtype: float

Meaning:

w_ck[c,k] is the weight assigned to character c when calculating
the relational center and spread of dimension k.

Required only when:

    --center-weight external
""",

    "profile_npz": """
PROFILE NPZ

Expected array:

profile
    shape: (437,)
    dtype: float

Meaning:

A known-information relational measurement profile supplied
during inference.

Target identity is not part of this input.
""",

    "frozen_directory": """
FROZEN GEOMETRY DIRECTORY

Expected files:

geometry.npz
compile_manifest.json
frozen_manifest.json

geometry.npz contains:

G
H
lambda
rho
mu
sigma
E

plus supporting validation quantities.
""",

    "out": """
OUTPUT DIRECTORY

The compile command creates the frozen relational geometry here.

Expected result:

<directory>/
    geometry.npz
    compile_manifest.json
    frozen_manifest.json
""",
}


EXAMPLES = {

    "compile": (
        "python relational_compiler.py compile "
        "--registry canonical_registry.npz "
        "--evidence corpus_evidence.npz "
        "--out frozen_geometry "
        "--phi abs "
        "--center-weight occurrence_weight_mass"
    ),

    "validate-frozen": (
        "python relational_compiler.py "
        "validate-frozen frozen_geometry"
    ),

    "audit-coordinate": (
        "python relational_compiler.py "
        "audit-coordinate frozen_geometry "
        "--ascii 65 "
        "--dimension 17"
    ),

    "decode-profile": (
        "python relational_compiler.py "
        "decode-profile frozen_geometry "
        "--profile-npz profile.npz "
        "--mapper field_zscore "
        "--correspondence negative_squared_distance "
        "--top 10"
    ),

    "self-test": (
        "python relational_compiler.py self-test"
    ),
}


def load_compiler():

    if not COMPILER_PATH.exists():
        raise FileNotFoundError(
            f"Compiler not found:\n{COMPILER_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "relational_compiler_intrinsic",
        COMPILER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load relational_compiler.py"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # dataclasses resolves annotations through sys.modules while the
    # imported module is executing. Register it before exec_module().
    sys.modules[spec.name] = module

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "build_parser",
    ):
        raise RuntimeError(
            "relational_compiler.py does not expose build_parser()."
        )

    return module


def get_command_parsers(
    root_parser: argparse.ArgumentParser,
):

    for action in root_parser._actions:

        if isinstance(
            action,
            argparse._SubParsersAction,
        ):
            return action.choices

    raise RuntimeError(
        "No argparse subcommands were found."
    )


def useful_actions(
    parser: argparse.ArgumentParser,
):

    result = []

    for action in parser._actions:

        if action.dest == "help":
            continue

        if isinstance(
            action,
            argparse._SubParsersAction,
        ):
            continue

        result.append(
            action
        )

    return result


def action_flag(
    action: argparse.Action,
) -> str:

    if action.option_strings:
        return max(
            action.option_strings,
            key=len,
        )

    return action.dest


def action_required(
    action: argparse.Action,
) -> bool:

    if not action.option_strings:
        return True

    return bool(
        getattr(
            action,
            "required",
            False,
        )
    )


def type_name(
    action: argparse.Action,
) -> str:

    if isinstance(
        action,
        argparse._StoreTrueAction,
    ):
        return "boolean flag"

    if isinstance(
        action,
        argparse._StoreFalseAction,
    ):
        return "boolean flag"

    if action.type is None:
        return "text"

    return getattr(
        action.type,
        "__name__",
        str(action.type),
    )


def nargs_description(
    action: argparse.Action,
) -> str:

    nargs = action.nargs

    if nargs is None:
        return "one value"

    if nargs == "+":
        return "one or more values"

    if nargs == "*":
        return "zero or more values"

    if nargs == "?":
        return "zero or one value"

    return str(
        nargs
    )


def build_command_description(
    name: str,
    parser: argparse.ArgumentParser,
) -> str:

    lines = []

    lines.append(
        f"COMMAND\n{name}\n"
    )

    if parser.description:
        lines.append(
            "PURPOSE\n"
            + parser.description.strip()
            + "\n"
        )

    lines.append(
        "USAGE"
    )

    lines.append(
        parser.format_usage().strip()
    )

    lines.append(
        "\nEXPECTED INPUT"
    )

    for action in useful_actions(
        parser
    ):

        flag = action_flag(
            action
        )

        requirement = (
            "REQUIRED"
            if action_required(action)
            else "OPTIONAL"
        )

        lines.append(
            f"\n{flag}"
        )

        lines.append(
            f"    status:  {requirement}"
        )

        lines.append(
            f"    type:    {type_name(action)}"
        )

        lines.append(
            f"    values:  {nargs_description(action)}"
        )

        if action.choices:
            lines.append(
                "    choices: "
                + ", ".join(
                    str(x)
                    for x in action.choices
                )
            )

        default = getattr(
            action,
            "default",
            None,
        )

        if (
            default is not None
            and default is not argparse.SUPPRESS
        ):
            lines.append(
                f"    default: {default}"
            )

        if action.help:
            lines.append(
                f"    help:    {action.help}"
            )

        doc_key = action.dest

        if doc_key in INPUT_FORMAT_DOCS:
            lines.append(
                "\n"
                + INPUT_FORMAT_DOCS[
                    doc_key
                ].strip()
            )

    example = EXAMPLES.get(
        name
    )

    if example:
        lines.append(
            "\nEXAMPLE"
        )

        lines.append(
            example
        )

    return "\n".join(
        lines
    )


def shell_join(
    parts: list[str],
) -> str:

    if os.name == "nt":
        return subprocess.list2cmdline(
            parts
        )

    return shlex.join(
        parts
    )


class FormField:

    def __init__(
        self,
        action: argparse.Action,
        parent: ttk.Frame,
        row: int,
        changed_callback,
    ):

        self.action = action
        self.frame = parent
        self.changed_callback = changed_callback
        self.variable = None

        flag = action_flag(
            action
        )

        label_text = flag

        if action_required(
            action
        ):
            label_text += " *"

        ttk.Label(
            parent,
            text=label_text,
        ).grid(
            row=row,
            column=0,
            sticky="nw",
            padx=(4, 10),
            pady=5,
        )

        if isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
            ),
        ):

            self.variable = tk.BooleanVar(
                value=bool(
                    action.default
                )
            )

            widget = ttk.Checkbutton(
                parent,
                variable=self.variable,
                command=self.changed_callback,
            )

            widget.grid(
                row=row,
                column=1,
                sticky="w",
                pady=5,
            )

            return

        if action.choices:

            default = (
                ""
                if action.default in (
                    None,
                    argparse.SUPPRESS,
                )
                else str(
                    action.default
                )
            )

            self.variable = tk.StringVar(
                value=default
            )

            widget = ttk.Combobox(
                parent,
                textvariable=self.variable,
                values=[
                    str(x)
                    for x in action.choices
                ],
                state="readonly",
                width=43,
            )

            widget.grid(
                row=row,
                column=1,
                sticky="ew",
                pady=5,
            )

            widget.bind(
                "<<ComboboxSelected>>",
                lambda _event: (
                    self.changed_callback()
                ),
            )

            return

        default = ""

        if (
            action.default is not None
            and action.default is not argparse.SUPPRESS
        ):
            default = str(
                action.default
            )

        self.variable = tk.StringVar(
            value=default
        )

        entry = ttk.Entry(
            parent,
            textvariable=self.variable,
            width=46,
        )

        entry.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=5,
        )

        self.variable.trace_add(
            "write",
            lambda *_: (
                self.changed_callback()
            ),
        )

        if is_path_action(
            action
        ):

            ttk.Button(
                parent,
                text="Browse",
                command=self.browse,
            ).grid(
                row=row,
                column=2,
                padx=(6, 2),
                pady=5,
            )

    def browse(
        self,
    ):

        action = self.action

        if action.dest == "out":
            value = filedialog.askdirectory()

        elif action.dest == "frozen_directory":
            value = filedialog.askdirectory()

        else:
            value = filedialog.askopenfilename()

        if value:
            self.variable.set(
                value
            )

    def raw_value(
        self,
    ):
        return self.variable.get()


def is_path_action(
    action: argparse.Action,
) -> bool:

    dest = action.dest.lower()

    known = {
        "registry",
        "evidence",
        "out",
        "center_weight_npz",
        "profile_npz",
        "frozen_directory",
    }

    if dest in known:
        return True

    return (
        "path" in dest
        or "file" in dest
        or "directory" in dest
        or dest.endswith(
            "_npz"
        )
    )


class IntrinsicMapApp(
    tk.Tk
):

    def __init__(
        self,
    ):

        super().__init__()

        self.title(
            "Relational Compiler — Intrinsic Map"
        )

        self.geometry(
            "1350x820"
        )

        self.minsize(
            1000,
            650,
        )

        self.compiler = load_compiler()

        self.root_parser = (
            self.compiler.build_parser()
        )

        self.commands = (
            get_command_parsers(
                self.root_parser
            )
        )

        self.current_command = None
        self.form_fields: list[FormField] = []
        self.scan_root = tk.StringVar(
            value=str(HERE)
        )
        self.available_files: list[Path] = []

        self._build_interface()

        names = list(
            self.commands.keys()
        )

        if names:
            self.command_list.selection_set(
                0
            )

            self.select_command(
                names[0]
            )

        self.scan_files()

    def _build_interface(
        self,
    ):

        self.columnconfigure(
            0,
            weight=1,
        )

        self.rowconfigure(
            0,
            weight=1,
        )

        main = ttk.Panedwindow(
            self,
            orient=tk.HORIZONTAL,
        )

        main.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        left = ttk.Frame(
            main,
            padding=8,
        )

        main.add(
            left,
            weight=1,
        )

        ttk.Label(
            left,
            text="Compiler Commands",
            font=(
                "",
                12,
                "bold",
            ),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        self.command_list = tk.Listbox(
            left,
            exportselection=False,
        )

        self.command_list.pack(
            fill="both",
            expand=True,
        )

        for name in self.commands:
            self.command_list.insert(
                tk.END,
                name,
            )

        self.command_list.bind(
            "<<ListboxSelect>>",
            self._on_command_selected,
        )

        middle = ttk.Frame(
            main,
            padding=8,
        )

        main.add(
            middle,
            weight=3,
        )

        ttk.Label(
            middle,
            text="Intrinsic Command Map",
            font=(
                "",
                12,
                "bold",
            ),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        self.documentation = tk.Text(
            middle,
            wrap="word",
            font=(
                "Consolas",
                10,
            ),
        )

        self.documentation.pack(
            fill="both",
            expand=True,
        )

        self.documentation.configure(
            state="disabled"
        )

        right = ttk.Frame(
            main,
            padding=8,
        )

        main.add(
            right,
            weight=2,
        )

        ttk.Label(
            right,
            text="Command Form",
            font=(
                "",
                12,
                "bold",
            ),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        form_canvas = tk.Canvas(
            right,
            highlightthickness=0,
        )

        scrollbar = ttk.Scrollbar(
            right,
            orient="vertical",
            command=form_canvas.yview,
        )

        self.form_frame = ttk.Frame(
            form_canvas
        )

        self.form_frame.bind(
            "<Configure>",
            lambda _event: (
                form_canvas.configure(
                    scrollregion=(
                        form_canvas.bbox(
                            "all"
                        )
                    )
                )
            ),
        )

        form_canvas.create_window(
            (0, 0),
            window=self.form_frame,
            anchor="nw",
        )

        form_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        form_canvas.pack(
            side="top",
            fill="both",
            expand=True,
        )

        scrollbar.place(
            relx=1.0,
            rely=0.0,
            relheight=0.48,
            anchor="ne",
        )

        files_frame = ttk.LabelFrame(
            right,
            text="Available Files",
            padding=8,
        )

        files_frame.pack(
            fill="both",
            pady=(8, 0),
        )

        scan_toolbar = ttk.Frame(
            files_frame
        )

        scan_toolbar.pack(
            fill="x",
            pady=(0, 6),
        )

        ttk.Entry(
            scan_toolbar,
            textvariable=self.scan_root,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        ttk.Button(
            scan_toolbar,
            text="Folder",
            command=self.choose_scan_root,
        ).pack(
            side="left",
            padx=(6, 0),
        )

        ttk.Button(
            scan_toolbar,
            text="Scan",
            command=self.scan_files,
        ).pack(
            side="left",
            padx=(6, 0),
        )

        file_list_frame = ttk.Frame(
            files_frame
        )

        file_list_frame.pack(
            fill="both",
            expand=True,
        )

        file_scroll = ttk.Scrollbar(
            file_list_frame,
            orient="vertical",
        )

        self.available_file_list = tk.Listbox(
            file_list_frame,
            height=9,
            exportselection=False,
            yscrollcommand=file_scroll.set,
            font=(
                "Consolas",
                9,
            ),
        )

        file_scroll.configure(
            command=self.available_file_list.yview
        )

        self.available_file_list.pack(
            side="left",
            fill="both",
            expand=True,
        )

        file_scroll.pack(
            side="right",
            fill="y",
        )

        self.available_file_list.bind(
            "<Double-Button-1>",
            lambda _event: self.use_selected_file(),
        )

        file_buttons = ttk.Frame(
            files_frame
        )

        file_buttons.pack(
            fill="x",
            pady=(6, 0),
        )

        ttk.Button(
            file_buttons,
            text="Use Selected",
            command=self.use_selected_file,
        ).pack(
            side="left",
        )

        self.file_count = tk.StringVar(
            value="0 files"
        )

        ttk.Label(
            file_buttons,
            textvariable=self.file_count,
        ).pack(
            side="right",
        )

        generated = ttk.LabelFrame(
            right,
            text="Generated Command",
            padding=8,
        )

        generated.pack(
            fill="x",
            pady=(8, 0),
        )

        self.generated_command = tk.Text(
            generated,
            height=7,
            wrap="word",
            font=(
                "Consolas",
                10,
            ),
        )

        self.generated_command.pack(
            fill="x",
        )

        buttons = ttk.Frame(
            generated
        )

        buttons.pack(
            fill="x",
            pady=(8, 0),
        )

        ttk.Button(
            buttons,
            text="Generate",
            command=self.generate_command,
        ).pack(
            side="left",
        )

        ttk.Button(
            buttons,
            text="Validate",
            command=self.validate_form,
        ).pack(
            side="left",
            padx=6,
        )

        ttk.Button(
            buttons,
            text="Copy",
            command=self.copy_command,
        ).pack(
            side="left",
        )

        ttk.Button(
            buttons,
            text="Clear",
            command=self.clear_form,
        ).pack(
            side="right",
        )

        self.status = tk.StringVar(
            value="Ready"
        )

        ttk.Label(
            right,
            textvariable=self.status,
        ).pack(
            fill="x",
            pady=(6, 0),
        )

    def choose_scan_root(
        self,
    ):
        value = filedialog.askdirectory(
            initialdir=self.scan_root.get() or str(HERE)
        )

        if value:
            self.scan_root.set(value)
            self.scan_files()

    def classify_available_file(
        self,
        path: Path,
    ) -> str:
        name = path.name.lower()
        suffix = path.suffix.lower()

        if suffix in {".xlsx", ".xls"}:
            if (
                "ascii95" in name
                or "canonical" in name
                or "matrix" in name
            ):
                return "CANONICAL MATRIX"
            return "SPREADSHEET"

        if suffix == ".npz":
            if "registry" in name:
                return "REGISTRY NPZ"
            if "evidence" in name:
                return "EVIDENCE NPZ"
            if "profile" in name:
                return "PROFILE NPZ"
            if "weight" in name:
                return "CENTER-WEIGHT NPZ"
            return "NPZ"

        if suffix == ".py":
            return "PYTHON"

        if suffix in {".csv", ".tsv"}:
            return "TABLE"

        if suffix == ".json":
            return "JSON"

        if suffix in {".md", ".txt"}:
            return "TEXT"

        return "FILE"

    def scan_files(
        self,
    ):
        try:
            root = Path(
                self.scan_root.get()
            ).expanduser().resolve()

            if not root.exists():
                raise FileNotFoundError(root)

            if not root.is_dir():
                raise NotADirectoryError(root)

            discovered = [
                path
                for path in root.rglob("*")
                if path.is_file()
            ]

            discovered.sort(
                key=lambda path: (
                    self.classify_available_file(path),
                    str(path).lower(),
                )
            )

            self.available_files = discovered

            self.available_file_list.delete(
                0,
                tk.END,
            )

            for path in discovered:
                category = self.classify_available_file(path)

                try:
                    shown = path.relative_to(root)
                except ValueError:
                    shown = path

                self.available_file_list.insert(
                    tk.END,
                    f"{category:<18}  {shown}",
                )

            self.file_count.set(
                f"{len(discovered)} files"
            )

            self.status.set(
                f"Scanned {root}: {len(discovered)} files found."
            )

        except Exception as exc:
            self.available_files = []

            if hasattr(
                self,
                "available_file_list",
            ):
                self.available_file_list.delete(
                    0,
                    tk.END,
                )

            if hasattr(
                self,
                "file_count",
            ):
                self.file_count.set(
                    "0 files"
                )

            self.status.set(
                f"SCAN ERROR — {exc}"
            )

            messagebox.showerror(
                "Scan Error",
                str(exc),
            )

    def use_selected_file(
        self,
    ):
        selection = self.available_file_list.curselection()

        if not selection:
            messagebox.showinfo(
                "Select a file",
                "Select a file from Available Files first.",
            )
            return

        path = self.available_files[
            selection[0]
        ]

        suffix = path.suffix.lower()
        name = path.name.lower()

        preferred_destinations: list[str] = []

        if suffix == ".npz":
            if "registry" in name:
                preferred_destinations.append(
                    "registry"
                )

            if "evidence" in name:
                preferred_destinations.append(
                    "evidence"
                )

            if "profile" in name:
                preferred_destinations.append(
                    "profile_npz"
                )

            if "weight" in name:
                preferred_destinations.append(
                    "center_weight_npz"
                )

        if suffix in {".xlsx", ".xls"}:
            messagebox.showinfo(
                "Canonical spreadsheet detected",
                (
                    f"{path.name}\n\n"
                    "This is available and has been detected, but the "
                    "current compiler's --registry field requires an NPZ "
                    "registry rather than an Excel workbook."
                ),
            )
            return

        candidates = [
            field
            for field in self.form_fields
            if is_path_action(field.action)
            and field.action.dest != "out"
            and field.action.dest != "frozen_directory"
        ]

        target = None

        for destination in preferred_destinations:
            for field in candidates:
                if field.action.dest == destination:
                    target = field
                    break

            if target is not None:
                break

        if target is None:
            for field in candidates:
                if not str(
                    field.raw_value()
                ).strip():
                    target = field
                    break

        if target is None and candidates:
            target = candidates[0]

        if target is None:
            messagebox.showinfo(
                "No compatible field",
                (
                    "The selected command has no file-input field "
                    "that can accept this file."
                ),
            )
            return

        target.variable.set(
            str(path)
        )

        self.status.set(
            f"Inserted {path.name} into {action_flag(target.action)}."
        )

    def _on_command_selected(
        self,
        _event,
    ):

        selection = self.command_list.curselection()

        if not selection:
            return

        name = self.command_list.get(
            selection[0]
        )

        self.select_command(
            name
        )

    def select_command(
        self,
        name: str,
    ):

        self.current_command = name

        parser = self.commands[
            name
        ]

        description = (
            build_command_description(
                name,
                parser,
            )
        )

        self.documentation.configure(
            state="normal"
        )

        self.documentation.delete(
            "1.0",
            tk.END,
        )

        self.documentation.insert(
            "1.0",
            description,
        )

        self.documentation.configure(
            state="disabled"
        )

        self.build_form(
            parser
        )

        self.generate_command()

    def build_form(
        self,
        parser: argparse.ArgumentParser,
    ):

        for child in self.form_frame.winfo_children():
            child.destroy()

        self.form_fields.clear()

        actions = useful_actions(
            parser
        )

        self.form_frame.columnconfigure(
            1,
            weight=1,
        )

        for row, action in enumerate(
            actions
        ):

            field = FormField(
                action=action,
                parent=self.form_frame,
                row=row,
                changed_callback=self.generate_command,
            )

            self.form_fields.append(
                field
            )

    def parsed_values(
        self,
        require_required: bool,
    ):

        result = []
        missing = []

        for field in self.form_fields:

            action = field.action
            raw = field.raw_value()

            if isinstance(
                action,
                argparse._StoreTrueAction,
            ):

                if raw:
                    result.extend(
                        action.option_strings[
                            -1:
                        ]
                    )

                continue

            if isinstance(
                action,
                argparse._StoreFalseAction,
            ):

                if not raw:
                    result.extend(
                        action.option_strings[
                            -1:
                        ]
                    )

                continue

            text = str(
                raw
            ).strip()

            if not text:

                if (
                    require_required
                    and action_required(
                        action
                    )
                ):
                    missing.append(
                        action_flag(
                            action
                        )
                    )

                continue

            if action.nargs in (
                "+",
                "*",
            ):

                values = shlex.split(
                    text
                )

            else:
                values = [
                    text
                ]

            if action.type is not None:

                for value in values:
                    try:
                        action.type(
                            value
                        )

                    except Exception as exc:
                        raise ValueError(
                            f"{action_flag(action)} "
                            f"expects {type_name(action)}; "
                            f"received {value!r}"
                        ) from exc

            if action.choices:

                valid_choices = {
                    str(x)
                    for x in action.choices
                }

                for value in values:

                    if value not in valid_choices:
                        raise ValueError(
                            f"{action_flag(action)} "
                            f"must be one of "
                            f"{sorted(valid_choices)}"
                        )

            if action.option_strings:
                result.append(
                    action_flag(
                        action
                    )
                )

            result.extend(
                values
            )

        if missing:
            raise ValueError(
                "Missing required input:\n\n"
                + "\n".join(
                    missing
                )
            )

        return result

    def generate_command(
        self,
    ):

        if self.current_command is None:
            return

        try:

            values = self.parsed_values(
                require_required=False
            )

            parts = [
                "python",
                "relational_compiler.py",
                self.current_command,
            ]

            parts.extend(
                values
            )

            command = shell_join(
                parts
            )

            self.generated_command.delete(
                "1.0",
                tk.END,
            )

            self.generated_command.insert(
                "1.0",
                command,
            )

            self.status.set(
                "Command generated from compiler definition."
            )

        except Exception as exc:

            self.status.set(
                str(exc)
            )

    def validate_form(
        self,
    ):

        if self.current_command is None:
            return

        try:

            values = self.parsed_values(
                require_required=True
            )

            argv = [
                self.current_command,
                *values,
            ]

            self.root_parser.parse_args(
                argv
            )

            self.generate_command()

            self.status.set(
                "VALID — compiler accepts this command format."
            )

            messagebox.showinfo(
                "Valid",
                "The compiler's own parser accepts this command.",
            )

        except SystemExit:

            self.status.set(
                "INVALID — rejected by compiler parser."
            )

            messagebox.showerror(
                "Invalid",
                "The compiler rejected the generated command.",
            )

        except Exception as exc:

            self.status.set(
                f"INVALID — {exc}"
            )

            messagebox.showerror(
                "Invalid",
                str(exc),
            )

    def copy_command(
        self,
    ):

        value = self.generated_command.get(
            "1.0",
            tk.END,
        ).strip()

        if not value:
            return

        self.clipboard_clear()
        self.clipboard_append(value)

        self.status.set(
            "Command copied to clipboard."
        )

    def clear_form(
        self,
    ):

        for field in self.form_fields:

            action = field.action

            if isinstance(
                action,
                (
                    argparse._StoreTrueAction,
                    argparse._StoreFalseAction,
                ),
            ):

                field.variable.set(
                    bool(
                        action.default
                    )
                )

            else:

                default = getattr(
                    action,
                    "default",
                    None,
                )

                if (
                    default is None
                    or default is argparse.SUPPRESS
                ):

                    field.variable.set(
                        ""
                    )

                else:

                    field.variable.set(
                        str(default)
                    )

        self.generate_command()


def main():

    try:
        app = IntrinsicMapApp()

    except Exception as exc:

        root = tk.Tk()
        root.withdraw()

        messagebox.showerror(
            "Intrinsic Map Error",
            str(exc),
        )

        raise

    app.mainloop()


if __name__ == "__main__":
    main()
