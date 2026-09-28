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
import datetime
import threading
import queue
import sys
import tkinter as tk

import numpy as np

from pathlib import Path
from tkinter import filedialog, messagebox, ttk


HERE = Path(__file__).resolve().parent
COMPILER_PATH = HERE / "relational_compiler.py"

WEIGHT_COMPILER_PATH = HERE / "compile_weights.py"
CANONICAL_WORKBOOK_PATH = HERE / "data" / "Canonical_ASCII95_Attribute_Matrix_v1.xlsx"
LEGACY_ARCHITECTURE_PATH = HERE / "legacy" / "architecture.json"

MODEL_OBJECTIVES = (
    "LM_HOLE",
    "SFT_RESPONSE",
    "CHAT_ASSISTANT",
    "PROBLEM_SOLVING",
    "CODE_CAUSAL",
    "CODE_INSTRUCTION",
)



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

        generator_box = ttk.LabelFrame(
            left,
            text="Model Generator",
            padding=8,
        )

        generator_box.pack(
            fill="x",
            pady=(0, 10),
        )

        ttk.Label(
            generator_box,
            text="One clean dataset → weights run",
        ).pack(
            anchor="w",
            pady=(0, 6),
        )

        ttk.Button(
            generator_box,
            text="Open Model Generator",
            command=self.open_model_generator,
        ).pack(
            fill="x",
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

    def open_model_generator(
        self,
    ):
        window = tk.Toplevel(
            self
        )

        window.title(
            "Model Generator — One Clean Run"
        )

        window.geometry(
            "1080x820"
        )

        window.minsize(
            900,
            650,
        )

        window.columnconfigure(
            0,
            weight=1,
        )

        window.rowconfigure(
            1,
            weight=1,
        )

        header = ttk.Frame(
            window,
            padding=10,
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        ttk.Label(
            header,
            text="MODEL GENERATOR",
            font=(
                "",
                14,
                "bold",
            ),
        ).pack(
            anchor="w",
        )

        ttk.Label(
            header,
            text=(
                "Working control compiler: dataset → compiled weights → verification. "
                "This executable control is the repository's historical 254-D compiler. "
                "The canonical 437-D workbook is inspected but is not falsely treated as "
                "a completed 437-D Q/K/V/FFN compiler."
            ),
            wraplength=1020,
        ).pack(
            anchor="w",
            pady=(4, 0),
        )

        body = ttk.Frame(
            window,
            padding=10,
        )

        body.grid(
            row=1,
            column=0,
            sticky="nsew",
        )

        body.columnconfigure(
            1,
            weight=1,
        )

        body.rowconfigure(
            9,
            weight=1,
        )

        source_var = tk.StringVar()
        dataset_var = tk.StringVar()
        objective_var = tk.StringVar(
            value="CODE_CAUSAL"
        )
        output_root_var = tk.StringVar(
            value=str(
                HERE / "model_runs"
            )
        )

        workbook_var = tk.StringVar(
            value=str(
                CANONICAL_WORKBOOK_PATH
            )
        )

        compiler_var = tk.StringVar(
            value=str(
                WEIGHT_COMPILER_PATH
            )
        )

        architecture_var = tk.StringVar(
            value=str(
                LEGACY_ARCHITECTURE_PATH
            )
        )

        status_var = tk.StringVar(
            value="Ready."
        )

        run_dir_holder = {
            "path": None
        }

        log_queue: queue.Queue = queue.Queue()

        def add_path_row(
            row,
            label,
            variable,
            browse_command=None,
        ):
            ttk.Label(
                body,
                text=label,
            ).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=4,
            )

            ttk.Entry(
                body,
                textvariable=variable,
            ).grid(
                row=row,
                column=1,
                sticky="ew",
                pady=4,
            )

            if browse_command:
                ttk.Button(
                    body,
                    text="Browse",
                    command=browse_command,
                ).grid(
                    row=row,
                    column=2,
                    padx=(6, 0),
                    pady=4,
                )

        def choose_source():
            choice = messagebox.askyesnocancel(
                "Source selection",
                "Yes = choose a source file.\nNo = choose a source folder.\nCancel = do nothing.",
                parent=window,
            )

            if choice is None:
                return

            if choice:
                value = filedialog.askopenfilename(
                    parent=window
                )
            else:
                value = filedialog.askdirectory(
                    parent=window
                )

            if value:
                source_var.set(
                    value
                )

        def choose_dataset():
            value = filedialog.askopenfilename(
                parent=window,
                filetypes=[
                    (
                        "JSONL evidence",
                        "*.jsonl",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )

            if value:
                dataset_var.set(
                    value
                )

        def choose_output():
            value = filedialog.askdirectory(
                parent=window
            )

            if value:
                output_root_var.set(
                    value
                )

        add_path_row(
            0,
            "Source file/folder",
            source_var,
            choose_source,
        )

        add_path_row(
            1,
            "Evidence JSONL",
            dataset_var,
            choose_dataset,
        )

        ttk.Label(
            body,
            text="Objective",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=4,
        )

        ttk.Combobox(
            body,
            textvariable=objective_var,
            values=list(
                MODEL_OBJECTIVES
            ),
            state="readonly",
        ).grid(
            row=2,
            column=1,
            sticky="ew",
            pady=4,
        )

        add_path_row(
            3,
            "Output root",
            output_root_var,
            choose_output,
        )

        add_path_row(
            4,
            "Canonical 437 workbook",
            workbook_var,
            None,
        )

        add_path_row(
            5,
            "Weight compiler",
            compiler_var,
            None,
        )

        add_path_row(
            6,
            "Control architecture",
            architecture_var,
            None,
        )

        controls = ttk.Frame(
            body
        )

        controls.grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(10, 6),
        )

        log_frame = ttk.LabelFrame(
            body,
            text="Run Log",
            padding=6,
        )

        log_frame.grid(
            row=9,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=(8, 0),
        )

        log_frame.columnconfigure(
            0,
            weight=1,
        )

        log_frame.rowconfigure(
            0,
            weight=1,
        )

        log_text = tk.Text(
            log_frame,
            wrap="word",
            font=(
                "Consolas",
                9,
            ),
        )

        log_text.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        log_scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=log_text.yview,
        )

        log_scroll.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        log_text.configure(
            yscrollcommand=log_scroll.set
        )

        ttk.Label(
            body,
            textvariable=status_var,
        ).grid(
            row=8,
            column=0,
            columnspan=3,
            sticky="ew",
        )

        running = {
            "value": False
        }

        def append_log(
            text_value,
        ):
            log_text.insert(
                tk.END,
                text_value
                + (
                    ""
                    if text_value.endswith("\n")
                    else "\n"
                ),
            )

            log_text.see(
                tk.END
            )

        def poll_log():
            try:
                while True:
                    item = log_queue.get_nowait()

                    if item[0] == "line":
                        append_log(
                            item[1]
                        )

                    elif item[0] == "status":
                        status_var.set(
                            item[1]
                        )

                    elif item[0] == "done":
                        running["value"] = False
                        status_var.set(
                            item[1]
                        )

            except queue.Empty:
                pass

            if window.winfo_exists():
                window.after(
                    100,
                    poll_log,
                )

        def printable_ascii95(
            text_value,
        ):
            return all(
                32 <= ord(ch) <= 126
                for ch in text_value
            )

        def source_files(
            source: Path,
        ):
            if source.is_file():
                return [
                    source
                ]

            allowed = {
                ".py",
                ".txt",
                ".md",
                ".ps1",
                ".bat",
                ".cmd",
                ".c",
                ".h",
                ".cpp",
                ".hpp",
                ".js",
                ".ts",
            }

            skip = {
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "node_modules",
                "site-packages",
                "dist",
                "build",
            }

            result = []

            for directory, dirnames, filenames in os.walk(
                source
            ):
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name not in skip
                    and not name.startswith(".")
                ]

                for filename in filenames:
                    path = Path(
                        directory
                    ) / filename

                    if path.suffix.lower() in allowed:
                        result.append(
                            path
                        )

            result.sort(
                key=lambda p: str(p).lower()
            )

            return result

        def build_dataset_file():
            source_text = source_var.get().strip()

            if not source_text:
                raise ValueError(
                    "Choose a source file or folder, or choose an existing Evidence JSONL."
                )

            source = Path(
                source_text
            ).expanduser().resolve()

            if not source.exists():
                raise FileNotFoundError(
                    source
                )

            root = (
                source
                if source.is_dir()
                else source.parent
            )

            output_root = Path(
                output_root_var.get().strip()
            ).expanduser().resolve()

            output_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            dataset_path = (
                output_root
                / "generated_evidence.jsonl"
            )

            files = source_files(
                source
            )

            if not files:
                raise ValueError(
                    "No supported source files were found."
                )

            dataset_name = (
                source.name
                if source.name
                else "local-source"
            )

            objective = (
                objective_var.get()
            )

            record_count = 0
            character_count = 0

            with dataset_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as out:
                for path in files:
                    try:
                        relative = str(
                            path.relative_to(
                                root
                            )
                        )
                    except ValueError:
                        relative = path.name

                    with path.open(
                        "r",
                        encoding="utf-8",
                    ) as src:
                        for line_number, raw in enumerate(
                            src,
                            1,
                        ):
                            text_value = raw.rstrip(
                                "\r\n"
                            )

                            if not text_value:
                                continue

                            if not printable_ascii95(
                                text_value
                            ):
                                bad = next(
                                    (
                                        (
                                            index,
                                            ch,
                                            ord(ch),
                                        )
                                        for index, ch in enumerate(
                                            text_value
                                        )
                                        if not 32 <= ord(ch) <= 126
                                    ),
                                    None,
                                )

                                raise ValueError(
                                    f"{path}:{line_number} contains non-ASCII95 data "
                                    f"at character {bad}; no implicit normalization was performed."
                                )

                            record = {
                                "id": (
                                    f"{relative}:{line_number}"
                                ),
                                "dataset": dataset_name,
                                "objective": objective,
                                "text": text_value,
                                "provenance": {
                                    "source_file": relative,
                                    "line_number": line_number,
                                },
                            }

                            out.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                            record_count += 1
                            character_count += len(
                                text_value
                            )

            if record_count == 0:
                raise ValueError(
                    "No nonempty evidence records were produced."
                )

            dataset_var.set(
                str(dataset_path)
            )

            append_log(
                f"DATASET BUILT: {dataset_path}"
            )

            append_log(
                f"records={record_count} characters={character_count} objective={objective}"
            )

            return dataset_path

        def run_subprocess(
            arguments,
            label,
        ):
            log_queue.put(
                (
                    "line",
                    "\n=== "
                    + label
                    + " ===",
                )
            )

            log_queue.put(
                (
                    "line",
                    subprocess.list2cmdline(
                        [
                            str(x)
                            for x in arguments
                        ]
                    ),
                )
            )

            process = subprocess.Popen(
                [
                    str(x)
                    for x in arguments
                ],
                cwd=str(
                    HERE
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                log_queue.put(
                    (
                        "line",
                        line.rstrip(
                            "\n"
                        ),
                    )
                )

            code = process.wait()

            if code != 0:
                raise RuntimeError(
                    f"{label} failed with exit code {code}"
                )

        def require_base_files():
            for path, label in (
                (
                    Path(
                        compiler_var.get()
                    ),
                    "weight compiler",
                ),
                (
                    Path(
                        workbook_var.get()
                    ),
                    "canonical workbook",
                ),
                (
                    Path(
                        architecture_var.get()
                    ),
                    "control architecture",
                ),
            ):
                if not path.exists():
                    raise FileNotFoundError(
                        f"{label}: {path}"
                    )

        def create_run_directory():
            output_root = Path(
                output_root_var.get().strip()
            ).expanduser().resolve()

            output_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            stamp = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            run_dir = (
                output_root
                / f"run_{stamp}"
            )

            counter = 1

            while run_dir.exists():
                run_dir = (
                    output_root
                    / f"run_{stamp}_{counter}"
                )
                counter += 1

            return run_dir

        def do_inspect():
            require_base_files()

            run_subprocess(
                [
                    sys.executable,
                    compiler_var.get(),
                    "inspect",
                    workbook_var.get(),
                ],
                "INSPECT CANONICAL 437 MATRIX",
            )

        def do_compile():
            require_base_files()

            dataset_text = dataset_var.get().strip()

            if dataset_text:
                dataset = Path(
                    dataset_text
                ).expanduser().resolve()
            else:
                dataset = build_dataset_file()

            if not dataset.exists():
                raise FileNotFoundError(
                    dataset
                )

            run_dir = create_run_directory()

            run_subprocess(
                [
                    sys.executable,
                    compiler_var.get(),
                    "compile",
                    "--dataset",
                    str(dataset),
                    "--architecture",
                    architecture_var.get(),
                    "--output",
                    str(run_dir),
                ],
                "COMPILE WEIGHTS",
            )

            run_dir_holder[
                "path"
            ] = run_dir

            return run_dir

        def do_verify(
            run_dir=None,
        ):
            require_base_files()

            if run_dir is None:
                run_dir = run_dir_holder[
                    "path"
                ]

            if run_dir is None:
                value = filedialog.askopenfilename(
                    parent=window,
                    title="Select weights.npz",
                    filetypes=[
                        (
                            "NPZ weights",
                            "*.npz",
                        )
                    ],
                )

                if not value:
                    raise ValueError(
                        "No weights file selected."
                    )

                weights = Path(
                    value
                )

            else:
                weights = (
                    Path(run_dir)
                    / "weights.npz"
                )

            if not weights.exists():
                raise FileNotFoundError(
                    weights
                )

            run_subprocess(
                [
                    sys.executable,
                    compiler_var.get(),
                    "verify",
                    str(weights),
                ],
                "VERIFY WEIGHTS",
            )

        def launch(
            operation,
            label,
        ):
            if running[
                "value"
            ]:
                messagebox.showinfo(
                    "Run active",
                    "A model-generator operation is already running.",
                    parent=window,
                )
                return

            running[
                "value"
            ] = True

            status_var.set(
                label
            )

            def worker():
                try:
                    operation()

                    log_queue.put(
                        (
                            "done",
                            label
                            + " — DONE",
                        )
                    )

                except Exception as exc:
                    log_queue.put(
                        (
                            "line",
                            "ERROR: "
                            + str(exc),
                        )
                    )

                    log_queue.put(
                        (
                            "done",
                            label
                            + " — FAILED",
                        )
                    )

            threading.Thread(
                target=worker,
                daemon=True,
            ).start()

        def build_only():
            path = build_dataset_file()
            append_log(
                "READY DATASET: "
                + str(path)
            )

        def clean_run():
            require_base_files()

            dataset_text = dataset_var.get().strip()

            if dataset_text:
                dataset = Path(
                    dataset_text
                ).expanduser().resolve()

                if not dataset.exists():
                    raise FileNotFoundError(
                        dataset
                    )

            else:
                dataset = build_dataset_file()

            run_subprocess(
                [
                    sys.executable,
                    compiler_var.get(),
                    "inspect",
                    workbook_var.get(),
                ],
                "STEP 1 — INSPECT CANONICAL 437 MATRIX",
            )

            run_dir = create_run_directory()

            run_subprocess(
                [
                    sys.executable,
                    compiler_var.get(),
                    "compile",
                    "--dataset",
                    str(dataset),
                    "--architecture",
                    architecture_var.get(),
                    "--output",
                    str(run_dir),
                ],
                "STEP 2 — COMPILE WEIGHTS",
            )

            run_dir_holder[
                "path"
            ] = run_dir

            weights = (
                run_dir
                / "weights.npz"
            )

            run_subprocess(
                [
                    sys.executable,
                    compiler_var.get(),
                    "verify",
                    str(weights),
                ],
                "STEP 3 — VERIFY WEIGHTS",
            )

            append_log(
                "\nOUTPUT:"
            )

            append_log(
                str(run_dir)
            )

            append_log(
                "weights.npz"
            )

            append_log(
                "compile_report.json"
            )

            status_var.set(
                "ONE CLEAN RUN — VERIFIED"
            )

        ttk.Button(
            controls,
            text="Build Dataset",
            command=lambda: launch(
                build_only,
                "BUILD DATASET",
            ),
        ).pack(
            side="left",
            padx=(0, 6),
        )

        ttk.Button(
            controls,
            text="Inspect 437 Matrix",
            command=lambda: launch(
                do_inspect,
                "INSPECT MATRIX",
            ),
        ).pack(
            side="left",
            padx=(0, 6),
        )

        ttk.Button(
            controls,
            text="Compile Weights",
            command=lambda: launch(
                do_compile,
                "COMPILE WEIGHTS",
            ),
        ).pack(
            side="left",
            padx=(0, 6),
        )

        ttk.Button(
            controls,
            text="Verify Weights",
            command=lambda: launch(
                do_verify,
                "VERIFY WEIGHTS",
            ),
        ).pack(
            side="left",
            padx=(0, 6),
        )

        ttk.Button(
            controls,
            text="ONE CLEAN RUN",
            command=lambda: launch(
                clean_run,
                "ONE CLEAN RUN",
            ),
        ).pack(
            side="right",
        )

        poll_log()

    def choose_scan_root(
        self,
    ):
        value = filedialog.askdirectory(
            initialdir=self.scan_root.get() or str(HERE)
        )

        if value:
            self.scan_root.set(value)
            self.scan_files()

    def inspect_npz_kind(
        self,
        path: Path,
    ) -> str | None:
        try:
            with np.load(
                path,
                allow_pickle=False,
            ) as data:
                keys = set(
                    data.files
                )

                if {
                    "ascii_code",
                    "dimension_name",
                    "participation",
                }.issubset(keys):
                    if (
                        np.asarray(data["ascii_code"]).shape == (95,)
                        and np.asarray(data["dimension_name"]).shape == (437,)
                        and np.asarray(data["participation"]).shape == (95, 437)
                    ):
                        return "REGISTRY NPZ"

                if {
                    "y",
                    "g",
                    "w",
                }.issubset(keys):
                    g = np.asarray(
                        data["g"]
                    )

                    if (
                        g.ndim == 2
                        and g.shape[1] == 437
                    ):
                        return "EVIDENCE NPZ"

                if "profile" in keys:
                    if np.asarray(
                        data["profile"]
                    ).shape == (437,):
                        return "PROFILE NPZ"

                if "w_ck" in keys:
                    if np.asarray(
                        data["w_ck"]
                    ).shape == (95, 437):
                        return "CENTER-WEIGHT NPZ"

        except Exception:
            return None

        return None

    def classify_available_file(
        self,
        path: Path,
    ) -> str | None:
        if path.is_dir():
            required = {
                "geometry.npz",
                "compile_manifest.json",
                "frozen_manifest.json",
            }

            try:
                names = {
                    child.name
                    for child in path.iterdir()
                    if child.is_file()
                }
            except OSError:
                return None

            if required.issubset(
                names
            ):
                return "FROZEN GEOMETRY"

            return None

        name = path.name.lower()
        suffix = path.suffix.lower()

        if suffix == ".npz":
            return self.inspect_npz_kind(
                path
            )

        if suffix in {
            ".xlsx",
            ".xls",
        }:
            if (
                "ascii95" in name
                or "canonical" in name
                or "matrix" in name
            ):
                return "SOURCE MATRIX"

        return None

    def category_allowed_for_command(
        self,
        category: str,
    ) -> bool:
        command = self.current_command

        if command == "compile":
            return category in {
                "REGISTRY NPZ",
                "EVIDENCE NPZ",
                "CENTER-WEIGHT NPZ",
                "SOURCE MATRIX",
            }

        if command == "validate-frozen":
            return category == "FROZEN GEOMETRY"

        if command == "audit-coordinate":
            return category == "FROZEN GEOMETRY"

        if command == "decode-profile":
            return category in {
                "FROZEN GEOMETRY",
                "PROFILE NPZ",
            }

        return False

    def scan_files(
        self,
    ):
        try:
            root = Path(
                self.scan_root.get()
            ).expanduser().resolve()

            if not root.exists():
                raise FileNotFoundError(
                    root
                )

            if not root.is_dir():
                raise NotADirectoryError(
                    root
                )

            skip_names = {
                ".git",
                "__pycache__",
                "node_modules",
                ".venv",
                "venv",
                "site-packages",
                "dist",
                "build",
                ".mypy_cache",
                ".pytest_cache",
            }

            root_depth = len(
                root.parts
            )

            max_depth = 5
            max_results = 500

            discovered: list[
                tuple[str, Path]
            ] = []

            for directory, dirnames, filenames in os.walk(
                root
            ):
                current = Path(
                    directory
                )

                depth = (
                    len(current.parts)
                    - root_depth
                )

                dirnames[:] = [
                    name
                    for name in dirnames
                    if name not in skip_names
                    and not name.startswith(".")
                ]

                if depth >= max_depth:
                    dirnames[:] = []

                directory_category = (
                    self.classify_available_file(
                        current
                    )
                )

                if (
                    directory_category
                    and self.category_allowed_for_command(
                        directory_category
                    )
                ):
                    discovered.append(
                        (
                            directory_category,
                            current,
                        )
                    )

                wanted_suffixes = {
                    ".npz",
                    ".xlsx",
                    ".xls",
                }

                for filename in filenames:
                    path = current / filename

                    if (
                        path.suffix.lower()
                        not in wanted_suffixes
                    ):
                        continue

                    category = (
                        self.classify_available_file(
                            path
                        )
                    )

                    if category is None:
                        continue

                    if not self.category_allowed_for_command(
                        category
                    ):
                        continue

                    discovered.append(
                        (
                            category,
                            path,
                        )
                    )

                    if (
                        len(discovered)
                        >= max_results
                    ):
                        break

                if (
                    len(discovered)
                    >= max_results
                ):
                    break

            category_order = {
                "REGISTRY NPZ": 0,
                "EVIDENCE NPZ": 1,
                "CENTER-WEIGHT NPZ": 2,
                "PROFILE NPZ": 3,
                "FROZEN GEOMETRY": 4,
                "SOURCE MATRIX": 5,
            }

            discovered.sort(
                key=lambda item: (
                    category_order.get(
                        item[0],
                        99,
                    ),
                    str(
                        item[1]
                    ).lower(),
                )
            )

            self.available_files = [
                path
                for _category, path
                in discovered
            ]

            self.available_file_list.delete(
                0,
                tk.END,
            )

            for category, path in discovered:
                try:
                    shown = path.relative_to(
                        root
                    )
                except ValueError:
                    shown = path

                self.available_file_list.insert(
                    tk.END,
                    f"{category:<20}  {shown}",
                )

            self.file_count.set(
                f"{len(discovered)} usable"
            )

            if (
                len(discovered)
                >= max_results
            ):
                status = (
                    f"Showing first {max_results} compatible inputs "
                    f"under {root}. Choose a narrower project folder "
                    "for a complete list."
                )
            elif discovered:
                status = (
                    f"{len(discovered)} compatible inputs found "
                    f"for '{self.current_command}'."
                )
            elif self.current_command == "self-test":
                status = (
                    "self-test requires no input files."
                )
            else:
                status = (
                    f"No compatible inputs found for "
                    f"'{self.current_command}' under {root}."
                )

            self.status.set(
                status
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
                    "0 usable"
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
                "Select an input",
                "Select an input from the compatible-input list first.",
            )
            return

        path = self.available_files[
            selection[0]
        ]

        category = self.classify_available_file(
            path
        )

        if category == "SOURCE MATRIX":
            messagebox.showinfo(
                "Canonical source matrix",
                (
                    f"{path.name}\n\n"
                    "This is the canonical spreadsheet source. "
                    "The current --registry field cannot consume it "
                    "directly; it must first be converted to a REGISTRY NPZ."
                ),
            )
            return

        destination_by_category = {
            "REGISTRY NPZ": "registry",
            "EVIDENCE NPZ": "evidence",
            "CENTER-WEIGHT NPZ": "center_weight_npz",
            "PROFILE NPZ": "profile_npz",
            "FROZEN GEOMETRY": "frozen_directory",
        }

        destination = (
            destination_by_category.get(
                category
            )
        )

        if destination is None:
            messagebox.showinfo(
                "Unsupported input",
                "The selected item does not map to an input field.",
            )
            return

        target = next(
            (
                field
                for field in self.form_fields
                if field.action.dest
                == destination
            ),
            None,
        )

        if target is None:
            messagebox.showinfo(
                "No matching field",
                (
                    f"{category} is not used by the "
                    f"'{self.current_command}' command."
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

        if hasattr(
            self,
            "available_file_list",
        ):
            self.scan_files()

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
