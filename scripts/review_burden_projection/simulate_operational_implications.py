#****************************************
# MIT License
# Copyright (c) 2026 Jinying Chen
#  
# author(s): Jinying Chen, Boston University Chobanian & Avedisian School of Medicine
# date: 2026-8-15
# ver: 1.0
# 
# This code was written to support model evaluation for the 2026 paper published 
# in JMIR Medical Informatics. 
# The code is for research use only, and is provided as it is.
# 
# See LICENSE in the project root for license terms.

import argparse
import contextlib
import csv
import io
import json
import math
import os
import re
import struct
import zlib
from pathlib import Path


DEFAULT_PREVALENCES = [0.005, 0.01, 0.02, 0.05]
FIGURE_METRICS = [
    ("review_burden_tp_plus_fp", "Review burden (TP + FP)", "review_burden"),
    ("missed_errors_fn", "Missed errors (FN)", "missed_errors"),
]
MODEL_COLOR_PALETTE = [
    ("#2563eb", (37, 99, 235)),
    ("#be123c", (190, 18, 60)),
    ("#38bdf8", (56, 189, 248)),
    ("#ff5fb7", (255, 95, 183)),
    ("#047857", (4, 120, 87)),
    ("#111827", (17, 24, 39)),
    ("#7c3aed", (124, 58, 237)),
    ("#a16207", (161, 98, 7)),
    ("#475569", (71, 85, 105)),
]
MODEL_SUBSCRIPT_SUFFIXES = {
    "biowordvecml": ("BioWordVec", "ML"),
    "fasttextml": ("fastText", "ML"),
}
DEFAULT_ARGUMENTS = {
    "metrics": None,
    "output_file": None,
    "entries_per_batch": 1000,
    "prevalences": DEFAULT_PREVALENCES,
    "model_column": "model",
    "recall_column": "recall",
    "specificity_column": "specificity",
    "round_digits": 2,
    "make_figures": True,
    "no_figures": False,
    "figure_folder": None,
    "figure_mode": "per_model",
    "figure_models": None,
}
MATPLOTLIB_WARNING_SHOWN = False
MATPLOTLIB_RENDERER_NOTICE_SHOWN = False
MATPLOTLIB_UNAVAILABLE_REASON = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Project expected model outputs under assumed misspelling prevalences. "
            "Model-level recall and specificity must be provided in the JSON config."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config file containing the metrics list and output settings.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_args(cli_args):
    config = load_config(cli_args.config)
    merged = dict(DEFAULT_ARGUMENTS)
    merged.update({key: value for key, value in config.items() if value is not None})

    if not merged.get("metrics"):
        raise ValueError(
            "Config file must contain a non-empty 'metrics' list with model, recall, and specificity values."
        )
    if not isinstance(merged["metrics"], list):
        raise ValueError("Config field 'metrics' must be a list of model metric objects.")
    for index, row in enumerate(merged["metrics"], start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Config metric row {index} must be an object.")

    return argparse.Namespace(**merged)


def fieldnames_from_rows(rows):
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def normalize_metric(value, column_name):
    if is_missing(value):
        raise ValueError(f"Missing value in required metric column {column_name!r}")
    text = str(value).strip().replace("%", "")
    if not text:
        raise ValueError(f"Empty value in required metric column {column_name!r}")
    metric = float(text)
    if metric > 1.0:
        metric = metric / 100.0
    if metric < 0.0 or metric > 1.0:
        raise ValueError(
            f"Metric {column_name!r} must be between 0 and 1, or 0 and 100 percent; got {value!r}"
        )
    return metric


def normalize_prevalence(value):
    prevalence = float(value)
    if prevalence > 1.0:
        prevalence = prevalence / 100.0
    if prevalence < 0.0 or prevalence > 1.0:
        raise ValueError(
            f"Prevalence must be between 0 and 1, or 0 and 100 percent; got {value!r}"
        )
    return prevalence


def validate_columns(fieldnames, columns):
    available_columns = set(fieldnames)
    missing_columns = [column for column in columns if column not in available_columns]
    if missing_columns:
        raise ValueError(
            "Config metrics are missing required column(s): "
            + ", ".join(repr(column) for column in missing_columns)
        )


def project_model_outputs(metrics_df, args):
    rows = []
    prevalences = [normalize_prevalence(value) for value in args.prevalences]
    entries_per_batch = int(args.entries_per_batch)

    for row in metrics_df:
        model = str(row.get(args.model_column, "")).strip()
        if not model:
            continue

        recall = normalize_metric(row.get(args.recall_column), args.recall_column)
        specificity = normalize_metric(row.get(args.specificity_column), args.specificity_column)

        for prevalence in prevalences:
            expected_errors = entries_per_batch * prevalence
            expected_non_errors = entries_per_batch - expected_errors
            true_positives = expected_errors * recall
            false_negatives = expected_errors * (1.0 - recall)
            true_negatives = expected_non_errors * specificity
            false_positives = expected_non_errors * (1.0 - specificity)
            review_burden = true_positives + false_positives
            projected_precision = (
                true_positives / review_burden if review_burden > 0 else float("nan")
            )

            rows.append(
                {
                    "model": model,
                    "entries": entries_per_batch,
                    "prevalence": prevalence,
                    "prevalence_percent": prevalence * 100.0,
                    "observed_recall": recall,
                    "observed_specificity": specificity,
                    "expected_true_errors": expected_errors,
                    "expected_non_errors": expected_non_errors,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "true_negatives": true_negatives,
                    "review_burden_tp_plus_fp": review_burden,
                    "missed_errors_fn": false_negatives,
                    "projected_precision": projected_precision,
                }
            )

    count_columns = [
        "expected_true_errors",
        "expected_non_errors",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "review_burden_tp_plus_fp",
        "missed_errors_fn",
    ]
    metric_columns = [
        "prevalence",
        "prevalence_percent",
        "observed_recall",
        "observed_specificity",
        "projected_precision",
    ]
    for row in rows:
        for column in count_columns:
            row[column] = round(row[column], args.round_digits)
        for column in metric_columns:
            row[column] = round(row[column], 6)

    return rows


def infer_output_file(output_file):
    if output_file:
        return Path(output_file)
    return Path("operational_projection.tsv")


def sanitize_filename(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "model"


def escape_svg_text(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def model_label_parts(model):
    model_text = str(model)
    return MODEL_SUBSCRIPT_SUFFIXES.get(model_text.lower(), (model_text, ""))


def model_label_svg(model):
    base_label, subscript_label = model_label_parts(model)
    escaped_base = escape_svg_text(base_label)
    if not subscript_label:
        return escaped_base
    return (
        f"{escaped_base}"
        f'<tspan baseline-shift="sub" font-size="70%">'
        f"{escape_svg_text(subscript_label)}</tspan>"
    )


def model_label_matplotlib(model):
    base_label, subscript_label = model_label_parts(model)
    if not subscript_label:
        return str(base_label)
    escaped_subscript = re.sub(r"([_{}\\$])", r"\\\1", str(subscript_label))
    return f"{base_label}$_{{\\mathrm{{{escaped_subscript}}}}}$"


def configure_matplotlib_cache(figure_folder):
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = Path(figure_folder) / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def load_matplotlib_pyplot():
    global MATPLOTLIB_UNAVAILABLE_REASON
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception as error:
        error_message = str(error).splitlines()[0]
        if len(error_message) > 220:
            error_message = error_message[:217] + "..."
        MATPLOTLIB_UNAVAILABLE_REASON = f"{type(error).__name__}: {error_message}"
        return None
    MATPLOTLIB_UNAVAILABLE_REASON = None
    return plt


def note_matplotlib_renderer_once():
    global MATPLOTLIB_RENDERER_NOTICE_SHOWN
    if MATPLOTLIB_RENDERER_NOTICE_SHOWN:
        return
    print("Using matplotlib renderer for PNG figures")
    MATPLOTLIB_RENDERER_NOTICE_SHOWN = True


def warn_matplotlib_unavailable_once():
    global MATPLOTLIB_WARNING_SHOWN
    if MATPLOTLIB_WARNING_SHOWN:
        return
    reason = ""
    if MATPLOTLIB_UNAVAILABLE_REASON:
        reason = f" Reason: {MATPLOTLIB_UNAVAILABLE_REASON}"
    message = (
        "Warning: matplotlib is not installed in the active Python environment. "
        "PNG figures will use the lower-quality built-in renderer; install matplotlib "
        "or run with an environment that includes matplotlib for publication-quality PNGs."
    )
    if MATPLOTLIB_UNAVAILABLE_REASON and "No module named" not in MATPLOTLIB_UNAVAILABLE_REASON:
        message = (
            "Warning: matplotlib could not be initialized in the active Python environment. "
            "PNG figures will use the lower-quality built-in renderer. "
            "If matplotlib is installed, check that it can access a writable cache/temp "
            "directory, such as MPLCONFIGDIR."
        )
    print(message + reason)
    MATPLOTLIB_WARNING_SHOWN = True


def text_width(text, scale=2):
    return len(str(text)) * 6 * scale


def draw_model_label_png(canvas, x, y, model, color, scale=2):
    base_label, subscript_label = model_label_parts(model)
    draw_text_preserve_case(canvas, x, y, base_label, color, scale=scale)
    if subscript_label:
        subscript_x = x + text_width(base_label, scale=scale)
        subscript_y = y + 4 * scale
        draw_text_preserve_case(
            canvas,
            subscript_x,
            subscript_y,
            subscript_label,
            color,
            scale=max(1, scale - 1),
        )


def group_rows_by_model(projection_rows):
    grouped = {}
    for row in projection_rows:
        grouped.setdefault(row["model"], []).append(row)
    for model_rows in grouped.values():
        model_rows.sort(key=lambda item: float(item["prevalence"]))
    return grouped


def parse_figure_models(raw_models):
    if not raw_models:
        return None
    models = []
    for value in raw_models:
        models.extend(part.strip() for part in str(value).split(","))
    return [model for model in models if model]


def filter_projection_rows_for_figures(projection_rows, figure_models):
    selected_models = parse_figure_models(figure_models)
    if not selected_models:
        return list(projection_rows)

    available_models = {row["model"] for row in projection_rows}
    selected_model_set = set(selected_models)
    missing_models = [model for model in selected_models if model not in available_models]
    if missing_models:
        print(
            "Warning: requested figure model(s) not found and skipped: "
            + ", ".join(missing_models)
        )

    return [row for row in projection_rows if row["model"] in selected_model_set]


def get_color(index):
    return MODEL_COLOR_PALETTE[index % len(MODEL_COLOR_PALETTE)]


def get_metric_config(metric_name):
    for column, label, file_label in FIGURE_METRICS:
        if metric_name == column:
            return label, file_label
    return metric_name, sanitize_filename(metric_name)


def format_count(value):
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def build_polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def create_canvas(width, height, color):
    return [[list(color) for _ in range(width)] for _ in range(height)]


def set_pixel(canvas, x, y, color):
    height = len(canvas)
    width = len(canvas[0]) if height else 0
    if 0 <= x < width and 0 <= y < height:
        canvas[y][x] = list(color)


def draw_rect(canvas, x1, y1, x2, y2, color):
    for y in range(max(0, y1), min(len(canvas), y2 + 1)):
        for x in range(max(0, x1), min(len(canvas[0]), x2 + 1)):
            set_pixel(canvas, x, y, color)


def draw_line(canvas, x1, y1, x2, y2, color, thickness=1):
    x1, y1, x2, y2 = map(lambda value: int(round(value)), [x1, y1, x2, y2])
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    error = dx + dy
    x, y = x1, y1
    radius = max(0, thickness // 2)
    while True:
        draw_rect(canvas, x - radius, y - radius, x + radius, y + radius, color)
        if x == x2 and y == y2:
            break
        e2 = 2 * error
        if e2 >= dy:
            error += dy
            x += sx
        if e2 <= dx:
            error += dx
            y += sy


def draw_circle(canvas, cx, cy, radius, color):
    cx, cy, radius = int(round(cx)), int(round(cy)), int(round(radius))
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                set_pixel(canvas, x, y, color)


def draw_text(canvas, x, y, text, color, scale=2):
    cursor_x = int(round(x))
    cursor_y = int(round(y))
    for char in str(text).upper():
        glyph = FONT_5X7.get(char, FONT_5X7[" "])
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == "1":
                    draw_rect(
                        canvas,
                        cursor_x + col_index * scale,
                        cursor_y + row_index * scale,
                        cursor_x + (col_index + 1) * scale - 1,
                        cursor_y + (row_index + 1) * scale - 1,
                        color,
                    )
        cursor_x += 6 * scale


def lowercase_glyph(uppercase_char):
    glyph = FONT_5X7[uppercase_char]
    return ["00000"] + glyph[:-1]


FONT_5X7_PRESERVE_CASE = dict(FONT_5X7)
for lowercase_char in "abcdefghijklmnopqrstuvwxyz":
    FONT_5X7_PRESERVE_CASE[lowercase_char] = lowercase_glyph(lowercase_char.upper())


def draw_text_preserve_case(canvas, x, y, text, color, scale=2):
    cursor_x = int(round(x))
    cursor_y = int(round(y))
    for char in str(text):
        glyph = FONT_5X7_PRESERVE_CASE.get(char, FONT_5X7_PRESERVE_CASE[" "])
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == "1":
                    draw_rect(
                        canvas,
                        cursor_x + col_index * scale,
                        cursor_y + row_index * scale,
                        cursor_x + (col_index + 1) * scale - 1,
                        cursor_y + (row_index + 1) * scale - 1,
                        color,
                    )
        cursor_x += 6 * scale


def png_chunk(chunk_type, data):
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def encode_png(canvas):
    height = len(canvas)
    width = len(canvas[0]) if height else 0
    raw_rows = []
    for row in canvas:
        raw_rows.append(b"\x00" + b"".join(bytes(pixel) for pixel in row))
    raw_data = b"".join(raw_rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw_data, level=9))
        + png_chunk(b"IEND", b"")
    )


def figure_to_png_bytes(plt, fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def render_model_png_matplotlib(model, rows):
    plt = load_matplotlib_pyplot()
    if plt is None:
        return None

    prevalences = [float(row["prevalence_percent"]) for row in rows]
    prevalence_labels = [f"{prevalence:g}%" for prevalence in prevalences]
    review_values = [float(row["review_burden_tp_plus_fp"]) for row in rows]
    missed_values = [float(row["missed_errors_fn"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(
        prevalences,
        review_values,
        color="#047857",
        marker="o",
        linewidth=2.8,
        markersize=6,
        label="Review burden (TP + FP)",
    )
    ax.plot(
        prevalences,
        missed_values,
        color="#1d4ed8",
        marker="o",
        linewidth=2.8,
        markersize=6,
        label="Missed errors (FN)",
    )

    ax.set_title(
        f"Operational projection: {model_label_matplotlib(model)}",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=24,
    )
    ax.text(
        0,
        1.02,
        f"Expected counts per {rows[0]['entries']} medication entries",
        transform=ax.transAxes,
        fontsize=13,
        color="#52616b",
    )
    ax.set_xlabel("Assumed misspelling prevalence", fontsize=14, labelpad=10)
    ax.set_ylabel("Expected count", fontsize=14, labelpad=10)
    ax.set_xticks(prevalences)
    ax.set_xticklabels(prevalence_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#d9e2ec", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=13, loc="upper right")

    max_value = max(review_values + missed_values + [1.0])
    ax.set_ylim(bottom=0, top=max_value * 1.12)
    fig.tight_layout()
    return figure_to_png_bytes(plt, fig)


def render_metric_png_matplotlib(metric_name, metric_label, grouped_rows):
    plt = load_matplotlib_pyplot()
    if plt is None:
        return None

    all_rows = [row for rows in grouped_rows.values() for row in rows]
    prevalences = sorted({float(row["prevalence_percent"]) for row in all_rows})
    prevalence_labels = [f"{prevalence:g}%" for prevalence in prevalences]
    values = [float(row[metric_name]) for row in all_rows]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for model_index, (model, rows) in enumerate(grouped_rows.items()):
        color_hex, _ = get_color(model_index)
        x_values = [float(row["prevalence_percent"]) for row in rows]
        y_values = [float(row[metric_name]) for row in rows]
        ax.plot(
            x_values,
            y_values,
            color=color_hex,
            marker="o",
            linewidth=2.6,
            markersize=5.5,
            label=model_label_matplotlib(model),
        )

    ax.set_title(
        f"Operational projection: {metric_label}",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=24,
    )
    ax.text(
        0,
        1.02,
        "All selected models across assumed misspelling prevalences",
        transform=ax.transAxes,
        fontsize=13,
        color="#52616b",
    )
    ax.set_xlabel("Assumed misspelling prevalence", fontsize=14, labelpad=10)
    ax.set_ylabel("Expected count", fontsize=14, labelpad=10)
    ax.set_xticks(prevalences)
    ax.set_xticklabels(prevalence_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#d9e2ec", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=13, bbox_to_anchor=(1.02, 1), loc="upper left")

    max_value = max(values + [1.0])
    ax.set_ylim(bottom=0, top=max_value * 1.12)
    fig.tight_layout()
    return figure_to_png_bytes(plt, fig)


def render_model_png(model, rows):
    png_bytes = render_model_png_matplotlib(model, rows)
    if png_bytes is not None:
        note_matplotlib_renderer_once()
        return png_bytes
    warn_matplotlib_unavailable_once()
    return render_model_png_builtin(model, rows)


def render_metric_png(metric_name, metric_label, grouped_rows):
    png_bytes = render_metric_png_matplotlib(metric_name, metric_label, grouped_rows)
    if png_bytes is not None:
        note_matplotlib_renderer_once()
        return png_bytes
    warn_matplotlib_unavailable_once()
    return render_metric_png_builtin(metric_name, metric_label, grouped_rows)


def render_model_png_builtin(model, rows):
    width = 940
    height = 540
    margin_left = 84
    margin_right = 42
    margin_top = 92
    margin_bottom = 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_count = max(1, len(rows) - 1)

    background = (255, 255, 255)
    axis_color = (47, 58, 69)
    grid_color = (217, 226, 236)
    text_color = (31, 41, 51)
    muted_text = (82, 97, 107)
    review_color = (4, 120, 87)
    missed_color = (29, 78, 216)

    review_values = [float(row["review_burden_tp_plus_fp"]) for row in rows]
    missed_values = [float(row["missed_errors_fn"]) for row in rows]
    max_value = max(review_values + missed_values + [1.0])
    y_axis_max = math.ceil(max_value / 10.0) * 10.0 if max_value > 10 else math.ceil(max_value)

    def x_at(index):
        if len(rows) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / x_count)

    def y_at(value):
        return margin_top + plot_height - (float(value) / y_axis_max * plot_height)

    canvas = create_canvas(width, height, background)
    draw_text(canvas, margin_left, 28, "OPERATIONAL PROJECTION: ", text_color, scale=2)
    draw_model_label_png(
        canvas,
        margin_left + text_width("OPERATIONAL PROJECTION: ", scale=2),
        28,
        model,
        text_color,
        scale=2,
    )
    draw_text(canvas, margin_left, 56, f"EXPECTED COUNTS PER {rows[0]['entries']} ENTRIES", muted_text, scale=2)

    legend_x = width - margin_right - 300
    draw_line(canvas, legend_x, 34, legend_x + 28, 34, review_color, thickness=3)
    draw_text(canvas, legend_x + 38, 24, "REVIEW BURDEN (TP+FP)", text_color, scale=2)
    draw_line(canvas, legend_x, 56, legend_x + 28, 56, missed_color, thickness=3)
    draw_text(canvas, legend_x + 38, 46, "MISSED ERRORS (FN)", text_color, scale=2)

    y_ticks = [0, y_axis_max * 0.25, y_axis_max * 0.5, y_axis_max * 0.75, y_axis_max]
    for tick in y_ticks:
        y = y_at(tick)
        draw_line(canvas, margin_left, y, width - margin_right, y, grid_color)
        draw_text(canvas, 12, y - 8, format_count(tick), text_color, scale=2)

    draw_line(canvas, margin_left, margin_top, margin_left, height - margin_bottom, axis_color, thickness=2)
    draw_line(canvas, margin_left, height - margin_bottom, width - margin_right, height - margin_bottom, axis_color, thickness=2)
    draw_text(canvas, width / 2 - 178, height - 40, "ASSUMED MISSPELLING PREVALENCE", text_color, scale=2)
    draw_text(canvas, 12, 76, "EXPECTED COUNT", text_color, scale=2)

    review_points = [(x_at(index), y_at(value)) for index, value in enumerate(review_values)]
    missed_points = [(x_at(index), y_at(value)) for index, value in enumerate(missed_values)]
    for point_a, point_b in zip(review_points, review_points[1:]):
        draw_line(canvas, point_a[0], point_a[1], point_b[0], point_b[1], review_color, thickness=3)
    for point_a, point_b in zip(missed_points, missed_points[1:]):
        draw_line(canvas, point_a[0], point_a[1], point_b[0], point_b[1], missed_color, thickness=3)

    for index, row in enumerate(rows):
        x = x_at(index)
        prevalence_label = f'{float(row["prevalence_percent"]):g}%'
        draw_line(canvas, x, height - margin_bottom, x, height - margin_bottom + 6, axis_color)
        draw_text(canvas, x - 18, height - margin_bottom + 16, prevalence_label, text_color, scale=2)

        review_y = y_at(row["review_burden_tp_plus_fp"])
        missed_y = y_at(row["missed_errors_fn"])
        draw_circle(canvas, x, review_y, 5, review_color)
        draw_text(canvas, x - 18, review_y - 26, format_count(row["review_burden_tp_plus_fp"]), review_color, scale=2)
        draw_circle(canvas, x, missed_y, 5, missed_color)
        draw_text(canvas, x - 18, missed_y + 14, format_count(row["missed_errors_fn"]), missed_color, scale=2)

    return encode_png(canvas)


def render_model_svg(model, rows):
    width = 940
    height = 540
    margin_left = 84
    margin_right = 42
    margin_top = 72
    margin_bottom = 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_count = max(1, len(rows) - 1)

    review_values = [float(row["review_burden_tp_plus_fp"]) for row in rows]
    missed_values = [float(row["missed_errors_fn"]) for row in rows]
    max_value = max(review_values + missed_values + [1.0])
    y_axis_max = math.ceil(max_value / 10.0) * 10.0 if max_value > 10 else math.ceil(max_value)

    def x_at(index):
        if len(rows) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / x_count)

    def y_at(value):
        return margin_top + plot_height - (float(value) / y_axis_max * plot_height)

    review_points = [(x_at(index), y_at(value)) for index, value in enumerate(review_values)]
    missed_points = [(x_at(index), y_at(value)) for index, value in enumerate(missed_values)]

    y_ticks = [0, y_axis_max * 0.25, y_axis_max * 0.5, y_axis_max * 0.75, y_axis_max]
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #52616b; }",
        ".axis { stroke: #2f3a45; stroke-width: 1.2; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".review { fill: none; stroke: #047857; stroke-width: 3; }",
        ".missed { fill: none; stroke: #1d4ed8; stroke-width: 3; }",
        ".review-dot { fill: #047857; }",
        ".missed-dot { fill: #1d4ed8; }",
        ".label { font-size: 16px; }",
        ".legend { font-size: 17px; font-weight: 600; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="34" class="title">Operational projection: {model_label_svg(model)}</text>',
        f'<text x="{margin_left}" y="56" class="subtitle">Expected review burden and missed errors per {rows[0]["entries"]} medication entries</text>',
    ]

    for tick in y_ticks:
        y = y_at(tick)
        elements.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" class="grid"/>')
        elements.append(f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" class="label">{format_count(tick)}</text>')

    elements.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="axis"/>',
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis"/>',
            f'<text x="{width / 2:.2f}" y="{height - 26}" text-anchor="middle" class="label">Assumed misspelling prevalence</text>',
            f'<text x="22" y="{height / 2:.2f}" transform="rotate(-90 22 {height / 2:.2f})" text-anchor="middle" class="label">Expected count</text>',
        ]
    )

    elements.append(f'<polyline points="{build_polyline(review_points)}" class="review"/>')
    elements.append(f'<polyline points="{build_polyline(missed_points)}" class="missed"/>')

    for index, row in enumerate(rows):
        x = x_at(index)
        prevalence_label = f'{float(row["prevalence_percent"]):g}%'
        elements.append(f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}" class="axis"/>')
        elements.append(f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="label">{prevalence_label}</text>')

        review_y = y_at(row["review_burden_tp_plus_fp"])
        missed_y = y_at(row["missed_errors_fn"])
        elements.append(f'<circle cx="{x:.2f}" cy="{review_y:.2f}" r="5" class="review-dot"/>')
        elements.append(f'<text x="{x:.2f}" y="{review_y - 10:.2f}" text-anchor="middle" class="label">{format_count(row["review_burden_tp_plus_fp"])}</text>')
        elements.append(f'<circle cx="{x:.2f}" cy="{missed_y:.2f}" r="5" class="missed-dot"/>')
        elements.append(f'<text x="{x:.2f}" y="{missed_y + 20:.2f}" text-anchor="middle" class="label">{format_count(row["missed_errors_fn"])}</text>')

    legend_x = width - margin_right - 310
    legend_y = 34
    elements.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" class="review"/>',
            f'<text x="{legend_x + 38}" y="{legend_y + 4}" class="legend">Review burden (TP + FP)</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 24}" x2="{legend_x + 28}" y2="{legend_y + 24}" class="missed"/>',
            f'<text x="{legend_x + 38}" y="{legend_y + 28}" class="legend">Missed errors (FN)</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def get_prevalence_positions(rows, margin_left, plot_width):
    prevalences = sorted({float(row["prevalence"]) for row in rows})
    x_count = max(1, len(prevalences) - 1)
    positions = {}
    for index, prevalence in enumerate(prevalences):
        if len(prevalences) == 1:
            positions[prevalence] = margin_left + plot_width / 2
        else:
            positions[prevalence] = margin_left + (plot_width * index / x_count)
    return prevalences, positions


def render_metric_svg(metric_name, metric_label, grouped_rows):
    width = 1040
    height = 560
    margin_left = 84
    margin_right = 270
    margin_top = 78
    margin_bottom = 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_rows = [row for rows in grouped_rows.values() for row in rows]
    prevalences, x_positions = get_prevalence_positions(all_rows, margin_left, plot_width)
    values = [float(row[metric_name]) for row in all_rows]
    max_value = max(values + [1.0])
    y_axis_max = math.ceil(max_value / 10.0) * 10.0 if max_value > 10 else math.ceil(max_value)

    def y_at(value):
        return margin_top + plot_height - (float(value) / y_axis_max * plot_height)

    y_ticks = [0, y_axis_max * 0.25, y_axis_max * 0.5, y_axis_max * 0.75, y_axis_max]
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2933; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #52616b; }",
        ".axis { stroke: #2f3a45; stroke-width: 1.2; }",
        ".grid { stroke: #d9e2ec; stroke-width: 1; }",
        ".label { font-size: 16px; }",
        ".legend { font-size: 16px; font-weight: 600; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="34" class="title">Operational projection: {escape_svg_text(metric_label)}</text>',
        '<text x="{margin_left}" y="56" class="subtitle">All selected models across assumed misspelling prevalences</text>',
    ]

    for tick in y_ticks:
        y = y_at(tick)
        elements.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" class="grid"/>')
        elements.append(f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" class="label">{format_count(tick)}</text>')

    elements.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="axis"/>',
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis"/>',
            f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 26}" text-anchor="middle" class="label">Assumed misspelling prevalence</text>',
            f'<text x="22" y="{height / 2:.2f}" transform="rotate(-90 22 {height / 2:.2f})" text-anchor="middle" class="label">Expected count</text>',
        ]
    )

    for prevalence in prevalences:
        x = x_positions[prevalence]
        prevalence_label = f"{prevalence * 100:g}%"
        elements.append(f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}" class="axis"/>')
        elements.append(f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" class="label">{prevalence_label}</text>')

    legend_x = width - margin_right + 28
    legend_y = margin_top
    for model_index, (model, rows) in enumerate(grouped_rows.items()):
        color_hex, _ = get_color(model_index)
        points = [
            (x_positions[float(row["prevalence"])], y_at(row[metric_name]))
            for row in rows
        ]
        if points:
            elements.append(
                f'<polyline points="{build_polyline(points)}" fill="none" stroke="{color_hex}" stroke-width="3"/>'
            )
        for x, y in points:
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color_hex}"/>')
        current_legend_y = legend_y + model_index * 22
        elements.append(f'<line x1="{legend_x}" y1="{current_legend_y}" x2="{legend_x + 26}" y2="{current_legend_y}" stroke="{color_hex}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 36}" y="{current_legend_y + 4}" class="legend">{model_label_svg(model)}</text>')

    elements.append("</svg>")
    return "\n".join(elements)


def render_metric_png_builtin(metric_name, metric_label, grouped_rows):
    width = 1040
    height = 560
    margin_left = 84
    margin_right = 270
    margin_top = 98
    margin_bottom = 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_rows = [row for rows in grouped_rows.values() for row in rows]
    prevalences, x_positions = get_prevalence_positions(all_rows, margin_left, plot_width)
    values = [float(row[metric_name]) for row in all_rows]
    max_value = max(values + [1.0])
    y_axis_max = math.ceil(max_value / 10.0) * 10.0 if max_value > 10 else math.ceil(max_value)

    background = (255, 255, 255)
    axis_color = (47, 58, 69)
    grid_color = (217, 226, 236)
    text_color = (31, 41, 51)
    muted_text = (82, 97, 107)

    def y_at(value):
        return margin_top + plot_height - (float(value) / y_axis_max * plot_height)

    canvas = create_canvas(width, height, background)
    draw_text(canvas, margin_left, 28, f"OPERATIONAL PROJECTION: {metric_label}", text_color, scale=2)
    draw_text(canvas, margin_left, 56, "ALL SELECTED MODELS", muted_text, scale=2)

    y_ticks = [0, y_axis_max * 0.25, y_axis_max * 0.5, y_axis_max * 0.75, y_axis_max]
    for tick in y_ticks:
        y = y_at(tick)
        draw_line(canvas, margin_left, y, width - margin_right, y, grid_color)
        draw_text(canvas, 12, y - 8, format_count(tick), text_color, scale=2)

    draw_line(canvas, margin_left, margin_top, margin_left, height - margin_bottom, axis_color, thickness=2)
    draw_line(canvas, margin_left, height - margin_bottom, width - margin_right, height - margin_bottom, axis_color, thickness=2)
    draw_text(canvas, margin_left + plot_width / 2 - 178, height - 40, "ASSUMED MISSPELLING PREVALENCE", text_color, scale=2)
    draw_text(canvas, 12, 76, "EXPECTED COUNT", text_color, scale=2)

    for prevalence in prevalences:
        x = x_positions[prevalence]
        prevalence_label = f"{prevalence * 100:g}%"
        draw_line(canvas, x, height - margin_bottom, x, height - margin_bottom + 6, axis_color)
        draw_text(canvas, x - 18, height - margin_bottom + 16, prevalence_label, text_color, scale=2)

    legend_x = width - margin_right + 28
    legend_y = margin_top - 4
    for model_index, (model, rows) in enumerate(grouped_rows.items()):
        _, color_rgb = get_color(model_index)
        points = [
            (x_positions[float(row["prevalence"])], y_at(row[metric_name]))
            for row in rows
        ]
        for point_a, point_b in zip(points, points[1:]):
            draw_line(canvas, point_a[0], point_a[1], point_b[0], point_b[1], color_rgb, thickness=3)
        for x, y in points:
            draw_circle(canvas, x, y, 5, color_rgb)
        current_legend_y = legend_y + model_index * 22
        draw_line(canvas, legend_x, current_legend_y, legend_x + 26, current_legend_y, color_rgb, thickness=3)
        draw_model_label_png(canvas, legend_x + 36, current_legend_y - 8, model, text_color, scale=2)

    return encode_png(canvas)


def infer_figure_folder(output_path, figure_folder):
    if figure_folder:
        return Path(figure_folder)
    return output_path.with_name(f"{output_path.stem}_figures")


def write_per_model_figures(projection_rows, output_path, figure_folder):
    if not projection_rows:
        print("No projection rows available; skipped figure generation")
        return []

    folder = infer_figure_folder(output_path, figure_folder)
    folder.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_cache(folder)

    written_paths = []
    for model, rows in group_rows_by_model(projection_rows).items():
        figure_stem = folder / f"{sanitize_filename(model)}_operational_projection"
        svg_path = figure_stem.with_suffix(".svg")
        png_path = figure_stem.with_suffix(".png")
        svg_path.write_text(render_model_svg(model, rows), encoding="utf-8")
        png_path.write_bytes(render_model_png(model, rows))
        written_paths.extend([svg_path, png_path])
    return written_paths


def write_per_metric_figures(projection_rows, output_path, figure_folder):
    if not projection_rows:
        print("No projection rows available; skipped figure generation")
        return []

    folder = infer_figure_folder(output_path, figure_folder)
    folder.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_cache(folder)
    grouped_rows = group_rows_by_model(projection_rows)

    written_paths = []
    for metric_name, metric_label, file_label in FIGURE_METRICS:
        figure_stem = folder / f"{file_label}_all_models_operational_projection"
        svg_path = figure_stem.with_suffix(".svg")
        png_path = figure_stem.with_suffix(".png")
        svg_path.write_text(
            render_metric_svg(metric_name, metric_label, grouped_rows),
            encoding="utf-8",
        )
        png_path.write_bytes(render_metric_png(metric_name, metric_label, grouped_rows))
        written_paths.extend([svg_path, png_path])
    return written_paths


def write_figures(projection_rows, output_path, figure_folder, figure_mode, figure_models):
    figure_rows = filter_projection_rows_for_figures(projection_rows, figure_models)
    if figure_models and not figure_rows:
        print("Warning: no rows matched --figure_models; skipped figure generation")
        return []

    if figure_mode == "per_metric":
        return write_per_metric_figures(figure_rows, output_path, figure_folder)
    return write_per_model_figures(figure_rows, output_path, figure_folder)


def main():
    args = resolve_args(parse_args())
    metrics_df = list(args.metrics)
    fieldnames = fieldnames_from_rows(metrics_df)
    validate_columns(
        fieldnames,
        [args.model_column, args.recall_column, args.specificity_column],
    )
    projection_rows = project_model_outputs(metrics_df, args)

    output_path = infer_output_file(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "model",
        "entries",
        "prevalence",
        "prevalence_percent",
        "observed_recall",
        "observed_specificity",
        "expected_true_errors",
        "expected_non_errors",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "review_burden_tp_plus_fp",
        "missed_errors_fn",
        "projected_precision",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(projection_rows)

    print(f"Wrote operational projection to {output_path}")
    print(f"Models projected: {len({row['model'] for row in projection_rows})}")
    print(f"Rows written: {len(projection_rows)}")
    should_make_figures = bool(args.make_figures) and not bool(args.no_figures)
    if should_make_figures:
        figure_paths = write_figures(
            projection_rows,
            output_path,
            args.figure_folder,
            args.figure_mode,
            args.figure_models,
        )
        print(f"Wrote figure file(s): {len(figure_paths)}")
        if figure_paths:
            print(f"Figure folder: {figure_paths[0].parent}")
    else:
        print("Figure output disabled by --no_figures")


if __name__ == "__main__":
    main()
