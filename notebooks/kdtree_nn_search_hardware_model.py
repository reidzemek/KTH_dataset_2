# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.22.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", css_file="")


@app.cell
def _(mo):
    mo.md(r"""
    # 🌳 k-d tree nearest neighbor search

    This app can be used to generate data to generate trace data for functional validation of the k-d tree nearest neighbor search hardware implementation.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    import marimo as mo
    import os
    from rich.progress import track
    from pypcd4 import PointCloud
    import numpy as np
    import matplotlib.pyplot as plt
    import debugpy
    from glom import glom
    import yaml
    import pandas as pd
    from scipy.spatial import KDTree as scipy_KDTree
    import statistics
    import seaborn as sns
    import math

    return (
        Path,
        PointCloud,
        debugpy,
        math,
        mo,
        np,
        os,
        pd,
        plt,
        sns,
        statistics,
        sys,
    )


@app.cell
def _(mo, os):
    # Set script flag if running in script mode
    is_script = mo.app_meta().mode == "script"

    # Check if any environment variable starts with "VSCODE_"
    is_vscode = any(key.startswith("VSCODE_") for key in os.environ)
    return is_script, is_vscode


@app.cell
def _(debugpy, is_script, is_vscode, os):
    # Suppress frozen module warnings
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    # Start debugpy listener if client is not already running
    if not is_script and is_vscode:
        try:
            debugpy.listen(5678)
            print("Debugger listening on port 5678")
        except RuntimeError:
            print("Debugger already running")
    return


@app.cell
def _(Path, sys):
    # Source code
    src_paths = [
        Path().resolve().parent,
        Path().resolve().parent.parent / "icp-python"
    ]

    # Add to path of not already there
    for _path in src_paths:
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    return


@app.cell
def _():
    from KDTree import KDTree
    import utilities
    import utils
    import rc_themes

    return rc_themes, utilities, utils


@app.cell
def _(mo):
    validation_path_label = mo.md("Validation dataset path")
    validation_path_field = mo.ui.text(value="../Validation_Data", full_width=True)

    submit_button = mo.ui.run_button(label="📤 Submit")
    return submit_button, validation_path_field, validation_path_label


@app.cell
def _(
    mo,
    submit_button,
    utilities,
    validation_path_field,
    validation_path_label,
):
    _icon, validation_path_status = utilities.validate_path(validation_path_field.value)
    validation_path_row = mo.hstack(
        [validation_path_label.style(white_space="nowrap"), validation_path_field, _icon],
        widths=[0, 1, 0]
    )

    mo.vstack([
        validation_path_row,
        submit_button
    ])
    return (validation_path_status,)


@app.cell
def _(
    is_script,
    is_vscode,
    mo,
    submit_button,
    validation_path_field,
    validation_path_status,
):
    mo.stop(not is_script and not submit_button.value and not is_vscode, mo.md("Please configure the validation dataset path and click 📤 **Submit** to continue.").callout(kind="warn"))
    mo.stop(not validation_path_status, mo.md("⚠️ **Aborted:** Invalid dataset path. Please fix ❌ above and try again.").callout(kind="danger"))

    validation_path = validation_path_field.value
    return (validation_path,)


@app.cell
def _(rc_themes, sns, utils, validation_path):
    validation_path

    # Apply theme
    if utils.system_dark_mode():
        rc_theme = rc_themes.marimo_dark_rc
    else:
        rc_theme = rc_themes.marimo_light_rc
    sns.set_theme(context="notebook", rc=rc_theme, palette="muted")
    return (rc_theme,)


@app.cell
def _(rc_theme):
    figure_style = {
        "border-radius": "20px", 
        "overflow": "hidden",    # Essential to clip the figure's sharp corners
        "border": f"1px solid {rc_theme["grid.color"]}",
        "box-shadow": f"0 4px 12px {rc_theme["figure.facecolor"]}"
    }
    return (figure_style,)


@app.cell
def _(Path, validation_path):
    # List source and target point cloud paths
    sources = list(Path(validation_path).rglob("*filtered_full*.pcd"))
    targets = list(Path(validation_path, "targets").rglob("*.pcd"))
    return sources, targets


@app.cell
def _(
    PointCloud,
    figure_style,
    is_vscode,
    mo,
    plt,
    sources,
    statistics,
    targets,
    utilities,
):
    # Gather point counts
    n_p = [PointCloud.from_path(s).points for s in sources]
    n_q = [PointCloud.from_path(t).points for t in targets]

    # Compute min, median, max
    stats_n_p = {"minimum": min(n_p), "median": statistics.median(n_p), "maximum": max(n_p)}
    stats_n_q = {"minimum": min(n_q), "median": statistics.median(n_q), "maximum": max(n_q)}

    plt.figure(figsize=[plt.rcParams["figure.figsize"][0] * 2, plt.rcParams["figure.figsize"][1]])

    # Left subplot: sources
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 2nd subplot
    plt.hist(n_p, bins=20)
    for _label, _position in stats_n_p.items():
        utilities.add_vline(_position, label=_label, label_format="{:.0f}")
    plt.legend()
    plt.title('Source Point Counts')
    plt.xlabel('Number of points')
    plt.ylabel('Frequency')

    # Right subplot: targets
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 1st subplot
    plt.hist(n_q, bins=20)
    for _label, _position in stats_n_q.items():
        utilities.add_vline(_position, label=_label, label_format="{:.0f}")
    plt.legend()
    plt.title('Target Point Counts')
    plt.xlabel('Number of points')
    plt.ylabel('Frequency')

    plt.tight_layout()
    _figure = mo.style(plt.gcf(), figure_style) if not is_vscode else (plt.show() or "")

    mo.md(f"""
        ## Point count distributions

        First, we will consider the distributions of the total number of points in both the source 
        and target point clouds.

        Based on these distributions, we can evaluate several design decisions, including:

        - The static point count for the source point cloud, which influences the amount of data to be
        processed in subsequent DRRA-2 resources.
        - Maximum point count for the target point clouds, which will define:
            - The required on-chip memory for tree storage
            - The address bit-width needed to access tree memory

        {_figure}
    """)
    return stats_n_p, stats_n_q


@app.cell
def _(
    PointCloud,
    figure_style,
    is_vscode,
    mo,
    np,
    plt,
    sources,
    targets,
    utilities,
):
    # Gather coordinate values
    p_coords = np.concatenate([PointCloud.from_path(t).numpy(("x", "y", "z")) for t in sources]).ravel().tolist()
    q_coords = np.concatenate([PointCloud.from_path(t).numpy(("x", "y", "z")) for t in targets]).ravel().tolist()

    # Compute min, median, max
    stats_p_coords = {"minimum": min(p_coords), "maximum": max(p_coords)}
    stats_q_coords = {"minimum": min(q_coords), "maximum": max(q_coords)}

    plt.figure(figsize=[plt.rcParams["figure.figsize"][0] * 2, plt.rcParams["figure.figsize"][1]])

    plt.subplot(1, 2, 1)
    _counts, _bins = np.histogram(p_coords, bins=20)
    _bins = 0.5 * (_bins[:-1] + _bins[1:])
    plt.bar(_bins, _counts, width=_bins[1] - _bins[0])
    plt.gca()._get_lines.get_next_color()
    for _label, _position in stats_p_coords.items():
        utilities.add_vline(_position, label=_label, label_format="{:.0f}")
    plt.legend()
    plt.title("Source coordinate values (x, y & z)")
    plt.xlabel("Coordinate values")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    _counts, _bins = np.histogram(q_coords, bins=20)
    _bins = 0.5 * (_bins[:-1] + _bins[1:])
    plt.bar(_bins, _counts, width=_bins[1] - _bins[0])
    plt.gca()._get_lines.get_next_color()
    for _label, _position in stats_q_coords.items():
        utilities.add_vline(_position, label=_label, label_format="{:.0f}")
    plt.legend()
    plt.title("Target coordinate values (x, y & z)")
    plt.xlabel("Coordinate values")
    plt.ylabel("Frequency")

    plt.tight_layout()
    _figure = mo.style(plt.gcf(), figure_style) if not is_vscode else (plt.show() or "")

    mo.md(f"""
        ## Coordinate value distributions

        {_figure}
    """)
    return (stats_q_coords,)


@app.cell
def _(mo):
    memory_params_label = mo.md("### Memory Parameters")
    n_coord_bits_label = mo.md("Quantization bit width for point cloud coordinates (P & Q)")
    n_coord_bits_field = mo.ui.dropdown(options=list(range(8, 33)), value=16)
    n_addrs_label = mo.md("Addressable memory for k-d tree nodes")
    n_addrs_field = mo.ui.dropdown(options=list(2**i for i in range(9, 16)), value=2**14)
    return (
        memory_params_label,
        n_addrs_field,
        n_addrs_label,
        n_coord_bits_field,
        n_coord_bits_label,
    )


@app.cell
def _(mo, n_addrs_field):
    n_P_value, n_P_set_value = mo.state(3824)
    n_Q_value, n_Q_set_value = mo.state(n_addrs_field.value)
    return n_P_set_value, n_P_value, n_Q_set_value, n_Q_value


@app.cell
def _(
    mo,
    n_P_set_value,
    n_P_value,
    n_Q_set_value,
    n_Q_value,
    stats_n_p,
    stats_n_q,
):
    n_pc_label = mo.md("### Number of point cloud points")
    n_P_label = mo.md("Source (P)")
    n_P_slider = mo.ui.slider(
        start=1,
        stop=stats_n_p["maximum"],
        value=n_P_value(),
        on_change=n_P_set_value,
        full_width=True
    )
    n_Q_label = mo.md("Target (Q)")
    n_Q_slider = mo.ui.slider(
        start=10,
        stop=stats_n_q["maximum"],
        value=min(stats_n_q["maximum"], n_Q_value()),
        on_change=n_Q_set_value, full_width=True
    )
    general_heading = mo.md("### General")
    n_pairs_label = mo.md("Total number of source (P) and target (Q) point cloud pairs to process")
    n_pairs_field = mo.ui.dropdown(options=[1, 10, 200], value=1)
    trace_path_label = mo.md("Output path for k-d tree search trace")
    trace_path_field = mo.ui.text(value="../KDTree_Search_Trace", full_width=True)

    run_button = mo.ui.run_button(label="🏃‍♂️‍➡️ Run")
    return (
        general_heading,
        n_P_label,
        n_P_slider,
        n_Q_label,
        n_Q_slider,
        n_pairs_field,
        n_pairs_label,
        n_pc_label,
        run_button,
        trace_path_field,
        trace_path_label,
    )


@app.cell
def _(
    mo,
    n_P_set_value,
    n_P_value,
    n_Q_set_value,
    n_Q_value,
    stats_n_p,
    stats_n_q,
):
    n_P_field = mo.ui.number(
        start=1,
        stop=stats_n_p["maximum"],
        value=n_P_value(),
        on_change=n_P_set_value
    )
    n_Q_field = mo.ui.number(
        start=10,
        stop=stats_n_q["maximum"],
        value=min(stats_n_q["maximum"], n_Q_value()),
        on_change=n_Q_set_value
    )
    return n_P_field, n_Q_field


@app.cell
def _(math, n_addrs_field):
    addr_width = int(math.log2(n_addrs_field.value))
    return (addr_width,)


@app.cell
def _(
    addr_width,
    mo,
    n_addrs_field,
    n_addrs_label,
    n_coord_bits_field,
    n_coord_bits_label,
):
    n_coord_bits_row = mo.hstack(
        [n_coord_bits_label, n_coord_bits_field],
        justify="start"
    )
    n_addrs_row = mo.hstack(
        [
            n_addrs_label,
            n_addrs_field,
            mo.md(f"<em>(Address width: <strong>{addr_width if addr_width is not None else '--'} bits</strong>)</em>"),
        ],
        justify="start",
    )
    return n_addrs_row, n_coord_bits_row


@app.cell
def _(mo, n_P_field, n_P_label, n_P_slider, n_Q_field, n_Q_label, n_Q_slider):
    n_P_row = mo.hstack(
        [
            n_P_label.style(white_space="nowrap"),
            n_P_slider,
            n_P_field.style(width="12ch"),
            mo.md("*(required number of points - smaller: removed, larger: downsampled)*").style(white_space="nowrap")
        ],
        widths=[0, 1, 0, 0],
        justify="start"
    )
    n_Q_row = mo.hstack(
        [
            n_Q_label.style(white_space="nowrap"),
            n_Q_slider,
            n_Q_field.style(width="12ch"),
            mo.md("*(maximum number of points - larger: downsampled)*").style(white_space="nowrap")
        ],
        widths=[0, 1, 0, 0],
        justify="start"
    )
    return n_P_row, n_Q_row


@app.cell
def _(mo, n_pairs_field, n_pairs_label, trace_path_field, trace_path_label):
    n_pairs_row = mo.hstack(
        [n_pairs_label.style(white_space="nowrap"), n_pairs_field],
        justify="start"
    )
    trace_path_row = mo.hstack(
        [trace_path_label.style(white_space="nowrap"), trace_path_field],
        widths=[0, 1]
    )
    return n_pairs_row, trace_path_row


@app.cell
def _(
    general_heading,
    memory_params_label,
    mo,
    n_P_row,
    n_Q_row,
    n_addrs_row,
    n_coord_bits_row,
    n_pairs_row,
    n_pc_label,
    run_button,
    trace_path_row,
):
    mo.vstack([
        mo.md("## Options"),
        memory_params_label,
        n_coord_bits_row,
        n_addrs_row,
        n_pc_label,
        n_P_row,
        n_Q_row,
        general_heading,
        n_pairs_row,
        trace_path_row,
        run_button
    ])
    return


@app.cell
def _(
    addr_width,
    is_script,
    mo,
    n_P_field,
    n_Q_field,
    n_coord_bits_field,
    n_pairs_field,
    run_button,
    stats_q_coords,
    trace_path_field,
    utilities,
    validation_path_field,
):
    mo.stop(not is_script and not run_button.value, mo.md("Please configure options and click 🏃‍♂️‍➡️ **Run** to continue.").callout(kind="warn"))

    log_leaf, log_best, log_branch, log, log_result = utilities.process(
        validation_path_field.value,
        trace_path_field.value,
        n_P_field.value,
        n_Q_field.value,
        n_pairs_field.value,
        n_coord_bits_field.value,
        addr_width,
        stats_q_coords["minimum"],
        stats_q_coords["maximum"]
    )
    return log, log_best, log_branch, log_leaf, log_result


@app.cell
def _(log, log_best, log_branch, log_leaf, log_result, mo, pd):
    # VALIDATION_DATASET_PATH = validation_path_field.value

    search_trace_heading = mo.md("## 🔍 Anlaysis: 🧭 k-d tree search trace")

    leaf_heading = mo.md("### Leaf nodes on consecutive downward passes")
    leaf_df = pd.DataFrame(log_leaf)
    leaf_df.columns = [f"pass_{i}" for i in range(leaf_df.shape[1])]

    best_heading = mo.md("### Best nodes on consecutive downward passes")
    best_df = pd.DataFrame(log_best)
    best_df.columns = [f"pass_{i}" for i in range(best_df.shape[1])]

    rows = []

    for point_idx, point_log in enumerate(log_branch):
        for down_pass_idx, nodes in enumerate(point_log):
            rows.append({
                "point": point_idx,
                "down_pass": down_pass_idx,
                "nodes": nodes
            })

    branch_df = pd.DataFrame(rows)

    branch_pivot = branch_df.pivot(index="point", columns="down_pass", values="nodes")
    branch_pivot.columns = branch_pivot.columns.astype(str)

    branch_pivot.columns = [f"pass {i}" for i in branch_pivot.columns]

    branch_heading = mo.md("### Branch nodes")

    unified_log_heading = mo.md("### Unified log")
    columns = [
        "query_addr",
        "query_x", "query_y", "query_z",
        "trav_id",
        "node_addr",
        "node_x", "node_y", "node_z",
        "node_type",
        "point_dist",
        "plane_dist"
    ]
    unified_log = pd.DataFrame(log, columns=columns)

    mo.vstack([
        search_trace_heading,
        leaf_heading,
        leaf_df,
        best_heading,
        best_df,
        branch_heading,
        branch_pivot,
        unified_log_heading,
        log,
        mo.md("### Nearest Neighbours"),
        log_result
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
