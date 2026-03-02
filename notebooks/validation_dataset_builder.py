import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Validation Dataset Builder
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    from pypcd4 import PointCloud, Encoding
    import marimo as mo
    import os
    import json
    import shutil
    import numpy as np
    import yaml

    return Encoding, Path, PointCloud, json, mo, np, os, sys, yaml


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
    import utilities
    import utils
    import kdtree
    import icp

    return icp, kdtree, utilities


@app.cell
def _(mo, os):
    dataset_path_ui = mo.ui.text(label="Dataset Path", value="../KTH_dataset_2")
    validation_path_ui = mo.ui.text(label="Validation Dataset Path", value="../Validation_Data")
    point_count_ui = mo.ui.number(label="Source point count", value=3824)
    build_button = mo.ui.run_button(label="🛠️ Build")

    def validate_path(path):
        exists = os.path.exists(path or "")
        if not exists:
            return mo.md("❌ **Path not found**"), exists
        return mo.md("✅"), exists

    return (
        build_button,
        dataset_path_ui,
        point_count_ui,
        validate_path,
        validation_path_ui,
    )


@app.cell
def _(
    build_button,
    dataset_path_ui,
    mo,
    point_count_ui,
    validate_path,
    validation_path_ui,
):
    dataset_validity_icon, dataset_validity_status = validate_path(dataset_path_ui.value)
    dataset_row = mo.hstack([dataset_path_ui, dataset_validity_icon], justify="start")

    mo.vstack([
        mo.md("## Options"),
        dataset_row,
        point_count_ui,
        validation_path_ui,
        build_button
    ])
    return (dataset_validity_status,)


@app.cell
def _(
    build_button,
    dataset_path_ui,
    dataset_validity_status,
    mo,
    point_count_ui,
    validation_path_ui,
):
    is_browser = mo.app_meta().mode != "script"
    mo.stop(is_browser and not build_button.value, mo.md("Please configure options and click 🛠️ **Build** to continue.").callout(kind="warn"))
    mo.stop(not dataset_validity_status, mo.md("⚠️ **Build aborted:** Invalid dataset path. Please fix ❌ above and try again.").callout(kind="danger"))
    DATASET_PATH = dataset_path_ui.value
    SOURCE_POINT_COUNT = point_count_ui.value
    VALIDATION_DATASET_PATH = validation_path_ui.value

    mo.md("## Building 🛠️")
    return DATASET_PATH, SOURCE_POINT_COUNT, VALIDATION_DATASET_PATH


@app.cell
def _():
    import time
    from collections import defaultdict
    from ruamel.yaml import YAML
    from ruamel.yaml import CommentedMap, CommentedSeq
    from ruamel.yaml.tokens import CommentToken
    from ruamel.yaml.scalarstring import PlainScalarString

    return CommentedMap, CommentedSeq, YAML, defaultdict


@app.cell
def _(yaml):
    # 1. DEFINE ALIGNMENT LOGIC
    # This ensures all floats are written with 15 decimal places and aligned 
    def aligned_float_representer(dumper, data):
        # 22.15f: 22 total width, 15 decimals. 
        # High width ensures even large translation values don't break the alignment.
        return dumper.represent_scalar('tag:yaml.org,02:float', f"{data:22.15f}")

    yaml.add_representer(float, aligned_float_representer)
    return


@app.cell
def _(
    CommentedMap,
    CommentedSeq,
    DATASET_PATH,
    Encoding,
    Path,
    PointCloud,
    SOURCE_POINT_COUNT,
    VALIDATION_DATASET_PATH,
    YAML,
    defaultdict,
    icp,
    json,
    kdtree,
    mo,
    np,
    os,
    utilities,
):
    # List the scan directories in the provided dataset path
    from docutils.nodes import meta
    scan_dirs = [d for d in Path(DATASET_PATH).glob("source_*") if d.is_dir()]

    # Initialize the scan progress bar UI
    with mo.status.progress_bar(total=len(scan_dirs), title="Currently processing scan") as scan_progress:

        for scan_dir in scan_dirs:

            # Update the scan progress bar UI
            scan_progress.update(increment=0, subtitle=f"{scan_dir.name}")

            # Group file paths in the scan directory by their leading numeric prefix
            frame_map = defaultdict(list)
            for f in Path(scan_dir).glob("*_*"):
                frame_map[f.name.split('_')[0]].append(f)

            # Initialize the frame progress bar UI
            with mo.status.progress_bar(total=len(frame_map), title="Currently processing frame", remove_on_exit=True) as frame_progress:

                for prefix in sorted(frame_map.keys()):

                    # Update the frame progress bar UI
                    frame_progress.update(increment=0, subtitle=f"{prefix}")

                    # Extract point cloud and metadata file paths
                    files = frame_map[prefix]
                    source_path = next(f for f in files if f.name.endswith("_filtered_full_pointcloud.pcd"))
                    metadata_path = next(f for f in files if f.name.endswith("_metadata.json"))

                    # Read, downsample and add source point cloud to the validation dataset
                    P = PointCloud.from_path(source_path).numpy(('x', 'y', 'z'))
                    P_d = utilities.farthest_point_sampling(P, SOURCE_POINT_COUNT)
                    validation_path = os.path.join(VALIDATION_DATASET_PATH, scan_dir.name, prefix)
                    Path(validation_path).mkdir(parents=True, exist_ok=True)
                    PointCloud.from_xyz_points(P_d).save(os.path.join(validation_path, "source.pcd"), encoding=Encoding.ASCII)

                    # Prepare validation dataset metadata file
                    metadata_yaml = YAML()
                    metadata_yaml.width = 512
                    metadata_yaml.indent(mapping=2, sequence=4, offset=2)
                    metadata_validation = CommentedMap()

                    # Add the current source path to the validation metadata
                    metadata_validation["source"] = str(source_path)

                    # List to hold validation metadata for filtered targets 
                    filtered_targets_validation = []

                    # Load the metadata file for the current source point cloud
                    with open(metadata_path, 'r') as file:
                        metadata = json.load(file)

                    filtered_targets = metadata.get("filtered_targets")

                    # Initialize the target progress bar UI
                    with mo.status.progress_bar(total=len(filtered_targets), title="Currently processing target", remove_on_exit=True) as target_progress:

                        for target in filtered_targets:

                            # Update the target progress bar UI
                            target_progress.update(increment=0, subtitle=f"{target.get("bolt")} - {target.get("id")} (Bolt - ID)")

                            # Read the transformation matrix and the corresponding point to plate transformation error from the metadata file
                            T_matrix = np.array(target.get("transformation_matrix"), dtype=float).T
                            T_error = target.get("transformation_error")

                            # Target path
                            target_path = os.path.join(DATASET_PATH, "targets", str(target.get("bolt")), str(target.get("id")) + ".pcd")

                            # Load the target point cloud
                            Q = PointCloud.from_path(target_path).numpy(('x', 'y', 'z'))
                            N = PointCloud.from_path(target_path).numpy(('normal_x', 'normal_y', 'normal_z'))

                            # Compute the point to point and point to plane transformation errors
                            Q_tree = kdtree.build(Q, N)
                            Q_nearest, N_nearest = kdtree.nn_search(Q_tree, P)
                            T_error_p2p = icp.p2p_error(P, Q_nearest)
                            T_error_p2pl = icp.p2pl_error(P, Q_nearest, N_nearest)

                            target_validation = CommentedMap()
                            target_validation["path"] = str(target_path)
                            matrix_rows = CommentedSeq()
                            for row in T_matrix:
                                # Convert row to a CommentedSeq to access formatting attributes
                                row_seq = CommentedSeq([float(val) for val in row])

                                # Set the flow style for this specific row (square brackets)
                                row_seq.fa.set_flow_style() 

                                matrix_rows.append(row_seq)

                            target_validation["transformation_matrix"] = matrix_rows

                            target_validation["metrics"] = {
                                "original": {
                                    "transformation_error": T_error
                                },
                                "computed": {
                                    "transformation_error_p2p": float(T_error_p2p),
                                    "transformation_error_p2pl": float(T_error_p2pl)
                                }
                            }
                            filtered_targets_validation.append(target_validation)

                            target_progress.update()

                    metadata_validation["targets"] = filtered_targets_validation
                    metadata_validation["criterion"] = {
                        "metric": "metrics.original.transformation_error",
                        "objective": "minimize"
                    }
                    with open(os.path.join(validation_path, "metadata.yaml"), "w") as f:
                        metadata_yaml.dump(metadata_validation, f)

                    frame_progress.update()
            scan_progress.update()





            # for path in groups["01"]:
            #     print(path)

                            # time.sleep(0.1)
    return


@app.cell
def _(Path, VALIDATION_DATASET_PATH, mo):
    _validation_path = Path(VALIDATION_DATASET_PATH).resolve()

    mo.md(
        f"""
        ## 💯 All done, happy architecting! 👨‍💻

        You can find the validation dataset at: `{_validation_path}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
