import marimo

__generated_with = "0.20.2"
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
    from collections import defaultdict
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from rich.progress import track

    return (
        CommentedMap,
        CommentedSeq,
        Encoding,
        Path,
        PointCloud,
        YAML,
        defaultdict,
        json,
        mo,
        np,
        os,
        sys,
        track,
    )


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
def _(YAML):
    yaml_width = 100
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = yaml_width
    return yaml, yaml_width


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
    defaultdict,
    icp,
    json,
    kdtree,
    mo,
    np,
    os,
    track,
    utilities,
    yaml,
    yaml_width,
):
    # List the scan directories in the provided dataset path
    from docutils.nodes import meta
    scan_dirs = [d for d in Path(DATASET_PATH).glob("source_*") if d.is_dir()]

    # Initialize the scan progress bar UI
    with mo.status.progress_bar(total=len(scan_dirs), title="Currently processing scan") as scan_progress:

        for scan_dir in track(scan_dirs, description="Processing scans: "):

            # Update the scan progress bar UI
            scan_progress.update(increment=0, subtitle=f"{scan_dir.name}")

            # Group file paths in the scan directory by their leading numeric prefix
            frame_map = defaultdict(list)
            for f in Path(scan_dir).glob("*_*"):
                frame_map[f.name.split('_')[0]].append(f)

            # Initialize the frame progress bar UI
            with mo.status.progress_bar(total=len(frame_map), title="Currently processing frame", remove_on_exit=True) as frame_progress:

                for prefix in track(sorted(frame_map.keys()), description="Professing frames: "):

                    # Update the frame progress bar UI
                    frame_progress.update(increment=0, subtitle=f"{prefix}")

                    # Extract point cloud, metadata and amplitude file paths
                    files = frame_map[prefix]
                    source_path = next(f for f in files if f.name.endswith("_filtered_full_pointcloud.pcd"))
                    metadata_path = next(f for f in files if f.name.endswith("_metadata.json"))
                    amplitude_path = next(f for f in files if f.name.endswith("_amplitude.json"))

                    # Initialize the path for the validation dataset
                    validation_path = os.path.join(VALIDATION_DATASET_PATH, scan_dir.name, prefix)
                    Path(validation_path).mkdir(parents=True, exist_ok=True)

                    # Read, downsample and add source point cloud to the validation dataset
                    P = PointCloud.from_path(source_path).numpy(('x', 'y', 'z'))
                    P_ds = utilities.farthest_point_sampling(P, SOURCE_POINT_COUNT)
                    PointCloud.from_xyz_points(P_ds).save(os.path.join(validation_path, source_path.name), encoding=Encoding.ASCII)

                    # Read the point cloud amplitude data
                    with open(amplitude_path, 'r') as file:
                        amplitude = np.array(json.load(file)["data"])

                    # Load the metadata file for the current source point cloud
                    with open(metadata_path, 'r') as file:
                        metadata = json.load(file)

                    filtered_targets = metadata.get("filtered_targets")

                    # Initialize structured validation metadata for YAML
                    metadata_validation = {}

                    # Add the current source path to the validation metadata
                    metadata_validation["source"] = os.path.join(validation_path, source_path.name)

                    # List to hold validation metadata for filtered targets 
                    filtered_targets_validation = []

                    # Initialize the target progress bar UI
                    with mo.status.progress_bar(total=len(filtered_targets), title="Currently processing target", remove_on_exit=True) as target_progress:

                        for target in track(filtered_targets, description="Professing targets: "):

                            # Update the target progress bar UI
                            target_progress.update(increment=0, subtitle=f"{target.get("bolt")} - {target.get("id")} (Bolt - ID)")

                            # Read the transformation matrix and the corresponding point to plate transformation error from the metadata file
                            T_matrix = np.array(target.get("transformation_matrix"), dtype=float).T
                            T_error = np.array(target.get("transformation_error"))

                            # Target path
                            target_path = Path(os.path.join(DATASET_PATH, "targets", str(target.get("bolt")), str(target.get("id")) + ".pcd"))

                            print(f"\nSource: {source_path}\nTarget: {target_path}")

                            # Load the target point cloud
                            Q = PointCloud.from_path(target_path).numpy(('x', 'y', 'z'))
                            N = PointCloud.from_path(target_path).numpy(('normal_x', 'normal_y', 'normal_z'))

                            # Compute the point to point and point to plane transformation errors
                            Q_tree = kdtree.build(Q, N)
                            Q_nearest, N_nearest = kdtree.nn_search(Q_tree, P)
                            T_error_p2p = icp.p2p_error(P, Q_nearest)
                            T_error_p2pl = icp.p2pl_error(P, Q_nearest, N_nearest)

                            # Structured validation metadata for current target
                            target_validation = CommentedMap({
                                "path": os.path.join(validation_path, target_path.name),
                                "transformation_matrix": [
                                    (lambda r: r.fa.set_flow_style() or r)(
                                        CommentedSeq(row)
                                    ) for row in T_matrix.tolist()
                                ],
                                "metrics": {
                                    "original": {
                                        "transformation_error": float(T_error)
                                    },
                                    "computed": {
                                        "transformation_error_p2p": float(T_error_p2p),
                                        "transformation_error_p2pl": float(T_error_p2pl)
                                    }
                                }       
                            })

                            T_matrix_visual = "Visual Representation:\n"
                            for row in T_matrix:
                                T_matrix_visual += f"[ {', '.join(f'{val:8.3f}' for val in row)} ]\n"

                            target_validation.yaml_set_comment_before_after_key('metrics', before=T_matrix_visual, indent=2)

                            filtered_targets_validation.append(target_validation)

                            target_progress.update()

                    filtered_targets_validation_formatted = CommentedSeq(filtered_targets_validation)
                    for i in range(0, len(filtered_targets_validation_formatted)):
                        label = f" TARGET {i+1} ".center(yaml_width - 4, "#")
                        filtered_targets_validation_formatted.yaml_set_comment_before_after_key(i, before=f"\n{label}", indent=2)
                    
                    metadata_validation["targets"] = filtered_targets_validation_formatted
                    metadata_validation["criterion"] = {
                        "metric": "metrics.original.transformation_error",
                        "objective": "minimize"
                    }

                    metadata_validation_formatted = CommentedMap(metadata_validation)
                    for key in list(metadata_validation_formatted.keys())[1:]:
                        metadata_validation_formatted.yaml_set_comment_before_after_key(key, before="\n")

                    metadata_validation_formatted.yaml_set_comment_before_after_key("targets", before="Target point clouds identified as potential matches")
                    metadata_validation_formatted.yaml_set_comment_before_after_key("criterion", before="Best target selection criterion")
                 
                    with open(os.path.join(validation_path, "metadata.yaml"), "w") as f:
                        yaml.dump(metadata_validation_formatted, f)

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
