import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Validation Dataset Builder OLD
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

    return Encoding, Path, PointCloud, json, mo, np, os, shutil, sys, yaml


@app.cell
def _(Path, sys):
    # Source code
    src_paths = [
        Path().resolve().parent,
        Path().resolve().parent.parent / "icp-python"
    ]

    # Add to path if not already there
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

    return (utilities,)


@app.cell
def _(mo):
    options = (
        mo.md(
            """
            ## Options
            {dataset_path}<br>
            {source_point_count}<br>
            {validation_dataset_path}
            """
        ).batch(
            dataset_path=mo.ui.text(label="Dataset Path: ", value="../KTH_dataset_2"), # type: ignore
            source_point_count=mo.ui.number(label="Source point count:", value=3824), # type: ignore
            validation_dataset_path=mo.ui.text(label="Validation Dataset Path: ", value="../Validation_Data_OLD") # type: ignore
        ).form()
    )

    options
    return (options,)


@app.cell
def _(mo, options):
    mo.stop(options.value is None, mo.md("Please configure options and click **Submit** to continue."))

    DATASET_PATH: str = options.value["dataset_path"] # type:ignore
    SOURCE_POINT_COUNT: int = options.value["source_point_count"] # type: ignore
    VALIDATION_DATASET_PATH: str = options.value["validation_dataset_path"] # type: ignore
    return DATASET_PATH, SOURCE_POINT_COUNT, VALIDATION_DATASET_PATH


@app.cell
def _(
    DATASET_PATH: str,
    Encoding,
    Path,
    PointCloud,
    SOURCE_POINT_COUNT: int,
    VALIDATION_DATASET_PATH: str,
    json,
    mo,
    np,
    os,
    shutil,
    utilities,
    yaml,
):
    # Initialize the progress bar UI
    with  mo.status.progress_bar(total=len(list(Path(DATASET_PATH).glob("source_2025-12-12_*"))), title="Building Validation Dataset (scans)") as progress:

        # For each scan
        for scan in Path(DATASET_PATH).glob("source_2025-12-12_*"):

            # Update the progress bar UI
            progress.update(increment=1, subtitle=f"{scan}")

            # For each full resolution source point cloud
            for _path in scan.glob("*_filtered_full_pointcloud.pcd"):

                # Index for the current full resolution source point cloud
                source_idx = _path.name.removesuffix("_filtered_full_pointcloud.pcd")

                # Metadata file for the current source point cloud
                metadata_path = os.path.join(_path.parent, source_idx + "_metadata.json")

                # Directory name for validation pair and downsampled source point cloud path
                pair_path = os.path.join(VALIDATION_DATASET_PATH, _path.parent.name, source_idx)
                _path_fx = os.path.join(pair_path, "source.pcd")

                # Current full resolution source point cloud
                pc = PointCloud.from_path(_path).numpy(("x", "y", "z"))

                # Skip if too few points of save if point count is correct
                Path(pair_path).mkdir(parents=True, exist_ok=True)
                if pc.shape[0] < SOURCE_POINT_COUNT:
                    continue
                if pc.shape[0] == SOURCE_POINT_COUNT:
                    PointCloud.from_xyz_points(pc).save(_path_fx, encoding=Encoding.ASCII)

                # Downsample and save point clouds with too many points
                pc_fx = utilities.farthest_point_sampling(pc, SOURCE_POINT_COUNT)
                PointCloud.from_xyz_points(pc_fx).save(_path_fx, encoding=Encoding.ASCII)

                # Load the metadata file for the current source point cloud
                with open(metadata_path, 'r') as file:
                    metadata = json.load(file)

                # For each target identified as a potential match
                for target in metadata.get("filtered_targets"):

                    # Extract the bolt number and the ID for the current target
                    target_bolt = str(target.get("bolt"))
                    target_id = str(target.get("id"))

                    # Path for the current target
                    target_path = os.path.join(_path.parent.parent, "targets", target_bolt, target_id + ".pcd")

                    # Validation dataset path for current target
                    best = ""
                    target_base_path = os.path.join(pair_path, best)
                    Path(target_base_path).mkdir(parents=True, exist_ok=True)
                    target_path_dest = os.path.join(target_base_path, "target_bolt-" + target_bolt + "_id-" + target_id + ".pcd")

                    # Make a copy of the target point cloud in the validation dataset
                    shutil.copy(target_path, target_path_dest)

                    # Extract the transformation matrix and the point to plane transformation error of the current target from the metadata file
                    transformation_matrix = np.array(target.get("transformation_matrix")).T
                    transformation_error_p2pl = np.array(target.get("transformation_error"))

                    # Structured transformation parameters for YAML
                    data_to_save = {
                        "transformation_matrix": transformation_matrix.tolist(),
                        "transformation_error_p2pl": transformation_error_p2pl.tolist()
                    }

                    # 2. Create the "Visual Alignment" comment
                    # We use 8.4f for a compact, readable look
                    visual_matrix = "# Visual Representation:\n"
                    for row in transformation_matrix:
                        visual_matrix += f"# [ {', '.join(f'{val:8.3f}' for val in row)} ]\n"

                    # Path for transformation parameters YAML
                    transformation_path = os.path.join(target_base_path, "target_bolt-" + target_bolt + "_id-" + target_id + "_transformation.yaml")

                    # Write transformation parameters YAML file
                    with open(transformation_path, 'w') as f:
                        yaml.dump(data_to_save, f, default_flow_style=None, sort_keys=False)

                        # Append the visual comment at the end
                        f.write("\n" + visual_matrix)

                    # Place the best transformation 
                    if (target_bolt == str(metadata.get("best_target").get("bolt"))) & (target_id == str(metadata.get("best_target").get("id"))):
                        best = "best"

                        target_base_path = os.path.join(pair_path, best)

                        Path(target_base_path).mkdir(parents=True, exist_ok=True)

                        target_path_dest = os.path.join(target_base_path, "target_bolt-" + target_bolt + "_id-" + target_id + ".pcd")
                        shutil.copy(target_path, target_path_dest)
                        transformation_path = os.path.join(target_base_path, "target_bolt-" + target_bolt + "_id-" + target_id + "_transformation.yaml")
                        # Write to file
                        with open(transformation_path, 'w') as f:
                            yaml.dump(data_to_save, f, default_flow_style=None, sort_keys=False)

                            # Append the visual comment at the end
                            f.write("\n" + visual_matrix)
    return


@app.cell
def _(Path, VALIDATION_DATASET_PATH: str, mo):
    _validation_path = Path(VALIDATION_DATASET_PATH).resolve()

    mo.md(
        f"""
        ## All done, happy architecting! 👨‍💻

        You can find the validation dataset at: `{_validation_path}`
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
