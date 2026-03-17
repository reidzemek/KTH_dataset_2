import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Validation Dataset Tools

    This app has a couple of different tools to to evaluate the validation dataset.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    import marimo as mo
    import os
    from rich.progress import track
    import yaml
    from glom import glom
    import matplotlib.pyplot as plt

    return Path, glom, mo, os, plt, sys, track, yaml


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
    import utils

    return


@app.cell
def _(mo, os):
    validation_path_ui = mo.ui.text(label="Validation Dataset Path", value="../Validation_Data")
    run_button = mo.ui.run_button(label="🏃‍♂️‍➡️ Run")

    def validate_path(path):
        exists = os.path.exists(path or "")
        if not exists:
            return mo.md("❌ **Path not found**"), exists
        return mo.md("✅"), exists

    return run_button, validate_path, validation_path_ui


@app.cell
def _(mo, run_button, validate_path, validation_path_ui):
    dataset_validity_icon, dataset_validity_status = validate_path(validation_path_ui.value)
    validation_row = mo.hstack([validation_path_ui, dataset_validity_icon], justify="start")

    mo.vstack([
        mo.md("## Options"),
        validation_row,
        run_button
    ])
    return (dataset_validity_status,)


@app.cell
def _(dataset_validity_status, mo, run_button, validation_path_ui):
    is_script = mo.app_meta().mode == "script"
    mo.stop(not is_script and not run_button.value, mo.md("Please configure options and click 🏃‍♂️‍➡️ **Run** to continue.").callout(kind="warn"))
    mo.stop(not dataset_validity_status, mo.md("⚠️ **Aborted:** Invalid dataset path. Please fix ❌ above and try again.").callout(kind="danger"))

    VALIDATION_DATASET_PATH = validation_path_ui.value

    mo.md("## Anlaysis 🔍")
    return VALIDATION_DATASET_PATH, is_script


@app.cell
def _(Path, VALIDATION_DATASET_PATH, glom, mo, track, yaml):
    original_transformation_error = []
    computed_transformation_error_p2p = []
    computed_transformation_error_p2pl = []

    computed_p2p_accuracy = []
    computed_p2pl_accuracy = []

    # List the scan directories in the provided dataset path
    scan_dirs = list(Path(VALIDATION_DATASET_PATH).glob("source*"))

    # Initialize the scan progress bar UI
    with mo.status.progress_bar(total=len(scan_dirs), title="Currently processing scan") as scan_progress:

        for scan_dir in track(scan_dirs, description="Processing scans: "):

            # Update the scan progress bar UI
            scan_progress.update(increment=0, subtitle=f"{scan_dir.name}")

            # List the frame directories for the current scan
            frame_dirs = scan_dir.iterdir()

            for frame_dir in frame_dirs:

                # Read the metadata file for the current frame
                metadata_path = frame_dir / "metadata.yaml"
                with open(metadata_path, 'r') as file:
                    metadata = yaml.safe_load(file)

                for target in metadata["targets"]:

                    original_transformation_error.append(glom(target, "metrics.original.transformation_error"))
                    computed_transformation_error_p2p.append(glom(target, "metrics.computed.transformation_error_p2p"))
                    computed_transformation_error_p2pl.append(glom(target, "metrics.computed.transformation_error_p2pl"))


                T_errors_og = glom(metadata, ("targets", ["metrics.original.transformation_error"]))
                T_errors_p2p = glom(metadata, ("targets", ["metrics.computed.transformation_error_p2p"]))
                T_errors_p2pl = glom(metadata, ("targets", ["metrics.computed.transformation_error_p2pl"]))
                computed_p2p_accuracy.append(T_errors_og.index(min(T_errors_og)) == T_errors_p2p.index(min(T_errors_p2p)))
                computed_p2pl_accuracy.append(T_errors_og.index(min(T_errors_og)) == T_errors_p2pl.index(min(T_errors_p2pl)))

            scan_progress.update()
    return (
        computed_p2p_accuracy,
        computed_p2pl_accuracy,
        computed_transformation_error_p2p,
        computed_transformation_error_p2pl,
        original_transformation_error,
    )


@app.cell
def _(VALIDATION_DATASET_PATH, mo):
    VALIDATION_DATASET_PATH
    mo.md(
        """
        ### Transformation Error Comparison ⚖️

        First, we will plot the distribution of the transformation errors for each frame of each scan 
        in the dataset. Specifically, the final figure will show 3 distributions:

        - original transformation error - *point to plane*
        - computed transformation error - *point to point*
        - computed transformation error - *point to plane*

        **NOTE:** we are not performing any registration (ICP) of the point clouds. We are only 
        comparing the point to point transformation error for the source target pairs provided in the 
        validation dataset to computed transformation error metrics used in our ICP reference 
        implementation.

        The reason for the discrepancy between the original point to plane transformation error and the 
        computed one is that the original calculation ignored small per point errors in the point cloud 
        amplitude data while the computed transformation error does not.
        """
    )
    return


@app.cell
def _(
    Path,
    computed_transformation_error_p2p,
    computed_transformation_error_p2pl,
    is_script,
    mo,
    original_transformation_error,
    plt,
):
    plt.hist(original_transformation_error, bins=300, alpha=0.5, label="original: point to plane")
    plt.hist(computed_transformation_error_p2p, bins=50, alpha=0.5, label="computed: point to point")
    plt.hist(computed_transformation_error_p2pl, bins=50, alpha=0.5, label="computed: point to plane")
    plt.xlim([0, 100])
    plt.legend()
    if is_script:
        T_error_distribution_path = Path("distribution.png")
        plt.savefig(T_error_distribution_path)
        print(f"Figure saved to: {T_error_distribution_path.resolve()}")
    mo.center(plt.gca())
    return


@app.cell
def _(
    VALIDATION_DATASET_PATH,
    computed_p2p_accuracy,
    computed_p2pl_accuracy,
    mo,
):
    VALIDATION_DATASET_PATH

    p2p_percentage = round(sum(computed_p2p_accuracy)/len(computed_p2p_accuracy)*100)
    p2pl_percentage = round(sum(computed_p2pl_accuracy)/len(computed_p2pl_accuracy)*100)

    mo.md(
        f"""
        ### Best Target Selection Accuracy Evaluation 🎯

        Now will compare how the computed transformation errors compare to the original point to plane 
        transformation error for the selection of the best target.

        The best target corresponds to the minimum transformation error, and should result in the 
        highest spatial accuracy of the retrieval based localization system.

        If we consider the original transformation error from the validation dataset as the reference, 
        we have following best target selection accuracies:

        **Computed point to point transformation error accuracy:** {p2p_percentage}%<br>
        **Computed point to plane transformation error accuracy:** {p2pl_percentage}%
        """
    )
    return


@app.cell
def _(VALIDATION_DATASET_PATH, mo):
    VALIDATION_DATASET_PATH

    mo.md(
        """
        ## Conclusions

        Since there appears to be no correlation between the best target selection based on the 
        original transformation error, and the best target selection based on the computed 
        computed transformation error metrics, we can conclude that the computed metrics would be 
        insufficient to evaluate quantization parameters for our ICP implementation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
