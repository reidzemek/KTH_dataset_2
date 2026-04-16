import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pypcd4 import PointCloud
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    return Path, PointCloud, np, plt, sns, sys


@app.cell
def _(sys):
    sys.path.append("/Users/reidzemek/Developer/icp-python")
    sys.path
    return


@app.cell
def _():
    # CELL test
    import utils
    import rc_themes

    return rc_themes, utils


@app.cell
def _(rc_themes, sns, utils):
    # CELL Apply theme
    if utils.system_dark_mode():
        rc_theme = rc_themes.monokai_classic_rc
    else:
        rc_theme = rc_themes.monokai_pro_light_rc
    sns.set_theme(context="notebook", rc=rc_theme, palette="muted")
    colors = sns.color_palette() # Access the muted palette colors
    return colors, rc_theme


@app.cell
def _(PointCloud, utils):
    _pc_target = PointCloud.from_path("KTH_dataset_2/targets/1/9686.pcd").numpy(("x", "y", "z"))
    utils.plot(_pc_target, "Target Point Cloud")
    return


@app.cell
def _(PointCloud, utils):
    _pc_source_filtered_full = PointCloud.from_path(
        "KTH_dataset_2/source_2025-12-12_12.30.27/01_filtered_full_pointcloud.pcd"
    ).numpy(("x", "y", "z"))
    utils.plot(_pc_source_filtered_full, "Source Point Cloud - Filtered Full")

    _pc_source_keypoints = PointCloud.from_path(
        "KTH_dataset_2/source_2025-12-12_12.30.27/01_keypoints_pointcloud.pcd"
    ).numpy(("x", "y", "z"))
    utils.plot(_pc_source_keypoints, "Source Point Cloud - Keypoints")

    _pc_source_preselect = PointCloud.from_path(
        "KTH_dataset_2/source_2025-12-12_12.30.27/01_preselect_pointcloud.pcd"
    ).numpy(("x", "y", "z"))
    utils.plot(_pc_source_preselect, "Source Point Cloud - Preselect")
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can now look at the total number of points in both the target, and the full version of each source point cloud by plotting histograms over the entire dataset for each.
    """)
    return


@app.cell
def _(Path, PointCloud, plt):
    num_pts_target = []

    for dir in Path("KTH_dataset_2/targets").iterdir():
        for _path in Path(dir).iterdir():
            _pc = PointCloud.from_path(_path).numpy(("x", "y", "z"))
            num_pts_target.append(_pc.shape[0])

    plt.hist(num_pts_target, bins=30);
    plt.title("Target Point Clouds - Point Count")
    plt.show()
    return


@app.cell
def _(Path, PointCloud, colors, np, plt, rc_theme):
    num_pts_source = []

    for _d in Path("KTH_dataset_2").glob("source_2025-12-12_*"):
        for _path in _d.glob("*_filtered_full_pointcloud.pcd"):
            _pc = PointCloud.from_path(_path).numpy(("x", "y", "z"))
            num_pts_source.append(_pc.shape[0])

    _p = 15
    _p_val = np.percentile(num_pts_source, _p)

    plt.hist(num_pts_source, bins=30)

    plt.axvline(
        _p_val,
        linestyle='--',
        linewidth=1.2,
        label=f'Percentile: {_p}',
        color=colors[1]
    )

    # Define reusable bbox style
    bbox_style = {
        "facecolor": rc_theme["legend.facecolor"],
        "edgecolor": rc_theme["legend.edgecolor"],
        "alpha": 0.85,
        "boxstyle": "round,pad=0.2",
    }

    plt.annotate(
        f"{_p_val:.0f}",
        xy=(_p_val, plt.ylim()[0]),
        xytext=(10, 10),
        textcoords="offset points",
        color=colors[1],
        bbox=bbox_style,
        fontweight='bold'
    )

    plt.title("Full Source Point Clouds - Point Count")
    plt.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Algorithm Implementations (AlImps)

    #### Centroid

    Since the centroid AlImp operates only on the source point cloud (`P`) and its corresponding nearest points in the target point cloud (`Q_nearest`) - both of which have the same number of points (# points in `P`) - we only need to restrict the number of points in `P`;  the full target point cloud (`Q`) can remain unchanged.

    The memory layout will be 8 points (3 10-bit values per point) per memory row (240/256 bit = 94% utilization). After loading one row into the AlImp, 8 coordinate values (one from each point) are passed passed into 3 8-value adder trees followed by shift based division by 8 arranged in parallel, one for each coordinate axis (x, y, z). The output is written to a register file which enables hierarchical averaging of the point cloud. The averaging resource can be configured for either 8, 4 or 2 values, allowing the centroid AlImp to accept any even number of total input points.

    The number of rows required in the AlImp register file will depend on the total number of input points. For 3824 input points identified above, we will require a 4 row register file. This is outlined in the [following](./centroid.drawio) flowchart which shows how the average resource would need to be programmed (in Proto-Assembly Language (PASM)) to process 3824 total points.

    We can think of each row of the register file as one hierarchical averaging level. In the first input row, each point is just itself. In the second row, each point is initially the average of 8 points; in the third row, the average of 64 (8²) points; and in the fourth row, the average of 512 (8³) points.

    "Initially" means that, unless the input is a perfect power of 8 (in our case it isn't), there will be blocks of 4, 2, and/or 1 point(s) that must be collapsed using the alternative operating modes of the averaging resource. After the fourth row has been fully collapsed, only one point - the centroid - remains, which can be sent directly to the output.
    """)
    return


if __name__ == "__main__":
    app.run()
