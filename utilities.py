import numpy as np
import marimo as mo
import os
from pathlib import Path
from pypcd4 import PointCloud
import yaml
from scipy.spatial import KDTree as scipy_KDTree
from KDTree import KDTree
from natsort import natsorted
import matplotlib.pyplot as plt
import csv

# Function to add vertical lines with number annotations
def add_vline(position, color=None, linestyle="--", label_format="{:.2f}", label=None):
    
    # Get the current axis
    ax = plt.gca()

    # Get the y position for the text label
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    height = ymin + 0.05 * y_range

    c = color if color else ax._get_lines.get_next_color()

    ax.axvline(position, linestyle=linestyle, color=c, label=label)

    ax.text(
        position,
        height,
        label_format.format(position),
        ha="center",
        va="bottom",
        bbox=dict(
            facecolor=plt.rcParams["figure.facecolor"],
            edgecolor="none",
            boxstyle="round",
            alpha=0.9
        )
    )

def farthest_point_sampling(points, k):
    N = points.shape[0]
    centroids = np.zeros(k, dtype=np.int64)
    distances = np.full(N, np.inf)

    # Deterministic first point: farthest from the center
    center = points.mean(axis=0)
    farthest = np.argmax(np.linalg.norm(points - center, axis=1))

    for i in range(k):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return points[centroids]

def validate_path(path):
    exists = os.path.exists(path or "")
    if not exists:
        return mo.md("❌ **Path not found**").style(white_space="nowrap"), exists
    return mo.md("✅"), exists

# TODO this function should be added to the an ICP class along with all other ICP functionality
def load_pc(path: Path) -> np.ndarray:
    """
    Load a point cloud from a CSV file into an nx3 NumPy array.

    Args:
        path (Path or str): path to the CSV file

    Returns:
        np.ndarray: nx3 array of points (x, y, z)
    """
    path = Path(path)
    pc = np.loadtxt(path, delimiter=",")
    
    if pc.ndim == 1:
        # Single row case
        pc = pc.reshape(1, 3)
        
    if pc.shape[1] != 3:
        raise ValueError("CSV file must have 3 columns for x, y, z")
    
    return pc

# TODO: should also be added to an ICP class
def write_pc_csv(pc: np.ndarray, path: Path):
    """
    Write a point cloud to a CSV file.
    
    Args:
        pc (np.ndarray): nx3 array of points (x, y, z)
        path (Path or str): path to the output CSV file
    """
    if pc.shape[1] != 3:
        raise ValueError("Point cloud array must have shape (n, 3)")
    
    path.parent.mkdir(parents=True, exist_ok=True)

    
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for point in pc:
            writer.writerow(point)

# TODO this should be part of the ICP class as well
def write_pc_bin(pc: np.ndarray, path: Path, n_bits: int):
    """
    Write a point cloud to a binary-style CSV with fixed bit width per coordinate.

    Args:
        pc (np.ndarray): nx3 array of points (signed integers)
        path (Path or str): path to the output file
        n_bits (int): bit width for each coordinate
    """
    if pc.shape[1] != 3:
        raise ValueError("Point cloud array must have shape (n, 3)")
    
    path.parent.mkdir(parents=True, exist_ok=True)


    n_points = pc.shape[0]
    
    with path.open("w") as f:
        for i, point in enumerate(pc):
            # Convert each coordinate to n_bits binary
            line = "".join(np.binary_repr(val, width=n_bits) for val in point)
            # Line terminator
            line += ',' if i < n_points - 1 else ';'
            f.write(line + "\n")

def process(validation_path, trace_path, n_P, n_Q, n_pairs, n_coord_bits, addr_width, min_val, max_val):

    # List the scan directories in the provided dataset path
    scan_dirs = list(Path(validation_path).glob("source*"))

    # Counter for the number of processed point cloud pairs
    pc_pair_count = 0

    for scan_dir in natsorted(scan_dirs):

        # List the frame directories for the current scan
        frame_dirs = scan_dir.iterdir()

        for frame_dir in natsorted(frame_dirs):

            # Source point cloud (current frame)
            source_path = next(frame_dir.glob("*.pcd"))

            print(source_path)

            # Construct trace source path
            trace_source_subpath = Path(*source_path.parts[
                next(i for i, p in enumerate(source_path.parts) if p.startswith("source")):
            ]).with_suffix("")
            trace_source_path = Path(trace_path, f"{trace_source_subpath}-{n_P}_{n_coord_bits}.csv")

            print(trace_source_path)

            # Load source point cloud (P) if it already exists
            if trace_source_path.exists():

                # Initialize the source point cloud
                P = load_pc(trace_source_path)

            # Load source point cloud (P) for downsampling and quantization
            else:

                # Load point cloud from validation dataset
                P = PointCloud.from_path(source_path).numpy(("x", "y", "z"))

                # Downsample
                P = farthest_point_sampling(P, n_P)

                # Quantize
                P_norm = 2 * (P - min_val) / (max_val - min_val) - 1
                P = np.clip(np.rint(P_norm * 2**(n_coord_bits-1)*0.9), -2**(n_coord_bits-1), 2**(n_coord_bits-1)-1).astype(np.int64)

                write_pc_csv(P, trace_source_path)
                write_pc_bin(P, trace_source_path.with_suffix(".bin"), n_coord_bits)

            # Read the metadata file for the current frame
            metadata_path = frame_dir / "metadata.yaml"
            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)

            for target in metadata["targets"]:
                if n_pairs is not None and pc_pair_count >= n_pairs:
                    return Q_tree._log_leaf, Q_tree._log_best, Q_tree._log_branch, log, log_result
                pc_pair_count += 1

                # Target point cloud
                target_path = Path(validation_path) / target["path"]
                Q = PointCloud.from_path(target_path).numpy(("x", "y", "z"))

                print(target_path)

                # Construct tree path
                tree_subpath = Path(*target_path.parts[target_path.parts.index("targets"):]).with_suffix("")
                tree_path = Path(trace_path, f"{tree_subpath}-{n_Q}_{n_coord_bits}.csv")

                # Load tree data structure for target point cloud (Q) if it already exists
                if False:#tree_path.exists():

                    # Initialize the k-d tree data structure for the target point cloud (Q)
                    Q_tree = KDTree(tree_path)

                # Load target point cloud (Q) for downsampling, quantization and tree construction
                else:

                    # Downsample 
                    Q = farthest_point_sampling(Q, n_Q)

                    # Quantize
                    Q_norm = 2 * (Q - min_val) / (max_val - min_val) - 1
                    Q = np.clip(np.rint(Q_norm * 2**(n_coord_bits-1)*0.9), -2**(n_coord_bits-1), 2**(n_coord_bits-1)-1).astype(np.int64)

                    # Initialize k-d tree data structure for the target point cloud (Q)
                    Q_tree = KDTree(Q)

                    Q_tree.write_tree(tree_path)
                    Q_tree.write_tree_bin(tree_path.with_suffix(".bin"), n_coord_bits, addr_width)
                    
                Q_scipy_tree = scipy_KDTree(Q)
                _, indices = Q_scipy_tree.query(P)
                Q_scipy_nearest = Q[indices]

                # Find the nearest neighbor for each point in the source point cloud (P)
                Q_nearest, _ = Q_tree.nn_search(P)

                print(f"CORRECT: {np.array_equal(Q_nearest, Q_scipy_nearest)}")

                print(f"\nmax tree depth: {Q_tree.max_depth}")
                print(f"total number of visited nodes: {Q_tree._visited_count}")

                # Trace path for the current target
                target_trace_path = Path(trace_path) / scan_dir.name / frame_dir.name / Path(f"{target_path.parts[-3]}_{target_path.parts[-2]}_{target_path.stem}")

                print(target_trace_path)

                nearest_path = Path(target_trace_path, f"nearest-{n_P}_{n_Q}_{n_coord_bits}.csv")
                write_pc_csv(Q_nearest.astype(np.int64), nearest_path)
                write_pc_bin(Q_nearest.astype(np.int64), nearest_path.with_suffix(".bin"), n_coord_bits)

                Q_tree.write_search_trace(target_trace_path, n_P, n_Q, n_coord_bits)
                log, log_result = Q_tree.write_unified_search_trace(target_trace_path, n_P, n_Q, n_coord_bits)
