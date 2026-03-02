# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.0",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pypcd4==1.4.3",
#     "pyzmq>=27.1.0",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # ICP Quantization Analysis
    """)
    return


@app.cell
def _():
    import sys
    import seaborn as sns
    import marimo as mo
    from pathlib import Path
    import os
    from pypcd4 import PointCloud
    import numpy as np
    import matplotlib.pyplot as plt
    import logging

    return Path, PointCloud, logging, mo, np, os, plt, sns, sys


@app.cell
def _(sys):
    sys.path.append("/home/reidzemek/icp-python")
    for path in sys.path:
        print(path)
    return


@app.cell
def _():
    import utils
    import rc_themes
    import kdtree
    import icp

    return icp, kdtree, rc_themes, utils


@app.cell
def _(logging, sys):
    logger = logging.getLogger(__name__)

    # Configure the format to show: TIME - LOGGER NAME - LEVEL - MESSAGE
    logging.basicConfig(
        format="%(name)s.%(funcName)s | %(levelname)s | %(message)s",
        level=logging.ERROR,
        stream=sys.stderr,
        force=True
    )
    return (logger,)


@app.cell
def _(rc_themes, sns, utils):
    # Apply theme
    if utils.system_dark_mode():
        rc_theme = rc_themes.monokai_classic_rc
    else:
        rc_theme = rc_themes.monokai_pro_light_rc
    sns.set_theme(context="notebook", rc=rc_theme, palette="muted")
    colors = sns.color_palette() # Access the muted palette colors
    return


@app.cell
def _(Path, PointCloud, icp, kdtree, logger, np, os):
    # LOG file, bit width and the number of iterations
    log = []
    n_bits = range(10, 17)
    n_iters = [2, 5, 10, 20, 50]

    # from docutils.nodes import target
    root = "../Validation_Data"

    for scan_dir in os.listdir(root):
        for frame_dir in os.listdir(os.path.join(root, scan_dir)):

            logger.info(f"Currently processing: {os.path.join(root, scan_dir, frame_dir)}")

            # Get the paths for the source and the best target point clouds
            frame_path = os.path.join(root, scan_dir, frame_dir, "source.pcd")
            # target_path = next(Path(root, scan_dir, frame_dir, "best").glob("*.pcd"))

            for target_path in Path(os.path.join(root, scan_dir, frame_dir)).glob('target_*.pcd'):

                logger.info(f"Current target: {target_path}")

                # Initialize source (P) and target (Q) point clouds
                P_in = PointCloud.from_path(frame_path).numpy(("x", "y", "z"))
                Q = PointCloud.from_path(target_path).numpy(("x", "y", "z"))
                Q_N = PointCloud.from_path(target_path).numpy(
                    ("normal_x", "normal_y", "normal_z")
                )

                # Find global bounds across both clouds to preserve relative spatial frame
                all_pts = np.concatenate([P_in, Q])
                min_val, max_val = all_pts.min(), all_pts.max()

                # Normalize P and Q to [-1, 1] using shared global bounds
                P_in_norm = 2 * (P_in - min_val) / (max_val - min_val) - 1
                Q_norm = 2 * (Q - min_val) / (max_val - min_val) - 1

                for n_bit in n_bits:

                    # Convert P and Q to signed 10-bit integers
                    P_q_in = np.clip(np.rint(P_in_norm * 2**(n_bit-1)*0.9), -2**(n_bit-1), 2**(n_bit-1)-1).astype(np.int16)
                    Q_q = np.clip(np.rint(Q_norm * 2**(n_bit-1)*0.9), -2**(n_bit-1), 2**(n_bit-1)-1).astype(np.int16)

                    # # Convert target point cloud surface normals to 10-bit integers
                    # # TODO: update this to use fixed point representation (update p2pl_error()
                    # # function as well)
                    # Q_N_q = np.clip(np.rint(Q_N * 511), -512, 511).astype(np.int16)

                    # Arrays to store the source point clouds as they are transformed
                    P = P_in
                    P_q = P_q_in

                    # Build the k-d tree to enable efficient nearest neighbor search
                    Q_tree = kdtree.build(Q, Q_N)
                    Q_q_tree = kdtree.build(Q_q)

                    # Total transformation accumulator
                    T_total = np.identity(4)
                    T_q_total = np.identity(4)

                    for n_iter in n_iters:

                        # Iterative closest point algorithm (10 iterations)
                        for i in range(n_iter):
                            logger.info(f"ICP Iteration: {i}")

                            # For each point in P, find the nearest point in Q
                            Q_nearest, Q_N_nearest = kdtree.nn_search(Q_tree, P)
                            Q_q_nearest, _ = kdtree.nn_search(Q_q_tree, P_q)

                            # FULL PRECISION IMPLEMENTATION
                            if True:
                                # Compute centroids (mean) and center the point clouds
                                P_mean = np.mean(P, axis=0).reshape(1, 3)
                                P_centered = P - P_mean
                                Q_nearest_mean = np.mean(Q_nearest, axis=0).reshape(1, 3)
                                Q_nearest_centered = Q_nearest - Q_nearest_mean

                                # Compute the cross covariance matrix
                                H = P_centered.T @ Q_nearest_centered

                                # Compute the 4x4 rigid transformation matrix
                                T = icp.T_matrix(H, P_mean, Q_nearest_mean)

                                # Apply the transformation to P
                                P = (T @ np.hstack([P, np.ones((P.shape[0], 1))]).T).T[:, :3]

                                # Update the total transformation (order should be consistent)
                                T_total = T @ T_total

                            # QUANTIZED IMPLEMENTATION
                            if True:
                                # Compute centroids (mean) and center the point clouds
                                P_q_mean = icp.mean(P_q, "P_q", 16, 28)
                                P_q_centered = icp.center(P_q, P_q_mean, "P_q", 16, 16, 17)
                                Q_q_nearest_mean = icp.mean(Q_q_nearest, "Q_q_nearest", 16, 28)
                                Q_q_nearest_centered = icp.center(Q_q_nearest, Q_q_nearest_mean, "P_q_nearest", 16, 16, 17)

                                # Compute the cross covariance matrix H
                                H_q = icp.xcovariance(P_q_centered, Q_q_nearest_centered, 16, 16, 43)

                                # Compute the 4x4 rigid transformation matrix for P
                                R_q, t_q =icp.transformation(H, P_q_mean, Q_q_nearest_mean, 33, 16, 16, 16, 11)

                                # Apply the transformation to P
                                P_q = icp.transform(P_q, R_q, t_q, 16, 16, 16)

                                logger.debug(f"{P_q.dtype}, {P_q_mean.dtype}, {P_q_centered.dtype}, {H_q.dtype}, {R_q.dtype}, {t_q.dtype}, {P_q.dtype}")

                                # Update the total transformation
                                # TODO write a function to do this
                                # T_q_total = T_q @ T_q_total

                        Q_og = (((Q_q/(2**(n_bit-1)*0.9))+1)/2)*(max_val - min_val) + min_val
                        P_og = (((P_q/(2**(n_bit-1)*0.9))+1)/2)*(max_val - min_val) + min_val

                        Q_og_tree = kdtree.build(Q_og, Q_N)
                        Q_og_nearest, Q_N_nearest1 = kdtree.nn_search(Q_og_tree, P_og)

                        log_line = {
                            "source": os.path.join(scan_dir, frame_dir), "target": target_path.name, "bitwidth": str(n_bit), "n_iter": n_iter, "t_error_p2pl": str(icp.p2pl_error(P_og, Q_og_nearest, Q_N_nearest1)) 
                        }
                        print("log: ", log_line)
                        log.append(log_line)

    return (
        H,
        P,
        P_in,
        P_q,
        P_q_in,
        P_q_mean,
        Q,
        Q_nearest,
        R_q,
        T,
        T_total,
        frame_path,
        log,
    )


@app.cell
def _(log):
    print(log[0].keys())

    import csv

    with open('output.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=log[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(log)
    return


@app.cell
def _():
    # scale = 30000*(2/(max_val - min_val))

    # Q_og = (((Q_q/32000)+1)/2)*(max_val - min_val) + min_val
    # P_og = (((P_q/32000)+1)/2)*(max_val - min_val) + min_val

    # Q_og_tree = kdtree.build(Q_og, Q_N)
    # Q_og_nearest, Q_N_nearest1 = kdtree.nn_search(Q_og_tree, P_og)

    # print("transformation_error_p2pl: ", icp.p2pl_error(P_og, Q_og_nearest, Q_N_nearest1))

    # TODO also compute the point to point transformation errors for the transformation in the validation data for another comparison




    # _Q_q_nearest, _ = kdtree.nn_search(Q_q_tree, P_q)
    # # print("transformation_error_p2pl: ", icp.p2pl_error(P_q, _Q_q_nearest, _Q_N_q_nearest, 10), "\n")
    # print("transformation_matrix:\n", T_q_total, "\n")
    return


@app.cell
def _(P, icp, np):
    print(P)
    print(np.mean(P, axis=0))
    print(icp.mean(P))
    return


@app.cell
def _(P, np, utils):
    print(utils.is_intn(np.sum(P, axis=0), 32))

    tree_depth = P.shape[0].bit_length()

    print((2**tree_depth / P.shape[0]) * (2**16))
    return


@app.cell
def _(P_q_mean):
    print(P_q_mean.shape)
    return


@app.cell
def _(P_q, R_q, icp, mo):



    # print("previous")
    # warnings.warn("test")
    # print("next")

    a = 5

    print(a, "hello")

    print(__name__)

    # logger = logging.getLogger(__name__)

    print(P_q)

    with mo.redirect_stderr():

        icp.mean(P_q, 10, 21)

        print(R_q)
    return


@app.cell
def _(P, P_in, P_q, P_q_in, plt, utils):
    plt.figure(figsize=[x*y for x, y in zip(plt.rcParams["figure.figsize"], [2, 2])])
    _ax1 = plt.subplot(2, 2, 1, projection='3d')
    utils.plot(P_in, "P_in", ax=_ax1)
    _ax2 = plt.subplot(2, 2, 2, projection='3d')
    utils.plot(P_q_in, "P_int11_in", ax=_ax2)
    _ax3 = plt.subplot(2, 2, 3, projection='3d')
    utils.plot(P, "P", ax=_ax3)
    _ax4 = plt.subplot(2, 2, 4, projection='3d')
    utils.plot(P_q, "P_q", ax=_ax4)
    plt.show()
    return


@app.cell
def _():
    # _Q_nearest, _Q_N_nearest = kdtree.nn_search(Q_tree, P)
    # print("transformation_error_p2p: ", icp.p2p_error(P, _Q_nearest))
    # print("transformation_error_p2pl: ", icp.p2pl_error(P, _Q_nearest, _Q_N_nearest), "\n")
    # print("transformation_matrix:\n", T_total, "\n")

    # _Q_int11_nearest, _Q_N_int11_nearest = kdtree.nn_search(Q_int11_tree, P_int11)
    # print("transformation_error_p2p: ", icp.p2p_error(P_int11, _Q_int11_nearest))
    # print("transformation_error_p2pl: ", icp.p2pl_error(P_int11, _Q_int11_nearest, _Q_N_int11_nearest, 10), "\n")
    # print("transformation_matrix:\n", T_int11_total, "\n")
    return


@app.cell
def _(P, T, np):
    (T @ np.hstack([P, np.ones((P.shape[0], 1))]).T).T[:, :3]
    return


@app.cell
def _():
    # P_norm = P_int11/1023
    # P_og = ((P_norm+1)/2)*(max_val - min_val) + min_val

    # print(P_in, "\n")

    # print(P, "\n")
    # print(P_og)
    return


@app.cell
def _(Q_N_int11, np):
    # print(P_mean)
    # # print(P_int10_mean)
    # print(np.mean(P, axis=0).reshape(1, 3))

    # print(Q_int10_nearest)
    # plt.hist(Q_N.flatten())

    # _min, _max = np.min(Q_N), np.max(Q_N)

    # print(_min, _max)

    # Q_N_int10 = np.clip(np.rint(Q_N * 511), -512, 511).astype(np.int16)

    print((Q_N_int11.size - np.unique(Q_N_int11, axis=0).size)/Q_N_int11.size*100, "%")
    return


@app.cell
def _(P_int11, Q_int11_tree, T_int11_total, T_total, kdtree):


    # print(np.min(P_int10), np.max(P_int10))

    # P_norm = P_int10/511
    # P_og = ((P_norm+1)/2)*(max_val - min_val) + min_val

    # _Q_nearest_og, _Q_nearest_N_og = kdtree.nn_search(Q_tree, P_og)

    # print(icp.p2p_error(P_og, _Q_nearest_og))
    # print(icp.p2pl_error(P_og, _Q_nearest_og, _Q_nearest_N_og))

    _Q_int11_nearest, _Q_N_int11_nearest = kdtree.nn_search(Q_int11_tree, P_int11)

    # print(icp.p2p_error(P_int11, _Q_int11_nearest))

    # print(utils.it_intn(P_int11, 11))

    print(T_int11_total)
    print(T_total)

    # print(np.clip(T[:3, :3], -1.0, 1.0 - (2**-15)))
    # print(T[:3, :3])

    # print(np.rint(np.clip(T[:3, :3], -1.0, 1.0-(2**-15)) * 2**15).astype(np.int16))

    # print(np.rint(T[:3, 3].reshape(3, 1)).astype(np.int16))

    # R_Q1_15 = np.rint(np.clip(T_int11_total[:3, :3], -1.0, 1.0-(2**-15)) * 2**15).astype(np.int32)
    # t_int11 = np.rint(T_int11_total[:3, 3].reshape(3, 1)).astype(np.int32)

    # print(R_Q1_15)
    # print(t_int11)

    # print(P_int11_in)

    # # print(((R_Q1_15[0, 0] * P_int11_in[0, 0] + R_Q1_15[0, 1] * P_int11_in[0, 1] + R_Q1_15[0, 2] * P_int11_in[0, 2])))
    # print((((R_Q1_15 @ P_int11_in.T) >> 15) + t_int11).T)

    # print((T_int11_total[:3, :3] @ P_int11_in.T).T)

    # print(T[:3, 3].reshape(3, 1))
    # print((T_total[:3, :3] @ P_in.T + T_total[:3, 3].reshape(3, 1)).T, "\n")
    # print(P)
    # print(T_int11)




    # print(icp.p2pl_error(P_int10, _Q_int10_nearest, _Q_N_int10_nearest, 10))


    # print(T_int10_total)

    # S = 1022/(max_val - min_val)
    # O_scalar = -511 - (S * min_val)
    # O = np.full(3, O_scalar)
    # _T_total = np.eye(4)
    # _T_total[:3, :3] = T_int10_total[:3, :3]
    # _T_total[:3, 3] = (T_int10_total[:3, 3] + (np.eye(3) - T_int10_total[:3, :3]) @ O) / S

    # print(_T_total)
    return


@app.cell
def _(P_in, P_int10_in, plt, utils):
    plt.figure(figsize=[x*y for x, y in zip(plt.rcParams["figure.figsize"], [2, 1])])
    _ax1 = plt.subplot(1, 2, 1, projection='3d')
    utils.plot(P_in, "P", ax=_ax1)
    _ax2 = plt.subplot(1, 2, 2, projection='3d')
    utils.plot(P_int10_in, "P_int10", ax=_ax2)
    plt.show()
    return


@app.cell
def _(T_int10_total, T_total):
    # P_transformed = icp.transform(P, T)
    # P_int10_transformed = icp.transform(P_int10, T_int10)
    print(T_total)
    print(T_int10_total)
    return


@app.cell
def _(P, P_int10, plt, utils):
    plt.figure(figsize=[x*y for x, y in zip(plt.rcParams["figure.figsize"], [2, 1])])
    _ax1 = plt.subplot(1, 2, 1, projection='3d')
    utils.plot(P, "P_transformed", ax=_ax1)
    _ax2 = plt.subplot(1, 2, 2, projection='3d')
    utils.plot(P_int10, "P_int10_transformed", ax=_ax2)
    plt.show()
    return


@app.cell
def _(Q, Q_int10, plt, utils):
    plt.figure(figsize=[x*y for x, y in zip(plt.rcParams["figure.figsize"], [2, 1])])
    _ax1 = plt.subplot(1, 2, 1, projection='3d')
    utils.plot(Q, "Q", ax=_ax1)
    _ax2 = plt.subplot(1, 2, 2, projection='3d')
    utils.plot(Q_int10, "Q_int10", ax=_ax2)
    plt.show()
    return


@app.cell
def _(frame_path):
    frame_path
    return


@app.cell
def _(T, np):
    T.astype(np.int16)
    return


@app.cell
def _(np):
    np.ones(4)
    return


@app.cell
def _():
    # T_scaled = np.multiply(T, np.ones([4, 4])*2**16).astype(np.int64)
    return


@app.cell
def _():
    # P_int10_transformed = icp.transform(P_int10, T_scaled).astype(np.int64) >> 16
    return


@app.cell
def _(P_int10_transformed):
    P_int10_transformed.dtype
    return


@app.cell
def _(H):
    H.dtype
    return


@app.cell
def _(P_int10, utils):
    utils.plot(P_int10)
    return


@app.cell
def _(P_int10_transformed, utils):
    utils.plot(P_int10_transformed)
    return


@app.cell
def _(Q_nearest, utils):
    utils.plot(Q_nearest)
    return


@app.cell
def _(P_int10, Q_nearest, icp):
    icp.p2p_error(P_int10, Q_nearest)
    return


@app.cell
def _(Q, kdtree):
    Q_tree_full = kdtree.build(Q)
    return (Q_tree_full,)


@app.cell
def _(P, Q_tree_full, kdtree):
    Q_nearest_full = kdtree.nn_search(Q_tree_full, P)
    return (Q_nearest_full,)


@app.cell
def _(H, P, Q_nearest_full, icp, np):
    P_full_centroid = np.mean(P, axis=0).reshape(1, 3)
    P_full_centered = icp.center(P, P_full_centroid)
    Q_nearest_full_centroid = np.mean(Q_nearest_full, axis=0).reshape(1, 3)
    Q_nearest_full_centered = icp.center(Q_nearest_full, Q_nearest_full_centroid)
    H_full = icp.xcovariance(P_full_centered, Q_nearest_full_centered)
    T_full = icp.transformation(H, P_full_centroid, Q_nearest_full_centroid)
    P_full_transformed = icp.transform(P, T_full)
    return (P_full_transformed,)


@app.cell
def _(P, utils):
    utils.plot(P)
    return


@app.cell
def _(P_full_transformed, utils):
    utils.plot(P_full_transformed)
    return


@app.cell
def _(Q, utils):
    utils.plot(Q)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
