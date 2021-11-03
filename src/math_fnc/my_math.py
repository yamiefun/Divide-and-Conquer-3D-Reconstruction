import numpy as np
from utils import utils


def L2_dist(pnt1, pnt2):
    return np.sqrt(sum((pnt1-pnt2)**2))


def calculate_point_err(mtx, matched_set, blk1, blk2, thresh):
    err = 0
    dist_thresh = thresh
    outlier = 0
    for pnt_idx in matched_set:
        pnt1 = np.array(blk1[pnt_idx[0]].coor)
        pnt2 = np.array(blk2[pnt_idx[1]].coor)
        # print(pnt1.shape)
        pnt1 = np.concatenate((pnt1, np.array([1])))
        warp_pnt = mtx.dot(pnt1)
        warp_pnt = (warp_pnt/warp_pnt[-1])[:3]
        dist = L2_dist(warp_pnt, pnt2)
        if dist > dist_thresh:
            outlier += 1
    #     err += dist
    # err /= len(matched_set)
    # print(f"Error: {err}")
    inlier = len(matched_set) - outlier
    out_ratio = outlier / len(matched_set)
    print(f"Inlier: {inlier}, Outlier: {outlier}")
    print(f"Outlier Ratio: {out_ratio*100}%")

    return outlier


def calculate_cam_err(mtx, pnt_set1, pnt_set2, thresh):
    assert len(pnt_set1) == len(pnt_set2),\
        f"Different anchor images."
    outlier = 0
    inlier = 0
    warpped = np.dot(mtx, pnt_set1)
    warpped = warpped.T
    pnt_set2 = pnt_set2.T
    for pnt1, pnt2 in zip(warpped, pnt_set2):
        dist = L2_dist(pnt1[:3], pnt2[:3])
        if dist > thresh:
            outlier += 1
        else:
            inlier += 1
    return outlier


def calculate_err_two_matrix(R, t, matched_set, blk1, blk2):
    err = 0
    for pnt_idx in matched_set:
        pnt1 = np.array(blk1[pnt_idx[0]].coor).T
        pnt2 = np.array(blk2[pnt_idx[1]].coor).T

        warp_pnt = (R@pnt1) + t
        dist = pnt2 - warp_pnt
        dist = dist*dist
        dist = np.sum(dist)
        dist = np.sqrt(dist)
        # print(f"dist{dist}")
        err += dist
    err /= len(matched_set)
    print(f"Error: {err}")

    return err
