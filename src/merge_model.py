import argparse
import os
from collections import namedtuple
import numpy as np
import random
# from transform import find_transform_matrix
from transform import superimposition_matrix
from shutil import copyfile
# from rigid_transform_3D import rigid_transform_3D
from utils import utils
from math_fnc import my_math as mm


def ransac_find_transform(matched_3d_set, blk1, blk2, thresh):
    matched_set = np.array(matched_3d_set)
    print(f"Total number of matched point: {len(matched_set)}")
    n_sample = 10000
    outlier_ratio = 0.5
    # n_iter = int(np.log(1-0.99)/np.log(1-(1-outlier_ratio)**n_sample))
    n_iter = 10
    print(f"Total round for RANSAC: {n_iter}")
    err = float('inf')
    mtx = None
    R = None
    t = None
    for it in range(n_iter):
        print(f'RANSAC round {it}')
        # random sample matched 3d points
        blk1_sample = []
        blk2_sample = []
        # rand_idx = random.sample(range(len(matched_set)), n_sample)
        # sample_pnt_id = matched_set[rand_idx]
        # for pnt_idx in sample_pnt_id:
        #     blk1_sample.append(blk1[pnt_idx[0]].coor)
        #     blk2_sample.append(blk2[pnt_idx[1]].coor)
        sampled = set()
        while len(blk1_sample) < n_sample:
            idx = random.randrange(len(matched_set))
            if idx not in sampled:
                pnt_idx = matched_set[idx]
                sampled.add(idx)
                # select only err < 1 3d points as anchor
                if blk1[pnt_idx[0]].err > 1 or blk2[pnt_idx[1]].err > 1:
                    continue
                blk1_sample.append(blk1[pnt_idx[0]].coor)
                blk2_sample.append(blk2[pnt_idx[1]].coor)

        blk1_mod = utils.modify_point_format(blk1_sample)
        blk2_mod = utils.modify_point_format(blk2_sample)

        # ret_R, ret_t = rigid_transform_3D(blk1_mod, blk2_mod)
        # tmp_err = calculate_err_two_matrix(
        #   ret_R, ret_t, matched_set, blk1, blk2)

        # tmp_mtx = find_transform_matrix(blk1_sample, blk2_sample)
        tmp_mtx = superimposition_matrix(blk1_mod, blk2_mod, scale=True)
        # tmp_err = calculate_err(tmp_mtx, matched_set, blk1, blk2)
        tmp_err = mm.calculate_err(tmp_mtx, matched_set, blk1, blk2)
        # print(tmp_err)
        if tmp_err < err:
            err = tmp_err
            mtx = tmp_mtx
            # R = ret_R
            # t = ret_t
    print(f"Best mapping outlier ratio: {err/len(matched_set)}")
    # return R, t
    return mtx


def main():
    args = utils.parse_args()
    blk1_image_txt_path = os.path.join(args.blk1, f"images.txt")
    blk2_image_txt_path = os.path.join(args.blk2, f"images.txt")
    blk1_point3d_path = os.path.join(args.blk1, f"points3D.txt")
    blk2_point3d_path = os.path.join(args.blk2, f"points3D.txt")
    blk1_ply_path = os.path.join(args.blk1, f"model.ply")
    blk2_ply_path = os.path.join(args.blk2, f"model.ply")

    print(f"Start parsing COLMAP log files.")
    # get images.txt info
    blk1_anchor_2d_info = utils.parse_images_txt(blk1_image_txt_path)
    blk2_anchor_2d_info = utils.parse_images_txt(blk2_image_txt_path)

    # get points3D.txt info
    blk1_3d_info = utils.parse_points3d_txt(blk1_point3d_path)
    blk2_3d_info = utils.parse_points3d_txt(blk2_point3d_path)

    # assert len(blk1_anchor_2d_info) == len(blk2_anchor_2d_info), \
    #     ('Anchor image number are not the same in two image set',
    #      f'blk1 has {len(blk1_anchor_2d_info)}, '
    #      f'blk2 has {len(blk2_anchor_2d_info)}')

    if args.match == "":
        print(f"Start finding anchor image feature points.")
        matched_3d_set = []  # save the match of 3d point id in two set
        match_pnt_path = utils.get_default_block_path(-1)
        match_pnt_path = os.path.join(match_pnt_path, 'matchPoints.txt')
        with open(match_pnt_path, "w") as f:
            for image_name in blk1_anchor_2d_info:
                if image_name in blk2_anchor_2d_info:
                    image1_info = blk1_anchor_2d_info[image_name]
                    image2_info = blk2_anchor_2d_info[image_name]

                    # find useful 2d point appear in both image set
                    for pnt2d in image1_info.Pnt2D:
                        for target in image2_info.Pnt2D:
                            if pnt2d.x == target.x and pnt2d.y == target.y:
                                matched_3d_set.append((pnt2d.Pnt3DID,
                                                       target.Pnt3DID))
                                f.write(f"{pnt2d.x} {pnt2d.y} {pnt2d.Pnt3DID} "
                                        f"{target.Pnt3DID}\n")
                                break
    else:
        print(f"Load matched anchor image feature points from file.")
        matched_3d_set = []
        match_pnt_path = args.match
        with open(match_pnt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                x, y, id1, id2 = line.split()
                matched_3d_set.append((int(id1), int(id2)))
    print(f"Found total {len(matched_3d_set)} matched anchor points.")

    print(f"Start finding transform matrix.")
    thresh = 0.8
    transform_mtx = ransac_find_transform(
        matched_3d_set, blk1_3d_info, blk2_3d_info, thresh)
    print(f"Transformation matrix:\n{transform_mtx}")

    print(f"Start writing ply file for the merged model.")
    utils.modify_ply(blk1_ply_path, blk2_ply_path, transform_mtx,
                     f"merge_model_by_pnt")


if __name__ == "__main__":
    main()
