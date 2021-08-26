import argparse
import os
from collections import namedtuple
import numpy as np
import random

# from scipy.spatial import transform
# from transform import find_transform_matrix
from transform import superimposition_matrix
# from rigid_transform_3D import rigid_transform_3D
from utils import utils
from math_fnc import my_math as mm
from shutil import copyfile


def ransac_find_camera_transform(used_list, blk1, blk2, thresh):
    cam1_list = []
    cam2_list = []
    # cam1_list_all = []
    # cam2_list_all = []
    # Extract camera position
    for image_name in used_list:
        cam1 = blk1[image_name]
        cam2 = blk2[image_name]
        cam1_list.append(cam1)
        cam2_list.append(cam2)

    # for image_name in matched_list:
    #     cam1 = blk1[image_name]
    #     cam2 = blk2[image_name]
    #     cam1_list_all.append(cam1)
    #     cam2_list_all.append(cam2)

    assert len(cam1_list) == len(cam2_list),\
        f"The number of anchor images should be the same."

    cam1_list = np.array(cam1_list)
    cam2_list = np.array(cam2_list)
    # cam1_list_all = np.array(cam1_list_all)
    # cam2_list_all = np.array(cam2_list_all)

    # RANSAC
    n_sample = int(max(3, len(used_list)/10))
    assert len(used_list) >= 3,\
        f"The number of anchors is less than 3, cannot calculate transform matrix."
    outlier_ratio = 0.1
    # n_iter = int(np.log(1-0.99)/np.log(1-(1-outlier_ratio)**n_sample))
    n_iter = 10000
    print(f"--------------------------------------")
    print(f"RANSAC information:")
    print(f"# of sample points: {n_sample}")
    print(f"# of rounds: {n_iter}")
    print(f"--------------------------------------")
    err = 1e6
    mtx = None
    random_range = len(cam1_list)
    cam1_list = utils.modify_point_format(cam1_list)
    cam2_list = utils.modify_point_format(cam2_list)
    for it in range(n_iter):
        print(f'RANSAC round {it}')
        # random sample matched 3d points
        blk1_sample = []
        blk2_sample = []
        rand_idx = random.sample(range(random_range), n_sample)
        # blk1_sample = cam1_list[rand_idx]
        # blk2_sample = cam2_list[rand_idx]
        blk1_sample = cam1_list[:, rand_idx]
        blk2_sample = cam2_list[:, rand_idx]
        # blk1_cam = utils.modify_point_format(blk1_sample)
        # blk2_cam = utils.modify_point_format(blk2_sample)

        tmp_mtx = superimposition_matrix(blk1_sample, blk2_sample, scale=True)
        tmp_err = mm.calculate_cam_err3(tmp_mtx, cam1_list, cam2_list, thresh)
        if tmp_err < err:
            mtx = tmp_mtx
            err = tmp_err
    print(f"--------------------------------------")
    print(f"Result of RANSAC:")
    print(f"# of anchor points: {len(used_list)}")
    print(f"# of inliers : {len(used_list)-err}")
    print(f"# of outliers: {err}")
    print(f"Best mapping outlier ratio: {err/len(used_list):.3f}")
    print(f"--------------------------------------")

    return mtx


def get_common_anchor(blk1, blk2):
    return blk1.keys() & blk2.keys()


def modify_ply(blk1_ply_path, blk2_ply_path, mtx):
    out_path = utils.get_default_block_path(-1)
    out_path = os.path.join(out_path, 'merged_model.ply')
    copyfile(blk2_ply_path, out_path)
    coor = [None]*4
    rgba = [None]*4
    fout = open(out_path, 'a')
    with open(blk1_ply_path, "r") as f:
        lines = f.readlines()
        header_end = False
        for line in lines:
            if "end_header" in line:
                header_end = True
                continue
            if header_end:
                coor = [None]*4
                coor[0], coor[1], coor[2], *tmp = line.split()
                coor[3] = 1
                coor = [float(val) for val in coor]
                coor = np.array(coor)
                warp_coor = np.dot(mtx, coor)
                warp_coor = warp_coor[:3]
                coor = [str(val) for val in warp_coor]
                result = coor + tmp
                result = " ".join(result)
                fout.write(f"{result}\n")
    fout.close()

def main():
    args = utils.parse_args()
    blk1_image_txt_path = os.path.join(args.blk1, f"images.txt")
    blk2_image_txt_path = os.path.join(args.blk2, f"images.txt")
    # blk1_point3d_path = os.path.join(args.blk1, f"points3D.txt")
    # blk2_point3d_path = os.path.join(args.blk2, f"points3D.txt")
    blk1_ply_path = os.path.join(args.blk1, f"model.ply")
    blk2_ply_path = os.path.join(args.blk2, f"model.ply")

    print(f"Start parsing COLMAP log files.")
    # get images.txt info
    blk1_anchor_2d_info = utils.parse_images_txt(blk1_image_txt_path)
    blk2_anchor_2d_info = utils.parse_images_txt(blk2_image_txt_path)

    # calculate 3d coor of each image
    blk1_image_coor = utils.calculate_images_coor(blk1_image_txt_path, blk1_anchor_2d_info)
    blk2_image_coor = utils.calculate_images_coor(blk2_image_txt_path, blk2_anchor_2d_info)

    common_anchor = blk1_anchor_2d_info.keys() & blk2_anchor_2d_info.keys()
    print(f'blk1 has {len(blk1_anchor_2d_info)}, blk2 has {len(blk2_anchor_2d_info)}')
    print(f"Common anchor images number: {len(common_anchor)}")

    # get points3D.txt info
    # blk1_3d_info = utils.parse_points3d_txt(blk1_point3d_path)
    # blk2_3d_info = utils.parse_points3d_txt(blk2_point3d_path)

    if args.match == "":
        print(f"Start finding anchor image feature points.")

        # matched_anchor_img = []
        used_anchor_img = []
        match_pnt_path = utils.get_default_block_path(-1)
        match_pnt_path = os.path.join(match_pnt_path, 'cam_coor.log')
        with open(match_pnt_path, "w") as f:
            for image_name in common_anchor:
                image1_info = blk1_anchor_2d_info[image_name]
                image2_info = blk2_anchor_2d_info[image_name]
                if utils.check_useful_image(image1_info) and\
                    utils.check_useful_image(image2_info):
                    used_anchor_img.append((image_name))
                    info1 = blk1_image_coor[image_name]
                    info2 = blk2_image_coor[image_name]
                    f.write(f"{image_name} "+\
                        f"{info1[0]} {info1[1]} {info1[2]} "+\
                        f"{info2[0]} {info2[1]} {info2[2]}\n")
                # matched_anchor_img.append((image_name))
    
    # print(f"Found total {len(matched_anchor_img)} matched anchor images.")
    print(f"Use total {len(used_anchor_img)} matched anchor images.")

    print(f"Start finding transform matrix.")

    thresh = 0.4    # This thresh is used to determine inlier or outlier.
    transform_mtx = ransac_find_camera_transform(
        used_anchor_img, blk1_image_coor, blk2_image_coor, thresh)

    print(f"Transformation matrix:\n{transform_mtx}")

    print(f"Start writing ply file for the merged model.")
    modify_ply(blk1_ply_path, blk2_ply_path, transform_mtx)

    log_path = utils.get_default_block_path(-1)
    log_path = os.path.join(log_path, "matchlog.log")

    utils.log_inliers(transform_mtx, used_anchor_img,
                      blk1_image_coor, blk2_image_coor, log_path, thresh)

if __name__ == "__main__":
    main()
