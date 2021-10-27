import os
import numpy as np
from utils import utils
from math_fnc import my_math as mm
from transform import superimposition_matrix

from merge_camera import ransac_find_camera_transform
from merge_model import ransac_find_transform


def main():
    args = utils.parse_args()
    blk1_image_txt_path = os.path.join(args.blk1, f"images.txt")
    blk2_image_txt_path = os.path.join(args.blk2, f"images.txt")
    whole_image_txt_path = os.path.join(args.whole, f"images.txt")

    blk1_point3d_path = os.path.join(args.blk1, f"points3D.txt")
    blk2_point3d_path = os.path.join(args.blk2, f"points3D.txt")
    whole_point3d_path = os.path.join(args.whole, f"points3D.txt")

    blk1_ply_path = os.path.join(args.blk1, f"model.ply")
    blk2_ply_path = os.path.join(args.blk2, f"model.ply")
    whole_ply_path = os.path.join(args.whole, f"model.ply")

    print(f"Start parsing COLMAP log files.")
    # get images.txt info
    blk1_anchor_2d_info = utils.parse_images_txt(blk1_image_txt_path)
    blk2_anchor_2d_info = utils.parse_images_txt(blk2_image_txt_path)
    whole_anchor_2d_info = utils.parse_images_txt(whole_image_txt_path)

    # get points3D.txt info
    blk1_3d_info = utils.parse_points3d_txt(blk1_point3d_path)
    blk2_3d_info = utils.parse_points3d_txt(blk2_point3d_path)
    whole_3d_info = utils.parse_points3d_txt(whole_point3d_path)

    # calculate 3d coor of each image
    blk1_image_coor = utils.calculate_images_coor(
        blk1_image_txt_path, blk1_anchor_2d_info)
    blk2_image_coor = utils.calculate_images_coor(
        blk2_image_txt_path, blk2_anchor_2d_info)
    whole_image_coor = utils.calculate_images_coor(
        whole_image_txt_path, whole_anchor_2d_info)

    common_anchor = blk1_anchor_2d_info.keys() & blk2_anchor_2d_info.keys()
    print(f'blk1 has {len(blk1_anchor_2d_info)}, '
          f'blk2 has {len(blk2_anchor_2d_info)}')
    print(f"Common anchor images number: {len(common_anchor)}")

    print(f"Start finding anchor image feature points for merge by camera.")
    used_anchor_img = []
    for image_name in common_anchor:
        image1_info = blk1_anchor_2d_info[image_name]
        image2_info = blk2_anchor_2d_info[image_name]
        if utils.check_useful_image(image1_info) and\
                utils.check_useful_image(image2_info):
            used_anchor_img.append((image_name))
    print(f"Use total {len(used_anchor_img)} matched anchor images.")

    if args.match == "":
        print(f"Start finding anchor image feature points for merge by point.")
        matched_3d_set = []  # save the match of 3d point id in two set
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

    thresh = 0.4    # This thresh is used to determine inlier or outlier.
    # camera merge
    print(f"Start finding transform matrix.")
    cam_mtx = ransac_find_camera_transform(
        used_anchor_img, blk1_image_coor, blk2_image_coor, thresh)
    print(f"Transformation matrix (Camera):\n{cam_mtx}")
    print(f"Start writing ply file for the merged model.")
    utils.modify_ply(blk1_ply_path, blk2_ply_path, cam_mtx,
                     f"merge_model_by_cam")

    # point merge
    print(f"Start finding transform matrix.")
    pnt_mtx = ransac_find_transform(
        matched_3d_set, blk1_3d_info, blk2_3d_info, thresh)
    print(f"Transformation matrix (Point):\n{pnt_mtx}")
    print(f"Start writing ply file for the merged model.")
    utils.modify_ply(blk1_ply_path, blk2_ply_path, pnt_mtx,
                     f"merge_model_by_pnt")

    # Calculate outlier ratio without using GT.
    print('')
    print('Outlier ratio without using GT:')
    print('camera error: ')
    mm.calculate_err(cam_mtx, matched_3d_set,
                     blk1_3d_info, blk2_3d_info)
    print('point error: ')
    mm.calculate_err(pnt_mtx, matched_3d_set,
                     blk1_3d_info, blk2_3d_info)

    # save the match of 3d point id in two set blk1 and whole
    print('')
    print(f"Start finding common image feature points in blk1 and whole.")

    matched_3d_set_blk1_whole = []
    for image_name in blk1_anchor_2d_info:
        if image_name in whole_anchor_2d_info:
            image1_info = blk1_anchor_2d_info[image_name]
            image2_info = whole_anchor_2d_info[image_name]

            # find useful 2d point appear in both image set
            for pnt2d in image1_info.Pnt2D:
                for target in image2_info.Pnt2D:
                    if pnt2d.x == target.x and pnt2d.y == target.y:
                        matched_3d_set_blk1_whole.append((pnt2d.Pnt3DID,
                                                          target.Pnt3DID))
                        break

    # save the match of 3d point id in two set blk2 and whole
    print(f"Start finding common image feature points in blk2 and whole.")
    
    matched_3d_set_blk2_whole = []
    for image_name in blk2_anchor_2d_info:
        if image_name in whole_anchor_2d_info:
            image1_info = blk2_anchor_2d_info[image_name]
            image2_info = whole_anchor_2d_info[image_name]

            # find useful 2d point appear in both image set
            for pnt2d in image1_info.Pnt2D:
                for target in image2_info.Pnt2D:
                    if pnt2d.x == target.x and pnt2d.y == target.y:
                        matched_3d_set_blk2_whole.append((pnt2d.Pnt3DID,
                                                          target.Pnt3DID))
                        break

    # Find transform matrix between blk2 and whole.
    print('Find transform matrix between blk2 and whole.')
    matched_set_blk2_whole = np.array(matched_3d_set_blk2_whole)
    blk2_temp = []
    whole_temp = []
    for idx in range(len(matched_set_blk2_whole)):
        pnt = matched_set_blk2_whole[idx]
        blk2_temp.append(blk2_3d_info[pnt[0]].coor)
        whole_temp.append(whole_3d_info[pnt[1]].coor)
    blk2_mod = utils.modify_point_format(blk2_temp)
    whole_mod = utils.modify_point_format(whole_temp)
    blk2_whole_mtx = superimposition_matrix(blk2_mod, whole_mod, scale=True)

    print('Calculate ouliers.')
    # Calculate distance between blk2 and whole
    dist_thresh = 1.5
    outlier_pnt = 0
    outlier_cam = 0
    for pnt_idx in matched_set_blk2_whole:
        pnt1_coor = np.array(blk2_3d_info[pnt_idx[0]].coor)
        pnt1_coor = np.concatenate((pnt1_coor, np.array([1])))
        pnt2_coor = np.array(whole_3d_info[pnt_idx[1]].coor)
        
        warp_coor = np.dot(blk2_whole_mtx, pnt1_coor)
        warp_coor = (warp_coor/warp_coor[-1])[:3]
        
        dist = mm.L2_dist(warp_coor, pnt2_coor)
        if dist > dist_thresh:
            outlier_pnt += 1
            outlier_cam += 1

    # Calculate distance between blk1 and whole
    matched_set_blk1_whole = np.array(matched_3d_set_blk1_whole)
    blk1_a = []
    for pnt in blk1_3d_info:
        coor = blk1_3d_info[pnt].coor
        blk1_a.append(coor)
    blk1_mod = utils.modify_point_format(blk1_a)

    for pnt_idx in matched_set_blk1_whole:
        pnt1_coor = np.array(blk1_3d_info[pnt_idx[0]].coor)
        pnt1_coor = np.concatenate((pnt1_coor, np.array([1])))
        pnt2_coor = np.array(whole_3d_info[pnt_idx[1]].coor)

        warp_pnt_coor = np.dot(pnt_mtx, pnt1_coor)
        warp_pnt_coor = (warp_pnt_coor/warp_pnt_coor[-1])
        warp_pnt_coor = np.dot(blk2_whole_mtx, warp_pnt_coor)
        warp_pnt_coor = (warp_pnt_coor/warp_pnt_coor[-1])[:3]
      
        dist_pnt = mm.L2_dist(warp_pnt_coor, pnt2_coor)
        if dist_pnt > dist_thresh:
            outlier_pnt += 1
        
        warp_cam_coor = np.dot(cam_mtx, pnt1_coor)
        warp_cam_coor = (warp_cam_coor/warp_cam_coor[-1])
        warp_cam_coor = np.dot(blk2_whole_mtx, warp_cam_coor)
        warp_cam_coor = (warp_cam_coor/warp_cam_coor[-1])[:3]

        dist_cam = mm.L2_dist(warp_cam_coor, pnt2_coor)
        if dist_cam > dist_thresh:
            outlier_cam += 1

    # Calculate outlier ratio for both methods
    out_ratio_pnt = outlier_pnt / (len(matched_set_blk2_whole) + \
        len(matched_set_blk1_whole))
    print(f"Outlier ratio of merge by point with GT: {out_ratio_pnt*100}%")
    out_ratio_cam = outlier_cam / (len(matched_set_blk2_whole) + \
        len(matched_set_blk1_whole))
    print(f"Outlier ratio of merge by camera with GT: {out_ratio_cam*100}%")


if __name__ == '__main__':
    main()
