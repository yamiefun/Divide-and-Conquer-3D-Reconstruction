import os
import argparse
from shutil import copyfile
from typing import List
import numpy as np
from math_fnc import my_math as mm
from scipy.spatial.transform import Rotation
from itertools import combinations as comb
from .classes import Match, VIO, Point3D, Point2D, AnchorImg, Image


def get_default_block_path(blk) -> str:
    """
        Generate path for log files. The logs should be put in separate
        folders, named 'blk' and a number behind, ex, 'blk1', 'blk2', etc.
        All of them should be placed under 'test' folder.
        When 'blk' is -1, this function will return where 'src' is.

        Args:
            blk (int): The integer behind 'blk' in the folder name.

        Returns:
            string: Absolute path to the log folders.
    """
    file_path = os.path.realpath(__file__)
    pth = os.path.split(file_path)[0]
    pth = pth.split(os.sep)[0:-2]
    if blk >= 0:
        pth = os.path.join(*pth, f"test", f"blk{blk}")
    else:
        pth = os.path.join(*pth, f"test")

    return f"/{pth}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blk1", default=get_default_block_path(1))
    parser.add_argument("--blk2", default=get_default_block_path(2))
    parser.add_argument("--match", default="")

    args = parser.parse_args()
    return args


def modify_point_format(pnt):
    """
        This function will modify the point format to fit the input of
        superimposition_matrix function.
        Input format is numpy array with shape (N, 3), where N means the
        number of points. This function will transfer all points into
        homogeneous coordinate by append a "1" in the fourth dimention,
        and then return the transpose of the matrix. So the return will be
        a numpy array with shape (4, N), where all elements in the 4th row
        are 1.

        Args:
            pnt (numpy array): with shape (N, 3), where N means the number
                of points.

        Returns:
            ret (numpy array): with shape (4, N) where all elements in the
                4th row are 1.
    """
    pnt = np.array(pnt)
    ret = np.zeros((pnt.shape[0], pnt.shape[1]+1))
    ret[:, :-1] = pnt
    ret[:, -1] = 1
    ret = ret.T
    return ret


def parse_images_txt(image_pth) -> dict:
    """
        This function will extract useful information in `images.txt`,
        including anchor image name, 2d points info, corresponding 3d
        points id.

        The format in images.txt is:
            - Image list with two lines of data per image:
                - IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                - POINTS2D[] as (X, Y, POINT3D_ID)

        Args:
            image_pth (str): Path to 'images.txt' file.

        Returns:
            ret (dict): Key is the NAME in `images.txt`, value is other
                information in `images.txt` except NAME.
    """
    find_anchor = False
    ret = {}
    img_id = None
    Q = [None]*4
    T = [None]*3
    cam_id = None
    img_name = None
    with open(image_pth, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            if find_anchor:
                find_anchor = False
                pnt_2d = line.split()
                pnt_2d = [float(val) for val in pnt_2d]
                useful_2d = []
                for i in range(2, len(pnt_2d), 3):
                    if pnt_2d[i] >= 0:
                        # not -1, means this 2d point can map to 3d
                        new_pnt_2d = Point2D(
                            pnt_2d[i-2], pnt_2d[i-1], int(pnt_2d[i]))
                        useful_2d.append(new_pnt_2d)
                new_anchor = AnchorImg(
                    img_id, Q.copy(), T.copy(), cam_id, useful_2d)
                # ret.append(new_anchor)
                ret[img_name] = new_anchor

                # with open(anchor_pth, "a") as fout:
                #     fout.write(line)

            # if "anchor/2453" in line or "anchor/0222" in line:
            if "anchor" in line:
                find_anchor = True
                img_id, Q[0], Q[1], Q[2], Q[3],\
                    T[0], T[1], T[2], cam_id, img_name = line.split()
                img_id = int(img_id)
                Q = [float(val) for val in Q]
                T = [float(val) for val in T]
                cam_id = int(cam_id)
                # with open(anchor_pth, "a") as fout:
                #     fout.write(line)
    return ret


def parse_points3d_txt(pnt3d_path):
    coor = [None]*3
    rgb = [None]*3
    ret = {}
    with open(pnt3d_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            pnt_id, coor[0], coor[1], coor[2],\
                rgb[0], rgb[1], rgb[2], err, *track = line.split()
            coor = [float(val) for val in coor]
            rgb = [int(val) for val in rgb]
            err = float(err)
            new_pnt3d = Point3D(coor.copy(), rgb.copy(), err, track.copy())
            ret[int(pnt_id)] = new_pnt3d
    return ret


def modify_ply(blk1_ply_path, blk2_ply_path, mtx, out_name):
    out_path = get_default_block_path(-1)
    out_path = os.path.join(out_path, f'{out_name}.ply')
    fmod = open(f"{get_default_block_path(1)}/mod_model.ply", "w")
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
                fmod.write(f"{result}\n")
    fout.close()
    fmod.close()


def modify_ply_icp(blk1_ply_path, blk2_ply_path, mtx):
    out_path = get_default_block_path(-1)
    out_path = os.path.join(out_path, 'merged_model_icp.ply')
    copyfile(blk1_ply_path, out_path)
    coor = [None]*4
    rgba = [None]*4
    fout = open(out_path, 'a')
    fmod = open(f"{get_default_block_path(2)}/mod_model_icp.ply", "w")
    with open(blk2_ply_path, "r") as f:
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
                warp_coor = warp_coor.T
                warp_coor = np.squeeze(warp_coor)
                warp_coor = warp_coor[:3]
                coor = [str(val) for val in warp_coor]
                result = coor + tmp
                result = " ".join(result)
                fout.write(f"{result}\n")
                fmod.write(f"{result}\n")
    fout.close()


def check_useful_image(image_info):
    """
        Check if there is any 2d image point on the anchor image that is built
        into the 3d model.

        Args:
            image_info (namedTuple): The anchor image info parsed by
                `parse_images_txt` function.

        Returns:
            (bool): Ture if there is at least one image point is used in 3d
                model, False otherwise.
    """
    for pnt in image_info.Pnt2D:
        if pnt.Pnt3DID >= 0:
            return True
    return False


def log_inliers(mtx, match_list, blk1, blk2, log_path, thresh):
    """
        This function will generate the log file "matchlog.log".
    """
    cam1_list = []
    cam2_list = []
    inl = []
    outl = []
    inlier, outlier = 0, 0
    name_list = []

    # Extract camera position
    for image_name in match_list:
        name_list.append(image_name)
        cam1 = blk1[image_name]
        cam2 = blk2[image_name]
        cam1_list.append(cam1)
        cam2_list.append(cam2)
    cam1_list = np.array(cam1_list)
    cam2_list = np.array(cam2_list)
    for idx, (pnt1, pnt2) in enumerate(zip(cam1_list, cam2_list)):
        pnt1 = np.concatenate((pnt1, np.array([1])))
        warp_pnt = np.dot(mtx, pnt1)
        warp_pnt = warp_pnt[:3]
        dist = mm.L2_dist(warp_pnt, pnt2)
        if dist > thresh:
            outlier += 1
            outl.append((name_list[idx], warp_pnt, pnt2, dist))
        else:
            inlier += 1
            inl.append((name_list[idx], warp_pnt, pnt2, dist))
    inl.sort()
    outl.sort()
    with open(log_path, "w") as f:
        f.write(f"Inliers:\n")
        for name, p1, p2, dist in inl:
            f.write(f"{name} {p1} {p2} {dist:.3f}\n")
        f.write(f"Outliers:\n")
        for name, p1, p2, dist in outl:
            f.write(f"{name} {p1} {p2} {dist:.3f}\n")


def calculate_images_coor(pth, blk_info):
    """
        This function will calculate the 3d coordinate of every anchor images.
        The result will be returned, and also be written to log file under
        blk folder, called "mod_images.txt".

        Args:
            pth (string): Path to "images.txt" file.
            blk_info (dict): The dictionary returned by function
                `parse_iamges_txt`.

        Returns:
            dic (dict): Key is the image name, value is the 3d coordinates of
                the image.
    """
    pth = os.path.split(pth)[:-1]
    pth = os.path.join(pth[0], f"mod_images.txt")
    dic = {}
    with open(pth, 'w') as f:
        for image_name in blk_info:
            Q = blk_info[image_name].Q
            T = blk_info[image_name].T
            rot = Rotation.from_quat([Q[1], Q[2], Q[3], Q[0]])
            R = rot.as_matrix()
            ret = np.dot(-R.T, T)
            dic[image_name] = ret
            ret = [str(val) for val in ret]
            ret.append(image_name)
            ret = " ".join(ret)
            f.write(f"{ret}\n")
    return dic


def parse_img_list(pth):
    """
        Parse `image_list`, and store them in Image (dataclass) format.

        In this function, (x, y, z) will be filled in all 0s. These field will
        then be updated after matching these image with VIO data.

        Args:
            pth (str): Path of `image_list`.

        Returns:
            img_list (list): List of image information.
    """
    with open(pth, "r") as f:
        lines = f.readlines()
        img_list = [Image(line) for line in lines]
    return img_list


def parse_vio(pth) -> List[VIO]:
    """
        Parse VIO log file. Only extract the timestamp and first three number,
        i.e., (x, y, z) of VIO sensor from the log file. These information will
        be stored in a dataclass call VIO.

        The return will be a list containing all records in the vio log file.

        Args:
            pth (str): Path to VIO log file.

        Returns:
            vio_list (list): List of vio information.
    """
    with open(pth, 'r') as f:
        lines = f.readlines()
        vio_list = [VIO(line) for line in lines]
    return vio_list


def parse_match(pth) -> List[Match]:
    """
        Parse `match.out`.
        Every line in `match.out` will be parse into a dataclass called
        `Match`, including id1, id2, and similarity between them.

        Args:
            pth (str): Path of `match.out`.

        Returns:
            match_list (list): List of image matches.
    """
    with open(pth, 'r') as f:
        lines = f.readlines()
        match_list = [Match(line) for line in lines]
    return match_list


def log_image_sim(graph, thresh=-1) -> None:
    """
        This function will log the modified image matches to
        `test/mod_match.out`.

        Args:
            graph (np.array): The adjacency map of the graph.
            thresh (int): Max number of edges of a node.

        Returns:
            None
    """
    out_path = get_default_block_path(-1)
    out_path = os.path.join(out_path, f'mod_match.out')
    with open(out_path, 'w') as fout:
        for id1, tmp in enumerate(graph):
            tmp = [(idx, sim) for idx, sim in enumerate(tmp)]
            tmp.sort(key=lambda tup: tup[1], reverse=True)
            if thresh != -1:
                tmp = tmp[:thresh]
            for id2, sim in tmp:
                fout.write(f"{id1} {id2} {sim}\n")


def log_node(init_num) -> None:
    """ This function will generate `node.csv` file. """
    out_path = get_default_block_path(-1)
    out_path = os.path.join(out_path, f'node.csv')
    with open(out_path, 'w') as fout:
        fout.write(f"Id,Label,Color,idx\n")
        for i in range(init_num):
            fout.write(f"{i},{i},red,{i}\n")


def log_edge(init_num, graph) -> None:
    """ This function will generate `edge.csv` file. """
    out_path = get_default_block_path(-1)
    out_path = os.path.join(out_path, f'edge.csv')
    with open(out_path, 'w') as fout:
        fout.write(f"Source,Target,vio\n")
        for i, j in list(comb(np.arange(init_num), 2)):
            if graph[i, j] != 0:
                if j == i+1:
                    fout.write(f"{i},{j},1\n")
                else:
                    fout.write(f"{i},{j},0\n")


def log_match_import(pth, graph) -> None:
    """ This function will generate `match_import.txt` file. """
    out_path = get_default_block_path(-1)
    out_path = os.path.join(out_path, f'match_import.txt')
    fout = open(out_path, 'w')
    with open(pth, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line[0] == '#':
                continue
            if line[0] == '\n':
                line1 = f.readline()
                line2 = f.readline()
                num = int(f.readline())
                id1, img1 = line1.split()
                id2, img2 = line2.split()
                if graph[int(id1), int(id2)] == 0:
                    for i in range(num):
                        _ = f.readline()
                else:
                    fout.write(f"{img1} {img2} {num}\n")
                    id_list = []
                    for i in range(num):
                        line = f.readline()
                        line = line.split()
                        try:
                            id_list.append([line[0], line[3]])
                        except Exception as e:
                            print(f"EOF of match.txt")
                            break
                    for i in range(2):
                        line = [s[i] for s in id_list]
                        fout.write(f"{' '.join(line)}\n")
    fout.close()


def log_graph(graph, image_num) -> None:
    """ This function will generate `graph.log` file. """
    out_path = get_default_block_path(-1)
    out_path = os.path.join(out_path, f'graph.log')
    with open(out_path, 'w') as fout:
        for i in range(image_num):
            for j in range(image_num):
                fout.write(f"{int(graph[i, j])} ")
            fout.write(f"\n")
