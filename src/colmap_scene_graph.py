import sys
import sqlite3
from typing import List
import numpy as np
import os
import argparse

from utils.classes import AdjacencyMap, ColmapImage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path",
                        help="path to database.db",
                        required=True)
    parser.add_argument("--image_list_path",
                        help="output path to image_list",
                        required=True)
    parser.add_argument("--sim_graph_path",
                        help="output path to similarity graph",
                        required=True)
    # parser.add_argument("--dif_graph_path",
    #                     help="output path to difference graph",
    #                     required=True)
    parser.add_argument("--min_num_matches", type=int, default=15)
    args = parser.parse_args()
    return args


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)


def blob_to_array(blob, dtype, shape=(-1,)):
    # return np.fromstring(blob, dtype=dtype).reshape(*shape)
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


def pair_id_to_image_ids(pair_id):
    MAX_IMAGE_ID = 2**31 - 1
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def generate_image_list(output_path, db):
    """
        Generate `image_list` file.
        The output image will be sorted by image_id(start from 1) in database.
        The return is a dictionary of ColmapImage. Key is image_id, value is
        image informations.

        Args:
            output_path (str): Output file path.
            db (str): Path to database.

        Returns:
            image_dict (dict): Key: image_id,
                               Value: ColmapImage.
    """
    rows = db.execute("SELECT image_id, name FROM images ORDER BY image_id;")
    image_dict = {}
    with open(output_path, 'w') as fout:
        for image_id, image_name in rows:
            fout.write(f"{image_name}\n")
            image_dict[image_id] = \
                ColmapImage(ImgID=image_id, ImgName=image_name)
    return image_dict


def find_keypoint_num_per_image(db, images):
    """
        Find number of keypoints in each image in image_list and fill into the
        field "KeypointNum".

        Args:
            db (str): Path to database.
            image_list (list): List of ColmapImg.

        Returns:
            None
    """

    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints ORDER BY image_id;"))
    # for key in keypoints:
    #     print(f"image id: {key} # of keypoints: {len(keypoints[key])}")
    # for image in images:
    #     image.KeypointNum = len(keypoints[image.ImgID])
    for image_id in images:
        images[image_id].KeypointNum = len(keypoints[image_id])


def find_inlier_match(min_match_thresh, db) -> List:
    """
        Parse matches conform to geometry verification, where number of inliers
        should greater then a minimum match threshold.

        Args:
            min_match_thresh (int): Minimum match threshold.
            db (str): Path to database.

        Returns:
            match_list (list): List of tuple. Each tuple contains
                (image_id1, image_id2, number of match).
    """
    rows = db.execute(
            "SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;",
            (min_match_thresh,))
    match_list = []
    for row in rows:
        pair_id = row[0]
        inlier_matches = \
            np.frombuffer(row[1], dtype=np.uint32).reshape(-1, 2)
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        match_list.append((image_id1, image_id2, inlier_matches.shape[0]))

    return match_list


def build_graphs(images, matches):
    """
        Build similarity graph and difference graph for given images.

        Args:
            images (dict): Key is image_id, value is ColmapImg.
            matches (list): Function `find_inlier_match` output.
    """
    image_ids = [image_id for image_id in images]
    sim_graph = AdjacencyMap(image_ids, self_weight=1.0)
    dif_graph = AdjacencyMap(image_ids, self_weight=0.0)
    for match in matches:
        image_id1, image_id2, match_num = match

        sim_graph_weight = match_num / \
            (images[image_id1].KeypointNum + images[image_id2].KeypointNum)

        sim_graph.add_edge(image_id1=image_id1,
                           image_id2=image_id2,
                           weight=sim_graph_weight)
        dif_graph.add_edge(image_id1=image_id1,
                           image_id2=image_id2,
                           weight=1-sim_graph_weight)

    sim_graph.map_sort(reverse=True)
    dif_graph.map_sort(reverse=False)

    return sim_graph, dif_graph


def log_graph(output_path, graph):
    with open(output_path, "w") as fout:
        for image_id1 in graph.map:
            for image_id2, weight in graph.map[image_id1]:
                fout.write(f"{image_id1} {image_id2} {weight:.3f}\n")
    pass


def main():
    args = parse_args()
    # db = COLMAPDatabase.connect(args.db_path)
    # connect to database
    db = sqlite3.connect(args.db_path)
    # rows = db.execute("SELECT * FROM images;")

    # generate image_list and images info
    images = generate_image_list(args.image_list_path, db)

    # find keypoint number for each image
    find_keypoint_num_per_image(db, images)

    # find match information
    matches = find_inlier_match(args.min_num_matches, db)

    # build viewing graph
    similarity_graph, difference_graph = build_graphs(images, matches)

    # record similarity graph
    log_graph(args.sim_graph_path, similarity_graph)
    # rows = db.execute("pragma table_info(keypoints);")
    # for row in rows:
    #     print(row)
    # print(images)
    # print(matches)


if __name__ == "__main__":
    main()
