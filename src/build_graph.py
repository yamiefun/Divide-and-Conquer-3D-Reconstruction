import argparse
from utils import utils
import bisect
from math_fnc import my_math as mm
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_list", help='path to image_list', required=True)
    parser.add_argument("--init_num", help='number of initial graph',
                        type=int, default=-1)
    parser.add_argument("--mo", help='path to `match.out` file', required=True)
    parser.add_argument("--mt", help='path to `match.txt` file', required=True)
    parser.add_argument("--vio", help='path to vio file', required=True)

    args = parser.parse_args()
    return args


def image_vio_matching(image_list, vio_list) -> None:
    """
        This function will fill the VIO information field in image tuple.
    """
    vio_info = []
    ts_table = [vio.ts for vio in vio_list]
    vio_num = len(ts_table)
    for idx, image in enumerate(image_list):
        image_ts = image.ts
        cloest_idx = bisect.bisect_left(ts_table, image_ts)
        if cloest_idx >= vio_num:
            cloest_idx = vio_num-1
        x, y, z = vio_list[cloest_idx].x, \
            vio_list[cloest_idx].y, \
            vio_list[cloest_idx].z
        image_list[idx] = image_list[idx]._replace(x=x, y=y, z=z)


def image_dist(img1, img2) -> float:
    """
        Calculate distance between two image VIO.
        Distance means (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2.
        Args:
            img1 (namedTuple: Image): Informatoin of image 1.
            img2 (namedTuple: Image): Informatoin of image 2.

        Returns:
            (float): Distance of two input images.
    """
    return (img1.x-img2.x)**2 + (img1.y-img2.y)**2 + (img1.z-img2.z)**2


def add_edge(graph, i, j, weight=1) -> None:
    """
        This function adds edge to graph between node i and j.
    """
    graph[i, j] = weight
    graph[j, i] = weight


def main():
    args = parse_args()

    # Parse files.
    image_list = utils.parse_img_list(args.img_list)
    vio_list = utils.parse_vio(args.vio)
    match_list = utils.parse_match(args.mo)

    # Match image and VIO informations.
    image_vio_matching(image_list, vio_list)

    image_num = len(image_list)
    match_num = len(match_list)

    if args.init_num == -1:
        args.init_num = image_num

    """
    Calculate average distant between two node in init graph.
    Dist between two node = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
    avg_dist = 20 * (sum of dist / # of dist)
    """
    avg_dist = 0
    for i in range(args.init_num-1):
        avg_dist += image_dist(image_list[i], image_list[i+1])
    avg_dist = 20 * (avg_dist/(args.init_num-1))
    print(f"Average distance between two consecutive image in init graph: "
          f"{avg_dist}")

    """
    Calculate height constraint.
    """
    # height = [img.z for img in image_list]
    # height_constraint = (max(height)-min(height)) / 8

    """
    Build init graph.
    I and J is connected if dist(i, j) <= constraint
    """
    graph = np.zeros((image_num, image_num))
    for i, img1 in enumerate(image_list):
        for j, img2 in enumerate(image_list):
            if j < i:
                continue
            # if image_dist(img1, img2) <= avg_dist and \
            #       abs(img1.z-img2.z) <= height_constraint:
            if image_dist(img1, img2) <= avg_dist:
                add_edge(graph, i, j)
            # don't know why need this
            if j <= 20:
                add_edge(graph, i, j)

    """
    SKIP LOGGING PART
    :149~:173
    """

    # Gradually add images to initial graph.
    connect = np.zeros(image_num)
    for i in range(args.init_num):
        connect[i] = 1

    for _ in range(image_num-args.init_num):
        img1, img2, max_sim = 0, 0, 0

        for match in match_list:
            id1, id2, sim = match.id1, match.id2, match.sim
            if (id1 < args.init_num and id2 >= args.init_num) and \
                    (connect[id1] == 1 and connect[id2] == 0) and \
                    sim >= max_sim:     # and sim >= 0.0163
                max_sim = sim
                img1, img2 = id1, id2
        else:
            for match in match_list:
                id1, id2, sim = match.id1, match.id2, match.sim
                if connect[id1] + connect[id2] == 1 and sim > max_sim:
                    max_sim = sim
                    img1, img2 = id1, id2

        if connect[id1] == 1:
            connect[id2] = 1
            for i in range(image_num):
                if graph[i, id1] != 0:
                    add_edge(graph, i, id2)
        else:
            connect[id1] = 1
            for i in range(image_num):
                if graph[i, id2] != 0:
                    add_edge(graph, i, id1)

    # Map image similarity as weigth of edge.
    for match in match_list:
        id1, id2, sim = match.id1, match.id2, match.sim
        if graph[id1, id2] != 0:
            graph[id1, id2] = sim

    # Print modified image similarity file
    utils.log_image_sim(graph)



if __name__ == "__main__":
    main()
