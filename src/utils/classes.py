from dataclasses import dataclass
from typing import List
import os


@dataclass
class Match:
    """ Match class for storing image pairs parsed from `match.out`.

        Attributes:
            id1 (int): The image id of the first image.
            id2 (int): The image id of the second image.
            sim (float): The image similarity between image id1 and id2.
    """
    id1: int
    id2: int
    sim: float

    def __init__(self, line) -> None:
        id1, id2, sim = line.split()
        self.id1, self.id2, self.sim = int(id1), int(id2), float(sim)


@dataclass
class VIO:
    """ VIO information parsed from vio log file.

        Attributes:
            ts (int): Timestamp of the record.
            x (float): X value in the 3d coordinate.
            y (float): Y value in the 3d coordinate.
            z (float): Z value in the 3d coordinate.
    """
    ts: int
    x: float
    y: float
    z: float

    def __init__(self, line) -> None:
        ts, x, y, z, *_ = line.split(',')
        self.ts, self.x, self.y, self.z = int(ts), float(x), float(y), float(z)


@dataclass
class Image:
    """ Image information parsed from `image_list`.

        Attributes:
            ts (int): Timestamp of the image, which should be parsed from the
                        image name.
            path (str): Path of the image.
            x (float): The X value of the corresponding VIO record.
            y (float): The Y value of the corresponding VIO record.
            z (float): The Z value of the corresponding VIO record.
    """
    ts: int
    path: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, line) -> None:
        img_name = os.path.split(line)[1]
        self.ts = int(os.path.splitext(img_name)[0])
        self.path = line

    def update(self, coor) -> None:
        self.x = coor[0]
        self.y = coor[1]
        self.z = coor[2]


@dataclass
class Point3D:
    """ 3D point information parsed from `points3D.txt`.

        Attributes:
            coor (List): [X, Y, Z] value of the 3d point.
            rgb (List): [R, G, B] value of the 3d point.
            err (float): Error of the 3d point.
            track (List): Other useless informations.

    """
    coor: List[float]
    rgb: List[int]
    err: float
    track: List[str]


@dataclass
class Point2D:
    """ 2D point information parsed from `images.txt`.

        Attributes:
            x (float): X.
            y (float): Y.
            Pnt3DID (int): POINT3D_ID.
    """
    x: float
    y: float
    Pnt3DID: int


@dataclass
class AnchorImg:
    """ Anchor image information parsed from `images.txt`.

        Attributes:
            ImgID (int): IMAGE_ID.
            Q (list): QW, QX, QY, QZ corresponding to Q[0] to Q[3].
            T (list): TX, TY, TZ corresponding to T[0] to T[2].
            CamID (int): CAMERA_ID.
            Pnt2D (list): Include useful 2d points informations, stored
                in Point2D (dataclass) format.
    """
    ImgID: int
    Q: List[float]
    T: List[float]
    CamID: int
    Pnt2D: List[Point2D]


@dataclass
class ClusterList:
    def __init__(self, num, labels) -> None:
        self.clust = [set() for _ in range(num)]
        for idx, i in enumerate(labels):
            self.clust[i].add(idx)

    def print_lens(self) -> None:
        """
            Print len of every cluster.

            Args:
                None

            Returns:
                None
        """
        for idx, clust in enumerate(self.clust):
            print(f"Number of image in cluster {idx}: {len(clust)}")

    def add_node(self, tar=0, ref=0) -> None:
        """
            Add target to cluster. Target should be in the same cluster as the
            reference.
        """
        clust_id = self.find(ref)
        assert clust_id != -1,\
            (f"ClusterList add_node() fail because reference node doesn't "
             f"belong to any cluster.")
        self.clust[clust_id].add(tar)

    def find(self, idx) -> int:
        """
            Find the target idx in all clusters. If founded, return the id of
            the cluster, otherwise, return -1 instead.

            Args:
                idx (int): Target index.

            Returns:
                (int): Cluster id or -1.
        """
        for clust_id, clust in enumerate(self.clust):
            if idx in clust:
                return clust_id
        return -1

    def in_different_clust(self, pair) -> bool:
        clust1 = self.find(pair.id1)
        clust2 = self.find(pair.id2)
        assert clust1 != -1 and clust2 != -1,\
            (f"ClusterList in_different_clust error: id not found in any"
             f" cluster.")
        return clust1 != clust2
