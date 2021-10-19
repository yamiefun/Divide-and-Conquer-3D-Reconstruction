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
class MatchPair:
    """ Match pairs parsed from `match.out`.

        Attributes:
            id1 (int): Image1's id.
            id2 (int): Image2's id.
            score (float): The similarity score between image1 and image2.
    """
    id1: int
    id2: int
    score: float

    def __init__(self, line) -> None:
        id1, id2, score = line.split()
        self.id1, self.id2, self.score = int(id1), int(id2), float(score)