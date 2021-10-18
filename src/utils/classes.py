from dataclasses import dataclass
from typing import Any, List


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
    x: float
    y: float
    z: float


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
