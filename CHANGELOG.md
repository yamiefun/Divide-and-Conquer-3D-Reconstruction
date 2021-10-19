# Changelog
All notable changes to this project will be documented in this file.

## 2021-10-19
### Changed
+ Change `namedtuple` to [`dataclass`](https://docs.python.org/3.7/library/dataclasses.html). Need Python3.7 for fully support.

## 2021-10-18
### Fixed
+ Fixed `mod_match.out` file bugs. Similarity of image pairs should be printed in descending order. 


## 2021-09-30
### Added
+ `merge_model.py` can merge two sub-model by match anchor 3d points cloud. The usage is the same as `merge_camera.py`.

### Changed
+ The merged model file name will be `merged_model_cam.ply` if running with `merge_camera.py`, and `merged_model_pnt.ply` if running with `merge_model.py`.

### Fixed
+ Fixed `merge_model.py` bugs. Calculate transformation matrix with scale, and store the matrix with the smallest error.



## 2021-08-31
### Fixed
+ The 3d coordinate of images in the `images.txt` should be obtained by `-R^t * T`, where `R^t` is transpose of rotation matrix, `T` is translation vector. Rotation matrix is calculated from the quaternion representation using [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) module.
+ After calculating the transformation matrix between two 3d models, the first model will be warpped to align the second model.

### Changed
+ The filename of `ply` file should be `model.ply`, not `model2.ply`.
### Added
+ The explanation of each log file is added in `README`.