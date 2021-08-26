# Changelog
All notable changes to this project will be documented in this file.

## 2021-08-31
### Fixed
+ The 3d coordinate of images in the `images.txt` should be obtained by `-R^t * T`, where `R^t` is transpose of rotation matrix, `T` is translation vector. Rotation matrix is calculated from the quaternion representation using [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) module.
+ After calculating the transformation matrix between two 3d models, the first model will be warpped to align the second model.

### Changed
+ The filename of `ply` file should be `model.ply`, not `model2.ply`.
### Added
+ The explanation of each log file is added in `README`.