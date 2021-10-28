# README
This document will teach you how to apply IMU information and using divide-and-conquer algorithm to improve the efficiency of SFM  completely.
## Contents
+ [Combine IMU Information with Video](#Combine-IMU-Information-with-Video)
+ [Image Clustering](#Image-Clustering)
+ [3D Reconstruction](#3D-Reconstruction)
+ [Merge Sub-models](#Merge-Sub-models)
+ [Output Explanation](#Output-Explanation)
## Combine IMU Information with Video
The main feature of our work is using IMU information to improve the result of 3d reconstructure. This tutorial will teach you how to merge IMU information and video frames together.

Note: Some steps will be different if you're using different equipment to record IMU information and videos.
### A. Create Rosbag File
Record bag file, including camera and imu information. Open 3 terminals, run the commands below in them separately.
1. usb_cam
    
    Capture USB camera frames.
    [REF](https://blog.csdn.net/dengheCSDN/article/details/78983993)
    ```
    # terminal 1
    $ cd catkin_cam
    $ source devel/setup.bash
    $ roslaunch usb_cam usb_cam-test.launch
    ```
2. IMU

    Capture IMU information.
    [REF](http://wiki.ros.org/razor_imu_9dof)
    ```
    # terminal 2
    $ cd catkin_ws
    $ source devel/setup.bash
    $ roslaunch razor_imu_9dof razor-pub-and-display.launch
    ```
3. Recode bag file
    ```
    # terminal 3
    $ rosbag record /usb_cam/image_raw /imu
    ```
Because we need to resize the frames, we need to separate the bag file into frames and IMU information.

4. Separate bag into frames and IMU
    [REF](https://github.com/ethz-asl/kalibr)
    ```
    $ cd kalibr_ws
    $ source devel/setup.bash
    # extract rosbag to image & imu
    $ kalibr_bagextractor --image-topics /usb_cam/image_raw --imu-topics /imu --output-folder <output path> --bag <bag file>
    # for example,
    $ kalibr_bagextractor --image-topics /usb_cam/image_raw --imu-topics /imu --output-folder out_fold --bag ~/Desktop/2019-11-18-20-06-22.bag
    ```
    A folder called `out_fold` will be generated, and there will be two files in it, `cam0` folder contains all frames in the bag file and `imu0.csv` records all IMU information.

5. Subsample frames

    Reduce the number of frames.
6. Resize images

    Resize all image to 640*360.
7. Merge images and IMU to create a modified bag file

    **(Note: Start from this step if you're using `iii_video2image.py` to parse video and IMU info.)**
    ```
    $ kalibr_bagcreater --folder <folder path> --output-bag <output bag file name>
    ```
    where `folder path` is the folder containing `cam0` and `imu0.csv`.
    
---
### B. Vins_mono
Run vins_mono to obtain extrinsic matrix and `vins_result`.
The output will be stored in the path set in the config file 
```
~/vins-mono-catkin_ws/src/VINS-Mono/config/euroc/test_gopro.yaml
```
[REF](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)

Open 3 terminals and run the commands below in them separately.
```
# terminal 1, launch gopro
$ cd vins-mono-catkin_ws
$ source devel/setup.bash
$ roslaunch vins_estimator test_gopro.launch
```
```
# terminal 2, launch vins_mono gui
$ cd vins-mono-catkin_ws
$ source devel/setup.bash
$ roslaunch vins_estimator vins_rviz.launch
```
```
# terminal 3, play bag file
$ rosbag play <bag file>
```
After finish playing the bag file in terminal 3, two files will be automatically saved in the path set in the `yaml` config file, which are `vins_result_loop.csv` and `vins_result_no_loop.csv`.

---
### C. VisualSFM
Use VisualSFM to caluclate pairwise image matching and sift feature for each image.

1. Convert `.png` to `.jpg` in `cam0` folder.

    **(Note: Dont't need this step if you're running with `iii_video2image.py`.)**
    ```
    $ mogrify -format jpg *.png && rm *.png
    ```
2. Run Visual SFM
    ```
    # start VisualSFM gui
    $ VisualSFM
    # run the process below in VSFM
    1. (click) open multiple images
    2. (click) compute missing matches for input images
    # export match.txt
    3. (click) SfM -> Pairwise Matching -> Export F-Matrix Matches, save as `match.txt`
    ```
    VisualSFM will generate `.mat` and `.sift` file for each frame in `cam0`.

---
### D. libvot
Use libvot to compute the similarity between images by sift feature.
1. Generate list of image name and sift name.
    There are two kinds of images, one is called "route frames", which are recorded with IMU information, the other is called "additional frames", which don't have IMU information. In this step, please list all route frames into `image_list` first before additional frames. 
    ```
    # list route frames
    $ ls $PWD/cam0/*.jpg > image_list
    $ ls $PWD/cam0/*.sift > sift_list
    ```
    If there're images in different folders, use `>>` to append them to the list file.
    ```
    # list additional frames
    $ ls $PWD/cam0/*.jpg >> image_list
    $ ls $PWD/cam0/*.sift >> sift_list
    ```
    Remember to use absolute path, not related path.

2. Run `image_search`.
    ```
    $ image_search <sift list> <output folder path>
    # for example,
    $ image_search ./sift_list ./output
    ```

---
### E. Build Viewing Graph
Run our code. This code will generate file `match_import.txt`.

Run `build_graph.py` to build viewing graph. This program will trim some unreasonable edge in viewing graph by checking IMU information. 
Usage: 
```
$ python3 build_graph.py \
    --img_list <path to `image_list`>   \
    --init_num <number of route frames> \
    --mo       <path to `match.out`>    \
    --mt       <path to `match.txt`>    \
    --vio      <path to vins_result file>
```
This code will generate two improtant files, `match_import.txt` and `mod_match.out`.
+ `match_import.txt` is modified from `match.txt`, you could directly use VisualSFM to build 3d model with images and this file. See [step F](#F-VisualSFM) for more information.
+ `mod_match.out` is modified from `match.out`, contains modified viewing graph. You could apply divide and conquer algorithm on this file.
---
### F. VisualSFM
Use VisualSFM to reconstruct the 3d model.
```
1. (click) Open multiple images
2. (click)SfM -> Pairwise Matching -> Export Features Matches
          Select `match_import.txt`.
3. (click) Compute 3D Reconstruction
```
---

## Image Clustering
Using `decomposition.py` to separate images into multi-clusters.

Usage:
```
$ python3 decomposition.py \
    --match     <path to `match.out` or `mod_match.out`> \
    --img_list  <path to `image_list`>   \
    --init_num  <number of route images> \
    --clust_num <number of clusters>
```
This code will automatically generate a folder in the same place of `src`, called `test`. Inside `test`, there will be some folders, i.e., `block<i>` for `i` from `0` to `clust_num-1`. All images will be separated and placed in these folder. Additionally, there will be a folder called `anchor` in all `block` folder, which contains a set of anchor images. Anchor images in these clusters are all identical.
   
    **Notes: If the separated result is bad or not reasonable, you can separate the images into multi-clusters by yourself manually.**
---

## 3D Reconstruction
Use [COLMAP](https://colmap.github.io/) to build 3D models for those clusters generated above. 

Some log file generated by COLMAP will be used in the later steps, including `cameras.txt`, `images.txt`, `points3D.txt`, and a `ply` model file. Note that you should convert the `ply` file from binary to ASCII format, and name it `model.ply`.

---

## Merge Sub-models
For convenience, you should create two folders in `test` folder generated in step 1, called `blk1` and `blk2`. Then you can placed the log files generated by COLMAP mentioned above into them.
```
|_src/
|    |_math_fnc/
|    |_utils/
|    |_decomposition.py
|    |_...
|
|_test/
     |_blk1/
     |    |_cameras.txt
     |    |_images.txt
     |    |_points3D.txt
     |    |_model.ply
     |
     |_blk2/
          |_(same content as blk1)
```

Usage:`$ python3 merge_camera.py` or `$ python3 merge_model.py`

This code will calculate the transformation matrix between block1 and block2, then align block1 with block2 by multiply all 3D points in block by the transformation matrix. A `ply` file called `merged_model_cam.ply` or `merged_model_pnt.ply` will be generated, which is the result of merging the two sub-model.
A log file called `mod_points.log` be created, recording all 3d points in block1 after multipling by the transformation matrix.

If you're merging more than 2 clusters, you should select one of the sub-model as a pivot, aligning all other sub-models at it. You should manually copy all 3d points recorded in `mod_points.log` and paste them into the pivot sub-model `.ply` file.

---

## Output Explanation
### cam_coor.log
This file will be generated under `test` folder.
Every anchor image that be built in "both" block will be recorded in this file.
Each line contains the information of the anchor images 3d coordinate in both block.
```
# IMAGE_NAME X Y Z U V W
# ex:
anchor/100.jpg	1.003 -0.328 2.500 1.199 0.574 -3.455
```
(X, Y, Z) and (U, V, W) is the 3d coordinate of the anchor image in block1 and block2.
### matchlog.log
This file will be generated under `test` folder. It records the result of warpped anchor images coordinate and the distance to the corresponding anchor images in another block.
```
# IMAGE_NAME [X Y Z] [U V W] L2_DISTANCE
# ex:
anchor/100.jpg [ 1.264  0.576 -3.478] [ 1.199  0.574 -3.455] 0.069
```
(X, Y, Z) is the warpped 3d coordinate of anchor image in block1, (U, V, W) is the original 3d coordinate of anchor image in block2. 
### merged_model.ply
This file will be generated under `test` folder. This is the merged model file, which is based on block2 `model.ply` file with warpped block1 model information appended. You can use tools like `colmap` to view the merged model.
Note that the "element vertex" number in this file is wrong but harmless. You can modify it manually if needed.
### mod_images.txt
This file will be generated under both block folders. It contains the 3d coordinates of all anchor images in that block.
```
# X Y Z IMAGE_NAME
# ex:
1.003 -0.328 2.500 anchor/100.jpg
```
(X, Y, Z) is the 3d coordinate of the anchor image in that block.
### mod_points.log
This file will record all 3d points in a model, multiplied by the transformation matrix. The format will be the same as other `ply` file.

---