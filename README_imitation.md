# "Good Robot! Now Watch This!": Building Generalizable Embeddings via Reinforcement Learning

## Key Components

**`demo.py`**

This file defines the `Demonstration` class, which is a data structure to store task demonstrations. The `Demonstration` class reads demos from a parent folder. This parent folder contains subfolders for
(1) color images, (2) color heightmaps, (3) depth images, and (4) depth heightmaps. It also contains a `transitions` directory which contains information on executed actions. It is compatible with any task 
demonstration generated with the `touch.py` script.

**`touch.py`**

This is a utility to generate task demonstrations by specifying actions with key presses and specifying action location by clicking on images of the simulated workspace. The saved demonstrations can be directly loaded in downstream scripts using the interface provided by `demo.py`. A description of how to use the script is provided below.

**`models.py`**

This file defines `PixelNet`, the architecture of the DQN RL policies we use in See-SPOT-Run. These
policies are used to extract embedding functions *E*. The models defined here are instantiated,
 trained, and evaluated in `trainer.py`.

**`trainer.py`**

This file defines the `Trainer` class, a wrapper for `PixelNet` that implements a training and
 testing loop, Experience Replay, and loading and preprocessing data from demos and the live test
 environment.

We provide instructions to train these policies on each of 4 tasks (row, stack, unstack,
 vertical square) below. Training these DQN RL policies is ***Step I. Before Demonstration***
 in Reinforcement Learning Before Demonstration (RLBD).

**`robot.py`**

This file defines all simulation and real robot related code. It contains utilities to get data from
the simulated sensors as well as information about the simulated state (*e.g.* positions,
orientations, IDs, etc. of objects in the scene). It also contains analogous utilities to get data
from the real robot sensors. Lastly, it contains functions to assess task progress for each task
(row, stack, unstack, and vertical square). These functions are `check_row`, `check_stack`,
`unstacking_partial_success`, and `vertical_square_partial_success`. We also provide a
`manual_progress_check` function that is used in some of the real robot experiments where we did
not implement a scripted observer to assess progress (don't have absolute positions and orientations
of blocks during real experiments, and must estimate them from depth values, which can be
inaccurate).

**`utils.py`**

This file defines various miscellaneous utility functions.

The first important function, `compute_demo_dist`, has two steps: first, it computes a
candidate test action by selecting the test action with minimal Euclidean distance to the
target demo action (distance computed in embedding space). It does this for each Policy-Demonstration
Pair (PDP, *M x N* total), then picks the action from the *M x N* total candidate actions by
comparing their Euclidean match distance with the demo action. This Euclidean match distance is
what we refer to as the L2 Consistency Distance (L2CD) in Algorithm 2, and this function is
 our implementation of the first part of Algorithm 2.

The next important function, `compute_cc_dist`, has the same first step as `compute_demo_dist`.
However, instead of computing the L2CD, it instead iterates through each of the *M x N* candidate
actions and computes the CCD. This is calculated by rematching the test action to the demo
space, and computing the distance between the **coordinates** of the original target demo
action and the **coordinates** of the rematched action. Finally, the action with minimal
CCD is returned. This function is our implementation of the second part of Algorithm 2.

In our experiments, we have 3 pretrained policies and 2 demonstrations of the test task.
Thus, *M* = 3, *N* = 2, and *M x N* = 6.

**`evaluate\_demo\_correspondence.py`**

**`main.py`**

This is the driver file from which all model trains and experiments are run. It makes all necessary
calls to classes and functions from the other files listed above (aside from `touch.py`, which needs
to be run separately to generate demonstrations). We provide commands to run each component of our
experiments below.

## To Run

### Starting the V-REP Simulation

To pretrain RL policies, collect demonstrations, and run any of our simulation experiments,
 we need to first start a CoppeliaSim instance, which will run in the background during training.

[Download CoppeliaSim](http://www.coppeliarobotics.com/index.html) (formerly V-REP) and run it to start the simulation. You may need to adjust the paths below to match your V-REP folder.

To start with the GUI on, use the below command.

```bash
cd ~/real_good_robot
~/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE -s ~/src/real_good_robot/simulation/simulation.ttt
```

To start in headless mode (no GUI), use the below command.

```bash
cd ~/real_good_robot
~/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE -s ~/src/real_good_robot/simulation/simulation.ttt -h
```

The number used in the `gREMOTEAPISERVERSERVICE` argument is the `TCP_PORT` to specify in the calls below. Note that the default value of `19997` is required for the `touch.py` script.


### Generating Demonstrations

To generate demonstrations, use the below command. Note that this requires starting the simulator as described above.

`python touch.py -t [TASK_TYPE] --save`, where `[TASK_TYPE]` is one of `[row, stack, unstack,
vertical_square]`.

This command opens two windows, a window with a depth image, and a window with a color image.
At each step, the user must specify either a grasp or place action with a keypress: `g` for
grasp, `p` for place. After specifying the action with a keypress, the user must then click on 
the **color image** at the location at which the grasp/place should be centered.

Following this mouseclick, the gripper will execute the action, and after it is finished executing,
the next action can be specified. Once a demonstration has been completed, the user has two options.
(1) Pressing the `c` key to collect another demonstration, and (2) pressing the `q` key to
exit. To reproduce our results, it is important to collect both demonstrations of the task in
one run of `touch.py` so that the filestructure of the demonstrations is consistent.

We recommend using `touch.py` without the `--save` flag set a couple times to get used to
the workflow. Collected demonstrations should be ideal for our method to function optimally,
and location of the keypress can appear to be different from the location that the gripper
comes down (this is due to the angle of the camera and depth ambiguity).


### Pretraining RL Policies

We provide commands to pretrain RL policies for each task below. Note that this requires starting the simulator as described above.

**Row**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --push_rewards --experience_replay
 --explore_rate_decay --trial_reward --common_sense --place --future_reward_discount 0.65 --tcp_port [TCP_PORT] --random_seed 1238 --max_train_actions 40000
 --random_actions --task_type row --depth_channels_history
```

**Stack**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 8 --push_rewards --experience_replay
 --explore_rate_decay --trial_reward --common_sense --check_z_height --tcp_port [TCP_PORT] --place --future_reward_discount 0.65 --random_seed 1238
 --max_train_actions 40000 --random_actions --task_type stack --depth_channels_history
```

**Unstack**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --push_rewards --experience_replay
 --explore_rate_decay --trial_reward --common_sense --place --future_reward_discount 0.65 --tcp_port [TCP_PORT] --random_seed 1238 --max_train_actions 40000
 --random_actions --task_type unstack --trial_reward
```

**Vertical Square**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --push_rewards --experience_replay
 --explore_rate_decay --trial_reward --common_sense --place --future_reward_discount 0.65 --tcp_port [TCP_PORT] --random_seed 1238 --max_train_actions 40000
 --random_actions --task_type vertical_square --depth_channels_history
```

We additionally provide pretrained model files with the code to allow running imitation experiments without the need for this time-consuming pretraining step.

### Precomputing Demo Embeddings

Once demonstrations have been generated, we recommend precomputing demo embeddings to speed up the imitation test runs.
This is not a necessary step, but it greatly increases runtime, and particularly for simulated experiments, bugs
can arise due to long periods of inactivity.

Dictionary of demo embeddings are saved in `demos/<demo_folder>/embedding/embed_dict.pickle`. Note that this
dictionary takes up significant disk space (~20 GB), so ensure that you have the requisite storage space available.

**Row**

```bash
python3 evaluate_demo_correspondence.py -e demos/PATH_TO_ROW_DEMOS -d demos/PATH_TO_ROW_DEMOS -t row --stack_snapshot_file logs/PATH_TO_STACK_TRAINED_POLICY
--vertical_square_snapshot_file base_models/PATH_TO_VERT_SQUARE_TRAINED_POLICY --unstack_snapshot_file base_models/PATH_TO_UNSTACK_TRAINED_POLICY --write_embed
 --depth_channels_history --cycle_consistency
```

**Stack**

```bash
python3 evaluate_demo_correspondence.py -e demos/PATH_TO_STACK_DEMOS -d demos/PATH_TO_STACK_DEMOS -t stack --row_snapshot_file logs/PATH_TO_ROW_TRAINED_POLICY
--vertical_square_snapshot_file base_models/PATH_TO_VERT_SQUARE_TRAINED_POLICY --unstack_snapshot_file base_models/PATH_TO_UNSTACK_TRAINED_POLICY --write_embed
 --depth_channels_history --cycle_consistency
```

**Unstacking**

```bash
python3 evaluate_demo_correspondence.py -e demos/PATH_TO_UNSTACK_DEMOS -d demos/PATH_TO_UNSTACK_DEMOS -t unstack --stack_snapshot_file logs/PATH_TO_STACK_TRAINED_POLICY
--vertical_square_snapshot_file base_models/PATH_TO_VERT_SQUARE_TRAINED_POLICY --row_snapshot_file base_models/PATH_TO_ROW_TRAINED_POLICY --write_embed
 --depth_channels_history --cycle_consistency
```

**Vertical Square**

```bash
python3 evaluate_demo_correspondence.py -e demos/PATH_TO_VERT_SQUARE_DEMOS -d demos/PATH_TO_VERT_SQUARE_DEMOS -t vertical_square
 --row_snapshot_file logs/PATH_TO_ROW_TRAINED_POLICY --stack_snapshot_file base_models/PATH_TO_STACK_TRAINED_POLICY
 --unstack_snapshot_file base_models/PATH_TO_UNSTACK_TRAINED_POLICY --write_embed --depth_channels_history --cycle_consistency
```

### Running Imitation Experiments

#### Simulation

We first provide commands to run imitation experiments in simulation. The results of these experiments are enumerated in Table 2.
Note that this requires starting the simulator as described above.

**Row**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19998 --random_seed 1238 --max_test_trials 50 --task_type row --is_testing --use_demo --demo_path PATH_TO_ROW_DEMOS --stack_snapshot_file logs/PATH_TO_STACK_POLICY --vertical_square_snapshot_file logs/PATH_TO_VERT_SQUARE_POLICY --unstack_snapshot_file logs/PATH_TO_UNSTACK_POLICY --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65
```

**Stack**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19990 --random_seed 1238 --max_test_trials 50 --task_type stack --is_testing --use_demo --demo_path PATH_TO_STACK_DEMOS --row_snapshot_file logs/PATH_TO_ROW_POLICY --vertical_square_snapshot_file logs/PATH_TO_VERT_SQUARE_POLICY --unstack_snapshot_file logs/PATH_TO_UNSTACK_POLICY --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65
```

**Unstack**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19999 --random_seed 1238 --max_test_trials 50 --task_type unstack --is_testing --use_demo --demo_path PATH_TO_UNSTACK_DEMOS --stack_snapshot_file logs/PATH_TO_STACK_POLICY --vertical_square_snapshot_file logs/PATH_TO_VERT_SQUARE_POLICY --row_snapshot_file logs/PATH_TO_ROW_POLICY --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65
```

**Vertical Square**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 20000 --random_seed 1238 --max_test_trials 50 --task_type vertical_square --is_testing --use_demo --demo_path PATH_TO_VERT_SQUARE_DEMOS --stack_snapshot_file logs/PATH_TO_STACK_POLICY --unstack_snapshot_file logs/PATH_TO_UNSTACK_POLICY --row_snapshot_file logs/PATH_TO_ROW_POLICY --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65
```

#### Real

Now we provide commands to run imitation experiments on the real robot. The results of these experiments are enumerated in Table 3.
Note that this requires a calibrated UR5 as per the instructions in the below section.

**Row**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --obj_mesh_dir objects/blocks --num_obj 4 --check_z_height --common_sense --place --random_seed 1238 --max_test_trials 10 --task_type row --is_testing --use_demo --demo_path PATH_TO_ROW_DEMOS --stack_snapshot_file logs/PATH_TO_STACK_POLICY --vertical_square_snapshot_file logs/PATH_TO_VERT_SQUARE_POLICY --unstack_snapshot_file logs/PATH_TO_UNSTACK_POLICY --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop
```

**Stack**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --obj_mesh_dir objects/blocks --num_obj 4 --num_extra_obj 4 --check_z_height --common_sense --place --random_seed 1238 --max_test_trials 10 --task_type stack --is_testing --use_demo --demo_path PATH_TO_STACK_DEMOS --row_snapshot_file logs/PATH_TO_ROW_POLICY --vertical_square_snapshot_file logs/PATH_TO_VERT_SQUARE_POLICY --unstack_snapshot_file logs/PATH_TO_UNSTACK_POLICY --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop
```

**Unstack**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --check_z_height --disable_two_step_backprop --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --random_seed 1238 --max_test_trials 10 --task_type unstack --is_testing --use_demo --demo_path PATH_TO_UNSTACK_DEMOS --stack_snapshot_file logs/PATH_TO_STACK_POLICY --vertical_square_snapshot_file logs/PATH_TO_VERT_SQUARE_POLICY --row_snapshot_file logs/PATH_TO_ROW_POLICY --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop             
```

**Vertical Square**

```bash
export CUDA_VISIBLE_DEVICES="0" && python3 main.py --check_z_height --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --random_seed 1238 --max_test_trials 10 --task_type vertical_square --is_testing --use_demo --demo_path PATH_TO_VERT_SQUARE_DEMOS --stack_snapshot_file logs/PATH_TO_STACK_POLICY --unstack_snapshot_file logs/PATH_TO_UNSTACK_POLICY --row_snapshot_file logs/PATH_TO_ROW_POLICY --grasp_only --cycle_consistency --no_common_sense_backprop
```

#### Files

We provide both demonstration directories and pretrained model files in the `experiment_files` directory. These can be used rather than
generating them yourself.

## Running on a Real UR5 with ROS Based Image Collection

### ROS Based Image Collection Setup

We require python3, so you'll need to ensure `export ROS_PYTHON_VERSION=3` is set for the build. A couple additional steps below will need to be added in the middle. We advise installing in the folder:

```
~/ros_catkin_ws
```

Follow instructions in the [ROS Melodic steps to build ros from source](http://wiki.ros.org/melodic/Installation/Source).

In particular fix up this command:

```
export ROS_PYTHON_VERSION=3 && rosinstall_generator desktop_full --rosdistro melodic --deps --tar > melodic-desktop-full.rosinstall && wstool init -j8 src melodic-desktop-full.rosinstall
```

For the primesense camera add in the [openni2_launch](https://github.com/ros-drivers/openni2_launch), and [rgbd_launch](https://github.com/ros-drivers/rgbd_launch) repositories, and for handeye calibration between the camera and robot add [UbiquityRobotics/fiducials](https://github.com/UbiquityRobotics):

```
cd ~/catkin_ros_ws
git clone https://github.com/ros-drivers/openni2_launch.git
git clone https://github.com/ros-drivers/rgbd_launch.git
git clone https://github.com/UbiquityRobotics/fiducials.git
```

Run the build and install.

```
cd ~/ros_catkin_ws
rosdep install --from-paths src --ignore-src --rosdistro melodic -y && ./src/catkin/bin/catkin_make_isolated --install
```

Source the ros setup so you get access to the launch commands:
```
source ~/ros_catkin_ws/install_isolated/setup.zsh
```

Running ROS with depth image processing:

```bash
taskset 0x00000FFF roslaunch openni2_launch openni2.launch depth_registration:=true
```

We use the [linux taskset command](https://linux.die.net/man/1/taskset) ([examples](https://www.howtoforge.com/linux-taskset-command/)) to limit ROS to utilizing 8 cores or fewer, so other cores are available for training.

In a separate tab run our small test script:

```bash
python test_ros_images.py
```

Running RVIZ to look at the images:

```
rosrun rviz rviz
```

The correct images are from the following ROS topics:

```

        self.rgb_topic = "/camera/rgb/image_rect_color"
        # raw means it is in the format provided by the openi drivers, 16 bit int
        self.depth_topic = "/camera/depth_registered/hw_registered/image_rect"
```

#### Calibrating Camera Intrincics

You must first calibrate your rgb and depth camera intrinsics and rectify your images to ensure you can accurately convert camera positions to robot poses. We do this using [camera_calibration](http://wiki.ros.org/camera_calibration) in the [ros-perception/image_pipeline](https://github.com/ros-perception/image_pipeline) library.

You will need to generate and load a calibration yaml file which goes in a location like `~/.ros/camera_info/rgb_PS1080_PrimeSense.yaml`. We have an examle from our robot in this repository saved at [real/rgb_PS1080_PrimeSense.yaml](real/rgb_PS1080_PrimeSense.yaml).

#### Calibrating Camera Extrinsics

1. Print an [ArUco Tag](http://chev.me/arucogen/), we use 70mm tags with the 4x4 design (dictionary 1), so it can fit in the gripper. Make sure the ArUco dictionary id in the launch files is correct. Attach the ArUco Tag on the robot.

2. Edit the fiducials ROS package [aruco_detect.launch](https://github.com/UbiquityRobotics/fiducials/blob/kinetic-devel/aruco_detect/launch/aruco_detect.launch) file in `~/ros_catkin_ws/src/fiducials/aruco_detect/launch/aruco_detect.launch` from the [fiducials github repository](https://github.com/UbiquityRobotics/fiducials) you cloned earlier, see [the fiducials wiki for reference](http://wiki.ros.org/fiducials). Modify the launch file in [fiducials/aruco to detect your markers and receive images from your sensor. Here is our configuration:

```

<!-- Run the aruco_detect node -->
<launch>
  <!-- namespace for camera input -->
  <!-- /camera/rgb/image_rect_color/compressed -->
  <arg name="camera" default="/camera/rgb"/>
  <arg name="image" default="image_rect_color"/>
  <arg name="transport" default="compressed"/>
  <arg name="fiducial_len" default="0.07"/>
  <arg name="dictionary" default="1"/>
  <arg name="do_pose_estimation" default="true"/>
  <arg name="ignore_fiducials" default="" />
  <arg name="fiducial_len_override" default="" />

  <node pkg="aruco_detect" name="aruco_detect"
    type="aruco_detect" output="screen" respawn="false">
    <param name="image_transport" value="$(arg transport)"/>
    <param name="publish_images" value="true" />
    <param name="fiducial_len" value="$(arg fiducial_len)"/>
    <param name="dictionary" value="$(arg dictionary)"/>
    <param name="do_pose_estimation" value="$(arg do_pose_estimation)"/>
    <param name="ignore_fiducials" value="$(arg ignore_fiducials)"/>
    <param name="fiducial_len_override" value="$(arg fiducial_len_override)"/>
    <remap from="/camera/compressed"
        to="$(arg camera)/$(arg image)/$(arg transport)"/>
    <remap from="/camera_info" to="$(arg camera)/camera_info"/>
  </node>
</launch>
```

5. You must predefine the `workspace_limits` python variables in the `calibration_ros.py`, `touch.py`, `main.py`, `robot.py`, and `demo.py`. To modify these locations, change the variables `workspace_limits` at the end of `calibrate_ros.py`. You may define it in the `Calibrate` class or in the function `collect_data` for data collection.

3. The code directly communicates with the robot via TCP. At the top of `calibrate_ros.py`, change variable `tcp_host_ip` to point to the network IP address of your UR5 robot controller.

4. Roslaunch the camera with, for example:
```shell
taskset 0x00000FFF roslaunch openni2_launch openni2.launch depth_registration:=true

We use the [linux taskset command](https://linux.die.net/man/1/taskset) ([examples](https://www.howtoforge.com/linux-taskset-command/)) to limit ROS to utilizing 8 cores or fewer, so other cores are available for training.

```

5. The script is subscribed to the rostopic `/fiducial_transform` to get the pose of the tag in the camera frame. Roslaunch aruco_detect:
```shell
taskset 0x00000FFF roslaunch aruco_detect aruco_detect.launch
```

The robot will move suddenly and rapidly. Users **must** be ready to push the **emergency stop** button at any time.

6. CAREFULLY run `python touch.py` to start the arm, it will move suddenly!

7. Center the AR tag on the gripper manually using the teach mode button on the robot.

8. Click on the title bar of the `color` image window, do not click on the general color area the robot may move suddenly!

8. Press `-` to close the gripper (`=` will open it), and check that the center of the AR tag is where you want your gripper center to be defined.

9. Press `k` to calibrate, and after going to a number of positions you should see a calibration result like the following:

```
Total number of poses: 26
Invalid poses number: 0
Robot Base to Camera:
[[ 0.1506513   0.87990966 -0.45062533  0.79319678]
 [ 0.98857761 -0.13210593  0.07254191 -0.14601768]
 [ 0.00430005 -0.45640664 -0.88976092  0.55173518]
 [ 0.          0.          0.          1.        ]]
Total number of poses: 26
Invalid poses number: 0
Tool Tip to AR Tag:
[[ 0.18341198 -0.01617267 -0.98290309  0.0050482 ]
 [ 0.03295954  0.99940367 -0.01029385  0.01899328]
 [ 0.98248344 -0.03050802  0.18383565  0.10822485]
 [ 0.          0.          0.          1.        ]]
```

Backup procedure (in place of the steps 6 and later from above):
with caution, run the following to move the robot and calibrate:

The robot will move suddenly and rapidly. Users **must** be ready to push the **emergency stop** button at any time.

```shell
python calibrate_ros.py
```

The script will record the pose of the robot and the ArUco tag in the camera frame with correspondence. Then it uses the [Park and Martin Method](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=326576) to solve the AX=XB problem for the hand-eye calibration. And the method is implemented in the `utils.py`. The script will generate a `robot_base_to_camera_pose.txt` in `real/`. This txt basically is the pose of the camera in the robot base frame.

If you already have corresponded pose file of the robot and the ArUco tag, you can also use the `calibrate()` function in the `calibrate_ros.py` to directly calculate the pose of the camera without the data collection step.

### Collecting the Background heightmap

The real robot also uses a background heightmap of the scene with no objects present.

1. Completely clear the table or working surface.
2. Back up and remove the current `real/background_heightmap.depth.png`.
3. Run pushing and grasping data collection with the `--show_heightmap` flag.
4. View the heightmap images until you see one with no holes (black spots), and save the iteration number at the top.
5. Copy the good heightmap from `logs/<run_folder>/data/depth_heightmaps/<iteration>.0.depth.png` and rename it to `real/background_heightmap.depth.png`.
6. Stop and re-run pushing and grasping with the `--show_heightmap` flag.

Here is an example of the matplotlib visualization of a good depth heightmap, there are no black specks aside from one corner which is out of the camera's field of view:

![example_background_depth_map](images/example_background_depth_map.png)

Your updated depth heightmaps should be good to go!
