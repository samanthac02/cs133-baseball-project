## Progress
1. Downloaded the official G1 model (located in src/g1_baseball/orig_unitree) from https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description. Moved all the STL files from this repo into src/g1_baseball/meshes. Moved the URDF file into src/g1_baseball/urdf.

2. CADed a baseball bat in Fusion and exported it as STL into src/g1_baseball/meshes directory

3. Connected the bat to the G1's right hand in the URDF (with a small offset that I determined by visually looking at the viewer)

## Build
cd ~/robotws/cs133-baseball-project
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch utils viewurdf.launch.py urdf:="$PWD/src/g1_baseball/urdf/g1_baseball.urdf"

