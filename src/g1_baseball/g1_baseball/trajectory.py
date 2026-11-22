'''
trajectory.py

First trajectory generator for the G1 baseball project.
Moves from an initial configuration to a ready pose, performs
a slow demonstration sweep, and returns.

Node:    /trajectory
Publish: /joint_states    sensor_msgs/JointState

This publishes a kinematic animation for RVIZ.
'''

import rclpy
import numpy as np

from asyncio import Future
from pathlib import Path
import xml.etree.ElementTree as ET

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from ament_index_python.packages import get_package_share_directory
from hw5code.KinematicChain import KinematicChain
from utils.TransformHelpers import *
from utils.TrajectoryUtils import *


#
# Trajectory Utility
#
# Same calling pattern as the homework goto().
# This version uses a quintic spline with zero initial/final
# velocity and acceleration. Works with scalars or NumPy arrays.
#
def goto(t, T, p0, pf):
    u = np.clip(t / T, 0.0, 1.0)

    s = 10*u**3 - 15*u**4 + 6*u**5
    sdot = (30*u**2 - 60*u**3 + 30*u**4) / T

    p = p0 + (pf - p0) * s
    v = (pf - p0) * sdot

    return (p, v)


#
# Trajectory Generator Node Class
#
# This inherits all the standard ROS node stuff, but adds an
# update() method to be called regularly by an internal timer
# and a shutdown() method to stop the timer.
#
# Arguments are the node name and a future object.
#
class TrajectoryNode(Node):

    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object.
        super().__init__(name)
        self.future = future

        ##############################################################
        # INITIALIZE YOUR TRAJECTORY DATA!

        # Find the installed package and robot description.
        packagepath = Path(
            get_package_share_directory('g1_baseball')
        )

        self.declare_parameter(
            'urdf_file',
            str(packagepath / 'urdf' / 'g1_baseball.urdf')
        )

        urdf_file = self.get_parameter('urdf_file').value
        robot = ET.parse(urdf_file).getroot()

        # Define the list of movable joints from the URDF.
        # Publish ALL joints, including the stationary legs/left arm.
        joints = [
            joint for joint in robot.findall('joint')
            if joint.get('type') in ('revolute', 'continuous')
        ]

        self.jointnames = [
            joint.get('name') for joint in joints
        ]

        self.N = len(self.jointnames)

        # Map a joint name to its index in the joint vector.
        self.jointindex = {
            name: i for i, name in enumerate(self.jointnames)
        }

        # Define the initial joint configuration.
        self.q0 = np.zeros(self.N)

        # Define the ready configuration.
        # Angles below are in DEGREES, converted to radians.
        self.qready = self.q0.copy()

        ready_angles = {
            'waist_yaw_joint':            -15.0,
            'right_shoulder_pitch_joint': -30.0,
            'right_shoulder_roll_joint':  -25.0,
            'right_shoulder_yaw_joint':   -15.0,
            'right_elbow_joint':           60.0,
        }

        for jointname, angle in ready_angles.items():
            i = self.jointindex[jointname]
            self.qready[i] = np.radians(angle)

        # Define the follow-through configuration.
        self.qfollow = self.qready.copy()

        follow_angles = {
            'waist_yaw_joint':             15.0,
            'right_shoulder_yaw_joint':    15.0,
            'right_elbow_joint':           35.0,
        }

        for jointname, angle in follow_angles.items():
            i = self.jointindex[jointname]
            self.qfollow[i] = np.radians(angle)

        # Check the specified poses against the URDF joint limits.
        for joint in joints:
            if joint.get('type') == 'continuous':
                continue

            i = self.jointindex[joint.get('name')]
            limit = joint.find('limit')

            lower = float(limit.get('lower'))
            upper = float(limit.get('upper'))

            for q in (self.q0, self.qready, self.qfollow):
                if not lower <= q[i] <= upper:
                    raise ValueError(
                        f"Pose exceeds limits for {joint.get('name')}"
                    )

        # Define the duration of each phase.
        self.Tready = 3.0
        self.Tswing = 4.0
        self.Treturn = 3.0
        self.Thold = 1.0

        ##############################################################
        # SETUP THE LOGISTICS OF THE NODE.

        # Add the publisher to send joint commands.
        self.pubjoint = self.create_publisher(
            JointState, '/joint_states', 10
        )

        # Wait for a subscriber inside update(), so initialization
        # does not block ROS callbacks.
        self.started = False

        self.get_logger().info(
            "Waiting for a /joint_states subscriber..."
        )

        # Set up the timer to update at 100 Hz.
        self.dt = 0.01
        self.t = -self.dt

        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info(
            "Running with dt of %f seconds (%fHz)"
            % (self.dt, 1/self.dt)
        )

    # Shutdown.
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.cancel()
        self.destroy_timer(self.timer)
        self.destroy_node()

    # Update - send a new joint command every time step.
    def update(self):
        # Do not advance the trajectory until the viewer is ready.
        if not self.started:
            if not self.count_subscribers('/joint_states'):
                return

            self.started = True
            self.get_logger().info("Starting trajectory")

        # Increment trajectory time explicitly to avoid system jitter.
        self.t = self.t + self.dt

        # Use the current ROS time for the message timestamp.
        self.now = self.get_clock().now()

        ##############################################################
        # COMPUTE THE TRAJECTORY AT THIS TIME INSTANCE.

        # Define the phase boundaries.
        t1 = self.Tready
        t2 = t1 + self.Tswing
        t3 = t2 + self.Treturn
        t4 = t3 + self.Thold

        # Stop after the final hold.
        if self.t > t4:
            if not self.future.done():
                self.future.set_result("Trajectory has ended")
            return

        # Decide which phase we are in.
        if self.t < t1:
            # Move from the initial pose to the ready pose.
            qc, qcdot = goto(
                self.t,
                self.Tready,
                self.q0,
                self.qready
            )

        elif self.t < t2:
            # Perform the demonstration sweep.
            qc, qcdot = goto(
                self.t - t1,
                self.Tswing,
                self.qready,
                self.qfollow
            )

        elif self.t < t3:
            # Return to the initial pose.
            qc, qcdot = goto(
                self.t - t2,
                self.Treturn,
                self.qfollow,
                self.q0
            )

        else:
            # Hold the final configuration.
            qc = self.q0.copy()
            qcdot = np.zeros(self.N)

        ##############################################################
        # FINISH BY PUBLISHING THE DATA.

        header = Header(stamp=self.now.to_msg())

        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=qc.tolist(),
            velocity=qcdot.tolist()
        ))


#
# Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    trajectory = None

    try:
        # Initialize the trajectory generator node.
        trajectory = TrajectoryNode('trajectory', future)

        # Keep running until interrupted or the trajectory completes.
        rclpy.spin_until_future_complete(trajectory, future)

        # Report the reason for shutting down.
        if future.done():
            trajectory.get_logger().info(
                "Stopping: " + future.result()
            )
        else:
            trajectory.get_logger().info(
                "Stopping: Interrupted"
            )

    except KeyboardInterrupt:
        pass

    finally:
        # Shutdown the node and ROS.
        if trajectory is not None:
            trajectory.shutdown()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
