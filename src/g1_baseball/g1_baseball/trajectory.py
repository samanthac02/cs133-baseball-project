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

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

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
        # Initial configuration of the full robot.
        self.q0 = np.zeros(self.N)
        self.qc = self.q0.copy()

        # Active joints in the pelvis-to-bat kinematic chain.
        # The order must match the order along the chain.
        self.activenames = [
            'waist_yaw_joint',
            'waist_roll_joint',
            'waist_pitch_joint',

            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
            'right_wrist_roll_joint',
            'right_wrist_pitch_joint',
            'right_wrist_yaw_joint',
        ]

        self.activeindex = np.array([
            self.jointindex[name] for name in self.activenames
        ])

        self.chain = KinematicChain(
            self, 'pelvis', 'baseball_bat', self.activenames
        )

        # Ready configuration.
        #
        # This pose was computed using the supplied URDF so that the
        # bat link's local z axis is approximately horizontal, while
        # the right wrist yaw starts at 90 degrees.
        #
        # Active joint order:
        # waist yaw/roll/pitch,
        # shoulder pitch/roll/yaw,
        # elbow,
        # wrist roll/pitch/yaw.
        self.qready = self.q0.copy()

        self.qready[self.activeindex] = np.radians(np.array([
            -45.000000,
            7.283509,
            8.548961,

            -88.073999,
            -19.888281,
            19.210660,
            119.492473,
            -30.540658,
            30.441813,
            90.000000,
        ]))

        # Compute the matched starting pose for the Cartesian swing.
        self.qa_ready = self.qready[self.activeindex].copy()

        self.pstart, self.Rstart, _, _ = self.chain.fkin(
            self.qa_ready
        )

        # Assumed long axis of the bat, expressed in its own frame.
        self.bat_axis = np.array([0.0, 0.0, 1.0])

        # Verify the ready pose is horizontal under that assumption.
        bat_direction = self.Rstart @ self.bat_axis

        if abs(bat_direction[2]) > 0.02:
            raise ValueError(
                "Ready pose is not horizontal for this robot description. "
                "Check the bat attachment rotation and bat-axis definition."
            )

        # Center of the horizontal swing arc, expressed in pelvis.
        # The arc rotates around the pelvis z axis at grip height.
        self.center = np.array([
            0.0, 0.0, self.pstart[2]
        ])

        self.radius = self.pstart - self.center

        # Swing parameters.
        self.swing_angle = np.radians(100.0)

        # A small downward dip in the middle of the swing.
        self.swing_drop = 0.025

        # Roll the tool orientation around the bat's own long axis.
        # Negative direction was selected for this starting pose.
        self.roll_angle = np.radians(-90.0)

        # Secondary preference for the actual wrist-roll joint.
        # This is a preference, not an enforced joint trajectory.
        self.wrist_roll_change = np.radians(30.0)

        # Damped inverse and feedback parameters.
        self.lam = 20.0
        self.gamma = 0.05
        self.lam_secondary = 2.0

        # Scale translational rows before computing the damped inverse.
        self.Wtask = np.diag([
            1/0.4, 1/0.4, 1/0.4, 1.0, 1.0, 1.0
        ])

        # Joint limits for checking the commanded configuration.
        self.qmin = np.full(self.N, -np.inf)
        self.qmax = np.full(self.N,  np.inf)

        for joint in joints:
            if joint.get('type') == 'continuous':
                continue

            i = self.jointindex[joint.get('name')]
            limit = joint.find('limit')

            self.qmin[i] = float(limit.get('lower'))
            self.qmax[i] = float(limit.get('upper'))

        for q in (self.q0, self.qready):
            if np.any(q < self.qmin) or np.any(q > self.qmax):
                raise ValueError("Initial or ready pose exceeds joint limits")

        # Phase durations.
        self.Tready = 3.0
        self.Tpause = 1.0
        self.Tswing = 6.0
        self.Tfinish = 1.0
        self.Treturn = 4.0
        self.Thold = 1.0

        # Save the actual IK endpoint before starting the joint return.
        self.qreturn = None

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
        t2 = t1 + self.Tpause
        t3 = t2 + self.Tswing
        t4 = t3 + self.Tfinish
        t5 = t4 + self.Treturn
        t6 = t5 + self.Thold

        if self.t > t6:
            if not self.future.done():
                self.future.set_result("Trajectory has ended")
            return

        if self.t < t1:
            # JOINT MOTION:
            # Move from the initial configuration to the ready pose.
            qc, qcdot = goto5(
                self.t,
                self.Tready,
                self.q0,
                self.qready
            )

        elif self.t < t2:
            # Hold the ready pose before swinging.
            qc = self.qready.copy()
            qcdot = np.zeros(self.N)

        elif self.t < t3:
            # CARTESIAN MOTION:
            # Follow the desired bat pose using damped IK.
            t = self.t - t2

            # Smooth swing progress from 0 to 1.
            s, sdot = goto5(
                t, self.Tswing, 0.0, 1.0
            )

            theta = self.swing_angle * s
            thetadot = self.swing_angle * sdot

            Rz = Rotn(
                np.array([0.0, 0.0, 1.0]),
                theta
            )

            # Desired grip position:
            # a horizontal arc with a small downward dip.
            radial = Rz @ self.radius

            pd = (
                self.center
                + radial
                + np.array([
                    0.0,
                    0.0,
                    -self.swing_drop * sin(pi*s)
                ])
            )

            vd = (
                thetadot * np.cross(
                    np.array([0.0, 0.0, 1.0]),
                    radial
                )
                + np.array([
                    0.0,
                    0.0,
                    -self.swing_drop * pi * cos(pi*s) * sdot
                ])
            )

            # Begin the rollover during the final 30% of the swing.
            troll = t - 0.70 * self.Tswing
            Troll = 0.30 * self.Tswing

            if troll <= 0.0:
                sroll = 0.0
                srolldot = 0.0
            else:
                sroll, srolldot = goto5(
                    troll, Troll, 0.0, 1.0
                )

            phi = self.roll_angle * sroll
            phidot = self.roll_angle * srolldot

            # Desired orientation:
            # sweep around pelvis z, then roll about the bat axis.
            #
            # Rotating about the bat's own long axis preserves
            # its horizontal direction.
            Rd = (
                Rz
                @ self.Rstart
                @ Rotn(self.bat_axis, phi)
            )

            wd = (
                np.array([0.0, 0.0, thetadot])
                + phidot * (
                    Rz @ self.Rstart @ self.bat_axis
                )
            )

            ##########################################################
            # INVERSE KINEMATICS

            # Current active joint configuration.
            qa = self.qc[self.activeindex].copy()

            ptip, Rtip, Jv, Jw = self.chain.fkin(qa)
            J = np.vstack((Jv, Jw))

            # Task reference velocity with feedback.
            e = np.concatenate((
                ep(pd, ptip),
                eR(Rd, Rtip)
            ))

            xr = np.concatenate((vd, wd)) + self.lam * e

            # Scaled damped inverse.
            Jbar = self.Wtask @ J
            xrbar = self.Wtask @ xr

            A = (
                Jbar @ Jbar.T
                + self.gamma**2 * np.eye(6)
            )

            Jdamp = np.linalg.solve(A, Jbar).T

            # Secondary joint preferences.
            qpreferred = self.qa_ready.copy()
            vpreferred = np.zeros(10)

            # Encourage the waist to rotate throughout the swing.
            qpreferred[0] += theta
            vpreferred[0] = thetadot

            # Encourage late motion of the wrist-roll joint.
            qpreferred[7] += self.wrist_roll_change * sroll
            vpreferred[7] = (
                self.wrist_roll_change * srolldot
            )

            qdot_secondary = (
                vpreferred
                + self.lam_secondary * (qpreferred - qa)
            )

            N = np.eye(10) - Jdamp @ Jbar

            qadot = (
                Jdamp @ xrbar
                + N @ qdot_secondary
            )

            # Integrate the active joints.
            qc = self.qc.copy()
            qc[self.activeindex] = qa + self.dt * qadot

            # All other joints remain stationary.
            qcdot = np.zeros(self.N)
            qcdot[self.activeindex] = qadot

        elif self.t < t4:
            # Save and hold the actual endpoint reached by IK.
            if self.qreturn is None:
                self.qreturn = self.qc.copy()

            qc = self.qreturn.copy()
            qcdot = np.zeros(self.N)

        elif self.t < t5:
            # JOINT MOTION:
            # Return from the actual IK endpoint to the initial pose.
            if self.qreturn is None:
                self.qreturn = self.qc.copy()

            qc, qcdot = goto5(
                self.t - t4,
                self.Treturn,
                self.qreturn,
                self.q0
            )

        else:
            # Hold the initial configuration.
            qc = self.q0.copy()
            qcdot = np.zeros(self.N)

        ##############################################################
        # CHECK AND SAVE THE COMMANDED CONFIGURATION.

        if (
            not np.all(np.isfinite(qc))
            or np.any(qc < self.qmin)
            or np.any(qc > self.qmax)
        ):
            self.get_logger().error(
                "Stopping: trajectory exceeded a joint limit "
                "or produced a nonfinite command."
            )

            if not self.future.done():
                self.future.set_result("Invalid joint command")
            return

        self.qc = qc.copy()

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
