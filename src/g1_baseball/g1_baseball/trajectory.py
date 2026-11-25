'''
trajectory.py

One-handed swing

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
        joints = [
            joint for joint in robot.findall('joint')
            if joint.get('type') in ('revolute', 'continuous', 'prismatic')
        ]

        self.jointnames = [
            joint.get('name') for joint in joints
        ]

        self.N = len(self.jointnames)

        # Map a joint name to its index in the joint vector.
        self.jointindex = {
            name: i for i, name in enumerate(self.jointnames)
        }

        self.contact_distance = 0.50    # Location of ball contact on the bat
        self.slidername = 'bat_tip_joint'
        self.sliderindex = self.jointindex[self.slidername]

        # Define the initial joint configuration.
        # Initial configuration of the full robot.
        self.q0 = np.zeros(self.N)
        self.q0[self.sliderindex] = self.contact_distance
        self.qc = self.q0.copy()

        # Active joints in the pelvis-to-bat kinematic chain.
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
            self, 'pelvis', 'bat_tip', self.activenames + [self.slidername]
        )

        # Ready configuration.
        #
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

            0,
            0, 
            0, 
            0,
            0, 
            0,
            0.000000,
        ]))

        # Compute the matched starting pose for the Cartesian swing.
        self.qa_ready = self.qready[self.activeindex].copy()

        self.pstart, self.Rstart, _, _ = self.contact_fkin(
            self.qa_ready
        )

        # Defining desired bat orientation at contact (want bat to be horizontal)
        qa_contact_reference = self.qa_ready.copy()
        qa_contact_reference[7] = np.radians(90.0)
        p_reference, R_reference, _, _ = self.contact_fkin(qa_contact_reference)
        _, _, _, Jw_ready = self.contact_fkin(self.qa_ready)    
        self.roll_axis_local = self.Rstart.T @ Jw_ready[:, 7]
        self.roll_axis_local /= np.linalg.norm(self.roll_axis_local)
        self.contact_roll_angle = np.radians(90.0) - self.qa_ready[7]

        # Phase durations.
        self.Tready = 3.0
        self.Tcontact = 3.0
        self.Tfollow = 3.0
        self.Thold = 1.0

        # Sweep angles to choose initial endpoint pose
        self.theta_contact = np.radians(70.0)
        self.theta_finish = np.radians(150.0)

        # Center of the horizontal swing arc
        center = np.array([0.0, 0.0, self.pstart[2]])
        radius = p_reference - center

        # initial contact and finish positions
        self.pcontact = np.array([0.757, -0.445, -0.012]) # center + Rc @ radius       # CHANGEABLE PARAMETER
        self.pfinish = center + Rotz(self.theta_finish) @ radius

        self.contact_speed = 0.20

        # Direction should be tangent to horizontal reference arc
        direction = np.cross(np.array([0, 0, 1]), self.pcontact - center)
        direction_norm = np.linalg.norm(direction)

        # Check for 0 radius
        if direction_norm < 1e-6: 
            raise ValueError("Can't define contact direction with 0 radius")

        self.contact_direction = direction / direction_norm
        self.vcontact = self.contact_speed * self.contact_direction     # CHANGEABLE PARAMETER

        # orientation sweep velocity at contact
        self.omega_contact = self.contact_speed / np.linalg.norm(radius)


        # IK parameters.
        self.lam = 20.0
        self.gamma = 0.05
        # self.lam_secondary = 2.0

        self.Wtask = np.diag([ 
            1/0.4, 1/0.4, 1/0.4, 1.0, 1.0, 1.0  # CHECK
        ])

        # # Secondary task - CHECK
        # self.wrist_yaw_relief = np.radians(10.0)

        # Joint velocity costs (follows same order as active joints): 
        # Want to discourage leaning and bending
        self.joint_cost = np.array([
            1.0, 
            100.0, 
            100.0, 

            1.0, 
            1.0, 
            1.0, 
            1.0,
            0.25, 
            1.0, 
            1.0
        ])
        
        self.mobility = np.diag (1.0 / self.joint_cost)

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

        # Ensure all joints are within limits
        for q in (self.q0, self.qready):
            if np.any(q < self.qmin) or np.any(q > self.qmax):
                raise ValueError("Initial or ready pose exceeds joint limits")


        self.qfinish = None

        # log info about chosen contact point and velo
        self.get_logger().info(
            "Contact point [pelvis, m]: "
            + np.array2string(self.pcontact, precision=3)
        )
        
        self.get_logger().info(
            "Contact velocity [pelvis, m/s]: "
            + np.array2string(self.vcontact, precision=3)
        )

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

        # Set up the timer to update at 100 Hz, with (t=0)) occuring in
        # the first update cycle (dt) from now.
        self.dt = 0.01                      # 100Hz
        self.t = -self.dt                   # Seconds since start
        self.now = self.get_clock().now()   # ROS time since 1970
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

    def contact_fkin(self, qa): 
        '''
        Helper function
        The contact point on the bat is modeled as a prismatic joint, however we do not want 
        that joint to be moved during inverse kinematics so it must be removed from Jacobian
        '''
        qchain = np.concatenate((qa, np.array([self.contact_distance])))
        p, R, Jv, Jw = self.chain.fkin(qchain)

        return p, R, Jv[:, :10], Jw[:, :10] # Only keep the 10 active waist/arm joints in Jacobian

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
        t2 = t1 + self.Tcontact
        t3 = t2 + self.Tfollow
        t4 = t3 + self.Thold

        if self.t > t4:
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

        # In case we want a pause 
        # elif self.t < t2:
        #     # Hold the ready pose before swinging.
        #     qc = self.qready.copy()
        #     qcdot = np.zeros(self.N)

        elif self.t < t3:
            # Phase 2 and 3: cartesian swing
            t = self.t - t1

            if t < self.Tcontact: 
                # Phase 2 (ready --> contact)
                pd, vd = spline5(t, self.Tcontact, self.pstart, self.pcontact, np.zeros(3), self.vcontact, np.zeros(3), np.zeros(3))
                theta, thetadot = spline5(t, self.Tcontact, 0.0, self.theta_contact, 0.0, self.omega_contact, 0.0, 0.0)

                beta, betadot = goto5(t, self.Tcontact, 0.0, self.contact_roll_angle)
            else: 
                # Phase 3 (contact --> follow through)
                tfollow = t - self.Tcontact

                pd, vd = spline5(tfollow, self.Tfollow, self.pcontact, self.pfinish, self.vcontact, np.zeros(3), np.zeros(3), np.zeros(3))
                theta, thetadot = spline5(tfollow, self.Tfollow, self.theta_contact, self.theta_finish, self.omega_contact, 0.0, 0.0, 0.0)

                # Keep tilt during follow through
                beta = self.contact_roll_angle
                betadot = 0.0

            theta = float(theta)
            thetadot = float(thetadot)

            Rd = Rotz(theta) @ self.Rstart @ Rotn(self.roll_axis_local, beta)
            wd = np.array([0.0, 0.0, thetadot]) + betadot * (Rotz(theta) @ self.Rstart @ self.roll_axis_local)

            ##########################################################
            # INVERSE KINEMATICS

            # Current active joint configuration.
            qa = self.qc[self.activeindex].copy()

            ptip, Rtip, Jv, Jw = self.contact_fkin(qa)  # Using defined helper function
            J = np.vstack((Jv, Jw))

            # Task reference velocity with feedback.
            e = np.concatenate((ep(pd, ptip), eR(Rd, Rtip)))

            xr = np.concatenate((vd, wd)) + self.lam * e

            # Scaled damped inverse.
            Jbar = self.Wtask @ J
            xrbar = self.Wtask @ xr

            A = Jbar @ Jbar.T + self.gamma**2 * np.eye(6)

            Jdamp = np.linalg.solve(A, Jbar @ self.mobility).T

            # SEE BELOW FOR WHEN WE NEED SECONDARY TASKS
            # # Secondary joint preferences.
            # qpreferred = self.qa_ready.copy()
            # vpreferred = np.zeros(10)

            # # Encourage the waist to rotate throughout the swing.
            # qpreferred[0] += theta
            # vpreferred[0] = thetadot

            # # Move wrist yaw slightly away from its lower limit 
            # fraction = theta / self.theta_finish
            # fractiondot = thetadot / self.theta_finish

            # # Encourage late motion of the wrist-roll joint.
            # qpreferred[9] += self.wrist_yaw_relief * fraction
            # vpreferred[9] = self.wrist_yaw_relief * fractiondot

            # qdot_secondary = vpreferred + self.lam_secondary * (qpreferred - qa)

            # N = np.eye(10) - Jdamp @ Jbar

            # qadot = Jdamp @ xrbar + N @ qdot_secondary


            qadot = Jdamp @ xrbar

            # Integrate the active joints.
            qc = self.qc.copy()
            qc[self.activeindex] = qa + self.dt * qadot

            # All other joints remain stationary.
            qcdot = np.zeros(self.N)
            qcdot[self.activeindex] = qadot

        else:
            # Hold endpoint reached by inverse kinematics for now
            if self.qfinish is None: 
                self.qfinish = self.qc.copy()

            qc = self.qfinish.copy()
            qcdot = np.zeros(self.N)

        # Keep the bat contact position fixed
        qc[self.sliderindex] = self.contact_distance
        qcdot[self.sliderindex] = 0.0

        # Make sure commanded config is valid - catches edge cases

        if not np.all(np.isfinite(qc)) or np.any(qc < self.qmin) or np.any(qc > self.qmax):
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
