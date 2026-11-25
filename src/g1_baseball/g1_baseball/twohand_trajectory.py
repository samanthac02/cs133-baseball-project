'''
twohand_trajectory.py

Two-handed swing

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
        packagepath = Path(get_package_share_directory('g1_baseball'))

        self.declare_parameter('urdf_file', str(packagepath / 'urdf' / 'g1_baseball.urdf'))

        urdf_file = self.get_parameter('urdf_file').value
        robot = ET.parse(urdf_file).getroot()

        # Define the list of movable joints from the URDF.
        joints = [
            joint for joint in robot.findall('joint')
            if joint.get('type') in ('revolute', 'continuous', 'prismatic')
        ]

        self.jointnames = [joint.get('name') for joint in joints]
        self.N = len(self.jointnames)

        # Map a joint name to its index in the joint vector.
        self.jointindex = {name: i for i, name in enumerate(self.jointnames)}

        self.contact_distance = 0.50    # Location of ball contact on the bat
        self.slidername = 'bat_tip_joint'
        self.sliderindex = self.jointindex[self.slidername]

        self.grip_spacing = 0.08 # distance from right hand to left hand on bat
        self.left_offset = np.array([0.0, 0.0, -self.contact_distance - self.grip_spacing])

        waistnames = [
            'waist_yaw_joint',
            'waist_roll_joint',
            'waist_pitch_joint',
        ]

        suffixes = [
            'shoulder_pitch_joint',
            'shoulder_roll_joint',
            'shoulder_yaw_joint',
            'elbow_joint',
            'wrist_roll_joint',
            'wrist_pitch_joint',
            'wrist_yaw_joint',
        ]

        rightnames = ['right_' + s for s in suffixes]
        leftnames = ['left_' + s for s in suffixes]

        # Active joints in the pelvis-to-bat kinematic chain.
        self.activenames = waistnames + rightnames + leftnames

        self.activeindex = np.array([
            self.jointindex[name] for name in self.activenames
        ])

        self.rightcolumns = np.arange(10)
        self.leftcolumns = np.array([0, 1, 2, 10, 11, 12, 13, 14, 15, 16])

        self.rightchain = KinematicChain(
            self, 'pelvis', 'bat_tip', waistnames + rightnames + [self.slidername]
        )

        self.leftchain = KinematicChain(
            self, 'pelvis', 'left_grip', waistnames + leftnames + [self.slidername]
        )

        # Define the initial joint configuration.
        # Initial configuration of the full robot.
        self.q0 = np.zeros(self.N)
        self.q0[self.sliderindex] = self.contact_distance
        self.qc = self.q0.copy()


        # Ready configuration.
        #
        #
        # Active joint order:
        # waist yaw/roll/pitch,
        # shoulder pitch/roll/yaw,
        # elbow,
        # wrist roll/pitch/yaw.
        self.qready = self.q0.copy()

        # Initial position before swing (CHANGE)
        self.qready[self.activeindex] = np.radians(np.array([
            -30.000000,
            0.000000,
            0.000000,

            -18.241866,
            47.989218,
            -12.668237,
            55.614555,
            37.844100,
            -9.818378,
            27.550576,

            12.978320,
            24.388031,
            -30.134193,
            -25.957338,
            79.698408,
            62.724505,
            -4.417448,
        ]))

        # Compute the matched starting pose for the Cartesian swing.
        self.qa_ready = self.qready[self.activeindex].copy()

        pR, RR, _, pL, RL, _ = self.both_fkin(self.qa_ready)

        self.pstart = pR.copy()
        self.Rstart = RR.copy()

        # Grip Debugger
        grip_error = np.linalg.norm(pL - (pR + RR @ self.left_offset))
        rotation_error = np.linalg.norm(eR(RR, RL))
        if grip_error > 0.002 or rotation_error > 0.01:
            raise ValueError('Something wrong with grip - check URDF and start pose')


        # Phase durations.
        self.Tready = 4.0
        self.Tcontact = 3.0
        self.Tfollow = 3.0
        self.Thold = 1.0

        # Sweep angles to choose initial endpoint pose
        self.theta_contact = np.radians(40.0)
        self.theta_finish = np.radians(70.0)

        # Center of the horizontal swing arc
        center = np.array([0.0, 0.0, self.pstart[2]])
        radius = self.pstart - center

        # initial contact and finish positions
        self.pcontact = center + Rotz(self.theta_contact) @ radius       # CHANGEABLE PARAMETER
        self.pfinish = center + Rotz(self.theta_finish) @ radius

        self.contact_speed = 0.20

        # Direction should be tangent to horizontal reference arc
        direction = np.cross(np.array([0, 0, 1]), self.pcontact - center)
        direction_norm = np.linalg.norm(direction)

        # Check for 0 radius
        if direction_norm < 1e-6: 
            raise ValueError("Can't define contact direction with 0 radius")

        self.vcontact = self.contact_speed * direction / direction_norm     # CHANGEABLE PARAMETER

        # orientation sweep velocity at contact
        self.omega_contact = self.contact_speed / np.linalg.norm(radius)


        # IK parameters.
        self.lam = 20.0
        self.gamma = 0.02

        self.Wtask = np.diag( 
            [2.5, 2.5, 2.5, 1.0, 1.0, 1.0] + [7.5, 7.5, 7.5, 3.0, 3.0, 3.0]  # CHECK
            )


        # Joint velocity costs (follows same order as active joints): 
        # Want to discourage leaning and bending and prefer arm motion 
        self.joint_cost = np.array([
            [1.0, 100.0, 100.0] + [1.0] * 14
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

    def both_fkin(self, qa): 
        '''
        Helper function
        The contact point on the bat is modeled as a prismatic joint, however we do not want 
        that joint to be moved during inverse kinematics so it must be removed from Jacobian
        '''
        # Right arm chain
        qr = np.concatenate((
            qa[self.rightcolumns],
            np.array([self.contact_distance])
        ))
        pR, RR, JvR, JwR = self.rightchain.fkin(qr)


        # Left arm chain
        ql = np.concatenate((
            qa[self.leftcolumns]
        ))
        pL, RL, JvL, JwL = self.leftchain.fkin(ql)

        JR = np.zeros((6, 17))
        JL = np.zeros((6, 17))

        # Remove slider column from right chain
        JR[:, self.rightcolumns] = np.vstack((JvR[:, :10], JwR[:, :10]))

        JL[:, self.leftcolumns] = np.vstack((JvL, JwL))
        

        return pR, RR, JR, pL, RL, JL 

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

            else: 
                # Phase 3 (contact --> follow through)
                tfollow = t - self.Tcontact

                pd, vd = spline5(tfollow, self.Tfollow, self.pcontact, self.pfinish, self.vcontact, np.zeros(3), np.zeros(3), np.zeros(3))
                theta, thetadot = spline5(tfollow, self.Tfollow, self.theta_contact, self.theta_finish, self.omega_contact, 0.0, 0.0, 0.0)

            theta = float(theta)
            thetadot = float(thetadot)

            Rd = Rotz(theta) @ self.Rstart
            wd = np.array([0.0, 0.0, thetadot])

            # Left grip relative to right
            offset = Rd @ self.left_offset
            pLd = pd + offset
            RLd = Rd
            vLd = vd + np.cross(wd, offset)
            wLd = wd

            ##########################################################
            # INVERSE KINEMATICS

            # Current active joint configuration.
            qa = self.qc[self.activeindex].copy()

            pR, RR, JR, pL, RL, JL = self.both_fkin(qa)  # Using defined helper function

            # Two pose errors
            e = np.concatenate((
                ep(pd, pR),
                eR(Rd, RR), 
                eP(pLd, pL),
                eR(RLd, RL)
            ))

            xr = np.concatenate((vd, wd, vLd, wLd)) + self.lam * e

            # Combined jacobian
            J = np.vstac((JR, JL))

            # Scaled damped inverse.
            Jbar = self.Wtask @ J
            xrbar = self.Wtask @ xr

            A = Jbar @ Jbar.T + self.gamma**2 * np.eye(12)

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
