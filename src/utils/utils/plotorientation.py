'''plotorientation.py

   Plot the orientation of /pose /twist recorded in the ROS2 bag.

   Usage:
     plotorientation <bagname>
   where
     <bagname> is either 'latest' or the name of a bag

'''

import rclpy
import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

from rosbag2_py                 import SequentialReader
from rclpy.serialization        import deserialize_message

from geometry_msgs.msg          import PoseStamped, TwistStamped


######################################################################
# Utilities:
#
#  Grab the latest ROS Bag name.  If none, raise a FileNotFoundError.
#
def latestBag():
    # Report.
    print("Looking for latest ROS bag...")
        
    # Look at all bags, making sure we have at least one!
    dbfiles = glob.glob('*/metadata.yaml')
    if not dbfiles:
        raise FileNotFoundError('Unable to find a ROS2 bag')

    # Grab the modification times and the index of the newest.
    dbtimes = [os.path.getmtime(dbfile) for dbfile in dbfiles]
    i = dbtimes.index(max(dbtimes))

    # Select and return the newest.
    return os.path.dirname(dbfiles[i])


#
#   ROS BagReader Class
#
#   reader = BagReader(bagname, verbose=True)
#   t0   = reader.t0()
#   msgs = reader.msgs(topicname, topictype)
#          reader.close()
#
#   The methods: give the start time (in ROS seconds), give a list of
#   messages of the specified topic/type, and close the reader.
#
class BagReader:
    def __init__(self, bagname, verbose=True):
        # Set up the BAG reader.
        self.reader = SequentialReader()
        try:
            self.reader.open_uri(bagname)
        except Exception as e:
            print("Unable to read the ROS bag '%s'!" % bagname)
            print("Does it exist and WAS THE RECORDING Ctrl-c KILLED?")
            raise OSError("Error reading bag - did recording end?") from None

        # Report the contained topics and types:
        if verbose:
            print("The bag contains messages for:")
            for x in self.reader.get_all_topics_and_types():
                print("  topic %-20s of type %s" % (x.name, x.type))

    def close(self):
        # Close the reader's file.
        self.reader.close()

    def t0(self):
        # Get the starting time.
        return self.reader.get_metadata().starting_time.nanoseconds * 1e-9

    def msgs(self, topicname, topictype):
        # Restart the reader at the beginning of the file.
        self.reader.seek(0)

        # Pull out the relevant messages.
        msglist = []
        while self.reader.has_next():
            # Grab a message.
            (topic, rawdata, timestamp) = self.reader.read_next()

            # Pull out the deserialized message.
            if topic == topicname:
                msglist.append(deserialize_message(rawdata, topictype))

        # Return the list.
        return msglist


######################################################################
# Data Plotting:
#
#  Plot the Orientation Data
#
def plotorientation(posemsgs, twistmsgs, t0, bagname):
    # Process the pose messages.
    sec  = np.array([msg.header.stamp.sec     for msg in posemsgs])
    nano = np.array([msg.header.stamp.nanosec for msg in posemsgs])
    tp = sec + nano*1e-9 - t0

    q = np.array([[msg.pose.orientation.x for msg in posemsgs],
                  [msg.pose.orientation.y for msg in posemsgs],
                  [msg.pose.orientation.z for msg in posemsgs],
                  [msg.pose.orientation.w for msg in posemsgs]]).T

    # Unwrap the quaternion equivalent solutions.
    for i in range(1, q.shape[0]):
        if np.linalg.norm(q[i,:] - q[i-1,:]) > 1.5:
            q[i,:] *= -1

    # Process the twist messages.
    sec  = np.array([msg.header.stamp.sec     for msg in twistmsgs])
    nano = np.array([msg.header.stamp.nanosec for msg in twistmsgs])
    tv = sec + nano*1e-9 - t0

    w = np.array([[msg.twist.angular.x for msg in twistmsgs],
                  [msg.twist.angular.y for msg in twistmsgs],
                  [msg.twist.angular.z for msg in twistmsgs]]).T

    # Re-zero time.
    tstart = min((min(tp), min(tv)))
    print("Starting at time ", tstart)
    tp = tp - tstart
    tv = tv - tstart


    # Create a figure to plot orientation.
    fig, axs = plt.subplots(2, 1)

    # Plot the data in the subplots.
    axs[0].plot(tp, q)
    axs[0].set(ylabel='Quaternion')
    axs[1].plot(tv, w)
    axs[1].set(ylabel='Angular Velocity (rad/sec)')

    # Connect the time.
    axs[1].set(xlabel='Time (sec)')
    axs[1].sharex(axs[0])

    # Add the title and legend.
    axs[0].set(title="Task Orientation in '%s'" % bagname)
    axs[0].legend(['x', 'y', 'z', 'w'])
    axs[1].legend(['x', 'y', 'z'])

    # Draw grid lines and allow only "outside" ticks/labels in each subplot.
    for ax in axs.flat:
        ax.grid()
        ax.label_outer()


######################################################################
# Main Code:
#
def main():
    # Grab the arguments.
    bagname = 'latest' if len(sys.argv) < 2 else sys.argv[1]

    # Check for the latest ROS bag:
    if bagname == 'latest':
        bagname = latestBag()

    # Report.
    print("Reading ROS bag '%s'" % bagname)

    # Set up the BAG reader, grab the time and messages, then close.
    reader = BagReader(bagname)
    t0     = reader.t0() - 0.01

    posemsgs  = reader.msgs('/pose', PoseStamped)
    twistmsgs = reader.msgs('/twist', TwistStamped)

    reader.close()

    # Process the task
    if posemsgs and twistmsgs:
        print("Plotting orientation data...")
        plotorientation(posemsgs, twistmsgs, t0, bagname)
    elif posemsgs or twistmsgs:
        raise ValueError("Requiring both twist/pose data!")
    else:
        raise ValueError("No orientation data!")

    # Show
    plt.show()


#
#   Run the main code.
#
if __name__ == "__main__":
    main()
