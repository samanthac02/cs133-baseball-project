'''plotjoints.py

   Plot the /joint_states recorded in the ROS2 bag.

   Usage:
     plotjoints <bagname> <joints>
   where
     <bagname> is either 'latest' or the name of a bag
     <joints>  is either 'all' or joint numbers or joint names

'''

import rclpy
import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

from rosbag2_py                 import SequentialReader
from rclpy.serialization        import deserialize_message

from sensor_msgs.msg            import JointState


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
#  Plot the Joint Data
#
def plotjoints(jointmsgs, t0, bagname, jointnames=['all']):
    # Process the joint messages.
    names = jointmsgs[0].name

    sec  = np.array([msg.header.stamp.sec     for msg in jointmsgs])
    nano = np.array([msg.header.stamp.nanosec for msg in jointmsgs])
    t = sec + nano*1e-9 - t0

    pos = np.array([msg.position for msg in jointmsgs])
    vel = np.array([msg.velocity for msg in jointmsgs])

    # Grab/check the dimensions.
    Nnames = len(names)
    Npos   = pos.shape[1]
    Nvel   = vel.shape[1]
    if Npos != 0 and Npos != Nnames:
        raise ValueError("Position data does not match %d joints" % Nnames)
    if Nvel != 0 and Nvel != Nnames:
        raise ValueError("Velocity data does not match %d joints" % Nnames)
    
    # Extract the specified joint.
    if jointnames[0] != 'all':
        # Loop over all jointnames
        indices = []
        for jointname in jointnames:
            # Grab the joint index/name.
            try:
                index = int(jointname)
                try:
                    jointname = names[index]
                except Exception:
                    raise ValueError("Joint %d out of range 0...%d" %
                                     (index, Nnames))
            except Exception:
                try:
                    index = names.index(jointname)
                except Exception:
                    raise ValueError("Joint '%s' not in known joints %s" %
                                     (jointname, str(names)))

            # Append of the indices:
            indices.append(index)

        # Limit the data.
        names = [names[index] for index in indices]
        i     = np.array(indices)
        pos   = pos[:,i[i<Npos]]
        vel   = vel[:,i[i<Nvel]]

    # Re-zero time.
    tstart = min(t)
    print("Starting at time ", tstart)
    t = t - tstart


    # Create a figure to plot pos and vel vs. t
    fig, axs = plt.subplots(2, 1)

    # Plot the data in the subplots.
    axs[0].plot(t, pos)
    axs[0].set(ylabel='Position (rad)')
    axs[1].plot(t, vel)
    axs[1].set(ylabel='Velocity (rad/sec)')

    # Connect the time.
    axs[1].set(xlabel='Time (sec)')
    axs[1].sharex(axs[0])

    # Add the title and legend.
    axs[0].set(title="Joint Data in '%s'" % bagname)
    axs[0].legend(names)

    # Draw grid lines and allow only "outside" ticks/labels in each subplot.
    for ax in axs.flat:
        ax.grid()
        ax.label_outer()


######################################################################
# Main Code:
#
def main():
    # Grab the arguments.
    bagname    = 'latest' if len(sys.argv) < 2 else sys.argv[1]
    jointnames = ['all']  if len(sys.argv) < 3 else sys.argv[2:]

    # Check for the latest ROS bag:
    if bagname == 'latest':
        bagname = latestBag()

    # Report.
    print("Reading ROS bag:   " + str(bagname))
    print("Processing joints: " + str(jointnames))

    # Set up the BAG reader, grab the time and messages, then close.
    reader = BagReader(bagname)
    t0     = reader.t0() - 0.01

    jointmsgs = reader.msgs('/joint_states', JointState)

    reader.close()

    # Process the joints
    if jointmsgs:
        print("Plotting joint data...")
        plotjoints(jointmsgs, t0, bagname, jointnames)
    else:
        raise ValueError("No joint data!")

    # Show
    plt.show()


#
#   Run the main code.
#
if __name__ == "__main__":
    main()
