'''plotcondition.py

   Plot the /condition recorded in the ROS2 bag.

   Usage:
     plotcondition <bagname>
   where
     <bagname> is either 'latest' or the name of a bag

'''

import rclpy
import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

from rosbag2_py                 import SequentialReader
from rclpy.serialization        import deserialize_message

from std_msgs.msg               import Float64


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
#  Plot the Condition Data
#
def plotcondition(condmsgs, t0, bagname):
    # Process the condition messages.
    condition = np.array([msg.data for msg in condmsgs])

    # Set up time, assuming a 100Hz rate.
    N  = len(condition)
    dt = 0.01
    t  = dt * np.linspace(0, N, N, endpoint=False)

    # Create a figure to plot condition vs. t
    fig, ax = plt.subplots(1, 1)

    # Plot the data.
    ax.plot(t, condition)
    ax.set(ylabel='Condition Number')
    ax.set(xlabel='Time (sec)')
    ax.set(title="Condition Number in '%s'" % bagname)
    ax.grid()


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

    condmsgs = reader.msgs('/condition', Float64)

    reader.close()

    # Process the condition number.
    if condmsgs:
        print("Plotting condition data...")
        plotcondition(condmsgs, t0, bagname)
    else:
        raise ValueError("No condition data!")

    # Show
    plt.show()


#
#   Run the main code.
#
if __name__ == "__main__":
    main()
