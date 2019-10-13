"""
Message for communication with the robotiq 2f-85 gripper.
@author: Hongtao Wu
Oct 12, 2019
"""

from collections import namedtuple

class outputMsg(object):
    def __init__(self):
        self.rACT = 0
        self.rGTO = 0
        self.rATR = 0
        self.rPR  = 0
        self.rSP  = 0
        self.rFR  = 0

class inputMsg(object):
    def __init__(self):
        self.gACT = None
        self.gGTO = None
        self.gSTA = None
        self.gOBJ = None
        self.gFLT = None
        self.gPR  = None
        self.gPO  = None
        self.gCU  = None