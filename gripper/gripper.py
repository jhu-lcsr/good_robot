#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:33:20 2018
Driver to control robotiq gripper via python
@author: Benoit CASTETS

Dependencies:
*************
MinimalModbus: https://pypi.org/project/MinimalModbus/
"""
#Libraries importation
import minimalmodbus as mm
import time
import binascii

#Communication setup
mm.BAUDRATE=115200
mm.BYTESIZE=8
mm.PARITY="N"
mm.STOPBITS=1
mm.TIMEOUT=0.2


__author__  = "Benoit CASTETS"
__email__   = "b.castets@robotiq.com"
#__license__ = "Apache License, Version 2.0"

class RobotiqGripper( mm.Instrument ):
    """"Instrument class for Robotiq grippers (2F85, 2F140, hande,...). 
    
    Communicates via Modbus RTU protocol (via RS232 or RS485), using the *MinimalModbus* Python module.    
    Args:
        * portname (str): port name
        * slaveaddress (int): slave address in the range 1 to 247
    Implemented with these function codes (in decimal):
        
    ==================  ====================
    Description         Modbus function code
    ==================  ====================
    Read registers      3
    Write registers     16
    ==================  ====================
    
    For more information for gripper communication please check gripper manual
    on Robotiq website.
    https://robotiq.com/support/2f-85-2f-140
    """
    
    def __init__(self, portname, slaveaddress=9):
        """Create a RobotiqGripper object use to control Robotiq grippers
        using modbus RTU protocol USB/RS485 connection.
        
        Parameters
        ----------
        portname:
            Name of the port (string) where is connected the gripper. Usually
            /dev/ttyUSB0 on Linux. It is necesary to allow permission to access
            this connection using the bash command sudo chmod 666 /dev/ttyUSB0
        slaveaddress:
            Address of the gripper (integer) usually 9.
        """
        mm.Instrument.__init__(self, portname, slaveaddress=9)
        self.debug=True
        self.mode=mm.MODE_RTU
        
        self.processing=False
        
        self.timeOut=10
        
        self.registerDic={}
        self._buildRegisterDic()
        
        self.paramDic={}
        self.readAll()
        
        self.closemm=None
        self.closebit=None
        
        self.openmm=None
        self.openbit=None
        
        self._aCoef=None
        self._bCoef=None
        
        print('Finish Initializing the Gripper!')
        print('################################')
        
    def _buildRegisterDic(self):
        """Build a dictionnary with comment to explain each register variable.
        The dictionnary is organize in 2 levels:
        Dictionnary key are variable names. Dictionnary value are dictionnary
        with comments about each statut of the variable 
        (key=variable value, value=comment)
        """
        ######################################################################
        #input register variable
        self.registerDic.update({"gOBJ":{},"gSTA":{},"gGTO":{},"gACT":{},
                                "kFLT":{},"gFLT":{},"gPR":{},"gPO":{},"gCU":{}})
        print(self.registerDic)
        #gOBJ
        gOBJdic=self.registerDic["gOBJ"]
        
        gOBJdic[0]="Fingers are in motion towards requested position. No object detected."
        gOBJdic[1]="Fingers have stopped due to a contact while opening before requested position. Object detected opening."
        gOBJdic[2]="Fingers have stopped due to a contact while closing before requested position. Object detected closing."
        gOBJdic[3]="Fingers are at requested position. No object detected or object has been loss / dropped."
        
        #gSTA
        gSTAdic=self.registerDic["gSTA"]
        
        gSTAdic[0]="Gripper is in reset ( or automatic release ) state. See Fault Status if Gripper is activated."
        gSTAdic[1]="Activation in progress."
        gSTAdic[3]="Activation is completed."
        
        #gGTO
        gGTOdic=self.registerDic["gGTO"]
        
        gGTOdic[0]="Stopped (or performing activation / automatic release)."
        gGTOdic[1]="Go to Position Request."
        #gGTOdic[2]="Stopped (or performing activation / automatic release)."
        #gGTOdic[3]="Go to Position Request."
        
        #gACT
        gACTdic=self.registerDic["gACT"]
        
        gACTdic[0]="Gripper reset."
        gACTdic[1]="Gripper activation."
        
        #kFLT
        kFLTdic=self.registerDic["kFLT"]
        i=0
        while i<256:
            kFLTdic[i]=i
            i+=1
        
        #See your optional Controller Manual (input registers & status).
        
        #gFLT
        gFLTdic=self.registerDic["gFLT"]
        i=0
        while i<256:
            gFLTdic[i]=i
            i+=1
        gFLTdic[0]="No fault (LED is blue)"
        gFLTdic[5]="Priority faults (LED is blue). Action delayed, activation (reactivation) must be completed prior to perfmoring the action."
        gFLTdic[7]="Priority faults (LED is blue). The activation bit must be set prior to action."
        gFLTdic[8]="Minor faults (LED continuous red). Maximum operating temperature exceeded, wait for cool-down."
        gFLTdic[9]="Minor faults (LED continuous red). No communication during at least 1 second."
        gFLTdic[10]="Major faults (LED blinking red/blue) - Reset is required (rising edge on activation bit rACT needed). Under minimum operating voltage."
        gFLTdic[11]="Major faults (LED blinking red/blue) - Reset is required (rising edge on activation bit rACT needed). Automatic release in progress."
        gFLTdic[12]="Major faults (LED blinking red/blue) - Reset is required (rising edge on activation bit rACT needed). Internal fault; contact support@robotiq.com."
        gFLTdic[13]="Major faults (LED blinking red/blue) - Reset is required (rising edge on activation bit rACT needed). Activation fault, verify that no interference or other error occurred."
        gFLTdic[14]="Major faults (LED blinking red/blue) - Reset is required (rising edge on activation bit rACT needed). Overcurrent triggered."
        gFLTdic[15]="Major faults (LED blinking red/blue) - Reset is required (rising edge on activation bit rACT needed). Automatic release completed."
        
        #gPR
        gPRdic=self.registerDic["gPR"]
        
        i=0
        while i<256:
            gPRdic[i]="Echo of the requested position for the Gripper: {}/255".format(i)
            i+=1
        
        #gPO
        gPOdic=self.registerDic["gPO"]
        i=0
        while i<256:
            gPOdic[i]="Actual position of the Gripper obtained via the encoders: {}/255".format(i)
            i+=1
        
        #gCU
        gCUdic=self.registerDic["gCU"]
        i=0
        while i<256:
            current=i*10
            gCUdic[i]="The current is read instantaneously from the motor drive, approximate current: {} mA".format(current)
            i+=1
    
    
        ######################################################################
        #output register varaible
        self.registerDic.update({"rARD":{},"rATR":{},"rGTO":{},"rACT":{},"rPR":{},
                                "rFR":{},"rSP":{}})
        
        ######################################################################
        
    def _extractKBits(integer,position,nbrBits): 
        """Function to extract ‘k’ bits from a given 
        position in a number.
        
        Parameters
        ----------
        integer:
            Integer to by process as a binary number
        position:
            Position of the first bit to be extracted
        nbrBits:
            Number of bits to be extracted form the first bit position.
        
        Return
        ------
        extractedInt:
            Integer representation of extracted bits.
        """ 
        # convert number into binary first 
        binary=bin(integer) 
      
        # remove first two characters 
        binary = binary[2:] 
      
        end = len(binary) - position
        start = end - nbrBits + 1
      
        # extract k  bit sub-string 
        extractedBits = binary[start : end+1] 
      
        # convert extracted sub-string into decimal again 
        extractedInt=int(extractedBits,2)
         
        return extractedInt
     
    def readAll(self):
        """Return a dictionnary will all variable saved in the register
        """
        self.paramDic={}
        
        registers=self.read_registers(2000,6)
        
        #########################################
        #Register 2000
        #gripperStatus
        gripperStatusReg0=bin(registers[0])[2:]
        gripperStatusReg0="0"*(16-len(gripperStatusReg0))+gripperStatusReg0
        gripperStatusReg0=gripperStatusReg0[:8]
        #########################################
        print(gripperStatusReg0)
        self.paramDic["gOBJ"]=gripperStatusReg0[0:2]
        #Object detection
        self.paramDic["gSTA"]=gripperStatusReg0[2:4]
        #Gripper status
        self.paramDic["gGTO"]=gripperStatusReg0[4:6]
        #Action status. echo of rGTO (go to bit)
        self.paramDic["gACT"]=gripperStatusReg0[7]
        #Activation status
        
        #########################################
        #Register 2002
        #fault status
        faultStatusReg2=bin(registers[2])[2:]
        faultStatusReg2="0"*(16-len(faultStatusReg2))+faultStatusReg2
        faultStatusReg2=faultStatusReg2[:8]
        #########################################
        self.paramDic["kFLT"]=faultStatusReg2[0:4]
        #Universal controler
        self.paramDic["gFLT"]=faultStatusReg2[4:]
        #Fault
        
        #########################################
        #Register 2003
        #fault status
        posRequestEchoReg3=bin(registers[3])[2:]
        posRequestEchoReg3="0"*(8-len(posRequestEchoReg3))+posRequestEchoReg3
        posRequestEchoReg3=posRequestEchoReg3[:8]
        #########################################
        self.paramDic["gPR"]=posRequestEchoReg3
        #Echo of request position
        
        #########################################
        #Register 2004
        #position
        positionReg4=bin(registers[4])[2:]
        positionReg4="0"*(16-len(positionReg4))+positionReg4
        positionReg4=positionReg4[:8]
        #########################################
        self.paramDic["gPO"]=positionReg4
        #Actual position of the gripper
        
        #########################################
        #Register 2005
        #current
        currentReg5=bin(registers[5])[2:]
        currentReg5="0"*(16-len(currentReg5))+currentReg5
        currentReg5=currentReg5[:8]
        #########################################
        self.paramDic["gCU"]=currentReg5
        #Current
        
        #Convert of variable from str to int
        for key,value in self.paramDic.items():
            self.paramDic[key]=int(value,2)
    
    def reset(self):
        """Reset the gripper (clear previous activation if any)
        """
        #Lexique:
        
        #byte=8bit
        #bit=1 OR 0
        
        #Memo:
        
        #write a value with dec,hex or binary number
        #binary
        #inst.write_register(1000,int("0000000100000000",2))
        #hex
        #inst.write_register(1000,int("0x0100", 0))
        #dec
        #inst.write_register(1000,256)
        
        #Register 1000: Action Request
        #A register have a size of 2Bytes
        #(7-6)reserved/(5)rARD/(4)rATR/(3)rGTO/(2-1)reserved/(0)rACT/+8 unused bits
        
        #Register 1001:RESERVED
        #register 1002:RESERVED
        
        #Reset the gripper
        self.write_registers(1000,[0,0,0])
        #09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30
    
    def activate(self):
        """If not already activated. Activate the gripper
        """
        #Turn the variable which indicate that the gripper is processing
        #an action to True
        self.processing=True
        #Activate the gripper
        self.write_registers(1000,[256,0,0])
        #09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1
        #Waiting for activation to complete
        timeIni=time.time()
        loop=True
        while loop:
            self.readAll()
            gSTA=self.paramDic["gSTA"]
            # Changes by HT: Before was ((time.time()-timeIni)<self.timeOut) in which the activation never ended.
            if ((time.time()-timeIni)>self.timeOut):
                loop=False
                print("Activation never ended. Time out.")
            elif gSTA==3:
                loop=False
                print("Activation completed")
            else:
                pass
        
        self.processing=False
    
    def resetActivate(self):
        """Reset the gripper (clear previous activation if any) and activate
        the gripper. During this operation the gripper will open and close.
        """
        #Reset the gripper
        self.reset()
        #Activate the gripper
        self.activate()
        
        #TO DO: wait for the activation to complete
    
    def _intToHex(self,integer,digits=2):
        """Convert an integrer into a hexadeciaml number represented by a string
        
        Parameters
        ----------
        integer:
            Integer to be converted in hexadecimal
        digits:
            Number of digits requested. ex: F, 0F, 00F
        """
        exadecimal=hex(integer)[2:]
        exadecimal="0"*(digits-len(exadecimal))+exadecimal
        return exadecimal
    
    def goTo(self,position,speed=255,force=255):
        """Go to the position with determined speed and force.
        
        Parameters
        ----------
        position:
            Position of the gripper. Integer between 0 and 255. 0 being the
            open position and 255 being the close position.
        speed:
            Gripper speed between 0 and 255
        force:
            Gripper force between 0 and 255
        
        Return
        ------
        objectDetected:
            True if object detected
        position:
            End position of the gripper
        """
        objectDetected=False

        print('goTo:', position, speed, force)
        
        #TO DO: Check if gripper is activated and if not activate?
        if position>255:
            print("maximum position is 255")
        else:
            #rARD(5) rATR(4) rGTO(3) rACT(0)
            self.write_registers(1000,[int("00001001"+"00000000",2),
                                       position,
                                       int(self._intToHex(speed)+self._intToHex(force),16)])
            timer=time.time()
            loop=True
            while loop or (time.time()-timer)>self.timeOut:
                self.readAll()
                gOBJ=self.paramDic["gOBJ"]

                if gOBJ==1 or gOBJ==2:
                    objectDetected=True
                    loop=False
                elif gOBJ==3:
                    objectDetected=False
                    loop=False
                elif (time.time()-timer)>self.timeOut:
                    loop=False
                    print("Gripper never reach its requested position and no\
                           object have been detected")
                    
        
        position=self.paramDic["gPO"]
        
        #TO DO: Check if gripper is in position. If no wait.
        
    def closeGripper(self,speed=255,force=255):
        """Close the gripper
        
        Parameters
        ----------
        speed:
            Gripper speed between 0 and 255
        force:
            Gripper force between 0 and 255
        """
        self.goTo(255,speed,force)
    
    def openGripper(self,speed=255,force=255):
        """Open the gripper
        
        Parameters
        ----------
        speed:
            Gripper speed between 0 and 255
        force:
            Gripper force between 0 and 255
        """
        self.goTo(0,force,speed)
    
    def goTomm(self,positionmm,speed=255,force=255):
        """Go to the requested opening expressed in mm
        
        Parameters
        ----------
        positionmm:
            Gripper opening in mm.
        speed:
            Gripper speed between 0 and 255
        force:
            Gripper force between 0 and 255
        
        Return
        ------
        Return 0 if succeed, 1 if failed.
        """
        if self.openmm is None or self.closemm is None:
            print("You have to calibrate the gripper before using \
                  the function goTomm()")
            return 1
        elif positionmm>self.openmm:
            print("The maximum opening is {}".format(positionmm))
            return 1
        else:
            position=int(self._mmToBit(positionmm))
            self.goTo(position,speed,force)
            return 0
        
    def getPositionCurrent(self):
        """Return the position of the gripper in bit.
        
        Return
        ------
        position:
            Gripper position in bit
        current:
            Motor current in bit. 1bit is about 10mA.
        """
        
        registers=self.read_registers(2002,1)
        register=self._intToHex(registers[0])
        position=int(register[:2],16)
        current=int(register[2:],16)
        return position,current
    
    def _mmToBit(self,mm):
        """Convert a mm gripper opening in bit opening.
        Calibration is needed to use this function.
        """
        bit=(mm-self._bCoef)/self._aCoef
        
        return bit
        
        
    def _bitTomm(self,bit):
        """Convert a bit gripper opening in mm opening.
        Calibration is needed to use this function.
        """
        mm=self._aCoef*bit+self._bCoef
        
        return mm
    
    def getPositionmm(self):
        """Return the position of the gripper in mm.
        Calibration is need to use this function.
        """
        position=self.getPositionCurrent()[0]
        
        positionmm=self._bitTomm(position)
        return positionmm
    
    def calibrate(self,closemm,openmm):
        """Calibrate the gripper for mm positionning
        """
        self.closemm=closemm
        self.openmm=openmm
        
        self.goTo(0)
        #get open bit
        self.openbit=self.getPositionCurrent()[0]
        obit=self.openbit
        
        self.goTo(255)
        #get close bit
        self.closebit=self.getPositionCurrent()[0]
        cbit=self.closebit
        
        self._aCoef=(closemm-openmm)/(cbit-obit)
        self._bCoef=(openmm*cbit-obit*closemm)/(cbit-obit)
    
    def printInfo(self):
        """Print gripper register info in the python treminal
        """
        self.readAll()
        for key,value in self.paramDic.items():
            print("{} : {}".format(key,value))
            print(self.registerDic[key][value])
            
#Test
#if True:
#    grip = RobotiqGripper("/dev/ttyUSB0")
#    grip.resetActivate()
#    time.sleep(1)
#    # grip.reset()
#    grip.printInfo()
#    # grip.activate()
#    grip.printInfo()
#    
#    
#    grip.closeGripper()
#    grip.openGripper()
#    # grip.goTo(20)
#    # grip.goTo(230)
#    # grip.goTo(40)
#    # grip.goTo(80)
#    
#    # grip.calibrate(0,40)
#    # grip.goTomm(20,255,255)
#    # grip.goTomm(40,1,255)
