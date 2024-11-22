#!/usr/bin/env python


"""
pystim: a Python Class for interacting with various bits of hardware to produce sounds and
record signals.

The audio output hardware is either an National Instruments DAC card or a system sound card
If the NI DAC is available, TDT system 3 hardware is assumed as well for the
attenuators (PA5) and an RP2.1. or RZ5D. We don't use the RP2.1 for sound generation.
The RZ5D is limited by a 48 kHz maximum output rate, and thus to less than stimului 
with components at a maximum of 24 kHz. It is not suitable for the higher frequency sounds 
tht might be required for some small animal (mouse) ABRs.

Hardware on the Manis Lab Rig 5 (ABR) system includes:
RP2.1 (vintage)
RZ5D
NI6731 (high speed 4 channel 16-bit dac)
2 x PA5 attenuators

If the system sound card is used, stimuli are generated. This is used only for testing.


12/17/2008-2024 Paul B. Manis, Ph.D.
UNC Chapel Hill
Department of Otolaryngology/Head and Neck Surgery
Supported by NIH Grants DC000425, DC004551 and DC015093 to PBM.
Tessa J. Ropp, Ph.D. also contributed to the development of this code.

Refactored and modified version, includes access to rz5d to help synchronize acquisition.
August, 2017 and later.

The Python requirements are listed in the requirements.txt file in the root directory of the repository.
Other requirements include:
nidaqmx for Python (https://nidaqmx-python.readthedocs.io/en/latest/)
pyaudio (https://people.csail.mit.edu/hubert/pyaudio/, or more recent versions; only for testing
when other hardware is not available).
tdt rco/rcx/rpx files for the TDT system. These were created with RPvdsEx, and
reside in the tdt directory of the repository. The rco files are compiled versions of the
rcx files, and are used by the RP2.1 and RZ5D systems to control the hardware.

"""


"""
Old:
    (TDT manual system 3): 
    Sweep Control
    To use the sweep control circuit constructs the following names are required:
    zSwPeriod: The period of the sweep duration. This is set in OpenWorkbench
    and can not be modified during block acquisition.
    If it is necessary to change this value during the experiment, an
    *asynchronous next sweep control circuit* construct should be used
     See Asynchronous Next Sweep Control, page 324 for more information.
    317
    OpenEx User's Guide
    318
    zSwCount: The maximum number of sweeps before the signal is terminated.
    If this requires manual or external control, the value should be set to -1 through the OpenWorkbench protocol.

New:
    (Synapse):
    Sweep cycling is controlled by PulseGen1 clock, which sets the
    interstimulus interval.
    The NIDAQ system is using a callback to reload the card at the end of every
    output, and is retriggered by the PulseGen1 (on digital out 0).
    The NIDAQ system can be turned off or on independently of the RZ5D state,
    and can be reloaded in between stimuli as well.

"""

import ctypes
from dataclasses import dataclass, field
from pyqtgraph.Qt.QtCore import QMutex
from pathlib import Path
import platform
import struct
import time
import numpy as np


# Check for the system we are running on, and what hardware/software is available.
opsys = platform.system()
nidaq_available = False
if opsys in ["nt", "Windows"]:
    try:
        print("Testing import of nidaqmx")
        import nidaq.nidaq as nidaq
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, Edge, VoltageUnits

        print("    nidaq, nidaqmx and nidaqmx.constants were imported ok.")
        print("\nTesting import of tdt.py")
        import tdt

        print("    tdt.py was imported ok.")
        print("\nTesting import win32com.client")
        import win32com.client
        print("    win32com.client was imported ok")

  
        nidaq_available = True
    except:
        raise ImportError("Some required imports failed - check the system and the installation of the required packages")

# If we are not on Windows, we can use the system sound card for testing.

if opsys in ["Darwin", "Linux"] or nidaq_available == False:
    import pyaudio

# The following are reference values for rough calibrations
# They do not correct for system frequency responses
# They are old.

REF_ES_dB = 86.0  # calibration info -  Assumes 10 dB padding with attenuator.
REF_ES_volt = 2.0  # output in volts to get refdb
REF_MAG_dB = 100.0  # right speaker is mag... different scaling.

#  RZ5D State flags when using Synapse
RZ5D_Idle = 0
RZ5D_Preview = 2
RZ5D_Standby = 1
RZ5D_Run = 3

def_soundcard_outputrate = 44100
def_soundcard_inrate = 44100
def_NI_outputrate = 1000000




# define empty list function for dataclasses
def defemptylist():
    return []


@dataclass
class Stimulus_Status:
    """
    Create data structure for the status of the stimulus generator
    """
    controller: object = None
    running: bool = False
    stimulus_count: int = 0
    done: bool = False
    index: int = 0
    debugFlag: bool = False
    NI_devicename: str = ""
    NIDAQ_task: object = None
    required_hardware: list = field(default_factory=defemptylist)
    hardware: list = field(default_factory=defemptylist)
    max_repetitions: int = 10


class Stimulus_Parameters:
    """
    Data structure for the stimulus parameters,
    populated with some default values
    The defaults need to be checked, as the hardware
    setup should have instantiated them at their correct values.
    """

    NI_out_sampleFreq: float = 500000
    RP21_out_sampleFreq: float = 0.0
    RP21_in_sampleFreq: float = 0.0
    soundcard_out_sampleFreq: float = def_soundcard_outputrate
    soundcard_in_sampleFreq: float = def_soundcard_inrate
    atten_left: float = 120.0
    atten_right: float = 120.0


class PyStim:
    """ PyStim class: a class to control the stimulus generation and data acquisition

    """
    def __init__(self, required_hardware=["Soundcard"], ni_devicename:str="Dev1", controller=None,
                 acquisition_mode:str="abr"):
        """
        During initialization, we identify what hardware is available.

        Parameters
        ----------
        required_hardware : list : (Default: ['Soundcard'])
            A list of the names of devices we expect to be able to use
            For example: ['PA5', 'NIDAQ', 'RZ5D'] for an attenuator, an NI
            card (for DAC output) and the TDT RZ5D DSP unit. other combinations
            are possible (but not all have been tested or are useful)
        nidevicename : str (Default: 'dev1')
            The device name for the NI device we will use. Get this from
            the NIDAQmx system configuration utility that is provided by National Instruments.
        controller : object
            The parent class that provides the controls.
        """

        self.State = Stimulus_Status()  # create instance of each data structure (class)
        self.State.required_hardware = required_hardware  # save the caller data
        self.State.NI_devicename = ni_devicename
        self.State.controller = controller
        self.Stimulus = Stimulus_Parameters()  # create instance of the stimulus

        self.find_hardware(acquisition_mode = acquisition_mode)  # look for the required hardware and make sure we can talk to it.

        self.ch1 = None  # These will be arrays to receive the a/d sampled data
        self.ch2 = None
        self.audio = None  # pyaudio object  - get later
        self.State.NIDAQ_task = None  # nidaq task object - get later
        self.stim_mutex = QMutex()

    def find_hardware(self, acquisition_mode="abr", verbose:bool=False):
        """
        Find the required hardware on the system.
        For non-windows systems, this just finds the system soundcard for testing
        Otherwise it looks for the requested hardware.
        Keeps track of available hardware in the self.State.hardware list

        Parameters
        ----------
        None

        """
        if (
            opsys in ["Darwin", "Linux"] or nidaq_available is False
        ):  # If we are not on a Windows system, just set up soundcard
            print(f"Found operation system: {opsys}; We only support the sound card")
            self.setup_soundcard()
            self.State.hardware.append("Soundcard")
            #  TODO: check for other sound card sample rates, and use the maximum rate
            # or a specified rate from the configuration file.
        elif opsys == "Windows":
            if "NIDAQ" in self.State.required_hardware: #  and self.setup_nidaq():
                self.State.hardware.append("NIDAQ")
                # self.setup_nidaq()
            if "RP21" in self.State.required_hardware:
                assert acquisition_mode in ["abr", "calibrate"]
                print("looking for RP21")
                if acquisition_mode == "abr":
                    if self.setup_RP21(
                     # "c:\\TDT\\OpenEx\\MyProjects\\Tetrode\\RCOCircuits\\tone_search.rcx"
                        "c:\\users\\experimenters\\desktop\\pyabr\\tdt\\abrs.rcx",
                        acquisition_mode="abr"
                    ):
                        self.State.hardware.append("RP21")
                    else:
                        print("RP21 expected, but was not found")
                        raise NotImplementedError("RP21 expected, but was not found")
                elif acquisition_mode == "calibrate":
                    if self.setup_RP21(
                        "c:\\Users\\experimenters\\Desktop\\pyabr\\tdt\\mic_record.rcx",
                        acquisition_mode="calibrate"):
                        self.State.hardware.append("RP21")
                    else:
                        print("RP21 expected, but was not found")
                        raise NotImplementedError("RP21 expected, but was not found")
               
                else:
                    raise ValueError(f"RP21 acquisition mode must be 'abr' or 'calibrate'; got: '{acquisition_mode:s}'")
            if "PA5" in self.State.required_hardware and self.setup_PA5():
                self.State.hardware.append("PA5")
            if "RZ5D" in self.State.required_hardware and self.setup_RZ5D():
                self.State.hardware.append("RZ5D")
        else:
            raise NotImplementedError("Unknown operating system: {opsys}")
        
        print("Hardware found: ", self.State.hardware)

    def getHardware(self):
        """getHardware: get some information about the hardware setup

        Returns
        -------
        tuple
            The hardware state data
            the current output and input sample rates

        Raises
        ------
        ValueError
            if not valid hardware is in the State list of hardware.
        """
        print("Hardware: self.state.hardware: ", self.State.hardware)
        if "NIDAQ" in self.State.hardware:
            sfout = self.Stimulus.NI_out_sampleFreq
        elif "RP21" in self.State.hardware:
            sfout = self.Stimulus.RP21_out_sampleFreq
        elif "Soundcard" in self.State.hardware:
            sfout = self.Stimulus.soundcard_out_sampleFreq
        else:
            raise ValueError("pystim.getHardware: No Valid OUTPUT hardware found")
        if "RP21" in self.State.hardware:
            sfin = self.Stimulus.RP21_in_sampleFreq
        elif "Soundcard" in self.State.hardware:
            sfin = self.Stimulus.soundcard_in_sampleFreq
        else:
            raise ValueError("pystim.getHardware: No Valid INTPUT hardware found")
        print("Hardware: ", self.State.hardware)
        print("Sample Freqs: ", sfout, sfin)
        return (self.State.hardware, sfin, sfout)

    def reset_hardware(self):
        """
        Reset the hardware to initial state
        """
        if "RZ5D" in self.State.hardware:
            if self.RZ5D is not None:
                self.RZ5D.setModeStr("Idle")
        if "PA5" in self.State.hardware:
            if self.PA5 is not None:
                self.PA5.SetAtten(120.0)
        if "RP21" in self.State.hardware:
            if self.RP21 is not None:
                self.RP21.Halt()
        if "NIDAQ" in self.State.hardware:
            if self.State.NIDAQ_task is not None:
                self.State.NIDAQ_task.close()
        if "pyaudio" in self.State.hardware:
            if self.audio is not None:
                self.audio.terminate()


    def setup_soundcard(self):
        if self.State.debugFlag:
            print("pystim:setup_soundcard: Your OS or available hardware only supports a standard sound card")
        self.State.hardware.append("pyaudio")

    def setup_nidaq(self):
        # get the drivers and the activeX control (win32com)
        self.NIDevice = nidaqmx.system.System.local()
        self.NIDevicename = self.NIDevice.devices.device_names
        self.Stimulus.NI_out_sampleFreq = 1000000  # output frequency, in Hz
        return True

    def show_nidaq(self):
        """
        Report some information regardign the nidaq setup
        """
        print("pystim:show_nidaq found the follwing nidaq devices:")
        print(f"    {self.NIDevice.devices.device_names:s}")
        # print ("devices: %s" % nidaq.NIDAQ.listDevices())
        print("    ", self.NIDevice)
        print(
            f"\nAnalog Output Channels: {self.NIDevice.devices[self.NIDevicename].ao_physical_chans.channel_names}"
        )

    def setup_PA5(self, devnum=1):
        """
        Set up the ActiveX connection to the TDT PA5 attenuators

        Parameters
        ----------
        devnum : int (default = 1)
            The device number to connect to for the attenuator
        """
        self.PA5 = win32com.client.Dispatch("PA5.x")
        a = self.PA5.ConnectPA5("USB", devnum)
        if a > 0:
            if self.State.debugFlag:
                print("pystim:setup_PA5: Connected to PA5 Attenuator %d" % devnum)
        else:
            if "PA5" in self.State.required_hardware:
                raise IOError("pystim:setup_PA5: This requirement was requested, but the device not found")
            else:
                return False
        self.PA5.SetAtten(120.0)  # set all attenuators to maximum attenuation
        return True

    def setup_RP21(self, rcofile: str = "", acquisition_mode="abr"):
        """
        Make an ActiveX connection to theTDT RP2.1 Real-Time Processor
        and load the RCO file.

        Parameters
        ----------
        rcofile : str (default : '')
            The RCO file to connect to. Must be the full path.
        """
        if self.State.debugFlag:
            print("Setting up RP21")
        self.RP21_rcofile = rcofile
        if not Path(self.RP21_rcofile).is_file():
            raise FileNotFoundError(f"The required RP2.1 RCO file was not found \n    (Looking for {self.RP21_rcofile})")
        self.RP21 = win32com.client.Dispatch("RPco.x")  # connect to RP2.1
        a = self.RP21.ConnectRP2("USB", 0)
        if a > 0:
            print(f"pystim.setup_RP21: RP2.1 connected, status: {a:d}")
        else:
            print("connect status: ", a)
            raise IOError(f"pystim.setup_RP21: RP2.1 requested in hardware, but connection failed with status: {a:d}")
            exit()
        self.RP21.ClearCOF()
        if acquisition_mode == "abr":
            self.samp_cof_flag = 4  # set for 97 kHz
        elif acquisition_mode == "calibrate":
             self.samp_cof_flag = 5 
        else:
            raise ValueError(f"Acquistion mode must be either 'abr' or 'calibrate', got '{acquisition_mode:s}'")
        self.samp_flist = [
            6103.5256125,
            12210.703125,
            24414.0625,
            48828.125,
            97656.25,
            195312.5,
        ]
        if self.samp_cof_flag > 5:
            self.samp_cof_flag = 5

        a = self.RP21.LoadCOFsf(self.RP21_rcofile, self.samp_cof_flag)
        if a > 0:
            print(
                "pystim.setup_RP21: File %s loaded\n      and sample rate set to %f"
                % (self.RP21_rcofile, self.samp_flist[self.samp_cof_flag])
            )
        else:
            raise FileNotFoundError(f"pystim.setup_RP21: There was an error loading RCO file {rcofile!s}\n    Error = {a:d}")

        # set the input and output sample frequencies to the same value
        self.Stimulus.RP21_out_sampleFreq = self.samp_flist[self.samp_cof_flag]
        self.Stimulus.RP21_in_sampleFreq = self.samp_flist[self.samp_cof_flag]
        return True

    def show_RP21(self):
        """
        TODO: maybe report RP2.1 info: cof rate, loaded circuit, sample freqs
        """
        pass

    def setup_RZ5D(self):
        try: 
            self.RZ5D = tdt.SynapseAPI()
            if self.RZ5D.getModeStr() != "Idle":
                self.RZ5D.setModeStr("Idle")
            return True
        except:
            raise IOError("pystim.setup_RZ5D: RZ5D requested, but not found")

    def get_RZ5D_Params(self):
        self.RZ5DParams = {}  # keep a local copy of the parameters
        self.RZ5DParams["device_name"] = self.RZ5D.getGizmoNames()
        self.RZ5DParams["device status"] = self.RZ5D.getModeStr()

    def show_RZ5D(self):
        print("Device Status: {0:d}".format(self.RZ5DParams["device_status"]))

    def get_RZ5D_Mode(self):
        return self.RZ5D.getModeStr()

    def RZ5D_close(self):
        if self.RZ5D.getModeStr() != "Idle":
            self.RZ5D.setModeStr("Idle")

    def cleanup_NIDAQ_RP21(self):
        self.stim_mutex.lock()
        if "PA5" in self.State.hardware:
            self.setAttens()  # attenuators down (there is noise otherwise)
        # if self.State.NIDAQ_task is not None:
        #     self.State.NIDAQ_task.stop()  # done, so stop the output.
        #     self.State.NIDAQ_task.close()
        #     self.State.NIDAQ_task = None
        if "RP21" in self.State.hardware:
            self.RP21.Halt()
        self.stim_mutex.unlock()

    # internal debug flag to control printing of intermediate messages
    def debugOn(self):
        self.State.debugFlag = True

    def debugOff(self):
        self.State.debugFlag = False

    def dbconvert(self, spl=0, chan=0):
        """
        compute voltage from reference dB level
        db = 20 * log10 (Vsignal/Vref)
        """
        ref = REF_ES_dB
        if chan == 1:
            ref = REF_MAG_dB

        zeroref = REF_ES_volt / (10 ** (ref / 20.0))
        sf = zeroref * 10 ** (spl / 20.0)
        # actually, the voltage needed to get spl out...
        if self.State.debugFlag:
            print("pystim.dbconvert: scale = %f for %f dB" % (sf, spl))
        return sf  # return a scale factor to multiply by a waveform normalized to 1
    

    def setAttens(self, atten_left=120.0, atten_right=120.0):
        if "PA5" in self.State.hardware:
            self.PA5.ConnectPA5("USB", 1)
            self.PA5.SetAtten(atten_left)
            if atten_right is not None:
                self.PA5.ConnectPA5("USB", 2)
                self.PA5.SetAtten(atten_right)

    def play_sound(
        self,
        wavel,
        waver=None,
        samplefreq=44100,
        postduration=0.025,
        attns=[20.0, 20.0],
        isi=1.0,
        reps=1,
        protocol="Search",
        mode="record", # or "calibrate"
        storedata=True,
    ):
        """
        play_sound sends the sound out to an audio device.
        In the absence of NI card, and TDT system, we (try to) use the system audio device (sound card, etc)
        The waveform is played in both channels on sound cards, possibly on both channels
        for other devices if there are 2 channels.

        Parameters
        ----------
        wavel : numpy array of floats
            Left channel waveform
        waver : numpy of floats
            Right channel waveform
        samplefreq : float
            output sample frequency (Hz)
        postduration : float (default: 0.35)
            Time after end of stimulus, in seconds
        attns : 2x1 list (default: [20., 20.])
            Attenuator settings to use for this stimulus
        isi : float (default 1.0)
            Interstimulus interval
        reps : int (default 1)
            Number of repetitions before returning.
        protocol: str (default "Search")
            protocol mode to use.
        storedata : bool (default: True)
            flag to force storage of data at end of run

        """
        if storedata:
            runmode = "Record"
        else:
            runmode = "Preview"

        # if we are just using pyaudio (linux, MacOS), set it up now
        if "pyaudio" in self.State.hardware:
            dur = len(wavel) / float(self.Stimulus.soundcard_out_sampleFreq)
            self.Ndata = int(np.ceil((dur + postduration) *  self.Stimulus.soundcard_out_sampleFreq))
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
            else:
                self.audio.terminate()
                self.audio = pyaudio.PyAudio()
            chunk = 1024
            FORMAT = pyaudio.paFloat32
            # CHANNELS = 2
            CHANNELS = 1
            if self.State.debugFlag:
                print(f"pystim.play_sound: samplefreq: {self.Stimulus.soundcard_out_sampleFreq:.1f} Hz")
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=int(self.Stimulus.soundcard_out_sampleFreq),
                output=True,
                input=True,
                frames_per_buffer=chunk,
            )
            wave = np.zeros(2 * len(wavel))
            if len(wavel) != len(waver):
                print(
                    f"pystim.play_sound: L,R output waves are not the same length: L = {len(wavel):d}, R = {len(waver):d}")
                return
            (waver, clipr) = self.clip(waver, 20.0)
            (wavel, clipl) = self.clip(wavel, 20.0)
            wave[0::2] = waver
            wave[1::2] = wavel  # order chosen so matches etymotic earphones on my macbookpro.
            postdur = int(float(postduration * self.Stimulus.soundcard_in_sampleFreq))

            write_array(self.stream, wave)
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            return

        if "NIDAQ" in self.State.hardware:
            if self.State.NIDAQ_task is not None:
                raise ValueError("NIDAQ task has not been released")
                exit()
            print("Using NIDAQ")
            ndata = len(wavel)
            dur = postduration + ndata / float(self.Stimulus.NI_out_sampleFreq)
            self.waveout = wavel


            if "RP21" in self.State.hardware:
                self.Ndata = int(np.ceil((dur + postduration) * self.Stimulus.RP21_out_sampleFreq))
                self.RP21.SetTagVal("REC_Size", self.Ndata)  # old version using serbuf  -- with
                # a = self.RP21.Run()  # start the RP2.1 processor...
                # print("RP2.1 started")
            if "PA5" in self.State.hardware:
                print("setting attns: ", attns)
                self.setAttens(atten_left=attns[0], atten_right=attns[1])

            # if "RP21" in self.State.hardware:
            #     a = self.RP21.SoftTrg(0)  # and trigger it. RP2.1 will in turn start the ni card
            #     print("softtrig set, returns", a)
            self.prepare_NIDAQ(wavel, None, timeout=dur, re_armable=False)
            dur = 1.5
            print("dur: ", dur)
            # self.State.NIDAQ_task.wait_until_done(timeout=dur)

            # while (
            #     (self.State.NIDAQ_task is not None) or (not self.State.NIDAQ_task.is_task_done())
            # ):  # wait for AO to finish?  The call back will set the task to non
            #     time.sleep(0.01)
            #     print("done? : ", self.State.NIDAQ_task.is_task_done())
            # print("task is done: ", self.State.NIDAQ_task.is_task_done())
            # self.stop_nidaq()

            if "PA5" in self.State.hardware:
                self.setAttens()  # attenuators down (there is noise otherwise)

            if "RP21" in self.State.hardware:
                self.RP21.SoftTrg(0)

                self.ch2 = self.RP21.ReadTagV("Data_out2", 0, self.Ndata)
                # ch2 = ch2 - mean(ch2[1:int(Ndata/20)]) # baseline: first 5% of trace
                self.ch1 = self.RP21.ReadTagV("Data_out1", 0, self.Ndata)
                self.RP21.Halt()
                # compute stimulus and recording waveforms
                if "NIDAQ" in self.State.hardware:
                    self.t_stim = np.arange(
                        0, len(wavel) / self.Stimulus.NI_out_sampleFreq, 1.0 /self.Stimulus.NI_out_sampleFreq
                    )
                else:  # output is via RP21
                    self.t_stim = np.arange(
                        0, len(wavel) / self.Stimulus.RP21_out_sampleFreq, 1.0/self.Stimulus.RP21_out_sampleFreq
                    )   
                self.t_record = np.arange(0, float(self.Ndata) / self.Stimulus.RP21_in_sampleFreq,
                                           1.0 / self.Stimulus.RP21_in_sampleFreq,)
            # self.cleanup_NIDAQ_RP21()




    def stop_nidaq(self, task_handle=None):
        """
        Only stop the DAC, not the RZ5D
        This is used when reloading a new stimulus.
        """
        self.stim_mutex.lock()
        if task_handle is None:
            task_handle = self.State.NIDAQ_task
        if task_handle is not None:
            task_handle.stop()
            task_handle.close()  # release resources
            self.State.NIDAQ_task = None  # need to destroy value
            self.State.running = False
        self.stim_mutex.unlock()

    def stop_recording(self):
        """
        Stop the entire system (DAC and RZ5D)
        """
        self.stop_nidaq()
        if "RZ5D" in self.State.hardware:
            self.RZ5D.setModeStr("Idle")
        self.setAttens()

    def arm_NIDAQ(self):
        """
        Load up the NI card output buffer, and set the triggers
        This gets the card ready to put out the buffer with the
        next trigger pulse
        """
        if self.State.NIDAQ_task is None:
            raise ValueError(f"NIDAQ task is not available? ")    
        self.State.NIDAQ_task.write(self.waveout, auto_start=False)
        trigger_source = f"/{self.State.NI_devicename:s}/PFI0"
        self.State.NIDAQ_task.triggers.start_trigger.trig_type.DIGITAL_EDGE
        self.State.NIDAQ_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=trigger_source,
            trigger_edge=Edge.RISING,
        )
        self.State.running = True

    def re_arm_NIDAQ(self, task_handle, status, callback_data):
        """
        Callback for when the daq is done...
        Re arm the dac card and start the task again
        """
        print("rearm: ", task_handle, status, callback_data)
        if not self.re_arm:  # one-shot
            return 0
    
        
        if task_handle.is_task_done():
            task_handle.stop()
            self.arm_NIDAQ()  # reload
            task_handle.start()
            self.State.stimulus_count += 1

            counter_elapsed = self.State.stimulus_count > self.repetitions
            #   controller_running = self.State.controller.running
            timeout = False  # (time.time() - self.start_time) > self.timeout
            if counter_elapsed or (not self.State.running) or timeout:
                self.stop_nidaq(task_handle=task_handle)
                self.State.done = True

        return 0

    def load_and_arm_NIDAQ(self, re_arm:bool=False):
        """
        Initial setup of NI card for AO.
        Creates a task for the card, sets parameters, clock rate,
        and does setup if needed.
        A callback is registered so that when the task is done, the
        board is either released or re-armed for the next trigger.
        The callback is used so that the task does not block the GUI.
        """
            
        self.re_arm = re_arm
        this_starttime = time.time()
        failed = False
        with nidaqmx.task.Task("NI_DAC_out") as self.State.NIDAQ_task:
            channel_name = f"/{self.State.NI_devicename:s}/ao0"
            self.State.NIDAQ_task.ao_channels.add_ao_voltage_chan(  # can only do this once...
                channel_name, min_val=-10.0, max_val=10.0, units=VoltageUnits.VOLTS
            )
            self.State.NIDAQ_task.timing.cfg_samp_clk_timing(
                rate = self.Stimulus.NI_out_sampleFreq,
                source="",
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=len(self.waveout),
            )

            self.State.NIDAQ_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source="/Dev1/PFI0",
                trigger_edge=Edge.RISING,
            )
            self.State.NIDAQ_task.write(self.waveout)
            self.State.NIDAQ_task.start()
            if "RP21" in self.State.hardware:
                self.RP21.Run()
                a = self.RP21.SoftTrg(1)  # and trigger it. RP2.1 will in turn start the ni card
                print("softtrig ste to start, returned:", a)
            while not self.State.NIDAQ_task.is_task_done():
                now_time = time.time()
                if now_time - this_starttime > 5.0:
                    failed = True
                    print("arming nidaq/task execution FAILED")
                    break
            self.RP21.SoftTrg(0)
            self.State.NIDAQ_task.stop()
        # self.State.NIDAQ_task.close()
        self.State.NIDAQ_task = None
        return True

    def prepare_NIDAQ(self, wavel, waver=None, repetitions: int = 1, timeout: float = 1200.0, re_armable:bool=False):
        """
        Set up and initialize the NIDAQ card for output,
        then let it run and keep up with each task completion
        so it can be retriggered on the next trigger pulse.
        Configured so that if we are currently running, the run is immediately stopped
        so we can setup right away.
        """

        # self.stop_nidaq()  # stop the DAC if it is running
        self.waveout = wavel
        self.repetitions = repetitions
        self.State.stimulus_count = 0
        (self.waveout, clipl) = self.clip(self.waveout, 10.0)  # clip the wave if it's >10V
        self.start_time = time.time()
        self.timeout = timeout
        self.load_and_arm_NIDAQ(re_arm=re_armable)

    def retrieveRP21_inputs(self):
        return (self.ch1, self.ch2)

    def HwOff(self):  # turn the hardware off.

        if "Soundcard" in self.State.hardware:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
            except:
                pass  # possible we never created teh stream...

        if "NIDAQ" in self.State.hardware:
            if self.State.NIDAQ_task is not None:
                self.Stat.NIDAQ_task.register_done_event(None)
                self.stop_nidaq(task_handle=self.State.NIDAQ_task)

        if "RP21" in self.State.hardware:
            self.RP21.Halt()

        if "RZ5D" in self.State.hardware:
            self.RZ5D_close()

    # clip data to max value (+/-) to avoid problems with daqs
    def clip(self, data, maxval):
        if self.State.debugFlag:
            print(
                "pysounds.clip: max(data) = %f, %f and maxval = %f" % (max(data), min(data), maxval)
            )
        clip = 0
        u = np.where(data >= maxval)
        ul = list(np.transpose(u).flat)
        if len(ul) > 0:
            data[ul] = maxval
            clip = 1  # set a flag in case we want to know
            if self.State.debugFlag:
                print("pysounds.clip: clipping %d positive points" % (len(ul)))
        minval = -maxval
        v = np.where(data <= minval)
        vl = list(np.transpose(v).flat)
        if len(vl) > 0:
            data[vl] = minval
            clip = 1
            if self.State.debugFlag:
                print("pysounds.clip: clipping %d negative points" % (len(vl)))
        if self.State.debugFlag:
            print(
                "pysounds.clip: clipped max(data) = %f, %f and maxval = %f"
                % (np.max(data), np.min(data), maxval)
            )
        return (data, clip)


"""
the following was taken from #http://hlzr.net/docs/pyaudio.html
it is used for reading and writing to the system audio device

"""


def write_array(stream, data):
    """
    Outputs a numpy array to the audio port, using PyAudio.
    """
    # Make Buffer
    buffer_size = struct.calcsize("@f") * len(data)
    output_buffer = ctypes.create_string_buffer(buffer_size)

    # Fill Up Buffer
    # struct needs @fffff, one f for each float
    dataformat = "@" + "f" * len(data)
    struct.pack_into(dataformat, output_buffer, 0, *data)

    # Shove contents of buffer out audio port
    stream.write(output_buffer)


def read_array(stream, size, channels=1):
    input_str_buffer = np.zeros((size, 1))  # stream.read(size)
    input_float_buffer = struct.unpack("@" + "f" * size * channels, input_str_buffer)
    return np.array(input_float_buffer)


if __name__ == "__main__":

    p = PyStim(required_hardware=["PA5", "NIDAQ", "RP21"], ni_devicename="dev1")

    ni_sample_frequency = 100000
    t = np.arange(0, 2.0, 1.0 / ni_sample_frequency)
    w = 10* np.cos(2 * np.pi * 1000.0 * t)
    # import matplotlib.pyplot as mpl
    # mpl.plot(t, w)
    # mpl.show()

    p.play_sound(w)

    # p.stop_nidaq()
