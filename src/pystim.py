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
    NI_task: object = None
    required_hardware: list = field(default_factory=defemptylist)
    hardware: list = field(default_factory=defemptylist)
    max_repetitions: int = 10


class Stimulus_Parameters:
    """
    Create data structure for the stimulus parameters,
    and populate with some default values
    """

    out_sampleFreq: float = 500000
    in_sampleFreq: float = 44100.0
    atten_left: float = 30.0
    atten_right: float = 120.0


class PyStim:
    """ PyStim class: a class to control the stimulus generation and data acquisition

    """
    def __init__(self, required_hardware=["Soundcard"], ni_devicename="dev1", controller=None):
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
        self.find_hardware()  # look for the required hardware and make sure we can talk to it.

        self.ch1 = None  # These will be arrays to receive the a/d sampled data
        self.ch2 = None
        self.trueFreq = None  # actual input acquisiton sample frequency
        self.audio = None  # pyaudio object  - get later
        self.NIDAQ_task = None  # nidaq task object - get later

    def find_hardware(self, verbose:bool=False):
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
            self.Stimulus.out_samplefreq = 44100  # use the default sound card sample rate
            #  TODO: check for other sound card sample rates, and use the maximum rate
            # or a specified rate from the configuration file.
        elif opsys == "Windows":
            if "NIDAQ" in self.State.required_hardware and self.setup_nidaq():
                self.State.hardware.append("NIDAQ")
                self.setup_nidaq()
            if "RP21" in self.State.required_hardware:
                print("looking for RP21")
                if self.setup_RP21(
                    # "c:\\TDT\\OpenEx\\MyProjects\\Tetrode\\RCOCircuits\\tone_search.rcx"
                    "c:\\users\\experimenters\\desktop\\pyabr\\tdt\\abrs.rcx"
                ):
                    self.State.hardware.append("RP21")
                else:
                    print("RP21 expected, but was not found")
                    raise NotImplementedError("RP21 expected, but was not found")
            if "PA5" in self.State.required_hardware and self.setup_PA5():
                self.State.hardware.append("PA5")
            if "RZ5D" in self.State.required_hardware and self.setup_RZ5D():
                self.State.hardware.append("RZ5D")
        else:
            raise NonImplementedError("Unknown operating system: {opsys}")
        
        print("Hardware found: ", self.State.hardware)

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
            if self.NIDAQ_task is not None:
                self.NIDAQ_task.close()
        if "pyaudio" in self.State.hardware:
            if self.audio is not None:
                self.audio.terminate()

    def setup_soundcard(self):
        if self.State.debugFlag:
            print("pystim:setup_soundcard: Your OS or available hardware only supports a standard sound card")
        self.State.hardware.append("pyaudio")
        self.Stimulus.out_sampleFreq = 44100.0
        self.Stimulus.in_sampleFreq = 44100.0

    def setup_nidaq(self):
        # get the drivers and the activeX control (win32com)
        self.NIDevice = nidaqmx.system.System.local()
        self.NIDevicename = self.NIDevice.devices.device_names
        self.Stimulus.out_sampleFreq = 500000  # output frequency, in Hz
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

    def setup_RP21(self, rcofile: str = ""):
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
        # print(self.RP21)
        # print(dir(self.RP21))
        # try to make the connection
        a = self.RP21.ConnectRP2("USB", 0)
        if a > 0:
            print(f"pystim.setup_RP21: RP2.1 connected, status: {a:d}")
        else:
            print("connect status: ", a)
            raise IOError(f"pystim.setup_RP21: RP2.1 requested in hardware, but connection failed with status: {a:d}")
        self.RP21.ClearCOF()
        self.samp_cof_flag = 4  # 2 is for 24.4 kHz
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
        self.Stimulus.out_sampleFreq = self.samp_flist[self.samp_cof_flag]
        self.Stimulus.in_sampleFreq = self.samp_flist[self.samp_cof_flag]
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

    def getHardware(self):
        return (self.State.hardware, self.Stimulus.out_sampleFreq, self.Stimulus.in_sampleFreq)

    def cleanup_NIDAQ(self):
        if self.NIDAQ_task is not None:
            self.NIDAQ_task.stop()  # done, so stop the output.
            self.NIDAQ_task.close()
            self.NIDAQ_task = None

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
        postduration=0.1,
        attns=[20.0, 20.0],
        isi=1.0,
        reps=1,
        protocol="Search",
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
            self.trueFreq = samplefreq
            dur = len(wavel) / float(samplefreq)
            self.Ndata = int(np.ceil((dur + postduration) * self.Stimulus.out_samplefreq))
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
                print(f"pystim.play_sound: samplefreq: {self.Stimulus.out_samplefreq:.1f} Hz")
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=int(self.Stimulus.out_samplefreq),
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
            postdur = int(float(postduration * self.Stimulus.in_sampleFreq))

            write_array(self.stream, wave)
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            return

        if "NIDAQ" in self.State.hardware:
            dev = self.State.NI_devicename
            self.NIDAQ_task = nidaqmx.Task()
            self.NIDAQ_task.ao_channels.add_ao_voltage_chan(
                f"{dev}/ao0", min_val=-10.0, max_val=10.0
            )
            ndata = len(wavel)
            print("NIDAQ clock: ", samplefreq)
            self.NIDAQ_task.timing.cfg_samp_clk_timing(
                rate=samplefreq,
                source="",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=ndata,
            )

            daqwave = np.zeros(ndata * 2)
            (wavel, clipl) = self.clip(wavel, 10.0)
            if len(waver) is not None:
                (waver, clipr) = self.clip(waver, 10.0)

            daqwave[0 : len(wavel)] = wavel
            if len(waver) is not None:
                daqwave[len(wavel) :] = waver
            # concatenate channels (using "groupbychannel" in writeanalogf64)
            dur = ndata / float(samplefreq)
            self.NIDAQ_task.write(wavel)

            if "RP21" in self.State.hardware:
                self.trueFreq = self.RP21.GetSFreq()
                self.Ndata = int(np.ceil((dur + postduration) * self.trueFreq))
                self.RP21.SetTagVal("REC_Size", self.Ndata)  # old version using serbuf  -- with
                # new version using SerialBuf, can't set data size - it is fixed.
                # however, old version could not read the data size tag value, so
                # could not determine when buffer was full/acquisition was done.
            if "PA5" in self.State.hardware:
                self.setAttens(atten_left=attns[0], atten_right=attns[1])

            self.NIDAQ_task.start()  # start the NI AO task
            if "RP21" in self.State.hardware:
                a = self.RP21.Run()  # start the RP2.1 processor...
                a = self.RP21.SoftTrg(1)  # and trigger it. RP2.1 will in turn start the ni card
            self.PPGo = False
            print("duration: ", dur)
            while (
                self.NIDAQ_task is not None and not self.NIDAQ_task.is_task_done()
            ):  # wait for AO to finish?
                if not self.PPGo:  # while waiting, check for stop.
                    time.sleep(dur)
                    if "RP21" in self.State.hardware:
                        self.RP21.Halt()
                    # self.NIDAQ_task.stop()
                    # return
            self.cleanup_NIDAQ()
            if self.NIDAQ_task is not None:
                self.NIDAQ_task.stop()  # done, so stop the output.
                self.NIDAQ_task.close()
            if "PA5" in self.State.hardware:
                self.setAttens()  # attenuators down (there is noise otherwise)
            # read the data...
            curindex1 = self.RP21.GetTagVal("Index1")
            # print(curindex1)
            curindex2 = self.RP21.GetTagVal("Index2")
            if "RP21" in self.State.hardware:
                self.RP21.Halt()
            #     while (
            #         curindex1 < Ndata or curindex2 < Ndata
            #     ):  # wait for input data to be sampled
            #         # if not self.PPGo:  # while waiting, check for stop.
            #         #     self.RP21.Halt()
            #         #     return
            #         curindex1 = self.RP21.GetTagVal("Index1")
            #         curindex2 = self.RP21.GetTagVal("Index2")
            #         print(curindex1, curindex2)

            if "RP21" in self.State.hardware:
                self.ch2 = self.RP21.ReadTagV("Data_out2", 0, self.Ndata)
                # ch2 = ch2 - mean(ch2[1:int(Ndata/20)]) # baseline: first 5% of trace
                self.ch1 = self.RP21.ReadTagV("Data_out1", 0, self.Ndata)
                self.RP21.Halt()
                self.t_stim = np.arange(
                    0, len(wavel) / samplefreq, 1.0 / samplefreq
                )
                self.t_record = np.arange(0, self.Ndata / self.trueFreq, 1 / self.trueFreq)
                # pg.plot(t_stim, wavel)
                # pg.plot(t_record, self.ch1)
                # if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
                #     pg.QtGui.QGuiApplication.instance().exec()

        if "PA5" in self.State.hardware:
            # print("539: attns: ", attns)
            self.setAttens(atten_left=attns[0], atten_right=attns[1])

        if "RZ5D" in self.State.hardware:
            swcount = -1
            timeout = isi * reps + 1
            # Start and run the stim/recording for specified # sweeps/time.
            # self.RZ5D.setModeStr(runmode)
            self._present_stim(
                wavel,
                stimulus_period=isi,
                repetitions=reps,
                runmode=runmode,
                protocol=protocol,
                timeout=timeout,
            )  # this sets up the NI card.
            # at this point we return to the main caller
            # stimuli will be presented and data collected (if in record mode)
            # The caller needs to check for the done_flag

            # while time.time()-start_time < deadmantimer:
            #     time.sleep(0.01)
            # sweeps_start = self.RZ5D.getSystemStatus()['recordSecs']
            # currTime = 0
            # prevTime = 0
            # print(f"ISI: {isi:.3f}  reps: {reps:d}, runmode: {runmode:s}")
            # print(f"Running for maximum of: {deadmantimer:.2f} seconds")
            # while currTime < deadmantimer:
            #     currTime = self.RZ5D.getSystemStatus()['recordSecs']-sweeps_start
            #     if prevTime != currTime:
            #         print(f"Running, sweeps time elapsed: {currTime:d} sec")
            #     prevTime = currTime
            # TFR end comment 10/12/21
            # print("nidaq has stopped")
            # if runmode == "Preview":
            #     return
            # else:
            #     self.RZ5D.setModeStr("Idle")  # was (RZ5D_Standby)

            # self.setAttens(atten_left=120)

    def _present_stim(
        self,
        waveforms,
        stimulus_period: float = 1.0,
        repetitions: int = 1,
        runmode: str = "Preview",
        protocol: str = "Search",
        timeout: float = 10.0,
    ):
        """ """
        self.State.done = False
        if self.RZ5D.getModeStr() != runmode:  # make sure the rz5d is in the requested mode first
            self.RZ5D.setModeStr(runmode)
        ##################################################################################
        # Set up the stimulus timing
        # We use the PulseGen1 to write to digital line out 0
        # This bit controls/triggers the timing of the stimuli (interstimulus interval)

        params = self.RZ5D.getParameterNames("PulseGen1")
        self.RZ5D.setParameterValue("PulseGen1", "PulsePeriod", stimulus_period)
        self.RZ5D.setParameterValue("PulseGen1", "DutyCycle", 1.0)  # 1 msec pulse
        self.RZ5D.setParameterValue("PulseGen1", "Enable", 1.0)
        ##################################################################################

        self.prepare_NIDAQ(waveforms, repetitions=repetitions)  # load up NIDAQ to go

    def stop_nidaq(self):
        """
        Only stop the DAC, not the RZ5D
        This is used when reloading a new stimulus.
        """
        if self.State.NI_task is not None:
            self.State.NI_task.close()  # release resources
            self.State.NI_task = None  # need to destroy value
            self.State.running = False

    def stop_recording(self):
        """
        Stop the entire system (DAC and RZ5D)
        """
        self.stop_nidaq()
        self.RZ5D.setModeStr("Idle")
        self.setAttens()

    def arm_NIDAQ(self):
        """
        Load up the NI card output buffer, and set the triggers
        This gets the card ready to put out the buffer with the
        next trigger pulse
        """
        self.State.NI_task.write(self.waveout, auto_start=False)
        #  self.State.NI_task.triggers.start_trigger.trig_type.DIGITAL_EDGE
        self.State.NI_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=f"{self.State.NI_devicename:s}/PFI0",
            trigger_edge=Edge.RISING,
        )
        self.State.running = True

    def re_arm_NIDAQ(self, task_handle, status, callback_data):
        """
        Callback for when the daq is done...
        Re arm the dac card and start the task again
        """

        if status != 0:
            self.stop_recording()  # nidaq failure?
            return False

        if self.State.NI_task.is_task_done():
            self.State.NI_task.stop()
            self.arm_NIDAQ()  # reload
            self.State.NI_task.start()
            self.State.stimulus_count += 1

            counter_elapsed = self.State.stimulus_count > self.repetitions
            #   controller_running = self.State.controller.running
            timeout = False  # (time.time() - self.start_time) > self.timeout
            if counter_elapsed or (not self.State.running) or timeout:
                self.stop_nidaq()
                self.State.done = True
                return False
        return True

    def load_and_arm_NIDAQ(self):
        """
        Initial setup of NI card for AO.
        Creates a task for the card, sets parameters, clock rate,
        and does setup if needed.
        A callback is registered so that when the task is done, the
        board is re-armed for the next trigger.
        This does not block the GUI.
        """
        self.State.NI_task = nidaqmx.task.Task("NI_DAC_out")
        channel_name = f"/{self.State.NI_devicename:s}/ao0"
        self.State.NI_task.ao_channels.add_ao_voltage_chan(  # can only do this once...
            channel_name, min_val=-10.0, max_val=10.0, units=VoltageUnits.VOLTS
        )
        self.State.NI_task.register_done_event(self.re_arm_NIDAQ)

        self.State.NI_task.timing.cfg_samp_clk_timing(
            rate = 500000, # self.Stimulus.out_sampleFreq,
            source="",
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=len(self.waveout),
        )

        # if not self.State.running:
        #     self.State.NI_task.stop()
        #     return False
        self.arm_NIDAQ()  # setup the DAC card
        self.State.NI_task.start()  # and start it
        return True

    def prepare_NIDAQ(self, wavel, waver=None, repetitions: int = 1, timeout: float = 1200.0):
        """
        Set up and initialize the NIDAQ card for output,
        then let it run and keep up with each task completion
        so it can be retriggered on the next trigger pulse.
        Configured so that if we are currently running, the run is immediately stopped
        so we can setup right away.
        """

        self.stop_nidaq()  # stop the DAC if it is running
        self.waveout = wavel
        self.repetitions = repetitions
        self.State.stimulus_count = 0
        (self.waveout, clipl) = self.clip(self.waveout, 10.0)  # clip the wave if it's >10V
        self.start_time = time.time()
        self.timeout = timeout
        self.load_and_arm_NIDAQ()

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
            self.stop_nidaq()

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

    p = PyStim(hdw=["PA5", "NIDAQ", "RP21"], devicename="dev1")

    ni_sampld_frequency = 100000
    w = np.cos(2 * np.pi * 2000.0 * np.arange(0, 0.2, 1.0 / ni_sample_frequency))
    p.setAttens(atten_left=30)
    p._present_stim(w)
    time.sleep(2.0)
    # p.RZ5D.setModeStr("Idle")
    # p.task.stop()
    p.setAttens()
