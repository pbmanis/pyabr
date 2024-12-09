#!/usr/bin/env python
"""
pyabr : a python program for ABR recordings. 
Parameter setup is defined in config/.cfg files (text files)

This program uses pystim for hardware control,
and sound.py for stimulus generation. 
"""

import atexit
import datetime
import copy
import pickle
import pprint
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import scipy.signal
from pyqtgraph import configfile
from pyqtgraph import dockarea as PGD
from pyqtgraph import reload as reload
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import QObject, pyqtSignal, pyqtSlot

from src import convert_nested_ordered_dict as convert_nested_ordered_dict
from src import presenter_thread as presenter_thread
from src import protocol_reader as protocol_reader
from src import pystim3 as pystim3
from src import wave_generator as wave_generator
from src import sound as sound  # for waveform generation
from src.build_parametertree import build_parametertree
import src.read_calibration as read_calibration
import src.read_abr as read_abr
import src.message_box as MBox

MAX_CLICK = 105.1  # SPL for maximum click

THREAD_PERIOD = 20  # thread period, in msec





class PyABR(QtCore.QObject):

    signal_pause = pyqtSignal()
    signal_resume = pyqtSignal()
    signal_stop = pyqtSignal()
    signal_quit = pyqtSignal()

    def __init__(self, configtype: str = "abr"):
        assert configtype in ["test", "abr"]
        if configtype == "test":
            configfilename = "config/abrs_test.cfg"
        else:
            configfilename = "config/abrs.cfg"
        self.QColor = QtGui.QColor
        atexit.register(self.quit)
        # get the configuration file:
        self.config = configfile.readConfigFile(configfilename)
        self.PS = pystim3.PyStim(
            required_hardware=self.config["required_hardware"],
            ni_devicename=self.config["NI_device"],
        )
        # check hardware
        self.hardware, self.sfin, self.sfout = self.PS.getHardware()
        self.calfile = Path(self.config["calfile"])  # get calibration file
        self.caldata = read_calibration.get_calibration_data(self.calfile)
        now = datetime.datetime.now()
        self.WG = wave_generator.WaveGenerator(caldata=self.caldata)
        self.Presenter = None  # the thread object that will be used for presenting stimuli
        self.wave_matrix: dict = {}
        self.ptreedata = None  # pyqtgraph parametertree  - set to None until the tree is built
        self.protocol = None
        self.debugFlag = True
        self.link_traces = True
        self.raw_clear_N = 10
        self.read_protocol_directory()
        self.acq_mode = "test"
        self.basefn = None  # base file path/name for saving data
        self.buildGUI()
        self.PR.update_protocol(self.ptreedata)
        self.update_devices(devices=self.hardware)
        self.counter = 0
        self.ch1 = None
        self.ch2 = None
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        super(PyABR, self).__init__()

        self.TrialTimer = pg.QtCore.QTimer()
        self.TrialTimer.timeout.connect(self.recurring_timer)
        self.TrialCounter = 1

        self.threadpool = QtCore.QThreadPool()
        self.Presenter = presenter_thread.Presenter(parameters=self)
        # self.TrialTimer.timeout.connect(self.Presenter.next_stimulus)
        # connect signals FROM the Presenter thread
        self.Presenter.signal_data_ready.connect(self.update_ABR)
        self.Presenter.signal_error.connect(self.thread_error)
        self.Presenter.signal_finished.connect(self.finished)
        self.Presenter.signal_progress.connect(self.update_progress)
        self.Presenter.signal_repetitions_done.connect(self.clear_average)
        self.Presenter.signal_mode_status.connect(self.update_mode)
        self.signal_pause.connect(self.Presenter.pause)
        self.signal_resume.connect(self.Presenter.resume)
        self.signal_stop.connect(self.Presenter.stop)
        self.signal_stop.connect(self.PS.cleanup_NIDAQ_RP21)
        self.signal_quit.connect(self.Presenter.quit)

        # note after building the GUI, everything is handled througn callbacks
        # in the command_dispatcher function.

    def read_protocol_directory(self):
        """read_protocol_directory Get the protocols from the protocols directory
        Chose the default (current) protocol, then read it.
        This must be called before the GUI is built, but can be called afterwards
        to update the list of available prototols.
        """
        self.known_protocols = [p.name for p in Path("protocols").glob("*.cfg")]
        self.current_protocol = self.known_protocols[0]
        self.PR = protocol_reader.ProtocolReader(ptreedata=self.ptreedata)
        self.PR.read_protocol(protocolname=self.current_protocol)
        self.stim = self.PR.get_current_protocol()

    def buildGUI(self):
        """Build GUI and window"""

        white = self.QColor(255, 255, 255)
        black = self.QColor(0, 0, 0)
        red = self.QColor(255, 0, 0)
        # Define the table style for various parts dark scheme
        dark_palette = QtGui.QPalette()
        dark_palette.setColor(QtGui.QPalette.ColorRole.Window, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Base, self.QColor(25, 25, 25))
        dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Text, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Button, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, red)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Link, self.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, self.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, self.QColor(0, 255, 0))

        self.app = pg.mkQApp()
        self.app.setStyle("fusion")
        self.app.setPalette(dark_palette)
        self.app.setStyleSheet(
            "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
        )
        ptreewidth = 250
        right_docks_width = 1024 - ptreewidth - 20
        right_docs_height = 800
        self.win = pg.QtWidgets.QMainWindow()

        self.dockArea = PGD.DockArea()
        self.win.setCentralWidget(self.dockArea)
        self.Dock_Params = PGD.Dock("Parameters", size=(ptreewidth, 1024))
        self.Dock_Recording = PGD.Dock("Recording", size=(right_docks_width, 1024))
        self.Dock_Preview = PGD.Dock("Preview", size=(right_docks_width, 1024))
        self.dockArea.addDock(self.Dock_Params, "left")
        self.dockArea.addDock(self.Dock_Recording, "right", self.Dock_Params)
        self.dockArea.addDock(self.Dock_Preview, "below", self.Dock_Recording)
        self.win.setWindowTitle("ABR Acquistion")
        self.win.setGeometry(100, 100, 1200, 1000)
        # print(dir(self.dockArea))
        self.Dock_Params.raise_()
        self.ptree, self.ptreedata = build_parametertree(
            self.known_protocols, self.current_protocol, stimuli=self.stim["stimuli"]
        )
        self.Dock_Params.addWidget(self.ptree)

        self.build_recording_graphs()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        self.win.show()

    def build_recording_graphs(self):
        # add space for the graphs
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(50, 50, 50))
        view.setCentralItem(layout)

        self.plot_ABR_Raw = layout.addPlot()
        self.plot_ABR_Raw.getAxis("left").setLabel("uV", color="#ff0000")
        self.plot_ABR_Raw.setTitle("Raw ABR Signal", color="#ff0000")
        self.plot_ABR_Raw.getAxis("bottom").setLabel("t (msec)", color="#ff0000")
        # self.plot_ABR_Raw.setYRange(-10.0, 10.0)  # microvolts

        layout.nextRow()  # averaged abr trace
        self.plot_ABR_Average = layout.addPlot()  #
        if self.link_traces:
            self.plot_ABR_Average.setXLink(self.plot_ABR_Raw)
        self.plot_ABR_Average.setTitle("Average ABR Signal", color="#ff0000")
        self.plot_ABR_Average.getAxis("left").setLabel("uV", color="#0000ff")
        # self.plot_ABR_Average.setYRange(-10, 10.0)
        self.plot_ABR_Average.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        layout.nextRow()  # waveforms
        self.stimulus_waveform = layout.addPlot()  #
        if self.link_traces:
            self.stimulus_waveform.setXLink(self.plot_ABR_Raw)
        self.stimulus_waveform.setTitle("Waveform", color="#ff0000")
        self.stimulus_waveform.getAxis("left").setLabel("V", color="#0000ff")
        # self.stimulus_waveform.setYRange(-10, 10.0)
        self.stimulus_waveform.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        layout.nextRow()  #
        l2 = layout.addLayout(colspan=3, border=(50, 0, 0))  # embed a new layout
        l2.setContentsMargins(2, 2, 2, 2)
        self.plot_Amplitude_Level = l2.addPlot(Title="Amplitude-Intensity")
        self.plot_Amplitude_Level.getAxis("bottom").setLabel("dB (SPL)")
        self.plot_Amplitude_Level.getAxis("left").setLabel("uV")

        self.plt_map = l2.addPlot(Title="Map")
        self.plt_map.getAxis("bottom").setLabel("F (kHz)")
        self.plt_map.getAxis("left").setLabel("dB (SPL)")

        self.Dock_Recording.addWidget(view)

        
        view2 = pg.GraphicsView()
        layout2 = pg.GraphicsLayout(border=(50, 50, 50))
        view2.setCentralItem(layout2)
        self.cal_plot = layout2.addPlot()
        self.Dock_Preview.addWidget(view2)
        self.Dock_Recording.raise_()

    def command_dispatcher(self, param, changes):
        """command_dispatcher callback for parametertree
        This function is called when the parametertree is changed, including
        when the user clicks on a button or changes a value.

        Parameters
        ----------
        param : parametertree parameter
            parametertree parameter that was changed
        changes : This holds all the real changes
            _description_
        """

        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            match path[0]:
                case "Quit":
                    self.quit(atexit=False)

                case "Protocol":
                    self.current_protocol = data
                    self.PR.read_protocol(data, update=False)
                case "Actions":
                    match path[1]:
                        case "New Filename":
                            self.new_filename()
                        case "Test Acquisition":
                            self.acquire(mode="test")
                        case "Start Acquisition":
                            self.acquire(mode="collect")
                        case "Stop":
                            self.stop()
                        case "Pause":
                            self.pause()
                        case "Resume":
                            self.continue_trials()
                        case "Save Visible":
                            self.save_visible()
                        case "Load Data File":
                            self.load_file()
                        case "Read Cal File":
                            self.load_cal_file()

                case "Status":
                    match path[1]:
                        case "dBSPL":
                            self.protocol["stimuli"]["default_spl"] = data
                        case "Freq (kHz)":
                            self.protocol["stimuli"]["default_frequency"] = data
        self.ptreedata.sigTreeStateChanged.disconnect(self.command_dispatcher)
        self.PR.update_protocol(ptreedata=self.ptreedata)
        self.protocol = self.PR.get_current_protocol()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

    def quit(self, atexit=True):
        self.TrialTimer.stop()
        self.signal_quit.emit()
        if self.Presenter is not None:
            self.Presenter.end_thread()
            self.threadpool.waitForDone(5 * THREAD_PERIOD)  # end thread'
        self.win.close()
        if not atexit:
            exit()

    def thread_error(self, data):
        print("Thread Error: ", data)
        self.quit()

    def new_filename(self):
        """
        Create a new filename for the data file
        """

        self.protocol["filename"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ptreedata.child("Parameters").setValue(self.protocol)
        print("new file: ", self.protocol["filename"])

    def save_visible(self):
        """
        Save the visible data to a file
        """
        print("Save Visible data set")
        self.new_filename()
        self.ptreedata.child("Parameters").setValue(self.protocol)
        with open(self.protocol["filename"] + ".pkl", "wb") as fh:
            pickle.dump(
                {
                    "abr": self.abr,
                    "abr_avg": self.abr_avg,
                    "waveform": self.stimulus_waveform,
                    "AI": self.AI,
                    "map": self.map,
                },
                fh,
            )

    def load_file(self):
        """
        Load a file
        """
        print("Load File")
        filename = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open File", ".", "Pickle Files (*.pkl)"
        )
        if filename[0]:
            with open(filename[0], "rb") as fh:
                data = pickle.load(fh)
                self.abr = data["abr"]
                self.abr_avg = data["abr_avg"]
                self.stimulus_wavedata = data["waveform"]
                self.AI = data["AI"]
                self.map = data["map"]
                self.update_plots()


    def load_cal_file(self):
        """
        Load a calibration file
        """
        calfile = self.config["calfile"]
        self.caldata = read_calibration.get_calibration_data(calfile)
        read_calibration.plot_calibration(self.caldata)

    def plot_stimulus_wave(self, n: int = 0): # , n_sample_delay:int=65):
        first_sound = self.wave_matrix[list(self.wave_matrix.keys())[n]]
        # samp_delay = n_sample_delay/first_sound["rate"]
        # print("nsam del: ", n_sample_delay, "samp_delay: ", samp_delay)
        t = np.arange(0, len(first_sound["sound"]) / float(first_sound["rate"]), 1.0 / first_sound["rate"])
        if len(t) > len(first_sound["sound"]):
            t = t[:len(first_sound["sound"])]
        t = t + 750e-6  # measured delay added in circuit in pystim3.abr
        if (self.raw_clear_N > 0) and ((n % self.raw_clear_N) == 0):
            self.stimulus_waveform.clear()
            self.stimulus_waveform.setXRange(0, np.max(t))
        self.stimulus_waveform.plot(
            t,
            first_sound["sound"],
            pen=pg.mkPen("y"),
        )

        # self.stimulus_waveform.autoRange()

    def update_plots(self):
        pass

    def finished(self):
        """
        Called when the Presenter thread has finished with valid data
        Nominally, this might automatically save the data to a file
        """
        pass

    def update_progress(
        self,
        wavetype: str = "",
        wave_count: int = 0,
        max_wave_count: int = 0,
        rep_count: int = 0,
        max_rep_count: int = 0,
        db: float = 0.0,
        fr: float = 0.0,
    ):
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Status":
                for childs in child:
                    if wavetype in ["click", "tonepip"] and childs.name() == "dBSPL":
                        childs.setValue(f"{db:5.1f}")
                    if wavetype in ["tonepip"] and childs.name() == "Freq (kHz)":
                        childs.setValue(f"{fr:8.1f}")
                    if childs.name() == "Wave #":
                        childs.setValue(f"{wave_count+1:d} / {max_wave_count:d}")
                    if childs.name() == "Rep #":
                        childs.setValue(f"{rep_count:d} / {max_rep_count:d}")

    def update_mode(self, mode: str):
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Status":
                for childs in child:
                    if childs.name() == "Mode":
                        childs.setValue(f"{mode:s}")

    def update_devices(self, devices: str):
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Status":
                for childs in child:
                    if childs.name() == "Devices":
                        childs.setValue(f"{devices!s}")

    def recurring_timer(self):
        self.counter += 1  # nothing really to do here.
        time.sleep(0.01)

    def make_waveforms(self, wavetype, dbspls, frequencies):
        """
        thin wrapper for the wavegenerator waveofmr maker.
        """
        self.WG.setup(protocol=self.protocol, frequency=self.PS.Stimulus.NI_out_sampleFreq, config=self.config)  # output rate
        self.WG.make_waveforms(wavetype=wavetype, dbspls=dbspls, frequencies=frequencies)
        self.wave_matrix = self.WG.wave_matrix

    def make_and_plot(self, n: int, wavetype: str, dbspls: list, frequencies: list):
        self.make_waveforms(wavetype=wavetype, dbspls=dbspls, frequencies=frequencies)
        # for k in list(self.wave_matrix.keys()):
        #     print(k, np.max(self.wave_matrix[k]["sound"]))
        self.plot_stimulus_wave(n)

    def acquire(self, mode: str = "test"):
        """
        present and collect responses without saving - using
        abbreviated protocol
        Here, we just set things up, and display the stimulus
        that will be presented.

        Then we can start the player.
        In test mode, we do not save the data....

        """
        self.acq_mode = mode
        self.Dock_Recording.raiseDock()
        self.app.processEvents()
        self.PS = pystim3.PyStim(
            required_hardware=self.config["required_hardware"],
            ni_devicename=self.config["NI_device"],
        )
        self.PS.reset_hardware()
        self.protocol = self.PR.get_current_protocol()  # be sure we have current protocol data
        self.TrialCounter = 1
        flist = self.protocol["stimuli"]["freqlist"]
        if self.protocol["protocol"]["stimulustype"] == "click":
            if len(flist) == 0:
                flist = [0] * len(self.protocol["stimuli"]["dblist"])
        self.make_and_plot(
            n=0,
            wavetype=self.protocol["protocol"]["stimulustype"],
            dbspls=self.protocol["stimuli"]["dblist"],
            frequencies=flist,
        )
        self.threadpool.start(self.Presenter.run)  # start the thread.
        if mode == "test":
            print("Starting acquisition in test mode")
        self.stimulus_waveform.enableAutoRange()
        self.Dock_Recording.raiseDock()
        self.Dock_Recording.update()
        self.TrialTimer.setInterval(int(self.protocol["stimuli"]["stimulus_period"]))
        self.TrialTimer.timeout.emit()
        self.TrialTimer.start()
        # finally, load up the data and trigger the thread to start.
        self.Presenter.setWaveforms(  # new instance of pystim3 each time
            wave=self.wave_matrix, protocol=self.protocol, sound_presentation_class=self.PS
        )

    def update_ABR(self, chdata, wave_counter, repetition_counter):
        """update_ABR Plot the most recent abr data, and keep
        track of the running average. 
        *** this function is called from the Presenter thread ***

        Parameters
        ----------
        chdata : np.array
            The data from the ABR acquisition
        wave_counter : int
            index into which wave was played
        repetition_counter : _type_
            _description_
        """
        if chdata[0] is None:  # no data, so fake it
            rng = np.random.default_rng()
            chdata = [
                rng.standard_normal(self.PS.Ndata) * 1e-6,
                rng.standard_normal(self.PS.Ndata) * 1e-6,
            ]
        self.ch1_data = np.array(chdata[0])
        # self.ch2_data = np.array(ch2[0]).T
        if self.TrialCounter == 1:
            self.summed_buffer = self.ch1_data
            self.plot_ABR_Raw.clear()
            self.plot_ABR_Average.clear()
        else:
            self.summed_buffer = self.summed_buffer + self.ch1_data
        
        _, self.sfin, self.sfout = self.PS.getHardware()
        self.t_record = np.linspace(0, len(self.ch1_data) / self.sfin, self.ch1_data.shape[0])
        if (self.raw_clear_N > 0) and (((self.TrialCounter-1) % self.raw_clear_N) == 0):
            self.plot_ABR_Raw.clear()
            self.plot_ABR_Raw.setXRange(0, np.max(self.t_record))

        self.plot_ABR_Raw.plot(self.t_record, self.ch1_data, pen=pg.mkPen(pg.intColor(self.TrialCounter-1, hues=10, values=10)))
        self.plot_ABR_Raw.setYRange(-15, 15)
 
        self.plot_ABR_Average.plot(
            self.t_record, self.summed_buffer / float(self.TrialCounter), pen=pg.mkPen("g")
        )
        self.TrialCounter += 1    

        if self.acq_mode == "test":
            return
        subject_data = {}
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Subject Data":
                for childs in child:
                    subject_data[childs.name()] = childs.value()
        # now assemble data and save it
        write_time = datetime.datetime.now()
        wave_copy = copy.deepcopy(self.wave_matrix)
        for k in wave_copy.keys():
            wave_copy[k]["sound"] = {}
        out_data = {
            "subject_data": subject_data,
            "calibration": self.caldata,
            "stimuli": wave_copy,
            "protocol": self.protocol,
            "wave_number": wave_counter,
            "repetition": repetition_counter,
            "data": self.ch1_data,
            "record_frequency": self.PS.trueFreq,
            "timestamp": write_time,
        }
        data_directory = Path(self.config["datapath"])
        if not data_directory.is_dir():
            Path.mkdir(data_directory, exist_ok=True)  # create if it does not exist.
        if self.basefn is None:
            self.basefn = write_time.strftime("%Y-%m-%d")
        # print("base fn", self.basefn)
        subject_data["Subject ID"] = "testing"
        # print("data dir: ", data_directory)
        # print("subject: ", subject_data['Subject ID'])
        self.subject_dir = Path(self.config["datapath"], self.basefn, subject_data["Subject ID"])
        # Path.mkdir(self.subject_dir, exist_ok = True)
        fn = Path(
            f"abr_data/{self.basefn:s}_{self.protocol['protocol']['stimulustype']:s}_{wave_counter:03d}_{repetition_counter:03d}"
        )
        fn = str(fn) + ".p"
        print("output file: ", fn)
        with open(fn, "wb") as fh:
            pickle.dump(out_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        del wave_copy    # delete the wave copy

    def clear_average(self):
        self.plot_ABR_Average.clear()

    def stop(self):
        """stop:  Stop stimulus presentation dead. 
        *** This function sends a signal to the Presenter thread
        to cleanly stop the stimulus presentation. The thread
        continues to run in the background (doing nothing..).
        """
        self.signal_stop.emit()  # this calls the cleanup directly, so don't call here
        self.TrialTimer.stop()

    def pause(self):
        """pause:  Pause the stimulus presentation.
        *** This function sends a signal to the Presenter thread
        to pause the stimulus presentation. The thread
        continues to run in the background (doing nothing..).
        It can be resumed with the continue_trials function.
        """
        self.signal_pause.emit()
        self.TrialTimer.stop()

    def continue_trials(self):
        """continue_trials:  resume the stimulus presentation when it
        is paused. If the presentation is not paused, this function
        does nothing. The timing picks up where it left off.

        """
        if self.Presenter._paused:
            self.Presenter.resume()
            self.TrialTimer.start()


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1:]
    if len(cmd) > 0:
        print("cmd: ", cmd)
        prog = PyABR(cmd[0])
    else:
        prog = PyABR()
    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()
