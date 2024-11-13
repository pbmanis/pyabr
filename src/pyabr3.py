#!/usr/bin/env python
"""
pyabr : a python program for ABR recordings. 
Parameter setup is defined in .toml files (text files),
and displayed in a window.

"""

import atexit
import datetime
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
from src import protocol_reader as protocol_reader
from src import pystim as pystim
from src import sound as sound  # for waveform generation
from src.build_parametertree import build_parametertree
import src.read_calibration as read_calibration

THREAD_PERIOD = 20  # thread period, in msec
MAX_CLICK = 105.1  # SPL for maximum click 

class Worker(QObject):  # (QtCore.QRunnable):
    """
    Worker thread class
    Inherits from QObject to handle worker thread setup and signals.

    Sent signals are:

        finished
            No data
        error
            tuple (exctype, value, traceback.format_exc() )
        result
            object data returned from processing, anything
        mode_status
            status of the flags
        data_ready: a signal with the result data from the A/D converter
        progress:
            a signal with the current stimulus information.


    received signals/slots:
        setWaveforms
        Pause
        Resume
        Stop


    """

    # define the signals.

    signal_finished = pyqtSignal()  # protocol is ALL done
    signal_error = pyqtSignal(tuple)  # protocol has errored
    signal_data_ready = pyqtSignal(np.ndarray, int, int)  #  returns 2 channels in the array, then the wave number and trial counter
    signal_paused = pyqtSignal()  # status of running: paused or not
    signal_stop = pyqtSignal()  # stop protocol entirely

    signal_progress = pyqtSignal(
        str, int, int, int, int, float, float
    )  # update display of progress. Pass name, wave count, nwaves, rep count, nreps, db and fr
    signal_mode_status = pyqtSignal(str)  # notify changes in mode

    def __init__(self, parameters=None):  # fn, *args, **kwargs):

        super(Worker, self).__init__()

        # Flags
        self._finished = False
        self._error = False
        self._paused = False
        self._running = False
        self._quit = False

        self._waveloaded = False
        self.wavedata = None
        self.wavetype = None
        self.protocol = None

        self.parameters = parameters
        self.wave_counter = 0
        self.repetiton_counter = 0

    @pyqtSlot()
    def run(self):
        """
        Initialize the runner function with passed args, kwargs.

        Within this function, we present successive stimuli.
        When the stimulus list is exhausted, we set the
        _finished signal and flag and return

        Calling the slot "setWaveforms" sets up and starts the acquisiton sequence.

        """

        # Retrieve args/kwargs here; and fire processing using them
        self._running = True
        self.wave_counter = 0
        while True:  # while running, keep processing
            time.sleep(float(THREAD_PERIOD / 1000.0))  # Short delay to prevent excessive CPU usage
            # State management:
            # print(self._quit, self._running, self._finished, self._paused, self._waveloaded)
            # time.sleep(0.2)
            if self._quit:
                return  # the final return (ends the thread)
            if not self._running:
                self.signal_mode_status.emit("Waiting to start")
                continue
            if self._finished:
                self.signal_mode_status.emit("Finished")
            if self._paused:
                self.signal_mode_status.emit("Paused")
            if not self._waveloaded:
                self.signal_mode_status.emit("No wave loaded")
            # running
            if self._running:
                self.signal_mode_status.emit("Running")
                self.next_stimulus()
                time.sleep(self.protocol["stimuli"]["stimulus_period"])

    def retrieve_data(self):
        res = self.sound_func.retrieveRP21_inputs()
        self.chdata = np.array(res)
        # self.ch2_data = np.array(res)
        self.signal_data_ready.emit(self.chdata, self.wave_counter, self.repetition_counter)

    def next_stimulus(self):
        """
        Play the next stimulus in the sequence and collect data.
        for nreps. When done, return the acquired data.

        This is done for each entry in the wave_matrix (wave_counter),
        and each entry is repeated n_repetition times.
        Each repetition is returned via a signal to the main program.

        """
        print(
            f"next stim, wave count: {self.wave_counter:d}/{self.n_waves} repetition count: {self.repetiton_counter:d}/{self.n_repetitions:d}"
        )
        print("    self._running: ", self._running)
        if not self._running or self.wavetype is None or not self._waveloaded:
            print("? missing something")
            return None, None

        if self.repetition_counter >= self.n_repetitions:
            self.wave_counter += 1
            self.repetition_counter = 0  # reset reps
        if self.wave_counter >= self.n_waves:
            self.signal_finished.emit()
            self._finished = True
            return
        self.repetition_counter += 1
        if len(self.wavekeys[0]) > 2:
            sec_key = self.wavekeys[self.wave_counter][2]
        else:
            sec_key = 0.0
        self.signal_progress.emit(
                self.wavetype,
                self.wave_counter,
                self.n_waves,
                self.repetition_counter,
                self.n_repetitions,
                self.wavekeys[self.wave_counter][1],
                sec_key,
            )
        print(self.wave_counter, self.wavekeys[self.wave_counter], self.repetition_counter, "/", self.n_repetitions)
        
        if self.wavetype in ["click", "tonepip"]:
            wave = self.wave_matrix[self.wavekeys[self.wave_counter]]["sound"]
            sfout = self.wave_matrix[self.wavekeys[self.wave_counter]]["rate"]
            if self.wave_counter >= self.n_waves:
                self.retrieve_data()
                self.signal_finished.emit()
                self._finished = True
                self._running = False
                return

            listed_attn = self.wavekeys[self.wave_counter][1]
            if self.wavetype in ["click"]:
                attn = MAX_CLICK - listed_attn
            else:
                attn = listed_attn # for tones, just use the attenuation
            
            self.sound_func.play_sound(
                wave,
                wave,
                samplefreq=sfout,
                attns=[attn, attn],
            )
            self.retrieve_data()

        else:  # other stimuli.
            listed_attn = self.wavekeys[self.wave_counter][1]
            wave = self.wave_matrix[self.wavekeys[0]]["sound"]
            sfout = self.wave_matrix[self.wavekeys[0]]["rate"]
            self.sound_func.play_sound(wave, wave, samplefreq=sfout, attns=[listed_attn, listed_attn])

            self.retrieve_data()

        return

    @pyqtSlot()
    def setWaveforms(self, wave: dict, protocol: dict, sound_function: object):
        # get the data we need and trigger the stimulus/acquisition.
        print("set waveforms")
        self._running = False
        self.sound_func = sound_function
        self.wave_matrix = wave
        self.protocol = protocol
        self.wavekeys = list(self.wave_matrix.keys())
        self.wave_counter = 0  # reset counter of the waveforms
        self.repetition_counter = 0
        self.wavetype = self.wavekeys[0][0]  # get the first key of the first waveform.
        self.n_waves = len(self.wavekeys)
        # print(self.protocol['stimuli'])
        self.n_repetitions = self.protocol["stimuli"]["nreps"]
        # print("nwaves, nreps: ", self.n_waves, self.n_repetitions)
        self._waveloaded = True
        self._running = True
        print("waveforms set")

    @pyqtSlot()
    def pause(self):
        # pause the thread
        self._paused = True  # set False to block the thread

    @pyqtSlot()
    def resume(self):
        # resume stimulation
        if self._running:
            self._paused = False

    @pyqtSlot()
    def stop(self):
        # stops stimulation, and
        self._running = False  # stops running
        self._paused = False  # also kills ability to resume.


    @pyqtSlot()
    def finished(self):
        self._finished = True
        self.signal_finished.emit()  # send a signal

    @pyqtSlot()
    def end_thread(self):
        """
        Stop the thread from running.
        This is needed at the end of the program to terminate cleanly.
        """
        self._paused = False  # resume if paused
        self._running = False  # set running False
        self.signal_finished.emit()  # send a signal

    @pyqtSlot()
    def quit(self):
        self._running = False
        self._quit = True
        self.signal_finished.emit()  # send a signal


class PyABR(QtCore.QObject):

    signal_pause = pyqtSignal()
    signal_resume = pyqtSignal()
    signal_stop = pyqtSignal()
    signal_quit = pyqtSignal()

    def __init__(self):
        self.QColor = QtGui.QColor
        atexit.register(self.quit)
        self.PS = pystim.PyStim(required_hardware=["PA5", "RP21", "NIDAQ"])  # PySounds.PySounds()

        # check hardware first
        self.hw, self.sfin, self.sfout = self.PS.getHardware()
        #
        # get the latest calibration file:
        self.config = configfile.readConfigFile("config/abrs.cfg")

        self.calfile = Path(self.config["calfile"])
        self.caldata = read_calibration.get_calibration_data(self.calfile)
        self.Presenter = None
        self.wave_matrix: dict = {}
        self.ptreedata = None  # until the tree is built
        self.protocol = None
        self.debugFlag = True
        self.known_protocols = [p.name for p in Path("protocols").glob("*.cfg")]
        self.current_protocol = self.known_protocols[0]
        self.PR = protocol_reader.ProtocolReader(ptreedata=self.ptreedata)
        self.PR.read_protocol(protocolname=self.current_protocol)
        self.stim = self.PR.get_current_protocol()
        self.acq_mode = "test"
        self.basefn = None  # base file path/name for saving data
        self.buildGUI()

        self.PR.update_protocol(self.ptreedata)
        self.counter = 0
        self.ch1 = None
        self.ch2 = None
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        super(PyABR, self).__init__()

        self.TrialTimer = pg.QtCore.QTimer()

        self.TrialTimer.timeout.connect(self.recurring_timer)

        self.TrialCounter = 0

        self.threadpool = QtCore.QThreadPool()
        self.Presenter = Worker(parameters=self)
        # self.TrialTimer.timeout.connect(self.Presenter.next_stimulus)
        # connect signals FROM the Presenter thread
        self.Presenter.signal_data_ready.connect(self.update_ABR)
        self.Presenter.signal_error.connect(self.thread_error)
        self.Presenter.signal_finished.connect(self.finished)
        self.Presenter.signal_progress.connect(self.update_progress)
        self.Presenter.signal_mode_status.connect(self.update_mode)
        self.signal_pause.connect(self.Presenter.pause)
        self.signal_resume.connect(self.Presenter.resume)
        self.signal_stop.connect(self.Presenter.stop)
        self.signal_stop.connect(self.PS.cleanup_NIDAQ)
        self.signal_quit.connect(self.Presenter.quit)

        # note after buildgui, everything is done with callbacks from the GUI

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

        self.build_graphs()

    def build_graphs(self):
        # add space for the graphs
        view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(50, 50, 50))
        view.setCentralItem(l)

        self.plot_ABR_Raw = l.addPlot()
        self.plot_ABR_Raw.getAxis("left").setLabel("uV", color="#ff0000")
        self.plot_ABR_Raw.setTitle("Raw ABR Signal", color="#ff0000")
        self.plot_ABR_Raw.getAxis("bottom").setLabel("t (msec)", color="#ff0000")
        # self.plot_ABR_Raw.setYRange(-10.0, 10.0)  # microvolts

        l.nextRow()  # averaged abr trace
        self.plot_ABR_Average = l.addPlot()  #
        # self.plot_ABR_Average.setXLink(self.plot_ABR_Raw)
        self.plot_ABR_Average.setTitle("Average ABR Signal", color="#ff0000")
        self.plot_ABR_Average.getAxis("left").setLabel("uV", color="#0000ff")
        # self.plot_ABR_Average.setYRange(-10, 10.0)
        self.plot_ABR_Average.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        l.nextRow()  # waveforms
        self.stimulus_waveform = l.addPlot()  #
        # self.stimulus_waveform.setXLink(self.plot_ABR_Raw)
        self.stimulus_waveform.setTitle("Waveform", color="#ff0000")
        self.stimulus_waveform.getAxis("left").setLabel("V", color="#0000ff")
        # self.stimulus_waveform.setYRange(-10, 10.0)
        self.stimulus_waveform.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        l.nextRow()  #
        l2 = l.addLayout(colspan=3, border=(50, 0, 0))  # embed a new layout
        l2.setContentsMargins(2, 2, 2, 2)
        self.plot_Amplitude_Level = l2.addPlot(Title="Amplitude-Intensity")
        self.plot_Amplitude_Level.getAxis("bottom").setLabel("dB (SPL)")
        self.plot_Amplitude_Level.getAxis("left").setLabel("uV")

        self.plt_map = l2.addPlot(Title="Map")
        self.plt_map.getAxis("bottom").setLabel("F (kHz)")
        self.plt_map.getAxis("left").setLabel("dB (SPL)")
        self.Dock_Recording.addWidget(view)
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        self.win.show()

    def command_dispatcher(self, param, changes):
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            # print("path: ", path)
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
                        case "Load File":
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
        print("Save Visible")
        self.protocol["filename"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        print("Load Cal File")
        calfile = self.config["calfile"]
        self.caldata = read_calibration.get_calibration_data(calfile)
        read_calibration.plot_calibration(self.caldata)


    def plot_stimulus_wave(self, n: int = 0):
        first_sound = self.wave_matrix[list(self.wave_matrix.keys())[n]]
        # print(first_sound['rate'])
        t = np.arange(0, len(first_sound["sound"]) / first_sound["rate"], 1.0 / first_sound["rate"])
        # print("np.max t: ", np.max(t))
        self.stimulus_waveform.clear()
        self.stimulus_waveform.plot(
            t,
            first_sound["sound"],
            pen=pg.mkPen("y"),
        )
        self.stimulus_waveform.setXRange(0, np.max(t))
        # self.stimulus_waveform.autoRange()

    def update_plots(self):
        pass

    def finished(self):
        """
        Called when the Presenter thread has finished with valid data
        """
        # self.data = self.Presenter.getWaveforms()
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
        # print("update progress called")
        children = self.ptreedata.children()
        # print(dir(children))
        for child in children:
            # print(dir(child))
            if child.name() == "Status":
                # print("Found child: ", child.name())
                for childs in child:
                    if wavetype in ["click", "tonepip"] and childs.name() == "dBSPL":
                        childs.setValue(f"{db:5.1f}")
                    if wavetype in ["tonepip"] and childs.name() == "Freq (kHz)":
                        childs.setValue(f"{fr:8.1f}")
                    if childs.name() == "Wave #":
                        childs.setValue(f"{wave_count:d} / {max_wave_count:d}")
                    if childs.name() == "Rep #":
                        childs.setValue(f"{rep_count:d} / {max_rep_count:d}")

    def update_mode(self, mode: str):
        # print("update mode called with: ", mode)
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Status":
                for childs in child:
                    if childs.name() == "Mode":
                        childs.setValue(f"{mode:s}")

    def recurring_timer(self):
        self.counter += 1  # nothing really to do here.
        time.sleep(0.01)

    def make_waveforms(self, wavetype: str, dbspl=None, frequency=None):
        """
        Generate all the waveforms we will need for this protocol.
        Waveforms, held in self.wave_matrix,
        are in a N x (wave) shape, where N is the number
        of different stimuli. Waveforms are played out in the order
        they appear in this array.
        """
        self.wave_matrix = {}
        if dbspl is None:  # get the list?
            dbspl = self.protocol["stimuli"]["dblist"]
        if dbspl is None:  # still None ? use the default
            dbspl = [self.protocol["stimuli"]["default_spl"]]
        if frequency is None:
            frequency = self.protocol["stimuli"]["frlist"]
        if frequency is None:
            frequency = [self.protocol["stimuli"]["default_frequency"]]
        print("WAVETYPE: ", wavetype)
        match wavetype:
            case "click":
                print("doing click")

                freqs = [0] * len(dbspl)  # set to all zeros
                starts = np.cumsum(
                    np.ones(self.protocol["stimuli"]["nstim"])
                    * self.protocol["stimuli"]["interval"]
                )
                starts += self.protocol["stimuli"]["delay"]
                for i, db in enumerate(dbspl):
                    wave = sound.ClickTrain(
                        rate=self.sfout,
                        duration=self.protocol["stimuli"]["wave_duration"],
                        dbspl=self.config["reference_dbspl"],
                        click_duration=self.protocol["stimuli"]["stimulus_duration"],
                        click_starts=starts,
                        alternate=self.protocol["stimuli"]["alternate"],
                    )
                    wave.generate()
                    self.wave_matrix[("click", db, freqs[i])] = {
                        "sound": wave.sound,
                        "rate": self.sfout,
                    }
                    self.wave_time = wave.time
                self.nwaves = len(dbspl)

            case "tonepip":

                starts = np.cumsum(
                    np.ones(self.protocol["stimuli"]["nstim"])
                    * self.protocol["stimuli"]["interval"]
                )
                starts += self.protocol["stimuli"]["delay"]
                # print("tonepip ", frequency, dbspl)
                dbref = 100.0
                for i, db in enumerate(dbspl):
                    for j, fr in enumerate(frequency):
                        wave = sound.TonePip(
                            rate=self.sfout,
                            duration=self.protocol["stimuli"]["wave_duration"],
                            f0=fr,
                            dbspl=dbref,
                            pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                            ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                            pip_start=starts,
                            alternate=self.protocol["stimuli"]["alternate"],
                        )
                        # print("tonepip generate: ", db, fr)
                        wave.generate()
                        self.wave_matrix[("tonepip", db, fr)] = {
                            "sound": wave.sound,
                            "rate": self.sfout,
                        }
                        self.wave_time = wave.time
                self.nwaves = len(dbspl) * len(frequency)

            case "interleaved_plateau":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                if frequency is None:
                    frequency = self.protocol["stimuli"]["default_frequency"]
                dt = self.protocol["stimuli"]["stimulus_period"]
                s0 = self.protocol["stimuli"]["delay"]
                dbs = eval(self.protocol["stimuli"]["dblist"])
                freqs = eval(self.protocol["stimuli"]["freqlist"])
                wave_duration = s0 + len(dbs) * len(freqs) * dt + dt  # duration of the waveform
                self.dblist = []
                self.freqlist = []
                n = 0
                dbref = 100.
                for j, dbspl in enumerate(dbs):
                    for i, frequency in enumerate(freqs):
                        wave_n = sound.TonePip(
                            rate=self.sfout,
                            duration=wave_duration,
                            f0=frequency,
                            dbspl=dbspl,
                            pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                            ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                            pip_start=[s0 + n * dt],
                            alternate=False,  # do alternation separately here. self.protocol["stimuli"]["alternate"],
                        )
                        self.dblist.append(dbspl)
                        self.freqlist.append(frequency)
                        if i == 0 and j == 0:
                            wave_n.generate()
                            self.wave_out = wave_n.sound
                            self.wave_time = wave_n.time
                        else:
                            self.wave_out += wave_n.generate()
                        n += 1
                self.wave_matrix["interleaved_plateau", 0] = {
                    "sound": self.wave_out,
                    "rate": self.sfout,
                    "dbspls": self.dblist,
                    "frequencies": self.freqlist,
                }
                self.nwaves = 1

            case "interleaved_ramp":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                if frequency is None:
                    frequency = self.protocol["stimuli"]["default_frequency"]
                dt = self.protocol["stimuli"]["stimulus_period"]
                s0 = self.protocol["stimuli"]["delay"]
                dbs = eval(self.protocol["stimuli"]["dblist"])
                freqs = eval(self.protocol["stimuli"]["freqlist"])
                self.dblist = []
                self.freqlist = []
                wave_duration = s0 + len(dbs) * len(freqs) * dt + dt  # duration of the waveform
                self.nwaves = self.protocol["stimuli"]["nreps"]
                n = 0
                dbref = 100.0
                for i, frequency in enumerate(freqs):
                    for j, dbspl in enumerate(dbs):

                        wave_n = sound.TonePip(
                            rate=self.sfout,
                            duration=wave_duration,
                            f0=frequency,
                            dbspl=dbspl,
                            pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                            ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                            pip_start=[s0 + n * dt],
                            alternate=False,  # do alternation separately here. self.protocol["stimuli"]["alternate"],
                        )
                        self.dblist.append(dbspl)
                        self.freqlist.append(frequency)
                        if i == 0 and j == 0:
                            wave_n.generate()
                            self.wave_out = wave_n.sound
                            self.wave_time = wave_n.time
                        else:
                            self.wave_out += wave_n.generate()
                        n += 1

                self.wave_matrix["interleaved_ramp", 0] = {
                    "sound": self.wave_out,
                    "rate": self.sfout,
                    "dbspls": self.dblist,
                    "frequencies": self.freqlist,
                }
                self.nwaves = 1 # self.protocol["stimuli"]["nreps"]

            case "interleaved_random":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                if frequency is None:
                    frequency = self.protocol["stimuli"]["default_frequency"]
                dt = self.protocol["stimuli"]["stimulus_period"]
                s0 = self.protocol["stimuli"]["delay"]
                dbs = eval(self.protocol["stimuli"]["dblist"])
                freqs = eval(self.protocol["stimuli"]["freqlist"])
                wave_duration = s0 + len(dbs) * len(freqs) * dt + dt  # duration of the waveform
                print("Wave duration: ", wave_duration)
                n = 0
                dbfr = np.tile(dbs, len(freqs))
                frdb = np.tile(freqs, len(dbs))
                indices = np.arange(len(dbfr))
                np.random.shuffle(indices)  # in place.
                self.dblist = dbfr[indices]
                self.freqlist = frdb[indices]  # save the order so we can match the responses
                dbref = 100.0
                for n, isn in enumerate(indices):
                    wave_n = sound.TonePip(
                        rate=self.sfout,
                        duration=wave_duration,
                        f0=frdb[isn],
                        dbspl=dbfr[isn],
                        pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                        ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                        pip_start=[s0 + n * dt],
                        alternate=False,  # do alternation separately here. self.protocol["stimuli"]["alternate"],
                    )
                    if n == 0:
                        wave_n.generate()
                        self.wave_out = wave_n.sound
                        self.wave_time = wave_n.time
                    else:
                        self.wave_out += wave_n.generate()

                    self.wave_matrix["interleaved_random", isn] = {
                        "sound": self.wave_out,
                        "rate": self.sfout,
                        "dbspls": self.dblist,
                        "frequencies": self.freqlist,
                        "indices": isn,
                    }
                self.nwaves = 1
                # self.protocol["stimuli"]["nreps"]
            case _:
                raise ValueError(f"Unrecongnized wavetype: {wavetype:s}")

    def make_and_plot(self, n: int, wavetype: str, dbspl: list, frequency: list):

        self.make_waveforms(wavetype=wavetype, dbspl=dbspl, frequency=frequency)
        for k in list(self.wave_matrix.keys()):
            print(k, np.max(self.wave_matrix[k]["sound"]))
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
        self.PS.reset_hardware()
        self.protocol = self.PR.get_current_protocol()  # be sure we have current protocol data
        print("Starting test acquisition")
        self.TrialCounter = 0
        flist = self.protocol["stimuli"]["freqlist"]
        if self.protocol["protocol"]["stimulustype"] == "click":
            if len(flist) == 0:
                flist = [0] * len(self.protocol["stimuli"]["dblist"])
        self.make_and_plot(
            n=0,
            wavetype=self.protocol["protocol"]["stimulustype"],
            dbspl=self.protocol["stimuli"]["dblist"],
            frequency=flist,
        )
        self.threadpool.start(self.Presenter.run)  # start the thread.
        print("Starting test acquisition")
        self.stimulus_waveform.enableAutoRange()
        self.Dock_Recording.raiseDock()
        self.Dock_Recording.update()
        self.TrialTimer.setInterval(int(self.protocol["stimuli"]["stimulus_period"]))
        self.TrialTimer.timeout.emit()
        self.TrialTimer.start()
        print("Starting acquisition?")
        self.Presenter.setWaveforms(self.wave_matrix, self.protocol, self.PS)

    def update_ABR(self, chdata, wave_counter, repetition_counter):
        if chdata[0] is None:  # no data, so fake it
            rng = np.random.default_rng()
            chdata = [rng.standard_normal(self.PS.Ndata) * 1e-6, rng.standard_normal(self.PS.Ndata) * 1e-6]
        self.ch1_data = np.array(chdata[0]).T
        # self.ch2_data = np.array(ch2[0]).T
        if self.TrialCounter == 0:
            self.summed_buffer = self.ch1_data
        else:
            self.summed_buffer += self.ch1_data
        if self.TrialCounter == 0:
            self.plot_ABR_Raw.clear()
            self.plot_ABR_Average.clear()
        self.TrialCounter = self.TrialCounter + 1
        # self.t_stim = np.arange(0, len(self.ch1_data)/self.PS.Stimulus.out_sampleFreq, 1./self.PS.Stimulus.out_sampleFreq )

        self.t_record = np.arange(0, len(self.ch1_data) / self.PS.trueFreq, 1 / self.PS.trueFreq)
        self.plot_ABR_Raw.clear()
        self.plot_ABR_Raw.plot(self.t_record, self.ch1_data, pen=pg.mkPen("c"))
        # self.plot_ABR_Raw.autoRange(True)
        self.plot_ABR_Raw.setXRange(0, np.max(self.t_record))
        avedata = None
        self.plot_ABR_Average.clear()
        self.plot_ABR_Average.plot(
            self.t_record, self.summed_buffer / float(self.TrialCounter), pen=pg.mkPen("g")
        )
        # self.plot_ABR_Raw.autoRange(True)
        self.plot_ABR_Raw.setXRange(0, np.max(self.t_record))

        if self.acq_mode == "test":
            return
        # now save all the data.... 
        subject_data = {}
        children = self.ptreedata.children()
        for child in children:
            if child.name() == "Subject Data":
                for childs in child:
                    subject_data[childs.name()] = childs.value()
        # now assemble data and save it
        write_time = datetime.datetime.now()
        out_data = {'subject_data': subject_data,
                    'calibration': self.caldata,
                    'stimuli': self.wave_matrix,
                    'protocol': self.protocol,
                    'wave_number': wave_counter,
                    'repetition': repetition_counter,
                    'data': self.ch1_data,
                    'record_frequency': self.PS.trueFreq,
                    'timestamp': write_time
                    }
        data_directory = Path(self.config["datapath"])
        if not data_directory.is_dir():
            Path.mkdir(data_directory, exist_ok=True)  # create if it does not exist.
        if self.basefn is None:
            self.basefn = write_time.strftime("%Y-%m-%d")
        # print("base fn", self.basefn)
        subject_data['Subject ID'] = "testing"
        # print("data dir: ", data_directory)
        # print("subject: ", subject_data['Subject ID'])
        self.subject_dir = Path(self.config["datapath"], self.basefn, subject_data['Subject ID'])
        print(str(self.subject_dir))
        # Path.mkdir(self.subject_dir, exist_ok = True)
        fn =  Path(f"abr_data/{self.basefn:s}_{self.protocol['protocol']['stimulustype']:s}_{wave_counter:03d}_{repetition_counter:03d}")
        fn = str(fn)+".p"
        print("output file: ", fn)
        with open(fn, "wb") as fh:
            pickle.dump(out_data, fh, protocol=pickle.HIGHEST_PROTOCOL)


    def stop(self):
        self.PS.cleanup_NIDAQ()
        self.signal_stop.emit()
        self.TrialTimer.stop()

    def pause(self):
        self.signal_pause.emit()
        self.TrialTimer.stop()

    def continue_trials(self):
        if self.Presenter._paused:
            self.Presenter.resume()
            self.TrialTimer.start()


if __name__ == "__main__":
    prog = PyABR()
    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()
