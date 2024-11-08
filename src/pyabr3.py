#!/usr/bin/env python
"""
pyabr : a python program for ABR recordings. 
Parameter setup is defined in .toml files (text files),
and displayed in a window.

"""

import sys
import datetime
import pickle
import platform
import time
import toml
from pathlib import Path
import pprint
import traceback

import numpy as np
import scipy.signal
import pyqtgraph as pg
import pyqtgraph.reload as reload
import pyqtgraph.dockarea as PGD
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph import configfile

import src.convert_nested_ordered_dict as convert_nested_ordered_dict
import sound  # for waveform generation
import src.PySounds as PySounds  # for access to hardware


class WorkerSignals(QtCore.QObject):
    """
    Defines the signals available from a running worker thread.
    """
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

class Worker(QtCore.QRunnable):
    """
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    """


    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress


    @QtCore.pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
            print("stim delivered")
        except Exception as e:
            self.signals.error.emit((e, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class PyABR(object):
    def __init__(self):
        self.QColor = QtGui.QColor
        self.PS = PySounds.PySounds()
        # check hardware first
        self.hw, self.sfin, self.sfout = self.PS.getHardware()

        self.ptreedata = None  # until the tree is built
        self.rawtoml = ""
        self.protocol = None
        self.debugFlag = True
        self.known_protocols = [p.name for p in Path('protocols').glob("*.cfg")]
        self.current_protocol = self.known_protocols[0]
        self.read_protocol(self.current_protocol, update=False)
        self.buildGUI()
        self.update_protocol()
        self.counter = 0

        self.threadpool = QtCore.QThreadPool()
        self.TrialTimer = QtCore.QTimer()
        self.TrialTimer.setInterval(1000)
        self.TrialTimer.timeout.connect(self.recurring_timer)
        # self.timer.start()
        self.TrialCounter = 0
        # note after buildgui, all is done with callbacks


    def read_protocol(self, protocolname, update:bool=False):
        """
        Read the current protocol
        """
        protocol = configfile.readConfigFile(Path('protocols', protocolname))
        # print("protocol: ", protocol)
        # if isinstance(protocol['stimuli']["dblist"], str):
        #     protocol['stimuli']["dblist"] = list(eval(str))
        # if "freqlist" in list(protocol["stimuli"].keys()):
        #     if isinstance(protocol['stimuli']["freqlist"], str):
        #         protocol['stimuli']["freqlist"] = list(eval(str))
        protocol = convert_nested_ordered_dict.convert_nested_ordered_dict(protocol)
        # paste the raw string into the text box for reference
        self.protocol = protocol   # the parameters in a dictionary... 
        if update:
            self.update_protocol()

    def update_protocol(self):
        children = self.ptreedata.children()
        data = None
        for child in children:
            if child.name() == "Parameters":
                 data = child
        if data is None:
            return
        for child in data.children():
            if child.name() == "wave_duration":
                child.setValue(self.protocol['stimuli']["wave_duration"])
            if child.name() == "stimulus_duration":
                child.setValue(self.protocol['stimuli']['stimulus_duration'])
            if child.name() == "stimulus_risefall":
                child.setValue(self.protocol['stimuli']['stimulus_risefall'])
            if child.name() == "delay":
                child.setValue(self.protocol['stimuli']['delay'])
            if child.name() == "nreps":
                child.setValue(self.protocol['stimuli']['nreps'])
            if child.name() == "stimulus_period":
                child.setValue(self.protocol['stimuli']['stimulus_period'])
            if child.name() == "nstim":
                child.setValue(self.protocol['stimuli']['nstim'])
            if child.name() == "interval":
                child.setValue(self.protocol['stimuli']['interval'])
            if child.name() == "alternate":
                child.setValue(self.protocol['stimuli']['alternate'])
            if child.name() == "default_frequency":
                child.setValue(self.protocol['stimuli']['default_frequency'])
            if child.name() == "default_spl":
                child.setValue(self.protocol['stimuli']['default_spl'])
            if child.name() == "freqlist":
                child.setValue(self.protocol['stimuli']['freqlist'])
            if child.name() == "dblist":
                child.setValue(self.protocol['stimuli']['dblist'])
        self.make_waveforms()
        self.stimulus_waveform.clearPlots()
        self.stimulus_waveform.plot(
                    self.wave_time,
                    self.wave_out,
                    pen=pg.mkPen("y"),
                )
        self.stimulus_waveform.enableAutoRange()
        self.Dock_Recording.raiseDock()



    def get_protocol(self): 
        """get_protocol Read the current protocol information and put it into
        the protocol dictionary
        """
        children = self.ptreedata.children()
        data = None
        for child in children:
            if child.name() == "Parameters":
                 data = child
        if data is None:
            return
        for child in data.children():
            if child.name() == "wave_duration":
                self.protocol['stimuli']["wave_duration"] = child.value()
            if child.name() == "stimulus_duration":
                self.protocol['stimuli']['stimulus_duration'] = child.value()
            if child.name() == "stimulus_risefall":
                self.protocol['stimuli']['stimulus_risefall'] = child.value()
            if child.name() == "delay":
                self.protocol['stimuli']['delay'] = child.value()
            if child.name() == "nreps":
                self.protocol['stimuli']['nreps'] = child.value()
            if child.name() == "stimulus_period":
                self.protocol['stimuli']['stimulus_period'] = child.value()
            if child.name() == "nstim":
                self.protocol['stimuli']['nstim'] = child.value()
            if child.name() == "interval":
                self.protocol['stimuli']['interval'] = child.value()
            if child.name() == "alternate":
                self.protocol['stimuli']['alternate'] = child.value()
            if child.name() == "default_frequency":
                self.protocol['stimuli']['default_frequency'] = child.value()
            if child.name() == "default_spl":
                self.protocol['stimuli']['default_spl'] = child.value()
            if child.name() == "freqlist":
                self.protocol['stimuli']['freqlist'] = child.value()
            if child.name() == "dblist":
                self.protocol['stimuli']['dblist'] = child.value()


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
        right_docks_width= 1024 - ptreewidth - 20
        right_docs_height  = 800
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
        self.win.resize(1380, 1024)
        # print(dir(self.dockArea))
        self.Dock_Params.raise_()
        stim = self.protocol['stimuli']
        # Define parameters that control aquisition and buttons...
        params = [
  
            {
                "name": "Protocol",  # for selecting stimulus protocol
                "type": "list",
                "limits": [str(p) for p in self.known_protocols],
                "value": str(self.current_protocol),
            },
            {
                "name": "Parameters",  # for displaying stimulus parameters
                "type": "group",
                "children": [
                    {"name": "wave_duration", "type": "float", "value":  stim["wave_duration"], "limits": [0.1, 10.0]},   # waveform duration in milli seconds
                    {"name": "stimulus_duration", "type": "float", "value":  5e-3, "limits": [1e-3, 20e-3]},  # seconds
                    {"name": "stimulus_risefall", "type": "float", "value":  5e-4, "limits": [1e-4, 10e-4]},  # seconds
                    {"name": "delay", "type": "float", "value":  3e-3, "limits": [1e-3, 1.0]},  # seconds
                    {"name": "nreps", "type": "float", "value":  500, "limits": [1, 2000]},  # number of repetitions
                    {"name": "stimulus_period", "type": "float", "value":  1, "limits": [0.1, 10.0]},  # seconds
                    {"name": "nstim", "type": "int", "value":  30, "limits": [1, 1000]},  # number of pips
                    {"name": "interval", "type": "float", "value":  25e-3, "limits": [1e-3, 1e-1]}, # seconds
                    {"name": "alternate", "type": "bool", "value": True},
                    {"name": "default_frequency", "type": "float", "value":  4000.0, "limits": [100.0, 100000.0]},
                    {"name": "default_spl", "type": "float", "value":  80.0, "limits": [0.0, 100.0]},
                    {"name": "freqlist", "type": "str", "value": ' [2000.0, 4000.0, 8000.0, 12000.0, 16000.0, 20000.0, 24000.0, 32000.0, 48000.0]'},
                    {"name": "dblist", "type": "str", "value":  '[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]'},
                ],
            },

            {"name": "Comment", "type": "text", "value": ""},  # open comment field
            {
                "name": "Actions",
                "type": "group",
                "children": [
                    {"name": "New Filename", "type": "action"},
                    {"name": "Test Acquisition", "type": "action"},
                    {"name": "Start Acquisition", "type": "action"},
                    {"name": "Pause", "type": "action"},
                    {"name": "Continue", "type": "action"},
                    {"name": "Stop", "type": "action"},
                    {"name": "Save Visible", "type": "action"},
                    {"name": "Load File", "type": "action"},
                ],
            },
            {
                "name": "Status",
                "type": "group",
                "children": [
                    {"name": "dBSPL", "type": "float", "value": 0.0, "readonly": True},
                    {
                        "name": "Freq (kHz)",
                        "type": "float",
                        "value": 0.0,
                        "readonly": True,
                    },
                ],
            },
            {
                    "name": "Quit",
                    "type": "action",
            },
          

        ]

        ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="params", type="group", children=params)
        ptree.setParameters(self.ptreedata)
        ptree.setMaximumWidth(350)
        self.Dock_Params.addWidget(ptree)

        # add space for the graphs
        view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(50, 50, 50))
        view.setCentralItem(l)

        self.plot_ABR_Raw = l.addPlot()
        self.plot_ABR_Raw.getAxis("left").setLabel("uV", color="#ff0000")
        self.plot_ABR_Raw.setTitle("Raw ABR Signal", color="#ff0000")
        self.plot_ABR_Raw.getAxis("bottom").setLabel("t (msec)", color="#ff0000")
        self.plot_ABR_Raw.setYRange(-10.0, 10.0)  # microvolts

        l.nextRow()  # averaged abr trace
        self.plot_ABR_Average = l.addPlot()  #
        self.plot_ABR_Average.setXLink(self.plot_ABR_Raw)
        self.plot_ABR_Average.setTitle("Average ABR Signal", color="#ff0000")
        self.plot_ABR_Average.getAxis("left").setLabel("uV", color="#0000ff")
        self.plot_ABR_Average.setYRange(-10, 10.0)
        self.plot_ABR_Average.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        l.nextRow()  # waveforms
        self.stimulus_waveform = l.addPlot()  #
        self.stimulus_waveform.setXLink(self.plot_ABR_Raw)
        self.stimulus_waveform.setTitle("Waveform", color="#ff0000")
        self.stimulus_waveform.getAxis("left").setLabel("V", color="#0000ff")
        self.stimulus_waveform.setYRange(-10, 10.0)
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
        self.win.show()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)


    def command_dispatcher(self, param, changes):

        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            # print("path: ", path)
            match path[0]:
                case "Quit":
                    exit()
                case  "Protocol":
                    self.current_protocol = data
                    self.read_protocol(data, update=False)
                case "Actions":
                    match path[1]:
                        case "New Filename":
                            self.new_filename()
                        case "Test Acquisition":
                            self.test_acquire()
                        case "Start Acquisition":
                            self.sequence_acquire()
                        case "Stop/Pause":
                            self.stop_pause()
                        case "Continue":
                            self.continue_trials()
                        case "Save Visible":
                            self.save_visible()
                        case "Load File":
                            self.load_file()
                case "Status":
                    match path[1]:
                        case "dBSPL":
                            self.protocol["stimuli"]["default_spl"] = data
                        case "Freq (kHz)":
                            self.protocol["stimuli"]["default_frequency"] = data
        self.ptreedata.sigTreeStateChanged.disconnect(self.command_dispatcher)
        self.update_protocol()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

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

    def update_plots(self):
        pass

          

    def sequence_acquire(self):
        """
        Present stimuli using sequences of intensities and/or
        frequencies
        """
        
        print("Starting sequenced acquisition")

        self.stimulus_waveform.clearPlots()
        self.Dock_Recording.raise_
        if 'freqlist' not in list(self.protocol["stimuli"].keys()):
            freqs = [1000.]
        else:
            freqs = self.protocol["stimuli"]["freqlist"]
            for freq in freqs:
                for db in self.protocol["stimuli"]["dblist"]:
                    for n in range(self.protocol["stimuli"]["nreps"]):
                        self.make_waveforms(dbspl=db, frequency=freq)
                        self.stimulus_waveform.plot(
                            self.wave_out.time,
                            self.wave_out.sound,
                            pen=pg.mkPen("y"),
                        )
                        self.stimulus_waveform.enableAutoRange()


    def update_ABR(self):
        print("Update ABR")

    def thread_complete(self):
        print("Thread complete")
    
    def update_progress(self, progress):
        print("Progress: ", progress)

    def recurring_timer(self):
        self.counter += 1  # nothing really to do here.

    
    def make_waveforms(self, dbspl=None, frequency=None):
        match self.protocol["protocol"]["stimulustype"]:
            case "click":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                starts = np.cumsum(
                    np.ones(self.protocol["stimuli"]["nstim"])*
                    self.protocol["stimuli"]["interval"]
                )
                starts += self.protocol["stimuli"]["delay"] 
                wave = sound.ClickTrain(
                    rate=self.sfout,
                    duration=self.protocol["stimuli"]["wave_duration"],
                    dbspl=dbspl,
                    click_duration=self.protocol["stimuli"]["stimulus_duration"],
                    click_starts=starts,
                    alternate=self.protocol["stimuli"]["alternate"],
                )
                wave.generate()
                self.wave_out = wave.sound
                self.wave_time = wave.time
    
            case "tonepip":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                if frequency is None:
                    frequency = self.protocol["stimuli"]["default_frequency"]
                starts = np.cumsum(
                    np.ones(self.protocol["stimuli"]["nstim"])*
                    self.protocol["stimuli"]["interval"]
                )
                starts += self.protocol["stimuli"]["delay"] 
                wave = sound.TonePip(
                    rate=self.sfout,
                    duration=self.protocol["stimuli"]["wave_duration"],
                    f0=frequency,
                    dbspl=dbspl,
                    pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                    ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                    pip_start=starts,
                    alternate=self.protocol["stimuli"]["alternate"],
                )
                wave.generate()
                self.wave_out = wave.sound
                self.wave_time = wave.time
            
            case "interleaved_plateau":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                if frequency is None:
                    frequency = self.protocol["stimuli"]["default_frequency"]
                dt = self.protocol["stimuli"]["stimulus_period"]
                s0 = self.protocol["stimuli"]["delay"]
                dbs = eval(self.protocol["stimuli"]["dblist"])
                freqs = eval(self.protocol["stimuli"]["freqlist"])
                wave_duration = s0 + len(dbs)*len(freqs)*dt + dt  # duration of the waveform
                print("Wave duration: ", wave_duration)
                n = 0
                for j, dbspl in enumerate(dbs):
                    for i, frequency in enumerate(freqs):
                        wave_n = sound.TonePip(
                            rate=self.sfout,
                            duration=wave_duration,
                            f0=frequency,
                            dbspl=dbspl,
                            pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                            ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                            pip_start=[s0+n*dt],
                            alternate=False, # do alternation separately here. self.protocol["stimuli"]["alternate"],
                            )
                        n += 1
                        if i == 0 and j == 0:
                            wave_n.generate()
                            self.wave_out = wave_n.sound
                            self.wave_time = wave_n.time
                        else:
                            self.wave_out += wave_n.generate()

            case "interleaved_ramp":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                if frequency is None:
                    frequency = self.protocol["stimuli"]["default_frequency"]
                dt = self.protocol["stimuli"]["stimulus_period"]
                s0 = self.protocol["stimuli"]["delay"]
                dbs = eval(self.protocol["stimuli"]["dblist"])
                freqs = eval(self.protocol["stimuli"]["freqlist"])
                wave_duration = s0 + len(dbs)*len(freqs)*dt + dt  # duration of the waveform
                print("Wave duration: ", wave_duration)
                n = 0
                for i, frequency in enumerate(freqs):
                    for j, dbspl in enumerate(dbs):

                        wave_n = sound.TonePip(
                            rate=self.sfout,
                            duration=wave_duration,
                            f0=frequency,
                            dbspl=dbspl,
                            pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                            ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                            pip_start=[s0+n*dt],
                            alternate=False, # do alternation separately here. self.protocol["stimuli"]["alternate"],
                            )
                        n += 1
                        if i == 0 and j == 0:
                            wave_n.generate()
                            self.wave_out = wave_n.sound
                            self.wave_time = wave_n.time
                        else:
                            self.wave_out += wave_n.generate()
                    
        
            case "interleaved_random":
                if dbspl is None:
                    dbspl = self.protocol["stimuli"]["default_spl"]
                if frequency is None:
                    frequency = self.protocol["stimuli"]["default_frequency"]
                dt = self.protocol["stimuli"]["stimulus_period"]
                s0 = self.protocol["stimuli"]["delay"]
                dbs = eval(self.protocol["stimuli"]["dblist"])
                freqs = eval(self.protocol["stimuli"]["freqlist"])
                wave_duration = s0 + len(dbs)*len(freqs)*dt + dt  # duration of the waveform
                print("Wave duration: ", wave_duration)
                n = 0
                dbfr = np.tile(dbs, len(freqs))
                frdb = np.tile(freqs, len(dbs))
                indices = np.arange(len(dbfr))
                np.random.shuffle(indices)  # in place.
                self.dblist = dbfr[indices]
                self.freqlist = frdb[indices]  # save the order so we can match the responses
                for n, isn in enumerate(indices):        
                        wave_n = sound.TonePip(
                            rate=self.sfout,
                            duration=wave_duration,
                            f0=frdb[isn],
                            dbspl=dbfr[isn],
                            pip_duration=self.protocol["stimuli"]["stimulus_duration"],
                            ramp_duration=self.protocol["stimuli"]["stimulus_risefall"],
                            pip_start=[s0+n*dt],
                            alternate=False, # do alternation separately here. self.protocol["stimuli"]["alternate"],
                            )
                        if n == 0:
                            wave_n.generate()
                            self.wave_out = wave_n.sound
                            self.wave_time = wave_n.time
                        else:
                            self.wave_out += wave_n.generate()


    def test_acquire(self):
        """
        present and collect responses without saving - using
        abbreviated protocol
        """
        
        self.Dock_Recording.raiseDock()
        self.app.processEvents()
        print("Starting test acquisition")
        self.TrialCounter = 0
        self.make_waveforms()
        self.stimulus_waveform.clearPlots()
        self.stimulus_waveform.plot(
            self.wave_time, self.wave_out, pen=pg.mkPen("y")
        )
        self.stimulus_waveform.enableAutoRange()
        self.Dock_Recording.raiseDock()
        self.Dock_Recording.update()
        self.TrialTimer.setInterval(int(self.protocol["stimuli"]["stimulus_period"]))
        self.TrialTimer.timeout.connect(self.next_trial)
        self.TrialTimer.start()
        self.stimulus_state = "running"

        worker = Worker(self.play_waveforms)
        worker.signals.result.connect(self.update_ABR)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.update_progress)

            
    def play_waveforms(self):
        # self.TrialTimer.timeout.connect(self.next_trial)
        for n in range(self.protocol["stimuli"]["nreps"]):
            self.next_trial()
            t = np.arange(0, 1, 0.001)
            self.plot_ABR_Raw.plot(t, np.random.randn(len(t)), pen=pg.mkPen("c"))
            self.plot_ABR_Raw.plot(np.arange(0,  1./44100.*len(self.ch1) , 1/44100.), self.ch1, pen=pg.mkPen("y"))
            self.plot_ABR_Raw.update()
        self.stop()

    def stop(self):
        self.TrialTimer.stop()
        self.stimulus_state = "stopped"
    
    def pause(self):
        self.TrialTimer.stop()
        self.stimulus_state = "paused"

    def continue_trials(self):
        if self.stimulus_state == "paused":
            self.stimulus_state = "running"
            self.TrialTimer.start()
            self.next_trial()

    # callback routine to stop timer when thread times out.
    def next_trial(self):
        if self.debugFlag:
            print("NextTrial: entering", self.TrialCounter)
        if self.stimulus_state in ['stopped', 'paused']:
            return
        self.TrialTimer.stop()
        if self.TrialCounter <= self.protocol["stimuli"]["nreps"]:
            self.TrialTimer.start(int(self.protocol["stimuli"]["stimulus_period"]))
            self.PS.playSound(self.wave_out, self.wave_out, self.sfout)
            self.TrialCounter = self.TrialCounter + 1
        if self.debugFlag:
            print("NextTrial: exiting")

        # (self.ch1, self.ch2) = sound.retrieveInputs()


if __name__ == "__main__":
    prog = PyABR()
    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()
