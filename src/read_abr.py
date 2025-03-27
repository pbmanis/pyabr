import datetime
import pathlib
import pickle
import platform
from pathlib import Path
import re
from string import ascii_letters
from typing import Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as mpl
from pylibrary.plotting import styler as ST
from pylibrary.tools import cprint as CP

# import ephys.tools.get_configuration as GETCONFIG
import get_configuration as GETCONFIG
import plothelpers as mpl_PH
import src.analyzer
from src import filter_util as filter_util
from src import parse_ages as PA
from src import read_abr4 as read_abr4
from src import read_calibration as read_calibration
from src import fit_thresholds

use_matplotlib = True
from matplotlib.backends.backend_pdf import PdfPages
from pylibrary.plotting import plothelpers as PH

# Check the operating system and set the appropriate path type
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

WaveAnalyzer = src.analyzer.Analyzer()

re_age = re.compile(r"[_ ]{1}[p|P]{0,1}[\d]{1,3}[d]{0,1}[_ ]{1}", re.IGNORECASE)
re_splfile = re.compile(r"^(?P<datetime>([\d]{8}-[\d]{4}))-(?P<type>([(n)(p)(SPL)])).txt$")
re_click_file = re.compile(r"^(?P<datetime>([\d]{8}-[\d]{4}))-n.txt$")
re_pyabr3_click_file = re.compile(
    r"^(?P<datetime>([\d]{4}-[\d]{2}-[\d]{2}))_click_(?P<serial>([0-9]{3}_[0-9]{3})).p$"
)
re_tone_file = re.compile(r"^(?P<datetime>([\d]{8}-[\d]{4}))-n-(?P<frequency>([0-9.]*)).txt$")
re_pyabr3_interleaved_tone_file = re.compile(
    r"^(?P<datetime>([\d]{4}-[\d]{2}-[\d]{2}))_interleaved_plateau_(?P<serial>([0-9]{3}_[0-9]{3})).p$"
)
re_subject = re.compile( 
    r"(?P<strain>([A-Za-z]{3,10}))[_ ]{1}(?P<sex>([M|F]{1}))[_ ]{1}(?P<subject>([A-Z0-9]{4}))[_ ]{1}(?P<age>([p|P\d]{1,4}))[_ ]{1}(?P<date>([0-9]{0,8}))[_ ]{0,1}(?P<treatment>([A-Za-z]{2}))"
)


class AnalyzeABR:
    def __init__(self):
        self.caldata = None
        self.gain = 1e4
        self.FILT = filter_util.Utility()
        self.frequencies = []
        self.hide_treatment = False
        self.experiment = (
            None  # this is the experiment dict from the project configuration directory.
        )

        # 24414.0625

    def set_hide_treatment(self, hide_treatment: bool):
        self.hide_treatment = hide_treatment

    def get_experiment(
        self, config_file_name: Union[Path, str] = "experiments.cfg", exptname: str = None
    ):
        datasets, experiments = GETCONFIG.get_configuration(config_file_name)
        if exptname in datasets:
            self.experiment = experiments[exptname]
        else:
            raise ValueError(
                f"Experiment {exptname} not found in the configuration file with datasets={datasets!s}"
            )

    def read_abr_file(self, fn):
        with open(fn, "rb") as fh:
            d = pickle.load(fh)
            self.caldata = d["calibration"]
            # Trim the data array to remove the delay to the stimulus.
            # to be consistent with the old ABR4 program, we leave
            # the first 1 msec prior to the stimulus in the data array
            delay = float(d["protocol"]["stimuli"]["delay"])
            i_delay = int((delay-1e-3) * d["record_frequency"])
            d["data"] = d["data"][i_delay:]
            d["data"] = np.append(d["data"], np.zeros(i_delay))
        return d

    def show_calibration(self, fn):
        # d = self.read_abr_file(fn)
        # dbc = self.convert_attn_to_db(20.0, 32000)
        # print(dbc)
        read_calibration.plot_calibration(self.caldata)

    def show_calibration_history(self):
        fn = "abr_data/calibration_history"
        files = list(Path(fn).glob("frequency_MF1*.cal"))
        cal_files = []
        creation_dates = []
        for j, f in enumerate(files):
            cal_files.append(read_calibration.get_calibration_data(f))
            creation_dates.append(f.stat().st_ctime)
        app = pg.mkQApp("Calibration Data Plot")
        win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        win.resize(800, 600)
        win.setWindowTitle(f"Calibration History")
        symbols = ["o", "s", "t", "d", "+", "x"]
        win.setBackground("w")
        pl = win.addPlot(title=f"Calibration")
        # li = pg.LegendItem()
        # li.setColumnCount(2)

        # li.setParentItem(pl)
        pl.setLogMode(x=True, y=False)
        pl.addLegend()
        pl.legend.setOffset((0.1, 0))
        pl.legend.setColumnCount(2)
        for cd in sorted(creation_dates):
            i = creation_dates.index(cd)
            caldata = cal_files[i]
            freqs = caldata["freqs"]
            if "2022" in caldata["date"]:
                continue
            # pl.plot(freqs, caldata['maxdb'], pen='r', name="Max SPL (0 dB Attn)")
            pl.plot(
                freqs,
                caldata["db_cs"],
                pen=pg.mkPen(i, len(cal_files), width=2),
                symbolPen=pg.mkPen(i, len(cal_files)),
                symbol=symbols[i % len(symbols)],
                symbolSize=7,
                name=f"{caldata['date']:s}",
            )
            # li.addItem(pl.legend.items[-1][1], f"{caldata['date']:s}")
            # pl.plot(freqs, caldata['db_bp'], pen='g', name=f"Measured dB SPL, attn={caldata['calattn']:.1f}, bandpass")
            # pl.plot(freqs, caldata['db_nf'], pen='b', name="Noise Floor")
            # pl.setLogMode(x=True, y=False)
        pl.setLabel("bottom", "Frequency", units="Hz")
        pl.setLabel("left", "dB SPL")
        pl.showGrid(x=True, y=True)
        # text_label = pg.LabelItem(txt, size="8pt", color=(255, 255, 255))
        # text_label.setParentItem(pl)
        # text_label.anchor(itemPos=(0.5, 0.05), parentPos=(0.5, 0.05))

        pg.exec()

    def convert_attn_to_db(self, attn, fr):
        """convert_attn_to_db converts the attenuation value at a particular
        frquency to dB SPL, based on the calibration file data

        Parameters
        ----------
        attn : float
            attenuator setting (in dB)
        fr : float
            the frequency of the stimulus (in Hz)
        """
        if self.caldata is None:
            raise ValueError(
                f"Calibration data not loaded; must load from a data file or a calibration file"
            )

        dBSPL = 0.0
        if fr in self.caldata["freqs"]:  # matches a measurement frequency
            ifr = np.where(self.caldata["freqs"] == fr)
            dBSPL = self.caldata["maxdb"][ifr]
            # print("fixed")
        else:
            # interpolate linearly between the two closest frequencies
            # first, we MUST sort the caldata and frequencies so that the freqs are in ascending
            # order, otherwise numpy gives the wrong result.
            ifr = np.argsort(self.caldata["freqs"])
            freqs = self.caldata["freqs"][ifr]
            maxdb = self.caldata["maxdb"][ifr]
            # for i, frx in enumerate(freqs):
            #     print(f"{frx:8.1f}  {maxdb[i]:8.3f}")
            dBSPL = np.interp([fr], freqs, maxdb)
        #     print("interpolated", dBSPL)
        # print(f"dBSPL = {dBSPL:8.3f} for freq={float(fr):9.2f} with {float(attn):5.1f} dB attenuation")
        dBSPL_corrected = dBSPL[0] - attn
        return dBSPL_corrected

    def average_within_traces(
        self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None
    ):
        # _i_ is the index into the acquisition series. There is one file for each repetition of each condition.
        # the series might span a range of frequencies and intensities; these are
        # in the protocol:stimulus dictionary (dblist, freqlist)
        # we average response across traces for each intensity and frequency
        # this function specifically works when each trace has one stimulus condition (db, freq), and is
        # repeated nreps times.
        # The returned data is the average of the responses across the nreps for this (the "ith") stimulus condition
        stim_type = str(Path(fd).stem)
        # print("Stim type: ", stim_type)
        # print("avg within traces:: pyabr3 Protocol: ", protocol)
        if stim_type.lower().startswith("tone"):
            name = "tonepip"
        elif stim_type.lower().startswith("interleaved"):
            name = "interleaved_plateau"
        elif stim_type.lower().startswith("click"):
            name = "click"
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]
        interstim_interval = protocol["stimuli"]["interval"]
        missing_reps = []
        for n in range(nreps):  # loop over the repetitions for this specific stimulus
            fn = f"{date}_{name}_{i:03d}_{n+1:03d}.p"
            if not Path(fd, fn).is_file():
                missing_reps.append(n)
                continue
            d = self.read_abr_file(Path(fd, fn))
            sample_rate = d["record_frequency"]
            if n == 0:
                data = d["data"]
                tb = np.linspace(0, len(data) / sample_rate, len(data))
            else:
                data += d["data"]
            
        if len(missing_reps) > 0:
            CP.cprint("r", f"Missing {len(missing_reps)} reps for {name} {i}", "red")
        data = data / (nreps - len(missing_reps))
        if high_pass_filter is not None:
            data = self.FILT.SignalFilter_HPFButter(
                data, high_pass_filter, sample_rate, NPole=4, bidir=True
            )
        # tile the traces.
        # first interpolate to 100 kHz
        # If you don't do this, the blocks will precess in time against
        # the stimulus, which is timed on a 500 kHz clock.
        # It is an issue because the TDT uses an odd frequency clock...

        trdur = len(data) / sample_rate
        newrate = 1e5
        tb100 = np.arange(0, trdur, 1.0 / newrate)

        one_response = int(interstim_interval * newrate)  # interstimulus interval
        arraylen = one_response * protocol["stimuli"]["nstim"]

        abr = np.interp(tb100, tb, data)
        sub_array = np.split(abr[:arraylen], protocol["stimuli"]["nstim"])
        sub_array = np.mean(sub_array, axis=0)
        tb100 = tb100[:one_response]
        return sub_array, tb100, newrate

    def average_across_traces(
        self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None
    ):
        """average_across_traces for abrs with multiple stimuli in a trace.
        This function averages the responses across multiple traces.
        and returns a list broken down by the individual traces.

        Parameters
        ----------
        fd : _type_
            _description_
        i : _type_
            _description_
        protocol : _type_
            _description_
        date : _type_
            _description_
        stim_type : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]

        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        stim_type = str(Path(fd).stem)
        # print("Stim type: ", stim_type)
        if stim_type.lower().startswith("interleaved"):
            stim_type = "interleaved_plateau"
        missing_reps = []
        ndata = 0
        for n in range(nreps):
            fn = f"{date}_{stim_type}_{i:03d}_{n+1:03d}.p"
            if not Path(fd, fn).is_file():
                missing_reps.append(n)
                continue
            d = self.read_abr_file(Path(fd, fn))

            if ndata == 0:
                data = d["data"]
            else:
                data += d["data"]
            if ndata == 0:

                tb = np.arange(0, len(data) / d['record_frequency'], 1.0 / d['record_frequency'])
            ndata += 1
        if len(missing_reps) > 0:
            CP.cprint("r", f"Missing {len(missing_reps)} reps for {fn!s}", "red")
        data = data / (nreps - len(missing_reps))
        if high_pass_filter is not None:
            print("average across traces hpf: ", high_pass_filter, d['record_frequency'])
            data = self.FILT.SignalFilter_HPFButter(
                data, HPF=high_pass_filter, samplefreq=d['record_frequency'], NPole=4, bidir=True
            )
        # tile the traces.
        # first linspace to 100 kHz
        trdur = len(data) / d['record_frequency']
        newrate = 1e5
        tb = np.linspace(0, trdur, len(data))
        # f, ax = mpl.subplots(2,1)
        # ax[0].plot(tb, data)

        tb100 = np.arange(0, trdur, 1.0 / newrate)
        # print('len tb100: ', len(tb100), np.mean(np.diff(tb100)))
        abr = np.interp(tb100, tb, data)
        # ax[0].plot(tb100, abr, '--r', linewidth=0.5)
        # mpl.show()
        # exit()

        one_response_100 = int(protocol["stimuli"]["stimulus_period"] * newrate)
        print("avg across traces:: pyabr3 .p file protocol[stimuli]: ", protocol["stimuli"])
        if isinstance(protocol["stimuli"]["freqlist"], str):
            frlist = eval(protocol["stimuli"]["freqlist"])
        if isinstance(protocol["stimuli"]["dblist"], str):
            dblist = eval(protocol["stimuli"]["dblist"])
        arraylen = one_response_100 * protocol["stimuli"]["nstim"]
        if stim_type.lower() in ["click", "tonepip"]:
            nsplit = protocol["stimuli"]["nstim"]
        elif stim_type.lower() in ["interleaved_plateau"]:
            nsplit = int(len(frlist) * len(dblist))
        else:
            raise ValueError(f"Stimulus type {stim_type} not recognized")
        arraylen = one_response_100 * nsplit
        # print(len(frlist), len(dblist))
        # print("one response: ", one_response_100)
        # print("nsplit: ", nsplit)
        # print("arranlen/nsplit: ", float(arraylen) / nsplit)
        # print("len data: ", len(data), len(abr), nsplit * one_response_100)
        sub_array = np.split(abr[:arraylen], nsplit)
        # print(len(sub_array))
        abr = np.array(sub_array)
        tb = tb100[:one_response_100]

        # print("abr shape: ", abr.shape, "max time: ", np.max(tb))
        stim = np.meshgrid(frlist, dblist)
        return abr, tb, newrate, stim

    def show_stimuli(self, fn):
        d = self.read_abr_file(fn)
        stims = list(d["stimuli"].keys())
        wave = d["stimuli"][stims[0]]["sound"]
        pg.plot(wave)
        pg.exec()

    ###########################################################################

    def parse_metadata(self, metadata, stim_type, acquisition):
        """parse_metadata : use the metadata dict to generate a
        title string for the plot, and the stack order for the data.

        Parameters
        ----------
        metadata : _type_
            _description_
        filename : _type_
            _description_
        stim_type : _type_
            _description_
        acquisition : _type_
            _description_


        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        fn = metadata["filename"]
        if metadata["type"] == "ABR4":
            filename = str(Path(fn).name)
            if filename in ["Click", "Clicks", "Tone", "Tones"]:
                filename = str(Path(fn).parent.name)
            else:
                filename = str(Path(fn).parent)
        else:
            filename = str(Path(fn).parent)
        subject_id = metadata["subject_id"]
        age = PA.ISO8601_age(metadata["age"])
        sex = metadata["sex"]
        amplifier_gain = metadata["amplifier_gain"]
        strain = metadata["strain"]
        weight = metadata["weight"]
        genotype = metadata["genotype"]
        rec_freq = metadata["record_frequency"]
        hpf = metadata["highpass"]

        title_file_name = filename
        page_file_name = filename
        if self.hide_treatment:
            # hide the treatment values in the title just in case...
            if acquisition == "ABR4":
                tf_parts = list(Path(fn).parts)
                fnp = tf_parts[-2]
                fnsplit = fnp.split("_")
                fnsplit[-1] = "*"
                tf_parts[-2] = "_".join([x for x in fnsplit])
                title_file_name = tf_parts[-2]
                page_file_name = str(Path(*tf_parts))
            elif acquisition == "pyabr3":
                tf_parts = list(Path(fn).parts)
                fnp = tf_parts[-3]
                fnsplit = fnp.split("_")
                fnsplit[-1] = "*"
                tf_parts[-3] = "_".join([x for x in fnsplit])
                page_file_name = str(Path(*tf_parts[:-1]))
                title_file_name = tf_parts[-3]

        title = f"\n{title_file_name!s}\n"
        title += f"Stimulus: {stim_type}, Amplifier Gain: {amplifier_gain}, Fs: {rec_freq}, HPF: {hpf:.1f}, Acq: {acquisition:s}\n"
        title += f"Subject: {subject_id:s}, Age: {age:s} Sex: {sex:s}, Strain: {strain:s}, Weight: {weight:.2f}, Genotype: {genotype:s}"

        # determine the direction for stacking the plots.
        # stack_dir = "up"
        # if metadata["type"] == "ABR4":
        #     stack_dir = "up"
        # elif metadata["type"] == "pyabr3" and stim_type.lower().startswith("click"):
        #     stack_dir = "up"
        # elif metadata["type"] == "pyabr3" and (
        #     stim_type.lower().startswith("tone") or stim_type.lower().startswith("interleaved")
        # ):
        #     stack_dir = "up"
        # else:
        #     raise ValueError(
        #         f"Stimulus type {stim_type} not recognized togethger with data type {metadata['type']}"
        #     )
        return title, page_file_name # , stack_dir

    def plot_abrs(
        self,
        abr_data: np.ndarray,
        tb: np.ndarray,
        waveana: object,
        scale: str = "V",
        ax_plot: Union[object, None] = None,
        acquisition: str = "pyabr3",
        V_stretch: float = 1.0,
        stim_type: str = "Click",
        dblist: Union[list, None] = None,
        frlist: Union[list, None] = None,
        maxdur: float = 10.0,
        thresholds: Union[list, None] = None,
        metadata: dict = {},
        use_matplotlib: bool = True,
        live_plot: bool = False,
        pdf: Union[object, None] = None,
    ):
        """plot_abrs : Make a plot of the ABR traces, either into a matplotlib figure or a pyqtgraph window.
        If ax_plot is None, then a new figure is created. If ax_plot is not None, then the plot is added to the
        current ax_plot axis.

        Parameters
        ----------
        abr_data : np.ndarray
            the abr data set to plot.
            Should be a 2-d array (db x time)
        tb : np.ndarray
            time base
        scale : str, optional
            scale representation of the data ["uV" or "V"], by default "V"
        ax_plot : Union[object, None], optional
            matplot axis to plot the data into, by default None
        acquisition : str, optional
            what kind of data is being plotted - from ABR4 or pyabr3, by default "pyabr3"
        V_stretch : float, optional
            Voltage stretch factor, by default 1.0
        stim_type : str, optional
            type of stimulus - click or tone, by default "click"
        dblist : Union[list, None], optional
            intensity values, by default None
        frlist : Union[list, None], optional
            frequency values, by default None
        maxdur : float, optional
            max trace duration, in msec, by default 14.0
        metadata : dict, optional
            a dictionary of metadata returned from reading the data file, by default {}
        use_matplotlib : bool, optional
            flag to force usage of matplotlib vs. pyqtgraph, by default True
        live_plot : bool, optional
            flag to allow a live plot (pyqtgraph), by default False
        pdf : Union[object, None], optional
            A pdf file object for the output, by default None

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        print("metadata: ", metadata)
        sninfo = Path(metadata['filename']).parent.parent.name
        subid = re_subject.search(sninfo)

        amplifier_gain = metadata["amplifier_gain"]
        if subid is not None:
            subj = subid.group("subject")
            if subj in  ["N004", "N006", "N005", "N007"]:
                amplifier_gain = 1

        if scale == "uV":
            # amplifier gain has already been accountd for.
            added_gain = 1.0
        elif scale == "V":
            added_gain = 1e6  # convert to microvolts
        else:
            raise ValueError(f"Scale {scale} not recognized, must be 'V' or 'uV'")

        if frlist is None or len(frlist) == 0 or stim_type == "click":
            frlist = [0]
            ncols = 1
            width = 5  # (1.0 / 3.0) * len(dblist) * ncols
            height = 1.0 * len(dblist)
            lmar = 0.15
        else:
            ncols = len(frlist)
            width = 2.0 * ncols
            height = 1.0 * len(dblist)
            lmar = 0.1

        stack_increment = self.experiment["ABR_settings"]["stack_increment"]
        if height > 10.0:
            height = 10.0 * (10.0 / height)
        if use_matplotlib:
            if ax_plot is None:  # no external plot to use
                P = mpl_PH.regular_grid(
                    cols=ncols,
                    rows=len(dblist),
                    order="rowsfirst",
                    figsize=(width, height),
                    verticalspacing=0.01,
                    horizontalspacing=0.03,
                    margins={
                        "leftmargin": lmar,
                        "rightmargin": 0.05,
                        "topmargin": 0.15,
                        "bottommargin": 0.10,
                    },
                )

            # check the data type to build the datasets

            title, page_file_name = self.parse_metadata(
                metadata, stim_type, acquisition
            )
            if ax_plot is None:
                P.figure_handle.suptitle(title, fontsize=8)
                ax = P.axarr
            else:
                ax = ax_plot

            v_min = 0.0
            v_max = 0.0
            n = 0
            colors = [
                "xkcd:raspberry",
                "xkcd:red",
                "xkcd:orange",
                "xkcd:golden yellow",
                "xkcd:green",
                "xkcd:blue",
                "xkcd:purple",
                "xkcd:bright violet",
            ]
            click_colors = [
                "xkcd:azure",
                "xkcd:lightblue",
                "xkcd:purple",
                "xkcd:orange",
                "xkcd:red",
                "xkcd:green",
                "xkcd:golden yellow",
            ]
            n_click_colors = len(click_colors)
            refline_ax = []
          
            for j, fr in enumerate(range(ncols)):  # enumerate(abr_data.keys()):
                for i, db in enumerate(dblist):
                    db = float(db)
                    delta_y = 0.0
                    if ax_plot is not None:  # single axis
                        ax = ax_plot
                        delta_y = stack_increment * i
                    else:   # multiple axes
                        ax = P.axarr[len(dblist) - i - 1, j]
                        delta_y = 0
                    if not i % 2:
                        ax_plot.text(
                            -0.2, delta_y, f"{int(db):d}", fontsize=8, ha="right", va="center"
                        )
                    npts = abr_data.shape[-1]
                    n_disp_pts = int(
                        maxdur * 1e-3 * metadata["record_frequency"]
                    )  # maxdur is in msec.
                    if n_disp_pts < npts:
                        npts = n_disp_pts
                    # print("added_gain: ", added_gain)
                    if abr_data.ndim > 3:
                        abr_data = abr_data.squeeze(axis = 0)
                    if stim_type == "Click":
                        plot_data = added_gain * abr_data[0, i, :] / amplifier_gain
                    else:
                        plot_data = added_gain * abr_data[i, j, :] / amplifier_gain

                    if isinstance(thresholds, list) and np.abs(db-thresholds[j]) < 1:
                        CP.cprint("c", f"** Found threshold, db: {db!s}, threshold: {thresholds[j]!s}")
                        ax.plot(
                            tb[:npts] * 1e3,
                            np.zeros_like(plot_data[:npts]) + delta_y,
                            color="xkcd:blue",
                            linewidth=4,
                            clip_on=False,
                            alpha=0.7,
                        )
                    else:
                        pass
                        # if thresholds is None:
                        #     CP.cprint("r", f"** Threshold value is NOne: db: {db!s}")

                        # else:
                        #     CP.cprint("r", f"** DID NOT find threshold, db: {db!s}, threshold: {thresholds[j]!s}")

                    if stim_type in ["Click"]:
                        ax.plot(
                            tb[:npts] * 1e3,
                            plot_data[:npts] + delta_y,
                            color=click_colors[i % n_click_colors],
                            linewidth=1,
                            clip_on=False,
                        )
                        # ax.plot(waveana.p1_latencies[i]*1e3, waveana.p1_amplitudes[i]+delta_y, 'ro', clip_on=False)
                        # ax.plot(waveana.n1_latencies[i]*1e3, waveana.n1_amplitudes[i]+delta_y, 'bo', clip_on=False)
                        marker_at = 1.0
                        ax.plot([marker_at, marker_at], [delta_y-1, delta_y+1], 'k-', clip_on=False, linewidth=0.5)
                        if i == len(dblist) - 1:
                            ax.text(marker_at, delta_y+1.5, f"{marker_at:.1f} ms", fontsize=8, ha="center", va="bottom")
                    else:
                        ax.plot(
                            (tb[:npts]) * 1e3,
                            plot_data[:npts] + delta_y,
                            color=colors[j],
                            clip_on=False,
                        )
                    ax.plot(
                        [tb[0] * 1e3, tb[npts] * 1e3],
                        [delta_y, delta_y],
                        linestyle="--",
                        linewidth=0.3,
                        color="grey",
                        zorder=0,
                    )
                    if ax not in refline_ax:
                        PH.referenceline(ax, reference=delta_y, linewidth=0.5)
                        refline_ax.append(ax)
                    # print(dir(ax))
                    ax.set_facecolor(
                        "#ffffff00"
                    )  # background will be transparent, allowing traces to extend into other axes
                    ax.set_xlim(0, maxdur)
                    # let there be an axis on one trace (at the bottom)

                    # if stack_dir == "up":
                    if i == len(dblist) - 1:
                        if ncols > 1:
                            ax.set_title(f"{frlist[j]} Hz")
                        else:
                            ax.set_title("Click")
                        PH.noaxes(ax)
                    elif i == 0:
                        PH.nice_plot(ax, direction="outward", ticklength=3)
                        ax.set_xlabel("Time (ms)")
                        ticks = np.arange(0, maxdur, 2)
                        ax.set_xticks(ticks, [f"{int(k):d}" for k in ticks])
                    else:
                        PH.noaxes(ax)

                    # elif stack_dir == "down":
                    #     if i == 0:
                    #         if ncols > 1:
                    #             ax.set_title(f"{frlist[j]} Hz")
                    #         else:
                    #             ax.set_title("Click")
                    #         PH.noaxes(ax)
                    #     elif i == len(dblist) - 1:
                    #         PH.nice_plot(ax, direction="outward", ticklength=3)
                    #         ticks = np.arange(0, maxdur, 2)
                    #         ax.set_xticks(ticks, [f"{int(k):d}" for k in ticks])
                    #         ax.set_xlabel("Time (ms)")
                    #     else:
                    #         PH.noaxes(ax)

                    if j == 0:
                        ax.set_ylabel(
                            f"dB SPL",
                            fontsize=8,
                            labelpad=18,
                            rotation=90,
                            ha="center",
                            va="center",
                        )
                        if i == 0:
                            muv = r"\u03BC"
                            PH.calbar(
                                ax,
                                calbar=[-2.5, stack_increment * len(dblist), 1, 1],
                                unitNames={"x": "ms", "y": f"uV"},
                                xyoffset=[0.5, 0.1],
                                fontsize=6,
                            )
                    # ax.set_xlim(0, np.max(tb[:npts]) * 1e3)
                    ax.set_xlim(0, maxdur)

                # ax.set_xticks([1, 3, 5, 7, 9], minor=True)

                n += 1
                if np.max(plot_data[:npts]) > v_max:
                    v_max = np.max(plot_data[:npts])
                if np.min(plot_data[:npts]) < v_min:
                    v_min = np.min(plot_data[:npts])

            # if metadata["type"] == "pyabr3" and stim_type.lower().startswith("click"):
            #     V_stretch = 10.0 * V_stretch

            amax = np.max([-v_min, v_max]) * V_stretch
            if amax < 0.5:
                amax = 0.5
            # print(P.axarr.shape, len(dblist), len(frlist))

            for isp in range(len(dblist)):
                if ax_plot is None:
                    for j in range(len(frlist)):
                        P.axarr[isp, j].set_ylim(-amax, amax)
                        P.axarr[isp, j].set_xlim([0, maxdur])
                    # PH.referenceline(ax, linewidth=0.5)
            if ax_plot is None:
                transform = P.figure_handle.transFigure
            else:
                transform = ax_plot.transAxes
            mpl.text(
                0.96, 0.01, s=datetime.datetime.now(), fontsize=6, ha="right", transform=transform
            )
            mpl.text(
                0.02,
                0.01,
                s=f"{page_file_name:s}",
                fontsize=5,
                ha="left",
                transform=transform,
            )

        # else:  # use pyqtgraph
        #     app = pg.mkQApp("ABR Data Plot")
        #     win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        #     win.resize(1200, 1000)
        #     win.setWindowTitle(f"File: {str(Path(fn).parent)}")
        #     win.setBackground("w")

        #     lw = pg.LayoutWidget(parent=win)
        #     lw.addLabel(text="Hi there", row=0, col=0, rowspan=1, colspan=len(frlist))
        #     lw.nextRow()

        #     plid = []
        #     if len(frlist) == 0:
        #         frlist = [1]
        #     col = 0

        #     print("stim_type (in pg plotting)", stim_type)
        #     if stim_type not in ["clicks", "click"]:  # this is for tones/interleaved, etc.
        #         ref_set = False
        #         v_min = 0
        #         v_max = 0
        #         for i, db in enumerate(dblist):
        #             row = i  # int(i/5)
        #             for j, fr in enumerate(frlist):
        #                 col = j
        #                 # if tb is None:
        #                 #     npts = len(abr_data[i, j])
        #                 #     tb = np.linspace(0, npts / rec_freq, npts)

        #                 pl = win.addPlot(
        #                     title=f"{dblist[-i-1]} dB, {fr} Hz", col=col, row=row
        #                 )  # i % 5)
        #                 if not ref_set:
        #                     ref_ax = pl
        #                     ref_set = True
        #                 plot_data = 1e6 * abr_data[len(dblist) - i - 1, j] / amplifier_gain
        #                 lpd = len(plot_data)
        #                 if stim_type in ["click", "tonepip"]:
        #                     pl.plot(
        #                         tb[:lpd] * 1e3,
        #                         plot_data,
        #                         pen=pg.mkPen(j, len(dblist), width=2),
        #                         clipToView=True,
        #                     )
        #                 else:
        #                     pl.plot(
        #                         tb[:lpd] * 1e3,
        #                         plot_data,
        #                         pen=pg.mkPen(j, len(dblist), width=2),
        #                         clipToView=True,
        #                     )
        #                 pl.plot(
        #                     tb[:lpd] * 1e3,
        #                     np.zeros_like(plot_data),
        #                     pen=pg.mkPen(
        #                         "grey", linetype=pg.QtCore.Qt.PenStyle.DashLine, width=0.33
        #                     ),
        #                     clipToView=True,
        #                 )
        #                 # pl.showGrid(x=True, y=True)
        #                 if j == 0:
        #                     pl.setLabel("left", "Amp", units="uV")
        #                 if i == len(dblist) - 1:
        #                     pl.setLabel("bottom", "Time", units="ms")
        #                 pl.setYRange(-3.0, 3.0)
        #                 pl.setXRange(0, 10)
        #                 if ref_set:
        #                     pl.setXLink(ref_ax)
        #                     pl.setYLink(ref_ax)
        #                 if np.max(plot_data) > v_max:
        #                     v_max = np.max(plot_data)
        #                 if np.min(plot_data) < v_min:
        #                     v_min = np.min(plot_data)

        #         ref_ax.setYRange(v_min * V_stretch, v_max * V_stretch)

        #     else:  # pyqtgraph
        #         v0 = 0
        #         v = []
        #         for i, db in enumerate(dblist):
        #             if i == 0:
        #                 pl = win.addPlot(title=f"{db} dB, {fr} Hz")  # i % 5)
        #             pl.plot(
        #                 tb * 1e3,
        #                 -v0 + abr_data[i, j] / amplifier_gain,
        #                 pen=pg.mkPen(pg.intColor(i, hues=len(dblist)), width=2),
        #                 clipToView=True,
        #             )
        #             v0 += 1e-6 * amplifier_gain
        #             v.append(v0)
        #             # pl.showGrid(x=True, y=True)
        #             pl.setLabel("left", "Amplitude", units="mV")
        #             pl.setLabel("bottom", "Time", units="s")
        #             label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
        #             label.setParentItem(pl)
        #             label.anchor(itemPos=(0.05, -v0 * 180), parentPos=(0.1, 0))
        #             # pl.setYRange(-2e-6, 2e-6)
        #             plid.append(pl)
        #         for i, db in enumerate(dblist):
        #             label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
        #             label.setParentItem(pl)
        #             label.anchor(itemPos=(0.05, -v[i] * 200), parentPos=(0.1, 0))
        #             # win.nextRow()
        #             for j, fr in enumerate(frlist):
        #                 ax.set_title(f"{self.convert_attn_to_db(db, fr)} dBSPL, {fr} Hz")
        #                 if i == 0:
        #                     ax.set_xlabel("Time (s)")
        #                 if j == 0:
        #                     ax.set_ylabel("Amplitude")
        #                 ax.set_ylim(-50, 50)
        #                 PH.noaxes(ax)
        #                 if i == 0 and j == 0:
        #                     PH.calbar(
        #                         ax,
        #                         calbar=[0, -20, 2, 10],
        #                         scale=[1.0, 1.0],
        #                         xyoffset=[0.05, 0.1],
        #                     )
        #                 PH.referenceline(ax, linewidth=0.5)
        #                 n += 1

        #     pg.exec()

        # # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)

    # def check_fsamp(self, d):
    #     if d["record_frequency"] is None:
    #         raise ValueError()
    #     d["record_frequency"] = 24414.0625  # 97656.25
    #     return d["record_frequency"]

    def read_and_average_abr_files(
        self,
        filename,
        amplifier_gain=1e4,
        scale: str = "V",
        high_pass_filter: Union[float, None] = None,

        maxdur: Union[float, None] = None,
        pdf: Union[object, None] = None,
    ):
        d = self.read_abr_file(filename)
        print("     Read and average abrs")

        if maxdur is None:
            maxdur = 25.0
        stim_type = d["protocol"]["protocol"]["stimulustype"]
        fd = Path(filename).parent
        fns = Path(fd).glob("*.p")
        # break the filenames into parts.
        # The first part is the date,
        # the second part is the stimulus type,
        # the third part is the index into the stimulus array
        # the fourth part is the repetition number for a given stimulus
        protocol = d["protocol"]
        rec = protocol["recording"]
        # print("rec: ", rec)
        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        if isinstance(dblist, str):
            dblist = eval(dblist)
        if isinstance(frlist, str):
            frlist = eval(frlist)

        subject_id = d["subject_data"]["Subject ID"]
        if d["subject_data"]["Age"] != "":
            age = PA.ISO8601_age(d["subject_data"]["Age"])
        else:
            age = 0
        sex = d["subject_data"]["Sex"]
        strain = d["subject_data"]["Strain"]
        if d["subject_data"]["Weight"] != "":
            weight = float(d["subject_data"]["Weight"])
        else:
            weight = 0.0
        genotype = d["subject_data"]["Genotype"]

        file_parts = Path(filename).stem.split("_")
        # print(file_parts)
        date = file_parts[0]
        stim_type = file_parts[1]
        print("stim type(before...): ", stim_type)
        if len(frlist) == 0:
            frlist = [1]

        if stim_type in ["Click", "click", "Tone"]:
            n = 0

            for i, db in enumerate(dblist):
                for j, fr in enumerate(frlist):
                    x, tb, sample_rate = self.average_within_traces(
                        fd,
                        n,
                        protocol,
                        date,
                        high_pass_filter=high_pass_filter,
                    )
                    if i == 0 and j == 0:
                        abr_data = np.zeros((len(dblist), len(frlist), len(x)))
                    abr_data[i, j] = x
                    n += 1
            print("raaabrf:: sample rate: ", sample_rate, "max time: ", np.max(tb))
            

        print("read and average py3abr: stim type:: ", stim_type)
        if stim_type in ["interleaved"]:  # a form of Tones... 
            n = 0
            abr_data, tb, sample_rate, stim = self.average_across_traces(
                fd, n, protocol, date, high_pass_filter=high_pass_filter
            )
            # print(len(frlist), len(dblist))
            abr_data = abr_data.reshape(len(dblist), len(frlist), -1)
            # print("calculated new tb")

        # print("pyabr3_data shape: ", abr_data.shape)
        # print("pyabr3 db list: ", dblist)
        if dblist[0] > dblist[-1]:  # need to reverse data order and dblist order
            dblist = dblist[::-1]
            abr_data = np.flip(abr_data, axis=0)
        print("sample_rate: " , sample_rate, 1./np.mean(np.diff(tb)))
        amp_gain = 1e4

        metadata = {
            "type": "pyabr3",
            "filename": filename,
            "subject_id": subject_id,
            "age": age,
            "sex": sex,
            "stim_type": stim_type,
            "stimuli": {"dblist": dblist, "freqlist": frlist},
            "amplifier_gain": amp_gain,
            "strain": strain,
            "weight": weight,
            "genotype": genotype,
            "record_frequency": sample_rate,
            "highpass": high_pass_filter,
        }

        if pdf is not None:
            self.plot_abrs(
                abr_data=abr_data,
                tb=tb,
                waveana = waveana,
                stim_type=stim_type,
                scale=scale,
                dblist=dblist,
                frlist=frlist,
                metadata=metadata,
                maxdur=maxdur,
                highpass=300.0,
                use_matplotlib=True,
                pdf=pdf,
            )
        print("max time base rad and average: " , np.max(tb))
        return abr_data, tb, metadata


def plot_click_stack(
    AR, ABR4, filename, directory_name: str = None, ax: Union[object, None] = None
):
    AR = AnalyzeABR()
    ABR4 = read_abr4.READ_ABR4()
    maxdur = AR.experiment["ABR_settings"]["maxdur"]
    HPF = AR.experiment["ABR_settings"]["HPF"]
    LPF = AR.experiment["ABR_settings"]["LPF"]
    stack_increment = AR.experiment["ABR_settings"]["stack_increment"]
    
    pdf = None
    if ax is None and pdf is None:
        raise ValueError("Must provide either an axis or a pdf file")

    print("Click file: ", filename)
    pfiles = list(Path(filename).glob("*_click_*.p"))
    if len(pfiles) > 0:
        AR.read_and_average_abr_files(
            filename=str(pfiles[0]), high_pass_filter=HPF, maxdur=maxdur, pdf=pdf
        )
    else:
        f = Path(filename)
        ABR4.plot_dataset(
            AR,
            datatype="click",
            subject=f.parent,
            topdir=directory_name,
            subdir=f.name,
            highpass=HPF,
            maxdur=maxdur,
            pdf=pdf,
        )


def do_directory(
    AR: AnalyzeABR,
    ABR4: object,
    directory_name: Union[Path, str],
    output_file: Union[Path, str],
    subject_prefix: str = "CBA",
    hide_treatment: bool = False,
):

    AR.set_hide_treatment(hide_treatment)

    subjs = Path(directory_name).glob(f"{subject_prefix}*")
    with PdfPages(output_file) as pdf:
        for subj in subjs:
            if not subj.is_dir() or subj.name.startswith("."):
                continue
            fns = subj.glob("*")
            for f in fns:
                if fname.startswith("Click"):
                    plot_click_stack(AR, ABR4, f, directory_name, ax=None)
                elif fname.startswith("Tone"):
                    print("Tone file: ", f)
                    ABR4.plot_dataset(
                        AR,
                        datatype="tone",
                        subject=f.parent,
                        topdir=directory_name,
                        subdir=f.name,
                        pdf=pdf,
                    )
                elif fname.startswith("Interleaved"):
                    print("Interleaved file: ", f)
                    files = list(Path(f).glob("*.p"))
                    print(f, "\n     # interleaved files: ", len(files))

                    AR.read_and_average_abr_files(
                        filename=str(files[0]),
                        pdf=pdf,
                    )
                else:
                    raise ValueError(f"File {f} for data file {fname:s} not recognized")


def get_treatment(subject):
    """get_treatment split off the treatment information from the subject name/file

    Parameters
    ----------
    subject : str
        directory name for data for this subject

    Returns
    -------
    The last underscore separated part of the subject name, which is the treatment
        treatment
    """
    subparts = subject.name.split("_")
    return subparts[-1]


def get_age(subject):
    """get_age split off the treatment information from the subject name/file
        age_categories:
            Preweaning: [7, 20]
            Pubescent: [21, 49]
            Young Adult: [50, 179]
            Mature Adult: [180, 364]
            Old Adult: [365, 1200]

    Parameters
    ----------
    subject : str
        directory name for data for this subject
    position: int
        position of the age in the subject name
    Returns
    -------
    The last underscore separated part of the subject name, which is the treatment
        treatment
    """
    subject_name = subject["name"]
    m = re_subject.match(subject_name)
    if m is None:
        raise ValueError(f"Cannot find age in {subject.name}")
    age = m.group("age")
    i_age = PA.age_as_int(PA.ISO8601_age(age))
    if i_age <= 20:
        age_str = "Preweaning"
    elif 21 <= i_age <= 49:
        age_str = "Pubescent"
    elif 50 < i_age <= 179:
        age_str = "Young Adult"
    elif 180 < i_age <= 364:
        age_str = "Mature Adult"
    elif i_age > 365:
        age_str = "Old Adult"
    else:
        print("age not found: ", i_age, age)
        raise ValueError(f"Age {age} not found in {subject_name}")
    return age_str


def get_categories(subjects, categorize="treatment"):
    """get_categories Find all of the categories in all of the subjects

    Parameters
    ----------
    subjects : list
        list of subject directories

    Returns
    -------
    list
        list of unique categories across all subjects
    """
    categories = []
    for subj in subjects:
        # print("subj: ", subj)
        # if subj.name.startswith("."):
        # continue
        if categorize == "treatment":
            category = get_treatment(subj)
        elif categorize == "age":
            category = get_age(subj)
        # print("treatment: ", treatment)
        if category not in categories:
            categories.append(category)

    return categories


def get_analyzed_click_data(filename, AR, ABR4, subj, HPF, maxdur, scale:float=1.0, invert:bool=False):
    pfiles = list(Path(filename).glob("*_click_*.p"))
    # f, ax = mpl.subplots(1, 1)  # for a quick view of the data
    if len(pfiles) > 0:
        waves, tb, metadata = AR.read_and_average_abr_files(
            filename=str(pfiles[0]), high_pass_filter=HPF, maxdur=maxdur, pdf=None
        )
        sym = "D"
        # print("metadata: ", metadata, pfiles[0])

        waves *= scale
        if invert:
            waves = -waves
        print("get analyzed click data:: metadata: ", metadata)
        print("gacd: wave shape: ", waves.shape)
        print("gacd: Max time: ", np.max(tb), "sample rate: ", metadata["record_frequency"])

    else:
        # sample frequency for ABR4 depends on when the data was collected. 
        # so we set it this way:
        subj_id = re_subject.match(subj["name"]).group("subject")
        # print(subj_id)
        if subj_id[0] == "R": # Ruili data, 100 kHz
            sample_frequency = 100000.
        elif subj_id[0] == "N":  # Reggie data, 50 kHz
            sample_frequency = 50000.
        else:
            print("Sample frequency cannot be determined with the information provided")
            raise ValueError(f"Subject ID {subj_id} not recognized")

        waves, tb, metadata = ABR4.read_dataset(
            subject=subj,
            datapath=subj["filename"],
            subdir=filename.name,
            datatype="Click",
            sample_frequency=sample_frequency,
            highpass=HPF,
        )
        waves *= scale
        if invert:
            waves = -waves
        # print("Read waves, shape= ", waves.shape)
        # print("metadata: ", metadata)
    dbs = metadata["stimuli"]["dblist"]
    # for i, db in enumerate(dbs):
    #     ax.plot(tb, waves[i, 0, :]/metadata["amplifier_gain"])
    # mpl.show()

    if waves is None:  # no data for this stimulus type
        return None
    # print(np.max(waves))
 
    WaveAnalyzer.analyze(timebase=tb, waves=waves[:, 0, :])
    dbs = metadata["stimuli"]["dblist"]
    # ABR4.dblist
    print("  get analyzed click data: dbs: ", dbs)
    return WaveAnalyzer, dbs, metadata

def get_analyzed_tone_data(filename, AR, ABR4, subj, HPF:float=100., maxdur:float=10., scale:float=1.0):
    pfiles = list(Path(filename).glob("*_interleaved_plateau_*.p"))
    print("pfiles: ", len(pfiles))
    if len(pfiles) > 0:
        waves, tb, metadata = AR.read_and_average_abr_files(
            filename=str(pfiles[0]), high_pass_filter=HPF, maxdur=maxdur, pdf=None
        )
        sym = "D"
    else:
        # sample frequency depends on whenth e data was collected. 
        # so we set it this way:
        subj_id = re_subject.match(subj["name"]).group("subject")


        waves, tb, metadata = ABR4.read_dataset(
            subject=subj['name'],
            datapath=subj["filename"],
            subdir=filename.name,
            datatype="Tone",
            highpass=HPF,
            scale=scale,
        )
        # print("Read waves, shape= ", waves.shape)
        # print("metadata: ", metadata)

    if waves is None:  # no data for this stimulus type
        return None
    # print(waves.shape)
    # print("metadata: ", metadata)
    # f, ax = mpl.subplots(1,1)
    subject_threshold = []
    subject_frequencies = []
    # print(metadata['stimuli'])
    if "frlist" in metadata['stimuli']:
        freq_list = metadata['stimuli']['frlist']
    elif "freqlist" in metadata['stimuli']:
        freq_list = metadata['stimuli']['freqlist']
    else:
        raise ValueError("No frequency list in metadata", metadata['stimuli'])    
    
    c = mpl.colormaps["rainbow"](np.linspace(0, 1, waves.shape[1]))

    for ifr in range(waves.shape[1]):
        WaveAnalyzer.analyze(timebase=tb, waves=waves[:, ifr, :])
        # WaveAnalyzer.ppio[np.isnan(WaveAnalyzer.ppio)] = 0
        # print("waves analyzed: ", WaveAnalyzer.ppio)
        # print("baseline: ", WaveAnalyzer.rms_baseline)
        valid_dbs = range(waves.shape[0])   # use this for ppio: #np.where(np.isfinite(WaveAnalyzer.ppio))[0]
        dbx = np.array(metadata['stimuli']['dblist'])[valid_dbs]
        spec_power = WaveAnalyzer.specpower(fr=[750, 1300], win=[1e-3, 8e-3], spls=metadata['stimuli']['dblist'])
        spec_noise = WaveAnalyzer.specpower(fr=[750, 1300], win=[1e-3, 8e-3], spls=metadata['stimuli']['dblist'],
            get_reference=True, i_reference=np.argmin(metadata['stimuli']['dblist']))  # use the lowest stimulus level to get the reference spectral baseline
        # print("valid)", valid)
        # ax.plot(metadata['stimuli']['dblist'], WaveAnalyzer.ppio, color=c[ifr], marker='o', linewidth=0, markersize=3, label=f"{freq_list[ifr]:.1f} Hz")
        # print("Waveana rms baseline: ", WaveAnalyzer.rms_baseline)
        threshold_value, threshold_index, fit = fit_thresholds.fit_thresholds(
                dbx,
                WaveAnalyzer.ppio[valid_dbs],
                # WaveAnalyzer.psdwindow[valid_dbs],
                # WaveAnalyzer.psdwindow[0],
                # WaveAnalyzer.reference_psd,
                WaveAnalyzer.rms_baseline,
                threshold_factor=AR.experiment["ABR_settings"]["tone_threshold_factor"],
            )
        frequency = float(freq_list[ifr])
        # mpl.plot(fit[0], fit[1], '-', color=c[ifr])
        threshold_value = round(threshold_value/2)*2  # round to nearest 2 dB
        subject_threshold.append(float(threshold_value))
        subject_frequencies.append(float(frequency))
    print("thresholds: ", subject_threshold, subject_frequencies)
    # ax.plot()
    # mpl.show()
    dbs = metadata["stimuli"]["dblist"]
    # ABR4.dblist
    print("  got analyzed tone data: dbs: ", dbs)
    return WaveAnalyzer, dbs, metadata, subject_threshold, subject_frequencies

def clean_subject_list(subjs):
    """clean_subject_list Remove cruft from the subject directory list.
    Things like hidden files, or unrecongized subdirectories, are deleted
    from the list.

    Parameters
    ----------
    subjs : list
        list of subject directoryies, Path objects
    """
    for subj in subjs:
        if not subj.is_dir():
            subjs.remove(subj)
        elif (
            subj.name.startswith(".")
            or subj.name.startswith(".DS_Store")
            or subj.name.endswith("Organization")
        ):
            subjs.remove(subj)
    return subjs


def _plot_io_data(
    waveana: object,
    dbs: list,
    V2uV: float,
    color: str,
    ax: object,
    symdata: list,
    threshold: float = None,
    threshold_index: int = None,
    dataset_name: str = None,
    add_label: bool = False,
):

    # CP.cprint("g", f"len(ppio), dbs: {len(waveana.ppio):d}, {len(dbs):d}")
    if len(waveana.ppio) != len(dbs):
        return  # skip

    if add_label:
        label = dataset_name[1]
    else:
        label = None
    if dataset_name[1] == "Old Adult":
        mec = "#33333380"
        melw = 0.5
    else:
        mec = color
        melw = 0.5
    linew = 0.33
    dbs = np.array(dbs)
    iwaves = np.argwhere((20.0 <= dbs) & (dbs <= 90.0))
    # v = np.array([float(w[1]) for w in waveana.p1n1_amplitudes])[iwaves]
    ax.plot(
        dbs[iwaves],
        waveana.ppio[iwaves] * V2uV,
        color=mec,
        linestyle="-",  # f"{color:s}{sym:s}-",
        linewidth=linew,
        # marker=symdata[0],
        # markersize=symdata[1],
        # markerfacecolor=color,
        # markeredgecolor=mec,
        # markeredgewidth=melw,
        alpha=0.33,
        label=label,
    )

    # if threshold is not None:
    #     ax.plot(
    #         dbs[threshold_index],
    #         waveana.ppio[threshold_index] * V2uV,
    #         color=color,
    #         marker="o",
    #         markersize=4,
    #         markerfacecolor="w",
    #     )


def plot_click_stacks(
    AR: object,
    example_subjects: list,
    filename: Union[str, Path],
    metadata: dict = None,
    dbs: list = None,
    thresholds: list = None,
    waveana: object = None,
    axlist: list = None,
):

    if len(axlist) != len(example_subjects):
        raise ValueError("Number of axes must match number of example subjects")
    subject = filename.parts[-2]
    if subject in example_subjects:
        fnindex = example_subjects.index(subject)
        waveana.waves = np.expand_dims(waveana.waves, 0)
        AR.plot_abrs(
            abr_data=waveana.waves,
            tb=waveana.timebase,
            scale="V",
            waveana=waveana,
            acquisition="pyabr3",
            V_stretch=1.0,
            metadata=metadata,
            stim_type="Click",
            dblist=dbs,
            frlist=None,
            thresholds = thresholds,
            maxdur=AR.experiment["ABR_settings"]["maxdur"],
            ax_plot=axlist[fnindex],
        )
    else:
        # print("Subject: ", subject, "not in examples: ", example_subjects)
        return


def remap_xlabels(ax):
    remapper = {
        "Preweaning": "PW",
        "Pubescent": "PB",
        "Young Adult": "YA",
        "Mature Adult": "MA",
        "Old Adult": "OA",
    }
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, label in enumerate(labels):
        if label in remapper:
            labels[i] = remapper[label]
    ax.set_xticks(range(len(labels)))  # need to do this to make sure ticks and labels are in sync
    ax.set_xticklabels(labels)

def set_gain_and_scale(subj, AR):
    # Set minimum latency and gain values from the configuratoin file
    if "ABR_parameters" not in AR.experiment.keys():
        raise ValueError("'ABR_parameters' missing from configuration file")
    
    scd = AR.experiment["ABR_parameters"]  # get the parameters dictionary
    if 'default' not in scd.keys():
        raise ValueError("'default' values missing from ABR_parameters")
    
    # print("Parameters: ", scd)
    # set the defaults.
    scale = scd['default']['scale']
    invert = scd['default']['invert']
    min_lat = scd['default']["minimum_latency"]
    # print("ABR keys: ", list(AR.experiment["ABR_parameters"].keys()))

    # if the full name is in the parameter list, use it instead of any defaults
    if subj["name"] in list(AR.experiment["ABR_parameters"].keys()):
        CP.cprint("g", f"\nsubject name: {subj['name']!s} is in experiment ABR_parameters list")
        scale = scd[subj["name"]]["scale"]
        invert = scd[subj["name"]]["invert"]
        min_lat = scd[subj["name"]]["minimum_latency"]
        fit_index = scd[subj["name"]]["fit_index"]

    else:
        CP.cprint("r", f"\nsubject name: {subj['name']!s} is NOT experiment ABR_parameters list - checking abbreviated versions")
        smatch = re_subject.match(subj["name"])
        CP.cprint("m", f"SMATCH: {smatch}, {subj['name']:s}")
        if smatch["subject"] is not None:
            sname = smatch["subject"]
            if sname.startswith("N0"):
                scale = scd['N0']['scale']
                invert = scd['N0']['invert']
                min_lat = scd['N0']["minimum_latency"]
                fit_index = scd['N0']["fit_index"]
            elif sname.startswith("T0"):
                scale = scd['T0']['scale']
                invert = scd['T0']['invert']
                min_lat = scd['T0']["minimum_latency"]
                fit_index = scd['T0']["fit_index"]
            else:
                raise ValueError(f"Subject name {sname:s} not recognized as a category for scaling")
        else:
            raise ValueError(f"Subject name {subj['name']:s} not in configuration ABR_parameters dictionary")
    return scale, invert, min_lat, fit_index

def do_one_subject(
    subj: dict,
    treat: str,
    requested_stimulus_type: str,
    AR: object,
    ABR4: object,
    V2uV: float,
    example_subjects: list = None,
    example_axes: list = None,
    test_plots: bool=False
):
    """do_one_subject : Compute the mean ppio, threshold and IO curve for one subject.

    Raises
    ------
    ValueError
        _description_
    """
    subject_threshold = []
    subject_threshold_unadjusted = []
    subject_ppio = []
    donefiles = []
    dbs = []
    threshold_index = None
    waveana = None  # in case the analysis fails or the dataset was excluded
    # print("requested file type: ", requested_stimulus_type)
    if requested_stimulus_type in ["Click", "Tone", "Interleaved_plateau"]:
        fns = list(subj["filename"].glob("*"))
    else:
        raise ValueError("Requested stimulus type not recognized")
    stim_type = None
    fns = [f for f in fns if not f.name.startswith(".")]
    for filename in sorted(fns):
        # check to see if we have done this file already (esp. ABR4 files)
        fname = filename.name
        if fname.startswith("."):  # skip hidden files
            continue

        filematch = re_splfile.match(fname)  # old ABR4 file type for click
        if filematch is not None:
            if filematch.group(1) in donefiles:
                break
            else:
                # print(f"    ABR4 File {fname:s} now being processed...")
                donefiles.append(filematch.group(1))
        else:
            # might be a new pyabr3 click file?
            if fname.endswith(".p"):
                if fname[:15] in donefiles:
                    break
                else:
                    # print(f"    pyabr3 File {fname:s} has not been processed, continuing")
                    donefiles.append(fname[:15])
        scale = 1

        CP.cprint("c", f"Testing for subject name: {subj['name']:s}, {subj['name'][6]:s}")

        if subj["name"] in list(AR.experiment["ABR_subject_excludes"].keys()):
            # print("Exclusion files: ", AR.experiment["ABR_subject_excludes"].keys())
            CP.cprint("r", f"Excluding subject file: {subj['name']: s}")
            continue
        
        scale, invert, min_lat, fit_index = set_gain_and_scale(subj, AR)

        fname = filename.name.lower()
        # determine the stimulus type.
        stim_type = None
        print("subject data type: ", subj["filename"].name, " but requested type: ", requested_stimulus_type)
        if subj["filename"].name != requested_stimulus_type:
            return subject_threshold, None, None, None, None, None
        match requested_stimulus_type:
            case "Click":
                if (
                    subj["filename"].name == "Click"
                    or fname.endswith("-n.txt")
                    or fname.endswith("-p.txt")
                ):
                    stim_type = "Click"
            case "Tone":
                if (
                    (subj["filename"].name == "Tone") 
                    or 
                    (subj["filename"].name == "Interleaved_plateau")
                ):
                    stim_type = "Tone"
            case "Interleaved_plateau":
                if subj["filename"].name == "Interleaved_plateau":
                    stim_type = "Interleaved_plateau"

            case _:
                print("Requested stimulus type not recognized", requested_stimulus_type)
                raise ValueError
        CP.cprint("g", f"    ***** do one subj: stim_type: {stim_type!s}, {fname:s}")
        if stim_type is None:
            return subject_threshold, None, None, None, None, None
        # print("do one sub req ::  stim type: ", requested_stimulus_type)
        if requested_stimulus_type == "Click":
            filename = Path(filename).parent
            CP.cprint("g", f"    ***** do one sub: filename: {filename!s}")
            waveana, dbs, metadata = get_analyzed_click_data(
                filename,
                AR,
                ABR4,
                subj,
                AR.experiment["ABR_settings"]["HPF"],
                AR.experiment["ABR_settings"]["maxdur"],
                scale=scale,
                invert=invert,
            )
            CP.cprint("g", f"    ***** do one subj: dbs: {dbs!s}")
            waveana.get_triphasic(min_lat=min_lat, dev=3)

            waveana.rms_baseline = waveana.rms_baseline # * 1./metadata["amplifier_gain"]
            # adjust the measures to follow a line of latency when the response gets small.
            waveana.adjust_triphasic(dbs, threshold_index=fit_index, window=0.0005)  
            waveana.ppio = waveana.ppio * 1./metadata["amplifier_gain"]

            metadata["stimuli"]["dblist"] = dbs

            # here we check to see if our current file is one of the example
            # subjects, and plot the traces if it is.
            plot_click_stacks(
                AR,
                example_subjects=example_subjects,
                filename=filename,
                waveana=waveana,
                metadata=metadata,
                dbs=dbs,
                axlist=example_axes,
            )

            if np.max(waveana.ppio) > 10:
                CP.cprint(
                    "r",
                    f"     {filename!s} has a max ppio of {np.max(waveana.ppio) :.2f} uV, which is too high",
                )
                raise ValueError
            above_thr = np.where(waveana.ppio > 3.0 * np.mean(waveana.rms_baseline))[0]

            threshold_index = None

            if subj["datatype"] == "pybar3":
                ref_db = (
                    10.0  # use first trace set at the lowest stimulus intensity (below threshold)
                )
            else:
                ref_db = None
            
            # thr_value, threshold_index, rms = waveana.thresholds(
            #     timebase=waveana.timebase,
            #     waves=waveana.waves,
            #     spls=dbs,
            #     response_window=[1.0e-3, 8e-3],
            #     baseline_window=[0, 0.8e-3],
            #     ref_db = ref_db,
            # )
            # fit with Hill function and get the threshold
            # from the baseline.
            # Note, we fit to the top of the dataset, in case there is
            # some non-monotonicity in the data.
            imax_index = np.argmax(waveana.p1n1_amplitudes)
            imax_amp = list(range(imax_index))
            # print("rms baseline: ", waveana.rms_baseline)
            threshold_value, threshold_index, fit = fit_thresholds.fit_thresholds(
                x=np.array(dbs)[imax_amp],
                y=np.array(waveana.p1n1_amplitudes)[imax_amp],
                baseline = waveana.rms_baseline,
                threshold_factor=AR.experiment["ABR_settings"]["click_threshold_factor"]
            )
            
            db_steps = np.abs(np.mean(np.diff(dbs)))
            if requested_stimulus_type == "Click":
                threshold_value_adj = round(threshold_value/2.5)*2.5  # round to nearest 2.5 dB
                threshold_value_unadj = dbs[(np.abs(dbs - threshold_value)).argmin()]
            subject_threshold.append(threshold_value_adj)
            subject_threshold_unadjusted.append(threshold_value_unadj)
            CP.cprint("m", f"    *****  do one subject: threshold: {threshold_value:.2f} dB")
            subject_ppio.append(float(np.max(waveana.ppio) * V2uV))
            

            if test_plots:
                f, ax = mpl.subplots(1,2, figsize = [8, 5])
                # print(filename, subj["filename"].name)
                plot_click_stacks(
                    AR,
                    example_subjects=[subj["filename"].parent.name],
                    filename=filename,
                    waveana=waveana,
                    metadata=metadata,
                    dbs=dbs,
                    thresholds=subject_threshold_unadjusted,
                    axlist=[ax[0]],
                )

                stacki = AR.experiment["ABR_settings"]["stack_increment"]
                dy = stacki*np.array(range(len(dbs)))
                ax[0].plot(waveana.fitline_p1_lat*1e3, dy+waveana.p1_amplitudes*1e6, '-o', color='r', linewidth=0.3)
                ax[0].plot(waveana.fitline_n1_lat*1e3, dy+waveana.n1_amplitudes*1e6, '-o', color='b', linewidth=0.3)
                
                ax[1].plot(dbs, waveana.p1n1_amplitudes, 'o-', color='k')
                ax[1].plot(fit[0], fit[1], '-', color='r')

                ax[0].set_title(f"{subj['filename'].name:s}")
                mpl.show()
        
        elif requested_stimulus_type in ["Tone", "Interleaved_plateau"]:
            filename = Path(filename).parent
            CP.cprint("g", f"    ***** do one sub: Tones, filename: {filename!s}")
            
            waveana, dbs, metadata, thresholds, frequencies = get_analyzed_tone_data(
                filename,
                AR,
                ABR4,
                subj,
                AR.experiment["ABR_settings"]["HPF"],
                AR.experiment["ABR_settings"]["maxdur"],
                scale=scale,
            )
            print("thr, freq: ", thresholds, frequencies)
            subject_threshold = [thresholds, frequencies]
        
        else:
            print("stimulus type not recognized", requested_stimulus_type)
            exit()

    if len(subject_threshold) == 0:
        subject_threshold = [np.nan]
    if waveana is None:
        return subject_threshold, None, None, None, None, None
    CP.cprint(
        "g",
        f"\n   *****  do one subject ({fns[0]!s}): # of thresholds measured:  {len(subject_threshold):d}, thrs: {subject_threshold!s}\n",
    )
    return subject_threshold, subject_ppio, dbs, threshold_index, waveana, stim_type

def stuffs(filename, sub, directory_name, HPF, maxdur):
    fname = filename.name
    if fname.startswith("Tone"):

        print("Tone file: ", filename)
        pfiles = list(Path(filename).glob("*_Interleaved_plateau_*.p"))
        if len(pfiles) > 0:
            waves, tb, metadata = AR.read_and_average_abr_files(
                filename=str(pfiles[0]), high_pass_filter=HPF, maxdur=maxdur, pdf=None
            )
            sym = "D"
        else:
            w, t = ABR4.read_dataset(
                subject=subj,
                datapath=directory_name,
                subdir=filename.name,
                datatype="tone",
                highpass=HPF,
            )
        if w is None:  # no data for this stimulus type
            return
    elif fname.lower().startswith("interleaved"):

        print("Interleaved file: ", filename)
        files = list(Path(filename).glob("*.p"))
        print(filename, "\n     # interleaved files: ", len(files))
        HPF = 300.0

        AR.read_and_average_abr_files(
            filename=str(files[0]),
            high_pass_filter=HPF,
            maxdur=maxdur,
            pdf=None,
        )


def compute_tone_thresholds(
    AR,
    ABR4,
    categorize: str = "treatment",
    requested_stimulus_type: str = "Tone",
    subjs: Union[str, Path, None] = None,
    thr_mapax=None,
    example_axes: list = None,
    example_subjects: list = None,
    categories_done: list = None,
    symdata: dict = None,
    test_plots: bool=False,
):
    overridecolor = False
    if categorize in ["treatment"]:
        categories = get_categories(subjs)
    elif categorize in ["age"]:
        categories = get_categories(subjs, categorize="age")
    else:
        raise ValueError(f"Category type {categorize} not recognized")
    # sort categories by the table in the configuration file
    ord_cat = []
    for cat in AR.experiment["plot_order"]["age_category"]:
        if cat in categories:
            ord_cat.append(cat)
    categories = ord_cat
    colors = ["r", "g", "b", "c", "m", "y", "k", "skyblue", "orange", "purple", "brown"]

    plot_colors = AR.experiment["plot_colors"]


    V2uV = 1e6  # factor to multiply data to get microvolts.
    treat = "NT"  # for no treatment
    baseline_dbs = []

    all_thr_freq = {}
    subjs_done = []
    print("requested stimulus type: ", requested_stimulus_type)
    for isubj, subj in enumerate(subjs):  # get data for each subject
        # print("checking sub: ", subj["name"], subj["datatype"])
        if subj["name"] in AR.experiment["ABR_subject_excludes"]:
            continue

                # if subj["name"] not in ["CBA_F_N023_p175_NT", "CBA_F_N024_p175_NT", "CBA_F_N000_p18_NT", "CBA_F_N002_p27_NT", "CBA_F_N015_p573_NT"]:
        #     continue
        # if isubj > 5:
        #     continue

        if subj["name"] in ["CBA_M_N011_p99_NT"]:
            continue

        if subj["name"] in subjs_done:
            continue
        subjs_done.append(subj["name"])
        CP.cprint("m", f"\nSubject: {subj['name']!s}")

        if categorize == "treatment":
            treat = get_treatment(subj)
        elif categorize == "age":
            treat = get_age(subj)

        # f str(subj).endswith("Organization") or str(subj).startswith(".") or str(subj).startswith("Store") or treat.startswith("Store") or treat.startswith("."):
        #     continue
        # print(subj)
        subject_threshold, subject_ppio, dbs, threshold_index, waveana, stim_type = do_one_subject(
            subj=subj,
            treat=treat,
            requested_stimulus_type=requested_stimulus_type,
            AR=AR,
            ABR4=ABR4,
            V2uV=V2uV,
            example_subjects=example_subjects,
            example_axes=example_axes,
            test_plots= test_plots,
        )
        if treat not in all_thr_freq.keys():
            all_thr_freq[treat] = [subject_threshold]
        else:
            all_thr_freq[treat].append(subject_threshold)

    # each treatment group in the dict consists of a list (subjects) of lists ([thresholds], [frequencies])
    # first we get all the frequencies in the list of lists
    omit_freqs = [3000., 24000., 240000.]
    allfr = []
    for treat in all_thr_freq.keys():
        for i, subn in enumerate(all_thr_freq[treat]):
            allfr.extend(subn[1])
    allfr = sorted(list(set(allfr)))
    allfr = [f for f in allfr if f not in omit_freqs]
    # next, for each treament group, we create a list of thresholds for each frequency
    # if a subject does not have a threshold for a frequency, we insert a NaN
    thrmap = {}
    for treat in all_thr_freq.keys():
        if treat not in thrmap.keys():
            thrmap[treat] = []
        for i, subn in enumerate(all_thr_freq[treat]):
            # print("subn: ", subn)
            if len(subn[0]) == 0:
                thrmap[treat].append([np.nan for f in allfr if f not in omit_freqs])
            else:
                thrmap[treat].append([subn[0][subn[1].index(f)] if f in subn[1] else np.nan for f in allfr if f not in omit_freqs])
    all_tone_map_data = {"frequencies": allfr, "thresholds": thrmap, "plot_colors": plot_colors}
    with open("all_tone_map_data.pkl", "wb") as f:
        pickle.dump(all_tone_map_data, f)

    # plot_tone_map_data(thr_mapax, all_tone_map_data)

    return categories_done, all_tone_map_data

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def plot_tone_map_data(thr_mapax, all_tone_map_data):
    thrmap = all_tone_map_data["thresholds"]
    allfr = all_tone_map_data["frequencies"]
    plot_colors = all_tone_map_data["plot_colors"]
    print("thrmap: ", thrmap)
    print("allfr: ", allfr)
    f, thr_mapax = mpl.subplots(1,1)
    t_done = []
    fr_offset = 0.02
    # define frequencies that where ABRs were not consistently measured
    freq_to_remove = [1000., 2000., 3000., 6000., 12000., 40000.]
    # for each condition (treatment, age, etc)
    for itr, treat in enumerate(thrmap.keys()):
        print("treat: ", treat)
        if treat == "Young Adult":
            continue
        if treat in t_done:
            t_label = ""
        else:
            t_label = treat
            t_done.append(treat)

        dtr = np.array(thrmap[treat])
        frs = np.array(fr_offset*(itr-len(thrmap.keys())/2) + np.log10(allfr))
        # stack the arrays together. col 0 is raw freq, col2 is log freq, the rest are the subject threshols
        # measured at this frequency
        data = np.vstack((allfr, frs))
        for dx in dtr:
            data = np.vstack((data, dx))
        data = data.T
        nmeas = data.shape[0]-2
        rem_fr = []
        for ifr, fr in enumerate(allfr):
            if fr in freq_to_remove:
                rem_fr.append(ifr)
        datao = data.copy()
        data = np.delete(data, rem_fr, axis=0)
        data = np.where(data == 0., np.nan, data)
        data = np.where(data > 90., np.nan, data)
        topmarker = np.where(datao >=90., datao, np.nan)
        print(topmarker)
        print(len(np.where(topmarker[:, 2:] == 100.)[0]))
        # for d in data:
        thr_mapax.errorbar(data[:,1], np.nanmean(data[:, 2:], axis=1), yerr=scipy.stats.sem(data[:, 2:], axis=1, nan_policy='omit'), 
                        marker='o', markersize=4, color=plot_colors[treat], capsize=4, linestyle="-", linewidth=1.5)
        thr_mapax.plot(data[:,1], data[:, 2:], color=plot_colors[treat], marker="o", linewidth=0., markersize=2.5, label = t_label)
        thr_mapax.plot(datao[:,1], topmarker[:, 2:]+np.random.uniform(-2, 2, size=topmarker[:, 2:].shape),
                                                                       color=plot_colors[treat], marker="^", linewidth=0., markersize=3.5)
    
    thr_mapax.plot([np.log10(3000.), np.log10(64000)], [90, 90], linewidth=0.35, color="grey")
    thr_mapax.set_xlabel("Frequency (kHz)")
    thr_mapax.set_ylabel("Threshold (dB SPL)")
    x_vals = [ np.log10(4000), np.log10(8000), np.log10(16000), np.log10(32000), np.log10(48000)]
    x_labs = [ "4", "8", "16", "32", "48"]
    thr_mapax.set_xticks(x_vals)
    thr_mapax.set_xticklabels(x_labs)

    legend_without_duplicate_labels(thr_mapax)
    leg = thr_mapax.get_legend()
    leg.set_draggable(state= 1)
    thr_mapax.set_ylim([0, 110])
    thr_mapax.set_xlim([np.log10(3000.), np.log10(64000)])
    # thr_mapax.set_xscale("log")
    mpl.show()





def do_tone_map_analysis(
    AR: AnalyzeABR,
    ABR4: object,
    subject_data: dict = None,
    output_file: Union[Path, str] = None,
    subject_prefix: str = "CBA",
    categorize: str = "treatment",
    requested_stimulus_type: str = "Tone",
    experiment: Union[dict, None] = None,
    example_subjects: list = None,
):

    # base directory
    subjs = []
    # combine subjects from the directories
    # for directory_name in directory_names.keys():
    #     d_subjs = list(Path(directory_name).glob(f"{subject_prefix:s}*"))
    #     subjs.extend([ds for ds in d_subjs if ds.name.startswith(subject_prefix)])
    print("subject data keys: ", subject_data.keys())
    if requested_stimulus_type == "Tone":  # include interleaved_plateau as well
        subjs = subject_data[requested_stimulus_type]
        subjs.extend(subject_data["Interleaved_plateau"])
    # print("subjs with stim type", subjs)
    categories_done = []
    if output_file is not None:
        STYLE = ST.styler("JNeurophys", figuresize="full", height_factor=0.6)
        with PdfPages(output_file) as pdf:
            P = make_tone_figure(subjs, example_subjects, STYLE)

            if example_subjects is None:
                thr_mapax = P.axarr[1][0]

                example_axes = None
            else:
                example_axes = [P.axarr[i][0] for i in range(len(example_subjects))]
                nex = len(example_subjects)
                thr_mapax = P.axarr[1][0]
 
            categories_done, all_tone_map_data = compute_tone_thresholds(
                AR,
                ABR4,
                requested_stimulus_type=requested_stimulus_type,
                categorize=categorize,
                subjs=subjs,
                thr_mapax=thr_mapax,
                example_subjects=example_subjects,
                example_axes=example_axes,
                categories_done=categories_done,
            )
            # plot_tone_map_data(thr_mapax, all_tone_map_data)

def make_tone_figure(subjs, example_subjects, STYLE):
    row1_bottom = 0.1
    vspc = 0.08
    hspc = 0.06
    ncols = 1
    if example_subjects is not None:
        ncols += len(example_subjects)

    up_lets = ascii_letters.upper()
    ppio_labels = [up_lets[i] for i in range(ncols)]
    sizer = {
        "A": {"pos": [0.05, 0.175, 0.1, 0.90], "labelpos": (-0.15, 1.05)},
        "B": {"pos": [0.25, 0.6, 0.1, 0.90], "labelpos": (-0.15, 1.05)},
        # "C": {"pos": [0.49, 0.20, 0.12, 0.83], "labelpos": (-0.15, 1.05)},
        # "D": {"pos": [0.76, 0.22, 0.59, 0.36], "labelpos": (-0.15, 1.05)},
        # "E": {"pos": [0.76, 0.22, 0.12, 0.36], "labelpos": (-0.15, 1.05)},
    }

    P = PH.arbitrary_grid(
        sizer=sizer,
        order="rowsfirst",
        figsize=STYLE.Figure["figsize"],
        font="Arial",
        fontweight=STYLE.get_fontweights(),
        fontsize=STYLE.get_fontsizes(),
    )
    
    # PH.show_figure_grid(P, STYLE.Figure["figsize"][0], STYLE.Figure["figsize"][1])
    return P


def compute_click_io_analysis(
    AR,
    ABR4,
    categorize: str = "treatment",
    requested_stimulus_type: str = "Click",
    subjs: Union[str, Path, None] = None,
    axio=None,
    axthr=None,
    axppio=None,
    example_axes: list = None,
    example_subjects: list = None,
    categories_done: list = None,
    symdata: dict = None,
    test_plots:bool=False,
):
    overridecolor = False
    if categorize in ["treatment"]:
        categories = get_categories(subjs)
    elif categorize in ["age"]:
        categories = get_categories(subjs, categorize="age")
    else:
        raise ValueError(f"Category type {categorize} not recognized")
    # sort categories by the table in the configuration file
    ord_cat = []
    for cat in AR.experiment["plot_order"]["age_category"]:
        if cat in categories:
            ord_cat.append(cat)
    categories = ord_cat
    colors = ["r", "g", "b", "c", "m", "y", "k", "skyblue", "orange", "purple", "brown"]

    plot_colors = AR.experiment["plot_colors"]

    color_dict = {}
    # print(categories)
    waves_by_treatment = {}
    baseline = {}
    thresholds_by_treatment = {}  # {"treat1": [thr1, thr2, thr3], "treat2": [thr1, thr2, thr3]}
    amplitudes_by_treatment = {}  # {"treat1": [amp1, amp2, amp3], "treat2": [amp1, amp2, amp3]}
    dbs_by_treatment = {}

    treat = "NT"  # for no treatment
    baseline_dbs = []

    for subj in subjs:  # get data for each subject
        if subj["name"] in AR.experiment["ABR_subject_excludes"]:
            continue
        print("subj: ", subj["name"])
        CP.cprint("m", f"\nSubject: {subj!s}")


        if categorize == "treatment":
            treat = get_treatment(subj)
        elif categorize == "age":
            treat = get_age(subj)
        V2uV = 1e6  # factor to multiply data to get microvolts.

        # f str(subj).endswith("Organization") or str(subj).startswith(".") or str(subj).startswith("Store") or treat.startswith("Store") or treat.startswith("."):
        #     continue
        # print(subj)
        subject_threshold, subject_ppio, dbs, threshold_index, waveana, stim_type = do_one_subject(
            subj=subj,
            treat=treat,
            requested_stimulus_type=requested_stimulus_type,
            AR=AR,
            ABR4=ABR4,
            V2uV=V2uV,
            example_subjects=example_subjects,
            example_axes=example_axes,
            test_plots = test_plots,
        )

        if waveana is None or stim_type is None:
            continue
            
            
        CP.cprint("r", f"Subject thr: {subject_threshold!s}  stim_type: {stim_type!s}")
        # if subject_threshold is None or stim_type is None:
        #     continue
        print("Subject: ", subj, "\n   dbs: ", dbs, "\n   subject threshold", subject_threshold)
        dataset_name = (stim_type, treat)
        if treat not in thresholds_by_treatment:
            thresholds_by_treatment[treat] = []
        thresholds_by_treatment[treat].append(float(np.nanmean(subject_threshold)))
        if treat not in amplitudes_by_treatment:
            amplitudes_by_treatment[treat] = []
        amplitudes_by_treatment[treat].append(float(np.nanmean(subject_ppio)))
        if treat not in dbs_by_treatment:
            dbs_by_treatment[treat] = []
        dbs_by_treatment[treat].extend(dbs)
        # print("PPIO data: ", waveana.ppio)
        # print("PPIO shape: ", subject_ppio)
        if np.all(np.isnan(waveana.ppio)):
            print("all nan? : ", waveana.ppio)
            raise ValueError
        else:
            if treat not in waves_by_treatment:
                waves_by_treatment[treat] = [np.array([dbs, waveana.ppio])]
            else:
                waves_by_treatment[treat].append(np.array([dbs, waveana.ppio]))
        CP.cprint(
            "c",
            f"   dataset name: {dataset_name!s}  categories_done: {categories_done!s}, group: {treat:s}",
        )
        # plot the individual io functions.
        if axio is not None:
            # if subj["datatype"] == "pyabr3":
            #     symdata = ["d", 4]
            V2uV = 1e6
            if subj["name"][6] == "R":  # Ruili data set
                V2uV = 1e6  # different scale.
            
            if dataset_name not in categories_done:
                color = plot_colors[treat]
                colors.pop(0)
                color_dict[dataset_name] = color
                categories_done.append(dataset_name)
                if overridecolor:
                    color = "k"
                _plot_io_data(
                    waveana=waveana,
                    dbs=dbs,
                    # threshold_index=threshold_index,
                    V2uV=V2uV,
                    color=color,
                    ax=axio,
                    symdata=symdata,
                    dataset_name=dataset_name,
                    add_label=True,
                )
            else:
                color = plot_colors[treat]  # color_dict[dataset_name]
                if overridecolor:
                    color = "k"
                _plot_io_data(
                    waveana=waveana,
                    dbs=dbs,
                    # threshold_index=threshold_index,
                    V2uV=V2uV,
                    color=color,
                    ax=axio,
                    symdata=symdata,
                    dataset_name=dataset_name,
                    add_label=False,
                )

            if requested_stimulus_type not in baseline.keys():
                baseline[stim_type] = waveana.rms_baseline
                baseline_dbs.extend(dbs)
            else:
                if baseline[stim_type].ndim == 1:
                    baseline[stim_type] = baseline[stim_type][np.newaxis, :]
                n_base = baseline[stim_type].shape[1]
                n_wave = waveana.rms_baseline.shape[0]
                if n_wave == n_base:
                    baseline[stim_type] = np.vstack((baseline[stim_type], waveana.rms_baseline))
                    baseline_dbs.extend(dbs)
            baseline_dbs = sorted(list(set(baseline_dbs)))


    #  Now plot the baseline and the mean for each stimulus type

    plot_colors = AR.experiment["plot_colors"]
    plot_order = AR.experiment["plot_order"]["age_category"]
    if stim_type is None:
        print("stimulus type is None for ", subj)
        return categories_done
    else:
        print("stimulus type: for subj", subj["name"], stim_type)

    if axio is not None:
        # get baseline data and plot a grey bar
        bl_mean = np.mean(baseline[stim_type], axis=0) * V2uV
        bl_std = np.std(baseline[stim_type], axis=0) * V2uV

        axio.fill_between(baseline_dbs, bl_mean - bl_std, bl_mean + bl_std, color="grey", alpha=0.7)
        # For each category
        for k, treat in enumerate(categories):
            print("getting mean for category: ", treat)
            dataset_name = (stim_type, treat)
            # get all the SPLs for this dataset.
            all_dbs = []
            if treat not in waves_by_treatment.keys():
                continue
            for i in range(len(waves_by_treatment[treat])):
                all_dbs.extend(waves_by_treatment[treat][i][0])
            # and just the unique ones, in order
            all_dbs = [float(d) for d in sorted(list(set(all_dbs)))]
            # build a np array for all the response measures
            wvs = np.nan*np.ones((len(waves_by_treatment[treat]), len(all_dbs)))
            for i in range(len(waves_by_treatment[treat])):
                for j, db in enumerate(waves_by_treatment[treat][i][0]):
                    k = all_dbs.index(db)
                    wvs[i, k] = waves_by_treatment[treat][i][1][j]
            waves_by_treatment[treat] = wvs
            all_dbs = np.array(all_dbs)
            
            # limit the plot to a common range  (this could be smarter - we could count nonnan observations
            # at each level and only plot where we have sufficient data, e.g., N = 3 or larger?
            valid_dbs = np.argwhere((all_dbs >= 20.0) & (all_dbs <= 90.0))
            valid_dbs = [int(v) for v in valid_dbs]
            if waves_by_treatment[treat].ndim > 1:
                d_mean = np.nanmean(waves_by_treatment[treat], axis=0) * V2uV
                d_std = np.nanstd(waves_by_treatment[treat] * V2uV, axis=0)
            else:
                d_mean = waves_by_treatment[treat] * V2uV
                d_std = np.zeros_like(d_mean)
            color = plot_colors[treat]
            n_mean = len(all_dbs)
            axio.plot(all_dbs[valid_dbs], d_mean[valid_dbs], color=color, linestyle="-", linewidth=1.5, alpha=0.75)
            if np.max(d_std) > 0:
                axio.fill_between(
                    all_dbs[valid_dbs],
                    d_mean[valid_dbs] - d_std[valid_dbs],
                    d_mean[valid_dbs] + d_std[valid_dbs],
                    color=color,
                    alpha=0.1,
                    edgecolor=color,
                    linewidth=1,
                )
        PH.do_talbotTicks(axio, axes="x", density=[1, 2], insideMargin=0.05)
        PH.do_talbotTicks(axio, axes="y", density=[0.5, 1], insideMargin=0.05)
        PH.nice_plot(axio, direction="outward", ticklength=3)

    if axthr is not None:
        df = pd.DataFrame.from_dict(thresholds_by_treatment, orient="index")
        df = df.transpose()
        sns.barplot(
            df,
            ax=axthr,
            palette=plot_colors,
            order=plot_order,
            alpha=0.7,
            linewidth=0.75,
            edgecolor="grey",
            capsize=0.05,
        )
        sns.swarmplot(
            data=df, ax=axthr, palette=plot_colors, order=plot_order, size=3.0, linewidth=0.5
        )
        axthr.set_ylabel("Threshold (dB SPL)")
        axthr.set_xlabel("Age Category")
        remap_xlabels(axthr)
        axthr.set_ylim(0, 100)
        PH.nice_plot(axthr, direction="outward", ticklength=3)
        PH.do_talbotTicks(axthr, axes="x", density=[1, 2], insideMargin=0.05)
        PH.do_talbotTicks(axthr, axes="y", density=[0.5, 1], insideMargin=0.05)

    if axppio is not None:
        df = pd.DataFrame.from_dict(amplitudes_by_treatment, orient="index")
        df = df.transpose()
        # print("Amplitudes: ", df)

        sns.barplot(
            df,
            ax=axppio,
            palette=plot_colors,
            order=plot_order,
            alpha=0.7,
            linewidth=0.75,
            edgecolor="grey",
            capsize=0.05,
        )
        sns.swarmplot(
            data=df, ax=axppio, palette=plot_colors, order=plot_order, size=3.0, linewidth=0.5
        )

        axppio.set_ylabel("Amplitude (uV)")
        axppio.set_xlabel("Age Category")
        remap_xlabels(axppio)
        axppio.set_ylim(0, 12)
        PH.nice_plot(axppio, direction="outward", ticklength=3)
        PH.do_talbotTicks(axppio, axes="x", density=[1, 2], insideMargin=0.05)
        PH.do_talbotTicks(axppio, axes="y", density=[0.5, 1], insideMargin=0.05)
    return categories_done
    # break
    # print("ax click2, 2, example: ", ax_click1, ax_click2, example_subjects)
    # if ax_click1 is not None and example_subjects is not None:
    #     plot_click_stack(filename=filename, directory_name=directory_name, ax=ax_click1)

    # if ax_click2 is not None and example_subjects is not None:
    #     plot_click_stack(filename=filename, directory_name=directory_name, ax=ax_click2)


def make_click_figure(subjs, example_subjects, STYLE):
    row1_bottom = 0.1
    vspc = 0.08
    hspc = 0.06
    ncols = 3
    if example_subjects is not None:
        ncols += len(example_subjects)

    up_lets = ascii_letters.upper()
    ppio_labels = [up_lets[i] for i in range(ncols)]
    sizer = {
        "A": {"pos": [0.05, 0.175, 0.1, 0.90], "labelpos": (-0.15, 1.05)},
        "B": {"pos": [0.25, 0.175, 0.1, 0.90], "labelpos": (-0.15, 1.05)},
        "C": {"pos": [0.49, 0.20, 0.12, 0.83], "labelpos": (-0.15, 1.05)},
        "D": {"pos": [0.76, 0.22, 0.59, 0.36], "labelpos": (-0.15, 1.05)},
        "E": {"pos": [0.76, 0.22, 0.12, 0.36], "labelpos": (-0.15, 1.05)},
    }

    P = PH.arbitrary_grid(
        sizer=sizer,
        order="rowsfirst",
        figsize=STYLE.Figure["figsize"],
        font="Arial",
        fontweight=STYLE.get_fontweights(),
        fontsize=STYLE.get_fontsizes(),
    )
    # P = PH.Plotter(
    #     (5,1),
    #     axmap = axmap,
    #     order="rowsfirst",
    #     figsize=STYLE.Figure["figsize"],
    #     # horizontalspacing=hspc,
    #     # verticalspacing=vspc,
    #     margins={
    #         "leftmargin": 0.07,
    #         "rightmargin": 0.07,
    #         "topmargin": 0.05,
    #         "bottommargin": row1_bottom,
    #     },
    #     labelposition=(-0.15, 1.05),
    #     # panel_labels=ppio_labels,
    #     font="Arial",
    #     fontweight=STYLE.get_fontweights(),
    #     fontsize=STYLE.get_fontsizes(),
    # )

    # PH.show_figure_grid(P, STYLE.Figure["figsize"][0], STYLE.Figure["figsize"][1])
    return P


def do_click_io_analysis(
    AR: AnalyzeABR,
    ABR4: object,
    subject_data: dict = None,
    output_file: Union[Path, str] = None,
    subject_prefix: str = "CBA",
    categorize: str = "treatment",
    requested_stimulus_type: str = "Click",
    experiment: Union[dict, None] = None,
    example_subjects: list = None,
    test_plots: bool=False,
):

    # base directory
    subjs = []
    # combine subjects from the directories
    # for directory_name in directory_names.keys():
    #     d_subjs = list(Path(directory_name).glob(f"{subject_prefix:s}*"))
    #     subjs.extend([ds for ds in d_subjs if ds.name.startswith(subject_prefix)])
    subjs = subject_data[requested_stimulus_type]
    categories_done = []
    if output_file is not None:
        STYLE = ST.styler("JNeurophys", figuresize="full", height_factor=0.6)
        with PdfPages(output_file) as pdf:
            P = make_click_figure(subjs, example_subjects, STYLE)

            if example_subjects is None:
                ioax = P.axarr[0][0]
                thrax = P.axarr[1][0]
                ppioax = P.axarr[2][0]
                example_axes = None
            else:
                example_axes = [P.axarr[i][0] for i in range(len(example_subjects))]
                nex = len(example_subjects)
                ioax = P.axarr[nex][0]
                thrax = P.axarr[nex + 1][0]
                ppioax = P.axarr[nex + 2][0]
 
            print("\n\nstimulus_type: ", requested_stimulus_type)

            categories_done = compute_click_io_analysis(
                AR,
                ABR4,
                requested_stimulus_type=requested_stimulus_type,
                categorize=categorize,
                subjs=subjs,
                axio=ioax,
                axthr=thrax,
                axppio=ppioax,
                example_subjects=example_subjects,
                example_axes=example_axes,
                categories_done=categories_done,
                test_plots = test_plots,
            )

            ioax.set_xlabel("dB SPL")
            ioax.set_clip_on(False)
            ioax.set_ylabel(f"{requested_stimulus_type:s} P1-N1 Peak to Peak Amplitude (uV)")
            leg = ioax.legend(
                prop={"size": 4, "weight": "normal", "family": "Arial"},
                fancybox=False,
                shadow=False,
                facecolor="none",
                loc="upper left",
                draggable=True,
            )
            leg.set_zorder(1000)
            ioax.set_title(
                f"Peak to Peak Amplitude for all subjects\n{subject_prefix:s}",
                va="top",
            )
            ioax.text(
                0.96,
                0.01,
                s=datetime.datetime.now(),
                fontsize=6,
                ha="right",
                transform=P.figure_handle.transFigure,
            )
            mpl.show()
            if output_file is not None:
                pdf.savefig(P.figure_handle)
                print("Saved to: ", output_file)
                P.figure_handle.clear()
    else:
        for directory_name in directory_names:
            categories_done = compute_io_analysis(
                AR,
                ABR4,
                stimulus_type=stimulus_type,
                categorize=categorize,
                subjs=subjs,
                directory_name=directory_name,
                axio=ioax,
                axthr=thrax,
                axppio=ppioax,
                example_subjects=example_subjects,
                example_axes=example_axes,
                symdata=directory_names[directory_name],
                categories_done=categories_done,
            )
    return categories_done


def test_age_re():
    import re

    re_age = re.compile(r"[_ ]{1}[p]{0,1}[\d]{1,3}[d]{0,1}[_ ]{1}", re.IGNORECASE)
    examples = [
        "CBA_M_N017_p572_NT",
        "CBA_M_N004_p29_NT",
    ]
    for s in examples:
        m = re_age.search(s)
        if m is not None:
            print(m.group())
            age = m.group().strip("_ ")
            parsed = PA.age_as_int(PA.ISO8601_age(age))
            print(parsed)
            print()
        else:
            print("No match")


def find_files_for_subject(dataset: Union[str, Path], subj_data: dict = None, stim_types:list = None):
    """find_files_for_subject : Find the files for a given subject
    Return the list of files for the requested stimulus type, and the
    directory that holds those files.
    If there is not match, we return None
    """

    filenames = list(
        Path(dataset).glob("*")
    )  # find all of the files and directories in this subject directory
    filenames = [fn for fn in filenames if not fn.name.startswith(".")]  # clean out .DS_Store, etc.
    # print("Top level fns: ", filenames)
    # these are possible directories in the subject directory
    # the data may be at the top level (common for older ABR4 datasets)
    # or they may be in these subdirectories. We will find out by looking at the
    # files in the subject directory.

    # dirs = ["click", "tone", "interleaved_plateau"]
    # if requested_stimulus_type not in dirs:
    #     raise ValueError(f"Requested stimulus type {requested_stimulus_type} not recognized")
    if subj_data is None:
        subj_data = {"Click": [], "Tone": [], "Interleaved_plateau": []}
    for filename in filenames:
        # print("filename: ", filename)
        if filename.name in [
            "Click",
            "Tone",
            "interleaved_plateau",
            "Interleaved_plateau",
            "Interleaved_plateau_High",
        ]:
            short_name = Path(filename.parent).name
        else:
            short_name = filename.name

        m_subject = re_subject.match(short_name)
        if m_subject is not None:
            subject = m_subject["subject"]
        else:
            subject = None
        # abr4 clicks in the main directory
        m_click = re_click_file.match(filename.name)
        if m_click is not None and filename.is_file():
            dset = {
                "datetime": m_click["datetime"],
                "subject": subject,
                "name": filename.parent.name,
                "filename": filename,
                "datatype": "ABR4",
            }
            if dset not in subj_data["Click"]:
                subj_data["Click"].append(dset)

        # abr4 tones in the main directory
        m_tone = re_tone_file.match(filename.name)
        if m_tone is not None and filename.is_file():
            dset = {
                "datetime": m_tone["datetime"],
                "subject": subject,
                "name": filename.parent.name,
                "filename": filename,
                "datatype": "ABR4",
            }
            if dset not in subj_data["Tone"]:
                subj_data["Tone"].append(dset)

        # subdirectory with Click name - either ABR4 or pyabr3
        if filename.name == "Click" and filename.is_dir():
            # first check for ABR4 files
            click_files = filename.glob("*.txt")
            for cf in click_files:
                m_click = re_click_file.match(cf.name)
                if m_click is None:
                    continue
                dset = {
                    "datetime": m_click["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "ABR4",
                }
                if dset not in subj_data["Click"]:
                    subj_data["Click"].append(dset)

            # next check for pyabr3 files
            click_files = filename.glob("*.p")
            for i, cf in enumerate(click_files):
                m_click = re_pyabr3_click_file.match(cf.name)
                if m_click is None:
                    continue
                dset = {
                    "datetime": m_click["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "pyabr3",
                }
                if dset not in subj_data["Click"]:
                    subj_data["Click"].append(dset)

        # Tones in a Tone subdirectory
        if filename.name == "Tone" and filename.is_dir():
            tone_files = filename.glob("*.txt")
            for tone_file in tone_files:
                m_tone = re_tone_file.match(tone_file.name)
                if m_tone is None:
                    continue
                dset = {
                    "datetime": m_tone["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "ABR4",
                }
                if dset not in subj_data["Tone"]:
                    subj_data["Tone"].append(dset)

        # next check for pyabr3 files
        if filename.name.startswith("Interleaved_plateau") and filename.is_dir():
            tone_files = filename.glob("*.p")
            # print("tone files: ", filename, list(tone_files))
            for i, tone_file in enumerate(tone_files):
                m_tone = re_pyabr3_interleaved_tone_file.match(tone_file.name)
                # print(tone_file.name, m_tone["datetime"], m_tone["serial"])
                if m_tone is None:
                    continue
                dset = {
                    "datetime": m_tone["datetime"],
                    "subject": subject,
                    "name": filename.parent.name,
                    "filename": filename,
                    "datatype": "pyabr3",
                }
                if dset not in subj_data["Interleaved_plateau"]:
                    subj_data["Interleaved_plateau"].append(dset)

    return subj_data

def get_datasets(directory_names):
    subdata = None
    for directory in list(directory_names.keys()):
        subs = [
            sdir
            for sdir in Path(directory).glob("*")
            if sdir.is_dir()
            and not sdir.name.startswith(".")
            and not sdir.name.startswith("Old Organization")
            and not sdir.name.startswith("NG_")
        ]
        # print("Subs: ", subs)

        filter = "CBA"
        for sub in subs:
            if not sub.name.startswith(filter):
                continue
            subdata = find_files_for_subject(
                dataset=sub,
                subj_data=subdata,
                stim_types = ["Click", "Interleaved_plateau", "Tone"]
            )
    # for s in subdata.keys():
    #     for d in subdata[s]:
    #         print(s, d)
    return subdata

if __name__ == "__main__":

    AR = AnalyzeABR()
    ABR4 = read_abr4.READ_ABR4()

    # do_io_analysis(
    #     directory_name="/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_NIHL",
    #     output_file="NIHL_VGAT-EYFP_IO_ABRs_combined.pdf",
    #     subject_prefix="VGAT-EYFP",
    #     categorize="treatment",
    #     # hide_treatment=True,
    # )

    # do_io_analysis(
    #     directory_name="/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_NIHL",
    #     subject_prefix="Glyt2-GFP",
    #     output_file="NIHL_Glyt2-GFP_ABRs_IOFunctions_combined.pdf",
    #     # output_file="NIHL_GlyT2_ABRs_IOFunctions_combined.pdf",
    #     categorize="treatment",
    #       )

    config_file_name = "/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg"

    AR.get_experiment(config_file_name, "CBA_Age")
    directory_names = {  # values are symbol, symbol size, and relative gain factor
        "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_CBA_Age": ["o", 3.0, 1.],
        # "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Tessa_CBA": ["s", 3.0, 1.],
        "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Ruilis ABRs": ["x", 3.0, 10.],
    }
 
    subdata = get_datasets(directory_names)
    # select subjects for tuning analysis parameters in the configuration file.
    # subdata = get_datasets(directory_names)
    tests  = False
    if tests:
        test_subjs = ["N004", "N005", "N006", "N007"]
        newsub = {"Click": []}
        # print(subdata.keys())
        for sub in subdata:
            # print(sub)
            for d in subdata[sub]:
                # print(d)
                if d["subject"] in test_subjs:
                    newsub["Click"].append(d)
        
        subdata = newsub
        # for ns in subdata["Click"]:
        # print(ns) 
    # exit()
    test_plots = False

    do_click_io_analysis(
        AR=AR,
        ABR4=ABR4,
        subject_data=subdata,
        subject_prefix="CBA_",
        output_file="CBA_Age_ABRs_Tones_combined.pdf",
        categorize="age",
        requested_stimulus_type="Click",
        example_subjects=["CBA_F_N002_p27_NT", "CBA_M_N017_p572_NT"],
        test_plots = test_plots
    )
    # do_tone_map_analysis(
    #     AR=AR,
    #     ABR4=ABR4,
    #     subject_data=subdata,
    #     subject_prefix="CBA_",
    #     output_file="CBA_Age_ABRs_Tones_combined.pdf",
    #     categorize="age",
    #     requested_stimulus_type="Tone",
    #     example_subjects=["CBA_F_N002_p27_NT", "CBA_M_N017_p572_NT"],
    # )

    # with open("all_tone_map_data.pkl", "rb") as f:
    #     all_tone_map_data = pickle.load(f)
    # f, ax = mpl.subplots(1, 1)
    # plot_tone_map_data(ax, all_tone_map_data=all_tone_map_data)

    mpl.show()
