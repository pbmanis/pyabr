import pickle
import matplotlib.pyplot as mpl
import platform
import pathlib
from pathlib import Path
from typing import Union
import numpy as np
import pyqtgraph as pg
import plothelpers as mpl_PH
import src.read_calibration as read_calibration
import src.filter_util as filter_util
import src.parse_ages as PA

use_matplotlib = True
from pylibrary.plotting import plothelpers as PH

# Check the operating system and set the appropriate path type
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


class AnalyzeABR:
    def __init__(self):
        self.caldata = None
        self.gain = 1e4
        self.fsamp = 24414.0625
        self.FILT = filter_util.Utility()

        # 24414.0625

    def read_abr_file(self, fn):
        with open(fn, "rb") as fh:
            d = pickle.load(fh)
            self.caldata = d["calibration"]
            self.fsamp = d["record_frequency"]
            self.fsamp = self.check_fsamp(d)
        return d

    def show_calibration(self, fn):
        d = self.read_abr_file(fn)
        dbc = self.convert_attn_to_db(20.0, 32200)
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

    def average_within_traces(self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None):
        # i is the index into the acquisition series. There is one file for each repetition of each condition.
        # the series might span a range of frequencies and intensities; these are
        # in the protocol:stimulus dictionary (dblist, freqlist)
        # we average response across traces for each intensity and frequency
        # this function specifically works when each trace has one stimulus condition (db, freq), and is
        # repeated nreps times.
        # The returned data is the average of the responses across the nreps for this (the "ith") stimulus condition
        stim_type = str(Path(fd).stem)
        print("Stim type: ", stim_type)
        if stim_type.startswith("tones"):
            name = "tonepip"
        elif stim_type.startswith("click"):
            name = "click"
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]

        nreps = protocol["stimuli"]["nreps"]
        delay = protocol["stimuli"]["delay"]
        dur = 0.010
        for n in range(nreps):  # loop over the repetitions for this specific stimulus
            fn = f"{date}_{name}_{i:03d}_{n+1:03d}.p"
            d = self.read_abr_file(Path(fd, fn))
            if n == 0:
                data = d["data"]
            else:
                data += d["data"]
            if n == 0:
                tb = np.linspace(0, len(data) / self.fsamp, len(data))
        data = data / nreps
        if high_pass_filter is not None:
            print("average within traces hpf: ", high_pass_filter, self.fsamp)
            data = self.FILT.SignalFilter_HPFButter(data, high_pass_filter, self.fsamp, NPole=4, bidir=True)
        # tile the traces.
        # first interpolate to 100 kHz
        # If you don't do this, the blocks will precess in time against
        # the stimulus, which is timed on a 500 kHz clock.
        # It is an issue because the TDT uses an odd frequency clock...

        trdur = len(data) / self.fsamp
        newrate = 1e5
        tb100 = np.arange(0, trdur, 1.0 / newrate)

        one_response = int(0.025 * newrate)
        arraylen = one_response * protocol["stimuli"]["nstim"]

        abr = np.interp(tb100, tb, data)
        sub_array = np.split(abr[:arraylen], protocol["stimuli"]["nstim"])
        sub_array = np.mean(sub_array, axis=0)
        tb = tb[:one_response]
        return sub_array, tb

    def average_across_traces(self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None):
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
        delay = protocol["stimuli"]["delay"]
        dur = 0.010
        dataset = []
        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]

        stim_type = str(Path(fd).stem)
        for n in range(nreps):
            fn = f"{date}_{stim_type}_{i:03d}_{n+1:03d}.p"
            d = self.read_abr_file(Path(fd, fn))

            if n == 0:
                data = d["data"]
            else:
                data += d["data"]
            if n == 0:
                print("original sample rate: ", self.fsamp)
                print("number of data points: ", len(data))
                tb = np.arange(0, len(data) / self.fsamp, 1.0 / self.fsamp)

        data = data / nreps
        #  high pass filter at 300 Hz
        # data = self.FILT.SignalFilter_LPFButter(data, LPF=3000, samplefreq=self.fsamp, NPole=4)
        if high_pass_filter is not None:
            print("average across traces hpf: ", high_pass_filter, self.fsamp)
            data = self.FILT.SignalFilter_HPFButter(
                data, HPF=high_pass_filter, samplefreq=self.fsamp, NPole=4, bidir=True
            )
        # tile the traces.
        # first linspace to 100 kHz
        trdur = len(data) / self.fsamp
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
        it0 = int(protocol["stimuli"]["delay"] * newrate)

        abr = abr[it0:]
        one_response_100 = int(protocol["stimuli"]["stimulus_period"] * newrate)
        print(protocol["stimuli"])
        if isinstance(protocol["stimuli"]["freqlist"], str):
            frlist = eval(protocol["stimuli"]["freqlist"])
        if isinstance(protocol["stimuli"]["dblist"], str):
            dblist = eval(protocol["stimuli"]["dblist"])
        arraylen = one_response_100 * protocol["stimuli"]["nstim"]
        if stim_type in ["click", "tonepip"]:
            nsplit = protocol["stimuli"]["nstim"]
        elif stim_type in ["interleaved_plateau"]:
            nsplit = int(len(frlist) * len(dblist))
        arraylen = one_response_100 * nsplit
        print(len(frlist), len(dblist))
        print("one response: ", one_response_100)
        print("nsplit: ", nsplit)
        print("arranlen/nsplit: ", float(arraylen) / nsplit)
        print("len data: ", len(data), len(abr), nsplit * one_response_100)
        sub_array = np.split(abr[:arraylen], nsplit)
        # print(len(sub_array))
        abr = np.array(sub_array)
        tb = tb100[:one_response_100]

        print("abr shape: ", abr.shape, "max time: ", np.max(tb))
        stim = np.meshgrid(frlist, dblist)
        self.fsamp = newrate
        # print(stim)
        return abr, tb, stim

    def show_stimuli(self, fn):
        d = self.read_abr_file(fn)
        stims = list(d["stimuli"].keys())
        wave = d["stimuli"][stims[0]]["sound"]
        pg.plot(wave)
        pg.exec()

    def plot_abrs(
        self,
        abrd: np.ndarray,
        tb: np.ndarray,
        scale: str = "V",
        highpass: Union[float, None] = None,
        stim_type: str = "click",
        dblist: Union[list, None] = None,
        frlist: Union[list, None] = None,
        maxdur: float = 14.0,
        metadata: dict = {},
        use_matplotlib: bool = True,
    ):

        amplifier_gain = metadata["amplifier_gain"]
        if scale == "uV":
            # amplifier gain has already been accountd for.
            amplifier_gain = 1.0

        if len(frlist) == 0 or stim_type == "click":
            frlist = [0]
            ncols = 1
            width = (1.0 / 3.0) * len(dblist) * ncols
            height = 1.0 * len(dblist)
            lmar = 0.12
        else:
            ncols = len(frlist)
            width = 2.0 * ncols
            height = 1.0 * len(dblist)
            lmar = 0.08

        if height > 10.0:
            height = 10.0 * (10.0 / height)
        if use_matplotlib:
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
                    "topmargin": 0.1,
                    "bottommargin": 0.05,
                },
            )
            fn = metadata["filename"]

            if metadata["type"] == "ABR4":
                filename = str(Path(fn).name)
                if filename in ["Click", "Clicks", "Tone", "Tones"]:
                    filename = str(Path(fn).parent.name)
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
            if highpass is None:
                hpftext = "Off"
            else:
                hpftext = f"{highpass:5.1f} Hz"
            title = f"File: {filename:s}, Stimulus: {stim_type}, Amplifier Gain: {amplifier_gain}, Fs: {rec_freq}, HPF: {hpftext:s}\n"
            title += f"Subject: {subject_id:s}, Age: {age:s} Sex: {sex:s}, Strain: {strain:s}, Weight: {weight:.2f}, Genotype: {genotype:s}"
            P.figure_handle.suptitle(title, fontsize=8)
            ax = P.axarr
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
            click_colors = ["xkcd:azure", "xkcd:lightblue", "xkcd:purple", "xkcd:orange", "xkcd:red", "xkcd:green",
                            "xkcd:golden yellow"]
            n_click_colors = len(click_colors)
            refline_ax = []
            for j, fr in enumerate(range(ncols)):  # enumerate(abrd.keys()):
                for i, db in enumerate(dblist):

                    ax = P.axarr[len(dblist) - i - 1, j]

                    PH.nice_plot(ax, direction="outward", ticklength=3)
                    if i != 0:
                        PH.noaxes(ax)
                    npts = len(abrd[i, j])
                    n_disp_pts = int(maxdur*1e-3 * rec_freq)  # maxdur is in msec.
                    if n_disp_pts < npts:
                        npts = n_disp_pts
                    plot_data = 1e6 * abrd[i, j] / amplifier_gain
                    if stim_type in ["click", "tonepip"]:
                        ax.plot(tb[:npts] * 1e3, plot_data[:npts], color=click_colors[i % n_click_colors],
                                linewidth=1, clip_on=False)
                    else:
                        ax.plot(tb[:npts] * 1e3, plot_data[:npts], color=colors[j], clip_on=False)
                    if ax not in refline_ax:
                        PH.referenceline(ax, linewidth=0.5)
                        refline_ax.append(ax)
                    # print(dir(ax))
                    ax.set_facecolor("#ffffff00")  # background will be transparent, allowing traces to extend into other axes
                    ax.set_xlim(0, maxdur)
                    if i == len(dblist) - 1:
                        if ncols > 1:
                            ax.set_title(f"{frlist[j]} Hz")
                        else:
                            ax.set_title("Click")
                    if j == 0:
                        ax.set_ylabel(f"{dblist[i]} dB")
                    ax.set_xlim(0, np.max(tb[:npts]) * 1e3)
                    if i == 0:
                        ticks = np.arange(0, maxdur, 2)
                        ax.set_xticks(
                            ticks, [f"{int(k):d}" for k in ticks]
                        )
                    else:
                        ax.set_xticks(
                            ticks, [" "]*len(ticks)
                        )
                        

                if i == 0:
                    ax.set_xlabel("Time (ms)")
                # ax.set_xticks([1, 3, 5, 7, 9], minor=True)


                n += 1
                if np.max(plot_data[:npts]) > v_max:
                    v_max = np.max(plot_data[:npts])
                if np.min(plot_data[:npts]) < v_min:
                    v_min = np.min(plot_data[:npts])
            amax = np.max([-v_min, v_max])/2.0
            # print(P.axarr.shape, len(dblist), len(frlist))

            for i in range(len(dblist)):
                for j in range(len(frlist)):
                    P.axarr[i, j].set_ylim(-amax, amax)
                    # PH.referenceline(ax, linewidth=0.5)

            # mpl.tight_layout()
            mpl.show()
        else:  # use pyqtgraph
            app = pg.mkQApp("ABR Data Plot")
            win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            win.resize(1200, 1000)
            win.setWindowTitle(f"File: {str(Path(fn).parent)}")
            win.setBackground("w")

            lw = pg.LayoutWidget(parent=win)
            lw.addLabel(text="Hi there", row=0, col=0, rowspan=1, colspan=len(frlist))
            lw.nextRow()

            plid = []
            if len(frlist) == 0:
                frlist = [1]
            col = 0

            print("stim_type (in pg plotting)", stim_type)
            if stim_type not in ["clicks", "click"]:
                ref_set = False
                v_min = 0
                v_max = 0
                for i, db in enumerate(dblist):
                    row = i  # int(i/5)
                    for j, fr in enumerate(frlist):
                        col = j
                        # if tb is None:
                        #     npts = len(abrd[i, j])
                        #     tb = np.linspace(0, npts / rec_freq, npts)

                        pl = win.addPlot(
                            title=f"{dblist[-i-1]} dB, {fr} Hz", col=col, row=row
                        )  # i % 5)
                        if not ref_set:
                            ref_ax = pl
                            ref_set = True
                        plot_data = 1e6 * abrd[len(dblist) - i - 1, j] / amplifier_gain
                        lpd = len(plot_data)
                        if stim_type in ["click", "tonepip"]:
                            pl.plot(
                                tb[:lpd] * 1e3,
                                plot_data,
                                pen=pg.mkPen(j, len(dblist), width=2),
                                clipToView=True,
                            )
                        else:
                            pl.plot(
                                tb[:lpd] * 1e3,
                                plot_data,
                                pen=pg.mkPen(j, len(dblist), width=2),
                                clipToView=True,
                            )
                        pl.plot(
                            tb[:lpd] * 1e3,
                            np.zeros_like(plot_data),
                            pen=pg.mkPen(
                                "grey", linetype=pg.QtCore.Qt.PenStyle.DashLine, width=0.33
                            ),
                            clipToView=True,
                        )
                        # pl.showGrid(x=True, y=True)
                        if j == 0:
                            pl.setLabel("left", "Amp", units="uV")
                        if i == len(dblist) - 1:
                            pl.setLabel("bottom", "Time", units="ms")
                        pl.setYRange(-3.0, 3.0)
                        pl.setXRange(0, 10)
                        if ref_set:
                            pl.setXLink(ref_ax)
                            pl.setYLink(ref_ax)
                        if np.max(plot_data) > v_max:
                            v_max = np.max(plot_data)
                        if np.min(plot_data) < v_min:
                            v_min = np.min(plot_data)
                ref_ax.setYRange(v_min, v_max)

            else:
                v0 = 0
                v = []
                for i, db in enumerate(dblist):
                    if i == 0:
                        pl = win.addPlot(title=f"{db} dB, {fr} Hz")  # i % 5)
                    pl.plot(
                        tb * 1e3,
                        -v0 + abrd[i, j] / amplifier_gain,
                        pen=pg.mkPen(pg.intColor(i, hues=len(dblist)), width=2),
                        clipToView=True,
                    )
                    v0 += 1e-6 * amplifier_gain
                    v.append(v0)
                    # pl.showGrid(x=True, y=True)
                    pl.setLabel("left", "Amplitude", units="mV")
                    pl.setLabel("bottom", "Time", units="s")
                    label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
                    label.setParentItem(pl)
                    label.anchor(itemPos=(0.05, -v0 * 180), parentPos=(0.1, 0))
                    # pl.setYRange(-2e-6, 2e-6)
                    plid.append(pl)
                for i, db in enumerate(dblist):
                    label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
                    label.setParentItem(pl)
                    label.anchor(itemPos=(0.05, -v[i] * 200), parentPos=(0.1, 0))
                    # win.nextRow()
                    for j, fr in enumerate(frlist):
                        ax.set_title(f"{self.convert_attn_to_db(db, fr)} dBSPL, {fr} Hz")
                        if i == 0:
                            ax.set_xlabel("Time (s)")
                        if j == 0:
                            ax.set_ylabel("Amplitude")
                        ax.set_ylim(-50, 50)
                        PH.noaxes(ax)
                        if i == 0 and j == 0:
                            PH.calbar(
                                ax,
                                calbar=[0, -20, 2, 10],
                                scale=[1.0, 1.0],
                                xyoffset=[0.05, 0.1],
                            )
                        PH.referenceline(ax, linewidth=0.5)
                        n += 1

            # else:  # use pyqtgraph
            #     app = pg.mkQApp("ABR Data Plot")
            #     win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            #     win.resize(1000, 1000)
            #     win.setWindowTitle(f"File: {str(Path(fn).parent)}")
            #     plid = []
            #     if len(frlist) == 0:
            #         frlist = [1]
            #     for i, db in enumerate(dblist):
            #         for j, fr in enumerate(frlist):
            #             pl = win.addPlot(
            #                 title=f"{self.convert_attn_to_db(db, fr):5.1f} dB, {fr:8.1f} Hz"
            #             )
            #             if stim_type in ["click", "tonepip"]:
            #                 pl.plot(
            #                     tb * 1e3,
            #                     abrd[i, j] / self.gain,
            #                     pen=pg.mkPen(j, len(dblist), width=2),
            #                     clipToView=True,
            #                 )
            #             else:
            #                 pl.plot(
            #                     tb * 1e3,
            #                     abrd[len(dblist) - i - 1, j] / self.gain,
            #                     pen=pg.mkPen(j, len(dblist), width=2),
            #                     clipToView=True,
            #                 )
            #             # pl.showGrid(x=True, y=True)
            #             if j == 0:
            #                 pl.setLabel("left", "Amplitude", units="mV")
            #             if i == 0:
            #                 pl.setLabel("bottom", "Time", units="s")
            #             pl.setYRange(-5e-6, 5e-6)
            #             plid.append(pl)
            #         win.nextRow()

            pg.exec()

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)


    def check_fsamp(self, d):
        if d["record_frequency"] is None:
            d["record_frequency"] = 24414.0625   # 97656.25
        return d['record_frequency']


    def read_and_average_abr_files(self, fn, amplifier_gain=1e4, high_pass_filter: Union[float, None] = None, maxdur:Union[float, None]=None):
        d = self.read_abr_file(fn)
        print("read and average abrs")
        # print(d["protocol"])
        # print("d keys: ", d.keys())
        self.fsamp = self.check_fsamp(d)
        if maxdur is None:
            maxdur = 25.0
        stim_type = d["protocol"]["protocol"]["stimulustype"]
        fd = Path(fn).parent
        fns = Path(fd).glob("*.p")
        # break the filenames into parts.
        # The first part is the date, the second part is the stimulus type,
        # the third part is the index into the stimulus array
        # the fourth part is the repetition number for a given stimulus
        protocol = d["protocol"]
        rec = protocol["recording"]
        print("rec: ", rec)
        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        if isinstance(dblist, str):
            dblist = eval(dblist)
        if isinstance(frlist, str):
            frlist = eval(frlist)
        # ndb = len(dblist)
        # nreps = protocol["stimuli"]["nreps"]
        # delay = protocol["stimuli"]["delay"]
        # dur = 0.010
        print(d["subject_data"])
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

        file_parts = Path(fn).stem.split("_")
        # print(file_parts)
        date = file_parts[0]
        stim_type = file_parts[1]
        print("stim type(before...): ", stim_type)
        if len(frlist) == 0:
            frlist = [1]
        if stim_type in ["click", "tonepip"]:
            n = 0
            for i, db in enumerate(dblist):
                for j, fr in enumerate(frlist):
                    x, tb = self.average_within_traces(
                        fd,
                        n,
                        protocol,
                        date,
                        high_pass_filter=high_pass_filter,
                    )
                    if i == 0 and j == 0:
                        abrd = np.zeros((len(dblist), len(frlist), len(x)))
                    abrd[i, j] = x
                    n += 1

        print("stim type:: ", stim_type)
        if stim_type in ["interleaved"]:
            n = 0
            abrd, tb, stim = self.average_across_traces(fd, n, protocol, date, high_pass_filter=high_pass_filter)
            # print(len(frlist), len(dblist))
            abrd = abrd.reshape(len(dblist), len(frlist), -1)
            print("calculated new tb")
        else:
            tb = np.linspace(0, len(abrd[0, 0]) / self.fsamp, len(abrd[0, 0]))
        metadata = {
            "type": "pyabr4",
            "filename": fn,
            "subject_id": subject_id,
            "age": age,
            "sex": sex,
            "amplifier_gain": 1e4,
            "strain": strain,
            "weight": weight,
            "genotype": genotype,
            "record_frequency": self.fsamp
        }

        self.plot_abrs(abrd, tb, stim_type, dblist=dblist, frlist=frlist, metadata=metadata,
                       stim_type=stim_type, maxdur=maxdur, highpass=300.0,
                       use_matplotlib=True)


if __name__ == "__main__":
    fn = "abr_data/2024-11-13/clicks"
    fn = "abr_data/2024-11-15-B/interleaved_plateau"
    fn = "abr_data/2024-12-10-p572-A/interleaved_plateau"
    # fn = "abr_data/2024-12-11-p573-A/interleaved_plateau"
    if not Path(fn).is_dir():
        print(f"Directory {fn} does not exist")
        print(list(Path("abr_data").glob("*")))
        exit()
    files = list(Path(fn).glob("*.p"))
    print(str(files[0]))
    print(files[0].is_file())

    AR = AnalyzeABR()
    # AR.show_calibration(files[0])
    # AR.show_calibration_history()
    # exit()
    d = AR.read_abr_file(str(files[0]))
    print(d["stimuli"].keys())
    HPF = 300.0
    # AR.show_stimuli(files[0])
    AR.read_and_average_abr_files(str(files[0]), high_pass_filter=HPF, maxdur=12.0)
