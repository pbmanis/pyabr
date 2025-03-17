import os.path
from pathlib import Path
from re import S
from termios import NL1
from tkinter import NONE
from typing import List, Union

import matplotlib.cm
import matplotlib.pyplot as mpl
import numpy as np
import scipy.signal

import src.peakdetect as peakdetect  # from Brad Buran's project, but cloned and modified here



class Analyzer(object):
    """
    Provide analysis functions for ABRs.
    """

    def __init__(self, sample_frequency: float = 1e5):
        """Initialize the analyzer

        Parameters
        ----------
        sample_frequency : float, optional
            sample frequency for the traces, by default 1e5 khz
        """
        self.ppioMarker = "s"
        self.rmsMarker = "o"
        self.psdMarker = "*"
        self.baselineMarker = "+"
        self.sample_freq = sample_frequency
        self.p1n1p2 = None

    def set_baseline(self, timebase, baseline: List = [0.020, 0.025]):
        if np.max(timebase) < baseline[0]:
            baseline = [np.max(timebase) - 0.002, np.max(timebase)]
        return baseline

    def analyze(
        self, timebase: np.ndarray, waves: np.ndarray, response_window: List = [0.0022, 0.0080], dev=2.5,
        spls: list = None
    ):
        """Perform initial analysis to get Buran's results, peak-to-peak IO,
        and the rms of the signal an the baseline.

        Parameters
        ----------
        timebase : np.ndarray
            The timebase for the data (msec)
        waves : np.ndarray
            The SPL x npoints array of abr waves (in V)
        response_window : list, optional (msec)
            Time window to use for the response, by default [2.2, 8.0]
        dev: for peak detection, deviation (4-5 seems good)
        """
        # self.sample_rate = 0.001 * (timebase[1] - timebase[0])
        # self.sample_freq = 1.0 / self.sample_rate
        self.sample_rate = 1.0 / self.sample_freq
        self.waves = waves
        self.timebase = timebase
        baseline = self.set_baseline(timebase)

        self.get_triphasic(min_lat=response_window[0], dev=dev)
        self.ppio = self.peaktopeak(response_window) - self.peaktopeak(baseline)
        self.rms_response = self.measure_rms(response_window)
        self.rms_baseline = self.measure_rms(baseline)

        # self.true_threshold, self.median_sd = self.thresholds(timebase=timebase, waves=waves, spls=spls)
        # self.specpower(waves)

    def peaktopeak(self, time_window: Union[List, np.ndarray]) -> np.ndarray:
        """Measure the peak to peak values in a set of traces
        Works on the data in self.waves, and computes the p-p values
        for each trace.

        Parameters
        ----------
        time_window : List, np.ndarray
            start and end times for the measurement.

        Returns
        -------
        pp : np.ndarray
            peak-to-peak measure of data in the window for each wave

        """
        tx = self.gettimeindices(time_window)
        pp = np.zeros(self.waves.shape[0])
        # print("wave shape p-p: ", self.waves.shape)
        # print("tx", tx)
        for i in range(self.waves.shape[0]):
            pp[i] = np.max(self.waves[i, tx]) - np.min(self.waves[i, tx])
        return pp

    def get_triphasic(self, min_lat: float = 2.2, dev: float = 2.5):
        """Use Brad Buran's peakdetect routine to find the peaks and return
        a list of peaks. Works 3 times - first run finds all the positive peaks,
        and the second run finds the negative peaks that follow the positive
        peaks. The last run finds the next positive peak after the negative
        peak. This yields P1-N1-P2, which is the returned value. Note that the
        peaks from peakdetect may not be "aligned" in the sense that it is
        possible to find two positive peaks in succession without a negative
        peak.

        Parameters
        ----------
        min_lat : float, optional
            Minimum latency, msec, by default 2.2
        dev : float, optional
            "deviation" or threshold, by default 2.5 x the reference or
            baseline time window.
        """

        p1 = {}
        n1 = {}
        p2 = {}
        for j in range(self.waves.shape[0]):
            p1[j] = peakdetect.find_np(
                self.sample_freq,
                self.waves[j, :],
                nzc_algorithm_kw={"dev": dev},
                guess_algorithm_kw={"min_latency": min_lat},
            )

            if len(p1[j]) > 0:
                n1[j] = peakdetect.find_np(
                    self.sample_freq,
                    -self.waves[j, :],
                    nzc_algorithm_kw={"dev": dev},
                    guess_algorithm_kw={"min_latency": self.timebase[p1[j][0]]},
                )  # find negative peaks after positive peaks
            else:
                n1[j] = np.nan
            if len(n1[j]) > 0:
                p2[j] = peakdetect.find_np(
                    self.sample_freq,
                    self.waves[j, :],
                    nzc_algorithm_kw={"dev": dev},
                    guess_algorithm_kw={"min_latency": self.timebase[n1[j][0]]},
                )  # find negative peaks after positive peaks
            else:
                p2[j] = np.nan
        self.p1n1p2 = {"p1": p1, "n1": n1, "p2": p2}

    def measure_rms(self, time_window: Union[List, np.ndarray]) -> np.ndarray:
        """Measure the rms values in a set of traces.
        Works on the data in self.waves, and computes the rms values
        for each trace.

        Parameters
        ----------
        time_window : List, np.ndarray
            start and end times for the measurement
        Returns
        -------
        rms : np.ndarray
            peak-to-peak measure of data in the window for each wave

        """

        tx = self.gettimeindices(time_window)
        rms = np.zeros(self.waves.shape[0])
        for i in range(self.waves.shape[0]):
            rms[i] = np.std(self.waves[i][tx])
        return rms

    def gettimeindices(self, tr):
        (x,) = np.where((tr[0] <= self.timebase) & (self.timebase < tr[1]))
        return x

    def specpower(
        self,
        waves,
        spls,
        fr=[500.0, 1500.0],
        win=[0, -1],
        ax=None,
        ax2=None,
        lt="-",
        cindex=0,
    ):
        fs = 1.0 / self.sample_rate
        psd = [None] * waves.shape[0]
        psdwindow = np.zeros(waves.shape[0])
        cmap = matplotlib.cm.get_cmap("tab20")
        nperseg = 256
        maxseg = win[1] - win[0]
        if maxseg < nperseg:
            nperseg = maxseg
        for i in range(waves.shape[0]):
            freqs, psd[i] = scipy.signal.welch(
                1e6 * waves[i][win[0] : win[1]],
                fs,
                nperseg=nperseg,
                nfft=8192,
                scaling="density",
            )
            (frx,) = np.where((freqs >= fr[0]) & (freqs <= fr[1]))
            psdwindow[i] = np.nanmax(psd[i][frx[0] : frx[-1]])
            if ax is not None:
                ax.semilogx(
                    freqs,
                    psd[i],
                    linestyle=lt,
                    label=f"{spls[i]:.1f}",
                    color=cmap(i / 20.0),
                )
                # ax.set_ylim([0.1e-4, 0.1])
                ax.set_xlim([10.0, 2500.0])
                ax.set_xlabel("F (Hz)")
                ax.set_ylabel(r"PSD ($\mu V^2/Hz$)")
            if ax2 is not None:
                tb = fs * np.arange(0, len(waves[i][win[0] : win[1]]))
                ax2.plot(tb, waves[i][win[0] : win[1]], linestyle=lt, color=cmap(i / 20.0))
        self.fr = freqs
        self.psd = psd
        self.psdwindow = psdwindow
        return psdwindow

    def thresholds(
        self,
        timebase: np.ndarray,
        waves: np.ndarray,
        spls: Union[List, np.ndarray],
        response_window=[1.0, 8.0],
        baseline_window=[20, 25],
        SD=3.0,
        ref_db:Union[int, None]=None,
    ):
        """Measure the threshold for a response in each wave
        Auto threshold detection: BMC Neuroscience200910:104  DOI:
        10.1186/1471-2202-10-104 Use last 10 msec of 25 msec window for SD
        estimates Computes SNR (max(abs(signal))/reference SD) for a group of
        traces The reference SD is the MEDIAN SD across the entire intensity
        run, to minimize the effects of noise in just one trace.

        Parameters
        ----------
        timebase: np.ndarray
            timebase, as 1D np.ndarray
        waves : np.ndarray
            waveforms, as a 2D array
        spls : Union[List, np.darray]
            List of sound pressure levels corresponding to the waveforms
        response_window : list, optional
            time window for measuring the responses, by default [1.0, 8.0]
        baseline_window : list, optional
            time window for the "baseline", by default [20, 25]
        SD : float, optional
            Size of response relative to baseline to be
            considered a signal, by default 3.0
        ref_db : Union[int, None], optional
            which trace to use for the reference SD. If None, then use a baseline
            window (as above). If an integer, then we use
            that trace (the entire trace) as the reference SD, by default None
            This allows you to select a below-threshold set of traces for comparision,
            or if that is not available, use the SD for each trace.
            When doing the "interleaved_plateau" method, there is not time between 
            stimuli to measure the SD well, so usually set ref_db to 0
            to use the lowest-level trace as the reference, if that is appropriate.
            If not, then use the first 1 msec for the baseline window.
            
        Returns
        -------
        float
            threshold value (SPL)
        """
        self.timebase = timebase
        refwin = self.gettimeindices(baseline_window)
        sds = np.std(waves[:, refwin[0] : refwin[-1]], axis=1)*SD
        self.median_sd = np.nanmean(sds)*SD # np.nanmedian(sds)
        tx = self.gettimeindices(response_window)
        self.max_wave = np.max(np.fabs(waves[:, tx[0] : tx[-1]]), axis=1)
        true_thr = 110.0  # np.max(spls) + 10  # force to have a special value
        n_consec_fails = 0
        last_p1_lat = 0
        # if self.p1n1p2 is None:
        #     self.get_triphasic()
        self.p1_latencies = []
        self.p1n1p2_amplitudes = []
        self.p1n1_amplitudes = []
        for i, s in enumerate(spls):
            j = len(spls) - i - 1
            p1_index = self.p1n1p2["p1"][j][0]
            n1_index = self.p1n1p2["n1"][j][0]
            p2_index = self.p1n1p2["p2"][j][0]
            p1_lat = self.timebase[p1_index]
            n1_lat = self.timebase[n1_index]
            p2_lat = self.timebase[p2_index]
            p1n1p2_j = ((waves[j, p1_index] + waves[j, p2_index]) / 2.0) - waves[j, n1_index]
            p1n1p2_i = ((waves[i, p1_index] + waves[i, p2_index]) / 2.0) - waves[i, n1_index]
            p1n1_i = waves[i, p1_index] - waves[i, n1_index]
            self.p1n1p2_amplitudes.append([s, p1n1p2_i])
            self.p1n1_amplitudes.append([s, p1n1_i])
            # print("spl:", s, "p1n1_i: ", p1n1_i)
            true_thr = 100.0 # start at the top
            # continue
            if p1_lat > 4.25:
                n_consec_fails += 1
                if n_consec_fails >= 2:
                    break
                continue
            if p1_lat > last_p1_lat:
                last_p1_lat = p1_lat  # first one, save it
                self.p1_latencies.append([s, p1_lat])
                n_consec_fails = 0
            if p1_lat < (last_p1_lat - 0.3):
                n_consec_fails += 1
                if n_consec_fails >= 2:
                    break
                continue
            # symmertry ratio of latencies
            sym = (n1_lat - p1_lat) / (p2_lat - n1_lat)
            # print("sym: ", sym)
            if (
                ((n1_lat - p1_lat) > 0.8)
                or (p2_lat < n1_lat)
                or (p2_lat - p1_lat) > 1.5
                or (p2_lat - n1_lat) > 1.25
                or sym > 1.6
                or sym < 0.5
            ):
                n_consec_fails += 1
                if n_consec_fails >= 2:
                    break
                continue

            # print('p1n1p2: p1: ', self.p1n1p2['p1'][j][0], waves[i, self.p1n1p2['p1'][j][0]])
            # print('p1n1p2: n1: ', self.p1n1p2['n1'][j][0], waves[i, self.p1n1p2['n1'][j][0]])
            # print(waves[j, self.p1n1p2['p1'][j][0]:self.p1n1p2['n1'][j][0]])

            # print(self.median_sd * SD)
            if p1n1p2_j >= self.median_sd:
                true_thr = spls[j]
                n_consec_fails = 0
            else:
                n_consec_fails += 1
                if n_consec_fails >= 2:
                    break

        # print("True Thr: ", true_thr, "SD: self.median_sd: ", self.median_sd)
        return true_thr, self.median_sd

    def threshold_spec(
        self,
        waves: Union[List, np.ndarray],
        spls: Union[List, np.ndarray],
        response_window=[1.0, 8.0],
        baseline_window=[20, 25],
        SD=4.0,
        spec_bandpass=[800.0, 1200.0],
    ):
        """Auto threshold detection:
        BMC Neuroscience200910:104  DOI: 10.1186/1471-2202-10-104 Use last part
        of the response window for SD estimates Computes SNR
        (max(abs(signal))/reference SD) for a group of traces The reference SD
        is the MEDIAN SD across the intensity run.

        MODIFIED version: criteria based on power spectrum in a narrow power
        window.

        Parameters
        ----------
        waves : Union[List, np.ndarray]
            waveforms to measure (2D array)
        spls : Union[List, np.ndarray]
            spls corresponding to first dimension of the waveforms
        response_window : list, optional
            response window, by default [1.0, 8.0] (msec)
        baseline_window : list, optional
            baseline window, by default [20, 25] (msec)
        spec_bandpass : list, optional
            bandpass window to measure the spectrum: by default [800.0, 1200.0]
        SD : float, optional
            relative size of the response in the response window, compared to
            the "baseline" window, to consider the presence of a valid response,
            by default 4.0

        Returns
        -------
        float
            SPL threshold for a response
        """
        showspec = False
        refwin = self.gettimeindices(self.set_baseline(self.timebase, baseline=baseline_window))
        if showspec:
            fig, ax = mpl.subplots(1, 2, figsize=(10, 5))
        else:
            ax = [None, None]
        sds = self.specpower(
            waves,
            spls,
            fr=spec_bandpass,
            win=[refwin[0], refwin[-1]],
            ax=ax[0],
            ax2=ax[1],
            lt="--",
        )
        self.median_sd = np.nanmedian(sds)
        tx = self.gettimeindices(response_window)
        self.max_wave = self.specpower(
            waves,
            spls,
            fr=spec_bandpass,
            win=[tx[0], tx[-1]],
            ax=ax[0],
            ax2=ax[1],
            lt="-",
        )

        true_thr = 110.0  # np.max(spls) + 10  # force to have a value outside the stimulus range
        # so we can detect the maximum as a threshold as well.
        for i, s in enumerate(spls):
            j = len(spls) - i - 1
            if self.max_wave[j] >= self.median_sd * SD:
                true_thr = spls[j]
            else:
                break

        if showspec:
            mpl.show()
        return true_thr
