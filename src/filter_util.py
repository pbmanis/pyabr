"""
Utility.py - general utility routines::
    - power spectrum
    - elliptical filtering
    - handling very long input lines for dictionaries
    - general measurement routines for traces (mean, std, spikes, etc)

"declassed", 7/28/09 p. manis
Use as::
    import Utility as Utils
    then call Utils.xxxxx()

"""

# January, 2009
# Paul B. Manis, Ph.D.
# UNC Chapel Hill
# Department of Otolaryngology/Head and Neck Surgery
# Supported by NIH Grants DC000425-22 and DC004551-07 to PBM.
# See license
#
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import fnmatch
import gc
import os

from typing import Union, List, Tuple
from random import sample

import numpy as np
import scipy.signal
from numba import jit
from numpy import ma as ma
from scipy import fftpack as spFFT

debugFlag = False


class ScriptError(Exception):
    pass


class Utility:
    """A class of various utility routines for doing signal processing,
    spike finding, threshold finding, spectral analysis, etc.
    """

    def __init__(self):
        self.debugFlag = False

    def setDebug(self, debug=False):
        if debug:
            self.debugFlag = True
        else:
            self.debugFlag = False

    def pSpectrum(
        self, data: np.ndarray, samplefreq: float = 44100
    ) -> Tuple[np.ndarray, np.ndarray]:
        npts = len(data)
        # we should window the data here
        if npts == 0:
            print("? no data in pSpectrum")
            return
        # pad to the nearest higher power of 2
        (a, b) = np.frexp(npts)
        if a <= 0.5:
            b = b = 1
        npad = 2**b - npts
        if debugFlag:
            print(("npts: %d   npad: %d   npad+npts: %d" % (npts, npad, npad + npts)))
        padw = np.append(data, np.zeros(npad))
        npts = len(padw)
        sigfft = spFFT.fft(padw)
        nUniquePts = np.ceil((npts + 1) / 2.0)
        sigfft = sigfft[0:nUniquePts]
        spectrum = abs(sigfft)
        spectrum = spectrum / float(npts)  # scale by the number of points so that
        # the magnitude does not depend on the length
        # of the signal or on its sampling frequency
        spectrum = spectrum**2  # square it to get the power
        spmax = np.amax(spectrum)
        spectrum = spectrum + 1e-12 * spmax
        # multiply by two (see technical document for details)
        # odd nfft excludes Nyquist point
        if npts % 2 > 0:  # we've got odd number of points fft
            spectrum[1 : len(spectrum)] = spectrum[1 : len(spectrum)] * 2
        else:
            spectrum[1 : len(spectrum) - 1] = (
                spectrum[1 : len(spectrum) - 1] * 2
            )  # we've got even number of points fft
        freqAzero = np.arange(0, nUniquePts, 1.0) * (samplefreq / npts)
        return (spectrum, freqAzero)

    def sinefit(self, x: np.ndarray, y: np.ndarray, F: float) -> Tuple[float, float]:
        """LMS fit of a sine wave with period T to the data in x and y
        aka "cosinor" analysis.

        """
        npar = 2
        w = 2.0 * np.pi * F
        A = np.zeros((len(x), npar), float)
        A[:, 0] = np.sin(w * x)
        A[:, 1] = np.cos(w * x)
        (p, residulas, rank, s) = np.linalg.lstsq(A, y)
        Amplitude = np.sqrt(p[0] ** 2 + p[1] ** 2)
        Phase = np.arctan2(p[1], p[0])  # better check this...
        return (Amplitude, Phase)

    def sinefit_precalc(self, x: np.ndarray, y: np.ndarray, F: float) -> Tuple[float, float]:
        """LMS fit of a sine wave with period T to the data in x and y
        aka "cosinor" analysis.
        assumes that A (in sinefit) is precalculated

        """
        (p, residulas, rank, s) = np.linalg.lstsq(x, y)
        Amplitude = np.sqrt(p[0] ** 2 + p[1] ** 2)
        Phase = np.arctan2(p[1], p[0])  # better check this...
        return (Amplitude, Phase)

    # filter signal with elliptical filter
    def SignalFilter(
        self, signal: np.ndarray, LPF: float, HPF: float, samplefreq: float
    ) -> np.ndarray:
        if debugFlag:
            print(("sfreq: %f LPF: %f HPF: %f" % (samplefreq, LPF, HPF)))
        flpf = float(LPF)
        fhpf = float(HPF)
        sf = float(samplefreq)
        sf2 = sf / 2
        wp = [fhpf / sf2, flpf / sf2]
        ws = [0.5 * fhpf / sf2, 2 * flpf / sf2]
        if debugFlag:
            print(
                "signalfilter: samplef: %f  wp: %f, %f  ws: %f, %f lpf: %f  hpf: %f"
                % (sf, wp[0], wp[1], ws[0], ws[1], flpf, fhpf)
            )
        filter_b, filter_a = scipy.signal.iirdesign(wp, ws, gpass=1.0, gstop=60.0, ftype="ellip")
        msig = np.mean(signal)
        signal = signal - msig
        w = scipy.signal.lfilter(filter_b, filter_a, signal)  # filter the incoming signal
        signal = signal + msig
        if debugFlag:
            print(
                ("sig: %f-%f w: %f-%f" % (np.amin(signal), np.amax(signal), np.amin(w), np.amax(w)))
            )
        return w

    # filter with Butterworth low pass, using time-causal lfilter
    def SignalFilter_LPFButter(
        self,
        signal: np.ndarray,
        LPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
    ) -> np.ndarray:
        wn = [LPF / (samplefreq / 2.0)]
        b, a = scipy.signal.butter(NPole, wn, btype="low", output="ba")
        zi = scipy.signal.lfilter_zi(b, a)
        if bidir:
            out = scipy.signal.filtfilt(b, a, signal)  # , zi=zi * signal[0])
        else:
            out = scipy.signal.lfilter(b, a, signal)  # , zi=zi * signal[0])
        return np.array(out)

    # filter with Butterworth high pass, using time-causal lfilter
    def SignalFilter_HPFButter(
        self,
        signal: np.ndarray,
        HPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
    ) -> np.ndarray:
        flpf = float(HPF)
        sf = float(samplefreq)
        wn = [flpf / (sf / 2.0)]
        b, a = scipy.signal.butter(NPole, wn, btype="high", output="ba")
        # zi = scipy.signal.lfilter_zi(b, a)
        if bidir:
            out = scipy.signal.filtfilt(b, a, signal)  # , zi=zi*signal[0])
        else:
            out = scipy.signal.lfilter(b, a, signal)  # , zi=zi*signal[0])
        return np.array(out)

    # filter with Bessel high pass, using time-causal lfilter
    def SignalFilter_HPFBessel(
        self,
        signal: np.ndarray,
        HPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
    ) -> np.ndarray:
        flpf = float(HPF)
        sf = float(samplefreq)
        wn = [flpf / (sf / 2.0)]
        b, a = scipy.signal.bessel(NPole, wn, btype="high", output="ba")
        # zi = scipy.signal.lfilter_zi(b, a)
        if bidir:
            out = scipy.signal.filtfilt(b, a, signal)  # , zi=zi*signal[0])
        else:
            out = scipy.signal.lfilter(b, a, signal)  # , zi=zi*signal[0])
        return np.array(out)

    # filter signal with low-pass Bessel
    def SignalFilter_LPFBessel(
        self,
        signal: np.ndarray,
        LPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
        reduce: bool = False,
    ) -> np.ndarray:
        """Low pass filter a signal, possibly reducing the number of points in the
        data array.
        Parameters
        ----------
        signal: a numpya array of dim = 1, 2 or 3. The "last" dimension is filtered.

        LPF: low pass filter frequency, in Hz

        samplefreq: sampline frequency (points/second)

        NPole: number of poles in the filter.

        reduce: Flag that controls whether the resulting data is subsampled or not

        """
        if self.debugFlag:
            print(f"sfreq: {samplefreq:f}  LPF: {LPF:f}")
        wn = [LPF / (samplefreq / 2.0)]
        filter_b, filter_a = scipy.signal.bessel(NPole, wn, btype="low", output="ba")
        reduction = 1
        if reduce:
            if LPF <= samplefreq / 2.0:
                reduction = int(samplefreq / LPF)
        if self.debugFlag is True:
            print(
                f"signalfilter: samplef: {samplefreq:f}  wn: {wn:f}  lpf: {LPF:f}  NPoles: {NPole:d}"
            )
            sm = np.mean(signal)
            if bidir:
                w = scipy.signal.filtfilt(
                    filter_b, filter_a, signal - sm
                )  # filter the incoming signal
            else:
                w = scipy.signal.lfilter(
                    filter_b, filter_a, signal - sm
                )  # filter the incoming signal

            w = w + sm
            if reduction > 1:
                w = scipy.signal.resample(w, reduction)
            return w
        if signal.ndim == 2:
            sh = np.shape(signal)
            for i in range(0, np.shape(signal)[0]):
                sm = np.mean(signal[i, :])
                if bidir:
                    w1 = scipy.signal.filtfilt(filter_b, filter_a, signal[i, :] - sm)
                else:
                    w1 = scipy.signal.lfilter(filter_b, filter_a, signal[i, :] - sm)

                w1 = w1 + sm
                if reduction == 1:
                    w1 = scipy.signal.resample(w1, reduction)
                if i == 0:
                    w = np.empty((sh[0], np.shape(w1)[0]))
                w[i, :] = w1
            return w
        if signal.ndim == 3:
            sh = np.shape(signal)
            for i in range(0, np.shape(signal)[0]):
                for j in range(0, np.shape(signal)[1]):
                    sm = np.mean(signal[i, j, :])
                    if bidir:
                        w1 = scipy.signal.filtfilt(filter_b, filter_a, signal[i, j, :] - sm)
                    else:
                        w1 = scipy.signal.lfilter(filter_b, filter_a, signal[i, j, :] - sm)
                    w1 = w1 + sm
                    if reduction == 1:
                        w1 = scipy.signal.resample(w1, reduction)
                    if i == 0 and j == 0:
                        w = np.empty((sh[0], sh[1], np.shape(w1)[0]))
                    w[i, j, :] = w1
            return w
        if signal.ndim > 3:
            print("Error: signal dimesions of > 3 are not supported (no filtering applied)")
            return signal
