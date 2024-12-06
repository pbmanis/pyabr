"""
Calibrations
"""

import numpy as np
from pathlib import Path
from typing import Union
import src.pystim3 as PS
import pyqtgraph as pg
import pyqtgraph.configfile as configfile
from pyqtgraph.Qt import QtGui
import scipy.fft as fft
from scipy.signal import get_window
import src.wave_generator
import read_calibration
# from librosa import load as load_audio
# from librosa import stft, amplitude_to_db
# from librosa.display import specshow
from scipy.signal import spectrogram
import pyqtspecgram
from scipy.signal import butter, filtfilt

SPLCAL_ToneVoltage = read_calibration.SPLCAL_Tone_Voltage  # 10.0  # from matlab calibration program.

configfilename = "config/abrs.cfg"
# get the latest calibration file:
cfg = configfile.readConfigFile(configfilename)
print(cfg)
fn = Path(cfg['calfile'])
# fn = Path("E:/Users/Experimenters/Desktop/ABR_Code/frequency_MF1.cal")
print(fn.is_file())
caldata = read_calibration.get_calibration_data(fn)
micdata = read_calibration.get_microphone_calibration("calfiles/microphone_7016#10252.cal")
WG = src.wave_generator.WaveGenerator(caldata=caldata)

# MICS = {
#     "10252": {
#         "size": 0.25,
#         "ref_V": 3.85 * 1e-3,  # mV/Pa
#         "ref_dB": 94.0,
#         "amp_gain": 20.0,
#     },
#     "9945": {
#         "size": 0.25,
#         "ref_V": 3.85 * 1e-3,  # mV/Pa
#         "ref_dB": 94.0,
#         "amp_gain": 20.0,
#     },
#     "39279": {
#         "size": 0.5,
#         "ref_V": 16.22 * 1e-3,  # mV/Pa
#         "ref_dB": 94.0,
#         "amp_gain": 20.0,
#     },
# }


class Calibrations:

    def __init__(self, caldata:Union[dict, None] = None, micdata:Union[dict, None]=None):

        # set up hardware...
        assert micdata is not None
        self.PS = PS.PyStim(
            required_hardware=["NIDAQ", "RP21", "PA5"], acquisition_mode="calibrate"
        )
        self.PS.setAttens()
        self.mic = micdata
        self.caldata = caldata


    def compute_SPL(self, mic_vrms_V):
        vgain = 10 ** (self.mic["cal_gain"] / 20.0)
        print(f"    self.mic ref db: {self.mic['cal_ref']:5.1f}")
        print(f"    actual mic voltage: {mic_vrms_V / vgain:9.5f}")
        print(f"    mic ref V rms: {self.mic['Vrms']:9.4f}")
        print(f"    log x 20: {20 * np.log10((mic_vrms_V / vgain) / (self.mic['Vrms'])):6.1f}")

        dBSPL = (
            self.mic["cal_ref"]
            # - self.mic["cal_gain"]
            + 20.0 * np.log10((mic_vrms_V) / (self.mic["Vrms"]))
        )
        return dBSPL

    def compute_dBPerVPa(self, mic_vrms_V):
        dBPerVPa = (
            20.0 * np.log10(mic_vrms_V)
            - self.mic["cal_gain"]
            - (self.mic["cal_ref"] - 94.0)
        )
        return dBPerVPa

    def compute_mVPerPa(self, dbperpa: float):
        mVPerPa = 1000.0 * 10.0 ** (dbperpa / 20.0)
        return mVPerPa

    def get_signal(self, waveform, attns=[120, 120]):
        """set_wave : set the waveform to play out the daq card
        Input is on Ch2 only when using mic_record.rcx
        """
        self.PS.play_sound(
            waveform, None, attns=attns, samplefreq=self.PS.Stimulus.NI_out_sampleFreq
        )

        pg.plot(self.PS.t_record, self.PS.ch2)
        pg.exec()

    def do_padded_fft(self, signal, sampling_rate):
        ns = 8
        N = ns * signal.shape[0]
        m = 513
        w = get_window("blackmanharris", signal.shape[0])

        fft_result = fft.rfft(signal * w, n=N)
        freqs = fft.rfftfreq(N, 1.0 / (sampling_rate))
        fft_result = (2.0 / len(signal * ns)) * np.abs(fft_result)
        return {"fft": fft_result, "freqs": freqs}

    def check_microphone(self, mic_ser_no):
        """
        Check the microphone sensitivity.
        Assume calibrator is set to 94 dB SPL
        """
        self.mic = MICS[mic_ser_no]
        print(f"Checking against mic: {mic_ser_no:s}")  #   "ref_V": 3.85, # mV/Pa
        print(f"   {self.mic['ref_V']} mV/Pa ")
        print(f"   {self.mic['ref_dB']: 6.1f} dBSPL")
        print(f"   {self.mic['amp_gain']:5.1f} amplifier gain")

        outrate = cal.PS.Stimulus.NI_out_sampleFreq
        tonedur = 1.0  # seconds
        npts = int(tonedur * outrate)
        t = np.linspace(0, tonedur, npts)
        wave = np.sin(2.0 * np.pi * 4000.0 * t)
        cal.PS.setAttens()
        attns = [120, 120]  # turn speaker off

        self.PS.play_sound(
            wave, None, attns=attns, samplefreq=outrate
        )
        self.ad_data = np.array(self.PS.ch2)
        self.ad_data = self.butter_highpass_filter(self.ad_data, 200.0, fs=1./self.PS.Stimulus.RP21_in_sampleFreq, order=5)

        re_freqs, re_amps = self.compute_microphone_spectrum()
        self.compute_mic_factors(re_freqs, re_amps)



    def butter_highpass(self, cutoff:float, fs:float, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff:float, fs:float, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y


    def compute_microphone_spectrum(self):
        # to check calculatoin:
        print(self.ad_data)
        # self.ad_data = np.sin(2.0 * np.pi * 1000.0 * self.PS.t_record)*0.01622*10 # multiply by mic gain (20 db = 10x V)
        res = self.do_padded_fft(self.ad_data, self.PS.Stimulus.RP21_in_sampleFreq)
        amps = res["fft"]
        freqs = res["freqs"]
        freqlen = freqs.shape[0]
        re_amps = 2 * amps[: int(freqlen / 2)]
        re_freqs = freqs[: int(freqlen / 2)]
        return re_freqs, re_amps

    def compute_mic_factors(self, re_freqs, re_amps):
        i_ampmax = np.argmax(re_amps)
        print("max amp: ", np.max(re_amps))
        ampmax = re_amps[i_ampmax]
        freqmax = re_freqs[i_ampmax]
        dbspl = self.compute_SPL(ampmax)
        print(f"Amplitude {ampmax:.3e}  at {freqmax:8.1f}, dB: {dbspl:5.1f} ")

        dbpervpa = self.compute_dBPerVPa(ampmax)
        mvperpa = self.compute_mVPerPa(dbpervpa)
        print(f"dB Per V Pa: {dbpervpa:9.4f}")
        print(f"mV per pA: {mvperpa:8.2f}")
        return mvperpa, dbpervpa, dbspl


    def plot_specgram(self, plot, f_a, t_a, S_a, NFFT, Fs, Fc):
        f_a = np.fft.fftshift(f_a) + Fc
        lSa = np.fft.fftshift(10 * np.log10(S_a), axes=0)
        print(lSa.shape)
        print(lSa[0].shape)
        plot.setXRange(0, lSa.shape[0] * NFFT / Fs)
        plot.setYRange(Fc + -Fs / 2, Fc + Fs / 2)
        plot.setLabel(axis='left', text='Frequency [Hz]')
        plot.setLabel(axis='bottom', text='time [s]')
        # set scroll limits
        # plot.setLimits(xMin=t_a[0] - (t_a[10] - t_a[0]), xMax=t_a[-1] + (t_a[10] - t_a[0]),
        #             yMin=f_a[0] - (f_a[10] - f_a[0]), yMax=f_a[-1] + (f_a[10] - f_a[0]))

        # Fit the plot to the axes
        tr = tr = QtGui.QTransform()
        tr.translate(0, Fc - Fs / 2)
        tr.scale(NFFT / Fs, Fs / NFFT)

        img = img = pg.ImageItem(border='w')
        # img.setImage(tt[0][:, :, 1:].astype(np.uint16))
        img.setImage(lSa)
        img.setTransform(tr)

        # Colourmap
        # cmap = pg.colormap.get('CET-L9')
        cmap = cmap = pg.colormap.get('viridis')  # matplotlib style
        minv = 0
        maxv = 120.0
        # minv, maxv = np.nanmin(np.nanmin(lSa[lSa != -np.inf])), np.nanmax(np.nanmax(lSa))
        bar = bar = pg.ColorBarItem(interactive=True, values=(minv, maxv), colorMap=cmap, label='magnitude [dB]')
        bar.setImageItem(img, insert_in=plot)

        plot.addItem(img)
        plot.showAxes(True)

    def plot_mic_data(self, spec_x: np.array = None, spec_y: np.array = None, specplot:bool=True, stimwave=None):
        app = pg.mkQApp("Microphone Data Plot")
        win = pg.GraphicsLayoutWidget(show=True, title="Microphone Data Plot")
        win.setGeometry(100, 200, 400, 600) 
        # win.setWindowTitle(f"File: {str(Path(fn).parent)}")
        p_wave = win.addPlot(title=f"Microphone Waveform")
        p_wave.plot(self.PS.t_record, self.ad_data)
        win.nextRow()
        p_spec = win.addPlot(title=f"Microphone Spectrum")

        if spec_x is not None:
            p_spec.plot(spec_x, spec_y, pen=pg.mkPen('y'))
        if specplot:
            NFFT=1024
            Fc = 0.
            Fs = self.PS.Stimulus.RP21_in_sampleFreq
            f_a, t_a, S_a, _ = pyqtspecgram.pyqtspecgram(self.ad_data, NFFT, Fs=Fs, Fc=0)
            self.plot_specgram(p_spec,f_a, t_a, S_a, NFFT, Fs, Fc)
        # print("stimwave: ", stimwave)
        if stimwave is not None:
            win.nextRow()
            p_stim = win.addPlot(title=f"Stimulus Waveform")
            # print(WG.wave_time)
            p_stim.plot(WG.wave_time, stimwave, pen=pg.mkPen("c"))
        pg.exec()

    def rms(self, signal):
        return np.sqrt(np.mean(np.square(signal)))

    def test_wave(self):
        WG.setup(protocol=WG.stim, frequency=1000000)
        WG.make_waveforms(WG.protocol["protocol"]["stimulustype"])
        # WG.plot_stimulus_wave()
        # print(WG.wave_matrix.keys())

        
        for wmk in WG.wave_matrix.keys():  # do each of the stimuli
            self.PS.play_sound(
                WG.wave_matrix[wmk]["sound"],
                WG.wave_matrix[wmk]["sound"],
                attns=[wmk[1], wmk[1]],
                samplefreq=WG.sfout,
                postduration = 0.05
            )
            self.PS.setAttens()


            self.ad_data = np.array(self.PS.ch2)
        
            # self.ad_data = self.butter_highpass_filter(self.ad_data, 200.0, fs=self.PS.Stimulus.RP21_in_sampleFreq, order=5)

            re_freqs, re_amps = self.compute_microphone_spectrum()
            self.plot_mic_data(
                spec_x=re_freqs,
                spec_y=re_amps,
                stimwave=WG.wave_matrix[(WG.protocol["protocol"]["stimulustype"], 0)][
                    "sound"
                ],
                specplot=False
            )
            w0 = list(WG.wave_matrix.keys())[0]
            print(WG.wave_matrix[w0].keys())
            pstarts = WG.wave_matrix[w0]["pip_starts"]
            pstarts = [x[0] for x in pstarts]
            pdurs = WG.wave_matrix[w0]["pip_durations"]
            print(pstarts)
            ps = pstarts[-7:]
            print("ps: ", ps)
            fr = WG.wave_matrix[w0]['frequencies'][-7:]
            fs=self.PS.Stimulus.RP21_in_sampleFreq
            last_set = int(ps[0]*fs)
            print(pdurs)
            end_set = int((ps[-1]+0.010)*fs)
            print("last_set: ", last_set, end_set, last_set/fs, end_set/fs)
            wdata = self.ad_data[last_set:(end_set)]
            t = np.linspace(0, len(wdata)/fs, len(wdata))
            print("ps 0: ", ps[0])
            for i in range(0, len(ps)):
                startt = int(ps[i]*fs)
                endt = startt+int(0.005*fs)
                print(startt, endt)
                rms = self.rms(self.ad_data[startt:endt])
                # dBSPL = mic["cal_gain"] - mic["cal_ref"] + 20.0*np.log10(rms*1e3/mic["mVPerPa"]);
                dBSPL = self.compute_SPL(rms)
                print(f"RMS ({i:d}), {fr[i]:8.1f} Hz {rms:6.5f} V = {dBSPL:6.1f} dBSPL")


        import matplotlib.pyplot as mpl
        mpl.plot(t, wdata)
        mpl.show()

if __name__ == "__main__":
    cal = Calibrations(micdata=micdata)
    # cal.check_microphone("9945")

    # outrate = cal.PS.Stimulus.NI_out_sampleFreq
    # tonedur = 0.5 # seconds
    # npts = int(tonedur*outrate)
    # t = np.linspace(0, tonedur, npts)
    # wave = np.sin(2.0*np.pi*4000.0*t)
    # cal.get_signal(wave, attns=[25, 25])
    # cal.PS.setAttens()

    cal.test_wave()
