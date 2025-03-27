pyabr
=====

A python program to do abrs with TDT systems (RP2.1, PA5, and possibily with
RZ/RX processors) and an NI DAC.

Installation (Windows)
----------------------

Data acquisiton only works under windows.

Install Python3.12 from the python.org site (don't use the Microsoft site)
Install TDT drivers from TDT (rpsvdx)
Install NIDAQ-mx (even if you are not using it!)
Install the active-x drivers from TDT. This may not be supported much longer however.

Run the installation script to create a virtual environment:
in the bash (git bash), run:

`
$ ./make_env.bat
`

Then activate the environment:

`
pyabr_venv\Scripts\activate
`

Run the test program:

To play a tone pip:

`
python tests/play_test_sounds.py pip
`

To run pyabr3:

`python pyabr3.py`

This will bring up a window with a panel on the left allowing access to parameters, loading protocols, recording, 
and a set of docks on the right showing the recorded responses, with a running average. 
Recording can be paused and restarted. 

The program is threaded, so that stimulus presentation and data acquisiton can occur simultaneously
with interactions with the graphics. The live graphs can be zoomed or panned; the graphs will however reset
on the next stimulus cycle. 

Installation (mac os)
---------------------
Note: Install on mac os only provides data analysis, not data acquisition.

Run the script:

```
./make_env.sh
```
 to create a working environment (do NOT do this as sudo).

Activate the environment:

```
source pyabr_venv/bin/activate
```

Analysis
--------
Analysis is provided by the script *read_abr.py*. 
This program will need access to a configuration file (see the __main__ section for how this is set up).
As currently configured, it will be necessary to edit the read_abr.py file in order to
do analysis. Items such as paths to data, specifics about file names, etc, will have to be
adjusted for your configuration. 

The program expects the data to be in a specific structure under a directory:
experimental_directory_name:
    Subject_identifier (a directory)
        Clicks
            either matlab output (.txt files) or .p files from pyabr3
        Tones
            matlab output (.txt files)
        Interleaved_plateau
            pyabr3 output, .p files.

The names Clicks, Tones and Interleaved_plateau are hard-coded.
the subject identifier is expected to have a specific format:
"CBA_M_SUBID_p48_(date)_etc."

Where:
```
    CBA is the strain or cross identifier 
    M or F is the sex
    SUBID is a 1-2 letter identifier, followed by 1-3 numbers. 
    p48 is the age (in postnatal days)
    (date) is an optional date string, 
    or in some cases, a combined animal identifier.
```



The configuration file (in pyqtgraph configuration format) must have:

```
ABR_settings:
            HPF: 300   # high pass filter, Hz
            LPF: 3000  # low pass filter, Hz
            maxdur: 12.0   # msec
            stack_increment: 3.0  # spacing between stacked traces, in uV (per example)
            click_threshold_factor: 3.0  # threshold for ABR detection, clicks relative to the noise level
            tone_threshold_factor: 4.0  # threshold for ABR detection, tones relative to the noise level
        
        ABR_subject_excludes:
            CBA_F_R024_P13_NT_CBA24: "no responses"
            CBA_F_R026_P223_NT_CBA26: "old - replaced with current data "
            CBA_F_R027_P309_NT_CBA27: "old - replaced with current data"
            CBA_M_R019_P13_NT_CBA19: "something wacky with data"
            CBA_F_N003_p27_NT: "only 4 levels, 20 dB apart"
        
        # ABR parameters holds 
        ABR_parameters:
            default:
                minimum_latency: 2.2-3  # seconds
                invert: False
                scale: 1e-1
                fit_index: 2  # index of the first db level to fit in the sequence (assumes ascending)
            N0:  # subject id - most of the ones used in this project
                minimum_latency: 2.2e-3  # seconds
                invert: True
                scale: 1
                fit_index: 4
```
The settings are general settings related to filtering. plot duration and spacing,
and the automated click and tone threshold factors. The next set of 'ABR_subject_excludes'
is specifically to exclude some files in the datasets from the analysis. 
The ABR_parameters define the parameters used in analysis, such as the minimum latency
to the start of the P1 wave, whether the data needs to be inverted (or not), the
voltage scale factor, and the index of the first level of ascending SPLs to be used
for the regression line that is fit to the P1 and N1 latencies, to aid in the determination
of the threshold (and to align the data points for SPLs that are lower than
the threshold, or lower than can be properly identified by the peakdetect routine). For example,
if 10 sound pressure levels were presented (say 10 to 100 dB SPL in 10 dB increments), a value
of 5 would only use the data from 50 to 100 dB SPL for the linear regression, and
the responses at lower SPLS would be determined by extrapolation of the regression
to measure the voltage in the trace at an appropriate latency. Note that at these lower
SPLs, the voltage is taken as the mean in a small window.


Note that for the parameters, the defaults are read first, then specific subjects
are listed if there are to be any changes. All fields must be specified for each
subject. 

Included modules
----------------

pysounds
--------

A set of modules that provide auditory stimulus generation, including hardware control.

It includes:

_sound.py_

A collection of routines that generate sound waveforms. Waveforms include clicks,
tone and noise pips, tone and noise trains, SAM tones and noise, dynamic ripple stimuli 
(though of limited capability), RSS stimuli, bandpass and notched noise. Test and display
routines are included.

_pystim.py_

    A module that interacts with some hardware (Rp2.1, PA5, RZ5D, nidaq, and soundcards) to play
generated sounds.

_pysounds.py_

    An early verion module that only knows about the RP2.1, PA5, and nidaq. This is in the
*archived* folder.

_calreader.py_

    Read and display calibration files generated in matlab ABR program.

_stimController_

*This module suite provides the following in conjunction with TDT hardware:*

    1. Generation of waveforms for acoustic stimulation, including tones, clicks, noises, bandpass and notched noise, modulated tones and noises, RSS, DMR, and comodulated masking stimuli (not all are currently implemented).

    2. The suite uses a graphical window to show the waveforms and long-term spectra, as well as an option to display the frequency-time spectra for the stimuli. 

    3. The suite provides for the presentation of stimuli, including control of their timing, repetitions, etc. 

    4. The suite interfaces with system soundcards (for testing), National Instruments DAC card(s)*, and Tucker-Davis Technologies RP2.1, RZ5D DSP processors and PA5 attenuators.

    5. For recording responses to acoustic stimulation, the suite expects to connect to an RZ5D with a specific circuit that is loaded. In this setup, the RZ5D (using appropriately configured OpenWorkbench) generates trigger pulses for the stimuli, and records multiple channels from the preamplifier. The current suite in turn controls the RZ5D, starting the acquisition, and following the storage of data in the data tank. The goal is that most of the experimental interaction with the user takes place with the current suite, rather than through the RZ5D/OpenWorkbench/OpenScope windows, although these need to be monitored. Communication with the RZ5D is limited to setting up the standard CoreSweepControl timing parameters, and initiating mode changes (Record, Preview, Standby, Idle). 


This sofware does not provide:
------------------------------

1. Complex stimulus control.

This software does provide:
---------------------------

1. data acquisition
2. data analysis. However, the analysis module is somewhat customized, and the
script will need to be modified (e.g., paths, datasets, etc) in order for it to be useful.





* Tested with NI6371 only.
