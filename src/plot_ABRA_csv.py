from pathlib import Path
import pandas as pd
import matplotlib.pyplot as mpl
import numpy as np
import seaborn as sns
from typing import Union
import pylibrary.plotting.plothelpers as PH

palette = sns.color_palette("tab10")
""" Takes output from the ABRA program (github.com/manorlab)
"""

def assign_treatment(df):
    df["treatment"] = {}

    def apply_treatment(row):
        if "Filename" in row.keys():
            rf = row.Filename
        elif "File Name" in row.keys():
            rf = row["File Name"]
        if "Sham" in rf:
            return "Sham"
        elif "NE2wks106" in rf:
            return "NE2wks106"
        elif "NE2wks115" in rf:
            return "NE2wks115"
        else:
            return "Unknown"

    df["treatment"] = df.apply(apply_treatment, axis=1)
    return df

def assign_subject(df):
    df["subject"] = {}

    def apply_subject(row):
        if "Filename" in row.keys():
            rf = row.Filename
        elif "File Name" in row.keys():
            rf = row["File Name"]
        return rf

    df["subject"] = df.apply(apply_subject, axis=1)
    return df

def plot_thresholds(filename: Union[str, Path], ax=None, palette=None, treat_order=None):
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    if ax is None:
        f, ax = mpl.subplots(1, 1)
    sns.boxplot(x="treatment", y="Threshold", data=df, order=treat_order, palette=palette, ax=ax, saturation=0.6)
    sns.swarmplot(x="treatment", y="Threshold", data=df,    hue="treatment",
        hue_order=treat_order,
        alpha=1.0,
        linewidth=0.25,
        edgecolor="black", ax=ax)
    mpl.title("Thresholds by Treatment")
    mpl.xlabel("Treatment")
    mpl.ylabel("Threshold (dB SPL)")
    mpl.xticks(rotation=45)


def plot_amplitude_data(filename, ax=None, palette=None, treat_order=None):
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    dfp = pd.DataFrame(columns=["subject", "treatment", "maxWave1"])
    for sub in df["File Name"].unique():
        subdf = df[df["File Name"] == sub]
        max_wave1 = subdf["Wave I amplitude (P1-T1) (μV)"].max()
        treatment = subdf["treatment"].iloc[0]
        dfp = dfp._append(
            {"subject": sub, "treatment": treatment, "maxWave1": max_wave1}, ignore_index=True
        )
    if ax is None:
        f, ax = mpl.subplots(1, 1)
    sns.boxplot(
        x="treatment",
        y="maxWave1",
        data=dfp,
        palette=palette,
        ax=ax,
        order=treat_order,
        saturation=0.6,
    )
    sns.swarmplot(
        x="treatment",
        y="maxWave1",
        data=dfp,
        palette=palette,
        hue="treatment",
        hue_order=treat_order,
        alpha=1.0,
        linewidth=0.25,
        edgecolor="black",
        ax=ax,
    )
    mpl.title("Wave I Amplitude by Treatment")
    mpl.xlabel("Treatment")
    mpl.ylabel("Wave I Amplitude (μV)")
    mpl.xticks(rotation=45)


def plot_IO_data(filename:Union[str, Path], ax: None, palette=None, treat_order=None):
    # Define the path to the CSV file
    fn = filename 
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    if "File Name" not in df.columns:
        df.rename(columns={"Filename": "File Name"}, inplace=True)
    if "Wave I amplitude (P1-T1) (μV)" not in df.columns:
        df.rename(columns={"Wave I amplitude (P1-T1) (μV)": "Wave1_amplitude"}, inplace=True)
    df = assign_subject(df)
    # print(df.head())
    dfp = pd.DataFrame(columns=["subject", "treatment", "Stimulus", "Frequency (Hz)", "Wave1_amplitude", "dB SPL"])
    subjects = list(df['File Name'].unique())
    freqs = df["Frequency (Hz)"].unique()
    freq_order = [float(x) for x in sorted(freqs)]
    treats = df["treatment"].unique()
    if len(freqs) == 1 and freqs == 100.:
        mode = "click"
        colors = sns.color_palette(palette, len(treats))
        if ax is None:
            f, ax = mpl.subplots(1, 1)
    else:
        mode = "tone"
        r, c = PH.getLayoutDimensions(len(subjects))
        fig, ax = mpl.subplots(r, c, figsize=(11, 8))
        axn = ax.ravel()
        for a in axn:
            PH.nice_plot(a)
        colors = sns.color_palette(palette, len(freq_order))
        print("freq_order: ", freq_order)
    for sub in df["File Name"].unique():
        subdf = df[df["File Name"] == sub]
        wave1 = np.array(subdf["Wave I amplitude (P1-T1) (μV)"].values)
        dbspl = np.array(subdf["dB Level"].values)
        if sub.find("WJ9") > 0:
            dbspl = dbspl[:15]
            wave1 = wave1[:15]
            # print("dbspl: ", sub, dbspl)
            # print("wave1: ", sub, wave1)

        treatment = subdf["treatment"].iloc[0]
        dfp = dfp._append(
            {"subject": sub, "treatment": treatment, "Wave1_amplitude": wave1, "dB SPL": dbspl, "Frequency (Hz)": subdf["Frequency (Hz)"]},
            ignore_index=True,
        )

    tmap = [1, 2, 0]
    npl = 0
    for i, treat in enumerate(treats):
        data = df[df["treatment"] == treat]

        for ns, s in enumerate(data["subject"].unique()):
            subdata = data[data["subject"] == s]

            if mode == "click":
                color = colors[tmap[i]]
            elif mode == 'tone':
                frqs = [float(f) for f in subdata["Frequency (Hz)"]]
                frx = int(freq_order.index(frqs[ns]))
                color = colors[frx]
                ax = axn[npl]
                npl += 1
            else:
                color = 'k'
            if len(subdata) > 0:
                # reassemble data by frequency (split the dB SPL and Wave1_amplitude into separate arrays by frequency))
                db = []
                amp = []
                u_freq = pd.Series(frqs).unique()
                # pd.set_option('display.max_columns', None)
                for ifr, freq in enumerate(u_freq):
                    freq = int(freq)
                    this_fr = subdata[subdata['Frequency (Hz)'] == freq]
                    db = np.array(this_fr["dB Level"])
                    amp = np.array(this_fr[u"Wave I amplitude (P1-T1) (μV)"])
                    color = colors[freq_order.index(freq)]
                    ax.plot(
                        db,
                        amp,
                        marker="o",
                        color=color,
                        label=f"{freq/1000} kHz",
                        linewidth=1.5,
                        markersize=2,
                        alpha=0.3,
                    )
                ax.set_title(f"{s}\n{treat}", fontsize=6)
                ax.legend(fontsize=6, loc="upper left", ncol=1, frameon=False)    
    if mode == "click":
        hue = "treatment"
        hue_order = treat_order
    elif mode == "tone":
        hue = "Frequency (Hz)"
        hue_order = freq_order
        for a in axn:
            a.set_ylim(-0.5, 4)
    # sns.lineplot(
    #     x="dB Level",
    #     y="Wave I amplitude (P1-T1) (μV)",
    #     hue=hue,
    #     hue_order=hue_order,
    #     data=df,
    #     palette=palette,
    #     ax=axn,
    # )
    mpl.xlabel("dB SPL")
    label = r"Wave I Amplitude (${\mu}$V)"
    mpl.ylabel(label)
    mpl.xticks(rotation=45)


def plot_tone_thresholds(filename:Union[Path, str],ax=None, palette=None, treat_order=None):
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    if ax is None:
        f, ax = mpl.subplots(1, 1, figsize=(6, 6))

    print(df.columns)
    # df['Frequency'] = np.log10(df['Frequency'])
    # sns.swarmplot(x="Frequency", y="Threshold", data=df,    hue="treatment",
    #     hue_order=treat_order,
    #     alpha=1.0,
    #     linewidth=0.25,
    #     edgecolor="black", ax=ax)
    # sns.boxplot(x="Frequency", y="Threshold", data=df, hue="treatment", hue_order=treat_order, palette=palette, ax=ax, saturation=0.6)
    sns.lineplot(x="Frequency", y="Threshold", data=df, hue="treatment", hue_order=treat_order, palette=palette, ax=ax,
                 errorbar=("sd", 1), linewidth=1.5)
    for s in df["Filename"].unique():
        subdf = df[df["Filename"] == s]
        if len(subdf) > 0:
            ax.plot(
                subdf["Frequency"] - 0.2 + np.random.uniform(0, subdf["Frequency"]*0.1, len(subdf["Frequency"])),
                subdf["Threshold"],
                marker="o",
                color=palette[treat_order.index(subdf["treatment"].values[0])],
                linewidth=0,
                markersize=2,
                alpha=0.6,
            )

            # sns.stripplot(
            #     x="Frequency",
            #     y="Threshold",
            #     data=subdf,
            #     hue="treatment",
            #     hue_order=treat_order,
            #     palette=palette,
            #     alpha=0.6,
            #     linewidth=0.25,
            #     edgecolor="black",
            #     ax=ax,
            #     native_scale=True,
            #     log_scale= True,
            #     jitter=20,
            #     dodge=True,
            # )
    ax.set_xscale("log")

    mpl.title("Thresholds by Frequency")
    mpl.xlabel("Frequency")
    mpl.ylabel("Threshold (dB SPL)")
    mpl.xticks(rotation=45)

stim_type = "Tone"

# stim_type = "Click"
treat_order = ["Sham", "NE2wks106", "NE2wks115"]

if stim_type == "Click":
    f, ax = mpl.subplots(1, 3, figsize=(9, 4))
    clickfile = Path("2025-07-18T18-36_export.csv")
    plot_IO_data(filename=clickfile, ax=ax[0], palette=palette, treat_order=treat_order)
    plot_amplitude_data(filename=clickfile, ax=ax[1], palette=palette, treat_order=treat_order)
    plot_thresholds(filename=Path("2025-07-18T18-37_export_thresholds.csv"), ax=ax[2], palette=palette, treat_order=treat_order)
    mpl.tight_layout()
    mpl.show()
if stim_type == "Tone":
    import os
    print("Current working directory:", os.getcwd())
    cwd = Path.cwd()
    print("abra path: ", Path(cwd, "abra", "Tones").is_dir())
    tonefile = Path("abra", "Tones", "2025-07-19T16-26_export_tones.csv")
    toneamplitudefile = Path("abra", "Tones", "2025-07-20T11-08_export_tone_amplitudes.csv")
    print(tonefile.is_file(), toneamplitudefile.is_file())
    plot_tone_thresholds(filename=tonefile, palette=palette, treat_order=treat_order)
    frpalette = sns.color_palette("nipy_spectral_r", 7)
    plot_IO_data(filename=toneamplitudefile, ax=None, palette=frpalette, treat_order=treat_order)
    mpl.tight_layout()
    mpl.show()