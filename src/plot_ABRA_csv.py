from pathlib import Path
import pandas as pd
import matplotlib.pyplot as mpl
import numpy as np
import seaborn as sns
from typing import Union
import pylibrary.plotting.plothelpers as PH
from matplotlib import rcParams

rcParams["text.latex.preamble"] = r"\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}"

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


def assign_cross(df, cross_key="NF107"):
    df["cross"] = {}

    def apply_cross(row):
        if "Filename" in row.keys():
            rf = row.Filename
        elif "File Name" in row.keys():
            rf = row["File Name"]
        if rf.find(cross_key) > 0:
            return cross_key
        else:
            return "C57Bl/6"

    df["cross"] = df.apply(apply_cross, axis=1)
    return df

def split_groups(split:bool=True, group_col="treatment", split_col="cross", cross_order=["C57Bl/6", "NF107"],
                 treat_order=["Sham", "NE2wks106", "NE2wks115"]):
    """
    Split the groups based on the treatment and cross columns.
    If split is True, use the split_col for hue, otherwise use group_col.
    """
    if split:
        hue = split_col
        hue_order = cross_order
        dodge = True
    else:
        hue = group_col
        hue_order = treat_order
        dodge = False
    return hue, hue_order, dodge

def plot_thresholds(
    filename: Union[str, Path], ax=None, palette=None, treat_order=None, split: bool = False,
    plottype: str = "bar",
):
    assert plottype in ["bar", "box"]
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    df = assign_cross(df, "NF107")
    hue, hue_order, dodge = split_groups(split=split, group_col="treatment", split_col="cross",
                                         cross_order=["C57Bl/6", "NF107"], treat_order=treat_order)

    if ax is None:
        f, ax = mpl.subplots(1, 1)
    match plottype:
        case "bar":
            kwds = {"errorbar": ("sd", 1), "saturation":0.45}
            fn = sns.barplot
        case "box": 
            kwds = {"saturation": 0.6}
            fn = sns.boxplot
        case _:
            raise ValueError(f"Unknown plot type: {plottype}")
    fn(
        x="treatment",
        y="Threshold",
        data=df,
        order=treat_order,
        hue_order=hue_order,
        hue=hue,
        palette=palette,
        ax=ax,
        **kwds,
    )
    sns.swarmplot(
        x="treatment",
        y="Threshold",
        data=df,
        order=treat_order,  # treat_order,
        palette=palette,
        hue=hue,
        hue_order=hue_order,
        # size=4,
        # markers = 'x',
        alpha=1,
        linewidth=0.25,
        edgecolor="black",
        dodge=dodge,
        ax=ax,
    )
    ax.legend(fontsize=7, loc="upper left", ncol=1, frameon=True)
    PH.nice_plot(ax, direction="outward")
    mpl.title("Thresholds by Treatment")
    mpl.xlabel("Treatment")
    mpl.ylabel("Threshold (dB SPL)")


def plot_amplitude_data(filename, ax=None, palette=None, treat_order=None, split: bool = False,
                        plottype: str = "bar"):
    assert plottype in ["bar", "box"]
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    df = assign_cross(df, "NF107")
    hue, hue_order, dodge = split_groups(split=split, group_col="treatment", split_col="cross",
                                         cross_order=["C57Bl/6", "NF107"], treat_order=treat_order)

    dfp = pd.DataFrame(columns=["subject", "cross", "treatment", "maxWave1"])
    for sub in df["File Name"].unique():
        subdf = df[df["File Name"] == sub]
        max_wave1 = subdf["Wave I amplitude (P1-T1) (μV)"].max()
        treatment = subdf["treatment"].iloc[0]
        dfp = dfp._append(
            {
                "subject": sub,
                "cross": subdf["cross"].iloc[0],
                "treatment": treatment,
                "maxWave1": max_wave1,
            },
            ignore_index=True,
        )
    if ax is None:
        f, ax = mpl.subplots(1, 1)
    match plottype:
        case "bar":
            kwds = {"errorbar": ("sd", 1), "saturation":0.45}
            fn = sns.barplot
        case "box": 
            kwds = {"saturation": 0.6}
            fn = sns.boxplot
        case _:
            raise ValueError(f"Unknown plot type: {plottype}")
    fn(
        x="treatment",
        y="maxWave1",
        data=dfp,
        palette=palette,
        ax=ax,
        order=treat_order,
        hue_order=hue_order,
        hue=hue,
        **kwds
    )
    sns.swarmplot(
        x="treatment",
        y="maxWave1",
        data=dfp,
        palette=palette,
        order=treat_order,
        hue_order=hue_order,
        hue=hue,
        alpha=0.6,
        linewidth=0.25,
        edgecolor="black",
        dodge=dodge,
        ax=ax,
    )
    ax.legend(fontsize=7, loc="upper right", ncol=1, frameon=True)
    PH.nice_plot(ax, direction="outward")
    mpl.title("Wave I Amplitude by Treatment")
    mpl.xlabel("Treatment")
    mpl.ylabel("Wave I Amplitude (μV)")
    # mpl.xticks(rotation=45)


def plot_IO_data(filename: Union[str, Path], ax=None, individual=False, palette=None, treat_order=None, split: bool = False, plottype: str = "bar"):
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    df = assign_cross(df, "NF107")
    if "File Name" not in df.columns:
        df.rename(columns={"Filename": "File Name"}, inplace=True)
    if "Wave I amplitude (P1-T1) (μV)" not in df.columns:
        df.rename(columns={"Wave I amplitude (P1-T1) (μV)": "Wave1_amplitude"}, inplace=True)
    df = assign_subject(df)
    # print(df.head())
    dfp = pd.DataFrame(
        columns=[
            "subject",
            "cross",
            "treatment",
            "Stimulus",
            "Frequency (Hz)",
            "Wave1_amplitude",
            "dB SPL",
        ]
    )
    subjects = list(df["File Name"].unique())
    freqs = df["Frequency (Hz)"].unique()
    freq_order = [float(x) for x in sorted(freqs)]
    treats = df["treatment"].unique()
    if len(freqs) == 1 and freqs == 100.0:
        mode = "click"
        colors = sns.color_palette(palette, len(treats))
        if ax is None and not individual:
            f, ax = mpl.subplots(1, 1)
            axn = [ax]
        elif ax is None and individual:
            r, c = PH.getLayoutDimensions(len(subjects))
            P = PH.regular_grid(r, c, order='rowsfirst', figsize=(17, 11), 
                                verticalspacing=0.05, horizontalspacing=0.03)
            # f, ax = mpl.subplots(r, c, figsize=(17, 11))
            ax = P.axarr
            axn = ax.ravel()
            for a in axn:
                PH.nice_plot(a)
        else:
            axn = [ax]
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
            {
                "subject": sub,
                "cross": subdf["cross"].iloc[0],
                "treatment": treatment,
                "Wave1_amplitude": wave1,
                "dB SPL": dbspl,
                "Frequency (Hz)": subdf["Frequency (Hz)"],
            },
            ignore_index=True,
        )

    tmap = [1, 2, 0]
    npl = 0
    for i, treat in enumerate(treats):
        data = df[df["treatment"] == treat]

        for ns, s in enumerate(data["subject"].unique()):
            subdata = data[data["subject"] == s]
            if s.find("NF107") > 0:
                marker = "s"
            else:
                marker = "o"
            if mode == "click":
                color = colors[tmap[i]]
                frqs = [100.0]
                if individual:
                    ax = axn[npl]
                    npl += 1
                else:
                    ax = ax
            elif mode == "tone":
                frqs = [float(f) for f in subdata["Frequency (Hz)"]]
                frx = int(freq_order.index(frqs[ns]))
                color = colors[frx]
                ax = axn[npl]
                npl += 1
            else:
                color = "k"
            if len(subdata) > 0:
                # reassemble data by frequency (split the dB SPL and Wave1_amplitude into separate arrays by frequency))
                db = []
                amp = []
                u_freq = pd.Series(frqs).unique()
                # pd.set_option('display.max_columns', None)
                for ifr, freq in enumerate(u_freq):
                    freq = int(freq)
                    this_fr = subdata[subdata["Frequency (Hz)"] == freq]
                    db = np.array(this_fr["dB Level"])
                    amp = np.array(this_fr["Wave I amplitude (P1-T1) (μV)"])
                    if mode == "click":
                        labl = None
                        lw = 0.5
                        alpha = 1
                    elif mode == "tone":
                        labl = f"{freq/1000} kHz"
                        lw = 1.5
                        color = colors[freq_order.index(freq)]
                        alpha = 0.5
                    else:
                        raise ValueError("Unknown mode: {}".format(mode))
                    ax.plot(
                        db,
                        amp,
                        marker=marker,
                        color=color,
                        label=labl,
                        linewidth=lw,
                        markersize=2,
                        alpha=alpha,
                    )
            if mode == "tone" or (mode == "click" and individual):
                sshort = '_'.join(s.split("_")[0 :5])
                ax.set_title(f"{sshort}\n{treat}", fontsize=6)
            ax.legend(fontsize=5, loc="upper left", ncol=1, frameon=False)
    if mode == "click":
        hue = "treatment"
        hue_order = treat_order
        for a in axn:
            a.set_ylim(-0.5, 8)

    elif mode == "tone":
        hue = "Frequency (Hz)"
        hue_order = freq_order
        for a in axn:
            a.set_ylim(-0.5, 4)
    sns.lineplot(
        x="dB Level",
        y="Wave I amplitude (P1-T1) (μV)",
        hue=hue,
        hue_order=hue_order,
        style="cross",
        style_order=["C57Bl/6", "NF107"],
        markers=["o", "s"],
        data=df,
        palette=palette,
        ax=ax,
    )
    mpl.xlabel("dB SPL")
    label = r"Wave I Amplitude (μV)"
    mpl.ylabel(label)
    mpl.xticks(rotation=45)
    ax.legend(fontsize=7, loc="upper left", ncol=1, frameon=True)
    PH.nice_plot(ax, direction="outward")


def plot_tone_thresholds(filename: Union[Path, str], ax=None, palette=None, treat_order=None):
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
    sns.lineplot(
        x="Frequency",
        y="Threshold",
        data=df,
        hue="treatment",
        hue_order=treat_order,
        palette=palette,
        ax=ax,
        errorbar=("sd", 1),
        linewidth=1.5,
    )
    for s in df["Filename"].unique():
        subdf = df[df["Filename"] == s]
        if len(subdf) > 0:
            ax.plot(
                subdf["Frequency"]
                - 0.2
                + np.random.uniform(0, subdf["Frequency"] * 0.1, len(subdf["Frequency"])),
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


# stim_type = "Tone"

stim_type = "Click"
treat_order = ["Sham", "NE2wks106", "NE2wks115"]

if stim_type == "Click":
    f, ax = mpl.subplots(1, 3, figsize=(9, 4))
    click_IO_file = Path('abra', "2025-07-21T19-53_export_click_IO.csv")
    click_threshold_file = Path('abra', "2025-07-21T19-54_export_click_thresholds.csv")
    split = True
    show_individual = True
    if show_individual:
        kwds = {"individual": True, "ax": None, "split": True}
    else:
        kwds = {"individual": False, "ax": ax[0], "split": False}
    plot_IO_data(filename=click_IO_file, palette=palette, treat_order=treat_order, **kwds)
    plot_amplitude_data(
        filename=click_IO_file, ax=ax[1], palette=palette, treat_order=treat_order, split=split, plottype="bar"
    )
    plot_thresholds(
        filename=Path(click_threshold_file),
        ax=ax[2],
        palette=palette,
        treat_order=treat_order,
        split=split,
        plottype="bar",
    )
    mpl.tight_layout()
    mpl.show()
if stim_type == "Tone":
    import os
    split = True  # split the data by cross 
    print("Current working directory:", os.getcwd())
    cwd = Path.cwd()
    print("abra path: ", Path(cwd, "abra", "Tones").is_dir())
    tonefile = Path("abra", "Tones", "2025-07-19T16-26_export_tones.csv")
    toneamplitudefile = Path("abra", "Tones", "2025-07-20T11-08_export_tone_amplitudes.csv")
    print(tonefile.is_file(), toneamplitudefile.is_file())
    plot_tone_thresholds(filename=tonefile, palette=palette, treat_order=treat_order)
    frpalette = sns.color_palette("nipy_spectral_r", 7)
    plot_IO_data(filename=toneamplitudefile, ax=None, palette=frpalette, treat_order=treat_order, split=split, plottype="bar")
    mpl.tight_layout()
    mpl.show()
