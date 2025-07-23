from pathlib import Path
import pandas as pd
import matplotlib.pyplot as mpl
import numpy as np
import seaborn as sns
from typing import Union
import pylibrary.plotting.plothelpers as PH
from matplotlib import rcParams
from sklearn.cluster import KMeans
from kneed import KneeLocator

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

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


def split_groups(
    split: bool = True,
    group_col="treatment",
    split_col="cross",
    cross_order=["C57Bl/6", "NF107"],
    treat_order=["Sham", "NE2wks106", "NE2wks115"],
):
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
    filename: Union[str, Path],
    ax=None,
    palette=None,
    treat_order=None,
    split: bool = False,
    split_col="cross",
    cross_order=["C57Bl/6", "NF107"],
    plottype: str = "bar",
    **kwargs,
):
    assert plottype in ["bar", "box"]

    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    if split_col == "cross":
        df = assign_cross(df, "NF107")
    hue, hue_order, dodge = split_groups(
        split=split,
        group_col="treatment",
        split_col=split_col,
        cross_order=cross_order,
        treat_order=treat_order,
    )

    if ax is None:
        f, ax = mpl.subplots(1, 1)
    match plottype:
        case "bar":
            kwds = {"errorbar": ("sd", 1), "saturation": 0.45}
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


def plot_amplitude_data(
    filename,
    ax=None,
    palette=None,
    treat_order=None,
    split: bool = False,
    plottype: str = "bar",
    split_col="cross",
    cross_order=["C57Bl/6", "NF107"],
):
    assert plottype in ["bar", "box"]
    # Define the path to the CSV file
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    if split_col == "cross":
        df = assign_cross(df, "NF107")
    hue, hue_order, dodge = split_groups(
        split=split,
        group_col="treatment",
        split_col=split_col,
        cross_order=cross_order,
        treat_order=treat_order,
    )

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
            kwds = {"errorbar": ("sd", 1), "saturation": 0.45}
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
        **kwds,
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


def plot_thr_amp_data(
    filename: Union[str, Path], ax=None, palette=None, treat_order=None, split: bool = False
):
    fn = filename
    df = pd.read_csv(fn)
    df = assign_treatment(df)
    df = assign_cross(df, "NF107")
    hue, hue_order, dodge = split_groups(
        split=split,
        group_col="treatment",
        split_col="cross",
        cross_order=["C57Bl/6", "NF107"],
        treat_order=treat_order,
    )
    print("hue: ", hue)
    print(df.columns)
    dfp = pd.DataFrame(columns=["subject", "cross", "treatment", "maxWave1", "threshold"])
    for sub in df["File Name"].unique():
        subdf = df[df["File Name"] == sub]
        max_wave1 = subdf["Wave I amplitude (P1-T1) (μV)"].max()
        treatment = subdf["treatment"].iloc[0]
        threshold = subdf["Estimated Threshold"].iloc[0]
        dfp = dfp._append(
            {
                "subject": sub,
                "cross": subdf["cross"].iloc[0],
                "treatment": treatment,
                "maxWave1": max_wave1,
                "threshold": threshold,
            },
            ignore_index=True,
        )
    if ax is None:
        f, ax = mpl.subplots(1, 1)

    # principalDf = compute_pca(ax, dfp)
    # sns.scatterplot(data=principalDf, x="PC1", y="PC2", alpha=0.5, palette=palette, hue=hue, hue_order=hue_order, ax=ax)
    # sns.scatterplot(data=principalDf, x='x_scaled', y='y_scaled', alpha=0.5, style=hue,
    #                 markers=['s', 'h', 'D'], hue=hue, hue_order=hue_order, edgecolor="black", linewidth=0.3, ax=ax)
    sns.scatterplot(
        data=dfp,
        x="threshold",
        y="maxWave1",
        style="cross",
        markers=["o", "s"],
        size="cross",
        sizes=[28, 32],
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
    )
    compute_kmeans_clusters(ax, dfp)

    PH.nice_plot(ax, direction="outward")
    mpl.title("PCA of Thresholds and Wave I Amplitude")
    mpl.xlabel("Threshold (dB SPL)")
    mpl.ylabel("Wave I Amplitude (μV)")
    ax.set_ylim(0, 8)
    ax.set_xlim(0, 100)


def compute_pca(ax, dfp):
    # Perform PCA
    features = ["threshold", "maxWave1"]
    print(dfp["treatment"].unique())

    def map_treat(row):
        if row["treatment"] == "Sham":
            return 0
        elif row["treatment"] == "NE2wks106":
            return 1
        elif row["treatment"] == "NE2wks115":
            return 2
        else:
            return 3

    dfp["treatment"] = dfp.apply(map_treat, axis=1)
    x = dfp[features].values
    x_scaled = StandardScaler().fit_transform(x)  # Normalize the data

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_scaled)
    principalDf = pd.DataFrame(data=principalComponents, columns=["PC1", "PC2"])
    principalDf["treatment"] = dfp["treatment"]
    principalDf["cross"] = dfp["cross"]
    principalDf["subject"] = dfp["subject"]
    principalDf["x_scaled"] = x_scaled[:, 0]
    principalDf["y_scaled"] = x_scaled[:, 1]
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(fontsize=7, loc="upper right", ncol=1, frameon=True)
    return principalDf


def compute_kmeans_clusters(ax, dfp):
    # kmeans clustering ?
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }
    X = dfp[["threshold", "maxWave1"]].values

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    f2, ax2 = mpl.subplots(1, 1)
    mpl.plot(range(1, 11), sse, marker="o")
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    print("Knee point: ", kl.elbow)

    km = KMeans(n_clusters=3, **kmeans_kwargs)
    km_result = km.fit(X)
    print(km_result.labels_)
    print(km_result.cluster_centers_)
    ax.plot(
        km_result.cluster_centers_[:, 0],
        km_result.cluster_centers_[:, 1],
        marker="x",
        color="red",
        linestyle="",
        markersize=8,
        label="Centroids",
    )
    for i in range(len(X)):
        ax.text(
            X[i, 0] - 1,
            X[i, 1],
            str(km_result.labels_[i]),
            fontsize=9,
            color="black",
            ha="right",
            va="center",
        )


def plot_IO_data(
    filename: Union[str, Path],
    ax=None,
    palette=None,
    treat_order=None,
    individual=False,
    split: bool = False,
    **kwargs,
    # split_col:Union[str, None]="cross",
    # cross_order:Union[list, None]=None,
    # plottype: str = "bar",
):
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
            P = PH.regular_grid(
                r,
                c,
                order="rowsfirst",
                figsize=(17, 11),
                verticalspacing=0.05,
                horizontalspacing=0.03,
            )
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

    tmap = [0, 1, 2, 3]
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
                sshort = "_".join(s.split("_")[0:5])
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
    if not split:
        kwds = {"style": None, "style_order": None, "markers": ["o"]}
        sns.lineplot(
            x="dB Level",
            y="Wave I amplitude (P1-T1) (μV)",
            hue=hue,
            hue_order=hue_order,
            data=df,
            palette=palette,
            ax=ax,
            **kwds,
        )
    else:
        kwds = {"style": "cross", "style_order": ["C57Bl/6", "NF107"], "markers": ["o", "s"]}

    ax.legend(fontsize=7, loc="upper left", ncol=1, frameon=True)
    mpl.xlabel("dB SPL")
    label = r"Wave I Amplitude (μV)"
    mpl.ylabel(label)
    mpl.xticks(rotation=45)

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



if __name__ == "__main__":
    # Example usage of the plotting functions
    # This is a script to plot ABR data from CSV files exported from ABRA.
    # It can plot click and tone data, and can split the data by treatment and cross.
    # The data is expected to be in a specific format, with columns for treatment, subject, cross, etc.
    # The script uses seaborn and matplotlib for plotting.

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10

    stim_type = "Click"
    stim_type = "Tone"
    treat_order = ["Sham", "NE2wks106", "NE2wks115"]
    strain = "GlyT2"
    # strain = "VGATEYFP"

    if stim_type == "Click":
        f, ax = mpl.subplots(1, 3, figsize=(9, 4))
        f.suptitle(f"{strain} ABR Click Data", fontsize=16)
        split_col = None
        cross = None
        if strain == "GlyT2":
            click_IO_file = Path("abra", "2025-07-21T19-53_export_click_IO.csv")
            click_threshold_file = Path("abra", "2025-07-21T19-54_export_click_thresholds.csv")
            cross = ["C57Bl/6", "NF107"]
            split_col = "cross"
        elif strain == "VGATEYFP":
            click_IO_file = Path("abra", "VGATEYFP_2025-07-23T17-50_export_click_IO.csv")
            click_threshold_file = Path("abra", "VGATEYFP_2025-07-23T17-49_export_click_thresholds.csv")
            cross = None
            split_col = None
        split = True
        show_individual = True
        if show_individual:
            kwds = {
                "individual": True,
                "ax": None,
                "split": True,
                "split_col": split_col,
                "cross_order": cross,
            }
            plot_IO_data(filename=click_IO_file, palette=palette, treat_order=treat_order, **kwds)
        else:
            kwds = {
                "individual": False,
                "ax": ax[0],
                "split": False,
                "split_col": split_col,
                "cross_order": cross,
            }
        kwds2 = {
            "individual": False,
            "ax": ax[0],
            "split": False,
            "split_col": split_col,
            "cross_order": cross,
        }
        # now the 3 panel plot
        kwds3 = kwds2.copy()
        kwds3["ax"] = ax[2]
        kwds3.pop("individual", None)
        plot_IO_data(filename=click_IO_file, palette=palette, treat_order=treat_order, **kwds2)

        plot_amplitude_data(
            filename=click_IO_file, ax=ax[1], palette=palette, treat_order=treat_order, plottype="bar"
        )
        plot_thresholds(
            filename=Path(click_threshold_file),
            palette=palette,
            treat_order=treat_order,
            **kwds3,
        )
        # mpl.tight_layout()
        # plot_thr_amp_data(filename=click_IO_file, ax=None, palette=palette, treat_order=treat_order,
        #                   split= False)
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
        plot_IO_data(
            filename=toneamplitudefile,
            ax=None,
            palette=frpalette,
            treat_order=treat_order,
            split=split,
            plottype="bar",
        )
        mpl.tight_layout()
        mpl.show()
