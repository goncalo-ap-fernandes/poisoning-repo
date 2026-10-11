"""Plot the three N4SID MaxAE panels from existing metrics; no fitting is run."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "N4SID_Results/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    expected = {("Normal", 0, 0.0)} | {
        (method, tau, epsilon) for method in ("Calculated", "Gaussian", "PRBS")
        for tau in (25, 50, 100) for epsilon in (0.05, 0.10, 0.15, 0.20)}
    actual = {(row["method"], int(row["tau"]), float(row["epsilon"])) for row in rows}
    if len(rows) != 37 or actual != expected:
        raise ValueError("Expected all 37 N4SID conditions. Run N4SID() to completion.")
    figures = ROOT / "Figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({key: 24 for key in ("font.size", "axes.titlesize",
        "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    clean = next(row for row in rows if row["method"] == "Normal")
    for tau in (25, 50, 100):
        # Match the manuscript panels' plot area, typography and line width.
        figure = plt.figure(figsize=(7.48, 5.48))
        axis = figure.add_axes([132 / 748, 92 / 548, 606 / 748, 433 / 548])
        condition = [row for row in rows if int(row["tau"]) == tau]
        for method, color, label in (("Calculated", "blue", "Designed"),
                ("Gaussian", "red", "Gaussian"), ("PRBS", "green", "PRBS")):
            group = sorted((row for row in condition if row["method"] == method
                and row["included"] == "1"), key=lambda row: float(row["epsilon"]))
            x = [float(row["profile_rms"]) for row in group]
            y = [float(row["transition_maxae"]) for row in group]
            axis.scatter(x, y, color=color, alpha=0.5)
            axis.plot(x, y, color=color, linewidth=3, label=label)
        rms = [float(row["profile_rms"]) for row in condition]
        margin = 0.05 * (max(rms) - min(rms))
        axis.set_xlim(min(rms) - margin, max(rms) + margin)
        axis.set_ylim(0, 0.30)
        axis.set_xlabel("Perturbation RMS")
        axis.set_ylabel("Maximum Absolute Error")
        axis.axhline(float(clean["transition_maxae"]), color="black",
                     linestyle=":", linewidth=1.5)
        axis.grid(True)
        axis.legend(loc="upper left")
        path = figures / f"n4sid_top{tau}MaxError.pdf"
        figure.savefig(path)
        plt.close(figure)
        print(path)


if __name__ == "__main__":
    main()
