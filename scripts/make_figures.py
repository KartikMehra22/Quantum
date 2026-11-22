"""Generate all submission figures from recorded results, without rerunning fits."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from qml_project.analysis import summarise

COLORS = {"OLS": "#24547a", "Ridge": "#d16128"}


def save(fig, directory, name):
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(directory / (name + ".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def main(result_dir, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    result = summarise(result_dir)
    rows = result["summary"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "pdf.fonttype": 42})
    tasks = ["sine", "mixed_harmonics"]
    for noise, name in [(0.0, "test_mse_clean"), (0.1, "test_mse_noisy")]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        for ax, task in zip(axes, tasks):
            for method in ["OLS", "Ridge"]:
                subset = sorted([r for r in rows if r["task"] == task and r["label_noise_std"] == noise and r["shots"] > 0 and r["method"] == method], key=lambda r:r["shots"])
                ax.errorbar([r["shots"] for r in subset], [r["test_mse_mean"] for r in subset],
                            yerr=[r["test_mse_sd"] for r in subset], color=COLORS[method], marker="o", capsize=4, label=method)
                exact = next(r for r in rows if (r["task"], r["label_noise_std"], r["shots"], r["method"]) == (task, noise, 0, method))
                ax.axhline(exact["test_mse_mean"], color=COLORS[method], linestyle="--", alpha=0.7, label=f"{method}, exact features")
            ax.set_xscale("log", base=2)
            ax.set_xticks([32, 128, 512], ["32", "128", "512"])
            ax.set_xlabel("Shots per setting (3 settings per input)")
            ax.set_ylabel("Test MSE against latent target")
            ax.set_title(task.replace("_", " ").title())
            ax.legend(fontsize=7)
        save(fig, figure_dir, name)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for ax, task in zip(axes, tasks):
        for method in ["OLS", "Ridge"]:
            subset = sorted([r for r in rows if r["task"] == task and r["label_noise_std"] == 0 and r["shots"] > 0 and r["method"] == method], key=lambda r:r["shots"])
            ax.errorbar([r["shots"] for r in subset], [r["coefficient_norm_mean"] for r in subset],
                        yerr=[r["coefficient_norm_sd"] for r in subset], label=method,
                        color=COLORS[method], marker="o", capsize=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32,128,512], ["32","128","512"])
        ax.set_xlabel("Shots per measurement setting")
        ax.set_ylabel("Coefficient norm in standardized coordinates")
        ax.set_title(task.replace("_", " ").title())
        ax.legend()
    save(fig, figure_dir, "coefficient_norms")
    # Feature precision depends only on acquisition, not the downstream target.
    subset = sorted([r for r in rows if (r["task"], r["label_noise_std"], r["method"]) == ("sine", 0, "OLS") and r["shots"] > 0], key=lambda r:r["shots"])
    shots = np.array([r["shots"] for r in subset])
    observed = np.array([r["feature_mse_mean"] for r in subset])
    ideals = [np.load(result_dir / "raw" / f"inputs_d{seed}.npz")["ideal_features"] for seed in [11,22,33,44,55]]
    theoretical = float(np.mean([np.mean(1 - x*x) for x in ideals])) / shots
    fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
    ax.errorbar(shots, observed, yerr=[r["feature_mse_sd"] for r in subset], marker="o", capsize=4, label="Observed feature MSE")
    ax.plot(shots, theoretical, "--", label="Born-rule variance prediction")
    ax.set_xscale("log",base=2);ax.set_yscale("log")
    ax.set_xticks(shots,[str(s) for s in shots])
    ax.set_xlabel("Shots per measurement setting");ax.set_ylabel("Mean squared feature-estimation error")
    ax.legend()
    save(fig, figure_dir, "feature_precision")
    # Example fixed in advance: first listed data and measurement seed, both tasks.
    predictions = json.loads((result_dir / "predictions.json").read_text())
    inputs = np.load(result_dir / "raw/inputs_d11.npz")["x"][96:]
    order = np.argsort(inputs)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True)
    for i, task in enumerate(tasks):
        target = np.load(result_dir / "raw" / f"labels_{task}_noise0_d11.npz")["latent"][96:]
        for j, shots in enumerate([32, 128, 512]):
            ax = axes[i,j]
            ax.plot(inputs[order], target[order], color="black", label="Target", linewidth=1.4)
            for method in ["OLS", "Ridge"]:
                r = next(r for r in predictions if (r["task"],r["label_noise_std"],r["data_seed"],r["measurement_seed"],r["shots"],r["method"]) == (task,0,11,101,shots,method))
                ax.scatter(inputs, r["prediction"], s=7, alpha=0.55, color=COLORS[method], label=method)
            ax.set_title(f"{task.replace('_',' ')} | S={shots}",fontsize=9)
            ax.set_xlabel("Input x");ax.set_ylabel("Prediction")
            if i == 0 and j == 0: ax.legend(fontsize=7)
    save(fig, figure_dir, "example_predictions")
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.set_axis_off()
    nodes = ["Generate inputs\nand targets", "Fixed two-qubit\nquantum circuit", "XX / YY / ZZ\njoint sampling", "Training-only scaling\nOLS / tuned Ridge", "Held-out error\nand repeated runs"]
    for i, label in enumerate(nodes):
        x = 0.01 + i * 0.2
        ax.add_patch(FancyBboxPatch((x,0.3),0.17,0.4,boxstyle="round,pad=0.01",facecolor="#edf4f7",edgecolor="#24547a",transform=ax.transAxes))
        ax.text(x+0.085,0.5,label,ha="center",va="center",fontsize=8,transform=ax.transAxes)
        if i<4: ax.annotate("",(x+0.198,0.5),(x+0.172,0.5),xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
    save(fig, figure_dir, "workflow")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_axis_off(); ax.set_xlim(-0.7, 10.6); ax.set_ylim(-0.6, 1.6)
    for y, label in [(1, r"$q_0: |0\rangle$"), (0, r"$q_1: |0\rangle$")]:
        ax.plot([-0.1,10.2],[y,y],color="#243746",lw=1)
        ax.text(-0.25,y,label,ha="right",va="center",fontsize=10)
    gates = [(0.5,1,r"$R_y(x)$"),(1.9,1,r"$R_z(0.4x)$"),
             (0.5,0,r"$R_y(2x+0.3)$"),(1.9,0,r"$R_z(-0.7x)$"),
             (4.5,1,r"$R_y(0.7)$"),(4.5,0,r"$R_y(-0.4)$"),
             (6.0,0,r"$R_z(0.2)$"),(9.1,1,"Basis +\nmeasure"),(9.1,0,"Basis +\nmeasure")]
    for x,y,label in gates:
        width=1.35 if x==9.1 else 1.25
        ax.add_patch(FancyBboxPatch((x-width/2,y-0.2),width,0.4,boxstyle="round,pad=0.02",facecolor="white",edgecolor="#24547a",zorder=3))
        ax.text(x,y,label,ha="center",va="center",fontsize=9,zorder=4)
    ax.plot([3.25,3.25],[0,1],color="#243746");ax.plot(3.25,1,"o",color="#243746")
    ax.text(3.25,0,r"$\oplus$",ha="center",va="center",fontsize=22)
    ax.text(3.25,1.32,"CNOT",ha="center",fontsize=8)
    ax.plot([7.3,7.3],[0,1],color="#243746");ax.plot([7.3,7.3],[0,1],"o",color="#243746")
    ax.text(7.3,1.32,"CZ",ha="center",fontsize=8)
    ax.text(5,-0.5,"Independent XX, YY and ZZ settings; each yields two singles and one pair feature.",ha="center",fontsize=9)
    save(fig,figure_dir,"quantum_circuit")
    print(f"Generated seven figure pairs and summaries in {figure_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT / "results/semester")
    parser.add_argument("--figures",type=Path,default=ROOT / "figures")
    args = parser.parse_args()
    main(args.results,args.figures)

