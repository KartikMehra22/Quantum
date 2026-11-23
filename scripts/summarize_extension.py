"""Generate four-method figures and a numerical note from saved results."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from measurement_extension.analysis import summarize

COLORS={"OLS":"#777777","Ridge":"#24547a","DiagonalCov":"#d16128","FullCov":"#168275"}


def main(directory,figures,note):
    directory,figures,note=Path(directory),Path(figures),Path(note)
    config=json.loads((directory/"config.json").read_text());data=summarize(directory)
    figures.mkdir(parents=True,exist_ok=True);note.parent.mkdir(parents=True,exist_ok=True)
    tasks=config["tasks"];shots=config["shots_per_setting"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.2,"pdf.fonttype":42})
    def save(fig,name):
        fig.savefig(figures/(name+".png"),dpi=190,bbox_inches="tight")
        fig.savefig(figures/(name+".pdf"),bbox_inches="tight");plt.close(fig)
    for noise in config["label_noise_stds"]:
        fig,axes=plt.subplots(1,len(tasks),figsize=(4.8*len(tasks),3.8),squeeze=False,constrained_layout=True)
        for ax,task in zip(axes.ravel(),tasks):
            for method in config["methods"]:
                rows=sorted([r for r in data["summary"] if (r["task"],r["label_noise_std"],r["method"])==(task,noise,method) and r["shots"]>0],key=lambda r:r["shots"])
                ax.errorbar([r["shots"] for r in rows],[r["test_mse_mean"] for r in rows],yerr=[r["test_mse_sd"] for r in rows],label=method,color=COLORS[method],marker="o",capsize=3)
            ax.set_xscale("log",base=2);ax.set_xticks(shots,[str(s) for s in shots])
            ax.set_xlabel("Shots per setting (3 settings)");ax.set_ylabel("Test MSE against latent target")
            ax.set_title(task.replace("_"," "));ax.legend(fontsize=8)
        save(fig,"test_mse_clean" if noise==0 else f"test_mse_noise_{noise:g}")
    fig,axes=plt.subplots(1,len(tasks),figsize=(4.8*len(tasks),3.8),squeeze=False,constrained_layout=True)
    for ax,task in zip(axes.ravel(),tasks):
        for offset,reference,candidate in [(-0.16,"Ridge","FullCov"),(0.,"Ridge","DiagonalCov"),(0.16,"DiagonalCov","FullCov")]:
            rows=sorted([r for r in data["contrasts"] if (r["task"],r["label_noise_std"],r["reference"],r["candidate"])==(task,0,reference,candidate) and r["shots"]>0],key=lambda r:r["shots"])
            x=np.arange(len(rows))+offset
            ax.errorbar(x,[r["reference_minus_candidate_mse"] for r in rows],yerr=[r["paired_difference_sd"] for r in rows],fmt="o",capsize=3,label=f"{reference} - {candidate}")
        ax.axhline(0,color="black",lw=1);ax.set_xticks(np.arange(len(shots)),[str(s) for s in shots])
        ax.set_xlabel("Shots per setting");ax.set_ylabel("Paired MSE difference (positive favours candidate)")
        ax.set_title(task.replace("_"," "));ax.legend(fontsize=7)
    save(fig,"paired_differences_clean")
    fig,axes=plt.subplots(1,len(tasks),figsize=(4.5*len(tasks),2.7),squeeze=False,constrained_layout=True)
    for ax,task in zip(axes.ravel(),tasks):
        matrix=np.array([[1-next(r["gamma_zero_fraction"] for r in data["summary"] if (r["task"],r["label_noise_std"],r["shots"],r["method"])==(task,0,s,m)) for s in shots] for m in ["DiagonalCov","FullCov"]])
        image=ax.imshow(matrix,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        for i in range(2):
            for j in range(len(shots)):ax.text(j,i,f"{matrix[i,j]:.0%}",ha="center",va="center",color="white" if matrix[i,j]>0.6 else "black")
        ax.set_xticks(range(len(shots)),[str(s) for s in shots]);ax.set_yticks([0,1],["DiagonalCov","FullCov"])
        ax.set_xlabel("Shots per setting");ax.set_title(task.replace("_"," "))
        fig.colorbar(image,ax=ax,label="Fraction selecting gamma > 0")
    save(fig,"nonzero_covariance_selection")
    lines=["# Measurement-aware extension: measured results","","Generated from preserved fresh-seed results. This is a technical results note, not a paper draft.","",
           f"{len(config['data_seeds'])} independent dataset realizations; {len(config['measurement_seeds'])} acquisitions averaged within each dataset. Means and sample SDs are descriptive, not confidence intervals.","",
           "Ridge, DiagonalCov and FullCov each search 32 validation candidates. Ridge includes every gamma=0 option available to the new methods. DiagonalCov and FullCov share identical grids. No method uses test targets to select its penalty."]
    for noise in config["label_noise_stds"]:
        lines += ["","## "+("Primary clean-label results" if noise==0 else f"Separate label-noise sensitivity: SD {noise:g}"),"",
                  "| Task | Shots/setting | OLS MSE | Ridge MSE | DiagonalCov MSE | FullCov MSE | FullCov reduction vs Ridge |","|---|---:|---:|---:|---:|---:|---:|"]
        for task in tasks:
            for s in [*shots,0]:
                entries={r["method"]:r for r in data["summary"] if (r["task"],r["label_noise_std"],r["shots"])==(task,noise,s)}
                contrast=next(r for r in data["contrasts"] if (r["task"],r["label_noise_std"],r["shots"],r["reference"],r["candidate"])==(task,noise,s,"Ridge","FullCov"))
                values=" | ".join(f"{entries[m]['test_mse_mean']:.6f}" for m in config["methods"])
                lines.append(f"| {task} | {'Exact' if s==0 else s} | {values} | {contrast['relative_mse_reduction_percent']:+.2f}% |")
    lines += ["","## Interpretation and limitations","",
              "Positive reduction means FullCov has a lower mean MSE; negative means it is worse. Small differences and descriptive SDs do not establish significance. Exact references are reported separately; with zero measurement covariance the new methods reduce to Ridge on their coarser lambda grid.","",
              "Nonzero-gamma selection is shown separately: gamma=0 choices are not evidence of a covariance benefit. Comparisons use the same data and shot records. Equal candidate counts do not make model families equally complex.","",
              "The covariance penalty is an adaptation of an existing approach. It is not a newly invented method, an unbiased risk correction, proof of quantum advantage, or a reproduction of a different paper's forecasting results."]
    note.write_text("\n".join(lines)+"\n")
    print(f"Saved extension summaries, {len(config['label_noise_stds'])+2} figures and {note}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"results/measurement_extension")
    p.add_argument("--figures",type=Path,default=ROOT/"figures/measurement_extension")
    p.add_argument("--note",type=Path,default=ROOT/"docs/EXTENSION_RESULTS.md")
    a=p.parse_args();main(a.results,a.figures,a.note)
