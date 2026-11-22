"""Write numerical implementation notes from the recorded results; no paper is generated."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


if __name__=="__main__":
    data=json.loads((ROOT/"results/semester/summary.json").read_text())
    meta=json.loads((ROOT/"results/semester/metadata.json").read_text())
    lines=["# Measured results", "", "Generated directly from recorded result JSON. This is a numerical note, not a paper draft.", "",
           "Test MSE is against the noiseless target. Values are averaged within dataset first, then across five dataset means. Sample SD bars are descriptive, not confidence intervals."]
    for noise,title in [(0,"Primary clean-label study"),(0.1,"Separate noisy-label sensitivity check")]:
        lines += ["", "## "+title,"", "| Task | Shots per setting | OLS MSE | Ridge MSE | Ridge relative MSE reduction |", "|---|---:|---:|---:|---:|"]
        for r in data["paired_comparisons"]:
            if r["label_noise_std"]!=noise:continue
            e={x["method"]:x for x in data["summary"] if (x["task"],x["label_noise_std"],x["shots"])==(r["task"],noise,r["shots"])}
            lines.append(f"| {r['task']} | {'Exact' if r['shots']==0 else r['shots']} | {e['OLS']['test_mse_mean']:.6f} | {e['Ridge']['test_mse_mean']:.6f} | {r['relative_reduction_percent']:+.2f}% |")
    lines += ["", "Positive relative reduction means lower MSE for Ridge; negative means Ridge was worse. Percentages near a small exact-reference baseline can look large even when the absolute difference is small.", "",
              "## Validation and accounting", "", f"- {meta['metrics_rows']} fitted models: 600 finite-shot and 40 exact references, not 640 independent datasets.",
              f"- {meta['raw_measurement_files']} unique acquisitions; {meta['unique_finite_shots']:,} simulated executions shared across both models, targets, label conditions and tuning candidates.",
              f"- Main simulation/fit runtime {meta['runtime_seconds']:.2f} seconds in the recorded environment, excluding setup, tests, figures and documentation; not a QPU runtime estimate.",
              "- See verification.json and reproduction_verification.json for the independent reconstruction and fresh-run comparison.", "",
              "## Interpretation", "", "Mean test MSE decreases with more shots for both models on both tasks in the primary study. Ridge improves five of six finite-shot task/budget means, usually modestly, and worsens mixed harmonics at 32 shots. The secondary noise condition also gives mixed results. This does not establish statistical significance, general superiority, a new method or quantum advantage."]
    (ROOT/"docs/RESULTS.md").write_text("\n".join(lines)+"\n")
    print("Updated docs/RESULTS.md from saved results")
