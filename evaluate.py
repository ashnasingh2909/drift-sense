import argparse
import json
import os
import time
import numpy as np
import cv2

from drift_sense.matcher import locate_reference, locate_reference_naive, locate_reference_dl
from drift_sense.cnn import DriftSenseCNN

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Drift-Sense Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #f7f8fa; color: #1c1f26; margin: 0; padding: 40px; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; margin: 40px 0 14px; }}
  .subtitle {{ color: #6b7280; margin-bottom: 32px; }}
  .note {{ color: #6b7280; font-size: 13px; margin-bottom: 18px; max-width: 760px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px 22px; min-width: 150px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #6b7280; }}
  .card .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; color: #111827; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); margin-bottom: 8px; }}
  th, td {{ padding: 10px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid #eef0f3; }}
  th {{ background: #f1f3f6; color: #374151; font-weight: 600; }}
  tr:hover {{ background: #fafbfc; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .warn {{ color: #d97706; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge.dram {{ background: #dbeafe; color: #1d4ed8; }}
  .badge.finfet {{ background: #ede9fe; color: #6d28d9; }}
  .flag {{ background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 999px; font-size: 11px; }}
  .method {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .method.voting {{ background: #dcfce7; color: #166534; }}
  .method.naive {{ background: #fee2e2; color: #991b1b; }}
</style>
</head>
<body>
  <h1>Drift-Sense Self-Evaluation Report</h1>
  <div class="subtitle">Generated {timestamp} &middot; {n} image pairs</div>

  <h2>Drift-Sense (multi-hypothesis voting) <span class="method voting">this project</span></h2>
  <div class="stats">
    <div class="card"><div class="label">Mean Error</div><div class="value">{mean_err:.2f} px</div></div>
    <div class="card"><div class="label">Median Error</div><div class="value">{median_err:.2f} px</div></div>
    <div class="card"><div class="label">P90 Error</div><div class="value">{p90_err:.2f} px</div></div>
    <div class="card"><div class="label">Within 15px</div><div class="value">{within15:.0f}%</div></div>
    <div class="card"><div class="label">Catastrophic Fails (&gt;100px)</div><div class="value">{n_fail}</div></div>
    <div class="card"><div class="label">Ambiguous Sites</div><div class="value">{n_ambiguous}</div></div>
    <div class="card"><div class="label">Avg Runtime</div><div class="value">{avg_time:.2f}s</div></div>
  </div>

  <h2>Baseline: single-best-peak NCC <span class="method naive">classical, no voting</span></h2>
  <div class="note">
    Same multi-scale/multi-angle correlation search, but it commits to whichever single
    (scale, angle) trial produced the highest raw peak, with no cross-trial voting and no
    periodicity disambiguation &mdash; this is the standard template-matching approach the
    problem statement says breaks down on periodic DRAM/FinFET layouts. Run for comparison only;
    not used by <code>inference.py</code>.
  </div>
  <div class="stats">
    <div class="card"><div class="label">Mean Error</div><div class="value">{b_mean_err:.2f} px</div></div>
    <div class="card"><div class="label">Median Error</div><div class="value">{b_median_err:.2f} px</div></div>
    <div class="card"><div class="label">P90 Error</div><div class="value">{b_p90_err:.2f} px</div></div>
    <div class="card"><div class="label">Within 15px</div><div class="value">{b_within15:.0f}%</div></div>
    <div class="card"><div class="label">Catastrophic Fails (&gt;100px)</div><div class="value">{b_n_fail}</div></div>
    <div class="card"><div class="label">Avg Runtime</div><div class="value">{b_avg_time:.2f}s</div></div>
  </div>

  <h2>Per-pair results</h2>
  <table>
    <tr><th>#</th><th>Style</th><th>GT (x, y)</th><th>Voting Pred</th><th>Voting Err</th><th>Baseline Pred</th><th>Baseline Err</th><th>Ambiguous</th></tr>
    {rows}
  </table>
</body>
</html>
"""


def error_class(err):
    if err < 15:
        return "good"
    if err < 40:
        return "warn"
    return "bad"


def main():
    parser = argparse.ArgumentParser(description="Self-evaluation on the generated dataset")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", default="./results/report.html")
    parser.add_argument("--no-baseline", action="store_true",
                         help="Skip the naive single-peak baseline comparison (faster)")
    parser.add_argument("--include-dl", action="store_true",
                         help="Also run the experimental from-scratch CNN matcher (slow, "
                              "and currently underperforms both other methods — see README)")
    parser.add_argument("--dl-weights", default="./models/drift_sense_cnn.npz")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "ground_truth.json")) as f:
        records = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    net = None
    if args.include_dl:
        net = DriftSenseCNN()
        net.load(args.dl_weights)

    errors, b_errors, dl_errors, rows, times, b_times, dl_times = [], [], [], [], [], [], []
    n_ambiguous = 0

    for i, r in enumerate(records):
        ref = cv2.imread(os.path.join(args.data_dir, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(args.data_dir, r["search_path"]), cv2.IMREAD_GRAYSCALE)

        t0 = time.time()
        result = locate_reference(ref, search)
        times.append(time.time() - t0)

        err = float(np.hypot(result.x - r["gt_x"], result.y - r["gt_y"]))
        errors.append(err)
        if result.ambiguous:
            n_ambiguous += 1

        if args.no_baseline:
            b_err, b_x, b_y = float("nan"), None, None
        else:
            t0 = time.time()
            b_result = locate_reference_naive(ref, search)
            b_times.append(time.time() - t0)
            b_err = float(np.hypot(b_result.x - r["gt_x"], b_result.y - r["gt_y"]))
            b_errors.append(b_err)
            b_x, b_y = b_result.x, b_result.y

        dl_status = ""
        if net is not None:
            t0 = time.time()
            dl_result = locate_reference_dl(ref, search, net)
            dl_times.append(time.time() - t0)
            dl_err = float(np.hypot(dl_result.x - r["gt_x"], dl_result.y - r["gt_y"]))
            dl_errors.append(dl_err)
            dl_status = f" dl_err={dl_err:8.2f}px"

        baseline_cell = (
            f"({b_x:.1f}, {b_y:.1f})" if b_x is not None else "-"
        )
        baseline_err_cell = (
            f"<td class='{error_class(b_err)}'>{b_err:.2f}</td>" if b_x is not None else "<td>-</td>"
        )

        rows.append(
            f"<tr><td>{i}</td><td><span class='badge {r['style']}'>{r['style']}</span></td>"
            f"<td>({r['gt_x']:.1f}, {r['gt_y']:.1f})</td>"
            f"<td>({result.x:.1f}, {result.y:.1f})</td>"
            f"<td class='{error_class(err)}'>{err:.2f}</td>"
            f"<td>{baseline_cell}</td>"
            f"{baseline_err_cell}"
            f"<td>{'<span class=\"flag\">ambiguous</span>' if result.ambiguous else '-'}</td></tr>"
        )
        print(f"[{i + 1}/{len(records)}] {r['style']:6s} voting_err={err:6.2f}px "
              f"baseline_err={b_err:6.2f}px{dl_status} ambiguous={result.ambiguous}")

    errors = np.array(errors)
    b_errors = np.array(b_errors) if b_errors else np.array([np.nan])

    html = HTML_TEMPLATE.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        n=len(records),
        mean_err=errors.mean(),
        median_err=np.median(errors),
        p90_err=np.percentile(errors, 90),
        within15=100.0 * np.mean(errors < 15),
        n_fail=int(np.sum(errors > 100)),
        n_ambiguous=n_ambiguous,
        avg_time=np.mean(times),
        b_mean_err=np.nanmean(b_errors),
        b_median_err=np.nanmedian(b_errors),
        b_p90_err=np.nanpercentile(b_errors, 90),
        b_within15=100.0 * np.nanmean(b_errors < 15),
        b_n_fail=int(np.nansum(b_errors > 100)),
        b_avg_time=np.mean(b_times) if b_times else 0.0,
        rows="\n".join(rows),
    )

    with open(args.output, "w") as f:
        f.write(html)

    print(f"\nDrift-Sense (voting)   mean={errors.mean():.2f}px  median={np.median(errors):.2f}px  "
          f"fails(>100px)={int(np.sum(errors > 100))}")
    if not args.no_baseline:
        print(f"Baseline (single-peak) mean={np.nanmean(b_errors):.2f}px  "
              f"median={np.nanmedian(b_errors):.2f}px  fails(>100px)={int(np.nansum(b_errors > 100))}")
    if dl_errors:
        dl_arr = np.array(dl_errors)
        print(f"DL (from-scratch CNN)  mean={dl_arr.mean():.2f}px  "
              f"median={np.median(dl_arr):.2f}px  fails(>100px)={int(np.sum(dl_arr > 100))}  "
              f"[experimental — see README, currently underperforms both methods above]")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
