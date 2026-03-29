import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from action_lipo.lipo import ActionLiPo

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and visualize LiPo solver variants on one figure."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/log_inference_actions.npy"),
        help="Path to the inference action log.",
    )
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--blend", type=int, default=10)
    parser.add_argument("--time-delay", type=int, default=3)
    parser.add_argument("--joint-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.0333)
    parser.add_argument("--x-max", type=float, default=7.0)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path. If omitted, the figure is shown.",
    )
    return parser.parse_args()


def make_solver_specs(chunk, blend, time_delay):
    CVXPY_kwargs = {
        "solver": "cvxpy",
        "chunk_size": chunk,
        "blending_horizon": blend,
        "len_time_delay": time_delay,
    }
    OSQP_kwargs = {
        "solver": "osqp",
        "chunk_size": chunk,
        "blending_horizon": blend,
        "len_time_delay": time_delay,
    }
    NUMPY_kwargs = {
        "solver": "numpy",
        "chunk_size": chunk,
        "blending_horizon": blend,
        "len_time_delay": time_delay,
    }

    return [
        ("LiPo", ActionLiPo, CVXPY_kwargs),
        ("OSQP", ActionLiPo, OSQP_kwargs),
        ("NUMPY", ActionLiPo, NUMPY_kwargs),
    ]


def run_solver(label, solver_cls, solver_kwargs, action_chunks, chunk, blend, dt, start_index):
    solver_instance = solver_cls(**solver_kwargs)

    prev_chunk = None
    solved_segments = []
    blended_segments = []
    stitched_times = []
    chunk_records = []
    solve_times = []
    
    total_chunks = len(action_chunks) // chunk
    bias_time = 4.5 * (chunk - blend) * dt

    for k in range(start_index, total_chunks):
        start_time = k * (chunk - blend) * dt - bias_time
        action_chunk = action_chunks[k * chunk : (k + 1) * chunk]
        chunk_time = np.arange(chunk, dtype=np.float64) * dt + start_time

        solved, blended = solver_instance.solve(
            action_chunk,
            prev_chunk,
            len_past_actions=blend if prev_chunk is not None else 0,
        )
        if solved is None:
            raise RuntimeError(f"{label} solve failed at chunk {k}: {blended}")

        solve_log = solver_instance.get_log()[-1]
        solve_times.append(float(solve_log["time"]))

        stitched_time = np.arange(chunk - blend, dtype=np.float64) * dt + start_time
        solved_segments.append(solved[: chunk - blend].copy())
        blended_segments.append(blended[: chunk - blend].copy())
        stitched_times.append(stitched_time)

        chunk_records.append(
            {
                "chunk_index": k,
                "time_full": chunk_time,
                "raw": action_chunk.copy(),
                "blended_full": blended.copy(),
            }
        )
        prev_chunk = solved.copy()

    return {
        "label": label,
        "solver": solver_instance,
        "chunk_records": chunk_records,
        "times": np.asarray(solve_times, dtype=np.float64),
        "stitched_time": np.concatenate(stitched_times),
        "stitched_solved": np.concatenate(solved_segments, axis=0),
        "stitched_blended": np.concatenate(blended_segments, axis=0),
    }


def build_summary(results, baseline_label):
    baseline = results[baseline_label]["stitched_solved"]
    summary = []
    for label, result in results.items():
        diff = result["stitched_solved"] - baseline
        mean_time_ms = float(np.mean(result["times"]) * 1000.0)
        std_time_ms = float(np.std(result["times"]) * 1000.0)
        mean_abs_diff = float(np.mean(np.abs(diff)))
        max_abs_diff = float(np.max(np.abs(diff)))
        summary.append(
            {
                "label": label,
                "mean_time_ms": mean_time_ms,
                "std_time_ms": std_time_ms,
                "mean_abs_diff": mean_abs_diff,
                "max_abs_diff": max_abs_diff,
            }
        )
    return summary


def print_summary(summary):
    print("LiPo Benchmark Summary")
    print(
        f"{'Implementation':<20} {'Avg [ms]':>10} {'Std [ms]':>10} "
        f"{'Mean |Δ| vs LiPo':>18} {'Max |Δ| vs LiPo':>17}"
    )
    for row in summary:
        print(
            f"{row['label']:<20} {row['mean_time_ms']:>10.3f} {row['std_time_ms']:>10.3f} "
            f"{row['mean_abs_diff']:>18.6f} {row['max_abs_diff']:>17.6f}"
        )


def plot_results(results, summary, joint_index, chunk, blend, time_delay, dt, x_max):
    fig, (ax_main, ax_time) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [3.2, 1.2]},
    )

    baseline = results["LiPo"]
    baseline_records = baseline["chunk_records"]

    raw_color = "#7c7f86"
    blended_color = "#1f1f1f"
    solver_styles = {
        "LiPo": {
            "color": "#0057E7",
            "linewidth": 2.8,
            "linestyle": "-",
            "alpha": 0.68,
            "zorder": 8,
        },
        "OSQP": {
            "color": "#FF5A1F",
            "linewidth": 2.8,
            "linestyle": "-",
            "alpha": 0.68,
            "zorder": 11,
        },
        "NUMPY": {
            "color": "#2ECC40",
            "linewidth": 2.8,
            "linestyle": "-",
            "alpha": 0.68,
            "zorder": 14,
        },
    }

    epsilon_blend = baseline["solver"].epsilon_blending
    epsilon_path = baseline["solver"].epsilon_path

    for idx, record in enumerate(baseline_records):
        start_time = record["time_full"][0]
        delay_time = time_delay * dt
        blending_time = blend * dt
        time_vals = record["time_full"]

        epsilons = np.concatenate(
            [
                np.full(time_delay + 1, 0.0),
                np.full(max(0, blend - time_delay - 1), epsilon_blend),
                np.full(len(time_vals) - blend, epsilon_path),
            ]
        )

        ax_main.plot(
            record["time_full"],
            record["raw"][:, joint_index],
            color=raw_color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.34,
            zorder=1,
            label="Raw Action Chunk" if idx == 0 else None,
        )
        ax_main.fill_between(
            time_vals,
            record["blended_full"][:, joint_index] - epsilons,
            record["blended_full"][:, joint_index] + epsilons,
            color="#AEB7C2",
            alpha=0.11,
            zorder=0,
            label="Epsilon Band" if idx == 0 else None,
        )
        ax_main.axvspan(
            start_time,
            start_time + delay_time,
            color="#9AA0A6",
            alpha=0.08,
            zorder=0,
            label="Inference Delay" if idx == 0 else None,
        )
        ax_main.axvspan(
            start_time + delay_time,
            start_time + blending_time,
            color="#FFD166",
            alpha=0.10,
            zorder=0,
            label="Blending Zone" if idx == 0 else None,
        )
        ax_main.axvline(
            x=start_time,
            color="#8B9096",
            linestyle="--",
            linewidth=0.5,
            alpha=0.45,
            zorder=1,
        )

    ax_main.plot(
        baseline["stitched_time"],
        baseline["stitched_blended"][:, joint_index],
        color=blended_color,
        linestyle="--",
        linewidth=1.9,
        alpha=0.82,
        zorder=3,
        label="Linearly Blended Action",
    )

    for row in summary:
        label = row["label"]
        style = solver_styles[label]
        legend_label = f"{label} ({row['mean_time_ms']:.2f} ms, max |Δ| {row['max_abs_diff']:.4f})"
        ax_main.plot(
            results[label]["stitched_time"],
            results[label]["stitched_solved"][:, joint_index],
            color="white",
            linewidth=style["linewidth"] + 2.0,
            linestyle=style["linestyle"],
            alpha=0.95,
            zorder=style["zorder"] - 0.1,
        )
        ax_main.plot(
            results[label]["stitched_time"],
            results[label]["stitched_solved"][:, joint_index],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=legend_label,
        )

    ax_main.set_xlim(0.0, x_max)
    ax_main.set_xlabel("Time (s)")
    ax_main.set_ylabel("Action - Joint Angle (rad)")
    ax_main.set_title("LiPo Solver Benchmark")
    ax_main.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax_main.legend(loc="upper left", frameon=True, framealpha=0.92)

    chunk_ids = np.arange(len(baseline["times"]), dtype=np.int32)
    for row in summary:
        label = row["label"]
        style = solver_styles[label]
        ax_time.plot(
            chunk_ids,
            results[label]["times"] * 1000.0,
            color=style["color"],
            linewidth=2.0,
            marker="o",
            markersize=5.0,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            alpha=0.95,
            label=label,
        )

    ax_time.set_xlabel("Solved Chunk Index")
    ax_time.set_ylabel("Solve Time (ms)")
    ax_time.grid(alpha=0.18, linewidth=0.6)
    ax_time.legend(loc="upper left", ncol=4, frameon=True, framealpha=0.92)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()

    action_chunks = np.asarray(
        np.load(args.data, allow_pickle=True),
        dtype=np.float64,
    )

    results = {}
    for label, solver_cls, solver_kwargs in make_solver_specs(
        args.chunk,
        args.blend,
        args.time_delay,
    ):
        results[label] = run_solver(
            label=label,
            solver_cls=solver_cls,
            solver_kwargs=solver_kwargs,
            action_chunks=action_chunks,
            chunk=args.chunk,
            blend=args.blend,
            dt=args.dt,
            start_index=args.start_index,
        )

    summary = build_summary(results, baseline_label="LiPo")
    print_summary(summary)

    fig = plot_results(
        results=results,
        summary=summary,
        joint_index=args.joint_index,
        chunk=args.chunk,
        blend=args.blend,
        time_delay=args.time_delay,
        dt=args.dt,
        x_max=args.x_max,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()