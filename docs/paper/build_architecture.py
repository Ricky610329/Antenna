"""Build the paper's learning, execution, and timing diagrams.

The figures deliberately separate the research learning loop from emforge's
execution contract. The timing graphic is an illustrative solver-only
calculation; it is not a measured scaling result.
"""

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "antenna-paper-architecture-v6",
        "axes.titlesize": 13,
    }
)

INK = "#17324A"
TEAL = "#087F83"
TEAL_DARK = "#07666A"
TEAL_PALE = "#EAF5F3"
BLUE = "#416B8A"
BLUE_PALE = "#EEF3F8"
GRAY = "#697782"
GRAY_PALE = "#F3F5F6"
BOUNDARY = "#AFCFCC"
PAPER = "#FFFFFF"


def rounded_box(ax, x, y, w, h, *, face, edge, linewidth=1.35, radius=0.12, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=TEAL, width=1.7, dashed=False, z=4):
    return ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            linestyle=(0, (4, 3)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def poly_arrow(ax, points, *, color=TEAL, width=1.7, dashed=False, z=4):
    style = (0, (4, 3)) if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=width,
            linestyle=style,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=z,
        )
    arrow(ax, points[-2], points[-1], color=color, width=width, dashed=dashed, z=z)


def arrow_label(ax, x, y, text, *, color=TEAL_DARK, size=9.0):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=size,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.12", "fc": PAPER, "ec": "none"},
        zorder=7,
    )


def bounded_text(ax, bounds, x, y, text, **kwargs):
    """Add text that must remain inside a module's data-coordinate bounds."""
    artist = ax.text(x, y, text, **kwargs)
    checks = getattr(ax, "_bounded_text_checks", [])
    checks.append((bounds, artist))
    ax._bounded_text_checks = checks
    return artist


def validate_bounded_text(fig):
    """Fail if module text crosses its box or any label crosses the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    bounded_count = 0
    canvas_count = 0
    for ax in fig.axes:
        for (x, y, w, h), artist in getattr(ax, "_bounded_text_checks", []):
            bounded_count += 1
            left, bottom = ax.transData.transform((x, y))
            right, top = ax.transData.transform((x + w, y + h))
            extent = artist.get_window_extent(renderer=renderer)
            tolerance = 0.5
            if not (
                left - tolerance <= extent.x0
                and extent.x1 <= right + tolerance
                and bottom - tolerance <= extent.y0
                and extent.y1 <= top + tolerance
            ):
                raise RuntimeError(
                    f"Text exceeds module bounds: {artist.get_text()!r} "
                    f"in {(x, y, w, h)}"
                )
        for artist in ax.texts:
            canvas_count += 1
            extent = artist.get_window_extent(renderer=renderer)
            if not (
                canvas.x0 - 0.5 <= extent.x0
                and extent.x1 <= canvas.x1 + 0.5
                and canvas.y0 - 0.5 <= extent.y0
                and extent.y1 <= canvas.y1 + 0.5
            ):
                raise RuntimeError(f"Text exceeds figure canvas: {artist.get_text()!r}")
    for artist in fig.texts:
        canvas_count += 1
        extent = artist.get_window_extent(renderer=renderer)
        if not (
            canvas.x0 - 0.5 <= extent.x0
            and extent.x1 <= canvas.x1 + 0.5
            and canvas.y0 - 0.5 <= extent.y0
            and extent.y1 <= canvas.y1 + 0.5
        ):
            raise RuntimeError(f"Figure text exceeds canvas: {artist.get_text()!r}")
    return bounded_count, canvas_count


def save_figure(fig, stem):
    svg = OUT / f"{stem}.svg"
    png = OUT / f"{stem}.png"
    bounded_count, canvas_count = validate_bounded_text(fig)
    fig.savefig(svg, metadata={"Date": None}, facecolor=PAPER)
    fig.savefig(png, dpi=200, facecolor=PAPER)
    plt.close(fig)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{stem}: validated {bounded_count} module labels and "
        f"{canvas_count} total text elements."
    )


def learning_module(ax, x, y, w, h, number, title, detail, *, model=False):
    face = BLUE_PALE if model else TEAL_PALE
    edge = BLUE if model else TEAL
    rounded_box(ax, x, y, w, h, face=face, edge=edge, linewidth=1.5)
    rounded_box(
        ax,
        x + 0.16,
        y + h - 0.50,
        0.45,
        0.31,
        face=edge,
        edge=edge,
        linewidth=0,
        radius=0.06,
        z=3,
    )
    bounded_text(
        ax,
        (x + 0.16, y + h - 0.50, 0.45, 0.31),
        x + 0.385,
        y + h - 0.345,
        str(number),
        ha="center",
        va="center",
        color=PAPER,
        fontsize=9.2,
        weight="bold",
        zorder=5,
    )
    bounded_text(
        ax,
        (x + 0.68, y + 0.58, w - 0.82, h - 0.70),
        x + 0.72,
        y + h - 0.34,
        title,
        ha="left",
        va="center",
        color=INK,
        fontsize=10.5,
        weight="bold",
        zorder=5,
    )
    bounded_text(
        ax,
        (x + 0.14, y + 0.08, w - 0.28, 0.42),
        x + 0.18,
        y + 0.27,
        detail,
        ha="left",
        va="center",
        color=INK,
        fontsize=9.2,
        zorder=5,
    )


def build_learning_loop():
    fig, ax = plt.subplots(figsize=(13.4, 6.5))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.94, bottom=0.06)
    ax.set(xlim=(0, 13.4), ylim=(0, 6.5))
    ax.axis("off")

    ax.text(0.45, 6.12, "BATCHWISE SURROGATE LEARNING", color=TEAL_DARK, fontsize=14.0, weight="bold")

    learning_module(ax, 0.55, 3.70, 2.55, 1.18, 1, "Candidate pool", "Mutate / recombine / sample")
    learning_module(ax, 3.70, 3.70, 3.15, 1.18, 2, "SM prediction + selection", "Score candidates; choose a batch", model=True)
    learning_module(ax, 7.45, 3.70, 2.60, 1.18, 3, "Run batch EM", "Simulate selected patterns")
    learning_module(ax, 7.45, 1.25, 3.15, 1.18, 4, "Compatible dataset", "Historical + new EM pairs")
    learning_module(ax, 3.70, 1.25, 2.95, 1.18, 5, "SM update", "Fit on cumulative data", model=True)

    arrow(ax, (3.10, 4.29), (3.70, 4.29))
    arrow_label(ax, 3.40, 5.13, "candidates")
    arrow(ax, (6.85, 4.29), (7.45, 4.29))
    arrow_label(ax, 7.15, 5.13, "selected batch")
    poly_arrow(ax, [(8.75, 3.70), (8.75, 2.43)])
    arrow_label(ax, 9.75, 3.07, "EM pairs")
    arrow(ax, (7.45, 1.84), (6.65, 1.84), color=BLUE)
    arrow_label(ax, 7.05, 0.98, "training data", color=BLUE)
    poly_arrow(ax, [(5.18, 2.43), (5.18, 3.42), (5.27, 3.42), (5.27, 3.70)], color=BLUE)
    arrow_label(ax, 4.38, 3.07, "updated SM", color=BLUE)

    # Evaluated designs provide the parent pool for the next generation.
    poly_arrow(
        ax,
        [(8.20, 1.25), (8.20, 0.72), (0.28, 0.72), (0.28, 4.05), (0.55, 4.05)],
        color=TEAL,
    )
    arrow_label(ax, 1.52, 0.72, "EM-evaluated parents", color=TEAL_DARK, size=8.6)

    rounded_box(ax, 10.95, 1.30, 1.85, 1.08, face=PAPER, edge=INK, linewidth=1.45)
    ax.text(11.875, 1.96, "EM-verified", ha="center", va="center", color=INK, fontsize=10.8, weight="bold")
    ax.text(11.875, 1.62, "best design", ha="center", va="center", color=INK, fontsize=10.8, weight="bold")
    arrow(ax, (10.60, 1.84), (10.95, 1.84), color=INK)
    arrow_label(ax, 10.78, 2.63, "rank", color=INK, size=8.6)

    save_figure(fig, "architecture")


def service_box(ax, x, y, w, h, title, detail, *, face=GRAY_PALE, edge=BLUE):
    rounded_box(ax, x, y, w, h, face=face, edge=edge, linewidth=1.35)
    bounded_text(
        ax,
        (x + 0.13, y + h - 0.55, w - 0.26, 0.42),
        x + 0.18,
        y + h - 0.34,
        title,
        ha="left",
        va="center",
        color=INK,
        fontsize=10.6,
        weight="bold",
    )
    bounded_text(
        ax,
        (x + 0.13, y + 0.08, w - 0.26, 0.42),
        x + 0.18,
        y + 0.27,
        detail,
        ha="left",
        va="center",
        color=INK,
        fontsize=8.9,
    )


def build_emforge_architecture():
    fig, ax = plt.subplots(figsize=(13.4, 6.6))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.94, bottom=0.06)
    ax.set(xlim=(0, 13.4), ylim=(0, 6.6))
    ax.axis("off")

    ax.text(0.42, 6.17, "EMFORGE EXECUTION ARCHITECTURE", color=INK, fontsize=14.0, weight="bold")
    ax.text(12.95, 6.17, "Abstract service contract", ha="right", color=GRAY, fontsize=9.4, style="italic")

    rounded_box(ax, 0.42, 0.75, 3.35, 4.85, face="#F8FBFA", edge=BOUNDARY, linewidth=1.2)
    rounded_box(ax, 4.18, 0.75, 5.12, 4.85, face="#F8FAFC", edge="#B9C9D5", linewidth=1.2)
    rounded_box(ax, 9.72, 0.75, 3.25, 4.85, face="#FAFAFA", edge="#CBD1D5", linewidth=1.2)
    ax.text(0.67, 5.28, "STRATEGY / ALGORITHM", color=TEAL_DARK, fontsize=11.6, weight="bold")
    ax.text(4.43, 5.28, "SHARED JOB + DATA SERVICES", color=BLUE, fontsize=11.6, weight="bold")
    ax.text(9.97, 5.28, "HFSS WORKERS", color=GRAY, fontsize=11.6, weight="bold")

    # One connected container makes algorithm ownership explicit.
    rounded_box(ax, 0.72, 1.12, 2.75, 3.72, face=TEAL_PALE, edge=TEAL, linewidth=1.5)
    bounded_text(
        ax,
        (0.90, 4.18, 2.35, 0.45),
        0.92,
        4.42,
        "Hosted algorithm",
        ha="left",
        va="center",
        color=INK,
        fontsize=11.0,
        weight="bold",
    )
    for y in (3.82, 2.80):
        ax.plot([0.92, 3.27], [y, y], color=BOUNDARY, linewidth=1.0, zorder=3)
    bounded_text(
        ax,
        (0.90, 3.05, 2.35, 0.62),
        0.92,
        3.47,
        "Candidate policy\nSubmit / query results",
        ha="left",
        va="center",
        color=INK,
        fontsize=9.5,
        linespacing=1.35,
    )
    bounded_text(
        ax,
        (0.90, 2.00, 2.35, 0.62),
        0.92,
        2.42,
        "Model + update cadence\nAlgorithm-owned",
        ha="left",
        va="center",
        color=INK,
        fontsize=9.5,
        linespacing=1.35,
    )
    bounded_text(
        ax,
        (0.90, 1.30, 2.35, 0.48),
        0.92,
        1.55,
        "State + checkpoint",
        ha="left",
        va="center",
        color=INK,
        fontsize=9.5,
    )

    service_box(ax, 4.50, 3.92, 4.38, 0.92, "Validate + deduplicate", "Identity check; reuse completed result")
    service_box(ax, 4.50, 2.53, 1.92, 0.98, "Queue", "Schedule new jobs")
    service_box(ax, 6.96, 2.53, 1.92, 0.98, "Collect", "Normalize results")
    service_box(ax, 4.50, 1.08, 4.38, 0.98, "Shared result store", "Design identity + solver outputs", face=PAPER, edge=INK)

    for lane_y, label in [(4.12, "Worker 1"), (3.12, "Worker 2"), (2.12, "Worker N")]:
        rounded_box(ax, 10.12, lane_y, 2.42, 0.67, face=PAPER, edge=GRAY, linewidth=1.25, radius=0.08)
        bounded_text(
            ax,
            (10.22, lane_y + 0.08, 2.22, 0.51),
            11.33,
            lane_y + 0.335,
            label + "  |  build + solve",
            ha="center",
            va="center",
            color=INK,
            fontsize=9.2,
        )
    ax.text(11.33, 2.92, ". . .", ha="center", va="center", color=GRAY, fontsize=11.0)

    arrow(ax, (3.47, 4.38), (4.50, 4.38), color=TEAL)
    arrow_label(ax, 3.99, 4.70, "requests", color=TEAL_DARK)

    # New identities enter the queue; the result store is queried for reuse.
    arrow(ax, (5.46, 3.92), (5.46, 3.51), color=BLUE)
    arrow_label(ax, 5.98, 3.70, "new jobs", color=BLUE, size=8.3)
    ax.plot([6.82, 6.82], [2.06, 3.61], color=BLUE, linewidth=1.7, linestyle=(0, (4, 3)), zorder=3)
    arrow(ax, (6.82, 3.79), (6.82, 3.92), color=BLUE, width=1.7, dashed=True, z=3)
    ax.text(
        6.82,
        2.93,
        "store lookup / reuse",
        ha="center",
        va="center",
        rotation=90,
        color=BLUE,
        fontsize=7.5,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.10", "fc": PAPER, "ec": "none"},
        zorder=6,
    )

    # Scheduled work moves right; worker results return through collection.
    poly_arrow(ax, [(6.42, 3.02), (6.55, 3.02), (6.55, 3.70), (9.72, 3.70)], color=BLUE)
    arrow_label(ax, 8.48, 3.70, "scheduled jobs", color=BLUE, size=8.2)
    arrow(ax, (9.72, 3.02), (8.88, 3.02), color=GRAY)
    arrow_label(ax, 9.30, 2.73, "results", color=GRAY, size=8.2)
    arrow(ax, (7.92, 2.53), (7.92, 2.06), color=BLUE)

    # Stored results feed the algorithm, which alone decides the next update.
    arrow(ax, (4.50, 1.57), (3.47, 1.57), color=TEAL)
    arrow_label(ax, 3.98, 1.90, "stored results", color=TEAL_DARK, size=8.2)
    save_figure(fig, "emforge-architecture")


def build_execution_time():
    fig = plt.figure(figsize=(13.4, 6.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.15)
    ax_flow = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.04, right=0.975, top=0.86, bottom=0.20)
    fig.suptitle("EXECUTION-TIME STRUCTURE AND SOLVER-ONLY ESTIMATE", x=0.04, ha="left", y=0.96, color=INK, fontsize=14.0, weight="bold")

    ax_flow.set(xlim=(0, 10.4), ylim=(0, 6))
    ax_flow.axis("off")
    ax_flow.text(0.05, 5.72, "(a)  SCHEMATIC SWIMLANE - NOT TO SCALE", color=TEAL_DARK, fontsize=11.0, weight="bold")

    row_y = {
        "Candidate / control": 4.30,
        "Worker 1": 3.37,
        "Worker 2": 2.68,
        "Worker 3": 1.99,
        "SM train / evaluate": 1.02,
    }
    lane_start, lane_end = 1.92, 10.02
    for label, y in row_y.items():
        ax_flow.text(1.72, y, label, ha="right", va="center", color=INK, fontsize=8.3, weight="bold")
        ax_flow.plot([lane_start, lane_end], [y, y], color="#D5DDE1", linewidth=0.85, zorder=0)
    arrow(ax_flow, (lane_start, 5.15), (lane_end, 5.15), color=GRAY, width=1.1)
    ax_flow.text(5.95, 5.33, "execution sequence", ha="center", va="center", color=GRAY, fontsize=8.3)

    def timeline_bar(x, y, w, text, *, face, edge, fontsize=7.0):
        h = 0.43
        rounded_box(ax_flow, x, y - h / 2, w, h, face=face, edge=edge, linewidth=1.05, radius=0.045, z=3)
        bounded_text(
            ax_flow,
            (x + 0.04, y - h / 2 + 0.04, w - 0.08, h - 0.08),
            x + w / 2,
            y,
            text,
            ha="center",
            va="center",
            color=INK,
            fontsize=fontsize,
            weight="bold",
            zorder=5,
        )

    batch_starts = (2.00, 4.18, 6.36)
    collect_centers = []
    for number, start in enumerate(batch_starts, start=1):
        ax_flow.text(start + 0.98, 4.85, f"Batch {number}", ha="center", va="center", color=BLUE, fontsize=8.2, weight="bold")
        timeline_bar(start, row_y["Candidate / control"], 0.55, "select", face=TEAL_PALE, edge=TEAL, fontsize=6.5)
        submit_x = start + 0.58
        ax_flow.plot(
            [submit_x, submit_x],
            [row_y["Worker 3"] - 0.29, row_y["Candidate / control"] - 0.22],
            color=TEAL,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
        for worker in ("Worker 1", "Worker 2", "Worker 3"):
            timeline_bar(submit_x, row_y[worker], 0.92, "solve", face=BLUE_PALE, edge=BLUE, fontsize=6.8)
        collect_x = submit_x + 0.94
        ax_flow.plot(
            [collect_x, collect_x],
            [row_y["Worker 3"] - 0.29, row_y["Candidate / control"] - 0.22],
            color=GRAY,
            linewidth=1.15,
            zorder=2,
        )
        timeline_bar(collect_x, row_y["Candidate / control"], 0.62, "collect", face=GRAY_PALE, edge=GRAY, fontsize=5.9)
        collect_centers.append(collect_x + 0.31)

    # Completed batches accumulate; the SM update begins only after batch 3.
    rounded_box(
        ax_flow,
        batch_starts[0],
        row_y["SM train / evaluate"] - 0.18,
        collect_centers[-1] - batch_starts[0],
        0.36,
        face="#F6F7F8",
        edge="#B7C0C6",
        linewidth=0.9,
        radius=0.04,
        z=1,
    )
    bounded_text(
        ax_flow,
        (batch_starts[0] + 0.08, row_y["SM train / evaluate"] - 0.14, collect_centers[-1] - batch_starts[0] - 0.16, 0.28),
        (batch_starts[0] + collect_centers[-1]) / 2,
        row_y["SM train / evaluate"],
        "wait / accumulate completed batches",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=7.0,
        style="italic",
        zorder=4,
    )
    for x in collect_centers:
        ax_flow.plot(
            [x, x],
            [row_y["SM train / evaluate"] + 0.18, row_y["Candidate / control"] - 0.22],
            color="#AAB5BC",
            linewidth=0.9,
            linestyle=(0, (2, 3)),
            zorder=0,
        )
    train_x = collect_centers[-1] + 0.35
    poly_arrow(
        ax_flow,
        [
            (collect_centers[-1] + 0.31, row_y["Candidate / control"]),
            (train_x - 0.12, row_y["Candidate / control"]),
            (train_x - 0.12, row_y["SM train / evaluate"]),
            (train_x, row_y["SM train / evaluate"]),
        ],
        color=TEAL,
        width=1.35,
    )
    timeline_bar(train_x, row_y["SM train / evaluate"], 1.43, "train / evaluate", face=TEAL_PALE, edge=TEAL, fontsize=7.0)

    workers = [1, 2, 3]
    minutes = [ceil(90 / count) * 160 / 60 for count in workers]
    bars = ax_bar.bar(workers, minutes, width=0.62, color=[TEAL, BLUE, GRAY], edgecolor=INK, linewidth=0.8)
    ax_bar.set_title("(b)  ILLUSTRATIVE ANTENNA SOLVER-ONLY ESTIMATE", loc="left", color=INK, fontsize=10.8, weight="bold", pad=25)
    ax_bar.text(
        0.0,
        0.985,
        "Assumed 90 patterns; historical median 160 s/pattern",
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        color=GRAY,
        fontsize=8.4,
    )
    ax_bar.set_xlabel("Parallel HFSS workers", color=INK, labelpad=8)
    ax_bar.set_ylabel("Estimated solver makespan (min)", color=INK, labelpad=8)
    ax_bar.set_xticks(workers)
    ax_bar.set_ylim(0, 300)
    ax_bar.set_yticks([0, 60, 120, 180, 240])
    ax_bar.grid(axis="y", color="#D9DFE2", linewidth=0.8)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.spines[["left", "bottom"]].set_color(GRAY)
    ax_bar.tick_params(colors=INK)
    for bar, value in zip(bars, minutes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, value + 7, f"{value:.0f} min", ha="center", va="bottom", color=INK, fontsize=10.0, weight="bold")
    ax_bar.text(0.5, -0.14, "ceil(90 / workers) x 160 s\nSolver only; equal tasks; excludes queue, setup, retries, selection, training", transform=ax_bar.transAxes, ha="center", va="top", color=GRAY, fontsize=8.2, linespacing=1.3)
    save_figure(fig, "execution-time")


if __name__ == "__main__":
    build_learning_loop()
    build_emforge_architecture()
    build_execution_time()
    print("Wrote architecture, emforge-architecture, and execution-time as SVG and PNG.")
