"""Build the paper's learning and emforge execution architecture diagrams.

The diagrams express module responsibilities and data dependencies, not timing
or measured worker utilization.
"""

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
    fig, ax = plt.subplots(figsize=(10.2, 4.35))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.94, bottom=0.06)
    ax.set(xlim=(0, 13.4), ylim=(0, 5.4))
    ax.axis("off")

    ax.text(0.45, 5.02, "BATCHWISE SURROGATE LEARNING", color=TEAL_DARK, fontsize=13.5, weight="bold")

    # The researcher and AI agent form executable proposal rules. They do not
    # appear as independent geometry generators or mandatory per-batch calls.
    def main_box(x, w, title, detail=None, *, model=False):
        face = BLUE_PALE if model else TEAL_PALE
        edge = BLUE if model else TEAL
        rounded_box(ax, x, 2.88, w, 1.45, face=face, edge=edge, linewidth=1.5)
        title_y = 3.96 if detail else 3.63
        title_bounds = (
            (x + 0.08, 3.69, w - 0.16, 0.54)
            if detail
            else (x + 0.08, 3.00, w - 0.16, 1.25)
        )
        bounded_text(
            ax,
            title_bounds,
            x + w / 2,
            title_y,
            title,
            ha="center",
            va="center",
            color=INK,
            fontsize=12.8,
            weight="bold",
            linespacing=0.95,
            zorder=5,
        )
        if detail:
            bounded_text(
                ax,
                (x + 0.10, 2.94, w - 0.20, 0.68),
                x + w / 2,
                3.28,
                detail,
                ha="center",
                va="center",
                color=INK,
                fontsize=11.8,
                linespacing=0.95,
                zorder=5,
            )

    main_box(0.48, 2.23, "Strategy\nformation", "Researcher +\nAI agent", model=True)
    main_box(3.12, 2.05, "Candidate\npool", "Structural ops.\nRecombine\nSample")
    main_box(5.73, 2.42, "SM-guided\nselection", model=True)
    main_box(8.71, 1.84, "EM\nevaluation")
    main_box(11.11, 1.84, "Compatible\ndata")

    arrow(ax, (2.71, 3.63), (3.12, 3.63))
    arrow_label(ax, 2.91, 4.58, "executable rules", color=BLUE, size=11.8)
    arrow(ax, (5.17, 3.63), (5.73, 3.63))
    arrow(ax, (8.15, 3.63), (8.71, 3.63))
    arrow(ax, (10.55, 3.63), (11.11, 3.63))

    # Keep model feedback local so the forward path remains visually primary.
    rounded_box(ax, 6.04, 1.18, 2.10, 0.88, face=BLUE_PALE, edge=BLUE, linewidth=1.45)
    bounded_text(
        ax,
        (6.20, 1.30, 1.78, 0.58),
        7.09,
        1.59,
        "SM update",
        ha="center",
        va="center",
        color=INK,
        fontsize=12.8,
        weight="bold",
        zorder=5,
    )
    poly_arrow(ax, [(12.03, 2.88), (12.03, 2.45), (7.09, 2.45), (7.09, 2.06)], color=BLUE, width=1.55)
    poly_arrow(ax, [(6.79, 2.06), (6.79, 2.44), (6.94, 2.44), (6.94, 2.88)], color=BLUE, width=1.55)
    arrow_label(ax, 9.72, 2.45, "training data", color=BLUE, size=11.8)

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


if __name__ == "__main__":
    build_learning_loop()
    build_emforge_architecture()
    print("Wrote architecture and emforge-architecture as SVG and PNG.")
