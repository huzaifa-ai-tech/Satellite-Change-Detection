import logging
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("satellite.charts")


def create_severity_chart(severity_stats, save_path):
    """Create a severity breakdown pie chart showing Low/Medium/High change areas."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    low = severity_stats.get("low_pixels", 0)
    med = severity_stats.get("medium_pixels", 0)
    high = severity_stats.get("high_pixels", 0)

    if low + med + high == 0:
        return None

    plt.figure(figsize=(7, 5), facecolor="#111827")
    ax = plt.gca()
    ax.set_facecolor("#111827")

    labels = []
    sizes = []
    colors = []
    if low > 0:
        labels.append(f"Low ({low:,} px)")
        sizes.append(low)
        colors.append("#fbbf24")
    if med > 0:
        labels.append(f"Medium ({med:,} px)")
        sizes.append(med)
        colors.append("#fb923c")
    if high > 0:
        labels.append(f"High ({high:,} px)")
        sizes.append(high)
        colors.append("#ef4444")

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"color": "#e2e8f0", "fontsize": 12},
    )
    for t in autotexts:
        t.set_color("#111827")
        t.set_fontweight("bold")

    ax.set_title("Change Severity Distribution", fontsize=16, fontweight="bold", color="#e2e8f0")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight", facecolor="#111827")
    plt.close()
    logger.info("Severity chart saved: %s", save_path)
    return str(save_path)


def create_class_chart(objects, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for obj in objects:
        class_name = obj.get("class_name", "Unknown")
        counts[class_name] += 1

    plt.figure(figsize=(9, 5), facecolor="#111827")
    ax = plt.gca()
    ax.set_facecolor("#111827")

    if not counts:
        plt.text(0.5, 0.5, "No Objects Detected", ha="center", va="center", fontsize=16, fontweight="bold", color="#94a3b8")
        plt.axis("off")
    else:
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]

        colors = ["#60a5fa", "#34d399", "#fbbf24", "#c084fc", "#fb923c", "#fb7185", "#94a3b8"]
        palette = (colors * (len(labels) // len(colors) + 1))[: len(labels)]
        bars = plt.barh(labels, values, color=palette)
        plt.xlabel("Number of Objects", color="#94a3b8")
        plt.ylabel("Land Class", color="#94a3b8")
        plt.title("Satellite Objects Detected", fontsize=16, fontweight="bold", color="#e2e8f0")
        plt.tick_params(colors="#94a3b8")
        plt.gca().spines["bottom"].set_color("#334155")
        plt.gca().spines["left"].set_color("#334155")
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().invert_yaxis()

        for bar, value in zip(bars, values):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(value), va="center", color="#94a3b8")

        plt.tight_layout()

    plt.savefig(str(save_path), dpi=300, bbox_inches="tight", facecolor="#111827")
    plt.close()
    logger.info("Chart saved: %s", save_path)
    return str(save_path)
