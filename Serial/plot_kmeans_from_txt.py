import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# Matches: Cluster N : (c1, c2, ..., cD) --> (p1...), (p2...), ... [then next Cluster...]
CLUSTER_BLOCK = re.compile(
    r"Cluster\s+(\d+)\s*:\s*\(([^)]*)\)\s*-->\s*(.*?)(?=(?:\s*Cluster\s+\d+\s*:)|\Z)",
    re.DOTALL
)

def parse_file(path: Path):
    text = path.read_text().strip()
    clusters = {}     # id -> list of points (tuples)
    centroids = {}    # id -> centroid tuple
    detected_dim = None

    for m in CLUSTER_BLOCK.finditer(text):
        cid = int(m.group(1))
        cstr = m.group(2).strip()
        pstr = m.group(3).strip()

        c_vals = tuple(float(x.strip()) for x in cstr.split(","))
        centroids[cid] = c_vals
        if detected_dim is None:
            detected_dim = len(c_vals)

        pts = []
        for grp in re.findall(r"\(([^)]*)\)", pstr):
            parts = [p.strip() for p in grp.split(",")]
            try:
                vals = tuple(float(p) for p in parts)
                pts.append(vals)
            except ValueError:
                pass
        clusters[cid] = pts

    return clusters, centroids, detected_dim

def main():
    ap = argparse.ArgumentParser(description="3D plot of clustered points with centroids (same color).")
    ap.add_argument("--file", "-f", default="clustering_result.txt", help="Path to output file")
    ap.add_argument("--xi", type=int, default=0, help="x-axis dim index (0-based)")
    ap.add_argument("--yi", type=int, default=1, help="y-axis dim index (0-based)")
    ap.add_argument("--zi", type=int, default=2, help="z-axis dim index (0-based)")
    ap.add_argument("--limit", type=int, default=0, help="plot first N points per cluster (0=all)")
    ap.add_argument("--elev", type=float, default=20.0, help="elevation angle")
    ap.add_argument("--azim", type=float, default=-60.0, help="azimuth angle")
    ap.add_argument("--save", type=str, default="", help="optional path to save PNG instead of showing")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    clusters, centroids, D = parse_file(path)
    if not clusters:
        print("No clusters parsed. Check file format.")
        return

    need = max(args.xi, args.yi, args.zi)
    if D is None or need >= D:
        print(f"Chosen axes out of range. Detected DIM={D}, requested indices: {args.xi},{args.yi},{args.zi}.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Stable color map: cluster id -> color
    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', []) or [f"C{i}" for i in range(10)]
    ccolor = {cid: palette[i % len(palette)] for i, cid in enumerate(sorted(clusters.keys()))}

    # Plot clusters
    for cid in sorted(clusters.keys()):
        pts = clusters[cid]
        if args.limit > 0:
            pts = pts[:args.limit]
        col = ccolor[cid]
        if pts:
            xs = [p[args.xi] for p in pts]
            ys = [p[args.yi] for p in pts]
            zs = [p[args.zi] for p in pts]
            ax.scatter(xs, ys, zs, s=12, alpha=0.9, label=f"Cluster {cid}", color=col, depthshade=True)

        c = centroids.get(cid)
        if c:
            ax.scatter(c[args.xi], c[args.yi], c[args.zi],
                       marker='X', s=180, color=col,
                       edgecolors='black', linewidths=1.0, depthshade=False)

    ax.set_title(f"K-Means Clusters 3D (DIM={D}) — x=dim{args.xi}, y=dim{args.yi}, z=dim{args.zi}")
    ax.set_xlabel(f"dim {args.xi}")
    ax.set_ylabel(f"dim {args.yi}")
    ax.set_zlabel(f"dim {args.zi}")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
    else:
        plt.show()

if __name__ == "__main__":
    main()
