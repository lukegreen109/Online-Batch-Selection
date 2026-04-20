import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

def load_method_runs(exp_dir, method_name):
    """Loads all epoch .npz files for a given method from the embeddings_cache."""
    cache_dir = os.path.join(exp_dir, method_name, "sample_tracking", "embeddings_cache")
    if not os.path.exists(cache_dir):
        print(f"Error: Cache directory not found: {cache_dir}")
        return {}
        
    npz_files = glob.glob(os.path.join(cache_dir, "epoch_*.npz"))
    runs = {}
    for fpath in npz_files:
        try:
            basename = os.path.basename(fpath)
            ep_str = basename.replace("epoch_", "").replace(".npz", "")
            ep = int(ep_str)
            data = np.load(fpath)
            runs[ep] = {
                "embeddings": data["embeddings"],
                "labels": data["labels"],
                "indices": data["indices"]
            }
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")
            
    return runs

def main():
    parser = argparse.ArgumentParser(description="Animate embedding evolution side-by-side using streamed subsets.")
    parser.add_argument("--exp", required=True, help="Path to the experiment output directory (e.g., ./exp/CIFAR10_ResNet...)")
    parser.add_argument("--m1", required=True, help="First selection method to compare (e.g., Uniform)")
    parser.add_argument("--m2", required=True, help="Second selection method to compare (e.g., DivBS)")
    parser.add_argument("--out", default="embedding_evolution.mp4", help="Output MP4 filename.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the animation.")
    args = parser.parse_args()

    m1_epochs = load_method_runs(args.exp, args.m1)
    m2_epochs = load_method_runs(args.exp, args.m2)

    if not m1_epochs:
        print(f"Error: No runs found or could not load cache for method '{args.m1}'")
        return
    if not m2_epochs:
        print(f"Error: No runs found or could not load cache for method '{args.m2}'")
        return

    # Match epochs exactly between the two methods
    common_epochs = sorted(list(set(m1_epochs.keys()) & set(m2_epochs.keys())))

    if not common_epochs:
        print("Error: No intersecting epochs between the two methods.")
        return

    print(f"Found {len(common_epochs)} common epochs to animate.")

    # We need a consistent color map for the classes (0-9)
    cmap = plt.get_cmap("tab10")

    # Fast 2D reduction on the small subsets
    try:
        import umap
    except ImportError:
        print("Error: umap-learn is required to animate high-D embeddings.")
        return
        
    print("Computing 2D UMAP projections for all frames... (Should be extremely fast on the subsets)")
    
    frames_data = []
    # To stabilize the animation, we fit a single UMAP reducer on the FINAL epoch 
    # to find the "target" layout, and then transform all previous epochs into that space.
    last_ep = common_epochs[-1]
    
    # Fit M1 baseline
    if m1_epochs[last_ep]["embeddings"].shape[1] > 2:
        reducer1 = umap.UMAP(n_components=2, metric="cosine", random_state=42)
        reducer1.fit(m1_epochs[last_ep]["embeddings"])
    else:
        reducer1 = None
        
    # Fit M2 baseline
    if m2_epochs[last_ep]["embeddings"].shape[1] > 2:
        reducer2 = umap.UMAP(n_components=2, metric="cosine", random_state=42)
        reducer2.fit(m2_epochs[last_ep]["embeddings"])
    else:
        reducer2 = None

    for ep in common_epochs:
        print(f"  Projecting epoch {ep}...", end="\r")
        r1 = m1_epochs[ep]
        r2 = m2_epochs[ep]
        
        if reducer1 is not None:
            pts1 = reducer1.transform(r1["embeddings"])
        else:
            pts1 = r1["embeddings"]
            
        if reducer2 is not None:
            pts2 = reducer2.transform(r2["embeddings"])
        else:
            pts2 = r2["embeddings"]
            
        frames_data.append((ep, pts1, r1["labels"], pts2, r2["labels"]))

    print("\nProjections complete. Rendering animation...")

    def calculate_bounds(pts_list):
        all_pts = np.vstack(pts_list)
        min_x, min_y = np.percentile(all_pts, 1, axis=0) - 2
        max_x, max_y = np.percentile(all_pts, 99, axis=0) + 2
        return min_x, max_x, min_y, max_y

    pts1_list = [f[1] for f in frames_data]
    pts2_list = [f[3] for f in frames_data]
    
    m1_minx, m1_maxx, m1_miny, m1_maxy = calculate_bounds(pts1_list)
    m2_minx, m2_maxx, m2_miny, m2_maxy = calculate_bounds(pts2_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Embedding Evolution: {args.m1} vs {args.m2}", fontsize=16)

    scat1 = ax1.scatter([], [], s=8, alpha=0.8, cmap=cmap, vmin=0, vmax=9)
    scat2 = ax2.scatter([], [], s=8, alpha=0.8, cmap=cmap, vmin=0, vmax=9)

    def init():
        ax1.set_xlim(m1_minx, m1_maxx)
        ax1.set_ylim(m1_miny, m1_maxy)
        ax1.set_title(f"{args.m1}")
        ax1.axis('off')

        ax2.set_xlim(m2_minx, m2_maxx)
        ax2.set_ylim(m2_miny, m2_maxy)
        ax2.set_title(f"{args.m2}")
        ax2.axis('off')
        
        return scat1, scat2

    def update(frame_idx):
        ep, pts1, labels1, pts2, labels2 = frames_data[frame_idx]
        
        scat1.set_offsets(pts1)
        scat1.set_array(labels1)
        ax1.set_title(f"{args.m1} - Epoch {ep}")
        
        scat2.set_offsets(pts2)
        scat2.set_array(labels2)
        ax2.set_title(f"{args.m2} - Epoch {ep}")
        
        return scat1, scat2

    ani = FuncAnimation(fig, update, frames=len(frames_data), init_func=init, blit=False)
    
    print(f"Saving to {args.out}...")
    try:
        ani.save(args.out, fps=args.fps, extra_args=['-vcodec', 'libx264'])
        print("Animation cleanly saved!")
    except Exception as e:
        print(f"Warning: Failed to save mp4 ({e}). Falling back to gif...")
        ani.save(args.out.replace(".mp4", ".gif"), fps=args.fps, writer='imagemagick')
        print("Fallback animation saved as gif.")

if __name__ == '__main__':
    main()
