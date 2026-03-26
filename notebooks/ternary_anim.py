# methods/method_utils/ternary_anim.py
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
from pathlib import Path



class TernaryAnimator:
    def __init__(
        self,
        method,
        out_dir='anim_frames',
        dpi=150,
        hex_gridsize=24,
        hex_cmap='viridis',
        corner_labels=('Airplanes', 'Birds', 'Automobiles'),
        density_norm='sqrt',
        density_vmin=0.0,
        density_vmax=None,
        density_vmax_quantile=99.0,
    ):
        self.method = method
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.hex_gridsize = int(hex_gridsize)
        self.hex_cmap = hex_cmap
        self.corner_labels = corner_labels
        self.density_norm = density_norm
        self.density_vmin = float(density_vmin)
        self.density_vmax = None if density_vmax is None else float(density_vmax)
        self.density_vmax_quantile = float(density_vmax_quantile)
        self.frame_idx = 0
        self.frames = []
        self.hb = None
        self.cbar = None
        self.norm = None

        # figure + static elements
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        y_max = math.sqrt(3) / 2.0
        verts = np.array([[0.0,0.0],[1.0,0.0],[0.5,y_max],[0.0,0.0]])
        self.ax.plot(verts[:,0], verts[:,1], color='black', linewidth=1, zorder=1)

        left_label, right_label, top_label = self.corner_labels
        self.ax.text(0.0, -0.025, left_label, ha='left', va='top', fontsize=10)
        self.ax.text(1.0, -0.025, right_label, ha='right', va='top', fontsize=10)
        self.ax.text(0.5, y_max + 0.006, top_label, ha='center', va='bottom', fontsize=10)

        # decision boundaries (midpoints to centroid)
        mid_bary = np.array([[0.0,0.5,0.5],[0.5,0.0,0.5],[0.5,0.5,0.0]])
        centroid = np.array([[1/3,1/3,1/3]])
        mid_cart = barycentric_to_cartesian(mid_bary)
        centroid_cart = barycentric_to_cartesian(centroid)[0]
        for m in mid_cart:
            self.ax.plot([m[0], centroid_cart[0]], [m[1], centroid_cart[1]], color='white', linewidth=1.2, zorder=4)

        # Clip density artist to simplex so hexes are flush with boundaries.
        self.simplex_clip = Polygon(verts[:-1], closed=True, transform=self.ax.transData)

        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_xlim(-0.05,1.05)
        self.ax.set_ylim(-0.05, y_max + 0.05)

    def _build_norm(self, vmax):
        vmax = max(float(vmax), max(self.density_vmin + 1e-8, 1e-8))
        norm_name = str(self.density_norm).lower()

        if norm_name == 'linear':
            return mcolors.Normalize(vmin=self.density_vmin, vmax=vmax, clip=True)
        if norm_name == 'log':
            # Log needs strictly positive vmin.
            vmin = max(self.density_vmin, 1e-3)
            vmax = max(vmax, vmin + 1e-8)
            return mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        # Default to sqrt scaling for stronger low-density contrast.
        return mcolors.PowerNorm(gamma=0.5, vmin=self.density_vmin, vmax=vmax, clip=True)

    def _ensure_norm_from_counts(self, counts):
        if self.norm is not None:
            return
        counts = np.asarray(counts, dtype=float)
        if counts.size == 0:
            vmax = 1.0
        elif self.density_vmax is not None:
            vmax = self.density_vmax
        else:
            vmax = np.percentile(counts, self.density_vmax_quantile)
        self.norm = self._build_norm(vmax)

    def update(self, bary_points, step, colors=None, save_frame=True):
        """
        bary_points: (N,3) numpy array (rows sum to 1)
        colors: deprecated for hex-density mode (ignored)
        save_frame: if True save current figure to disk (used to assemble video later)
        """
        bary = np.asarray(bary_points)
        # If a 1D array of labels is passed, convert to one-hot barycentric vectors
        if bary.ndim == 1:
            labels = bary.astype(int)
            # Map labels into 3 classes (ternary). If labels exceed 2, wrap with modulo.
            labels_mod = labels % 3
            n = labels_mod.shape[0]
            bary = np.zeros((n, 3), dtype=float)
            bary[np.arange(n), labels_mod] = 1.0
        # Ensure bary is (N,3) before converting
        if bary.ndim != 2 or bary.shape[1] != 3:
            raise ValueError('bary_points must be shape (N,3) or a 1D label array')

        cart = barycentric_to_cartesian(bary)

        # Refresh hex-density layer for the current frame.
        if self.hb is not None:
            self.hb.remove()

        hex_kwargs = dict(
            gridsize=self.hex_gridsize,
            cmap=self.hex_cmap,
            mincnt=0,
            extent=(0.0, 1.0, 0.0, math.sqrt(3) / 2.0),
            linewidths=0.0,
            alpha=0.95,
            zorder=2,
        )
        if self.norm is not None:
            hex_kwargs['norm'] = self.norm

        self.hb = self.ax.hexbin(
            cart[:, 0],
            cart[:, 1],
            **hex_kwargs,
        )

        # First frame calibrates the fixed color scale.
        self._ensure_norm_from_counts(self.hb.get_array())
        self.hb.set_norm(self.norm)
        self.hb.set_clip_path(self.simplex_clip)

        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.hb, ax=self.ax, label='Point density')
        else:
            self.cbar.update_normal(self.hb)

        # Update title to show step
        try:
            self.ax.set_title(f"{self.method} -- Step: {step}", fontsize=14, pad=22)
        except Exception:
            pass

        # draw and save
        if save_frame:
            fname = str(self.out_dir / f'frame_{self.frame_idx:05d}.png')
            self.fig.savefig(fname, dpi=self.dpi, bbox_inches='tight')
            self.frames.append(fname)
            self.frame_idx += 1

    def finalize(self, out_name='anim', fps=1, cleanup=True):
        """
        Assemble saved PNG frames into an MP4. Uses imageio (and imageio-ffmpeg if available).
        Returns the filepath to the produced MP4, or None on failure.
        """
        if len(self.frames) == 0:
            print('No frames to assemble.')
            return None

        out_path = str(self.out_dir / (out_name + '.mp4'))

        # Try to use imageio (preferred)
        try:
            writer = imageio.get_writer(out_path, fps=fps, codec='libx264')
            for f in self.frames:
                img = imageio.imread(f)
                writer.append_data(img)
            writer.close()
            print('Saved', out_path)
            if cleanup:
                for f in self.frames:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            return out_path
        except Exception as e:
            print('imageio failed to write MP4:', e)

        # Fallback: try using matplotlib.animation.FFMpegWriter by re-playing images
        try:
            try:
                import imageio_ffmpeg as iioff
                matplotlib.rcParams['animation.ffmpeg_path'] = iioff.get_ffmpeg_exe()
            except Exception:
                pass
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            fig2, ax2 = plt.subplots(figsize=(6,6))
            ax2.axis('off')
            with writer.saving(fig2, out_path, dpi=150):
                for f in self.frames:
                    img = imageio.imread(f)
                    ax2.imshow(img)
                    writer.grab_frame()
                    ax2.clear()
            plt.close(fig2)
            print('Saved', out_path)
            if cleanup:
                for f in self.frames:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            return out_path
        except Exception as e:
            print('FFMpegWriter fallback failed:', e)
            return None