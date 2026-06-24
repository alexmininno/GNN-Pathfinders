"""
Reusable JHEP-style plotting utilities for LaTeX-ready PDF figures.

Mirrors Mathematica's standardoptions2D convention:
    - Times New Roman font family, 11 pt
    - Figure width = 72 * Intextwidth (points → inches at 72 dpi)
    - No axis labels or titles (added later via tikzpicture)
    - Three-colour palette: prcolor / seccolor / tercolor

Usage:
    from plot_style import JHEPPlot

    jp = JHEPPlot()                       # defaults
    fig, ax = jp.create_figure()          # sized for half-textwidth
    ax.plot(x, y, color=jp.prcolor)
    jp.save("my_plot.pdf")
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class InterceptJP:
    """Wrapper around JHEPPlot to handle saving standard and _045 variations correctly."""
    def __init__(self, raw_jp, is_045=False, output_dir=None, baseline_label=None, make_pdf=False):
        self.raw_jp = raw_jp
        self.is_045 = is_045
        self.output_dir = output_dir
        self.baseline_label = baseline_label
        self.make_pdf = make_pdf

    def create_figure(self, *args, **kwargs):
        return self.raw_jp.create_figure(*args, **kwargs)

    def add_legend(self, ax=None, *args, **kwargs):
        self.raw_jp.add_legend(ax, *args, **kwargs)
        if self.baseline_label:
            if ax is None:
                ax = self.raw_jp._ax if not isinstance(self.raw_jp._ax, np.ndarray) else self.raw_jp._ax.flat[0]
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    if text.get_text() == 'BFS Baseline' or text.get_text() == 'LCA Baseline':
                        text.set_text(self.baseline_label)

    def save(self, path, **kwargs):
        if not self.make_pdf:
            # Only save the standard png
            if not self.is_045:
                save_path = path + ".png"
                self.raw_jp.save(save_path, **kwargs)
            return

        # PDF saving path
        if self.is_045:
            # When drawing the 0.45 version, we override the plot width
            # to be 0.45 of textwidth (since JHEP has two columns optionally)
            old_width = self.raw_jp._width
            old_height = self.raw_jp._height
            self.raw_jp._width = self.raw_jp.intextwidth * 0.45
            self.raw_jp._height = self.raw_jp._width * self.raw_jp.aspect_ratio
            if self.raw_jp._fig is not None:
                self.raw_jp._fig.set_size_inches(self.raw_jp._width, self.raw_jp._height)
            
            save_path = path + "_045.pdf"
            self.raw_jp.save(save_path, **kwargs)
            
            # restore
            self.raw_jp._width = old_width
            self.raw_jp._height = old_height
        else:
            save_path = path + ".pdf"
            self.raw_jp.save(save_path, **kwargs)
class JHEPPlot:
    """
    Encapsulates the JHEP journal figure style.

    Parameters
    ----------
    intextwidth : float
        Width in inches (Mathematica convention: 72 pts/inch with
        ImageSize -> {72 * Intextwidth, Automatic}).
        Default 6.6155 in ≈ full JHEP \textwidth.
    aspect_ratio : float
        height / width.  Default golden ratio inverse ≈ 0.618.
    fontsize : int
        Base font size in points.  Default 11.
    usetex : bool
        If True, render all text with LaTeX (requires a TeX installation).
        Default True.
    """

    # ── colour palette ──────────────────────────────────────────────
    prcolor  = (29 / 255, 53 / 255, 87 / 255)    # dark navy
    seccolor = (69 / 255, 123 / 255, 157 / 255)  # steel blue
    tercolor = (152 / 255, 193 / 255, 217 / 255) # pale blue

    # Extended palette for multi-curve plots (physics constraints)
    palette = {
        'reward':    (0 / 255,   0 / 255,   0 / 255),   # black
        'anomaly':   (29 / 255,  53 / 255,  87 / 255),   # dark navy  (prcolor)
        'stability': (69 / 255, 123 / 255, 157 / 255),   # steel blue (seccolor)
        'sum':       (230 / 255, 57 / 255,  70 / 255),   # coral red
        'range':     (168 / 255, 77 / 255, 153 / 255),   # muted purple
        'pair':      (42 / 255, 157 / 255, 143 / 255),   # teal
        'nontrivial':(244 / 255, 162 / 255, 97 / 255),   # warm amber
        'bounds':    (152 / 255, 193 / 255, 217 / 255),  # pale blue  (tercolor)
        'diversity': (128 / 255, 128 / 255, 128 / 255),  # grey
        'found':     (201 / 255, 148 / 255, 21 / 255),   # gold
    }

    # Line-style presets: continuous scores = solid, binary scores = dashed
    STYLE_CONT = {'linestyle': '-',  'linewidth': 1.2, 'alpha': 0.9}
    STYLE_BIN  = {'linestyle': '--', 'linewidth': 0.9, 'alpha': 0.7}
    STYLE_BOLD = {'linestyle': '-',  'linewidth': 1.6, 'alpha': 1.0}

    def __init__(
        self,
        intextwidth: float = 6.6155,
        aspect_ratio: float = 0.618,
        fontsize: int = 11,
        usetex: bool = True,
    ):
        self.intextwidth = intextwidth
        self.aspect_ratio = aspect_ratio
        self.fontsize = fontsize
        self.usetex = usetex

        self._width = intextwidth            # inches
        self._height = intextwidth * aspect_ratio

        # ── backend selection ───────────────────────────────────────
        # matplotlib.use() must be called before pyplot import; if pyplot
        # was already imported (e.g. by the caller), silently skip.
        try:
            if usetex:
                matplotlib.use("pgf")
            else:
                matplotlib.use("Agg")
        except Exception:
            pass  # backend already locked by prior pyplot import

        # ── global rcParams ─────────────────────────────────────────
        rc = {
            # --- font ---
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "STIX"],
            "font.size": fontsize,
            # --- axes ---
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "axes.linewidth": 0.6,
            # --- ticks ---
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            # --- legend ---
            "legend.fontsize": fontsize - 1,
            "legend.frameon": False,
            # --- figure ---
            "figure.figsize": (self._width, self._height),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }

        if usetex:
            rc.update({
                "text.usetex": True,
                "pgf.texsystem": "lualatex",
                "pgf.rcfonts": False,
                "pgf.preamble": "\n".join([
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{amsmath}",
                    r"\usepackage{amsfonts}",
                    r"\usepackage{amssymb}",
                    r"\usepackage{mathtools}",
                    r"\usepackage{amsthm}",
                    r"\usepackage{physics}",
                ]),
                "text.latex.preamble": "\n".join([
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{amsmath}",
                    r"\usepackage{amsfonts}",
                    r"\usepackage{amssymb}",
                    r"\usepackage{mathtools}",
                    r"\usepackage{amsthm}",
                    r"\usepackage{physics}",
                ]),
            })
        else:
            rc["text.usetex"] = False

        plt.rcParams.update(rc)

        self._fig = None
        self._ax = None

    # ── public helpers ──────────────────────────────────────────────

    def create_figure(self, nrows=1, ncols=1, **kwargs):
        """
        Return (fig, ax) with the correct size.
        Extra kwargs are forwarded to plt.subplots.
        """
        self._fig, self._ax = plt.subplots(
            nrows, ncols,
            figsize=(self._width, self._height),
            **kwargs,
        )
        # Remove labels/titles — user adds them in LaTeX
        if isinstance(self._ax, np.ndarray):
            for a in self._ax.flat:
                a.set_xlabel("")
                a.set_ylabel("")
                a.set_title("")
        else:
            self._ax.set_xlabel("")
            self._ax.set_ylabel("")
            self._ax.set_title("")
        return self._fig, self._ax

    def add_legend(self, ax=None, loc='best', ncols=1, ncol=None, **kwargs):
        """
        Add a JHEP-styled legend to *ax* (defaults to the last created axis).

        The legend is frameless, uses the document fontsize − 1, and
        respects the current TeX setting.  Extra kwargs are forwarded
        to ax.legend().
        """
        if ncol is not None:
            ncols = ncol
        if ax is None:
            ax = self._ax if not isinstance(self._ax, np.ndarray) else self._ax.flat[0]
        ax.legend(
            loc=loc,
            ncols=ncols,
            frameon=False,
            fontsize=self.fontsize - 1,
            handlelength=2.0,
            columnspacing=1.0,
            **kwargs,
        )

    def adjust_figsize_for_readability(self):
        """Dynamically increase figure size if there are many ticks (e.g. for heatmaps) or crowded x-labels."""
        if self._fig is None:
            return
            
        for ax in self._fig.axes:
            num_xticks = len(ax.get_xticklabels())
            num_yticks = len(ax.get_yticklabels())
            
            h_factor = 0.25
            
            # Detect if it is a heatmap to avoid blowing up regular line/scatter plots
            is_heatmap = any(isinstance(c, matplotlib.collections.QuadMesh) for c in ax.collections)
            has_thick_border = any(getattr(p, 'get_linewidth', lambda: 0)() > 1.5 for p in ax.patches)
            
            # If it's not a quadmesh and doesn't have many texts (one per cell) and no thick borders, skip
            if not is_heatmap and len(ax.texts) < (num_xticks * num_yticks) * 0.5 and not has_thick_border:
                continue
                
            if has_thick_border:
                h_factor = 0.45
                
            # Only calculate minimum height. If we increase width, LaTeX will scale the whole
            # image down to fit \textwidth, which shrinks our 11pt fonts to unreadable sizes!
            min_h = num_yticks * h_factor * (self.fontsize / 9.0)
            
            curr_w, curr_h = self._fig.get_size_inches()
            new_h = max(curr_h, min_h)
            
            if new_h > curr_h:
                self._fig.set_size_inches(curr_w, new_h)

    def save(self, path: str, **kwargs):
        """Save the current figure to *path* (typically .pdf)."""
        if self._fig is None:
            raise RuntimeError("No figure has been created yet.")
        self.adjust_figsize_for_readability()
        self._fig.savefig(path, **kwargs)
        plt.close(self._fig)
        print(f"Saved → {path}")

    # ── convenience colour accessors ────────────────────────────────

    @property
    def colors(self):
        """Return (prcolor, seccolor, tercolor) as a tuple."""
        return (self.prcolor, self.seccolor, self.tercolor)

def get_latex_name(name: str) -> str:
    """Format string name into a LaTeX mathematical representation."""
    import re
    if not name or name == "Unknown":
        return name
    if name.startswith("CC3/ZZ"):
        n = name.replace("CC3/ZZ", "")
        return f"$\\mathbb{{C}}^3/\\mathbb{{Z}}_{{{n}}}$"
    elif name.startswith("C(dP"):
        n = name.replace("C(dP", "").replace(")", "")
        return f"$C(dP_{{{n}}})$"
    elif name == "C(Y2,1)":
        return "$C(dP_1)$"
    elif name.startswith("C(Y"):
        pq = name.replace("C(Y", "").replace(")", "")
        return f"$C(Y^{{{pq}}})$"
    elif name == "CC3/Delta27":
        return "$\\mathbb{C}^3/\\Delta_{27}$"
    elif re.match(r"^D\d+$", name):
        n = name[1:]
        return f"$D_{{{n}}}$"
    elif name == "Q4":
        return "$C(Y^{2,0})$"
    elif name == "Q9":
        return "$\\mathbb{C}^3/\\mathbb{Z}_3$"
    elif re.match(r"^Q\d+$", name):
        n = name[1:]
        return f"$\\mathbf{{Q{n}}}$"
    elif re.match(r"^A\d+$", name):
        n = name[1:]
        return f"$A_{{{n}}}$"
    return name

