# if this script is called standalone
import numpy as np
import matplotlib as mpl

def TeX_figsize(xscale,yscale=0.):
	fig_width_pt = 440.
	fig_height_pt = 642.
	inches_per_pt = 1./72.27
	fig_width = fig_width_pt*inches_per_pt*xscale
	if yscale == 0.:
		goldenratio = (np.sqrt(5.)-1.)/2.
		fig_height = fig_width*goldenratio
	else:
		fig_height = fig_height_pt*inches_per_pt*yscale
	figsize = [fig_width,fig_height]
	return figsize

pgf_with_pdflatex = {                   # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
	"font.size": 16, #36, #
    "axes.linewidth": 0.5,                
    "axes.labelsize": 16, #36, #                # LaTeX default is 10pt font. 
    "axes.titlesize": 16, #36, #
    "patch.linewidth": 0.5,		# Width of box around legend
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0.3,
    #"text.fontsize": 8, 
    "legend.fontsize": 14, #24, #
	"legend.edgecolor": "black",
	"legend.borderpad": 0.3,			# width of whitespace between text and border (in units of times linewidth)
    "xtick.labelsize": 14, #24, #
    "ytick.labelsize": 14, #24, #
    "figure.figsize": TeX_figsize(1.),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r'\usepackage{amsmath}',
        ]
    }
mpl.rcParams.update(pgf_with_pdflatex) 
