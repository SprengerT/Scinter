import numpy as np
import matplotlib as mpl

from .defined_plots import defined_plots 

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

def set_figure(textsize,labelsize):
    pgf_with_pdflatex = {                   # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
#    "font.family": "serif",
#    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
#    "font.sans-serif": [],
#    "font.monospace": [],
	"font.size": labelsize,
    "axes.linewidth": 0.5,                
    "axes.labelsize": labelsize,               # LaTeX default is 10pt font. 
    "axes.titlesize": labelsize,
    "patch.linewidth": 0.5,		# Width of box around legend
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0.3,
    #"text.fontsize": 8, 
    "legend.fontsize": textsize,
	"legend.edgecolor": "black",
	"legend.borderpad": 0.3,			# width of whitespace between text and border (in units of times linewidth)
    "xtick.labelsize": textsize,
    "ytick.labelsize": textsize,
    "figure.figsize": TeX_figsize(1.),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r'\usepackage{amsmath}',
        ]
    }
    mpl.rcParams.update(pgf_with_pdflatex) 

class plot(defined_plots):
#class responsible for the plotting of subplots
    
    def __init__(self,name_plot,tag,dict_subplot):
        self.name_plot = name_plot
        self.dict_subplot = dict_subplot
    
    def plot(self,list_category,list_data,figure,ax):
        for index,category in enumerate(list_category):
            exec("self.{0}(list_data[index],figure,ax)".format(category))
        ax.set_xlim([self.dict_subplot['xmin'],self.dict_subplot['xmax']])
        ax.set_ylim([self.dict_subplot['ymin'],self.dict_subplot['ymax']])
        ax.set_title(self.dict_subplot['title'])
        ax.set_xlabel(self.dict_subplot['xlabel'])
        ax.set_ylabel(self.dict_subplot['ylabel'])
        print("Successfully plotted {0}.".format(self.name_plot))
        
    def _add_specification(self,spec_name,spec_value):
        #log standard value if not yet existent and read current value
        # - delete keys if you want to set them to maximum range automatically!
        if spec_name not in self.dict_subplot:
            self.dict_subplot.update({spec_name:spec_value})
        return self.dict_subplot[spec_name]
        
        
