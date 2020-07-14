import numpy as np
import skimage
from skimage.measure import block_reduce
import skimage.morphology as morphology
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

class defined_plots:
    
    def colormesh(self,data,figure,ax):
        #load data
        x = data['x']
        y = data['y']
        f_xy = data['f_xy']
        
        #load specifications
        x_sampling = self._add_specification("x_sampling",1)
        y_sampling = self._add_specification("y_sampling",1)
        hist_equalize_radius = self._add_specification("hist_equalize_radius",0)
        # - axis properties
        self._add_specification("xmin",float(np.min(x)))
        self._add_specification("xmax",float(np.max(x)))
        self._add_specification("ymin",float(np.min(y)))
        self._add_specification("ymax",float(np.max(y)))
        self._add_specification("title","f(x,y)")
        self._add_specification("xlabel","x")
        self._add_specification("ylabel","y")
        self._add_specification("axis_aspect",'auto')
        
        #Preprocess data
        # - downsampling
        f_xy = block_reduce(f_xy, block_size=(x_sampling,y_sampling), func=np.mean)
        coordinates = np.array([x,x])
        coordinates = block_reduce(coordinates, block_size=(1,x_sampling), func=np.mean, cval=x[-1])
        x = coordinates[0,:]
        coordinates = np.array([y,y])
        coordinates = block_reduce(coordinates, block_size=(1,y_sampling), func=np.mean, cval=y[-1])
        y = coordinates[0,:]
        if not hist_equalize_radius==0:
            # enhance contrast
            f_xy = cv2.normalize(f_xy, dst=f_xy, alpha=0.0, beta=0.99999, norm_type=cv2.NORM_MINMAX)
            selem = morphology.disk(hist_equalize_radius)
            f_xy = skimage.filters.rank.equalize(f_xy,selem=selem)
        # - compute offsets to center pccolormesh
        offset_x = (x[1]-x[0])/2.
        offset_y = (y[1]-y[0])/2.
        
        #colormesh specifications
        cmap = self._add_specification("cmap",'viridis')
        vmin = self._add_specification("vmin",float(np.min(f_xy)))
        vmax = self._add_specification("vmax",float(np.max(f_xy)))
        
        #draw the plot
        #print(map(list, zip(*f_xy)))
        #im = ax.pcolormesh((x-offset_x),(y-offset_y),map(list, zip(*f_xy)),cmap=cmap,vmin=vmin,vmax=vmax)
        im = ax.pcolormesh((x-offset_x),(y-offset_y),np.swapaxes(f_xy,0,1),cmap=cmap,vmin=vmin,vmax=vmax)
        figure.colorbar(im, ax=ax)
        
    def curve(self,data,figure,ax):
        #load data
        N_curve = data['N']
        x = []
        y = []
        # - allow for multiple plots
        for index in range(N_curve):
            x.append(data["x{0}".format(index)])
            y.append(data["y{0}".format(index)])
        
        #load specifications
        linestyle = []
        marker = []
        markersize = []
        curve_label = []
        curve_color = []
        linewidth = []
        for index in range(N_curve):
            linestyle.append(self._add_specification("linestyle{0}".format(index),"dotted"))
            marker.append(self._add_specification("marker{0}".format(index),"o"))
            markersize.append(self._add_specification("markersize{0}".format(index),8))
            linewidth.append(self._add_specification("linewidth{0}".format(index),1))
            curve_label.append(self._add_specification("curve_label{0}".format(index),"curve"))
            curve_color.append(self._add_specification("curve_color{0}".format(index),"black"))
        # - axis properties
        self._add_specification("xmin",float(np.min(x[0])))
        self._add_specification("xmax",float(np.max(x[0])))
        self._add_specification("ymin",float(np.min(y[0])))
        self._add_specification("ymax",float(np.max(y[0])))
        self._add_specification("title","f(x,y)")
        self._add_specification("xlabel","x")
        self._add_specification("ylabel","y")
        self._add_specification("axis_aspect",'auto')
        
        for index in range(N_curve):
            ax.plot(x[index],y[index],label=curve_label[index],linestyle=linestyle[index],marker=marker[index],markersize=markersize[index],color=curve_color[index],linewidth=linewidth[index])
        # - legend
        flag_legend = self._add_specification("flag_legend",False)
        if flag_legend:
            ax.legend(loc='best')
            
    def scatter(self,data,figure,ax):
        #load data
        x = data['x']
        y = data['y']
        f_xy = data['f_xy']
        
        #load specifications
        markersize = self._add_specification("markersize",None)
        markerstyle = self._add_specification("markerstyle",'o')
        alpha = self._add_specification("alpha",1.0)
        # - axis properties
        self._add_specification("xmin",float(np.min(x)))
        self._add_specification("xmax",float(np.max(x)))
        self._add_specification("ymin",float(np.min(y)))
        self._add_specification("ymax",float(np.max(y)))
        self._add_specification("title","f(x,y)")
        self._add_specification("xlabel","x")
        self._add_specification("ylabel","y")
        self._add_specification("axis_aspect",'auto')
        
        #colormap specifications
        cmap = self._add_specification("cmap",'viridis')
        vmin = self._add_specification("vmin",float(np.min(f_xy)))
        vmax = self._add_specification("vmax",float(np.max(f_xy)))
            
        im = ax.scatter(x,y,c=f_xy,cmap=cmap,vmin=vmin,vmax=vmax,linewidths=markersize,alpha=alpha,marker=markerstyle)
        color_bar = figure.colorbar(im, ax=ax)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        
        
    def cmap_curve(self,data,figure,ax):
        #load data
        x = data['x']
        y = data['y']
        f_xy = data['f_xy']
        
        #load specifications
        linewidth = self._add_specification("linewidth",1)
        # - axis properties
        self._add_specification("xmin",None)
        self._add_specification("xmax",None)
        self._add_specification("ymin",None)
        self._add_specification("ymax",None)
        self._add_specification("title","f(x,y)")
        self._add_specification("xlabel","x")
        self._add_specification("ylabel","y")
        self._add_specification("axis_aspect",'auto')
        
        #colormap specifications
        cmap = self._add_specification("cmap",'viridis')
        vmin = self._add_specification("vmin",float(np.min(f_xy)))
        vmax = self._add_specification("vmax",float(np.max(f_xy)))
        
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(vmin, vmax)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        # Set the values used for colormapping
        lc.set_array(f_xy)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        figure.colorbar(line, ax=ax)