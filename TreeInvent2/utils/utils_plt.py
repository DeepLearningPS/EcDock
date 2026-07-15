import numpy as np
import copy
from tqdm import tqdm 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

def Define_box_grid_figure(datadict,figtext={},grids=(2,1),size=(12,12),filename='plot.png',wspace=0.3,hspace=0.2):
    #datadict={"TreeINVENT":
    #               {
    #                   "data":{"Tree-INVENT":FcMol_max_ring_size,},
    #                   "xlabel":"Max ring size",
    #                   "ylabel":"Distribution",
    #                   "style":"distplot",
    #                   "xlim":(0,15),
    #                   "ylim":(0,3.0),
    #                   "legend":True
    #               }
    # }
    def set_ax_frame(ax):
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        return
    print (datadict)
    clist='abcdefghijklmnopqrstuvwxyz'
    colors=['red','green','orange','blue','pink','purple','yellow']
    plt.rc('font',size=12)
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    figure=plt.figure(figsize=size)
    xgrids=grids[0]
    ygrids=grids[1]

    gs=gridspec.GridSpec(xgrids,ygrids)
    ndatas=len(datadict.keys())
    for pid,key in enumerate(datadict.keys()):
        xid=pid//ygrids
        yid=pid%ygrids
        ax=plt.subplot(gs[xid,yid])
        subdatadict=datadict[key]
        if 'xlabel' in datadict[key].keys():
            xlabel=datadict[key]['xlabel']
        else:
            xlabel=''
        if 'ylabel' in datadict[key].keys():
            ylabel=datadict[key]['ylabel']
        else:
            ylabel=''
        style=datadict[key]['style']

        if 'xlim' in datadict[key].keys():
            xlim=datadict[key]['xlim']
        else:
            xlim=None
        if 'ylim' in datadict[key].keys():
            ylim=datadict[key]['ylim']
        else:
            ylim=None
        if 'xticks' in datadict[key].keys():
            xticks=np.arrange(*datadict[key]['xticks'])
        else:
            xticks=None
        if 'yticks' in datadict[key].keys():
            yticks=np.arrange(*datadict[key]['yticks'])
        else:
            yticks=None 
        if 'ylabel' in datadict[key].keys():
            ylabel=datadict[key]['ylabel']
        else:
            ylabel=''
        if 'xlabel' in datadict[key].keys():
            xlabel=datadict[key]['xlabel']
        else:
            xlabel=''
        if 'legend' in datadict[key].keys():
            iflegend=datadict[key]['legend']
        else:
            iflegend=False
        if 'xticks_rotation' in datadict[key].keys():
            xticks_rotation=datadict[key]['xticks_rotation']
        else:
            xticks_rotation=0
        if 'yticks_rotation' in datadict[key].keys():
            yticks_rotation=datadict[key]['yticks_rotation']
        else:
            yticks_rotation=0
        if "markersize" in datadict[key].keys():
            markersize=datadict[key]["markersize"]
        else:
            markersize=8
        char=clist[pid]
        for did,subkey in enumerate(subdatadict['data'].keys()):
            if 'pointline' in style:
                plt.plot(subdatadict['data'][subkey][0],subdatadict['data'][subkey][1],label=subkey,color=colors[did],marker='o',linewidth=2,markersize=markersize)
            if 'scatter' in style:
                plt.plot(subdatadict['data'][subkey][0],subdatadict['data'][subkey][1],label=subkey,color=colors[did],marker='o',markersize=markersize)
            if 'regplot' in style:
                sns.regplot(ax=ax,x=subdatadict['data'][subkey][0],y=subdatadict['data'][subkey][1],label=subkey,color=colors[did],scatter=True)
            if 'distplot' in style:
                sns.distplot(subdatadict['data'][subkey],ax=ax,bins=50,label=subkey,hist_kws={},color=colors[did],kde=True,kde_kws={"shade":True})
            if 'bar' in style:
                plt.bar(subdatadict['data'][subkey][0],height=subdatadict['data'][subkey][1],align="edge")
            if iflegend:
                leg=plt.legend(fancybox=True,framealpha=0,fontsize=12,markerscale=0.5)
        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)

        if xticks:
            plt.xticks(xticks,rotation=xticks_rotation)
        else:
            plt.xticks(rotation=xticks_rotation,horizontalalignment='right',verticalalignment='top')
        if yticks:
            plt.yticks(yticks,rotation=yticks_rotation)
        else:
            plt.yticks(rotation=yticks_rotation)

        plt.tick_params(length=5,top=True,bottom=True,left=True,right=True)
        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)
        if 'text' in datadict[key].keys():
            for textkey in datadict[key]['text'].keys():
                plt.text(datadict[key]['text'][textkey][0],datadict[key]['text'][textkey][1],s=textkey,transform=ax.transAxes, fontsize=12)
        #plt.text(0.1,0.85,s=f'({clist[pid]})',transform=ax.transAxes,fontsize=12)
        set_ax_frame(ax)
    for key in figtext.keys():
        pos=figtext[key]
        figure.text(pos[0],pos[1],key,fontsize=16,rotation=pos[2])
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    #plt.savefig(filename)
    #plt.show()
    figure.savefig(filename,format='png',dpi=300)