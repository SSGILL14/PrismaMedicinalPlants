"""Reproduce the ten data figures for the systematic review of 83 studies."""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import textwrap

import matplotlib

matplotlib.use('Agg')

from matplotlib import colors, font_manager
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch, Wedge
from matplotlib.path import Path as PlotPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageCms


# Chart drawing

AXIS = "#000000"
INK = "#282837"
MUTED = "#5f5c6d"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "mAP", "IoU", "AUC", "Kappa"]


PALE=['#75b6a7','#a291cb','#dc978a','#80acce','#d1ab65','#9fb783','#c98fb8','#92b5bf','#d59d7b']

TEXT=['#28695b','#644983','#8d4e48','#35668a','#7c5925','#4a6534','#834867','#3a616c','#885339']

SOFT = ['#8ab9ac', '#b2a3d3', '#d9a094', '#92b6d5', '#d2b478', '#a9b994', '#cc9cbc', '#99b6bb', '#d4a787']

DARK = ['#3f786e', '#6f538c', '#985c52', '#4b7296', '#84652f', '#5d7552', '#8f537d', '#537980', '#8d6152']

PALETTE = {
    'rose':    {300: '#fda4af', 400: '#fb7185', 500: '#f43f5e'},
    'pink':    {300: '#f9a8d4', 400: '#f472b6', 500: '#ec4899'},
    'fuchsia': {300: '#f0abfc', 400: '#e879f9', 500: '#d946ef'},
    'purple':  {300: '#d8b4fe', 400: '#c084fc', 500: '#a855f7'},
    'blue':    {300: '#93c5fd', 400: '#60a5fa', 500: '#3b82f6'},
    'sky':     {300: '#7dd3fc', 400: '#38bdf8', 500: '#0ea5e9'},
    'green':   {300: '#86efac', 400: '#4ade80', 500: '#22c55e'},
    'yellow':  {300: '#fde047', 400: '#facc15', 500: '#eab308'},
    'orange':  {300: '#fdba74', 400: '#fb923c', 500: '#f97316'},
    'red':     {300: '#fca5a5', 400: '#f87171', 500: '#ef4444'},
    'teal':    {300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6'},
}

PERIODS = {2021: 'sky', 2022: 'sky', 2023: 'sky', 2024: 'purple', 2025: 'orange', 2026: 'teal'}

RADIAL_SOURCES = ('teal', 'purple', 'blue', 'orange', 'pink', 'green', 'fuchsia', 'red', 'sky', 'yellow',
                  'purple', 'teal', 'rose', 'blue')

def tones(families, level=500):
    return [PALETTE[family][level] for family in families]

MODEL_DISPLAY = {
    'P27':'ResNet / GoogLeNet / ViT-B16',
    'P30':'DeepSeqCoco hybrid',
    'P43':'HGA–PSO',
    'P56':'DenseNet-201 / AlexNet / MobileNetV2',
    'P71':'LeafFusionNet',
    'P89':'PhytoV2Net / InceptionV3',
    'P90':'HSS-GNN-HHLF',
    'P98':'ViT–U-Net fusion',
}

def axis_only(ax, sides=('left','bottom')):
    ax.grid(False)
    for side, spine in ax.spines.items():
        spine.set_visible(side in sides)
        spine.set_linewidth(.7)
        spine.set_color(AXIS)
    ax.tick_params(length=0, pad=8)

def blend(a,b,t):
    a=np.array(colors.to_rgb(a))
    b=np.array(colors.to_rgb(b))
    return a+(b-a)*np.asarray(t)[...,None]

def paint(ax, patch, bounds, tone, *, mode='vertical', z=2, highlight=None):
    """Clip tonal shading to the exact footprint of a data mark."""
    if mode=='radial':
        xx,yy=np.meshgrid(np.linspace(-1,1,170),np.linspace(-1,1,170))
        light=np.clip((1-yy)*.30 + (1-xx)*.12,0,1)
        tex=blend(tone,'#ffffff' if highlight is None else highlight,
                  light*.48 if highlight is None else light/.84)
    else:
        light=np.linspace(.53,.09,70) if highlight is None else np.linspace(1,0,70)
        tex=blend(tone,'#ffffff' if highlight is None else highlight,np.broadcast_to(light[:,None],(70,3)))
    patch.set(facecolor='none',edgecolor='none')
    ax.add_patch(patch)
    raster=ax.imshow(tex,extent=bounds,origin='upper',interpolation='bilinear',aspect=ax.get_aspect(),zorder=z)
    raster.set_clip_path(patch)
    return raster

def filled_bar(ax,x,y,w,h,c,*,highlight=None):
    # Round corners in display space so differently scaled axes get the same shape.
    transform=ax.transData.get_affine().get_matrix()
    aspect=abs(transform[0,0]/transform[1,1])
    radius=min(w*.40,h*.30/aspect)
    bounds=(x,x+w,y-h/2,y+h/2)
    patch=FancyBboxPatch((x,y-h/2),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                         mutation_aspect=aspect)
    return paint(ax,patch,bounds,c,highlight=highlight)

def ring(ax,r,lower,upper,tone,width,z=2,tint=False,*,highlight=None):
    if upper<=lower:
        return
    patch=Wedge((0,0),r,lower,upper,width=width)
    if tint:
        patch.set(facecolor=tone,edgecolor='none',zorder=z)
        ax.add_patch(patch)
    else:
        paint(ax,patch,(-r,r,-r,r),tone,mode='radial',z=z,highlight=highlight)

def swarm(vals, row, step_x=.40, step_y=.068):
    placed = []
    offsets = [0] + [q*step_y for k in range(1, 32) for q in (k,-k)]
    for value in sorted(vals):
        for offset in offsets:
            if all(((value-x)/step_x)**2 + ((offset-dy)/step_y)**2 >= .999 for x,dy in placed):
                placed.append((value,offset))
                break
        else:
            raise ValueError('Insufficient space for metric swarm')
    xs, ys = np.array(placed).T
    return xs, row+ys

def keyword_label(term):
    if term == 'plant disease(s)':
        return 'Plant Disease(s)'
    return term[0].upper() + term[1:]


def hive(nodes, edges, *, overlay=False, emphasize_frequent_terms=False):
    communities = {1: '#735c9a', 2: '#217b76', 3: '#ae5843'}
    year_stops = ('#395a8b', '#278c87', '#e2ad49')
    year_limits = (2024, 2026)
    fig, ax = plt.subplots(figsize=(10.7, 8.8))
    ax.set_aspect('equal')
    ax.set(xlim=(-5.25, 5.25), ylim=(-2.28, 3.87))
    ax.axis('off')
    directions = {1: np.array([0., 1.]), 2: np.array([-.866, -.5]), 3: np.array([.866, -.5])}
    positions = {}
    sides = {}
    for group, direction in directions.items():
        group_nodes = nodes[nodes.cluster == group].sort_values(['weight<Occurrences>', 'label'], ascending=[False, True])
        for term, radius in zip(group_nodes.label, np.linspace(.57, 3.31, len(group_nodes))):
            positions[term] = direction * radius
            sides[term] = group
        start, end = direction * .37, direction * 3.47
        ax.plot([start[0], end[0]], [start[1], end[1]], lw=1.1, color='#b2b8b7', zorder=1)
    ribbon_edges = edges[edges.Cooccurrences >= 4].sort_values('Cooccurrences')
    for edge in ribbon_edges.itertuples(index=False):
        a, b = positions[edge.Source], positions[edge.Target]
        group = sides[edge.Source]
        within = group == sides[edge.Target]
        if within:
            r = directions[group]
            turn = {1: 1, 2: 1, 3: -1}[group]
            perpendicular = np.array([-r[1], r[0]]) * turn
            bulge = .48 * np.linalg.norm(a-b) + .09
            ctrl1, ctrl2 = a + perpendicular * bulge, b + perpendicular * bulge
        else:
            ctrl1, ctrl2 = a * .29 + b * .09, b * .29 + a * .09
        color = communities[group] if not overlay and within else '#667c8a'
        ax.add_patch(PathPatch(PlotPath([a, ctrl1, ctrl2, b], [1, 4, 4, 4]), fill=False,
                               lw=.48 + .105 * edge.Cooccurrences, edgecolor=color,
                               alpha=.52 if within else .32, zorder=2))
    year_scale = colors.Normalize(*year_limits)
    year_palette = colors.LinearSegmentedColormap.from_list('publication', year_stops)
    for row in nodes.to_dict('records'):
        term = row['label']
        x, y = positions[term]
        group = sides[term]
        color = communities[group] if not overlay else year_palette(year_scale(row['score<Avg. pub. year>']))
        ax.scatter(x, y, s=27 + row['weight<Occurrences>']*1.95, color=color,
                   edgecolor='white', lw=.8, zorder=4)
        label_on_left = group == 2
        offset = -.14 if label_on_left else .14
        frequent = row['weight<Occurrences>'] >= 35
        if frequent:
            offset *= 1.5
        ax.text(x+offset, y, keyword_label(term), va='center', ha='right' if label_on_left else 'left',
                weight='bold' if emphasize_frequent_terms and frequent else 'normal', zorder=5)
    if not overlay:
        handles = [Line2D([0],[0], color='#667c8a', lw=.48 + .105*n, label=str(n)) for n in (4, 10, 20)]
        ax.legend(handles=handles, title='Shared studies', frameon=False, loc='upper left',
                  bbox_to_anchor=(.06,.91), ncol=1, handlelength=2.2)
    else:
        bar_ax = fig.add_axes([.32, .118, .36, .016])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=year_scale, cmap=year_palette), cax=bar_ax, orientation='horizontal')
        bar.set_ticks(np.arange(year_limits[0], year_limits[1] + .01, .5))
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, pad=4)
        bar.set_label('Mean publication year', labelpad=5)
    fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.18 if overlay else .10)
    return fig

def chord(nodes, edges):
    palette = ['#6663BD', '#239DAD', '#DF806D']
    nodes = nodes.copy()
    keep=set(nodes.label)
    edges=edges[edges.Source.isin(keep)&edges.Target.isin(keep)&(edges.Cooccurrences>=3)].copy()
    strength={s:0 for s in keep}
    for e in edges.itertuples():
        strength[e.Source]+=e.Cooccurrences;strength[e.Target]+=e.Cooccurrences
    nodes['strength']=nodes.label.map(strength)
    nodes=nodes.sort_values(['cluster','strength','label'],ascending=[True,False,True])
    labels=nodes.label.tolist();idx={s:i for i,s in enumerate(labels)}
    clusters=dict(zip(nodes.label,nodes.cluster));total=sum(strength.values());gap=np.deg2rad(3.3)
    available=2*np.pi-len(labels)*gap;angle=np.pi/2
    arcs={};cursors={};segments={}
    for s in labels:
        width=available*strength[s]/total
        arcs[s]=(angle,angle+width);cursors[s]=angle;angle+=width+gap
    # Allocate ribbon ends within each arc in the order of their destination.
    for s in labels:
        peers=[]
        for e in edges.itertuples():
            if e.Source==s:peers.append((e.Target,int(e.Cooccurrences)))
            elif e.Target==s:peers.append((e.Source,int(e.Cooccurrences)))
        for other,n in sorted(peers,key=lambda p:idx[p[0]],reverse=True):
            a=cursors[s];b=a+available*n/total
            segments[(s,other)]=(a,b);cursors[s]=b
    def point(a,r=.93):return np.array([np.cos(a)*r,np.sin(a)*r])
    def ribbon(a,b,c,d,col):
        aa=np.linspace(a,b,14);bb=np.linspace(c,d,14)
        verts=[point(aa[0])];codes=[PlotPath.MOVETO]
        for t in aa[1:]:verts.append(point(t));codes.append(PlotPath.LINETO)
        verts.extend([point(b)*.12,point(c)*.12,point(c)]);codes.extend([PlotPath.CURVE4]*3)
        for t in bb[1:]:verts.append(point(t));codes.append(PlotPath.LINETO)
        verts.extend([point(d)*.12,point(a)*.12,point(a)]);codes.extend([PlotPath.CURVE4]*3)
        verts.append(point(a));codes.append(PlotPath.CLOSEPOLY)
        return PathPatch(PlotPath(verts,codes),facecolor=col,edgecolor='white',lw=.3,alpha=.43)
    fig,ax=plt.subplots(figsize=(12.2,11.3));ax.set_aspect('equal');ax.set_xlim(-1.48,1.48);ax.set_ylim(-1.43,1.43);ax.axis('off')
    for e in edges.sort_values('Cooccurrences',ascending=False).itertuples():
        a,b=segments[(e.Source,e.Target)];c,d=segments[(e.Target,e.Source)]
        color=palette[int(clusters[e.Source])-1]
        ax.add_patch(ribbon(a,b,c,d,color))
    for s in labels:
        a,b=arcs[s];c=palette[int(clusters[s])-1]
        ax.add_patch(Wedge((0,0),1.0,np.rad2deg(a),np.rad2deg(b),width=.055,facecolor=c,edgecolor='white',lw=1))
        ax.add_patch(Wedge((0,0),1.019,np.rad2deg(a),np.rad2deg(b),width=.008,facecolor=blend(c, 'white', .52),edgecolor='none'))
        mid=(a+b)/2;degrees=np.rad2deg(mid)%360;p=point(mid,1.075)
        left=90<degrees<270
        rotation=degrees+180 if left else degrees
        ax.text(p[0],p[1],textwrap.fill(keyword_label(s),26),rotation=rotation,rotation_mode='anchor',
            ha='right' if left else 'left',va='center',color='#253149',linespacing=1.2,weight='bold')
    fig.tight_layout(pad=.6)
    return fig

def keyword_frequency(data):
    fig,ax=plt.subplots(figsize=(9.1,1.15+.32*len(data)))
    for i,row in enumerate(data.itertuples(index=False)):
        c=SOFT[min(i//4,4)]
        ax.plot([0,row.Studies],[i,i], color=c, lw=2.6,solid_capstyle='round')
        ax.scatter(row.Studies,i,s=72,color=c,edgecolor='white',lw=.6,zorder=3)
        ax.text(row.Studies+1.1,i,str(row.Studies),va='center')
    labels=['YOLO' if s=='yolo' else keyword_label(s) for s in data.Keyword]
    ax.set_yticks(range(len(data)),labels)
    ax.set(xlim=(0,max(61,data.Studies.max()+6)),ylim=(len(data)-.35,-.65),xlabel='Studies')
    ax.set_xticks([0,15,30,45,60])
    axis_only(ax)
    fig.subplots_adjust(left=.37,right=.97,top=.99,bottom=.102)
    return fig

def radial_waterfall(sources):
    fig,ax=plt.subplots(figsize=(11.7,8.75))
    counts = sources.to_numpy()
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    families = [RADIAL_SOURCES[i % len(RADIAL_SOURCES)] for i in range(len(counts))]
    palette = tones(families)
    highlights = tones(families, 400)
    history = [blend(tone, 'white', .64) for tone in tones(families, 300)]
    angles=np.linspace(120,-205,len(counts)+1)
    r=lambda n: .47+2.04*n/total
    point=lambda t,rr: (np.cos(np.deg2rad(t))*rr,np.sin(np.deg2rad(t))*rr)
    # Each equal-angle wedge ends at the running total. Its two layers retain
    # the earlier accumulation and the contribution at this step.
    for i,(n,end) in enumerate(zip(counts,cumulative)):
        tone=palette[i]
        a,b=angles[i:i+2]
        prev=end-n
        if prev:
            ring(ax,r(prev),b+.8,a-.8,history[i],r(prev)-r(0),1,True)
        ring(ax,r(end),b+.8,a-.8,tone,r(end)-r(prev),3,highlight=highlights[i])
        mid=(a+b)/2
        x,y=point(mid,2.62 if i==0 else 2.73 if i==1 else 2.85)
        if abs(x)<.16:
            align='center'
        else:
            align='right' if x<0 else 'left'
        ax.text(x,y,f'{sources.index[i]}  +{n}',ha=align,va='center')
        if i in (0,1,2,3,len(counts)-1):
            x,y=point(mid,r(end)+.13)
            ax.text(x,y,str(end),ha='center',va='center',weight='bold',
                    bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=4)
    ax.text(0,0,str(total),ha='center',va='center',weight='bold')
    ax.text(0,-.19,'mentions',ha='center',va='center',color=MUTED)
    ax.set(xlim=(-4.52,4.52),ylim=(-3.13,3.17))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=.007,right=.993,top=.988,bottom=.025)
    return fig

def distributions(studies, *, axis_weight='bold'):
    cohort_size = len(studies)
    fig=plt.figure(figsize=(12.6,7.55))
    gs=fig.add_gridspec(1,2,left=.15,right=.965,bottom=.10,top=.91,width_ratios=[5.5,1.65],wspace=.10)
    ax=fig.add_subplot(gs[0])
    cover=fig.add_subplot(gs[1],sharey=ax)
    ax.set(xlim=(49.5,100.65),ylim=(-.62,8.65),xlabel='Reported score (%)')
    cover.set(xlim=(0,cohort_size+30))
    score_axis=np.linspace(50,100,400)
    light_fills = [blend(tone, 'white', .37) for tone in PALE]
    for j,(metric,tone,dark) in enumerate(zip(METRICS,PALE,TEXT)):
        vals=studies[metric].dropna().to_numpy()
        density=np.exp(-.5*((score_axis[:,None]-vals)/1.25)**2).sum(axis=1)
        density+=np.exp(-.5*((score_axis[:,None]-(200-vals))/1.25)**2).sum(axis=1)
        height=.6*density/max(density)
        row=8-j
        base=row-.25
        ax.fill_between(score_axis,base,base+height,facecolor=light_fills[j],linewidth=0,zorder=1)
        ax.fill_between(score_axis,base,base+height*.47,facecolor=tone,linewidth=0,alpha=.68,zorder=2)
        # Stack identical dots in small increments while preserving their true scores.
        x,v=swarm(vals,base-.085,step_x=.26,step_y=.029)
        ax.scatter(x,v,s=11,color=dark,linewidth=0,alpha=.80,zorder=4)
        med=np.median(vals)
        top=base+np.interp(med,score_axis,height)
        ax.vlines(med,base,top+.045,color=dark,linewidth=1.55,zorder=4)
        filled_bar(cover,0,row,len(vals),.31,tone)
        cover.text(cohort_size+1,row,f'{len(vals)} / {cohort_size}',ha='left',va='center',color=MUTED)
    ax.set_xticks([50,60,70,80,90,100])
    ax.set_yticks(range(8,-1,-1),METRICS[:-1]+['κ × 100'],weight=axis_weight)
    cover.tick_params(left=False,labelleft=False)
    cover.set_xticks([0,cohort_size//2,cohort_size])
    cover.set_xlabel('Studies')
    ax.set_title('Distributions of reported scores',loc='left',pad=14,weight='bold')
    cover.set_title('Reporting coverage',loc='left',pad=14,weight='bold')
    axis_only(ax)
    axis_only(cover)
    cover.spines['bottom'].set_bounds(0,cohort_size)
    for panel in (ax, cover):
        panel.xaxis.label.set_weight(axis_weight)
        for tick_label in panel.get_xticklabels():
            tick_label.set_weight(axis_weight)
    return fig

def architecture(studies):
    groups = studies.Category.value_counts()
    metrics = studies.groupby('Category', observed=True)[METRICS[:4]]
    means, ns = metrics.mean(), metrics.count()
    fig,axes=plt.subplots(1,4,figsize=(13.5,5.3),sharey=True)
    for i,(ax,metric,tone,dark) in enumerate(zip(axes,means.columns,PALE,TEXT)):
        ax.set(xlim=(80,117),ylim=(5.6,-.60))
        ax.set_aspect('auto')
        for j,(value,n) in enumerate(zip(means[metric],ns[metric])):
            filled_bar(ax,80,j,value-80,.34,tone)
            ax.text(102.2,j,f'{value:.2f} ({n})',va='center')
        ax.set_xticks([80,90,100])
        ax.set_title(metric,color=dark,weight='bold',pad=13)
        axis_only(ax)
        ax.spines['bottom'].set_bounds(80,100)
        if i==0:
            ax.set_yticks(range(6),[f'{name}\n{groups[name]} studies' for name in means.index])
    fig.text(.59,.025,'Mean score (%) · contributing studies in parentheses',ha='center',color=MUTED)
    fig.subplots_adjust(left=.205,right=.995,top=.90,bottom=.15,wspace=.13)
    return fig

def ribbon(ax, left, right, source_y, target_y, height, tone, *, highlight=None):
    mid = (left + right) / 2
    vertices = [
        (left, source_y), (mid, source_y), (mid, target_y), (right, target_y),
        (right, target_y + height), (mid, target_y + height),
        (mid, source_y + height), (left, source_y + height), (left, source_y)
    ]
    codes = [PlotPath.MOVETO, *([PlotPath.CURVE4] * 3), PlotPath.LINETO,
             *([PlotPath.CURVE4] * 3), PlotPath.CLOSEPOLY]
    shape = PathPatch(PlotPath(vertices, codes), facecolor='none', edgecolor='none')
    ax.add_patch(shape)
    steps = np.linspace(.22, .56, 128) if highlight is None else np.linspace(0, 1, 128)
    texture = blend(tone, 'white' if highlight is None else highlight, np.tile(steps, (2, 1)))
    fill = ax.imshow(texture, extent=(left, right, min(source_y, target_y), max(source_y, target_y) + height),
                     aspect='auto', interpolation='bilinear', alpha=.81, zorder=2)
    fill.set_clip_path(shape)

def stack(counts, unit, minimum, gap, height):
    """Center a stack of label cards and its independently scaled attachment bands."""
    flow = np.array(counts) * unit
    heights = np.maximum(flow, minimum)
    total = heights.sum() + gap * (len(flow) - 1)
    tops = (height - total) / 2 + np.r_[0, np.cumsum(heights[:-1] + gap)]
    return tops, heights, tops + (heights - flow) / 2

def crop_sankey(links, *, crop_weight='bold'):
    periods, crops, values = list(links.index), list(links.columns), links.to_numpy()
    totals, counts = values.sum(axis=1), values.sum(axis=0)
    unit, row_space = .032, .072
    height = float(np.maximum(counts * unit, .30).sum() + row_space * (len(crops) - 1))
    st, sh, so = stack(totals, unit, .53, .45, height)
    tt, th, to = stack(counts, unit, .30, row_space, height)
    families = [PERIODS[year] for year in (2021, 2024, 2025, 2026)]
    palette, highlights = tones(families, 400), tones(families, 300)
    fig, ax = plt.subplots(figsize=(12, height + .7))
    fig.subplots_adjust(left=.012, right=.988, top=.984, bottom=.018)
    ax.set(xlim=(0, 12), ylim=(height + .11, -.48))
    ax.axis('off')
    left, right = 2.49, 9.41
    source_starts = so[:, None] + (np.cumsum(values, axis=1) - values) * unit
    target_starts = to[None, :] + (np.cumsum(values, axis=0) - values) * unit
    for n, i, j in sorted(((values[i, j], i, j) for i, j in zip(*np.nonzero(values))), reverse=True):
        ribbon(ax, left - .045, right + .045, source_starts[i, j], target_starts[i, j], n * unit,
               palette[i], highlight=highlights[i])
    for i, tone in enumerate(palette):
        y, h = st[i], sh[i]
        card = FancyBboxPatch((.15, y), left - .15, h, boxstyle='round,pad=0,rounding_size=.08')
        card.set(facecolor='none', edgecolor='none')
        ax.add_patch(card)
        # The raw tones stay within the saturated range of the color grade.
        light = highlights[i]
        steps = np.linspace(.30, .09, 240) if light is None else np.linspace(1, 0, 240)
        texture = blend(tone, 'white' if light is None else light, np.tile(steps[:, None], (1, 2)))
        fill = ax.imshow(texture, extent=(.15, left, y, y + h),
                         aspect='auto', interpolation='bilinear', zorder=3)
        fill.set_clip_path(card)
        ax.text(.35, y + h / 2, periods[i], va='center', weight='bold', zorder=4)
        ax.text(left - .19, y + h / 2, str(totals[i]), va='center', ha='right', weight='bold', zorder=4)
    for j, crop in enumerate(crops):
        y, h = tt[j], th[j]
        ax.add_patch(FancyBboxPatch((right, y), 2.43, h, boxstyle='round,pad=0,rounding_size=.062',
                                    facecolor='#f8f9fb', edgecolor='#d9dce2', linewidth=.58, zorder=3))
        ax.text(right + .12, y + h / 2, crop, va='center', weight=crop_weight, zorder=4)
        ax.text(11.70, y + h / 2, str(counts[j]), va='center', ha='right',
                weight='bold', color='#000000', zorder=4)
    ax.text(.15, -.23, 'Publication period', weight='bold', va='center')
    ax.text(left - .19, -.23, 'Pairs', weight='bold', va='center', ha='right')
    ax.text(right, -.23, 'Crop', weight='bold', va='center')
    ax.text(11.70, -.23, 'Studies', weight='bold', va='center', ha='right')
    return fig

def ranked_accuracy(studies, *, model_display=None):
    display = MODEL_DISPLAY if model_display is None else model_display
    palette = {year: tuple(PALETTE[family][shade] for shade in (500, 400))
               for year, family in PERIODS.items()}
    top = studies.dropna(subset=['Accuracy']).sort_values(['Accuracy', 'Original_SrNo'], ascending=[False, True]).head(20)
    fig=plt.figure(figsize=(14.0,7.15))
    outer=fig.add_gridspec(1,2,left=.028,right=.985,top=.88,bottom=.105,wspace=.11)
    for panel in range(2):
        spec=outer[panel].subgridspec(1,3,width_ratios=[2.65,2.25,.64],wspace=.075)
        labels,ax,vals=[fig.add_subplot(spec[j]) for j in range(3)]
        for axis in (labels,ax,vals):axis.set_ylim(9.58,-.52)
        ax.set(xlim=(0,.83),xlabel='Gap to 100 (percentage points)')
        rows=top.iloc[panel*10:panel*10+10]
        for i,r in enumerate(rows.itertuples()):
            model=display.get(r.Paper_ID,str(r.Model).strip(' ,')).replace('RTR_Lite_MobileNet','RTR Lite MobileNet')
            study=' '.join(r.Study_label.split())
            gap=100-r.Accuracy
            assert np.isclose(gap+r.Accuracy,100) and gap>0
            tone,highlight=palette[int(r.Year)]
            labels.text(0,i-.175,model,ha='left',va='center',weight='bold')
            labels.text(0,i+.20,study,ha='left',va='center',color='black')
            filled_bar(ax,0,i,gap,.48,tone,highlight=highlight)
            vals.text(.99,i,f'{r.Accuracy:.2f}%',ha='right',va='center',weight='bold')
        labels.set_title('Ranks 1–10' if panel==0 else 'Ranks 11–20',loc='left',weight='bold',pad=14)
        vals.set_title('Accuracy',loc='right',weight='bold',pad=14)
        for axis in (labels,vals):
            axis.set(xlim=(0,1));axis.axis('off')
        ax.set_xticks([0,.2,.4,.6,.8],['0','0.2','0.4','0.6','0.8'])
        ax.set_yticks([])
        axis_only(ax)
    return fig


# Supplementary input loading

METRIC_NAMES = {name.casefold(): name for name in METRICS}
METRIC_ENTRY = re.compile(r'([A-Za-z][A-Za-z0-9]*)\s+([0-9]+(?:\.[0-9]+)?)')
SHORT_MODELS = {
    'VG-GNBNet (VGG16 + Gaussian Naive Bayes)': 'VG-GNBNet',
    'DenseNet-201, AlexNet, and MobileNet-V2': 'DenseNet-201 / AlexNet / MobileNetV2',
    'GCLH ensemble (VGG16, DenseNet121 and InceptionV3)': 'GCLH ensemble',
}


@dataclass(frozen=True)
class FigureInputs:
    studies: pd.DataFrame
    dataset_mentions: pd.DataFrame
    crop_pairs: pd.DataFrame
    model_display: dict | None = None


def parse_scores(text, ref):
    metrics = []
    for line in text.splitlines():
        if line.startswith('Reported results — '):
            break
        metrics.append(line)
    scores = {}
    for token in filter(None, (part.strip() for part in ';'.join(metrics).split(';'))):
        match = METRIC_ENTRY.fullmatch(token)
        if not match or match[1].casefold() not in METRIC_NAMES:
            raise ValueError(f'Invalid performance metric for {ref}: {token!r}')
        name = METRIC_NAMES[match[1].casefold()]
        value = float(match[2])
        if name in scores or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f'Invalid or repeated {name} for {ref}: {token!r}')
        scores[name] = value
    return scores


def aliases(path, key, value):
    rows = pd.read_csv(path, keep_default_na=False)
    return {
        label: tuple(filter(None, group[value]))
        for label, group in rows.groupby(rows[key].str.strip().str.casefold(), sort=False)
    }


def split_names(text):
    return [part.strip() for part in text.split(',') if part.strip()]


def terms(raw, mapping, fallback):
    entries = mapping.get(raw.strip().casefold())
    if entries is None:
        entries = [canonical for part in split_names(raw)
                   for canonical in mapping.get(part.casefold(), (fallback(part),))]
    return sorted(set(entries))


def load_workbook_inputs(path, data_dir):
    years, crops, performance, groups = ('Table 2 - Keywords', 'Table 3 - Crops',
                                         'Table 4 - Performance Metrics', 'Table 5 - Model Paradigms')
    tables = pd.read_excel(path, sheet_name=[years, crops, performance, groups], keep_default_na=False)
    mapping = pd.read_csv(data_dir / 'study_references.csv', keep_default_na=False).set_index('Reference')
    known = set(mapping.index)
    if not known or len(known) != len(mapping) or not mapping.Paper_ID.is_unique:
        raise ValueError('study_references.csv must identify each study exactly once')
    for sheet in (years, crops, performance):
        references = tables[sheet].References
        if not references.is_unique or set(references) != known:
            raise ValueError(f'{sheet} must contain one record for each of the {len(known)} references')
    architecture = tables[groups].set_index('Model Paradigm', verify_integrity=True)
    categories = {}
    for group, refs, count in architecture[['References', 'Studies ( n )']].itertuples(name=None):
        references = [piece.strip() for piece in refs.split(';') if piece.strip()]
        if not references or len(references) != count:
            raise ValueError(f'Table 5 study count does not match the reference list for {group}')
        for ref in references:
            if ref in categories or ref not in known:
                raise ValueError(f'Unexpected or repeated architecture assignment: {ref}')
            categories[ref] = group
    if categories.keys() != known:
        raise ValueError('Every study must be assigned in Table 5 - Model Paradigms')

    studies = mapping.join(tables[years].set_index('References')[['Year']])
    model_data = tables[performance].set_index('References')
    scores = pd.DataFrame.from_dict(
        {ref: parse_scores(text, ref) for ref, text in model_data.Metric.items()}, orient='index',
    ).reindex(index=studies.index, columns=METRICS)
    studies = studies.join(scores)
    studies['Model'] = model_data['Model Name']
    studies['Category'] = pd.Categorical(studies.index.map(categories), categories=architecture.index, ordered=True)
    studies['Year'] = pd.to_numeric(studies.Year, errors='raise')

    sources = aliases(data_dir / 'dataset_text_aliases.csv', 'Workbook_text', 'Figure_source')
    crops_lookup = aliases(data_dir / 'crop_text_aliases.csv', 'Workbook_text', 'Figure_crop')
    dataset_records, crop_records = [], []
    for ref, raw_dataset, raw_crop in tables[crops][['References', 'Dataset', 'Crop']].itertuples(index=False, name=None):
        paper = mapping.at[ref, 'Paper_ID']
        dataset_records.extend((paper, term) for term in terms(raw_dataset, sources, str))
        crop_records.extend((paper, term) for term in terms(raw_crop, crops_lookup, str.capitalize))
    return FigureInputs(
        studies.reset_index(drop=True),
        pd.DataFrame(dataset_records, columns=['Paper_ID', 'Source_or_environment']),
        pd.DataFrame(crop_records, columns=['Paper_ID', 'Crop']),
        {paper: SHORT_MODELS[model] for paper, model in studies[['Paper_ID', 'Model']].itertuples(index=False, name=None)
         if model in SHORT_MODELS},
    )


# Color grading

ADOBE_BLUE_BEFORE = np.array([.667, .755, .849])
ADOBE_BLUE_AFTER = np.array([.313, .493, 1.])


def profile_matrix(profile):
    colors = profile.profile
    return np.array([colors.red_colorant[0], colors.green_colorant[0], colors.blue_colorant[0]]).T


def rgb_decode(values):
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def rgb_encode(values):
    values = np.clip(values, 0, 1)
    return np.where(values <= .0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - .055)


def xyz_to_lab(xyz):
    relative = xyz / WHITE
    f = np.where(relative > (6 / 29) ** 3, np.cbrt(relative), relative / (3 * (6 / 29) ** 2) + 4 / 29)
    x, y, z = f.T
    return np.column_stack((116 * y - 16, 500 * (x - y), 200 * (y - z)))


def lab_to_xyz(lab):
    lightness, a, b = lab.T
    y = (lightness + 16) / 116
    f = np.column_stack((y + a / 500, y, y - b / 200))
    return WHITE * np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29))


def smoothstep(start, end, values):
    t = np.clip((values - start) / (end - start), 0, 1)
    return t * t * (3 - 2 * t)


def polar(lab):
    lightness, a, b = lab.T
    return lightness, np.hypot(a, b), np.arctan2(b, a)


def cartesian(lightness, chroma, hue):
    return np.column_stack((lightness, chroma * np.cos(hue), chroma * np.sin(hue)))


# D50-adapted XYZ colorants from the reference ICC workflow.
P3_TO_XYZ = np.array([
    (0.5151214599609375, 0.2411956787109375, -0.0010528564453125),
    (0.2919769287109375, 0.6922454833984375, 0.0418853759765625),
    (0.1571044921875, 0.0665740966796875, 0.7840728759765625),
]).T
SRGB_TO_XYZ = np.array([
    (0.436065673828125, 0.2224884033203125, 0.013916015625),
    (0.3851470947265625, 0.7168731689453125, 0.097076416015625),
    (0.14306640625, 0.06060791015625, 0.7140960693359375),
]).T
ADOBE_TO_XYZ = np.array([
    (0.6097412109375, 0.3111114501953125, 0.01947021484375),
    (0.2052764892578125, 0.62567138671875, 0.0608673095703125),
    (0.1491851806640625, 0.0632171630859375, 0.74456787109375),
]).T
XYZ_TO_P3 = np.linalg.inv(P3_TO_XYZ)
WHITE = P3_TO_XYZ.sum(axis=1)
REFERENCE_LAB = xyz_to_lab(np.stack((ADOBE_BLUE_BEFORE, ADOBE_BLUE_AFTER)) ** (563 / 256) @ ADOBE_TO_XYZ.T)
ANCHOR_L, ANCHOR_C, ANCHOR_H = polar(REFERENCE_LAB)


def grade(lab):
    lightness, chroma, hue = polar(lab)
    # Protect neutral ink, axes, and paper, including antialiasing around text.
    color_weight = smoothstep(10.5, 16.5, chroma)
    ink_weight = np.maximum(smoothstep(27, 42, lightness), smoothstep(13, 23, chroma))
    weight = color_weight * ink_weight
    target_l = np.interp(lightness, [0, ANCHOR_L[0], 100], [0, ANCHOR_L[1], 100])
    return cartesian(lightness + weight * (target_l - lightness),
                     chroma * (1 + weight * (ANCHOR_C[1] / ANCHOR_C[0] - 1)),
                     hue + weight * (ANCHOR_H[1] - ANCHOR_H[0]))


def display_p3(lab):
    lightness, chroma, hue = polar(lab)
    linear_rgb = lab_to_xyz(lab) @ XYZ_TO_P3.T
    outside = ((linear_rgb < -1e-7) | (linear_rgb > 1 + 1e-7)).any(axis=1)
    # For colors outside Display P3, retain lightness and hue and reduce chroma.
    if outside.any():
        lightness, chroma, hue = lightness[outside], chroma[outside], hue[outside]
        low, high = np.zeros_like(chroma), chroma.copy()
        for _ in range(16):
            mid = (low + high) / 2
            trial = lab_to_xyz(cartesian(lightness, mid, hue)) @ XYZ_TO_P3.T
            fits = ((trial >= 0) & (trial <= 1)).all(axis=1)
            low, high = np.where(fits, mid, low), np.where(fits, high, mid)
        linear_rgb[outside] = lab_to_xyz(cartesian(lightness, low, hue)) @ XYZ_TO_P3.T
    return rgb_encode(linear_rgb)


def colorize(source, destination, icc_profile):
    with Image.open(source) as image:
        raw = np.array(image.convert('RGB'), dtype=np.uint8)
        dpi = image.info['dpi']
        assert 'icc_profile' not in image.info, 'Expected Matplotlib PNG with sRGB color values'
    height, width, _ = raw.shape
    packed = raw[..., 0].astype(np.uint32) << 16
    packed |= raw[..., 1].astype(np.uint32) << 8
    packed |= raw[..., 2]
    values, index = np.unique(packed, return_inverse=True)
    rgb = np.column_stack((values >> 16, (values >> 8) & 255, values & 255)) / 255
    lab = xyz_to_lab(rgb_decode(rgb) @ SRGB_TO_XYZ.T)
    converted = np.rint(display_p3(grade(lab)) * 255).astype(np.uint8)
    output = converted[index.reshape(-1)].reshape(height, width, 3)
    Image.fromarray(output).save(destination, dpi=dpi, icc_profile=icc_profile, optimize=True)


def load_profile(path):
    profile = ImageCms.getOpenProfile(str(path))
    np.testing.assert_allclose(profile_matrix(profile), P3_TO_XYZ, atol=2e-4,
                               err_msg='Select a Display P3 ICC profile')
    return path.read_bytes()


# Figure export

DEFAULT_FONTS = Path('/System/Library/Fonts/Supplemental')
DEFAULT_PROFILE = Path('/System/Library/ColorSync/Profiles/Display P3.icc')


def style(font_dir):
    for item in sorted(font_dir.glob('*.ttf')):
        if 'times' in item.name.casefold():
            font_manager.fontManager.addfont(str(item))
    for weight in ('normal', 'bold'):
        font_manager.findfont(font_manager.FontProperties(family=['Times New Roman'], weight=weight),
                              fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 12, 'axes.titlesize': 12,
        'axes.labelsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'text.color': INK, 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': '#adb4b5',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def black_axes(figure):
    for ax in figure.axes:
        if not ax.axison:
            continue
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_edgecolor('black')
        ax.tick_params(axis='both', which='both', colors='black')
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_color('black')
            axis.get_offset_text().set_color('black')
    figure.canvas.draw()


def emphasize(labels):
    for label in labels:
        label.set(color='black', weight='bold')


def colorful_rgb(color):
    srgb = np.array([to_rgb(color)])
    p3 = display_p3(grade(xyz_to_lab(rgb_decode(srgb) @ SRGB_TO_XYZ.T)))
    return tuple(np.rint(p3[0] * 255) / 255)


def publication_years(studies):
    counts = studies.Year.value_counts().sort_index()
    x = np.arange(len(counts))
    line = colorful_rgb(DARK[0])
    markers = [colorful_rgb(SOFT[i]) for i in (1, 0, 2, 4, 3, 6)]
    figure, ax = plt.subplots(figsize=(8.3, 3.9))
    ax.plot(x, counts.values, color=line, linewidth=1.7)
    ax.scatter(x, counts.values, s=90, c=markers, edgecolor='white', linewidth=1.3, zorder=3)
    for i, count in enumerate(counts.values):
        ax.annotate(str(count), (i, count), xytext=(0, 11), textcoords='offset points',
                    ha='center', weight='bold')
    ax.set(ylim=(0, 41), xlim=(-.3, 5.3), ylabel='Studies')
    ax.set_xticks(x, counts.index.astype(str))
    ax.set_yticks([0, 10, 20, 30, 40])
    axis_only(ax)
    figure.tight_layout(pad=1)
    return figure


def keyword_inputs(studies, data_dir):
    membership = pd.read_csv(data_dir / 'keyword_membership.csv')
    communities = pd.read_csv(data_dir / 'keyword_communities.csv').set_index('Keyword').Community
    assert not membership.duplicated().any()
    assert set(membership.Paper_ID).issubset(set(studies.Paper_ID))
    assert set(membership.Keyword) == set(communities.index) and len(communities) == 16
    years = studies.set_index('Paper_ID').Year
    term_groups = membership.groupby('Keyword').Paper_ID
    counts = term_groups.size()
    mean_years = term_groups.agg(lambda papers: sum(years[p] for p in papers) / len(papers))
    nodes = pd.DataFrame({'label': counts.index, 'cluster': communities.loc[counts.index].to_numpy(),
                          'weight<Occurrences>': counts.values, 'score<Avg. pub. year>': mean_years.values})
    pairs = Counter(pair for _, rows in membership.groupby('Paper_ID')
                    for pair in combinations(sorted(rows.Keyword), 2))
    edges = pd.DataFrame([(first, second, n) for (first, second), n in sorted(pairs.items())],
                         columns=['Source', 'Target', 'Cooccurrences'])
    frequencies = (counts.rename('Studies').reset_index()
                   .sort_values(['Studies', 'Keyword'], ascending=[False, True]))
    return nodes, edges, frequencies


def crop_flows(studies, crop_pairs):
    pairs = crop_pairs.merge(
        studies[['Paper_ID', 'Year']], on='Paper_ID', how='left', validate='many_to_one')
    assert pairs.Year.notna().all() and not pairs.duplicated(['Paper_ID', 'Crop']).any()
    counts = (pairs.groupby('Crop').size().rename('Studies').reset_index()
              .query('Studies >= 2').sort_values(['Studies', 'Crop'], ascending=[False, True]))
    periods = ['2021–2023', '2024', '2025', '2026']
    pairs['Period'] = pairs.Year.astype(int).astype(str).where(pairs.Year >= 2024, periods[0])
    pairs = pairs[pairs.Crop.isin(counts.Crop)]
    flows = pd.crosstab(pairs.Period, pairs.Crop).reindex(index=periods, columns=counts.Crop, fill_value=0)
    assert flows.sum().sum() == len(pairs)
    return flows


def save_figures(output, profile, data_dir):
    inputs = load_workbook_inputs(data_dir / 'Final-Data.xlsx', data_dir)
    studies = inputs.studies
    nodes, edges, frequencies = keyword_inputs(studies, data_dir)
    sources = (inputs.dataset_mentions.groupby('Source_or_environment').size()
               .rename('Studies').reset_index().sort_values(['Studies', 'Source_or_environment'], ascending=[False, True])
               .set_index('Source_or_environment').Studies)

    figures = {
        3: publication_years(studies),
        4: hive(nodes, edges),
        5: hive(nodes, edges, overlay=True),
        6: chord(nodes, edges),
        7: keyword_frequency(frequencies),
        8: radial_waterfall(sources),
        9: distributions(studies),
        10: architecture(studies),
        11: crop_sankey(crop_flows(studies, inputs.crop_pairs)),
    }
    with plt.rc_context({'text.color': 'black'}):
        figures[12] = ranked_accuracy(studies, model_display=inputs.model_display)

    with TemporaryDirectory(dir=output) as temp:
        for number, figure in figures.items():
            black_axes(figure)
            large_label = None
            if number == 8:
                labels = figure.axes[0].texts
                emphasize(labels)
                large_label, = [label for label in labels if label.get_position() == (0, 0)]
                large_label.set(fontsize=25, y=.065)
            elif number == 9:
                emphasize(figure.axes[1].texts)
            elif number == 10:
                emphasize([*figure.axes[0].get_yticklabels(), *figure.texts])
                for panel in figure.axes:
                    emphasize([*panel.get_xticklabels(), *panel.texts])
            labels = [t for t in figure.findobj(plt.Text) if t.get_visible() and t.get_text()]
            assert all(t.get_fontsize() == (25 if t is large_label else 12)
                       and t.get_fontfamily() == ['Times New Roman'] for t in labels)
            manuscript_number = number - 1 if number < 12 else number
            destination = output / f'Fig-{manuscript_number}.png'
            graded = number in (4, 5, 7, 9, 10)
            raw = Path(temp) / destination.name if graded else destination
            kwargs = {'pil_kwargs': {'icc_profile': profile}} if number == 3 else {}
            padding = .2 if number in (4, 5) else .23 if number in (3, 7) else .22
            figure.savefig(raw, dpi=300, bbox_inches='tight', pad_inches=padding, **kwargs)
            plt.close(figure)
            if graded:
                colorize(raw, destination, profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'Data',
                        help='Directory with Final-Data.xlsx and the companion input CSVs (default: Data/)')
    parser.add_argument('--output', type=Path, default=Path('figures'),
                        help='Directory for the manuscript-numbered PNGs')
    parser.add_argument('--font-dir', type=Path, default=DEFAULT_FONTS,
                        help='Directory with Times New Roman TTFs')
    parser.add_argument('--display-p3', type=Path, default=DEFAULT_PROFILE,
                        help='Path to the Display P3 ICC profile')
    args = parser.parse_args()

    required = (
        'Final-Data.xlsx', 'study_references.csv', 'keyword_membership.csv',
        'keyword_communities.csv', 'dataset_text_aliases.csv', 'crop_text_aliases.csv',
    )
    missing = [name for name in required if not (args.data_dir / name).is_file()]
    if missing:
        parser.error(f'Missing inputs in {args.data_dir}: {", ".join(missing)}')

    try:
        style(args.font_dir)
        profile = load_profile(args.display_p3)
    except (ValueError, OSError, AssertionError) as exc:
        parser.error(f'Check Times New Roman (--font-dir) and Display P3 (--display-p3): {exc}')

    args.output.mkdir(parents=True, exist_ok=True)
    save_figures(args.output, profile, args.data_dir)


if __name__ == '__main__':
    main()
