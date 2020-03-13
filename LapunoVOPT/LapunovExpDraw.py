import pickle
import numpy as np
import sys
import math as mt
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.subplots import make_subplots

with open(sys.argv[1], 'rb') as f:
    dataLapExp = pickle.load(f)

with open(sys.argv[2], 'rb') as f:
    data = pickle.load(f)
x = []
y = []
N = len(data[0]["max"])
for i in range(N):
    x.append([])
    y.append([])

for di in data:
    for i in range(N):
        for j in range(len(di["max"][i])):
            y[i].append(di["max"][i][j])
            x[i].append(di["K"])



x1 = [] 
y1 = [] 
for e_i in dataLapExp:
    x1.append(e_i["K_i"])
    y1.append(e_i["Lexp"])



axis_style = dict(nticks=5,ticks="inside",tickwidth=7,ticklen=15,tickcolor="#000",tickfont=dict(size=20),showgrid=False,
                    linecolor='black',linewidth=7,mirror=True)
layout=go.Layout(
                plot_bgcolor="#fff",
                xaxis= axis_style,
                xaxis2=axis_style, 
                yaxis= axis_style,
                yaxis2=axis_style)
                
fig = make_subplots(rows=2, cols=1,shared_xaxes=True, vertical_spacing=0.032)
fig.update_layout(layout)

for i in range(3):
    fig.add_trace(go.Scatter(
    x = x[i],
    y = y[i],
    mode = 'markers',
	marker=dict(
        size=1),
    name = '{}'.format(i+1),
    ),row=1,col=1)

fig.add_trace(go.Scatter(
    x = x1,
    y = y1,
    mode = 'markers',
	marker=dict(
        size=4),
    name = 'lexp',
    ),row=2,col=1)


fig.show()

