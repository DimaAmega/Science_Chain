import pickle
import numpy as np
import sys
import math as mt
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.subplots import make_subplots


with open("{}Ahead.pickle".format(sys.argv[1]), 'rb') as f:
    data1 = pickle.load(f)

L1x = []
L1y = []

x1 = []
y1 = []
label1 = []
N1 = len(data1[0]["max"])
for i in range(N1):
    x1.append([])
    y1.append([])

for di in data1:
    for i in range(N1):
        for j in range(len(di["max"][i])):
            y1[i].append(di["max"][i][j])
            x1[i].append(di["K"])
    L1x.append(di["K"])
    L1y.append(di["Lexp"])
    label1.append(di["state"])
with open("{}Back.pickle".format(sys.argv[1]), 'rb') as f:
    data2 = pickle.load(f)


L2x = []
L2y = []

x2 = []
y2 = []
label2 = []
N2 = len(data2[0]["max"])
for i in range(N2):
    x2.append([])
    y2.append([])

for di in data2:
    for i in range(N2):
        for j in range(len(di["max"][i])):
            y2[i].append(di["max"][i][j])
            x2[i].append(di["K"])
    L2x.append(di["K"])
    L2y.append(di["Lexp"])
    label2.append(di["state"])


axis_style = dict(nticks=5,ticks="inside",tickwidth=7,ticklen=15,tickcolor="#000",showgrid=False,
                    linecolor='black',linewidth=7,mirror=True)
layout=go.Layout(
                plot_bgcolor="#fff",
                xaxis= axis_style,
                xaxis2=axis_style, 
                yaxis= axis_style,
                yaxis2=axis_style,
                xaxis3= axis_style,
                xaxis4=axis_style, 
                yaxis3= axis_style,
                yaxis4=axis_style,)
                
fig = make_subplots(rows=4, cols=1,shared_xaxes=True,shared_yaxes=True, vertical_spacing=0.035)
fig.update_layout(layout)
fig.update_yaxes(range=[-1e-2,0.1],row=2)
fig.update_yaxes(range=[-1e-2,0.1],row=4)
#fig.update_yaxes4(range=[1e-3, 0.1])

for i in range(N1):
    fig.add_trace(go.Scatter(
    x = x1[i],
    y = y1[i],
    mode = 'markers',
	marker=dict(
        size=1),
    name = '{}'.format(i+1),
    ),row=1,col=1)

fig.add_trace(go.Scatter(
    x = L1x,
    y = L1y,
    mode = 'markers+lines',
	marker=dict(
        size=1),
    name = 'Lexp',
    text = label1,
    ),row=2,col=1)


for j in range(N2):
    fig.add_trace(go.Scatter(
    x = x2[j],
    y = y2[j],
    mode = 'markers',
	marker=dict(
        size=1),
    name = '{}'.format(j+1),
    ),row=3,col=1)

fig.add_trace(go.Scatter(
    x = L2x,
    y = L2y,
    mode = 'markers+lines',
	marker=dict(
        size=1),
    name = 'Lexp',
    text = label2,
    ),row=4,col=1)

fig.show()
