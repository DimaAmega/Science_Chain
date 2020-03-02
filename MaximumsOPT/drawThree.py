import pickle
import numpy as np
import sys
import math as mt
import plotly.graph_objs as go
import chart_studio.plotly as py

with open(sys.argv[1], 'rb') as f:
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
    # print(di["K"],'FULL OUT 3 PEND \n',di["full_max"][2],"\n")

size = 60
layout=go.Layout(
                plot_bgcolor="#fff",
                xaxis=dict(nticks=5,ticks="inside",tickwidth=7,ticklen=15,tickcolor="#000",tickfont=dict(size=size),showgrid=False,
                
                    linecolor='black',linewidth=7,mirror=True), 
                yaxis=dict(nticks=5,ticks="inside",tickwidth=7,ticklen=15,tickcolor="#000",tickfont=dict(size=size),showgrid=False,
        
                    linecolor='black',linewidth=7,mirror=True))

fig = go.Figure(layout=layout)

for i in range(N):
    fig.add_trace(go.Scatter(
    x = x[i],
    y = y[i],
    mode = 'markers',
	marker=dict(
        size=2),
    name = '{}'.format(i+1),
    ))
fig.show()
