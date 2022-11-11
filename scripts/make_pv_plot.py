"""
Script to make SV + PVoutput.org map of sites
"""

import pandas as pd
import plotly.graph_objects as go

site_1 = {
    "name": "Belluno Hospital",
    "latitude": "46.138667",
    "longitude": "12.202361",
}

site_2 = {
    "name": "Bari Hospital 1",
    "latitude": "41.114139",
    "longitude": "16.860861",
}

site_3 = {
    "name": "Bari Hospital 2",
    "latitude": "41.112778",
    "longitude": "16.857528",
}

site_4 = {
    "name": "Ca' Foscari University",
    "latitude": "45.4788382",
    "longitude": "12.254642",
}

sv_sites = pd.DataFrame(data=[site_1,site_2,site_3,site_4])
pvoutput_org_metadata = pd.read_csv('data/PVOutput_Italy_systems_metadata.csv')


sv_sites = sv_sites[['latitude','longitude']]
pvoutput_org_metadata = pvoutput_org_metadata[['latitude','longitude']]

traces = []
traces.append(go.Scattermapbox(
        lon = sv_sites['longitude'],
        lat = sv_sites['latitude'],
        mode = 'markers',
        name='SV demo sites',
        marker=go.scattermapbox.Marker(size=25,color='#ff9736'),
        ))
traces.append(go.Scattermapbox(
        lon = pvoutput_org_metadata['longitude'],
        lat = pvoutput_org_metadata['latitude'],
        mode = 'markers',
        name='pvoutput.org',
        marker=go.scattermapbox.Marker(color='#086788'),
        ))

fig = go.Figure(data=traces)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=6, mapbox_center={"lat": 42, "lon": 12})

fig.update_layout(
        title = 'PV sites across italy',
    width=1000,
    height=1000,
    )
fig.show(renderer='browser')