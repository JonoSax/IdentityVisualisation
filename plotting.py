'''
File which contains all the plotting functions used in the dashboard
'''

import numpy as np
from plotly import graph_objects as go
from plotly import express as px
from plotly.colors import qualitative as colours
from plotly.colors import hex_to_rgb
import pandas as pd
from datetime import datetime
from utilities import *

def trackElements(dfIDIncl : pd.DataFrame, uidAttr : str, attribute : str, hover_data : list):

    '''
    Track the permission positions of all identities in the respective elements

    Inputs 
    -----

    dfIDIncl : pd.DataFrame
        The dataframe which contains all the information about the target identities to be modelled (modelled positions, identity information)

    uidAttr : str
        The attribute name which corresponds with the unique identifier in the dfIDIncl. NOTE this is note made the index of the dataframe because this contains temporal information therefore it is NOT a unique identifier of records, only of the identity

    attribute : str
        The attribute name which has been selected to be temporarlly analysed.

    hover_data : list
        List of the attributes to include in the hoverdata from graphing
    '''

    print(f"     Tracking historical data with 3D plotting for {attribute}")

    allTimes = dfIDIncl["_DateTime"].unique()
    allTimesFormat = dfIDIncl["_PermissionDateTime"].unique()

    allSizes = np.linspace(4, 12, len(allTimes)).astype(int)

    fig = go.Figure()
    colourDict = {}

    # create a dictionary to colour the traces depending on the attribute and time of the data
    transparency = np.linspace(0.4, 1, len(allTimes))
    for n_c, c in enumerate(sorted(dfIDIncl[attribute].unique())):
        colourDict[str(c)] = {}
        for n_a, a in enumerate(allTimes):
            colourDict[str(c)][a] = f"rgba{tuple(np.append(hex_to_rgb(colours.Plotly[n_c%len(colours.Plotly)]), transparency[n_a]))}"
    
    # create the size dictionary
    timeDict = {}
    for t, s in zip(allTimes, allSizes):
        timeDict[t] = s

    # Combine the individual positions and take the median positions of all identities for the particular
    # attribute and dates if the attribute selected is not the unique identifier
    if attribute != uidAttr:
        allAttrs = dfIDIncl[attribute].unique()
        data = []
        for attr in allAttrs:
            dfAttrId = dfIDIncl[dfIDIncl[attribute] == attr]
            for dt in allTimes:
                data.append([
                    np.median(dfAttrId[dfAttrId["_DateTime"] == dt]["Dim0"]), 
                    np.median(dfAttrId[dfAttrId["_DateTime"] == dt]["Dim1"]), 
                    np.median(dfAttrId[dfAttrId["_DateTime"] == dt]["Dim2"]), 
                    dt,
                    len(dfAttrId[dfAttrId["_DateTime"] == dt]),
                    attr,
                    datetime.fromtimestamp(int(dt)).strftime("%m/%d/%Y, %H:%M:%S")
                    ])

        hover_data = ["Count", attribute, "_PermissionDateTime"]
        dfTrack = pd.DataFrame(data, columns = ["Dim0", "Dim1", "Dim2", "_DateTime"] + hover_data)
        dfTrack = dfTrack.combine_first(pd.DataFrame(columns=dfIDIncl.columns))
        elements = dfTrack[attribute].unique()

    # if the selected attribute is the unique identifier, provide all data (there will
    # no combining of data)
    else:
        dfTrack = dfIDIncl
        elements = dfIDIncl[attribute].unique()

    dfTrack = dfTrack.sort_values("_DateTime")

    for ele in elements:
        # get all the unique entries for this unique identity
        uiddf = dfTrack[dfTrack[attribute] == ele]

        selected_colours = [colourDict[attr][unix] for attr, unix in zip(uiddf[attribute], uiddf["_DateTime"])]
        selected_sizes = [timeDict[t] for t in uiddf["_DateTime"]]
        # set the colours so that the newest data pont is 100% opacity and the oldest data point is 40% opacity
        name = [u for u in uiddf[attribute] if u != "None"][0]

        # doco: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html
        fig.add_trace(
                go.Scatter3d(
                    x=uiddf["Dim0"],
                    y=uiddf["Dim1"],
                    z=uiddf["Dim2"],
                    customdata=uiddf[hover_data],
                    hovertemplate = f"<b>Grouping: {ele}</b><br><i>%{'{customdata[0]}'}</i><br><br>" + "<br>".join([f"{h}: %{'{customdata['+str(n)+']}'}" for n, h in enumerate(hover_data)]),
                    # hovertext = 
                    # ['<br>'.join([f"{h}: {uiddf[h].iloc[n]}" for h in hover_data]) for n in range(len(uiddf))],
                    marker=dict(color=selected_colours, size=selected_sizes),
                    line = dict(color=selected_colours),
                    name = name,            # NOTE this must be a string/number
                    # legendgroup = name,     # NOTE this must be a string/number
                    # connectgaps=True        # NOTE for some reason this isn't acutally connecting gaps.... maybe wrong data type for empty? '
                    hoverlabel = dict(namelength=0)
                )
            )

    plotTitle = f"Tracking {len(elements)} identities grouped by {attribute} from {allTimesFormat[0]} to {allTimesFormat[-1]}"

    # remove duplicate legend entries
    # NOTE this may be useful to update the plots rather than re-generating them?
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.for_each_trace
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False) if (trace.name in names) else names.add(trace.name)
            )

    # for missing datapoints, connect traces
    fig.update_traces(connectgaps=True)

    # see legend doco: https://plotly.com/python/reference/layout/#layout-legend 
    fig.update_layout(
        legend=dict(
            traceorder= "normal",
        )
    )

    return fig, plotTitle

def clusterIdentities(dfIDIncl : pd.DataFrame, dfIDExcl : pd.DataFrame, uidAttr : str, attribute : str, hover_data : list, sliderDateValue : float, sliderRoundValue : float, sliderClusterValue : float):

    '''
    Plot the current time specified data but scale the dots to represent the relative number of identities
    at that position.

    Inputs
    ------

    dfIDIncl : pd.DataFrame
        The dataframe which contains all the target information about the identities to be modelled (modelled positions, identity information)
    
    dfIDExcl : pd.DataFrame
        The dataframe which contains all the information about the identities to be exlcuded from the main visual modelling and reporting (modelled positions, identity information)
    
    uidAttr : str
        The attribute name which corresponds with the unique identifier in the dfIDIncl. NOTE this is note made the index of the dataframe because this contains temporal information therefore it is NOT a unique identifier of records, only of the identity

    attribute : str
        The attribute name which has been selected to be temporarlly analysed.

    hover_data : list
        List of the attributes to include in the hoverdata from graphing

    sliderDateValue : float
        Value of the slider data. Limits the process to the selected time period
    
    sliderRoundValue : float
        Value of the cluster rounding value. Sets the spatial clustering tolerance
    
    sliderClusterValue : float
        Value of the cluster threhold value. Clusters for each element which are below the set fraction size (set by the sliderClusterValue) of the maximum sized cluster are included moved to the dfIDExcl dataframe


    NOTE this is just using an average position and clustering based on the spatial resolution instead of
    grouping identities based on their distance from each other for two reason:
        1 - This is much simpler
        2 - It conceptually makes sense to group information that is nearby to a certain point. The
            actual point is irrelevant because, like the actual distances between points, they are
            an arbituary representation of the relationships of each identity. 
            Yes it would be more accurate to centre the clustering in areas where there is higher identity
            density rather tan along grid lines. However this is significantly more complicated for 
            very little tangible improvement in the ideas you can extract from the data.
    '''

    print(f"     Plotting data with 3D plotting for {attribute} and relative number of identities")

    # process only data for this time period
    dfModIncl = dfIDIncl[dfIDIncl["_DateTime"] == sorted(dfIDIncl["_DateTime"].unique())[sliderDateValue]].reset_index(drop=True)

    # for all identity attributes, if they are all the same for all aggregated identities report it otherwise set as "Mixed". 
    aggDict = {hd: lambda x: list(set(x))[0] if len(set([str(n) for n in x]))==1 else "Mixed" for hd in hover_data if hd != attribute}
    
    # get the clustered data
    dfPosIncl = clusterData(dfModIncl, uidAttr, attribute, sliderRoundValue, aggDict)

    if len(dfIDExcl) > 0:

        # add the excluded identities trace
        dfModExcl = dfIDExcl[dfIDExcl["_DateTime"] == sorted(dfIDExcl["_DateTime"].unique())[sliderDateValue]].reset_index(drop=True)

        # create a dictionary to colour the traces depending on the attribute and time of the data
        dfPosExcl = clusterData(dfModExcl, uidAttr, attribute, sliderRoundValue, aggDict)

    else:
        dfPosExcl = pd.DataFrame(None, columns=dfPosIncl.columns)

    sizes = pd.concat([dfPosIncl["_Count"], dfPosExcl["_Count"]])

    # filter out some of the clusters PER element 
    if sliderClusterValue > 0:
        dfPosIncl, dfPosInclRem = getClusterLimit(dfPosIncl, attribute, sliderClusterValue)
        dfPosExcl = pd.concat([dfPosExcl, dfPosInclRem])
        clusteringInfo = f"displaying clusters with a minimum of {int(sliderClusterValue*100)}% of the max count of identities"
    else:
        clusteringInfo = ""

    # hover_data.append("_Count")
    hover_data = sorted(hover_data)

    # create the colour dictionary to be used for all visualisation
    colourDict = {}
    for n_c, c in enumerate(sorted(np.r_[dfIDIncl[attribute].unique(), dfIDExcl[attribute].unique()])):

        colourDict[str(c)] = colours.Plotly[n_c%len(colours.Plotly)]

    # ------
    fig = px.scatter_3d(pd.DataFrame(None))

    # the clusters to include
    for ele in sorted(dfPosIncl[attribute].unique()):
        dfPosInclAttr = dfPosIncl[dfPosIncl[attribute] == ele]
        selected_sizes = [int(np.ceil(c/sizes.max()*40)) for c in dfPosInclAttr["_Count"]]
        fig.add_scatter3d(
            connectgaps=False,
            customdata=dfPosInclAttr[["_ClusterID", "_Count"] + hover_data],
            x=dfPosInclAttr["Dim0"], 
            y=dfPosInclAttr["Dim1"], 
            z=dfPosInclAttr["Dim2"],
            mode = "markers",
            marker=dict(color=colourDict[ele], size=selected_sizes, opacity=1),      # include has an opacity of 1
            hovertemplate = f"<b>Grouping: {ele}</b><br><i>%{'{customdata[0]}'}</i><br><br>" + "<br>".join([f"{h}: %{'{customdata['+str(n)+']}'}" for n, h in enumerate(["Count"] + hover_data, 1)]),
            legendgroup=ele,
            name=ele,
            hoverlabel = dict(namelength=0)
            )

        # fig.data[-1].marker.opacity = 1
    
    if len(dfPosExcl) > 0:
        
        for ele in sorted(dfPosExcl[attribute].unique()):
            dfPosExclAttr = dfPosExcl[dfPosExcl[attribute] == ele]
            selected_sizes = [int(np.ceil(c/sizes.max()*40)) for c in dfPosExclAttr["_Count"]]
            fig.add_scatter3d(
                connectgaps=False,
                customdata=dfPosExclAttr[["_ClusterID", "_Count"] + hover_data],    # include the clusterID, counts and selected hoverdata only
                x=dfPosExclAttr["Dim0"], 
                y=dfPosExclAttr["Dim1"], 
                z=dfPosExclAttr["Dim2"],
                mode = "markers",
                marker=dict(color=colourDict[ele], size=selected_sizes, opacity=0.4), # exclude has a transparency of 0.5
                hovertemplate = f"<b>Grouping: {ele}</b><br><br>" + "<br>".join([f"{h}: %{'{customdata['+str(n)+']}'}" for n, h in enumerate(["Count"] + hover_data, 1)]),
                legendgroup=f"{ele} Excluded",
                name=f"{ele} Excluded",
                hoverlabel = dict(namelength=0)
                )

            # fig.data[-1].marker.opacity = 0.4

    # remove the initial non plot
    fig.data = fig.data[1:]
    fig.update_layout(legend= {'itemsizing': 'constant'})

    allTimesFormat = dfIDIncl["_PermissionDateTime"].unique()

    plotTitle = f"Plotting and overlaying {len(dfModIncl)} identities for {len(dfPosIncl)} clusters colored based on {attribute} with a spatial resolution of {sliderRoundValue} from {allTimesFormat[sliderDateValue]} {clusteringInfo}"

    return fig, plotTitle

def plotIdentities(dfIDIncl, uidAttr, attribute, hover_data, sliderDateValue):

    '''
    Plot the current time specified data but scale the dots to represent the relative number of identities
    at that position.

    Inputs
    -----

    dfIDIncl : pd.DataFrame
        The dataframe which contains all the target information about the identities to be modelled (modelled positions, identity information)
    
    uidAttr : str
        The attribute name which corresponds with the unique identifier in the dfIDIncl. NOTE this is note made the index of the dataframe because this contains temporal information therefore it is NOT a unique identifier of records, only of the identity

    attribute : str
        The attribute name which has been selected to be temporarlly analysed.

    hover_data : list
        List of the attributes to include in the hoverdata from graphing

    sliderDateValue : float
        Value of the slider data. Limits the process to the selected time period
    '''

    print(f"     Plotting data with 3D plotting for {attribute}")

    dfTime = dfIDIncl[dfIDIncl["_DateTime"] == dfIDIncl["_DateTime"].unique()[sliderDateValue]]
    dfTime = dfTime.sort_values(attribute)

    allTimesFormat = dfIDIncl["_PermissionDateTime"].unique()

    fig = px.scatter_3d(dfTime, 
            x="Dim0", 
            y="Dim1", 
            z="Dim2",
            hover_data = hover_data,
            color = attribute, 
            hover_name = uidAttr,  
            )

    plotTitle = f"Plotting {len(dfTime)} identities colored based on {attribute} with full identity information from {allTimesFormat[sliderDateValue]}"

    return fig, plotTitle