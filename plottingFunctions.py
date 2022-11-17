"""
File which contains all the plotting functions used in the dashboard
"""

from datetime import datetime

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from plotly.colors import hex_to_rgb
from plotly.colors import qualitative as colours

from utilities import *

"""
TODO
    - Using d3blocks to create a chord diagram, network diagram etc: https://towardsdatascience.com/d3blocks-the-python-library-to-create-interactive-and-standalone-d3js-charts-3dda98ce97d4
    
"""


def track_elements(
    dfIDIncl: pd.DataFrame,
    dfIDExcl: pd.DataFrame,
    uidAttr: str,
    attribute: str,
    hover_data: list,
    mesh_time_slider: int,
    colourDict: dict,
):

    """
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

    mesh_time_slider: int
        The value of the slider-meshtime object to select/turn off the mesh viewing
    """

    print(f"     Tracking historical data with 3D plotting for {attribute}")

    if attribute == uidAttr:
        print("Attribute is the same as the unique identifier, cannot plot")
        return go.Figure()

    allTimes = dfIDIncl["_DateTime"].unique()
    allTimesFormat = dfIDIncl["_PermissionDateTime"].unique()
    elements = dfIDIncl[attribute].unique()

    allSizes = np.linspace(4, 12, len(allTimes)).astype(int)

    fig = go.Figure()
    customColourDict = {}

    # create a dictionary to colour the traces depending on the attribute and time of the data
    transparency = np.linspace(0.4, 1, len(allTimes))
    for _, c in enumerate(sorted(dfIDIncl[attribute].unique())):
        customColourDict[str(c)] = {}
        for n_a, a in enumerate(allTimes):

            customColourDict[str(c)][
                a
            ] = f"rgba{tuple(np.append(hex_to_rgb(colourDict[c]), transparency[n_a]))}"

    # Identify identities with access movement which is outsides the norm per element
    dfIDInclSort = dfIDIncl.sort_values([uidAttr, "_DateTime"], ascending=[True, False])
    dfIDInclSort["diff"] = np.sum(
        dfIDInclSort.groupby([uidAttr])[["Dim0", "Dim1", "Dim2"]].diff() ** 2, 1
    ).sort_values()
    idMovement = np.round(dfIDInclSort.groupby([uidAttr, attribute])["diff"].sum(), 2)
    idsOutlier = []
    for attr in elements:
        idMoveAttr = idMovement.loc[slice(None), attr]
        idMoveDesc = idMoveAttr.describe()
        q3 = idMoveDesc.loc["75%"]
        iqr = q3 - idMoveDesc.loc["25%"]
        rng = iqr * 1.5 + q3
        idsOutlier += list(idMoveAttr[idMoveAttr > rng].index)

    # Calculate the aggregate statistics for the attribute plotting
    dfTrack = (
        dfIDInclSort.groupby([attribute, "_DateTime", "_PermissionDateTime"])
        .agg({"Dim0": "median", "Dim1": "median", "Dim2": "median", uidAttr: "count"})
        .reset_index()
        .rename(columns={uidAttr: "Count"})
        .sort_values("_DateTime", ascending=False)
    )

    # plot the volumetric area encomapssing all identities for the given selected time if mesh_time_slider is -1 then this means there is to be NO plotting of the mesh
    if mesh_time_slider == -1:
        timeSelect = -1
    else:
        timeSelect = sorted(dfIDIncl["_DateTime"].unique())[mesh_time_slider]

    # create the size dictionary
    timeDict = {}
    for t, s in zip(allTimes, allSizes):
        timeDict[t] = s

    custom_hover_data = ["Count", attribute, "_PermissionDateTime"]
    for ele in sorted(elements):

        # ----------------- Track median position of identities in elements -----------------
        # get all the unique entries for this unique identity
        eletrdf = dfTrack[dfTrack[attribute] == ele]

        # set the colours so that the newest data pont is 100% opacity and the oldest data point is 40% opacity
        selected_colours = [
            customColourDict[attr][unix]
            for attr, unix in zip(eletrdf[attribute], eletrdf["_DateTime"])
        ]
        selected_sizes = 20  # [timeDict[t] * 2 for t in eletrdf["_DateTime"]]

        # doco: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html
        fig.add_trace(
            go.Scatter3d(
                x=eletrdf["Dim0"],
                y=eletrdf["Dim1"],
                z=eletrdf["Dim2"],
                customdata=eletrdf,
                hovertemplate=f"<b>Grouping: {ele}</b><br>"
                + f"<i>Identity count: %{'{customdata[0]}'}</i><br><br>"
                + "<br>".join(
                    [
                        f"{h}: %{'{customdata['+str(n)+']}'}"
                        for n, h in enumerate(
                            [h for h in custom_hover_data if h.find("_") == -1]
                        )  # display custom data if it does NOT start with "_"
                    ]
                ),
                marker=dict(
                    color=selected_colours, size=selected_sizes, symbol="square"
                ),
                line=dict(color=selected_colours),
                name=ele,  # NOTE this must be a string/number
                legendgroup=ele,  # NOTE this must be a string/number
                # connectgaps=True        # NOTE for some reason this isn't acutally connecting gaps.... maybe wrong data type for empty? '
                hoverlabel=dict(namelength=0),
            )
        )

        # fig.data[-1].marker.size = (10,)
        # fig.data[-1].marker.color = colourDict[ele]

        # --------------- Volumetric area of the element ----------------
        uidVolDf = dfIDInclSort[
            (dfIDInclSort[attribute] == ele) & (dfIDInclSort["_DateTime"] == timeSelect)
        ]

        if len(uidVolDf) == 0:
            continue

        fig = mesh_layers(fig, uidVolDf, colourDict, ele)

    # plot the tracking for outlier identities
    for id in idsOutlier:

        eletrdf = dfIDInclSort[dfIDInclSort[uidAttr] == id]

        # set the colours so that the newest data pont is 100% opacity and the oldest data point is 40% opacity
        selected_colours = [
            customColourDict[attr][unix]
            for attr, unix in zip(eletrdf[attribute], eletrdf["_DateTime"])
        ]
        selected_sizes = 10  # [timeDict[t] for t in eletrdf["_DateTime"]]

        # doco: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html
        fig.add_trace(
            go.Scatter3d(
                x=eletrdf["Dim0"],
                y=eletrdf["Dim1"],
                z=eletrdf["Dim2"],
                customdata=eletrdf,
                hovertemplate=f"<b>{uidAttr}: %{'{customdata['}{hover_data.index(uidAttr)}{']}'}</b><br><br>"
                + "<br>".join(
                    [
                        f"{h}: %{'{customdata['+str(n)+']}'}"
                        for n, h in enumerate(
                            [h for h in hover_data if h.find("_") == -1]
                        )  # display custom data if it does NOT start with "_"
                    ]
                ),
                marker=dict(
                    color=selected_colours,
                    size=10,
                    line=dict(width=1, color="white"),
                ),  # include has an opacity of 1
                line=dict(color=selected_colours),
                name=f"Outlier:{id}",  # NOTE this must be a string/number
                legendgroup=[
                    f"{eletrdf[attribute].unique()[0]} outliers"
                    if len(eletrdf[attribute].unique()) == 1
                    else "Mixed attribute outliers"
                ][
                    0
                ],  # group the outliers by the selected attribute
                # legendgroup=list(set(eletrdf[uidAttr]))[0]  # use the uid as a legend group for the plotting
                # connectgaps=True        # NOTE for some reason this isn't acutally connecting gaps.... maybe wrong data type for empty? '
                hoverlabel=dict(namelength=0),
            )
        )

        # fig.data[-1].marker.color = colourDict[ele]
        # fig.data[-1].marker.color = colourDict[ele]

    plotTitle = f"Tracking {len(elements)} identities grouped by {attribute} from {allTimesFormat[0]} to {allTimesFormat[-1]}. {len(idsOutlier)} displayed as well."

    # remove duplicate legend entries
    # NOTE this may be useful to update the plots rather than re-generating them?
    # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.for_each_trace
    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )

    # see legend doco: https://plotly.com/python/reference/layout/#layout-legend
    fig.update_layout(
        legend=dict(
            traceorder="normal",
        )
    )

    return fig, plotTitle, dfIDIncl, dfIDExcl


def cluster_identities(
    dfIDIncl: pd.DataFrame,
    dfIDExcl: pd.DataFrame,
    uidAttr: str,
    attribute: str,
    hover_data: list,
    sliderDateValue: float,
    sliderRoundValue: float,
    sliderClusterValue: float,
    colourDict: dict,
):

    """
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
    """

    print(
        f"     Plotting data with 3D plotting for {attribute} and relative number of identities"
    )

    # process only data for this time period
    dfModIncl = dfIDIncl[
        dfIDIncl["_DateTime"] == sorted(dfIDIncl["_DateTime"].unique())[sliderDateValue]
    ].reset_index(drop=True)

    # for all identity attributes, if they are all the same for all aggregated identities report it otherwise set as "Mixed".
    aggDict = {
        hd: lambda x: list(x)[0] if len(set([str(n) for n in x])) == 1 else list(x)
        for hd in hover_data
        if hd != attribute
    }

    # get the clustered data
    dfPosIncl = clusterData(dfModIncl, uidAttr, attribute, sliderRoundValue, aggDict)

    if len(dfIDExcl) > 0:

        # add the excluded identities trace
        dfModExcl = dfIDExcl[
            dfIDExcl["_DateTime"]
            == sorted(dfIDExcl["_DateTime"].unique())[sliderDateValue]
        ].reset_index(drop=True)

        # create a dictionary to colour the traces depending on the attribute and time of the data
        dfPosExcl = clusterData(
            dfModExcl, uidAttr, attribute, sliderRoundValue, aggDict
        )

    else:
        dfPosExcl = pd.DataFrame(None, columns=dfPosIncl.columns)

    sizes = pd.concat([dfPosIncl["_Count"], dfPosExcl["_Count"]])
    sizes[
        len(sizes)
    ] = 2.5  # this prevents the maximum size of the marker (40 pixels) form being met until there are at least that many identities in a cluster

    # filter out some of the clusters PER element
    if sliderClusterValue > 0:
        dfPosIncl, dfPosInclRem = getClusterLimit(
            dfPosIncl, attribute, sliderClusterValue
        )
        dfPosExcl = pd.concat([dfPosExcl, dfPosInclRem])
        clusteringInfo = f"displaying clusters with a minimum of {int(sliderClusterValue*100)}% of the max count of identities"
    else:
        clusteringInfo = ""

    # hover_data.append("_Count")
    hover_data = sorted(hover_data)

    # ------
    fig = go.Figure()
    dfPosIncl.rename(
        columns={"_Count": "Count", "_ClusterID": "ClusterID"}, inplace=True
    )
    dfPosExcl.rename(
        columns={"_Count": "Count", "_ClusterID": "ClusterID"}, inplace=True
    )
    dfPosIncl = dfPosIncl[
        ["ClusterID", "Count"]
        + [l for l in list(dfPosIncl.columns) if l != "Count" and l != "ClusterID"]
    ]

    # the clusters to include
    for ele in sorted(dfPosIncl[attribute].unique()):
        dfPosInclAttr = dfPosIncl[dfPosIncl[attribute] == ele]
        selected_sizes = np.clip(
            dfPosInclAttr["Count"] / sizes.max() * 50, 5, 50
        ).astype(int)
        fig.add_scatter3d(
            connectgaps=False,
            customdata=dfPosInclAttr,
            x=dfPosInclAttr["Dim0"],
            y=dfPosInclAttr["Dim1"],
            z=dfPosInclAttr["Dim2"],
            mode="markers",
            marker=dict(
                color=colourDict[ele], size=selected_sizes, opacity=1
            ),  # include has an opacity of 1
            hovertext=create_hovertexts(
                dfPosInclAttr, attribute, ["ClusterID", "Count"], hover_data
            ),
            hoverinfo="text",
            legendgroup=ele,
            name=ele,
            hoverlabel=dict(namelength=0),
        )

        # fig.data[-1].marker.opacity = 1

    if len(dfPosExcl) > 0:

        for ele in sorted(dfPosExcl[attribute].unique()):
            dfPosExclAttr = dfPosExcl[dfPosExcl[attribute] == ele]
            selected_sizes = [
                int(np.ceil(c / sizes.max() * 40)) for c in dfPosExclAttr["Count"]
            ]
            fig.add_scatter3d(
                connectgaps=False,
                customdata=dfPosExclAttr,
                x=dfPosExclAttr["Dim0"],
                y=dfPosExclAttr["Dim1"],
                z=dfPosExclAttr["Dim2"],
                mode="markers",
                marker=dict(
                    color=colourDict[ele],
                    size=selected_sizes,
                    opacity=0.4,
                    line=dict(width=1, color="white"),
                ),  # exclude has a transparency of 0.5
                hovertemplate=f"<b>Grouping: {ele}</b><br><br>"
                + "<br>".join(
                    [
                        f"{h}: %{'{customdata['+str(n)+']}'}"
                        for n, h in enumerate(
                            ["Count"] + [h for h in hover_data if h.find("_") == -1], 1
                        )  # display custom data if it does NOT start with "_"
                    ]
                ),
                legendgroup=f"Excluded",  # group the excluded element together
                name=f"{ele} Excluded",
                hoverlabel=dict(namelength=0),
            )

            # fig.data[-1].marker.opacity = 0.4

    # remove the initial non plot
    fig.update_layout(legend={"itemsizing": "constant"})

    allTimesFormat = dfIDIncl["_PermissionDateTime"].unique()

    plotTitle = f"Plotting and overlaying {len(dfModIncl)} identities for {len(dfPosIncl)} clusters colored based on {attribute} with a spatial resolution of {sliderRoundValue} from {allTimesFormat[sliderDateValue]} {clusteringInfo}"

    return fig, plotTitle, dfPosIncl, dfPosExcl


def plot_identities(
    dfIDIncl: pd.DataFrame,
    dfIDExcl: pd.DataFrame,
    uidAttr: str,
    attribute: str,
    hover_data: list,
    sliderDateValue: int,
    colourDict: dict,
    emphasise=False,
    fig=None,
):

    """
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

    sliderDateValue : int
        Value of the slider data. Limits the process to the selected time period

    emphasise : boolean
        If True, improve the visuals to highlight the object, else leave as default.

    fig : go.Figure()
        Plotly figure object. Optional.
    """

    print(f"     Plotting data with 3D plotting for {attribute}")

    dfTimeIncl = dfIDIncl[
        dfIDIncl["_DateTime"] == dfIDIncl["_DateTime"].unique()[sliderDateValue]
    ]

    if len(dfIDExcl) > 0:

        # add the excluded identities trace
        dfTimeExcl = dfIDExcl[
            dfIDExcl["_DateTime"] == dfIDExcl["_DateTime"].unique()[sliderDateValue]
        ]

    else:
        dfTimeExcl = pd.DataFrame(None, columns=dfIDIncl.columns)

    allTimesFormat = dfIDIncl["_PermissionDateTime"].unique()

    # Remove the uid from the hoverdata so that it has to be explicitly included
    if fig is None:
        fig = go.Figure()

    # the clusters to include
    for ele in sorted(dfTimeIncl[attribute].unique()):
        dfPosInfo = dfTimeIncl[dfTimeIncl[attribute] == ele]

        fig.add_scatter3d(
            connectgaps=False,
            customdata=dfPosInfo,
            x=dfPosInfo["Dim0"],
            y=dfPosInfo["Dim1"],
            z=dfPosInfo["Dim2"],
            mode="markers",
            marker=dict(
                color=colourDict[ele],
                opacity=1,
                size=15 if emphasise else 10,
                line=dict(width=1, color="black")
                if emphasise
                else dict(width=1, color="white"),
                symbol="circle",
            ),  # include has an opacity of 1
            hovertext=create_hovertexts(dfPosInfo, attribute, uidAttr, hover_data),
            hoverinfo="text",
            legendgroup="Selected identities" if emphasise else ele,
            name="Selected identities" if emphasise else ele,
            hoverlabel=dict(namelength=0),
        )

    if len(dfTimeExcl) > 0:

        # the clusters to include
        for ele in sorted(dfTimeExcl[attribute].unique()):
            dfPosExclAttr = dfTimeExcl[dfTimeExcl[attribute] == ele]
            fig.add_scatter3d(
                connectgaps=False,
                customdata=dfPosExclAttr,
                x=dfPosExclAttr["Dim0"],
                y=dfPosExclAttr["Dim1"],
                z=dfPosExclAttr["Dim2"],
                mode="markers",
                marker=dict(
                    color=colourDict[ele], opacity=0.4
                ),  # exclude has an opacity of 0.4
                hovertext=create_hovertexts(
                    dfPosExclAttr, attribute, uidAttr, hover_data
                ),
                hoverinfo="text",
                legendgroup=f"{ele} Excluded",
                name=f"{ele} Excluded",
                hoverlabel=dict(namelength=0),
            )

    plotTitle = f"Plotting {len(dfTimeIncl)} identities colored based on {attribute} with full identity information from {allTimesFormat[sliderDateValue]}"

    return fig, plotTitle, dfTimeIncl, dfTimeExcl


def plot_roles(
    fig: go,
    dfIDIncl: pd.DataFrame,
    dfIDExcl: pd.DataFrame,
    dfRole: pd.DataFrame,
    uidAttr: str,
    roleAttr: str,
    plotRelations: bool,
    colourDict: dict,
):

    """
    Plot the volume of the roles to visualise the overlap of role identities
    """

    for role in sorted(dfRole[roleAttr].unique()):

        dfR = dfRole[dfRole[uidAttr] == role]
        dfIDRole = dfIDIncl[dfIDIncl[roleAttr] == role]

        fig.add_scatter3d(
            customdata=dfR[uidAttr],
            x=dfR["Dim0"],
            y=dfR["Dim1"],
            z=dfR["Dim2"],
            mode="markers",
            marker=dict(color=colourDict[role], opacity=1, symbol="diamond", size=10),
            hovertemplate=f"<b>Role: {role}</b><br>",
            legendgroup=role,
            name=role,
            hoverlabel=dict(namelength=0),
        )

    # if there is corresponding role information
    if plotRelations:
        for idRoles in dfIDIncl[roleAttr].unique():

            fig = mesh_layers(
                fig, dfIDIncl[dfIDIncl[roleAttr] == idRoles], colourDict, idRoles
            )

            # Don't include the legend of the line plots
            fig.data[-1].showlegend = False

    return fig


def mesh_layers(fig, df, colourDict, label):

    """
    Plot 3d meshes to highlight the position of the identities with respect to the median position of all identities within a given element
    """

    pos = df[["Dim0", "Dim1", "Dim2"]]
    midPos = np.median(pos, 0)
    diff = np.sum((pos - midPos) ** 2, 1)
    diffDesc = diff.describe()
    q3 = diffDesc.loc["75%"]
    iqr = q3 - diffDesc.loc["25%"]
    rng = iqr * 1.5 + q3

    # Highlight the outlier points (only if there are any)
    if not np.all((diff < rng) == True):

        fig.add_mesh3d(
            x=df["Dim0"],
            y=df["Dim1"],
            z=df["Dim2"],
            color=colourDict[label],
            opacity=0.02,  # opacity is from lightest to darkest on time
            alphahull=0,
            hoverlabel=dict(namelength=0),
            hovertemplate=f"<b>{label} Outliers</b><br>",
            legendgroup=label,
        )

    # highlight the points which are within the normal range and plot as slighly darker/core identities
    fig.add_mesh3d(
        x=df[diff <= rng]["Dim0"],
        y=df[diff <= rng]["Dim1"],
        z=df[diff <= rng]["Dim2"],
        color=colourDict[label],
        opacity=0.15,  # opacity is from lightest to darkest on time
        alphahull=0,
        hoverlabel=dict(namelength=0),
        hovertemplate=f"<b>{label} Core</b><br>",
        legendgroup=label,
    )

    return fig


def create_hovertexts(infodf, title, subtitles, hover_data=None, length=30):

    """
    Take an pandas array and produce the standard hovertext

    infodf : pd.DataFrame
        data frame containing info

    title : str
        the key info to identify in the hover text (bold)

    subtitles : str
        sub title info, can be multiple (italics)

    hover_data : list
        if supplied, select the columns of the data to include

    length : int
        length of the text to limit
    """

    if hover_data is None:
        hover_data = list(infodf.columns)

    if type(subtitles) is not list:
        subtitles = [subtitles]

    # create the formatted hovertext information
    hovertexts = []
    for _, info in infodf.iterrows():
        hovertexts.append(
            f"<b>{title}: {info[title]}</b><br>"
            + "<br>".join([f"<i>{sub}: {info[sub]}</i>" for sub in subtitles])
            + "<br><br>"
            + "<br>".join(
                [
                    f"{h}: {info[h] if type(info[h]) != list else 'Mixed'}"
                    if len(f"{h}: {info[h]}") < length or type(info[h]) == list
                    else f"{h}: {info[h]}"[:length] + "..."
                    for _, h in enumerate(
                        [h for h in hover_data if h.find("_") == -1]
                    )  # display custom data if it does NOT start with "_" and limit to 30 characters length. If the info is presented in a list then it will display "Mixed"
                ]
            )
        )

    return hovertexts
