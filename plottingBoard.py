import json
import os
from datetime import datetime
from textwrap import wrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback

import reporting
from layout import pageone_layout
from plottingFunctions import *
from utilities import create_colour_dict, filterIdentityDataFrame

# from dashboard import app

"""
cache = Cache(
    app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
)
"""

pd.options.mode.chained_assignment = None


# create the dash application
def createInteractivePlot(app: Dash, dataModel: object):

    # get the list of desireable attributes
    # Remove the dim values
    # Remove any attributes with over 100 unique values (makezs plotly run too slow and clutters the graphs)
    # Remove the key words "Count" and "Unnamed: 0" which are artefacts of the plotting

    print("---Creating web server for plotting---")
    dfID = dataModel.mdsResults

    # Specify what data from the dataframe is to be selected and shared with the webbrowser
    selectedData = [
        d
        for d in sorted(list(dfID.columns))
        if d.lower().find("unnamed") == -1  # removed the unnamed columns
        and d
        != dataModel.joinKeys[
            "identity"
        ]  # remove the identity uid (use the uid in the permission data)
    ]

    # Specify what data from the dataframe will be included in the hovertext
    hover_data = [s for s in selectedData if s[0] != "_"]  # and s.find("_") == -1
    # hover_data.remove("timeUnix")

    # create the colour dictionary to be used for all graphing
    colour_dict = {}
    for d in hover_data:
        colour_dict[d] = create_colour_dict(dfID[d])

    marks = {}
    # convert the datetime object of the permission extract into a human readable format
    if "_DateTime" in selectedData:
        dfID = dfID.sort_values(
            ["_DateTime", dataModel.joinKeys["identity"]], ascending=True
        )

        # role information is when the _DateTime == -1
        dfRoles = dfID.copy()[dfID["_DateTime"] == -1]
        dfID = dfID[dfID["_DateTime"] > 0]

        # create the marks for the slider
        dtformat = dfID["Permission Datetime"].unique()
        # dttimes = dfID["_DateTime"].unique()

        marks = {n: {"label": d} for n, d in enumerate(dtformat)}
        # marks = {n: {'label': d} for n, d in zip(dttimes, dtformat)}

    # create the drop down information
    attrArray = np.array(
        [[r, len(dfID[r].unique())] for r in hover_data if r.find("DateTime") == -1]
    )
    dropDownOpt = [
        f"{attr}: {idNo} elements"
        for attr, idNo in attrArray
        if attr.find("DateTime") == -1
    ]
    dropDownStart = dropDownOpt[attrArray[:, 1].astype(int).argsort()[0]]

    xRng = dfID["__Dim0"].max() - dfID["__Dim0"].min()
    yRng = dfID["__Dim1"].max() - dfID["__Dim1"].min()
    zRng = dfID["__Dim2"].max() - dfID["__Dim2"].min()
    sliderValue = 2 * np.sqrt(xRng**2 + yRng**2 + zRng**2)

    # make data point selection
    # https://dash.plotly.com/interactive-graphing
    # https://dash.plotly.com/datatable
    # https://dash.plotly.com/datatable/editable

    # take in inputs and create the html layout
    app.layout = pageone_layout(
        dataModel,
        dfID,
        hover_data,
        marks,
        dtformat,
        dropDownOpt,
        dropDownStart,
        sliderValue,
    )

    global DFID, DFROLE, DFPERM, DFPRIV, COLOR_DICT

    DFID = dfID[hover_data + [s for s in selectedData if s not in hover_data]]
    DFROLE = dfRoles
    DFPERM = dataModel.permissionData
    DFPRIV = dataModel.privilegedData
    COLOR_DICT = colour_dict

    print("     Web app created")

    return app


# plotly figure updates
@callback(
    Output("plotly_figure", "figure"),
    Output("identitiesPlotted", "data"),
    # Input('plotly_figure', 'figure'),
    State("plotly_figure", "relayoutData"),
    # State("identityRepresentations", "data"),
    # State("roleData", "data"),
    Input("selectedDropDown", "value"),
    Input("selectableDropDownExclude", "value"),
    Input("selectableDropDownInclude", "value"),
    Input("selectableIdentities", "value"),
    State("uidAttr", "data"),
    State("roleAttr", "data"),
    State("hover_data", "data"),
    Input("toggle-timeseries", "value"),
    Input("toggle-hoverinfo", "value"),
    Input("toggle-rolesurface", "value"),
    Input("slider-dates", "value"),
    Input("slider-rounding", "value"),
    Input("slider-clustering", "value"),
    Input("slider-meshtime", "value"),
    State("slider-outliertolerance", "value"),
)
def update_graph(
    figLayout,
    # dfIDjson,
    # dfRolejson,
    attribute,
    elementsExclude,
    elementsInclude,
    identitiesFind,
    uidAttr,
    roleAttr,
    hover_data,
    trackingToggle,
    hoverinfoToggle,
    rolesurfaceToggle,
    sliderDateValue,
    sliderRoundValue,
    sliderClusterValue,
    sliderMeshTimeValue,
    sliderErrortolValue,
):

    """
    Take in the raw data and selected information and create visualisation
    """

    print("----- Updating plotting information -----")

    print("Variables from the dash board")
    print(
        f"""
        attribute: {str(attribute)},
        trackingToggle: {str(trackingToggle)},
        sliderDateValue: {str(sliderDateValue)},
        sliderRoundValue: {str(sliderRoundValue)},
        sliderClusterValue: {str(sliderClusterValue)},
        """
    )

    attribute = attribute.split(":")[0]
    dfID = DFID.copy()  # pd.read_json(dfIDjson, orient="split")
    dfRole = DFROLE.copy()  # pd.read_json(dfRolejson, orient="table")

    # simplify the elements exclude info list for processing by the pandas df
    includeInfo = [e.split(": ") for e in elementsInclude]
    excludeInfo = [e.split(": ") for e in elementsExclude]

    dfID[hover_data] = dfID[hover_data].astype(str)
    dataColumns = list(dfID.columns)
    dims = sum(1 for x in list(dataColumns) if x.startswith("Dim"))
    colour_dict = COLOR_DICT.copy()

    dfIDIncl, dfIDExcl = filterIdentityDataFrame(
        dfID, uidAttr, includeInfo, excludeInfo
    )

    # if there is no info to include, just return an empty plot
    if len(dfIDIncl) == 0:
        return px.scatter_3d(pd.DataFrame(None)), None

    # remove the count info to match to the data frame
    plotTitle = f"{dims}D visualising {attribute} for {len(dfID)} data points"

    # ---------- Track attributes across the time inputs ----------
    if trackingToggle:

        fig, plotTitle, dfPlotIncl, dfPlotExcl = track_elements(
            dfIDIncl,
            dfIDExcl,
            uidAttr,
            attribute,
            hover_data,
            sliderMeshTimeValue,
            colour_dict[attribute],
            sliderErrortolValue,
        )

    #  ---------- Cluster data around reduced spatial resolution ----------
    elif sliderRoundValue >= 0:

        fig, plotTitle, dfPlotIncl, dfPlotExcl = cluster_identities(
            dfIDIncl,
            dfIDExcl,
            uidAttr,
            attribute,
            hover_data,
            sliderDateValue,
            sliderRoundValue,
            sliderClusterValue,
            colour_dict[attribute],
        )

    # ---------- Plot the raw identity data ----------
    else:

        fig, plotTitle, dfPlotIncl, dfPlotExcl = plot_identities(
            dfIDIncl,
            dfIDExcl,
            uidAttr,
            attribute,
            hover_data,
            sliderDateValue,
            colour_dict[attribute],
        )

    # ---------- Plot the role data ----------
    if rolesurfaceToggle:

        fig = plot_roles(
            fig,
            dfPlotIncl,
            dfPlotExcl,
            dfRole,
            uidAttr,
            roleAttr,
            rolesurfaceToggle and not trackingToggle,  # link roles only if not tracking
            colour_dict[roleAttr],
        )

    if len(identitiesFind) > 0:

        identitiesFocus = dfIDIncl[dfIDIncl[uidAttr].isin(identitiesFind)]

        fig, plotTitle, dfPlotIncl, dfPlotExcl = plot_identities(
            identitiesFocus,
            [],
            uidAttr,
            attribute,
            hover_data,
            sliderDateValue,
            colour_dict[attribute],
            emphasise=True,
            fig=fig,
        )

    # ---------- Scale and format the plot ----------

    r = 0.1  # The extra bit to add to the graphs for scaling
    xDif = dfID["__Dim0"].max() - dfID["__Dim0"].min()
    yDif = dfID["__Dim1"].max() - dfID["__Dim1"].min()
    zDif = dfID["__Dim2"].max() - dfID["__Dim2"].min()
    xMin = dfID["__Dim0"].min() - xDif * r
    xMax = dfID["__Dim0"].max() + xDif * r
    yMin = dfID["__Dim1"].min() - yDif * r
    yMax = dfID["__Dim1"].max() + yDif * r
    zMin = dfID["__Dim2"].min() - zDif * r
    zMax = dfID["__Dim2"].max() + zDif * r

    print(
        [xMin, xMax],
        [yMin, yMax],
        [zMin, zMax],
    )

    fig.update_layout(
        title="<br>".join(wrap(plotTitle, width=50)),  # set the title
        clickmode="event+select",  # all for data to be collected by clicking
        width=1000,
        height=600,
        hovermode="closest",
        scene={
            "xaxis": dict(nticks=7, range=[xMin, xMax]),
            "yaxis": dict(nticks=7, range=[yMin, yMax]),
            "zaxis": dict(nticks=7, range=[zMin, zMax]),
            "aspectmode": "cube",
        },
    )

    # remove information about the position in the plot because this is not useful
    for f in fig.data:

        # if the hoverinfotoggle is true, remove all hovering info
        if hoverinfoToggle:
            f.hovertemplate = None
            f.hoverinfo = "none"
            # fig.layout.scene.xaxis.showspikes = False
            # fig.layout.scene.yaxis.showspikes = False
            # fig.layout.scene.zaxis.showspikes = False

        # if there is a hovertemplate, remove frivilous info
        elif f.hovertemplate is not None:

            # remove the co-ordinate info (not useful)
            f.hovertemplate = f.hovertemplate.replace(
                "__Dim0=%{x}<br>__Dim1=%{y}<br>__Dim2=%{z}<br>", ""
            )

            # if there is color info, remove
            if (
                f.hovertemplate.find(
                    "color=rgba",
                )
                > -1
            ):
                fStart = f.hovertemplate.find("color=rgba")
                fEnd = f.hovertemplate.find("<br>", fStart)
                fCol = f"{f.hovertemplate[fStart:fEnd]}<br>"
                f.hovertemplate = f.hovertemplate.replace(fCol, "")

        # Remove the axis data
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(showticklabels=False, title=""),
            )
        )

    # re-load the figure with the previous camera angle
    if figLayout is not None:
        fig.layout.scene.camera = figLayout.get("scene.camera")

    # add logo
    fig.add_layout_image(
        source="assets/PwC_fl_c.png",
        xref="paper",
        yref="paper",
        x=0.1,
        y=0.1,
        sizex=0.35,
        sizey=0.35,
        xanchor="right",
        yanchor="top",
    )

    print("     Plot updated\n")
    return fig, dfIDIncl.to_json(orient="split")


"""
----------------------------------------------------------------------------
"""


@callback(
    Output("submit_plot", "n_clicks"),
    Input("submit_plot", "n_clicks"),
    State("plotly_figure", "figure"),
    Input("info", "data"),
    Input("selectedDropDown", "value"),
)
def save_plot(click, fig, info, selectedAttr):

    if click:

        print("----- Saving plot -----")

        dims = sum([l.find("axis") > 0 for l in list(fig["layout"]["scene"])])
        selectedAttr = selectedAttr.split(":")[0]
        plotName = f'{os.path.expanduser("~")}\\Downloads\\{info}_{dims}D_{selectedAttr}_{datetime.now().strftime("%y%m%d.%H%M")}.html'
        go.Figure(fig).write_html(plotName)

        print(f"     Plot saved at {plotName}\n")

    return 0


# report 1
@callback(
    Output("report_1", "n_clicks"),
    Output("report_1_sub", "children"),
    Input("report_1", "n_clicks"),
    State("identitiesPlotted", "data"),
    State("uidAttr", "data"),
    State("slider-dates", "value"),
    State("selectedDropDown", "value"),
    State("slider-outliertolerance", "value"),
)
def report_1(click, idPlotted, uid, sliderDate, selectedAttr, errorTol):

    """
    Create a report which identifies which identities are "outliers" relative to their peers in their respective element. It models all identities as seen when the slider-rounding value = 0.

    It models all identities which are selected ONLY from the "Select elements to include/exclude" drop down, it is NOT impacted by the cluster size slider (id slider-clustering) as this report does not perform the getClusterLimit function.
    """

    if click and idPlotted is not None:

        idPlotted = pd.read_json(idPlotted, orient="split")
        permData = DFPERM.copy()
        privData = DFPRIV.copy()
        selectedAttr = selectedAttr.split(":")[0]

        output = reporting.report_1(
            idPlotted,
            permData,
            privData,
            uid,
            sliderDate,
            "OutlierReport",
            selectedAttr,
            errorTol,
        )

    else:
        output = ""

    return 0, output


# report 2
@callback(
    Output("report_2", "n_clicks"),
    Output("report_2_sub", "children"),
    Input("report_2", "n_clicks"),
    State("plotly_figure", "figure"),
    State("identitiesPlotted", "data"),
    State("uidAttr", "data"),
    State("selectedDropDown", "value"),
    State("slider-rounding", "value"),
    State("slider-clustering", "value"),
    State("slider-dates", "value"),
    State("hover_data", "data"),
)
def report_2(
    click,
    fig,
    idPlotted,
    uid,
    selectedAttr,
    sliderRound,
    sliderCluster,
    sliderDate,
    hover_data,
):

    """
    Create a report which describes the permission and attribute break down of all the selected clusters.

    It models all cluster which are NOT EXCLUDED taking into account both the "Select elements to include/exclude" drop down AND the cluster size slider (id slider-clustering). If it
    """

    if click and idPlotted is not None:

        idPlotted = pd.read_json(idPlotted, orient="split")
        permData = DFPERM.copy()
        selectedAttr = selectedAttr.split(":")[0]

        reportName = "ClusterReport"

        output = reporting.report_2(
            idPlotted,
            permData,
            uid,
            selectedAttr,
            sliderRound,
            sliderCluster,
            sliderDate,
            hover_data,
            reportName,
        )
        save_plot(True, fig, reportName, selectedAttr)

    else:
        output = ""

    return 0, output


# report 3
@callback(
    Output("report_3", "n_clicks"),
    Input("report_3", "n_clicks"),
    State("identitiesPlotted", "data"),
    State("uidAttr", "data"),
    State("selectedDropDown", "value"),
    State("hover_data", "data"),
)
def report_3(click, idPlotted, uid, selectedAttr, hover_data):

    """
    Create a report which describes the permission and attribute break down of all the selected clusters.

    It models all cluster which are NOT EXCLUDED taking into account both the "Select elements to include/exclude" drop down AND the cluster size slider (id slider-clustering). If it
    """

    if click and idPlotted is not None:

        idPlotted = pd.read_json(idPlotted, orient="split")
        permData = DFPERM.copy()
        selectedAttr = selectedAttr.split(":")[0]

        reportName = "TrendReport"

        reporting.report_3(
            idPlotted, permData, uid, selectedAttr, hover_data, reportName
        )

    return 0


# killing the dash server
@callback(
    Output("stop_button", "children"),
    Input("stop_button", "n_clicks"),
    State("pid", "data"),
)
def update_exit_button(click, pid):
    if click:
        print("\n!!----- Stopping plottng, server disconnected -----!!\n")
        os.system(f"taskkill /IM {pid} /F")  # this kills the app
        return


# action to perform when a row is added
# NOTE change the graphing visualisation on clicking: https://plotly.com/python/click-events/
@callback(
    Output("selected_points_table", "data"),
    Output("table_info", "data"),
    Output("plotly_figure", "clickData"),
    Input("selected_points_table", "data"),
    Input("selected_points_table", "data_previous"),
    State("hover_data", "data"),
    Input("plotly_figure", "clickData"),
    State("table_info", "data"),
    State("uidAttr", "data"),
)
def table_management(current, previous, hover_data, inputSelection, table_info, uiddf):

    """
    Create a toggle which allows you to select the permissions of each identity selected
    Make the identities selected draggable
    On the graph make the identities standout
        Might need to make a callback with just the figure output and what happens is there is another callback which is called containing a dictionary with all the relevant variables to create/update the graph as necessary
    """

    # initiating condition where there is no information
    if inputSelection is None and previous is None:
        return current, table_info, None

    # protect the removed info process when first initialising the webapp
    elif previous is None:
        pass

    # if a row has been removed then process
    elif len(previous) > len(current) and inputSelection is None:

        # hash the dictionary containing the table info and identify which point is removed. Once identified, update the TABLE_INFO list store
        prevIDs = [hash(json.dumps(p, sort_keys=True)) for p in previous]
        currentIDs = [hash(json.dumps(c, sort_keys=True)) for c in current]
        keptRows = [p in currentIDs for p in prevIDs]
        table_info = [p for p, k in zip(table_info, keptRows) if k]
        return current, table_info, None

    # if there is no custom data
    elif inputSelection["points"][0].get("customdata") is None:
        return current, table_info, None

    # if information has been added to the table the process
    if inputSelection["points"][0].get("customdata") is None:
        return current, table_info, None
    inputData = inputSelection["points"][0]["customdata"]

    # if the selected data is cluster information just remove that info
    if str(inputData[-1]).find("Cluster ") > -1:
        clusterName = inputData[
            -1
        ]  # If this is a cluster, the last value in the custom data is the cluster name
    # if there is something wrong with the data (ie there is less data than columns specified) don't process it any further
    elif len(hover_data) > len(inputData):
        return current, table_info, None
    else:
        clusterName = None

    print("----- Data added -----")
    d = {}
    for n, hd in enumerate(hover_data):
        d[hd] = inputData[n] if type(inputData[n]) != list else "Mixed"

    # if there is a cluster, set the identity name to the cluster name
    if clusterName is not None:
        d[uiddf] = clusterName

    if current == [] or current is None:
        current = [d]
    elif all(current[-1][k] == "" for k in list(current[-1])):
        current = [d]
    else:
        # store the raw data of the table
        current.append(d)

    table_info.append(inputData)

    return current, table_info, None


"""# action to perform when row is removed
@callback(
    Output("output", "children"),
    Input("selected_points_table", "data_previous"),
    Input("selected_points_table", "data"),
    Input("table_info", "data"),
)
def remove_rows(previous, current):

    return ""  # [f'Just removed {row}' for row in previous if row not in current]
"""

# to save file name prompts and checks
@callback(
    Output("input_filename", "placeholder"),
    Output("input_filename", "value"),
    Output("submit_file", "n_clicks"),
    Input("submit_file", "n_clicks"),
    State("table_info", "data"),
    State("input_filename", "value"),
    State("hover_data", "data"),
    State("uidAttr", "data"),
)
def save_file(click, table_info, filename, hover_data, uiddf):

    # Save data as long as there is information etc
    if not click:
        placeholder = "Select data"
    elif table_info == [] or table_info is None:
        placeholder = "Select data"
    elif filename == "":
        placeholder = "Set file name"

    else:

        filePath = f"{os.path.expanduser('~')}\\Downloads\\{filename}.xlsx"
        # if the file is being saved then load and process the info
        if not os.path.exists(filePath):

            permData = DFPERM.copy()
            idData = DFID.copy()

            table_df = pd.DataFrame([pd.Series(i) for i in table_info])
            col = table_df.columns.to_list()
            col[: len(idData.columns)] = idData.columns
            col[len(idData.columns) :] = [
                f"__Other{n}" for n in range(len(col) - len(idData.columns))
            ]
            table_df.columns = col

            # reporting.save_selected_information(selectedIDdf, permData, uiddf, filePath)
            reporting.save_selected_information(table_df, permData, uiddf, filePath)

            placeholder = "File saved"
            print(f"     File saved at {filePath}")
        else:
            placeholder = "File exists"

    # always reset the text
    output = ""

    print(f"----- Save file, {placeholder}, {output} -----\n")
    return placeholder, output, 0


@callback(
    Output("toggle-timeseries-out", "children"),
    Output("toggle-hoverinfo-out", "children"),
    Output("toggle-rolesurface-out", "children"),
    Output("toggle-rolesurface", "disabled"),
    Output("slider-dates", "disabled"),
    Output("slider-rounding", "disabled"),
    Output("slider-clustering", "disabled"),
    Output("slider-meshtime", "disabled"),
    Input("toggle-timeseries", "value"),
    Input("toggle-hoverinfo", "value"),
    Input("toggle-rolesurface", "value"),
    State("roleAttr", "data"),
    Input("slider-rounding", "value"),
)
def update_output(
    toggleTimeSeriesValue,
    toggleHoverinfoValue,
    roleSurfaceValue,
    roleAttr,
    sliderRoundValue,
):

    """
    Update the toggles and sliders based on the corresponding value of each other
    """

    # default values of all sliders/toggles
    toggleTimeSeries = "Date of extract"
    toggleHoverinfo = "Hoverinfo disabled"
    roleSurfaceInfo = "No role overlay"
    roleSurfaceInfoDisabled = False
    sliderDateDisabled = False
    sliderRoundDisabled = False
    sliderClusterDisabled = False
    sliderMeshTimeDisabled = True

    """
    If displaying time tracking data then:
        - No date selection (all being displayed anyway)
        - No rounding of data 
        - No clustering of data
        - No role surface creation 
        - Enable the mesh time selection slider
    """
    if toggleTimeSeriesValue:
        toggleTimeSeries = "Tracking data"
        sliderDateDisabled = True
        sliderRoundDisabled = True
        sliderClusterDisabled = True
        roleSurfaceInfoDisabled = True
        sliderMeshTimeDisabled = False

    """
    If there is no rounding being performed:
        - There is no threshold of rounding to be performed
    """
    if sliderRoundValue < 0:
        sliderClusterDisabled = True

    """
    When the hover info is enabled display the appropriate text
    """
    if not toggleHoverinfoValue:
        toggleHoverinfo = "Hoverinfo enabled"

    """
    If the the role surface is selected AND there is role information then display the appropriate text.

    If there is no role information (determined by whether a role joining key exists) then the role surface viewer is disabled (and no role information will be displayed).
    """
    if roleSurfaceValue and roleAttr is not None:
        roleSurfaceInfo = "Role overlaying"
    elif roleAttr is None:
        roleSurfaceInfo = "No roles"
        roleSurfaceInfoDisabled = True

    return (
        toggleTimeSeries,
        toggleHoverinfo,
        roleSurfaceInfo,
        roleSurfaceInfoDisabled,
        sliderDateDisabled,
        sliderRoundDisabled,
        sliderClusterDisabled,
        sliderMeshTimeDisabled,
    )


@callback(
    Output("report_1", "children"),
    Output("report_1", "disabled"),
    Output("report_2", "children"),
    Output("report_2", "disabled"),
    Output("report_3", "children"),
    Output("report_3", "disabled"),
    Output("report_4", "disabled"),
    Output("report_5", "disabled"),
    Input("slider-rounding", "value"),
    Input("slider-clustering", "value"),
    Input("toggle-timeseries", "value"),
)
def update_buttons(sliderRounding, sliderClustering, trackingToggle):

    """
    Update the report buttons
    """

    report3 = False
    report4 = False
    report5 = False

    report1_child = "Outlier Report"
    report1_disabled = False

    report2_child = "Cluster Report"
    report2_disabled = False

    report3_child = "BETA Identity changes report"
    report3_disabled = False

    if sliderRounding >= 0 or trackingToggle:
        report1_child = "Disabled"
        report1_disabled = True

    if sliderRounding < 0 or trackingToggle:
        report2_child = "Disabled"
        report2_disabled = True

    if not trackingToggle:
        report3_child = "Disabled"
        report3_disabled = True

    return (
        report1_child,
        report1_disabled,
        report2_child,
        report2_disabled,
        report3_child,
        report3_disabled,
        report4,
        report5,
    )


@callback(
    Output("selectableDropDownExclude", "options"),
    Output("selectableDropDownInclude", "options"),
    Input("selectedDropDown", "value"),
    State("identitiesPlotted", "data"),
    Input("selectableDropDownExclude", "value"),
    Input("selectableDropDownInclude", "value"),
)
def generate_sub_categories(
    attribute,
    identitiesPlotted,
    selectedDropDownExclude,
    selectedDropDownInclude,
):

    """
    Get all the possible options for the include and/or exclude categories
    """

    # NOTE keeping track of what is included AND excluded becomes pretty complicated....

    if identitiesPlotted is not None:
        identitiesPlotted = pd.read_json(identitiesPlotted, orient="split")
        identitiesAll = DFID  # pd.read_json(identitiesAll, orient="split")
        attribute = attribute.split(":")[0]
        dropDownExclude = (
            sorted(
                [f"{attribute}: {ele}" for ele in identitiesPlotted[attribute].unique()]
            )
            + selectedDropDownExclude
        )
        dropDownInclude = (
            sorted([f"{attribute}: {ele}" for ele in identitiesAll[attribute].unique()])
            + selectedDropDownInclude
        )

        # ensure you cannot select include for elements that are excluded
        # [dropDownInclude.remove(exc) for exc in selectedDropDownExclude]
        # [dropDownExclude.remove(exc) for exc in selectedDropDownInclude]

    else:
        dropDownExclude = selectedDropDownExclude
        dropDownInclude = selectedDropDownInclude

    return dropDownExclude, dropDownInclude


# change the colour of selected points, https://plotly.com/python/click-events/
"""
# create our callback function
def update_point(trace, points, selector):
    c = list(scatter.marker.color)
    s = list(scatter.marker.size)
    for i in points.point_inds:
        c[i] = '#bae2be'
        s[i] = 20
        with f.batch_update():
            scatter.marker.color = c
            scatter.marker.size = s
"""

"""
NOTE need to figure out how to combine outputs
@callback(Output('clear_table', 'n_clicks'),
    Output('selected_points_table', 'data'),
    Input('clear_table', 'n_clicks'),
    Input('selected_points_table', 'data'),
    )
def clear_table(click, table):

    if click:
        return 0, []
"""
