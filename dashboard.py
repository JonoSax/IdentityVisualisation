import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, dash_table
import pandas as pd
import os
from datetime import datetime
import dash_daq as daq
import plotly.graph_objects as go
import numpy as np
from textwrap import wrap
import reporting
from utilities import filterIdentityDataFrame, create_colour_dict
from plotting import *

pd.options.mode.chained_assignment = None

"""
TODO
    - How to pass an object to the dash app, not just the attributes? Performance
    issues?
    - Optimise dash computations with caching, parallelisation: https://dash.plotly.com/sharing-data-between-callbacks
    - Persist the camera views of plots with data refreshes: https://plotly.com/python/reference/layout/#layout-uirevision 
"""

# Theme stuff: https://dash.plotly.com/external-resources
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
# external_stylesheets = [open('assets\\bWLwgP.css', "r")]
app = Dash(__name__)  # , external_stylesheets=external_stylesheets)


def launchApp(dataModel: object):

    appObj = createInteractivePlot(dataModel)
    # webbrowser.open("http://127.0.0.1:8050/", new = 0, autoraise = True)
    try:
        appObj.run_server(debug=True, port=8050)
    except Exception as e:
        print(e)
        launchApp(appObj)


# create the dash application
def createInteractivePlot(dataModel: object):

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
        and d != dataModel.joinKeys["identity"]  # remove the identity uid
    ]

    # Specify what data from the dataframe will be included in the hovertext
    hover_data = [s for s in selectedData if s.lower().find("dim") == -1]
    # hover_data.remove("timeUnix")

    # create the colour dictionary to be used for all graphing
    colour_dict = {}
    for d in hover_data:
        colour_dict[d] = create_colour_dict(dfID[d])

    marks = {}
    # convert the datetime object of the permission extract into a human readable format
    if "_DateTime" in hover_data:
        hover_data.remove("_DateTime")
        selectedData.append("_PermissionDateTime")
        hover_data.append("_PermissionDateTime")
        dfID = dfID.sort_values(
            ["_DateTime", dataModel.joinKeys["identity"]], ascending=True
        )

        # role information is when the _DateTime == -1
        dfRoles = dfID.copy()[dfID["_DateTime"] == -1]
        dfID = dfID[dfID["_DateTime"] > 0]

        # convert the datetime object into a human readable time
        dfID["_PermissionDateTime"] = dfID["_DateTime"].apply(
            lambda x: datetime.fromtimestamp(int(x)).strftime("%m/%d/%Y, %H:%M:%S")
        )

        # create the marks for the slider
        dtformat = dfID["_PermissionDateTime"].unique()
        # dttimes = dfID["_DateTime"].unique()

        marks = {n: {"label": d} for n, d in enumerate(dtformat)}
        # marks = {n: {'label': d} for n, d in zip(dttimes, dtformat)}

    attrArray = np.array([[r, len(dfID[r].unique())] for r in hover_data if r.find("DateTime") == -1])
    dropDownOpt = [
        f"{attr}: {idNo} elements"
        for attr, idNo in attrArray
        if attr.find("DateTime") == -1
    ]
    dropDownStart = dropDownOpt[attrArray[:, 1].astype(int).argsort()[0]]

    # for values which are numeric, convert their values into a ranked position so that
    # on the heat maps it can show up easily
    # NOTE this is not actually very useful as it assumes that data that is chronological is related
    """
    dfSelect = dfID[hover_data]
    dfRanked = dfSelect.rank(numeric_only = True, method = 'dense').astype(int)
    dfID[list(dfRanked.columns)] = dfRanked
    """

    # make data point selection
    # https://dash.plotly.com/interactive-graphing
    # https://dash.plotly.com/datatable
    # https://dash.plotly.com/datatable/editable

    app.layout = html.Div(
        [
            # drop down list of attribute options detected from data
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                dropDownOpt,
                                dropDownStart,  # select an attribute with the fewest variables initially
                                id="selectedDropDown",
                                multi=False,  # for selecting multiple values set to true
                                clearable=False,
                            )
                        ],
                        style={"width": "49%"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                options=[""],
                                value=[],  # select an attribute with the fewest variables initially
                                id="selectableDropDownExclude",
                                placeholder="Select elements to exclude",
                                multi=True,  # for selecting multiple values set to true
                                clearable=True,
                            ),
                            dcc.Dropdown(
                                options=[""],
                                value=[],  # select an attribute with the fewest variables initially
                                id="selectableDropDownInclude",
                                placeholder="Select elements to include",
                                multi=True,  # for selecting multiple values set to true
                                clearable=True,
                                disabled=False,
                            ),
                        ],
                        style={"width": "49%"},
                    ),
                ],
                style={"padding": "0px 5px"},
            ),
            # main figure
            html.Div(
                [
                    # plotly figure
                    html.Div(
                        [dcc.Graph(id="plotly_figure")],
                        style={
                            "width": "800",
                            "height": "600",
                            "display": "inline-block",
                        },
                    ),
                    # slider for clustering of points
                    html.Div(
                        [
                            # html.Label("Clustering"),
                            html.Div(
                                [
                                    dcc.Slider(
                                        0,
                                        2,
                                        value=0,
                                        id="slider-rounding",
                                        disabled=False,
                                        marks=None,
                                        vertical=True,
                                    ),
                                ]
                            ),
                            # html.Div(id='slider-output'),
                        ],
                        style={"display": "inline-block"},
                    ),
                    # report buttons
                    html.Div(
                        [
                            # report1 button
                            html.Div(
                                [
                                    html.Button(
                                        "Outlier report", id="report_1", n_clicks=0
                                    ),
                                ],
                                style={"margin-left": "15px", "margin-top": "15px"},
                            ),
                            # report2 button
                            html.Div(
                                [
                                    html.Button(
                                        "Cluster report",
                                        id="report_2",
                                        n_clicks=0,
                                        hidden=True,
                                    ),
                                ],
                                style={"margin-left": "15px", "margin-top": "15px"},
                            ),
                            # report2/clustering slider
                            html.Div(
                                [
                                    dcc.Slider(
                                        0,
                                        1,
                                        value=0,
                                        id="slider-clustering",
                                        disabled=False,
                                        marks=None,
                                        vertical=False,
                                    ),
                                ],
                                style={},
                            ),
                            # report3 button
                            html.Div(
                                [
                                    html.Button(
                                        "Trendline report", id="report_3", n_clicks=0
                                    ),
                                ],
                                style={"margin-left": "15px", "margin-top": "15px"},
                            ),
                            # report4 button
                            html.Div(
                                [
                                    html.Button("Report 4", id="report_4", n_clicks=0),
                                ],
                                style={"margin-left": "15px", "margin-top": "15px"},
                            ),
                            # report5 button
                            html.Div(
                                [
                                    html.Button("Report 5", id="report_5", n_clicks=0),
                                ],
                                style={"margin-left": "15px", "margin-top": "15px"},
                            ),
                        ],
                        style={"display": "inline-block"},
                    ),
                ]
            ),
            # first line of buttons below the plot
            html.Div(
                [
                    # toggle switch to change between tracking points and viewing historical data
                    html.Div(
                        [
                            daq.ToggleSwitch(
                                id="toggle-timeseries",
                                disabled=not "_PermissionDateTime" in hover_data,
                                value=False,
                            ),
                            html.Div(id="toggle-timeseries-out"),
                        ],
                        style={
                            "margin-left": "15px",
                            "margin-top": "15px",
                            "display": "inline-block",
                        },
                    ),
                    html.Div(
                        [
                            daq.ToggleSwitch(
                                id="toggle-hoverinfo",
                                value=False,
                            ),
                            html.Div(id="toggle-hoverinfo-out"),
                        ],
                        style={
                            "margin-left": "15px",
                            "margin-top": "15px",
                            "display": "inline-block",
                        },
                    ),
                    html.Div(
                        [
                            daq.ToggleSwitch(
                                id="toggle-rolesurface",
                                value=False,
                            ),
                            html.Div(id="toggle-rolesurface-out"),
                        ],
                        style={
                            "margin-left": "15px",
                            "margin-top": "15px",
                            "display": "inline-block",
                        },
                    ),
                    # slider for dates
                    html.Div(
                        [
                            dcc.Slider(
                                0,
                                len(dtformat) - 1,
                                1,
                                value=len(dtformat) - 1,
                                # dttimes.min(), dttimes.max(),
                                # value = dttimes.max(),
                                id="slider-dates",
                                disabled=not "_PermissionDateTime" in hover_data,
                                marks=marks,
                            ),
                            # html.Div(id='slider-output'),
                        ],
                        style={"margin-left": "30px", "margin-top": "15px"},
                    ),
                ]
            ),
            # second line of buttons below the plot
            html.Div(
                [
                    # Save figure button
                    html.Div(
                        [
                            html.Button("Save plot", id="submit_plot", n_clicks=0),
                        ],
                        style={
                            "margin-left": "15px",
                            "margin-top": "15px",
                            "display": "inline-block",
                        },
                    ),
                    # plotly stop running button
                    html.Div(
                        [
                            daq.StopButton(id="stop_button", n_clicks=0),
                        ],
                        style={
                            "margin-left": "15px",
                            "margin-top": "15px",
                            "display": "inline-block",
                        },
                    ),
                    # Save file button and text entry
                    html.Div(
                        [
                            dcc.Input(
                                id="input_filename",
                                type="text",
                                placeholder="File name",
                                value="",
                            ),
                            html.Button("Save file", id="submit_file", n_clicks=0),
                        ],
                        style={
                            "margin-left": "15px",
                            "margin-top": "15px",
                            "display": "inline-block",
                        },
                    ),
                ]
            ),
            # data table
            html.Div(
                [
                    dash_table.DataTable(
                        id="selected_points_table",
                        columns=[
                            {
                                "name": "{}".format(a),
                                "id": "{}".format(a),
                            }
                            for a in hover_data
                        ],
                        # data=[{a: "" for a in hover_data}],
                        editable=True,
                        row_deletable=True,
                        # export_format='xlsx',
                        # export_headers='display',
                        # merge_duplicate_headers=True
                    )
                ],
                style={
                    "margin-left": "15px",
                    "margin-top": "15px",
                },
            ),
            # data being transferred to call back functions
            # NOTE think about using @cache.memoize() to store some of these bigger dataframes to reduce the compute bottleneck? https://dash.plotly.com/sharing-data-between-callbacks
            # dcc.Store(
            #     data=dfID[selectedData].to_json(orient="split"),
            #     id="identityRepresentations",
            # ),
            # dcc.Store(
            #     data=dataModel.rawPermissionData.astype(np.int8).to_json(
            #         orient="table"
            #     ),
            #     id="rawPermissionData",
            # ),
            # dcc.Store(data=dfRoles.to_json(orient="table"), id="roleData"),
            dcc.Store(data=dataModel.joinKeys["permission"], id="uidAttr"),
            dcc.Store(data=dataModel.joinKeys["role"], id="roleAttr"),
            dcc.Store(data=hover_data, id="hover_data"),
            dcc.Store(data=os.getpid(), id="pid"),
            dcc.Store(data="info", id="info"),
            dcc.Store(
                data=None, id="identitiesPlotted"
            ),  # store the plotted identity data
            dcc.Store(data=None, id="figureLayout"),
            html.Div(id="output"),
        ]
    )

    global dfID_g, dfRole_g, dfPerm_g, dfPriv_g, colour_dict_g

    dfID_g = dfID[selectedData]
    dfRole_g = dfRoles
    dfPerm_g = dataModel.permissionData
    dfPriv_g = dataModel.privilegedData
    colour_dict_g = colour_dict

    print("     Web app created")

    return app


# plotly figure updates
@app.callback(
    Output("plotly_figure", "figure"),
    Output("identitiesPlotted", "data"),
    # Input('plotly_figure', 'figure'),
    State("plotly_figure", "relayoutData"),
    # State("identityRepresentations", "data"),
    # State("roleData", "data"),
    Input("selectedDropDown", "value"),
    Input("selectableDropDownExclude", "value"),
    Input("selectableDropDownInclude", "value"),
    State("uidAttr", "data"),
    State("roleAttr", "data"),
    State("hover_data", "data"),
    Input("toggle-timeseries", "value"),
    Input("toggle-hoverinfo", "value"),
    Input("toggle-rolesurface", "value"),
    Input("slider-dates", "value"),
    Input("slider-rounding", "value"),
    Input("slider-clustering", "value"),
)
def update_graph(
    figLayout,
    # dfIDjson,
    # dfRolejson,
    attribute,
    elementsExclude,
    elementsInclude,
    uidAttr,
    roleAttr,
    hover_data,
    trackingToggle,
    hoverinfoToggle,
    rolesurfaceToggle,
    sliderDateValue,
    sliderRoundValue,
    sliderClusterValue,
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
    dfID = dfID_g.copy()  # pd.read_json(dfIDjson, orient="split")
    dfRole = dfRole_g.copy()  # pd.read_json(dfRolejson, orient="table")

    # simplify the elements exclude info list for processing by the pandas df
    includeInfo = [e.split(": ") for e in elementsInclude]
    excludeInfo = [e.split(": ") for e in elementsExclude]

    dfID[hover_data] = dfID[hover_data].astype(str)
    dataColumns = list(dfID.columns)
    dims = sum(1 for x in list(dataColumns) if x.startswith("Dim"))
    colour_dict = colour_dict_g.copy()

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

        fig, plotTitle = track_elements(
            dfIDIncl, uidAttr, attribute, hover_data, colour_dict[attribute]
        )

        dfPlotExcl = dfIDExcl
        dfPlotIncl = dfIDIncl

    #  ---------- Cluster data around reduced spatial resolution ----------
    elif sliderRoundValue > 0:

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
    if len(dfRole) > 0:

        # fig = add_roles(fig, dfID, dfRole, uidAttr, roleAttr)

        fig = plot_roles(
            fig,
            dfPlotIncl,
            dfPlotExcl,
            dfRole,
            uidAttr,
            roleAttr,
            rolesurfaceToggle and not trackingToggle,   # link roles only if not tracking
            colour_dict[roleAttr],
        )

    # ---------- Scale and format the plot ----------

    r = 0.2  # The extra bit to add to the graphs for scaling
    xMin = dfID["Dim0"].min() - r
    xMax = dfID["Dim0"].max() + r
    yMin = dfID["Dim1"].min() - r
    yMax = dfID["Dim1"].max() + r
    zMin = dfID["Dim2"].min() - r
    zMax = dfID["Dim2"].max() + r

    fig.update_layout(
        title="<br>".join(wrap(plotTitle, width=70)),  # set the title
        clickmode="event+select",  # all for data to be collected by clicking
        width=800,
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
            fig.layout.scene.xaxis.showspikes = False
            fig.layout.scene.yaxis.showspikes = False
            fig.layout.scene.zaxis.showspikes = False

        # if there is a hovertemplate, remove frivilous info
        elif f.hovertemplate is not None:

            # remove the co-ordinate info (not useful)
            f.hovertemplate = f.hovertemplate.replace(
                "Dim0=%{x}<br>Dim1=%{y}<br>Dim2=%{z}<br>", ""
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

    # re-load the figure with the previous camera angle
    if figLayout is not None:
        fig.layout.scene.camera = figLayout.get("scene.camera")

    print("     Plot updated\n")
    return fig, dfIDIncl.to_json(orient="split")


@app.callback(
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
@app.callback(
    Output("report_1", "n_clicks"),
    Input("report_1", "n_clicks"),
    State("identitiesPlotted", "data"),
    State("uidAttr", "data"),
    State("slider-dates", "value"),
    State("selectedDropDown", "value"),
)
def report_1(click, idPlotted, uid, sliderDate, selectedAttr):

    """
    Create a report which identifies which identities are "outliers" relative to their peers in their respective element. It models all identities as seen when the slider-rounding value = 0.

    It models all identities which are selected ONLY from the "Select elements to include/exclude" drop down, it is NOT impacted by the cluster size slider (id slider-clustering) as this report does not perform the getClusterLimit function.
    """

    if click and idPlotted is not None:

        idPlotted = pd.read_json(idPlotted, orient="split")
        permData = dfPerm_g.copy()
        privData = dfPriv_g.copy()
        selectedAttr = selectedAttr.split(":")[0]

        reporting.report_1(
            idPlotted,
            permData,
            privData,
            uid,
            sliderDate,
            "OutlierReport",
            selectedAttr,
        )

    return 0


# report 2
@app.callback(
    Output("report_2", "n_clicks"),
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
        permData = dfPerm_g.copy()
        selectedAttr = selectedAttr.split(":")[0]

        reportName = "ClusterReport"

        reporting.report_2(
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

    return 0


# report 3
@app.callback(
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
        permData = dfPerm_g.copy()
        selectedAttr = selectedAttr.split(":")[0]

        reportName = "TrendReport"

        reporting.report_3(
            idPlotted, permData, uid, selectedAttr, hover_data, reportName
        )

    return 0


# killing the dash server
@app.callback(
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
@app.callback(
    Output("selected_points_table", "data"),
    State("selected_points_table", "data"),
    State("hover_data", "data"),
    Input("plotly_figure", "clickData"),
)
def add_row(rows, hover_data, inputSelection):

    # if there is no input exit
    if inputSelection is None:
        # rows = None
        return rows

    # extract the data
    inputData = inputSelection["points"][0]["customdata"]

    # if the selected data is cluster information just remove that info
    if inputData[0].find("Cluster ") > -1:
        inputData = inputData[2:]

    # if the input data still does't match the hover_data categories, pass
    if len(inputData) != len(hover_data):
        pass
    else:
        print("----- Data added -----")
        d = {}
        for n, hd in enumerate(hover_data):
            d[hd] = inputData[n]
        if rows == [] or rows is None:
            rows = [d]
        elif all(rows[-1][k] == "" for k in list(rows[-1])):
            rows = [d]
        else:
            rows.append(d)

    return rows


# action to perform when row is removed
@app.callback(
    Output("output", "children"), Input("selected_points_table", "data_previous")
)
def remove_rows(previous):
    if previous is not None:
        return ""  # [f'Just removed {row}' for row in previous if row not in current]


# to save file name prompts and checks
@app.callback(
    Output("input_filename", "placeholder"),
    Output("input_filename", "value"),
    Output("submit_file", "n_clicks"),
    Input("submit_file", "n_clicks"),
    State("selected_points_table", "data"),
    State("input_filename", "value"),
)
def save_file(click, tab_data, filename):

    # Save data as long as there is information etc
    if not click:
        placeholder = "Select data"
    elif tab_data == [] or tab_data is None:
        placeholder = "Select data"
    elif filename == "":
        placeholder = "Set file name"

    else:
        fileName = f"{os.path.expanduser('~')}\\Downloads\\{filename}.csv"
        if not os.path.exists(fileName):
            pd.DataFrame.from_records(tab_data).to_csv(fileName, index=False)
            placeholder = "File saved"
            print(f"     File saved at {fileName}")
        else:
            placeholder = "File exists"

    # always reset the text
    output = ""

    print(f"----- Save file, {placeholder}, {output} -----\n")
    return placeholder, output, 0


@app.callback(
    Output("toggle-timeseries-out", "children"),
    Output("toggle-hoverinfo-out", "children"),
    Output("toggle-rolesurface-out", "children"),
    Output("toggle-rolesurface", "disabled"),
    Output("slider-dates", "disabled"),
    Output("slider-rounding", "disabled"),
    Output("slider-clustering", "disabled"),
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

    """
    If displaying time tracking data then:
        - No date selection (all being displayed anyway)
        - No rounding of data 
        - No clustering of data
        - No role surface creation 
    """
    if toggleTimeSeriesValue:
        toggleTimeSeries = "Tracking data"
        sliderDateDisabled = True
        sliderRoundDisabled = True
        sliderClusterDisabled = True
        roleSurfaceInfoDisabled = True

    """
    If there is no rounding being performed:
        - There is no threshold of rounding to be performed
    """
    if sliderRoundValue == 0:
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
    )


@app.callback(
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

    report3_child = "Trendline Report"
    report3_disabled = False

    if sliderRounding > 0 or trackingToggle:
        report1_child = "Disabled"
        report1_disabled = True

    if sliderRounding == 0 or trackingToggle:
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


@app.callback(
    Output("selectableDropDownExclude", "options"),
    Output("selectableDropDownInclude", "options"),
    Input("selectedDropDown", "value"),
    Input("identitiesPlotted", "data"),
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
        identitiesAll = dfID_g  # pd.read_json(identitiesAll, orient="split")
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


"""
NOTE need to figure out how to combine outputs
@app.callback(Output('clear_table', 'n_clicks'),
    Output('selected_points_table', 'data'),
    Input('clear_table', 'n_clicks'),
    Input('selected_points_table', 'data'),
    )
def clear_table(click, table):

    if click:
        return 0, []
"""
