import os

import dash_daq as daq
from dash import dash_table, dcc, html


def pageone_layout(
    dataModel,
    dfID,
    hover_data,
    marks,
    dtformat,
    dropDownOpt,
    dropDownList,
    dropDownStart,
    sliderValue,
):

    layout = html.Div(
        [
            # main figure
            html.Div(
                [
                    # main plot and controls
                    html.Div(
                        [
                            # plotly figure
                            html.Div(
                                [dcc.Graph(id="plotly_figure")],
                                style={
                                    "width": "70%",
                                    "height": "600",
                                    "display": "inline-block",
                                },
                            ),
                            # first line of buttons below the plot
                            html.Div(
                                [
                                    # toggle, tracking data
                                    html.Div(
                                        [
                                            daq.ToggleSwitch(
                                                id="toggle-timeseries",
                                                disabled=not "Permission Datetime"
                                                in dfID.columns,
                                                value=False,
                                            ),
                                            html.Div(
                                                id="toggle-timeseries-out",
                                            ),
                                        ],
                                        style={
                                            "display": "inline-block",
                                            "marginRight": "10px",
                                            "marginLeft": "10px",
                                        },
                                    ),
                                    # toggle, hover info
                                    html.Div(
                                        [
                                            daq.ToggleSwitch(
                                                id="toggle-hoverinfo",
                                                value=False,
                                            ),
                                            html.Div(
                                                id="toggle-hoverinfo-out",
                                            ),
                                        ],
                                        style={
                                            "display": "inline-block",
                                            "marginRight": "10px",
                                            "marginLeft": "10px",
                                        },
                                    ),
                                    # toggle, role surface
                                    html.Div(
                                        [
                                            daq.ToggleSwitch(
                                                id="toggle-rolesurface",
                                                value=False,
                                            ),
                                            html.Div(
                                                id="toggle-rolesurface-out",
                                            ),
                                        ],
                                        style={
                                            "display": "inline-block",
                                            "marginRight": "10px",
                                            "marginLeft": "10px",
                                        },
                                    ),
                                    # mesh attribute selection
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                dropDownList,
                                                None,  # select an attribute with the fewest variables initially
                                                id="selectedMeshAttr",
                                                placeholder="Select mesh attribute",
                                                multi=False,  # for selecting multiple values set to true
                                                clearable=False,
                                                disabled=True,
                                            ),
                                        ],
                                        style={
                                            "display": "inline-block",
                                            "marginRight": "10px",
                                            "marginLeft": "10px",
                                            "width": 300,
                                        },
                                    ),
                                ],
                                style={
                                    # "display": "inline-block",
                                    "marginRight": "10px",
                                    "marginLeft": "10px",
                                },
                            ),
                        ],
                        style={
                            "display": "inline-block",
                            "marginRight": "10px",
                            "marginLeft": "10px",
                        },
                    ),
                    # slider for clustering of points
                    html.Div(
                        [
                            # html.Label("Clustering"),
                            dcc.Slider(
                                -0.5,
                                sliderValue,
                                step=0.5,
                                value=-0.5,
                                id="slider-rounding",
                                disabled=False,
                                marks=None,
                                vertical=True,
                            ),
                            # html.Div(id='slider-output'),
                        ],
                        style={
                            "display": "inline-block",
                            "marginRight": "10px",
                            "marginLeft": "10px",
                        },
                    ),
                    # filtering and reporting controls/buttons
                    html.Div(
                        [
                            # attribute dropdown
                            dcc.Dropdown(
                                dropDownOpt,
                                dropDownStart,  # select an attribute with the fewest variables initially
                                id="selectedDropDown",
                                multi=False,  # for selecting multiple values set to true
                                clearable=False,
                            ),
                            # elements to include
                            dcc.Dropdown(
                                options=[""],
                                value=[],  # select an attribute with the fewest variables initially
                                id="selectableDropDownExclude",
                                placeholder="Select elements to exclude",
                                multi=True,  # for selecting multiple values set to true
                                clearable=True,
                                # style={"max-height": "280px", "overflow-y": "auto"},
                            ),
                            # elements to exclude
                            dcc.Dropdown(
                                options=[""],
                                value=[],  # select an attribute with the fewest variables initially
                                id="selectableDropDownInclude",
                                placeholder="Select elements to include",
                                multi=True,  # for selecting multiple values set to true
                                clearable=True,
                                disabled=False,
                                # style={"max-height": "280px", "overflow-y": "auto"},
                            ),
                            # find identities
                            dcc.Dropdown(
                                options=list(
                                    dfID[dataModel.joinKeys["permission"]].unique()
                                ),
                                value=[],  # select an attribute with the fewest variables initially
                                id="selectableIdentities",
                                placeholder="Select identities to highlight",
                                multi=True,  # for selecting multiple values set to true
                                clearable=True,
                                disabled=False,
                                # style={"max-height": "280px", "overflow-y": "auto"},
                            ),
                            # report1 button
                            html.Button(
                                "Outlier report",
                                id="report_1",
                                n_clicks=0,
                                style={
                                    "margin-top": "15px",
                                },
                            ),
                            html.Div(
                                id="report_1_sub",
                            ),
                            # report1/outlier tolerance
                            dcc.Slider(
                                0,  # No tolerance, any identity not at the median is an outlier
                                2,  # double the tolerance
                                0.5,  # 5 degrees of outlier tolerance
                                value=1,  # standard tolerance
                                id="slider-outliertolerance",
                                disabled=False,
                                vertical=False,
                                marks={
                                    0: "Strict",
                                    0.5: "Mild",
                                    1: "Standard",
                                    1.5: "Loose",
                                    2: "Very loose",
                                },
                                included=False,
                            ),
                            # report2 button
                            html.Button(
                                "Cluster report",
                                id="report_2",
                                n_clicks=0,
                                hidden=True,
                                style={
                                    "margin-top": "15px",
                                },
                            ),
                            html.Div(
                                id="report_2_sub",
                            ),
                            # report2/clustering slider
                            dcc.Slider(
                                0,
                                1,
                                value=0,
                                id="slider-clustering",
                                disabled=False,
                                marks=None,
                                vertical=False,
                            ),
                            # report3 button
                            html.Button(
                                "BETA Identity changes report",
                                id="report_3",
                                n_clicks=0,
                                style={
                                    # "margin-top": "15px",
                                },
                            ),
                            html.Div(
                                id="report_3_sub",
                            ),
                            # report3/time slider
                            dcc.Slider(
                                # NOTE this is exactly the same as the date slide on the main graph EXCEPT it has an extra value which allows you to turn off the volume displayer (-1)
                                -1,
                                len(dtformat) - 1,
                                1,
                                value=-1,
                                # dttimes.min(), dttimes.max(),
                                # value = dttimes.max(),
                                marks={
                                    n: {"label": d}
                                    for n, d in enumerate(
                                        ["Off"] + [""] * len(dtformat), -1
                                    )
                                },
                                id="slider-meshtime",
                                disabled=False,
                                vertical=False,
                            ),
                            # report4 button
                            html.Button(
                                "BETA Privilegd Access",
                                id="report_4",
                                n_clicks=0,
                                style={"margin-top": "15px"},
                            ),
                            html.Div(
                                id="report_4_sub",
                            ),
                            # report5 button
                            html.Button(
                                "BETA Rare Access report",
                                id="report_5",
                                n_clicks=0,
                                style={"margin-top": "15px"},
                            ),
                            html.Div(
                                id="report_5_sub",
                            ),
                        ],
                        style={
                            "display": "inline-block",
                            "width": "20%",
                            # "draggable": "true",
                            "vertical-align": "bottom",
                        },
                    ),
                ]
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
                        disabled=not "Permission Datetime" in hover_data,
                        marks=marks,
                    ),
                    # html.Div(id='slider-output'),
                ],
                style={"margin-left": "30px", "margin-top": "15px"},
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
                        data=[],
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
            dcc.Store(data=dataModel.joinKeys["roleidkey"], id="roleAttr"),
            dcc.Store(data=hover_data, id="hover_data"),
            dcc.Store(data=os.getpid(), id="pid"),
            dcc.Store(data="info", id="info"),
            dcc.Store(data=[], id="table_info"),
            dcc.Store(
                data=None, id="identitiesPlotted"
            ),  # store the plotted identity data
            dcc.Store(data=None, id="figureLayout"),
            html.Div(id="output"),
        ]
    )

    return layout
