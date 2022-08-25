import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, dash_table
import pandas as pd
import os
from datetime import datetime
import dash_daq as daq
import webbrowser
import plotly.graph_objects as go
from plotly.colors import qualitative as colours
from plotly.colors import hex_to_rgb
import numpy as np
from textwrap import wrap
import reporting
from utilities import clusterData

pd.options.mode.chained_assignment = None


'''
TODO
    - How to pass an object to the dash app, not just the attributes? Performance
    issues?
    - Optimise dash computations with caching, parallelisation: https://dash.plotly.com/sharing-data-between-callbacks
'''

# Theme stuff: https://dash.plotly.com/external-resources 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = [open('data\\bWLwgP.css', "r")]
app = Dash(__name__, external_stylesheets=external_stylesheets)

def launchApp(dataModel : object, name = ""):
    
    appObj = createInteractivePlot(dataModel)   
    # webbrowser.open("http://127.0.0.1:8050/", new = 0, autoraise = True)
    try:
        appObj.run_server(debug = True)
    except Exception as e:
        print(e)
        launchApp(appObj)

# create the dash application
def createInteractivePlot(dataModel : object):

    # get the list of desireable attributes
    # Remove the dim values
    # Remove any attributes with over 100 unique values (makes plotly run too slow and clutters the graphs)
    # Remove the key words "Count" and "Unnamed: 0" which are artefacts of the plotting

    print("---Creating web server for plotting---")
    df = dataModel.mdsResults

    # Specify what data from the dataframe is to be selected and shared with the webbrowser
    selectedData = [d for d in sorted(list(df.columns)) 
        if d.lower().find("unnamed") == -1 and              # removed the unnamed columns
        d.find(dataModel.joinKeys["permission"]) == -1]     # remove the permission uid

    # Specify what data from the dataframe will be included in the hovertext
    hover_data = [s for s in selectedData if s.lower().find("dim") == -1]
    # hover_data.remove("timeUnix")

    r = 0.2         # The extra bit to add to the graphs for scaling
    xMinR = df["Dim0"].min()-r
    xMaxR = df["Dim0"].max()+r
    yMinR = df["Dim1"].min()-r
    yMaxR = df["Dim1"].max()+r
    zMinR = df["Dim2"].min()-r
    zMaxR = df["Dim2"].max()+r        

    marks = {}
    if "_DateTime" in hover_data:
        hover_data.remove("_DateTime")
        selectedData.append("_PermissionDateTime")
        hover_data.append("_PermissionDateTime")
        df = df.sort_values(["_DateTime", dataModel.joinKeys["identity"]], ascending = True)

        # convert the datetime object into a human readable time
        df["_PermissionDateTime"] = df["_DateTime"].apply(lambda x: datetime.fromtimestamp(int(x)).strftime("%m/%d/%Y, %H:%M:%S"))

        # create the marks for the slider
        dtformat = df["_PermissionDateTime"].unique()
        # dttimes = df["_DateTime"].unique()

        marks = {n: {'label': d} for n, d in enumerate(dtformat)}
        # marks = {n: {'label': d} for n, d in zip(dttimes, dtformat)}


    attrArray = np.array([[r, len(df[r].unique())] for r in hover_data])
    dropDownStart = ": ".join(attrArray[attrArray[:, 1].astype(int).argsort()][0])
    dropDownOpt = [": ".join(a) for a in attrArray]

    # for values which are numeric, convert their values into a ranked position so that 
    # on the heat maps it can show up easily
    # NOTE this is not actually very useful as it assumes that data that is chronological is related
    '''
    dfSelect = df[hover_data]
    dfRanked = dfSelect.rank(numeric_only = True, method = 'dense').astype(int)
    df[list(dfRanked.columns)] = dfRanked
    '''

    # make data point selection
    # https://dash.plotly.com/interactive-graphing
    # https://dash.plotly.com/datatable
    # https://dash.plotly.com/datatable/editable 

    app.layout = html.Div([
        # drop down list of attribute options detected from data 
        html.Div([
            html.Div([
                dcc.Dropdown(
                    dropDownOpt,
                    dropDownStart,     # select an attribute with the fewest variables initially
                    id='selectedDropDown',
                    multi = False,      # for selecting multiple values set to true
                    clearable = False
                )
            ],
            style={'width': '49%', 'display': 'inline-block'})
        ], style={
            'padding': '0px 5px'
        }),

        # main figure
        html.Div([

            # range slider for x axis
            html.Div([
                dcc.RangeSlider(xMinR, xMaxR,
                        value=[xMinR, xMaxR],
                        id='slider-xAxis',
                        disabled = False,
                        marks = None,
                        vertical = True,
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": True}
                ),
                # html.Div(id='slider-output'),
                ], style={"margin-left": "15px", 'display': 'inline-block'}
            ),

            # range slider for y axis
            html.Div([
                dcc.RangeSlider(yMinR, yMaxR,
                        value=[yMinR, yMaxR],
                        id='slider-yAxis',
                        disabled = False,
                        marks = None,
                        vertical = True,
                        allowCross=False, 
                        tooltip={"placement": "bottom", "always_visible": True}
                ),
                # html.Div(id='slider-output'),
                ], style={'display': 'inline-block'}
            ),

            # range slider for z axis
            html.Div([
                dcc.RangeSlider(zMinR, zMaxR,
                        value=[zMinR, zMaxR],
                        id='slider-zAxis',
                        disabled = False,
                        marks = None,
                        vertical = True,
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": True}
                ),
                # html.Div(id='slider-output'),
                ], style={'display': 'inline-block'}
            ),

            # plotly figure
            html.Div([
                dcc.Graph(
                    id='plotly_figure'
                )
            ], style={'width': '800', 'height':'600', 'display': 'inline-block'}
            ),

            # slider for clustering of points
            html.Div([
                dcc.Slider(0, 1,
                        value=0,
                        id='slider-rounding',
                        disabled = False,
                        marks = None,
                        vertical = True
                ),
                # html.Div(id='slider-output'),
                ], style={'display': 'inline-block'}
            ),
    
            # report buttons
            html.Div([
                # report1 button
                html.Div([
                    html.Button(
                        'Report 1', 
                        id='report_1', 
                        n_clicks=0
                        ),
                    ], style={"margin-left": "15px", "margin-top": "15px"}
                ),

                # report2 button
                html.Div([
                    html.Button(
                        'Report 2', 
                        id='report_2', 
                        n_clicks=0,
                        hidden = True
                        ),
                    ], style={"margin-left": "15px", "margin-top": "15px"}
                ),

                # report2/clustering slider
                html.Div([
                    dcc.Slider(0, 1,
                            value=0,
                            id='slider-clustering',
                            disabled = False,
                            marks = None,
                            vertical = False
                    ),
                    ], style={}
                ),


                # report3 button
                html.Div([
                    html.Button(
                        'Report 3', 
                        id='report_3', 
                        n_clicks=0
                        ),
                    ], style={"margin-left": "15px", "margin-top": "15px"}
                ),

                # report4 button
                html.Div([
                    html.Button(
                        'Report 4', 
                        id='report_4', 
                        n_clicks=0
                        ),
                    ], style={"margin-left": "15px", "margin-top": "15px"}
                ),

                # report5 button
                html.Div([
                    html.Button('Report 5', id='report_5', n_clicks=0),
                    ], style={"margin-left": "15px", "margin-top": "15px"}
                ),
            ], style={'display': 'inline-block'})
        ]),
    
        # first line of buttons below the plot
        html.Div([
            
            # toggle switch to change between tracking points and viewing historical data
            html.Div([
                daq.ToggleSwitch(
                    id='toggle-in',
                    disabled = not "_PermissionDateTime" in hover_data,
                    value=False, 
                ), 
                html.Div(id='toggle-out'),
                ], style={"margin-left": "15px", "margin-top": "15px", "display": "inline-block"}
            ),
            
            # slider for dates
            html.Div([
                dcc.Slider(
                        0, len(dtformat)-1, 1,
                        value=len(dtformat)-1,
                        # dttimes.min(), dttimes.max(),
                        # value = dttimes.max(),
                        id='slider-dates',
                        disabled = not "_PermissionDateTime" in hover_data,
                        marks = marks
                ),
                # html.Div(id='slider-output'),
                ], style={"margin-left": "30px", "margin-top": "15px"}
            )
        ]),

        # second line of buttons below the plot
        html.Div([

            # Save figure button
            html.Div([
                html.Button('Save plot', id='submit_plot', n_clicks=0),
                ], style={"margin-left": "15px", "margin-top": "15px", "display": "inline-block"}
            ),

            # plotly stop running button
            html.Div([
                daq.StopButton(
                    id='stop_button',
                    n_clicks=0),
                ], style={"margin-left": "15px", "margin-top": "15px", "display": "inline-block"}
            ),

            # Save file button and text entry
            html.Div([
                dcc.Input( 
                    id="input_filename", 
                    type="text", 
                    placeholder="File name", 
                    value = ""), 
                html.Button('Save file', id='submit_file', n_clicks=0),
                ], style={"margin-left": "15px", "margin-top": "15px", "display": "inline-block"}
            ),
        ]),

        # data table
        html.Div([
            dash_table.DataTable(
                    id='selected_points_table',
                    columns=[{
                        'name': '{}'.format(a),
                        'id': '{}'.format(a),
                    } for a in hover_data],
                    # data=[{a: "" for a in hover_data}],
                    editable=True,
                    row_deletable=True,
                    # export_format='xlsx',
                    # export_headers='display',
                    # merge_duplicate_headers=True
                )], style={
                "margin-left": "15px", 
                "margin-top": "15px", 
                }),

        # data being transferred to call back functions
        dcc.Store(data = df[selectedData].to_json(orient='split'), id = "identityRepresentations"),
        dcc.Store(data = dataModel.rawPermissionData.to_json(orient='split'), id = "rawPermissionData"),
        dcc.Store(data = dataModel.identityData.to_json(orient='split'), id = "identityData"),
        dcc.Store(data = dataModel.joinKeys["identity"], id = "uidAttr"),
        dcc.Store(data = os.getpid(), id = "pid"),
        dcc.Store(data = hover_data, id = "hover_data"),
        dcc.Store(data = "info", id = "info"),

        html.Div(id='output')
    ])

    print("     Web app created")
    return app

# plotly figure updates
@app.callback(Output('plotly_figure', 'figure'),
    Input('identityRepresentations', 'data'),
    Input('selectedDropDown', 'value'),
    Input('uidAttr', 'data'), 
    Input('hover_data', 'data'), 
    Input('toggle-in', 'value'), 
    Input('slider-dates', 'value'), 
    Input('slider-rounding', 'value'),
    Input('slider-clustering', 'value'),
    Input('slider-xAxis', 'value'),
    Input('slider-yAxis', 'value'),
    Input('slider-zAxis', 'value')
    )
def update_graph(dfjson, attribute, uidAttr, hover_data, trackingToggle, sliderDateValue, sliderRoundValue, sliderClusterValue, sliderXScale, sliderYScale, sliderZScale):

    '''
    Take in the raw data and selected information and create visualisation
    '''

    print("----- Updating plotting information -----")

    print("Variables from the dash board")
    print(f'''
        attribute: {str(attribute)},
        trackingToggle: {str(trackingToggle)},
        sliderDateValue: {str(sliderDateValue)},
        sliderRoundValue: {str(sliderRoundValue)},
        sliderClusterValue: {str(sliderClusterValue)},
        sliderXScale: {str(sliderXScale)},
        sliderYScale: {str(sliderYScale)},
        sliderZScale: {str(sliderZScale)},
        ''')

    df = pd.read_json(dfjson, orient='split')
    df[hover_data] = df[hover_data].astype(str)
    dataColumns = list(df.columns)
    dims = sum(1 for x in list(dataColumns) if x.startswith ("Dim"))
    allTimes = df["_DateTime"].unique()
    allTimesFormat = df["_PermissionDateTime"].unique()

    # remove the count info to match to the data frame
    attribute = attribute.split(":")[0]
    plotTitle = f"{dims}D visualising {attribute} for {len(df)} data points"

    # ---------- Track attributes across the time inputs ----------
    if trackingToggle:

        print(f"     Tracking historical data with 3D plotting for {attribute}")

        uniqueIDs = df[uidAttr].unique()
        allSizes = np.linspace(4, 12, len(allTimes)).astype(int)

        fig = go.Figure()
        colourDict = {}

        # create a dictionary to colour the traces depending on the attribute and time of the data
        transparency = np.linspace(0.4, 1, len(allTimes))
        for n_c, c in enumerate(sorted(df[attribute].unique())):
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
            allAttrs = df[attribute].unique()
            data = []
            for attr in allAttrs:
                dfAttrId = df[df[attribute] == attr]
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
            dfTrack = dfTrack.combine_first(pd.DataFrame(columns=df.columns))
            uniqueIDs = dfTrack[attribute].unique()

        # if the selected attribute is the unique identifier, provide all data (there will
        # no combining of data)
        else:
            dfTrack = df

        dfTrack = dfTrack.sort_values("_DateTime")
        uidAttr = attribute

        for uid in uniqueIDs:
            # get all the unique entries for this unique identity
            uiddf = dfTrack[dfTrack[uidAttr] == uid]

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
                        hovertemplate = "<br>".join([f"{h}: %{'{customdata['+str(n)+']}'}" for n, h in enumerate(hover_data)]),
                        # hovertext = 
                        # ['<br>'.join([f"{h}: {uiddf[h].iloc[n]}" for h in hover_data]) for n in range(len(uiddf))],
                        marker=dict(color=selected_colours, size=selected_sizes),
                        line = dict(color=selected_colours),
                        name = name,            # NOTE this must be a string/number
                        # legendgroup = name,     # NOTE this must be a string/number
                        # connectgaps=True        # NOTE for some reason this isn't acutally connecting gaps.... maybe wrong data type for empty? 
                    )
                )

        plotTitle = f"Tracking {len(uniqueIDs)} identities grouped by {attribute} from {allTimesFormat[0]} to {allTimesFormat[-1]}"

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

    #  ---------- Cluster data around reduced spatial resolution ----------
    elif sliderRoundValue > 0:
        
        '''
        plot the current time specified data but scale the dots to represent the relative number of identities
        at that position.

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
        dfMod = df[df["_DateTime"] == df["_DateTime"].unique()[sliderDateValue]].reset_index(drop=True)

        # for all identity attributes, if they are all the same for all aggregated identities report it otherwise set as "Mixed". 
        aggDict = {hd: lambda x: list(set(x))[0] if len(set([str(n) for n in x]))==1 else "Mixed" for hd in hover_data if hd != attribute}
        
        # get the clustered data
        # NOTE maybe create a unique ID number for each cluster so it can be referred to 
        # in reports
        dfPos = clusterData(dfMod, uidAttr, attribute, sliderRoundValue, aggDict)

        if sliderClusterValue > 0:
            dfPos = dfPos[dfPos["_Count"] > sliderClusterValue * dfPos["_Count"].max()]
        
        # hover_data.append("_Count")
        hover_data = sorted(hover_data)

        fig = px.scatter_3d(dfPos,
                x="Dim0", 
                y="Dim1", 
                z="Dim2",
                hover_data = hover_data,
                color = attribute, 
                title = plotTitle, 
                hover_name = "_ClusterID",
                size = '_Count', 
                size_max=40,
                opacity=1
                )

        plotTitle = f"Plotting and overlaying {len(dfMod)} identities for {len(dfPos)} unique positions colored based on {attribute} with a spatial resolution of {sliderRoundValue} from {allTimesFormat[sliderDateValue]}"
                        
    # ---------- Plot the raw identity data ----------
    else:
        print(f"     Plotting data with 3D plotting for {attribute}")

        dfTime = df[df["_DateTime"] == df["_DateTime"].unique()[sliderDateValue]]
        dfTime = dfTime.sort_values(attribute)

        fig = px.scatter_3d(dfTime, 
                x="Dim0", 
                y="Dim1", 
                z="Dim2",
                hover_data = hover_data,
                color = attribute, 
                title = plotTitle, 
                hover_name = uidAttr,  
                )

        plotTitle = f"Plotting {len(dfTime)} identities colored based on {attribute} with full identity information from {allTimesFormat[sliderDateValue]}"

    # ---------- Scale and format the plot ----------

    # get the user selected slider values
    xMin, xMax = sliderXScale
    yMin, yMax = sliderYScale
    zMin, zMax = sliderZScale

    fig.update_layout(
        title = "<br>".join(wrap(plotTitle, width = 70)),       # set the title
        clickmode='event+select',                               # all for data to be collected by clicking
        width=800, 
        height=600, 
        hovermode='closest',
        scene = {
            'xaxis': dict(nticks = 7, range=[xMin, xMax]),
            'yaxis': dict(nticks = 7, range=[yMin, yMax]),
            'zaxis': dict(nticks = 7, range=[zMin, zMax]),
            'aspectmode': 'cube'
            },
    )

    # remove information about the position in the plot because this is not useful
    for f in fig.data:
        if f.hovertemplate is not None:
            f.hovertemplate = f.hovertemplate.replace("Dim0=%{x}<br>Dim1=%{y}<br>Dim2=%{z}<br>", "")

    print("     Plot updated\n")
    return fig

@app.callback(Output('submit_plot', 'n_clicks'),
    Input('submit_plot', 'n_clicks'),
    Input('plotly_figure', 'figure'),
    Input('info', 'data'),
    Input('selectedDropDown', 'value')
    )
def save_plot(click, fig, info, selectedAttr):

    if click:

        print("----- Saving plot -----")

        dims = sum([l.find("axis")>0 for l in list(fig["layout"]["scene"])])
        selectedAttr = selectedAttr.split(":")[0]
        plotName = f'{os.path.expanduser("~")}\\Downloads\\{info}_{dims}D_{selectedAttr}_{datetime.now().strftime("%y%m%d.%H%M")}.html'
        go.Figure(fig).write_html(plotName)
        
        print(f"     Plot saved at {plotName}\n")

    return 0

# report 1
@app.callback(Output('report_1', 'n_clicks'),
    Input('report_1', 'n_clicks'),
    Input('identityRepresentations', 'data'),
    Input('rawPermissionData', 'data'),
    Input('identityData', 'data'),
    Input('uidAttr', 'data'),
    Input('selectedDropDown', 'value'),
    )
def report_1(click, idRep, idData, permData, uid, selectedAttr):

    if click:

        idRep = pd.read_json(idRep, orient='split')
        idData = pd.read_json(idData, orient='split')
        permData = pd.read_json(permData, orient='split')
        selectedAttr = selectedAttr.split(":")[0]

        reporting.report_1(idRep, idData, permData, uid, selectedAttr)

    return 0

# report 2
@app.callback(Output('report_2', 'n_clicks'),
    Input('report_2', 'n_clicks'),
    Input('plotly_figure', 'figure'),
    Input('identityRepresentations', 'data'),
    Input('rawPermissionData', 'data'),
    Input('identityData', 'data'),
    Input('uidAttr', 'data'),
    Input('selectedDropDown', 'value'),
    Input('slider-rounding', 'value'),
    Input('slider-clustering', 'value'),
    Input('slider-dates', 'value'), 
    Input('hover_data', 'data')
    )
def report_2(click, fig, idRep, idData, permData, uid, selectedAttr, sliderRound, sliderCluster, sliderDate, hover_data):

    if click:

        idRep = pd.read_json(idRep, orient='split')
        idData = pd.read_json(idData, orient='split')
        permData = pd.read_json(permData, orient='split')
        selectedAttr = selectedAttr.split(":")[0]

        reporting.report_2(idRep, idData, permData, uid, selectedAttr, sliderRound, sliderCluster, sliderDate, hover_data)
        save_plot(True, fig, "Report2_Plot", selectedAttr)

    return 0

# killing the dash server
@app.callback(Output('stop_button', 'children'),
    Input('plotly_figure', 'figure'),
    Input('stop_button', 'n_clicks'), 
    Input('pid', 'data')
    )
def update_exitButton(fig, click, pid):
    if click:
        print("\n!!----- Stopping plottng, server disconnected -----!!\n")
        fig.update
        os.system(f"taskkill /IM {pid} /F") # this kills the app
        return

# action to perform when a row is added
@app.callback(Output('selected_points_table', 'data'),
    State('selected_points_table', 'data'),
    Input('hover_data', 'data'),
    Input('plotly_figure', 'clickData')
    )
def add_row(rows, hover_data, inputData):
    if inputData is None:
        # rows = None
        pass
    elif len(inputData['points'][0]['customdata']) != len(hover_data):
        pass
    else:
        print("----- Data added -----")
        d = {}
        for n, hd in enumerate(hover_data):
            d[hd] = inputData['points'][0]['customdata'][n]
        if rows == [] or rows is None:
            rows = [d]
        elif all(rows[-1][k] == "" for k in list(rows[-1])): 
            rows = [d]
        else:
            rows.append(d)
    
    return rows

# action to perform when row is removed
@app.callback(Output('output', 'children'),
            Input('selected_points_table', 'data_previous')
            )
def remove_rows(previous):
    if previous is not None:
        return "" # [f'Just removed {row}' for row in previous if row not in current]

# to save file name prompts and checks
@app.callback(Output('input_filename', 'placeholder'),
    Output('input_filename', 'value'),
    Output('submit_file', 'n_clicks'),
    Input('submit_file','n_clicks'),
    State('selected_points_table','data'),
    State('input_filename','value')
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
            pd.DataFrame.from_records(tab_data).to_csv(fileName,index=False)
            placeholder = "File saved"
            print(f"     File saved at {fileName}")
        else:
            placeholder = "File exists"
    
    # always reset the text
    output = ""

    print(f"----- Save file, {placeholder}, {output} -----\n")
    return placeholder, output, 0

# enable/disable the toggle and sliders
@app.callback(
    Output('toggle-out', 'children'),
    Output('slider-dates', 'disabled'),
    Output('slider-rounding', 'disabled'),
    Output('slider-clustering', 'disabled'),
    Input('toggle-in', 'value'),
    Input('slider-rounding', 'value')
)
def update_output(toggleValue, sliderRoundValue):

    # default values of all sliders/toggles
    toggleOut = "Date of extract"
    sliderDateDisabled = False
    sliderRoundDisabled = False
    sliderClusterDisabled = False

    if toggleValue:
        toggleOut = "Tracking data"
        sliderDateDisabled = True
        sliderRoundDisabled = True
        sliderClusterDisabled = True
    
    if sliderRoundValue == 0:
        sliderClusterDisabled = True

    return toggleOut, sliderDateDisabled, sliderRoundDisabled, sliderClusterDisabled

@app.callback(
    Output('report_1', 'disabled'),
    Output('report_2', 'children'),
    Output('report_2', 'disabled'),
    Output('report_3', 'disabled'),
    Output('report_4', 'disabled'),
    Output('report_5', 'disabled'),
    Input('slider-rounding', 'value'),
    Input('slider-clustering', 'value')
)
def update_buttons(sliderRounding, sliderClustering):

    '''
    Update all buttons based on the various inputs
    '''

    report1 = False
    report3 = False
    report4 = False
    report5 = False

    report2_child = "Report 2"
    report2_disabled = False

    if sliderRounding == 0 or sliderClustering == 0:
        report2_child = "Disabled"
        report2_disabled = True

    return(report1, report2_child, report2_disabled, report3, report4, report5)

'''
NOTE need to figure out how to combine outputs
@app.callback(Output('clear_table', 'n_clicks'),
    Output('selected_points_table', 'data'),
    Input('clear_table', 'n_clicks'),
    Input('selected_points_table', 'data'),
    )
def clear_table(click, table):

    if click:
        return 0, []
'''