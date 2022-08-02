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

external_stylesheets = ['\\data\\bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

def launchApp(dataModel : object, name = ""):
    
    appObj = createInteractivePlot(dataModel, name)
    webbrowser.open("http://127.0.0.1:8050/", new = 0, autoraise = True)
    try:
        appObj.run_server(debug = False)
    except Exception as e:
        print(e)
        launchApp(appObj)

# create the dash application
def createInteractivePlot(dataModel : object, info : str):

    # get the list of desireable attributes
    # Remove the dim values
    # Remove any attributes with over 100 unique values (makes plotly run too slow and clutters the graphs)
    # Remove the key words "Count" and "Unnamed: 0" which are artefacts of the plotting

    print("---Creating web server for plotting---")
    if dataModel.trackHistorical is None:
        df = dataModel.mdsResults
    else:
        df = dataModel.trackHistorical

    # attrList = [l for l in sorted(formatColumns) if not (l.lower().startswith("dim") or l.lower().startswith("unnamed"))]
    selectedData = [d for d in sorted(list(df.columns)) if (d.lower().find("unnamed") == -1)]
    hover_data = [s for s in selectedData if s.lower().find("dim") == -1]
    # hover_data.remove("timeUnix")
    attrList = hover_data.copy()
    attrList.remove("DateTime")
    attrList = [r.replace(r, f"{r}: {len(df[r].unique())}") for r in attrList]

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
                    attrList,
                    [a for a in attrList if a.find("LONG") == -1][0],     # ensure an attribute which isn't LONG is selected
                    id='selectedDropDown',
                )
            ],
            style={'width': '49%', 'display': 'inline-block'})
        ], style={
            'padding': '0px 5px'
        }),

        # plotly figure
        html.Div([
            dcc.Graph(
                id='plotly_figure'
            )
        ], style={'width': '100%', 'height':'100%', 'display': 'inline-block', 'padding': '0 10'}),
    
        # Save figure button
        html.Div([
            html.Button('Save plot', id='submit_plot', n_clicks=0)
            ], style={
                "margin-left": "15px", 
                "margin-top": "15px", 
                "display": "inline-block",
                }),

        # plotly stop running button
        html.Div([
            daq.StopButton(
                id='stop_button',
                n_clicks=0
            )], style={
                "margin-left": "15px", 
                "margin-top": "15px", 
                "display": "inline-block",
                }),

        # Save file button and text entry
        html.Div([
            dcc.Input( 
                id="input_filename", 
                type="text", 
                placeholder="File name", 
                value = ""), 
            html.Button('Save file', id='submit_file', n_clicks=0)
            ], style={
                "margin-left": "15px", 
                "margin-top": "15px", 
                "display": "inline-block",
                }),

        # Clear table button
        # html.Div([
        #     html.Button('Clear table', id='clear_table', n_clicks=0)
        #     ], style={
        #         "margin-left": "15px", 
        #         "margin-top": "15px", 
        #         "display": "inline-block",
        #         }),

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
        dcc.Store(data = df[selectedData].to_json(orient='split'), id = "dataFrame"),
        dcc.Store(data = attrList, id = "attrList"),
        dcc.Store(data = info, id = "info"),
        dcc.Store(data = os.getpid(), id = "pid"),
        dcc.Store(data = hover_data, id = "hover_data"),

        html.Div(id='output')
    ])

    print("     Web app created")
    return app

# plotly figure updates
@app.callback(Output('plotly_figure', 'figure'),
    Input('dataFrame', 'data'),
    Input('selectedDropDown', 'value'),
    Input('info', 'data'), 
    Input('hover_data', 'data')
    )
def update_graph(dfjson, attribute, info, hover_data):

    '''
    Take in the raw data and selected information and create visualisation
    '''

    print("----- Updating plotting information -----")

    df = pd.read_json(dfjson, orient='split')
    df[hover_data] = df[hover_data].astype(str)
    dataColumns = list(df.columns)
    dims = sum(1 for x in list(dataColumns) if x.startswith ("Dim"))

    # remove the count info to match to the data frame
    attribute = attribute.split(":")[0]
    plotTitle = f"{info} {dims}D visualising {attribute} for {len(df)} data points"

    # set the constant for x, y, z scaling (0 = exact fit for data, 0.1 = 10% larger etc)
    r = 0.1

    # NOTE 2D plotting is not being suppored atm.
    if dims == 2:
        print(f"     Creating 2D plotting for {attribute}")
        fig = px.scatter(df, 
                x=df["Dim0"], 
                y=df["Dim1"],
                hover_data = hover_data,
                color = attribute,
                title = plotTitle, 
                hover_name= attribute
                )
        
        fig.update_layout(
            scene = dict(
                xaxis = dict(range=[min(df["Dim0"])*(1-r), max(df["Dim0"])*(1+r)]),
                yaxis = dict(range=[min(df["Dim1"])*(1-r), max(df["Dim1"])*(1+r)])
                ))

    elif dims == 3:

        # if there is temporal versions of the data, plot the traces 
        if "DateTime" in df.columns:
            print(f"     Tracking historical data with 3D plotting for {attribute}")

            allTimes = df["DateTime"].unique()

            fig = go.Figure()
            uniqueIDs = df["Username"].unique()
            colourDict = {}

            # create a dictionary to colour the traces depending on the attribute selected
            for n, c in enumerate(df[attribute].unique()):
                colourDict[str(c)] = hex_to_rgb(colours.Plotly[n%len(colours.Plotly)])

            # NOTE because the data is ordered from the newest (at the top) to the oldest (at the bottom) this means the 
            # size is DECREASING and the transparency is DECREASING. This is so that for the legend plotting, the markers
            # are more easily visible...
            for uid in uniqueIDs:
                # get all the unique entries for this unique identity
                uiddf = df[df["Username"] == uid]

                # set the colours so that the newest data pont is 100% opacity and the oldest data point is 40% opacity
                variable_colour = [f"rgba{tuple(np.append(colourDict[uiddf[attribute].iloc[n]], c))}" for n, c in zip(range(len(uiddf)), np.linspace(1, 0.4, len(allTimes)))]
                name = [u for u in uiddf[attribute] if u != "None"][0]
                # doco: https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Scatter3d.html
                fig.add_trace(
                        go.Scatter3d(
                            x=uiddf["Dim0"],
                            y=uiddf["Dim1"],
                            z=uiddf["Dim2"],
                            customdata=uiddf[hover_data],
                            hovertext = 
                            ['<br>'.join([f"{h}: {uiddf[h].iloc[n]}" for h in hover_data]) for n in range(len(uiddf))],
                            marker=dict(color=variable_colour, size=np.linspace(12, 4, len(uiddf)).astype(int)),
                            line = dict(color=variable_colour),
                            name = name,            # NOTE this must be a string/number
                            legendgroup = name,     # NOTE this must be a string/number
                            connectgaps=True        # NOTE for some reason this isn't acutally connecting gaps.... maybe wrong data type for empty? 
                        )
                    )

            # remove duplicate legend entries
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
                
        # plot only the current data
        else:
            print(f"     Plotting data with 3D plotting for {attribute}")

            fig = px.scatter_3d(df, 
                    x=df["Dim0"], 
                    y=df["Dim1"], 
                    z=df["Dim2"],
                    hover_data = hover_data,
                    color = attribute, 
                    title = plotTitle, 
                    hover_name= attribute
                    )
        # NOTE i don't think this is actually doing anything....
        
        fig.update_layout(
            scene = dict(
                xaxis = dict(range=[min(df["Dim0"])-r, max(df["Dim0"])+r]),
                yaxis = dict(range=[min(df["Dim1"])-r, max(df["Dim1"])+r]),
                zaxis = dict(range=[min(df["Dim2"])-r, max(df["Dim2"])+r])
                ))


    # allow for multiple point selection
    fig.update_layout(
        clickmode='event+select',
        width=1200, 
        height=600, 
        hovermode='closest'
    )

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

        '''
        # create the specific names for saving the plots
        if "Attribute" in info: 
            # for attribute analysis, the combination is important. Highligh the specific
            # attribute with ##
            selectedAttr = [a.replace(attribute, f"#{attribute}#") for a in sorted(attrList)]
        else:
            selectedAttr = attribute
        '''

        dims = sum([l.find("axis")>0 for l in list(fig["layout"]["scene"])])
        selectedAttr = selectedAttr.split(":")[0]
        plotName = f'{os.path.expanduser("~")}\\Downloads\\{info}_{dims}D_{selectedAttr}_{datetime.now().strftime("%y%m%d%H%M%S")}.html'
        go.Figure(fig).write_html(plotName)
        
        print(f"     Plot saved at {plotName}\n")

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