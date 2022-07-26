#!/usr/bin/python

import sys
import numpy as np
from sklearn import manifold
from glob import glob
# from sklearn.metrics import euclidean_distances
# from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, dash_table
import pandas as pd
import os
from datetime import datetime
import dash_daq as daq
import webbrowser
import plotly.graph_objects as go

# https://dash.plotly.com/interactive-graphing
# https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

# ---------- MDS calculations ----------

# read the sheet data from the excel sheet and get header/info 
def getdfInfo(posdf):

    # find which row/column the data start
    start = min(np.where(posdf[:, 0]!=False)[0]) + 1

    # get the category information
    categoriesRaw = [d for d in posdf[start::, :start]]
    categoriesArray = np.array(categoriesRaw)
    # categoriesHeader = [c.replace(" ", "") for c in list(posdf[start-1, 0:start]) if type(c) == str]
    categoriesHeader = [c.replace(" ", "") for c in list(posdf[start-1, 0:start]) if type(c) == str]

    print(f"     {len(categoriesHeader)} categories extracted")

    # remove the categories which are empty (ie false)
    categoriesArray = [categoriesArray[:, n] for n in range(categoriesArray.shape[1]) if (categoriesArray[:, n]  != False).all()]
    categories = np.transpose(np.array(categoriesArray))

    # find which row/column the data start
    start = min(np.where(posdf[:, 0]!=False)[0]) + 1
    data = posdf[start::, start::]
    dimY, dimX = np.where(data=="Dimensions")
    if len(dimY) >0 and len(dimX) >0:
        data[dimY[0], dimX[0]:(dimX[0]+2)] = False           # search for and remove the dimensions free text
    data[np.where(data==False)] = 1
    data = data.astype(np.float64)
    print(f"     {len(data)} data points extracted")

    return data, categories, categoriesHeader

# perform MDS fitting
def similarityCalculation(excelFile: str, sheetName: str, dims: int):

    '''
    Perform the dimensionality reduction
    '''

    print("---similarityCalculation beginning---")

    if "attributes" in sheetName.lower():
        print(f"     Loading {sheetName} from {excelFile}")
        posdf = pd.read_excel(excelFile, sheet_name=sheetName, header=None).fillna(False).to_numpy()

        # we need the excel sheet to do the processing
        data, categories, categoriesHeader = getdfInfo(posdf)

        print("     Excel file successfully loaded")

        # check if the data has already been calculated
        joinedCategoryHeaders =  '_'.join(sorted(categoriesHeader))
        csvName = f"{sheetName}_{dims}_{joinedCategoryHeaders}"
        csvFiles = sorted(glob(f"{os.path.dirname(excelFile)}\\{csvName}*.csv"))
        calculateMDS = len(csvFiles) == 0

    elif "identities" in sheetName.lower():

        # attribute combinations are not important for identity data
        csvName = f"{sheetName}_{dims}"

        # identities contain all information, just check the dims are calculated for the right sheet
        csvFiles = sorted(glob(f"{os.path.dirname(excelFile)}\\{csvName}*.csv"))
        calculateMDS = len(csvFiles) == 0

        # only load the identity excel sheet if we know that we have to calculate the info
        if calculateMDS:
            print(f"     Opening {sheetName} worksheet")
            posdf = pd.read_excel(excelFile, sheet_name=sheetName, header=None).fillna(False).to_numpy()
            data, categories, categoriesHeader = getdfInfo(posdf)

    if calculateMDS:

        print("     No relevant calculations found, new MDS calculation beginning")
                
        # create the pre-computed similartiies matrix
        similarities = data * data.transpose()

        # perform dimensionality reduction
        print("     Starting mds fit")

        mds = manifold.MDS( 
            n_components=dims, 
            # max_iter=3000,
            eps=1e-3,
            random_state=np.random.RandomState(seed=3),
            dissimilarity="precomputed",
            n_jobs=1, 
            verbose=False
        )
        pos = mds.fit(1-similarities).embedding_
        iters = mds.n_iter_
        disparities = mds.stress_

        print(f"     Fitting complete")

        # get n-dimension labels
        dimNames = [f"Dim{n}" for n in range(len(pos[0]))]

        joinedCategoryHeaders =  '_'.join(sorted(categoriesHeader))
        posdf = pd.DataFrame(np.hstack([pos, categories]), columns = [*dimNames, *categoriesHeader])

        # merge all identity information if available. Remove unnecessary columns and add warnings to long attribute lists
        if "Identities" in sheetName: 
            posidentitiesRaw = pd.read_excel(excelFile, sheet_name="IdentityInformation")
            selectColums = [r for r in list(posidentitiesRaw.columns) if r.lower().find("unnamed") == -1]
            posidentitiesSelect = posidentitiesRaw[selectColums]
            formatColumns0 = [r.replace(" ", "") for r in list(posidentitiesSelect.columns)]
            posidentities = posidentitiesSelect.set_axis(formatColumns0, axis=1, inplace=False)
            posdf = posdf.merge(posidentities, on=categoriesHeader[0], how='left')

        # date stamp the file
        dir = os.path.dirname(excelFile)
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        csvPath = f"{dir}\\{csvName}_{dt_string}.csv"
        posdf.to_csv(csvPath)

    else:
        # return the most recently created version
        csvPath = csvFiles[-1]
        print(f"     Loading {sheetName} from {excelFile}")

    return csvPath

# ---------- Plotly visualisation launched in Dash webserver ----------

def launchApp(app):
    webbrowser.open("http://127.0.0.1:8050/", new = 0, autoraise = True)
    try:
        app.run_server(debug = False)
    except Exception as e:
        print(e)
        launchApp(app)

# create the dash application
def createInteractivePlot(df, info = ""):

    # get the list of desireable attributes
    # Remove the dim values
    # Remove any attributes with over 100 unique values (makes plotly run too slow and clutters the graphs)
    # Remove the key words "Count" and "Unnamed: 0" which are artefacts of the plotting

    print("---Creating web server for plotting---")
    # attrList = [l for l in sorted(formatColumns) if not (l.lower().startswith("dim") or l.lower().startswith("unnamed"))]
    hover_data = [d for d in sorted(list(df.columns)) if (d.lower().find("unnamed") == -1 and d.lower().find("dim") == -1)]
    attrList = [r.replace(r, f"LONG:{r}") if len(df[r].unique()) > 100 else r for r in hover_data ]
    try: attrList.remove("Count")
    except: pass

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

        # Save file button
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
        dcc.Store(data = df.to_json(orient='split'), id = "dataFrame"),
        dcc.Store(data = attrList, id = "attrList"),
        dcc.Store(data = info, id = "info"),
        dcc.Store(data = os.getpid(), id = "pid"),
        dcc.Store(data = hover_data, id = "hover_data"),

        html.Div(id='output')
    ])

    print("     Web app created")
    return app
    
# ------ callbacks ------

# plotly figure updates
@app.callback(Output('plotly_figure', 'figure'),
    Input('dataFrame', 'data'),
    Input('selectedDropDown', 'value'),
    Input('attrList', 'data'),
    Input('info', 'data'), 
    Input('hover_data', 'data')
    )
def update_graph(dfjson, attribute, attrList, info, hover_data):

    df = pd.read_json(dfjson, orient='split')
    dataColumns = list(df.columns)
    dims = sum(1 for x in list(dataColumns) if x.startswith ("Dim"))
    plotTitle = f"{info} {dims}D visualising {attribute} for {len(df)} data points"

    # if there is a warning "LONG", remove so it matches the dataframe
    attribute = attribute.replace("LONG:", "")

    # set the constant for x, y, z scaling (0 = exact fit for data, 0.1 = 10% larger etc)
    r = 0.2

    if dims == 2:
        fig = px.scatter(df, x=df["Dim0"], y=df["Dim1"],
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
        fig = px.scatter_3d(df, x=df["Dim0"], y=df["Dim1"], z=df["Dim2"],
                hover_data = hover_data,
                color = attribute, 
                title = plotTitle, 
                hover_name= attribute
                )

        fig.update_layout(
            scene = dict(
                xaxis = dict(range=[min(df["Dim0"])*(1-r), max(df["Dim0"])*(1+r)]),
                yaxis = dict(range=[min(df["Dim1"])*(1-r), max(df["Dim1"])*(1+r)]),
                zaxis = dict(range=[min(df["Dim2"])*(1-r), max(df["Dim2"])*(1+r)])
                ))

    # allow for multiple point selection
    fig.update_layout(clickmode='event+select')

    fig.update_layout(width=1200, height=600, hovermode='closest')

    print("---Web server launched---")
    return fig

@app.callback(Output('submit_plot', 'n_clicks'),
    Input('submit_plot', 'n_clicks'),
    Input('plotly_figure', 'figure'),
    Input('info', 'data'),
    Input('selectedDropDown', 'value')
    )
def save_plot(click, fig, info, selectedAttr):

    if click:

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
        go.Figure(fig).write_html(f'{os.path.expanduser("~")}\\Downloads\\{info}_{dims}D_{selectedAttr}_{datetime.now().strftime("%y%m%d%H%M%S")}.html')
        
    return 0

# killing the dash server
@app.callback(Output('stop_button', 'children'),
    Input('plotly_figure', 'figure'),
    Input('stop_button', 'n_clicks'), 
    Input('pid', 'data')
)
def update_exitButton(fig, n_clicks, pid):
    if n_clicks > 0:
        fig.update
        os.system(f"taskkill /IM {pid} /F") # this kills the app
        return

# action to perform when a row is added
@app.callback(Output('selected_points_table', 'data'),
    State('selected_points_table', 'data'),
    Input('hover_data', 'data'),
    Input('plotly_figure', 'clickData'))
def add_row(rows, hover_data, inputData):
    if inputData is None:
        # rows = None
        pass
    else:
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
            Input('plotly_figure', 'figure'),
            Input('selected_points_table', 'data_previous'),
            State('selected_points_table', 'data'))
def remove_rows(fig, previous, current):
    if previous is not None:
        return "" # [f'Just removed {row}' for row in previous if row not in current]

# to save file name prompts and checks
@app.callback(Output('input_filename', 'placeholder'),
    Output('input_filename', 'value'),
    Input('submit_file','n_clicks'),
    State('selected_points_table','data'),
    State('input_filename','value')
)
def save_file(count, tab_data, filename):

    # Save data as long as there is information etc
    if tab_data == [] or tab_data is None:
        placeholder = "Select data"
    elif filename == "":
        placeholder = "Set file name"
    else:
        fileName = f"{os.path.expanduser('~')}\\Downloads\\{filename}.csv"
        if not os.path.exists(fileName):
            pd.DataFrame.from_records(tab_data).to_csv(fileName,index=False)
            placeholder = "File saved"
        else:
            placeholder = "File exists"
    
    # always reset the text
    output = ""

    return placeholder, output

# ----------- Data processing and visualisation (main function) ----------

# initiate the processing, visualisation and local server
def multiDimAnalysis(excelFile: str, sheetName: str, dims: int):

    '''
    Calculate and visualise a dimensionally reduced analysis of entitlements
    If there is a file which already exists, just visualise otherwise create the new file
    '''

    print("---Beginning multi dimensional analysis---")
    print(f"Arguments\nworkbook: {workbook}, worksheet: {worksheet}, dims: {dims}")

    latestcsv = similarityCalculation(excelFile, sheetName, dims)

    posdf = pd.read_csv(latestcsv)

    print(f"Loading {latestcsv} for plotting")

    app = createInteractivePlot(posdf, sheetName)
    # os.system("start \"\" http://127.0.0.1:8050/")
    launchApp(app)
    webbrowser.open("http://127.0.0.1:8050/", new = 0, autoraise = True)
    try:
        app.run_server(debug = False)          
    except Exception as e:
        print(e)
        # relaunch the application
        app.run_server(debug = False)     

if __name__ == "__main__":
   
    print("Loading....")
    if len(sys.argv) == 4:
        _, workbook, worksheet, dims = sys.argv

        multiDimAnalysis(str(workbook), str(worksheet), int(dims))
    else:
        workbook = "C:\\Users\\ResheJ\\Downloads\\WorkBook-Blankv6.xlsm"
        worksheet = "SimilarityScoreIdentities"
        dim = 3
        
        multiDimAnalysis(workbook, worksheet, dim)