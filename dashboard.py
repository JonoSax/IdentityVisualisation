from dash import Dash

from dataLoader import CsvData
from plottingBoard import createInteractivePlot

"""
TODO
    - How to pass an object to the dash app, not just the attributes? Performance
    issues?
    - Optimise dash computations with caching, parallelisation: https://dash.plotly.com/sharing-data-between-callbacks
    - Persist the camera views of plots with data refreshes: https://plotly.com/python/reference/layout/#layout-uirevision 
"""

# Theme stuff: https://dash.plotly.com/external-resources


def launchApp(dataModel):

    app = Dash(__name__)  # , external_stylesheets=external_stylesheets)

    appObj = createInteractivePlot(app, dataModel)
    # webbrowser.open("http://127.0.0.1:8050/", new = 0, autoraise = True)
    try:
        appObj.run_server(debug=True, port=8050)
    except Exception as e:
        print(e)
        launchApp(appObj)


if __name__ == "__main__":

    csvData = CsvData()
    launchApp(csvData)
