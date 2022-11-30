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

# Running on azure app services: https://learn.microsoft.com/en-us/azure/app-service/configure-language-python#flask-app
# https://resonance-analytics.com/blog/deploying-dash-apps-on-azure


"""
def launchApp(dataModel):

    dash_app = Dash(__name__)  # , external_stylesheets=external_stylesheets)
    app = dash_app.server
    dash_app = createInteractivePlot(dash_app, dataModel)
    # webbrowser.open("http://127.0.0.1:8050/", new = 0, autoraise = True)
    try:
        dash_app.run_server(debug=True)
    except Exception as e:
        print(e)
"""

dataModel = CsvData(False)
dash_app = Dash(__name__)  # , external_stylesheets=external_stylesheets)
app = dash_app.server
dash_app = createInteractivePlot(dash_app, dataModel)


if __name__ == "__main__":

    dash_app.run_server(debug=True)
