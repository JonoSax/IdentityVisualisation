#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
from sklearn import manifold
from glob import glob
# from sklearn.metrics import euclidean_distances
# from sklearn.decomposition import PCA
import plotly.express as px
import os
from datetime import datetime

# https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html 

def similarityCalculation(excelFile: str, sheetName: str, attribute: str, dims: int):

    '''
    Perform the dimensionality reduction
    '''

    print("---similarityCalculation beginning---")

    seed = np.random.RandomState(seed=3)

    df = pd.read_excel(excelFile, sheet_name=sheetName, header=None).fillna(False).to_numpy()
    # df = pd.read_excel(excelFile, sheet_name='SimilarityScoreIdentities').fillna(False).to_numpy()

    print("Excel file successfully loaded")

    # find which row/column the data start
    start = min(np.where(df[:, 0]!=False)[0]) + 1

    # get the category information
    categoriesRaw = [d for d in df[start::, :start]]
    categoriesArray = np.array(categoriesRaw)
    categoriesHeader = [c.replace(" ", "") for c in list(df[start-1, 0:start]) if type(c) == str]
    joinedCategoryHeaders =  '_'.join(sorted(categoriesHeader))

    print(f"{len(categoriesHeader)} categories extracted")

    # remove the categories which are empty (ie false)
    categoriesArray = [categoriesArray[:, n] for n in range(categoriesArray.shape[1]) if (categoriesArray[:, n]  != False).all()]
    categories = np.transpose(np.array(categoriesArray))

    # check if the data has already been calculated
    if "attributes" in sheetName.lower():

        # attributes sheet must contain all info
        csvFiles = sorted(glob(f"{os.path.dirname(excelFile)}\\{sheetName}_{dims}*{joinedCategoryHeaders}*.csv"))
    
    elif "identities" in sheetName.lower():
        # identities only need the attribute to be in the name somewhere, not the whole match
        csvFiles = sorted(glob(f"{os.path.dirname(excelFile)}\\{sheetName}_{dims}*{attribute}*.csv"))

    else: 
        print("No relevant data found")
        csvFiles = []

    # Get only the data, not the headers
    data = df[start::, start::]
    data[np.where(data==False)] = 1
    data = data.astype(np.float64)

    print(f"{len(data)} data extracted")

    if len(csvFiles) == 0:

        print("New csv being created")

        # create the pre-computed similartiies matrix
        similarities = data * data.transpose()

        # perform dimensionality reduction
        print("Starting mds fit")

        mds = manifold.MDS( 
            n_components=dims, 
            max_iter=3000,
            eps=1e-6,
            random_state=seed,
            dissimilarity="precomputed",
            n_jobs=1, 
        )
        pos = mds.fit(1-similarities).embedding_

        # get n-dimension labels
        dimNames = [f"Dim{n}" for n in range(len(pos[0]))]

        posdf = pd.DataFrame(np.hstack([pos, categories]), columns = [*dimNames, *categoriesHeader])

        # save the calculated positional dataframe to the same directory as the original excel file
        dir = os.path.dirname(excelFile)
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        csvPath = f"{dir}\\{sheetName}_{dims}_{joinedCategoryHeaders}_{dt_string}.csv"
        posdf.to_csv(csvPath)

    else:
        # return the most recently created version
        csvPath = csvFiles[-1]
        print(f"Existing csv: {csvPath}")

    return csvPath

def mdsVisualisation(latestcsv: str, attribute: str, dims: int):

    '''
    Visualise the dimensionally reduced information based on the attribute information to colourise
    '''

    # get the most recently calculated positional information based on the relevant attribute, 
    # dimension and sheet data

    print(f"---{dims}d plotting beginning---")
    htmlLink = latestcsv.replace(".csv", ".html")

    htmlLink = htmlLink.replace(attribute, f"#{attribute}#")\

    if not os.path.exists(htmlLink):

        posdf = pd.read_csv(latestcsv)
        if attribute not in posdf.columns: 
            print(f"Attribute {attribute} not in {latestcsv}")
            return

        # set the size of the dots to the number of identity occurences (currently deactivated)
        if "Count" in posdf.columns and False:
            option = "Count"
            posdf["Count"] = posdf["Count"].astype(int)
        else:
            option = None

        print(f"using {latestcsv} to load data")

        sheetName = latestcsv.split("\\")[-1].split("_")[0]
        plotTitle = f"From {sheetName}, {dims}d MDS of {len(posdf)} points with {attribute} classificaiton"
        if dims == 2:
            otherData = list(posdf.columns)[3:]
            fig = px.scatter(posdf, x = "Dim0", y = "Dim1", size = option, color=attribute, title=plotTitle, hover_name=attribute, hover_data=otherData)

        elif dims == 3:
            otherData = list(posdf.columns)[4:]
            fig = px.scatter_3d(posdf, x = "Dim0", y = "Dim1", z = "Dim2", size = option, color=attribute, title=plotTitle, hover_name=attribute, hover_data=otherData)

        # convert the plotly figure into raw html and display
        fig.write_html(htmlLink)
        print(f"Plot saved at {htmlLink}")
    
    print(f"Displaying plot from {htmlLink}")
    os.system(f"start {htmlLink}")
    
def multiDimAnalysis(excelFile: str, sheetName: str, attribute: str, dims: int):

    '''
    Calculate and visualise a dimensionally reduced analysis of entitlements
    If there is a file which already exists, just visualise otherwise create the new file
    '''

    print("---Beginning multi dimensional analysis---")
    print(f"Arguments\nworkbook: {workbook}, worksheet: {worksheet}, attribute: {attribute}, dims: {dims}")

    # remove all spaces from the attribute name. Makes things ways simpler...
    attribute = attribute.replace(" ", "")

    newCSVFile = similarityCalculation(excelFile, sheetName, attribute, dims)
    mdsVisualisation(newCSVFile, attribute, dims)            

if __name__ == "__main__":
   
    print("Loading....")
    if len(sys.argv) == 5:
        _, workbook, worksheet, attribute, dims = sys.argv

        multiDimAnalysis(str(workbook), str(worksheet), str(attribute), int(dims))
    else:
        workbook = "C:\\Users\\ResheJ\\Downloads\\WorkBook-Blankv5c.xlsm"
        worksheet = "SimilarityScoreAttributes"
        attribute = "Job Profile"
        dim = 2
        
        multiDimAnalysis(workbook, worksheet, attribute, dim)