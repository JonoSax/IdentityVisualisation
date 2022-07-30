#!/usr/bin/python

import sys
import numpy as np
from sklearn import manifold
from glob import glob
import pandas as pd
import os
from os.path import dirname as up
from datetime import datetime
from dashboard import launchApp
from dataModel import DataModel

# https://dash.plotly.com/interactive-graphing
# https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html 


# ---------- MDS calculations ----------

# read the sheet data from the excel sheet and get header/info 
class ExcelData(DataModel):

    '''
    Perform the data modelling and analysis for information stored in an excel sheet
    '''

    def getExcelSimilarityData(self, excelFile, sheetName):

        '''
        Read in the pre-computed similarity calculations from an excel file

        Inputs
        -----
        excelFile : str
            The name of the excel file to read

        sheetName : str
            The name of the excel sheet to read

        Outputs
        ----
        similarityPermissionData
        categories
        categoriesHeader
        processingType
        identityData
        '''

        excelPath = f"{self.dir['data']}{excelFile}"
        posdf = pd.read_excel(excelPath, sheet_name=sheetName, header=None).fillna(False).to_numpy()

        # find which row/column the data start
        start = min(np.where(posdf[:, 0]!=False)[0]) + 1

        # get the category information
        categoriesRaw = [d for d in posdf[start::, :start]]
        categoriesArray = np.array(categoriesRaw)
        categoriesHeader = [c.replace(" ", "") for c in list(posdf[start-1, 0:start]) if type(c) == str]
        categoriesArray = [categoriesArray[:, n] for n in range(categoriesArray.shape[1]) if (categoriesArray[:, n]  != False).all()]
        categories = np.transpose(np.array(categoriesArray))

        print(f"     {len(categoriesHeader)} categories extracted")

        # find which row/column the data start
        start = min(np.where(posdf[:, 0]!=False)[0]) + 1
        data = posdf[start::, start::]
        dimY, dimX = np.where(data=="Dimensions")
        if len(dimY) >0 and len(dimX) >0:
            data[dimY[0], dimX[0]:(dimX[0]+2)] = False           # search for and remove the dimensions free text
        data[np.where(data==False)] = 1
        data = data.astype(np.float64)
        print(f"     {len(data)} data points extracted")

        # update the DataModel attributes
        self.similarityPermissionData = data
        self.categories = categories
        self.categoriesHeader = categoriesHeader
        self.processingType = self.processType(sheetName)

        if "id" in self.processingType:
            self.identityData = pd.read_excel(excelPath, sheet_name="IdentityInformation")
            print(f"{len(self.identityData)} identities identified with {len(self.identityData.columns)} attributes")

    def getExcelIdentityData(self, excelFile, sheetName):

        '''
        Read in the raw identity data to use for the MDS transformation

        Inputs
        -----
        excelFile : str
            The name of the excel file to read

        sheetName : str
            The name of the excel sheet to read        
            
        '''

        excelPath = f"{self.dir['data']}{excelFile}"
        posdf = pd.read_excel(excelPath, sheet_name=sheetName, header=None).fillna(False).to_numpy()

        self.rawPermissionData

        self.identityData = pd.read_excel(excelPath, sheet_name="IdentityInformation")
        pass

    def getExcelAttributeData(self, excelFile, sheetName):

        '''
        Read in the raw identity data to use for the MDS transformation

        Inputs
        -----
        excelFile : str
            The name of the excel file to read

        sheetName : str
            The name of the excel sheet to read        
            
        '''

        excelPath = f"{self.dir['data']}{excelFile}"
        posdf = pd.read_excel(excelPath, sheet_name=sheetName, header=None).fillna(False).to_numpy()

        # process the posdf information to be only the desired data inputss
        infoPos = np.where(posdf[:, 0]!=False)[0]
        endPosInfo = infoPos[-1]+1
        valPos = np.where(posdf[1, :]!=False)[0]
        startPosVal = valPos[0] + 1
        endPosVal = valPos[-1] + 1

        # permissionInfo = posdf[6:endPosInfo, 0:2]
        # NOTE add in the count column and values
        self.rawPermissionData = posdf[6:endPosInfo, startPosVal:endPosVal]   # ensure only values included under the parameter info is included
        self.categoriesHeader = list(posdf[1:4, startPosVal-1])
        self.identityData = pd.read_excel(excelPath, sheet_name="IdentityInformation")
        categoriesArray = posdf[1:4, startPosVal:endPosVal]
        self.categories = [categoriesArray[:, n] for n in range(categoriesArray.shape[1]) if (categoriesArray[:, n]  != False).all()]
        self.processingType = self.processType(sheetName)

# ----------- Data processing and visualisation (main function) ----------

# initiate the processing, visualisation and local server
def multiDimAnalysis(excelFile: str, sheetName: str, dims: int):

    '''
    Calculate and visualise a dimensionally reduced analysis of entitlements
    If there is a file which already exists, just visualise otherwise create the new file
    '''

    print("---Beginning multi dimensional analysis---")
    print(f"Arguments\nworkbook: {workbook}, worksheet: {worksheet}, dims: {dims}")

    excelData = ExcelData()
    # excelData.getExcelAttributeData(excelFile, "EntitlementAnalysisAttributes")
    excelData.getExcelSimilarityData(excelFile, sheetName)
    excelData.calculateMDS()
    excelData.plotMDS()



if __name__ == "__main__":
   
    print("Loading....")
    if len(sys.argv) == 4:
        _, workbook, worksheet, dims = sys.argv

        multiDimAnalysis(str(workbook), str(worksheet), int(dims))
    else:
        workbook = "C:\\Users\\ResheJ\\Downloads\\WorkBook-Hashedv1.xlsm"
        workbook = "WorkBook-FakeData.xlsm"
        worksheet = "SimilarityScoreAttributes"
        dim = 3
        
        multiDimAnalysis(workbook, worksheet, dim)