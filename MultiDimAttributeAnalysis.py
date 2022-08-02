#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
from dataModel import DataModel

# https://dash.plotly.com/interactive-graphing


# ---------- MDS calculations ----------

# read the sheet data from the excel sheet and get header/info 
class ExcelData(DataModel):

    '''
    Perform the data modelling and analysis for information stored in an excel sheet.

    There are three ways data from an excel document can be ingested: 
        - useSimilarityData: Read in the pre-computed similarity data
        - useIdentityData: Read in the identity data tables
        - useAttributeData: Read in the attribute data tables
    '''

    def useSimilarityData(self, excelFile : str, sheetName : str, identityID: str, permissionID: str):

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

        if "Id" in self.processingType:
            self.identityData = pd.read_excel(excelPath, sheet_name="IdentityInformation")
            print(f"{len(self.identityData)} identities identified with {len(self.identityData.columns)} attributes")

        self.joinKeys['identity'] = identityID
        self.joinKeys['permission'] = permissionID
        # self.permissionValue = permissionValue

    def useIdentityData(self, excelFile : str, sheetName : str, identityID: str, permissionID: str):

        '''
        Read in the raw identity data to use for the MDS transformation

        Inputs
        -----
        excelFile : str
            The name of the excel file to read

        sheetName : str
            The name of the excel sheet to read        

        Outputs
        -----
        rawPermissionData
        categoriesHeader
        identityData
        categories
        processingType
            
        '''

        excelPath = f"{self.dir['data']}{excelFile}"
        posdf = pd.read_excel(excelPath, sheet_name=sheetName, header=None).fillna(False).to_numpy()

        # process the posdf information to be only the desired data inputss
        infoPos = np.where(posdf[:, 0]!=False)[0]
        endPosInfo = infoPos[-1]+1
        valPos = np.where(posdf[1, :]!=False)[0]
        startPosVal = valPos[0] + 1
        endPosVal = valPos[-1] + 1
        startRows = 4

        # permissionInfo = posdf[6:endPosInfo, 0:2]
        # NOTE add in the count column and values
        self.rawPermissionData = np.transpose(posdf[startRows:endPosInfo, startPosVal:endPosVal])   # ensure only values included under the parameter info is included
        self.categoriesHeader = list(posdf[:startRows, startPosVal-1])
        self.categoriesHeader.append("Count")
        self.identityData = pd.read_excel(excelPath, sheet_name="IdentityInformation")
        categoriesArray = posdf[:startRows, startPosVal:endPosVal]
        self.categories = [categoriesArray[:, n] for n in range(categoriesArray.shape[1]) if (categoriesArray[:, n]  != False).all()]
        self.processingType = self.processType(sheetName)

        self.joinKeys['identity'] = identityID
        self.joinKeys['permission'] = permissionID
        # self.permissionValue = permissionValue

    def useAttributeData(self, excelFile : str, sheetName : str):

        '''
        Read in the raw identity data to use for the MDS transformation

        Inputs
        -----
        excelFile : str
            The name of the excel file to read

        sheetName : str
            The name of the excel sheet to read        

        Output
        ------
        rawPermissionData
        categoriesHeader
        categories
        processingType
        '''

        excelPath = f"{self.dir['data']}{excelFile}"
        posdf = pd.read_excel(excelPath, sheet_name=sheetName, header=None).fillna(False).to_numpy()

        # process the posdf information to be only the desired data inputss
        infoPos = np.where(posdf[:, 0]!=False)[0]
        endPosInfo = infoPos[-1]+1
        valPos = np.where(posdf[1, :]!=False)[0]
        startPosVal = valPos[0] + 1
        endPosVal = valPos[-1] + 1
        startRows = 5

        # permissionInfo = posdf[6:endPosInfo, 0:2]
        # NOTE add in the count column and values
        self.rawPermissionData = posdf[startRows:endPosInfo, startPosVal:endPosVal]   # ensure only values included under the parameter info is included
        self.categoriesHeader = list(posdf[1:startRows-1, startPosVal-1])
        self.categoriesHeader.append("Count")
        categoriesArray = posdf[1:startRows, startPosVal:endPosVal]
        self.categories = [categoriesArray[:, n] for n in range(categoriesArray.shape[1]) if (categoriesArray[:, n]  != False).all()]
        self.processingType = self.processType(sheetName)

class CSVData(DataModel):

    '''
    Perform the data modelling and analysis for information stored in exported csvs.

    Data structures
    -----

    -- Identity data --

    This file should the unique identities, one per line, with columns containing their attributes:
    ID#,    Dept,   Job,    
    001,    d1,     j1,
    002,    d1,     j2,    
    etc.

    Where the first line is the attribute type name, each line contains a UNIQUE **identity**
    which can be identified by some unique key value (which has to be specified).

    -- Permission data --

    This file should contain the permission data, where each line is a unique identity and permission data combination, eg
    ID#,    Value,  Application,    
    001,    v1,     a1,
    001,    v2,     a1, 
    002,    v1,     a1   
    002,    v3,     a2,
    etc.

    Where the first line is the attribute type name, each line contains a UNIQUE 
    **permission assignment**.

    The attribute name (column header in the file) which is used to join to the permission data
    for the above example, "ID#".

    The attribute name (column header in the file) which is the permission name of interest to compare againist
    other identities, for the above example, "Value".

    '''

    def getData(self, identityPath: str, permissionPath: str, identityKey: str, permissionKey : str, permissionValue: str, trackHistorical = True):

        '''
        Ingest the raw information and process for the dataModel

        Inputs
        -----

        identityPath : str

        permissionPath : str

        identityKey : str
            The joining key used to connect the identity and permission dataframes on the identity data

        permissionKey : str
            The joining key used to connect the identity and permission dataframes on the permission data

        permissionValue : str
            The columns value used to model the relationships between the identities

        trackHistorical : boolean
            If True, if there are relevant files which contain historical data, track these. 

        Outputs
        -----
        DataModel : obj
            The attributes of the data model populated
        '''

        self.identityPath = identityPath
        self.permissionPath = permissionPath
        self.joinKeys["identity"] = identityKey
        self.joinKeys["permission"] = permissionKey
        self.permissionValue = permissionValue
        self.trackHistorical = trackHistorical

        self.getIdentityData()
        self.getPermissionData()

    def getIdentityData(self):

        '''
        Process the raw identity data
        '''

        self.identityData = pd.read_csv(self.identityPath).dropna(how = 'all')

    def getPermissionData(self):

        '''
        Process the raw permission data
        '''

        permissionData = pd.read_csv(self.permissionPath).dropna(how = 'all')

        # perform a pivot of the raw csv data to create a sparse matrix of occurence
        headers = list(permissionData.columns)
        headers.remove(self.joinKeys['permission'])
        headers.remove(self.permissionValue)
        permPivot = pd.pivot_table(permissionData, 
                    values = [headers[0]], 
                    columns=[self.joinKeys['permission']], 
                    index=[self.permissionValue], 
                    aggfunc = 'count', 
                    fill_value = 0)
        permPivot.columns = permPivot.columns.droplevel(0)
        permPivot = permPivot.reset_index()
        permPivot = permPivot.set_index(self.permissionValue)

        # this is joining the identity and permission data as only the identities in the
        # identityData are INCLUDED in the permission data
        commonIdentities = permPivot.columns[permPivot.columns.isin(self.identityData[self.joinKeys['identity']])]
        joinedPivot = permPivot[commonIdentities]
        
        self.rawPermissionData = np.transpose(joinedPivot.to_numpy())
        self.categoriesHeader = [self.joinKeys['permission']]
        self.categories = np.c_[joinedPivot.columns]
        self.processingType = self.processType("Identity")

# ----------- Data processing and visualisation (main function) ----------

# initiate the processing, visualisation and local server
def excelData(excelFile: str, sheetName: str, dims: int, identityID: str, permissionID: str, permissionValue: str):

    '''
    Calculate and visualise a dimensionally reduced analysis of entitlements
    If there is a file which already exists, just visualise otherwise create the new file
    '''

    print("---Beginning multi dimensional analysis---")
    print(f"Arguments\nworkbook: {workbook}, worksheet: {worksheet}, dims: {dims}")

    
    excelData = ExcelData()
    # excelData.useAttributeData(excelFile, "EntitlementAnalysisAttributes", permissionValue)
    # excelData.useIdentityData(excelFile, "EntitlementAnalysisIdentities", identityID, permissionID, permissionValue)
    excelData.useSimilarityData(excelFile, sheetName, identityID, permissionID, permissionValue)
    excelData.calculateMDS(recalculate=False)
    excelData.plotMDS()
    

def csvData():
    
    identityPath = "data\\IdentitiesFakeSmall.csv"
    permissionPath = "data\\EntitlementsFake.csv"
    
    csvData = CSVData()
    csvData.getData(identityPath, permissionPath, "Username", "Identity", "Value")
    csvData.calculateMDS(recalculate=False)
    csvData.plotMDS()




if __name__ == "__main__":
   
    print("Loading....")
    args = sys.argv
    if any(["LaunchExcel" in a for a in args]):
        workbook, worksheet, dims, identityID, permissionID = args = sys.argv[2:]
        excelData(str(workbook), str(worksheet), int(dims), str(identityID), str(permissionID))
    else:
        workbook = "C:\\Users\\ResheJ\\Downloads\\WorkBook-Hashedv1.xlsm"
        workbook = "WorkBook-FakeData.xlsm"
        worksheet = "SimilarityScoreIdentities"
        dims = 3
        csvData()
        excelData(workbook, worksheet, dims, "Username", "Identity")