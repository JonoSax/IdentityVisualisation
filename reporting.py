from functools import total_ordering
import os
import numpy as np
import pandas as pd
from time import localtime, strftime
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

pd.options.mode.chained_assignment = None

class Metrics:

    '''
    Object to store and calculate metrics

    The overall goal is to make this as modular as possible so that new reports can be created
    easily. ATM it is kind of bespoke but as new report types are created, parts of these function
    can be easily split off and set as seperate functions for reuse.
    '''

    def __init__(self, df, permissions, identities, uiddf, attribute = None):

        self.df = self.setdf(df)
        self.key = uiddf
        self.attribute = attribute
        self.permissions = permissions
        self.identities = self.setdf(identities)
        self.outlierdf = None
        self.dfDistances = None
        self.permissionFixes = None

    def setdf(self, df):

        '''
        Format the df to be useful for calculations
        '''

        # set the multidmiensional positions as floats if they are present
        for d in [dim for dim in df.columns if "Dim" in dim]:
            df[d] = df[d].astype(float)
        
        # set the datetime value to int
        if "_DateTime" in df.columns:
            df[["_DateTime"]] = df[["_DateTime"]].astype(int)

        # remove spacing from the dataframe column names
        formatColumns = [r.replace(" ", "") for r in list(df.columns)]
        df = df.set_axis(formatColumns, axis=1, inplace=False)

        return(df)

    def calculateDistances(self):

        '''
        Calculate the distances of all object in each attribute unless specified
        '''

        if self.attribute is None:
            attr = list(self.df.columns)
            attr = [a for a in attr if 
                "Dim" not in a and              # remove the positions
                "_DateTime" not in a and        # remove the date times
                "Unnamed" not in a and          # remove the unnamed column that appears randomly 
                self.key not in a and              # remove the unique identifier
                len(self.df[a].unique()) < len(self.df)/len(self.df["_DateTime"].unique())]    # remove any entry which appears only once
        else:
            attr = [self.attribute]
        
        dfAll = self.df.copy()
        newCols = [f"_Distance{a}" for a in attr]
        for a, newCol in zip(attr, newCols):

            attributeValues = self.df[a].unique()

            dfCopy = None
            for av in attributeValues:
                dfAttr = self.df[self.df[a] == av][["Dim0", "Dim1", "Dim2", self.key, "_DateTime", a]]
                centrePoint = np.median(dfAttr[["Dim0", "Dim1", "Dim2"]], 0)
                dists = np.sum((dfAttr[["Dim0", "Dim1", "Dim2"]] - centrePoint)**2, 1)
                dfAttr[newCol] = dists 
                
                if dfAll is None:
                    dfCopy = dfAttr
                else:
                    dfCopy = pd.concat([dfCopy, dfAttr])
        
            dfAll = pd.merge(dfAll, dfCopy[[self.key, "_DateTime", newCol]], on=[self.key, "_DateTime"])

        self.dfDistances = dfAll[[self.key, "_DateTime"] + newCols]
 
    def findOutliers(self):

        '''
        From the distances calculated within each attribute, find identities which are outliers
        relative to the other attribute values
        '''

        attrDist = [d for d in self.dfDistances.columns if "_Distance" in d]

        allAttrDesc = self.dfDistances[attrDist].describe()
        outlierdf = None
        for a in attrDist:
            std = allAttrDesc[a].loc["std"]
            mean = allAttrDesc[a].loc["mean"]
            outliers = self.dfDistances[self.dfDistances[a] > (mean + std*3)][[self.key, "_DateTime"]]
            outliers["type"] = a.replace("_Distance", "")
            if outlierdf is None:
                outlierdf = outliers
            else:
                outlierdf = pd.concat([outlierdf, outliers])

        self.outlierdf = outlierdf


    def outlierEntitlements(self, times = 1):

        '''
        Find the % likelihood of entitlements contributing to the identities uniqueness to that attribute
        
        times is the number of time periods to investigate from the latest (ie if times = 1 then only 
        report on the latest time interval)

        NOTE somewhere in here the privileged permissions need to be excluded and possibly
        a list of exceptions should be imported to prevent re-occuring warnings
        '''

        times = sorted(self.outlierdf["_DateTime"].unique(), reverse = True)[:times]

        info = None
        for t in times:

            outlierIdsTime = self.outlierdf[self.outlierdf["_DateTime"] == t]
            identityTimes = self.identities[self.identities["_DateTime"] == t]
            permissionTimes = self.permissions.iloc[[n for n, i in enumerate(self.permissions.index) if str(t) in i]].T
            permissionTimes.columns = [p.split("_")[0] for p in permissionTimes.columns]

            for _, data in outlierIdsTime.iterrows():
                id = data[self.key]
                idtype = data["type"]
                idattr = self.identities[(self.identities[self.key] == id) & (self.identities["_DateTime"] == t)][data["type"]].values[0]

                attrIdentities = [i for i in identityTimes[(identityTimes[idtype] == idattr)][self.key] if i in permissionTimes.columns]

                # get the permission to compare and the permissions of the target identity
                relevantPermissions = permissionTimes[attrIdentities]
                targetPermissions = relevantPermissions.pop(id)

                '''
                calculate the relative occurence of the entitlements for this attribute distribution:

                 - where diff >0 then the identity has a acquired a permission which is unusual relative to the rest 
                of the attribute population, where 1 means it is ONLY found on that identity
                 - where diff <0 then the identity is missing a permission which is unusual relative to the rest 
                of the attribute population, where -1 means it is ONLY missing for that identity
                '''
                
                diff = targetPermissions - relevantPermissions.sum(1)/relevantPermissions.shape[1]
                    
                targetValues = diff[(diff > 0.8) | (diff < -0.8)].to_frame("Occurence").reset_index()

                targetValues[[self.key, "UnixTime", "Attribute", "Element"]] = [id, t, idtype, idattr]

                if info is None: 
                    info = targetValues
                else:
                    info = pd.concat([info, targetValues])

        self.rawOutlierInfo = info

    def createReport(self, name = ""):

        '''
        Create a report with actional information for each permission for each relevant identity 
        and provide reasoning for the action.

        Sheets:
            - Summary,              Some key summary stats
            - PriorityActions,      contains up to 10 tasks which would improve the permission environment
                                    modelling is also provided to show the impact of taking these actions
            - Important,            up to 100 additional tasks which could improve the permission environment
            - AllInformation,       contains all the detailed information which was used for the calculation the previous sheets
            - Reference,            notes to explain what information is being provided
        
        # NOTE is a sunburst chart useful? https://plotly.com/python/sunburst-charts/
        '''

        def addToReport(ws, rowNo, content):

            for n, con in enumerate(content, 1):
                ws.cell(row=rowNo, column=n).value = con

        # create the report as an excel and save in the downloads
        timeInfo = strftime("%Y%m%d.%H%M", localtime())
        reportPath = f"{os.path.expanduser('~')}\\Downloads\\Report{name}_{timeInfo}.xlsx"
        wb = Workbook()
        del wb["Sheet"]
        # excelExport = pd.ExcelWriter(reportPath)

        # https://github.com/Khan/openpyxl/blob/master/doc/source/tutorial.rst
        

        # -------------- Summary ----------------
        wsSummary = wb.create_sheet("Summary")
        wsSummary.cell(row=1, column=1).value = "Summary statistics sheet"        

        # report on affected users and the number of actions needed to rememdy
        idsAffected = self.rawOutlierInfo[["Value", self.key]].groupby(self.key).count().sort_values(self.key)
        userValue = self.rawOutlierInfo.groupby([self.key, "Value"]).size().reset_index().rename(columns={0:"Count"})
        
        # report on the 10 most mentioned permissions and the number of identities it affects
        permissionsAffected = self.rawOutlierInfo["Value"].value_counts().to_frame("Count").reset_index().rename(columns={"index": "Value"})

        # report on the attributes and their elements most affected
        elementsAffected = self.rawOutlierInfo[["Value", "Element"]].groupby("Element").count().sort_values("Value", ascending = False)


        # -------------- Priority Actions --------------
        '''
        Taking into account both the number of times an individual permission appears as an important
        permission as well as the relative importance to any given identity, select up to 10 permissions
        which should be added to an identity or taken away to improve their relative identity position. 

        NetOccurence:   This value is the absolute sum of the occurences across all identities and the 
                            attributes it affects.
        
        Occurence:      This is the occurence relative to others in the equivalent element defined as:
                            1       = the identity is the only one in the elemtn with the permission 
                                        (unique exception)
                            1>      = the identity has the permission which is unusual 
                                        (rare exception)
                            1>>     = the identity DOES has a permission which most other identities in the element  
                                        also have 
                                        (common exception/other identities under-privileged)
                            0       = the permission is found in all identities 
                                        (element wide permission)
                            -1<<    = the identity DOES NOT have a permission which most other identities in
                                        the element also do not have 
                                        (uncommon exception/other identities over-privileged)
                            -1<     = the identity does not have a permission which is unusual 
                                        (rare exculision )
                            -1      = this identity is the only one in that element without the permission 
                                        (unique exculsion)
        '''

        wsPriorityActions = wb.create_sheet("PriorityActions")
        wsPriorityActions.cell(row=1, column=1).value = "Priority actions to resolve"   
        wsPriorityActions.cell(row=2, column=1).value = "The following information outlines the 10 permissions which are causing the greatest identity discrepancy"
        
            # ------ Specifying which permissions to remove off identities
        addToReport(wsPriorityActions, 4, 
                ["Permissions to action", 
                "Action to take", 
                "Identities impacted", 
                "Attributes impacted", 
                "Likelihood of impact"]
                )

            # ------ Calculations ------
        priority = self.rawOutlierInfo.copy()
        priority["NetOccurence"] = np.abs(priority["Occurence"])
        priority = priority.groupby("Value").sum("NetOccurence").sort_values("NetOccurence", ascending = False).reset_index()
        # MODEL CHANGES WITH PRIORITY PERMISSIONS MADE

        priorityPermissions = list(priority["Value"])

        # for n, pi in enumerate(priority["Value"][:10], 5):
        rowNo = 5
        removeC = 0
        addC = 0
        # For up to 10 permissions (max of 7 to either add or remove), investigate the permissions to 
        # prioritise actions for
        while removeC + addC < 10:
            pi = priorityPermissions.pop(0)
            pAnalysis = self.rawOutlierInfo[self.rawOutlierInfo["Value"] == pi]

            # To remove the permissions
            removeP = pAnalysis[pAnalysis["Occurence"]>0]
            if len(removeP)>0 and removeC < 7:
                removeinfo = [
                    pi,
                    "Remove from identity",
                    ", ".join(list(removeP[self.key].unique())),
                    ", ".join(list(removeP["Attribute"].unique())), 
                    f"{int((np.abs(removeP['Occurence']).min())*100)}% - {int((np.abs(removeP['Occurence']).max())*100)}%"
                ]
                addToReport(wsPriorityActions, rowNo, removeinfo)
                removeC += 1
                rowNo += 1

            # To add the permissions
            addP = pAnalysis[pAnalysis["Occurence"]<0]
            if len(addP)>0 and addC < 7:
                addinfo = [
                    pi,
                    "Add to identity",
                    ", ".join(list(addP[self.key].unique())),
                    ", ".join(list(addP["Attribute"].unique())), 
                    f"{int((np.abs(addP['Occurence']).min())*100)}% - {int((np.abs(addP['Occurence']).max())*100)}%"
                ]
                addToReport(wsPriorityActions, rowNo, addinfo)
                addC += 1
                rowNo += 1

                
        # -------------- Important Actions --------------
        wsImportantActions = wb.create_sheet("ImportAction")

        # -------------- All info ------------------
        wsAllInformation = wb.create_sheet("AllData")
        for r in dataframe_to_rows(self.rawOutlierInfo, index=True, header=True):
            wsAllInformation.append(r)

        
        # -------------- Reference documentation --------------
        wsReference = wb.create_sheet("Reference")

        wb.save(filename = reportPath)

def report_1(df, permissions, identities, uiddf, attribute = None):

    '''
    Report on any identities which have deviated significantly from other identiies with similar permissions as them
    Compare each identities position in relation to other identities with similar attribute information. 
    If their position has changed by some significant amount, include them in a report for possible 
    over/under provisioning
    '''

    distances = Metrics(df, permissions, identities, uiddf)
    distances.calculateDistances()
    distances.findOutliers()    
    distances.outlierEntitlements()
    distances.createReport()
    

    print("Test")

    pass

if __name__ == "__main__":

    df = pd.read_csv("C:\\Users\\jreshef\\Documents\\Projects\\PermissionAnalysis\\results\\Identity3D_1659964744.csv", dtype=str)

    attribute = "Department"
    uiddf = "Username"

    # report_1(df, permissions, identities, attribute, uiddf)