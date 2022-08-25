import os
import numpy as np
import pandas as pd
from time import localtime, strftime
from openpyxl import Workbook
from utilities import *

pd.options.mode.chained_assignment = None

class Metrics:

    '''
    Object to store and calculate metrics

    The overall goal is to make this as modular as possible so that new reports can be created
    easily. ATM it is kind of bespoke but as new report types are created, parts of these function
    can be easily split off and set as seperate functions for reuse.
    '''

    def __init__(self, mdsResult, permissions, identities, uiddf, attribute = None, specificTime = None):

        self.mdsResult = self.setdf(mdsResult)
        self.key = uiddf
        self.attribute = attribute
        self.permissions = permissions.copy()
        self.specificTime = self.getTime(self.mdsResult, specificTime)

        # NOTE this is only until the datamodel object is updated for multi-index as well
        self.permissionsNew = permissions.copy()
        self.permissionsNew[self.key] = [s.split("_")[0] for s in self.permissionsNew.index]
        self.permissionsNew["_DateTime"] = [int(s.split("_")[1]) for s in self.permissionsNew.index]
        self.permissions = self.permissionsNew.set_index([self.key, "_DateTime"])
        
        
        self.identities = self.setdf(identities)
        self.identities = self.identities.set_index([self.key, "_DateTime"])
        
        self.outlierdf = None
        self.dfDistances = None
        self.permissionFixes = None

    def getTime(self, df, val = "max"):

        '''
        For a given dataframe, get the latest time
        '''

        if val == 'max':
            dt = df.reset_index()["_DateTime"].max()
        elif val == 'min':
            dt = df.reset_index()["_DateTime"].min()
        elif type(val) == int:
            dt = sorted(df.reset_index()["_DateTime"].unique())[val]
        else:
            dt = None

        return dt

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

    def calculateDistances(self, df = None, attr = None):

        '''
        Calculate the distances of all object in each attribute unless specified

        Use only the most recent permission information to calculate the centre point, however 
        calculate the distances retrospectively to assess the movement of individuals over time
        '''

        ret = type(df) == pd.DataFrame
        if df is None:
            df = self.mdsResult

        if attr is None:
            attr = list(df.columns)
            attr = [a for a in attr if 
                "Dim" not in a and              # remove the positions
                "_DateTime" not in a and        # remove the date times
                "Unnamed" not in a and          # remove the unnamed column that appears randomly 
                self.key not in a and              # remove the unique identifier
                len(df[a].unique()) < len(df)/len(df["_DateTime"].unique())]    # remove any entry which appears only once
        elif type(attr) is str:
            attr = [attr]

        self.attribute = attr

        dfAll = df.copy()
        newCols = [f"_Distance{a}" for a in attr]
        for a, newCol in zip(attr, newCols):

            attributeValues = df[a].unique()
            
            # Calculate the distance of all identities in each element from the median point at the 
            # most recently recorded permission extract
            dfCopy = None
            for av in attributeValues:
                dfAttr = df[df[a] == av][["Dim0", "Dim1", "Dim2", self.key, "_DateTime", a]]

                centrePoint = np.median(dfAttr[dfAttr["_DateTime"] == self.getTime(dfAttr)][["Dim0", "Dim1", "Dim2"]], 0)
                dists = np.sum((dfAttr[["Dim0", "Dim1", "Dim2"]] - centrePoint)**2, 1)
                dfAttr[newCol] = dists 
                
                if dfAll is None:
                    dfCopy = dfAttr
                else:
                    dfCopy = pd.concat([dfCopy, dfAttr])
        
            # add the new column in with the info
            dfAll = pd.merge(dfAll, dfCopy[[self.key, "_DateTime", newCol]], on=[self.key, "_DateTime"], validate="many_to_one")

        dfAll = dfAll[[self.key, "_DateTime"] + newCols].sort_values([self.key, "_DateTime"]).set_index([self.key, "_DateTime"])

        if ret:
            return dfAll
        else:
            self.dfDistances = dfAll

    def findOutliers(self):

        '''
        From the distances calculated within each attribute, find identities which are 
        outliers (3std away from the median) relative to the other attribute values. 

        These identities are then associated with the attribute in which they are an 
        outlier
        '''

        attrDist = [d for d in self.dfDistances.columns if "_Distance" in d]

        allAttrDesc = self.dfDistances[attrDist].describe()
        outlierdf = None
        for attr in attrDist:
            std = allAttrDesc[attr].loc["std"]
            mean = allAttrDesc[attr].loc["mean"]
            q3 = allAttrDesc[attr].loc["75%"]
            iqr = q3 - allAttrDesc[attr].loc["25%"]

            # 3 standard deviations from the mean of distances
            # outliers = self.dfDistances[self.dfDistances[attr] > (mean + std*3)]

            # 1.5 x iqr beyond the q3 
            outliers = self.dfDistances[self.dfDistances[attr] > (iqr * 1.5 + q3)]

            outliers["type"] = attr.replace("_Distance", "")
            if outlierdf is None:
                outlierdf = outliers
            else:
                outlierdf = pd.concat([outlierdf, outliers])

        self.outlierdf = outlierdf # .reset_index()

    def outlierEntitlements(self, times = 1):

        '''
        Find the % likelihood of entitlements contributing to the identities uniqueness to that attribute
        
        times is the number of time periods to investigate from the most recent permission export
        (ie if times = 1 then only report on the latest time interval based on the outlier information calculated. 
        If times = 3 then report on the 3 most recent time intervals where there is outlier information based on
        the median positions of the permissions at the most recent time avaialbe)

        NOTE somewhere in here the privileged permissions need to be excluded and possibly
        a list of exceptions should be imported to prevent re-occuring warnings
        '''

        times = sorted(self.outlierdf.index.get_level_values(1).unique(),reverse=True)[:times]

        info = None
        for t in times:

            outlierIdsTime = self.outlierdf.loc[(slice(None), t), :].droplevel(1)
            identityTimes = self.identities.loc[(slice(None), t), :].droplevel(1)
            permissionTimes = self.permissions.loc[(slice(None), t), :].droplevel(1).T
            permissionTimes.columns = [p.split("_")[0] for p in permissionTimes.columns]

            for id, data in outlierIdsTime.iterrows():
                idtype = data["type"]

                idattr = self.identities.loc[(id, t)][data["type"]]
                attrIdentities = [i for i in identityTimes[(identityTimes[idtype] == idattr)].index if i in permissionTimes.columns]

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
                    
                # Using a cut of off 80% (either add or remove) because:
                #   1 - If I included all values then the data frame would become unweildly
                #   2 - 80% is just a nice number, coming from the 80/20 rule
                targetValues = diff[(diff > 0.8) | (diff < -0.8)].to_frame("Occurence").reset_index().rename(columns={'index': 'Value'})

                targetValues[[self.key, "UnixTime", "Attribute", "Element"]] = [id, t, idtype, idattr]

                if info is None: 
                    info = targetValues
                else:
                    info = pd.concat([info, targetValues])

        self.rawOutlierInfo = info

    def createReport1(self, name = "1"):

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

        # create the report as an excel and save in the downloads
        timeInfo = strftime("%Y%m%d.%H%M", localtime())
        reportPath = f"{os.path.expanduser('~')}\\Downloads\\Report_{name}_{timeInfo}.xlsx"
        wb = Workbook()
        del wb["Sheet"]
        # excelExport = pd.ExcelWriter(reportPath)

        # https://github.com/Khan/openpyxl/blob/master/doc/source/tutorial.rst
        

        # -------------- Summary ----------------
        wsSummary = wb.create_sheet("Summary")
        
        wsSummary.cell(row=1, column=1).value = "Summary statistics sheet"
        wsSummary.cell(row=2, column=1).value = "This sheet provides some high level summary information about your permission environment and actions to take"

            # ----- Summary stats on permissions and attributes/elements -----

        '''
        For the latest time period available, for each element group calculate the key statistics 
        regarding the number of permissions, number of identities and variance in permissivity 
        '''

        latestTime = self.getTime(self.identities)

        latestIdentities = self.identities.loc[(slice(None), latestTime), :].droplevel(1)
        latestPermissions = self.permissions.loc[(slice(None), latestTime), :].droplevel(1)
        elementAnalysis = pd.DataFrame(None, columns=["Attribute", "Element", "IDCount", "MinPermissions", "LowerQuartile", "Median", "UpperQuartile", "MaxPermissions"])
        elementSpread = pd.DataFrame(None, columns=["Attribute", "Element", "IDCount", "RelativeStandardDeviation", "RelativeSpread"])
        for attr in sorted(self.attribute):
            element = latestIdentities[attr].unique()

            for ele in sorted(element):
                # get the identities in the specific element from the permissions dataframe
                # DONT merge the data though just because we don't actually need a dataframe that big...
                ids = [id for id in latestIdentities[latestIdentities[attr] == ele].index.get_level_values(0) if id in latestPermissions.index]
                

                # Analyse the association of permissions per element
                if len(ids) > 0:
                    elePermissions = latestPermissions.loc[ids,:]
                    idSum = elePermissions.sum(1).describe()
                    elementAnalysis.loc[len(elementAnalysis)] = [attr, ele, idSum["count"], idSum["min"], idSum["25%"], idSum["50%"], idSum["75%"], idSum["max"]]

                # Analyse the dispersion of permissions per element (using a minimum of 5 identities)
                if len(ids) > 5:
                    eleSpread = self.dfDistances.loc[(ids, latestTime), :][f"_Distance{attr}"]
                    idSpread = eleSpread.describe()
                    elementSpread.loc[len(elementSpread)] = [attr, ele, idSum["count"], idSpread["std"], idSpread["max"]-idSpread["min"]]

            # ----- Most highly provisioned element groups -----

        '''
        Using the median number of permissions per identity in any given element
        '''
        rowNo = np.array(4)     # make a mutable object so the addToReport can iterate on the same variable
        wsSummary.cell(row=int(rowNo), column=1).value = "10 most highly provisioned element groups"
        rowNo += 1
        addToReport(wsSummary, rowNo, elementAnalysis.columns)
        elementAnalysis = elementAnalysis.sort_values(["Median", "Element"], ascending = [False, True])
        for _, info in elementAnalysis[:10].iterrows():
            addToReport(wsSummary, rowNo, info)


            # ----- Largest element groups -----
        '''
        Using the number of identities per element
        '''
        rowNo += 2
        wsSummary.cell(row=int(rowNo), column=1).value = "10 most populated elements"
        rowNo += 1
        addToReport(wsSummary, rowNo, elementAnalysis.columns)
        elementAnalysis = elementAnalysis.sort_values(["IDCount", "Element"], ascending = [False, True])
        for _, info in elementAnalysis[:10].iterrows():
            addToReport(wsSummary, rowNo, info)


            # ----- Greatest permission spread in element groups -----
        '''
        Using the normalised standard deviation from the median position of the entitlement per element
        NOTE std not spread becuase spread is closely correlatd with the number of ids vs the actual variance
        in permission position
        '''
        rowNo += 2
        wsSummary.cell(row=int(rowNo), column=1).value = "10 most disperse permission element groups"
        rowNo += 1
        elementSpread["RelativeStandardDeviation"] /= elementSpread["RelativeStandardDeviation"].max()
        elementSpread["RelativeSpread"] /= elementSpread["RelativeSpread"].max()
        elementSpread = elementSpread.sort_values(["RelativeStandardDeviation", "IDCount"], ascending = [False, False])
        addToReport(wsSummary, rowNo, elementSpread.columns)
        for _, info in elementSpread[:10].iterrows():
            addToReport(wsSummary, rowNo, info)

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

        wsPriorityActions = wb.create_sheet("Priority Actions")
        wsPriorityActions.cell(row=1, column=1).value = "Priority actions to resolve"   
        wsPriorityActions.cell(row=2, column=1).value = "The following information outlines the 10 permissions which are causing the greatest identity discrepancy"
        
            # ------ Calculations to specify the impact of permission modification ------
        priority = self.rawOutlierInfo.copy()
        priority["NetOccurence"] = np.abs(priority["Occurence"])
        priority = priority.groupby("Value").sum("NetOccurence").sort_values("NetOccurence", ascending = False).reset_index()

        priorityPermissions = list(priority["Value"])

        rowNo = np.array(4)
        removeC = 0
        addC = 0
        removePermission = {}
        addPermission = {}
        idsImpacted = []
        priorityInfo = []
        # For up to 10 permissions (max of 7 to either add or remove), investigate the permissions to 
        # prioritise actions for
        while removeC + addC < 10 and len(priorityPermissions) > 0:
            pi = priorityPermissions.pop(0)
            pAnalysis = self.rawOutlierInfo[self.rawOutlierInfo["Value"] == pi]

            # To remove the permissions
            removeP = pAnalysis[pAnalysis["Occurence"]>0]
            if len(removeP)>0 and removeC < 7:
                ids = list(removeP[self.key].unique())
                removeinfo = [
                    pi,
                    "Remove from identity",
                    ", ".join(ids),
                    ", ".join(list(removeP["Attribute"].unique())), 
                    f"{int((np.abs(removeP['Occurence']).min())*100)}% - {int((np.abs(removeP['Occurence']).max())*100)}%"
                ]
                priorityInfo.append(removeinfo)
                removeC += 1
                removePermission[pi] = ids
                idsImpacted += ids

            # To add the permissions
            addP = pAnalysis[pAnalysis["Occurence"]<0]
            if len(addP)>0 and addC < 7:
                ids = list(addP[self.key].unique())
                addinfo = [
                    pi,
                    "Add to identity",
                    ", ".join(ids),
                    ", ".join(list(addP["Attribute"].unique())), 
                    f"{int((np.abs(addP['Occurence']).min())*100)}% - {int((np.abs(addP['Occurence']).max())*100)}%"
                ]
                priorityInfo.append(addinfo)
                addC += 1
                addPermission[pi] = ids
                idsImpacted += ids

        # get the unique entries of ids
        idsImpacted = list(set(idsImpacted))

        '''        
        Re-run the MDS with the added and removed permissions and re-calculate the distances of each 
        identity. 

        Identify the % changed distances from the median of their element and whether it is now within 
        the acceptable boundaries or if it requires further action (ie see the important actions sheet). 
        '''

        latestPermissions = self.permissions.loc[(slice(None), latestTime), :].droplevel(1).copy()
        modLatestPermissions = latestPermissions.copy()

        # create the permission data frame from the latest time export
        latestPermissions[[self.key, "Type"]] = [[m.split("_")[0], "Original"] for m in latestPermissions.index]
        latestPermissions = latestPermissions.set_index([self.key, "Type"])

        # create a permission data frame to modify the permission environment
        modLatestPermissions[[self.key, "Type"]] = [[m.split("_")[0], "Modified"] for m in modLatestPermissions.index]
        modLatestPermissions = modLatestPermissions.set_index([self.key, "Type"])

        for a in addPermission:
            ids = addPermission[a]
            modLatestPermissions[a].loc[ids] = 1

        for r in removePermission:
            ids = removePermission[r]
            modLatestPermissions[r].loc[ids] = 0

        allPerm = pd.concat([latestPermissions,modLatestPermissions])

        # re-calculate the mds 
        perMos = mdsCalculation(allPerm)
        dimNames = [f"Dim{n}" for n in range(3)]

        # merge all the positional and identity data together
        entitleExtract = pd.DataFrame(np.hstack([perMos, np.array(list(allPerm.index))]), columns = [*dimNames, *allPerm.index.names])
        entitleExtract = entitleExtract.merge(self.identities.loc[(slice(None), latestTime), :].reset_index(), on=self.key)
        entitleExtract[["Dim0", "Dim1", "Dim2"]] = entitleExtract[["Dim0", "Dim1", "Dim2"]].astype(float)
        entitleExtract["_DateTime"] = entitleExtract["_DateTime"].astype(int)
        entitleExtract.loc[entitleExtract["Type"] == "Modified", "_DateTime"] = latestTime + 1      # iterate the latest time to create the "future" permission
        
        #### RERUN THE OUTLIER CALCULATIONS AND THE DISTANCE FROM THE MEDIAN POINT
        dfNew = self.calculateDistances(entitleExtract, self.attribute)
        oldDists = [dfNew.loc[i].loc[latestTime] for i in idsImpacted]
        newDists = [dfNew.loc[i].loc[latestTime + 1] for i in idsImpacted]
        distInfo = pd.DataFrame([1-n/o for n, o in zip(newDists, oldDists)], index = idsImpacted)
        distDesc = pd.Series(np.hstack(1-np.array(newDists)/np.array(oldDists))).describe()

        prioritySummary = [
            "The impact of performing the following modification to the identities and permissions is a:",
            f"{int(distDesc['min']*100)} - {int(distDesc['max']*100)}% reduction in permission distances across all elements for an median of {int(distDesc['50%']*100)}%",
            f"{distInfo.T.max().idxmax()} in attribute {distInfo.max().idxmax().replace('_Distance', '')} will improve the most"
        ]

        for p in prioritySummary:
            addToReport(wsPriorityActions, rowNo, [p])

        rowNo += 1

        addToReport(wsPriorityActions, rowNo, 
            ["Permissions to action", 
            "Action to take", 
            "Identities impacted", 
            "Attributes impacted", 
            "Likelihood of impact"]
            )

        # write the priority report
        for info in priorityInfo:
            addToReport(wsPriorityActions, rowNo, info)

        # -------------- Important Actions --------------
        wsImportantActions = wb.create_sheet("Important Actions")
        wsImportantActions.cell(row=1, column=1).value = "Actions which would improve your permission environment but should be secondary to the Priority actions"   
        wsImportantActions.cell(row=2, column=1).value = "The following information outlines the next 30 permissions which are causing the greatest identity discrepancy"
        
        addToReport(wsImportantActions, 4, 
            ["Permissions to action", 
            "Action to take", 
            "Identities impacted", 
            "Attributes impacted", 
            "Likelihood of impact"]
            )

        removeC = 0
        addC = 0
        rowNo = 5
        while removeC + addC < 30 and len(priorityPermissions) > 0:
            pi = priorityPermissions.pop(0)
            pAnalysis = self.rawOutlierInfo[self.rawOutlierInfo["Value"] == pi]

            # To remove the permissions
            removeP = pAnalysis[pAnalysis["Occurence"]>0]
            if len(removeP)>0 and removeC < 20:
                removeinfo = [
                    pi,
                    "Remove from identity",
                    ", ".join(list(removeP[self.key].unique())),
                    ", ".join(list(removeP["Attribute"].unique())), 
                    f"{int((np.abs(removeP['Occurence']).min())*100)}% - {int((np.abs(removeP['Occurence']).max())*100)}%"
                ]
                addToReport(wsImportantActions, rowNo, removeinfo)
                removeC += 1
                rowNo += 1

            # To add the permissions
            addP = pAnalysis[pAnalysis["Occurence"]<0]
            if len(addP)>0 and addC < 20:
                addinfo = [
                    pi,
                    "Add to identity",
                    ", ".join(list(addP[self.key].unique())),
                    ", ".join(list(addP["Attribute"].unique())), 
                    f"{int((np.abs(addP['Occurence']).min())*100)}% - {int((np.abs(addP['Occurence']).max())*100)}%"
                ]
                addToReport(wsImportantActions, rowNo, addinfo)
                addC += 1
                rowNo += 1

        # -------------- All info ------------------
        wsAllInformation = wb.create_sheet("Raw Outlier Data")
        wsAllInformation.cell(row=1, column=1).value = "Raw data used for the Priority and Important actions"           
        addToReport(wsAllInformation, 2, list(self.rawOutlierInfo.columns))

        '''
        for r in dataframe_to_rows(self.rawOutlierInfo, index=True, header=True):
            wsAllInformation.append(r)
        '''

        for n, (_, r) in enumerate(self.rawOutlierInfo.iterrows(), 3):
            addToReport(wsAllInformation, n, list(r))
        
        # -------------- Reference documentation --------------
        wsReference = wb.create_sheet("Reference")

        print(f"----- {reportPath} created -----")
        wb.save(filename = reportPath)

def report_1(df, permissions, identities, uiddf, attribute = None):

    '''
    Report on any identities which have deviated significantly from other identiies with similar permissions as them
    Compare each identities position in relation to other identities with similar attribute information. 
    If their position has changed by some significant amount, include them in a report for possible 
    over/under provisioning

    NOTE TODO
        Include a slider next to the report which allows you to dial the sensitiity up and down
    '''

    distances = Metrics(df, permissions, identities, uiddf)
    distances.calculateDistances(attr = attribute)
    distances.findOutliers()    
    distances.outlierEntitlements()
    distances.createReport1()
    
def report_2(df, permissions, identities, uiddf, attribute, sliderRound, sliderCluster, sliderDate, hover_data, name = "2"):

    '''
    Report on the clusters that are observed in the data based on the value of the sliders. 
    Report on the permissions and the other identity attributes and their % breakdown for 
    all identities within each partcular bubble. 

    NOTE TODO
        Report on the permissions which differentiate each clustering
    '''

    timeInfo = strftime("%Y%m%d.%H%M", localtime())
    reportPath = f"{os.path.expanduser('~')}\\Downloads\\Report_{name}_{timeInfo}.xlsx"
    wb = Workbook()
    del wb["Sheet"]
    # excelExport = pd.ExcelWriter(reportPath)

    # https://github.com/Khan/openpyxl/blob/master/doc/source/tutorial.rst
    
    # ---------- Identify the breakdown of permissions in each cluster ----------

    wsPermAnalysis = wb.create_sheet("Permission analysis")
    wsPermAnalysis.cell(row=1, column=1).value = "Analysis of the permission per cluster"   
    wsPermAnalysis.cell(row=2, column=1).value = "The following information breaks down the % of identities each cluster who have the corresponding permission."
    wsPermAnalysis.cell(row=3, column=1).value = "The permissions are ordered from the most to least common across all individuals (including those not clustered)."
    rowNo = np.array(5)   

    aggDict = aggDict = {hd: lambda x: list(set(x))[0] if len(set([str(n) for n in x]))==1 else x for hd in hover_data if hd != attribute}
    pos = Metrics(df, permissions, identities, uiddf, attribute, sliderDate)
    dfMod = pos.mdsResult
    dfMod = df[df["_DateTime"] == df["_DateTime"].unique()[sliderDate]].reset_index(drop=True)
    dfPos = clusterData(dfMod, uiddf, attribute, sliderRound, aggDict)
    dfPos = dfPos.set_index("_ClusterID")
    dfClusters = dfPos[dfPos["_Count"] > sliderCluster * dfPos["_Count"].max()]
    clusterNames = sorted(dfClusters.index)
    dfAggregate = pd.DataFrame(None, columns=clusterNames)

    # get the breakdown of the permissions in each cluster
    for clid, dfCluster in dfClusters.iterrows():
        uids = dfCluster[uiddf]
        if type(uids) is str:
            uids = [uids]
        
        dfAggregate[clid] = pos.permissions.loc[[(i, pos.specificTime) for i in uids]].sum(0)/len(uids)

    # sort the aggregate df by the order of the number of identities which have the permissions 
    # (highest to lowest)
    permissionOrder = pos.permissions.loc[(slice(None), pos.specificTime), :].sum(0).sort_values(ascending = False)
    dfAggregate = dfAggregate.reindex(permissionOrder.index)

    # remove entries where the value is 0 acroos all identities (ie no identity has this permission)
    dfAggregate[dfAggregate==0] = None
    dfAggregate = dfAggregate.dropna(how = "all").fillna(0)

    addToReport(wsPermAnalysis, rowNo, ["Cluster ID"] + clusterNames)
    addToReport(wsPermAnalysis, rowNo, ["Element clustering"] + [dfPos.loc[c][attribute] for c in clusterNames])
    addToReport(wsPermAnalysis, rowNo, ["Cluster Count"] + [int(dfPos.loc[c]["_Count"]) for c in clusterNames])
    addToReport(wsPermAnalysis, rowNo, ["Permission Value"])
    rowNo += 1
    for idx, dfag in dfAggregate.iterrows():
        addToReport(wsPermAnalysis, rowNo, [idx] + [np.round(d, 2) for d in dfag])

    # ---------- Identify the breakdown of elements of each attribute in each cluster ----------

    wsAttrAnalysis = wb.create_sheet("Attribute analysis")
    wsAttrAnalysis.cell(row=1, column=1).value = "Analysis of the attributes and elements per cluster"   
    wsAttrAnalysis.cell(row=2, column=1).value = "The following information breaks down the % of elements in cluster for each identity attribute."
    wsAttrAnalysis.cell(row=3, column=1).value = "Each table is ordered from the one with the least to the most number of distinct elements."
    rowNo = np.array(5)   

    # Select the attributes for analysis
    analyseAttrs = [h for h in hover_data if h.find("_") == -1 and h != uiddf and h != attribute]

    # get the breakdown of the other attributes in each cluster
    attrBreakDowns = {}
    for attr in analyseAttrs:
        allDfAttr = pos.identities.loc[(slice(None), pos.specificTime), :][attr].unique()
        dfAttr = pd.DataFrame(None, index = sorted(allDfAttr), columns = clusterNames)
        dfCluster = dfClusters[[attr]]
        for clid, dc in dfCluster.iterrows():
            elements = dc[attr]
            if type(elements) is str:
                elements = np.array([elements])

            # update each element with the proporption of each element in the cluster
            for e in list(set(elements)):
                dfAttr.loc[e].loc[clid] = (elements == e).sum()/elements.__len__()

        # drop rows where there are no inputted values and replace nan with 0's
        dfAttr = dfAttr.dropna(how = 'all').fillna(0)
        attrBreakDowns[attr] = dfAttr

    # order the attributes by the least number of unique elements first
    attrOrderX = np.array([len(attrBreakDowns[a]) for a in attrBreakDowns]).argsort()
    attrOrder = [list(attrBreakDowns.keys())[x] for x in attrOrderX]

    addToReport(wsAttrAnalysis, rowNo, ["Cluster ID"] + clusterNames)
    addToReport(wsAttrAnalysis, rowNo, ["Element Clustering"] + [dfPos.loc[c][attribute] for c in clusterNames])
    addToReport(wsAttrAnalysis, rowNo, ["Cluster Count"] + [int(dfPos.loc[c]["_Count"]) for c in clusterNames])
    rowNo += 1
    for attr in attrOrder:
        attrDf = attrBreakDowns[attr]
        addToReport(wsAttrAnalysis, rowNo, [f"Attribute: {attr}"])
        for idx, dfag in attrDf.iterrows():
            addToReport(wsAttrAnalysis, rowNo, [idx] + [np.round(d, 2) for d in dfag])
        rowNo += 1

    wb.save(filename = reportPath)

if __name__ == "__main__":

    df = pd.read_csv("C:\\Users\\jreshef\\Documents\\Projects\\PermissionAnalysis\\results\\Identity3D_1659964744.csv", dtype=str)

    attribute = "Department"
    uiddf = "Username"

    # report_1(df, permissions, identities, attribute, uiddf)