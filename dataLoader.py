#!/usr/bin/python

import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd

from dataModel import DataModel

# https://dash.plotly.com/interactive-graphing


"""
TODO
    - Move the process of the CSVs into the dataModel object. The file specific object should ONLY read in the data to the expected data format and the dataModel object should do the processing, pivot tables etc.
"""


class CSVData(DataModel):

    """
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

    """

    def getData(
        self,
        identityPath: str,
        permissionPath: str,
        privilegedPath: str,
        rolePath: str,
        identityKey: str,
        permissionKey: str,
        permissionValue: str,
        roleFileKey=None,
        roleIDKey=None,
        managerKey=None,
        managerIDs=None,
        limitData=None,
    ):

        """
        Ingest the raw information and process for the dataModel

        Inputs
        -----

        reclaculate : boolean
            If True then load all info, otherwise the pre-computed info will be loaded

        identityPath : str
            Path to the CSV of the identity information

        permissionPath : str
            Path to the CSV of the permission information

        privilegedPath : str
            Path to the CSV of the list of permissions which are associated with privileged/elevenated access

        rolePath : str
            Path to the CSV of the list of permissions which are associated with roles

        identityKey : str
            The joining key used to connect the identity and permission dataframes on the identity data

        permissionKey : str
            The joining key used to connect the identity and permission dataframes on the permission data

        permissionValue : str
            The columns value used to model the relationships between the identities

        roleFileKey : str
            The column in the role file which links the name of the role

        roleIDKey : str
            The column value used to identify the identities roles

        managerKey : str
            The column in the identity file which describes the position of the manager

        managerIDs : list
            The manager which you want to include in the identity visualisation process. If None then show all

        limitData : int, default = None (load all files available)
            The maximum number of historical permission files to process.

        Outputs
        -----
        DataModel : obj
            The attributes of the data model populated
        """

        self.identityPath = identityPath
        self.permissionPath = permissionPath
        self.privilegedPath = privilegedPath
        self.rolePath = rolePath
        self.joinKeys["identity"] = identityKey
        self.joinKeys["permission"] = permissionKey
        self.joinKeys["rolefilekey"] = roleFileKey
        self.joinKeys["roleidkey"] = roleIDKey
        self.joinKeys["managerkey"] = managerKey
        self.managerIDs = managerIDs
        self.permissionValue = permissionValue

        self.getIdentityData()
        self.getPermissionData(limitData)
        self.getPrivilegedData()
        self.getRoleData()

    def getIdentityData(self):

        """
        Read in the raw identity data
        """

        dataPaths = sorted(
            glob(self.identityPath), reverse=False
        )  # get up to the 5 most recent files
        iAll = None
        for path in dataPaths:
            _, fileName = os.path.split(path)

            # get the unix datetime from the file. If it is in the file name read it, otherwise get it from the file creation date
            if "_dt_" in fileName:
                date = int(fileName.split("_")[-1].split(".")[0])
            else:
                date = int(os.path.getmtime(path))
            date -= (
                date + 13 * 60**2
            ) % 86400  # round to the nearest day, compensate for the UTC timing
            iStore = pd.read_csv(path, dtype=str, on_bad_lines="skip").dropna(
                how="all"
            )  # [:500]
            iStore["_DateTime"] = str(date)  # this is only needed for the pivot table

            if iAll is None:
                iAll = iStore
            else:
                iAll = pd.concat([iAll, iStore])

        self.rawIdentityData = iAll.sort_values(
            [self.joinKeys["identity"], "_DateTime"]
        )

    def getPermissionData(self, limit):

        """
        extract the raw permission data
        """

        # Read in all permission data which is stored from the wildcard search
        pAll = None
        dataPaths = sorted(
            glob(self.permissionPath), reverse=False
        )  # get up to the 5 most recent files
        if limit is not None:
            dataPaths = dataPaths[-limit:]

        # read in multiple files and combine. This assumes that multiple files with the same
        # key name are temporal versions of the information
        for path in dataPaths:
            if "_dt_" in path:
                date = int(path.split("_")[-1].split(".")[0])
            else:
                date = int(os.path.getmtime(path))
            date -= (
                date + 13 * 60**2
            ) % 86400  # round to the nearest day, compensate for the UTC timing

            pStore = pd.read_csv(path, dtype=str, on_bad_lines="skip").dropna(how="all")
            pStore["_DateTime"] = date

            # remove any duplicate entries of the access and the identity
            if pAll is None:
                pAll = pStore
            else:
                pAll = pd.concat([pAll, pStore])

        pAll.drop_duplicates(
            subset=[self.permissionValue, self.joinKeys["permission"], "_DateTime"],
            inplace=True,
            keep="first",
        )

        self.rawPermissionData = pAll

    def getPrivilegedData(self):

        """
        Read in the privileged permission data only if there is information
        """

        if self.privilegedPath is not None:
            self.privilegedData = pd.read_csv(
                self.privilegedPath, dtype=str, on_bad_lines="skip"
            ).dropna(how="all")

    def getRoleData(self):

        """
        Read in the data which corresponds to the role descriptions
        """

        if self.rolePath is not None:
            self.rawRoleData = pd.read_csv(
                self.rolePath, dtype=str, on_bad_lines="skip"
            ).dropna(how="all")


# ----------- Data processing and visualisation (main function) ----------
def CsvData(forceRecalculate=False):

    identityPath = "data/RBACImplementationTest/IdentitiesFake_*.csv"
    permissionPath = "data/EntitlementsFakeAll_*.csv"
    permissionPath = "data/RBACImplementationTest/EntitlementsFake50_*.csv"
    limitData = 7

    identityPath = "data/IdentityPermissionCreepTest/IdentitiesFake_*.csv"
    permissionPath = "data/IdentityPermissionCreepTest/Small/EntitlementsFake50_*.csv"
    limitData = 7
    permissionPath = "data/IdentityPermissionCreepTest/Full/EntitlementsFake_*.csv"
    limitData = 4

    privilegedPath = "data/PrivliegedData.csv"
    privilegedPath = None

    rolePath = "data/RoleData.csv"
    rolePath = None

    privilegedPath = "fakedata/PrivliegedData.csv"
    privilegedPath = None

    rolePath = "fakedata/RoleData.csv"
    rolePath = None

    permissionPathn = "data/Sanitised_Permissions*.csv"
    identityPathn = "data/Sanitised_Identities*.csv"

    csvData = CSVData()

    csvData.getData(
        identityPath=identityPath,
        permissionPath=permissionPath,
        privilegedPath=privilegedPath,
        rolePath=rolePath,
        identityKey="Username",
        permissionKey="Identity",
        permissionValue="Value",
        roleIDKey="Department",
        roleFileKey="Role",
        managerKey="Manager",
        managerIDs=None,  # ["Manager0", "Manager1", "Test"],
        limitData=limitData,
    )
    csvData.processData(forceRecalculate)

    return csvData


if __name__ == "__main__":

    print("Loading....")
    args = sys.argv
    if any(["LaunchExcel" in a for a in args]):
        workbook, worksheet, dims, identityID, permissionID = args = sys.argv[2:]
        excelData(
            str(workbook), str(worksheet), int(dims), str(identityID), str(permissionID)
        )
    else:
        workbook = "C:/Users/ResheJ/Downloads/WorkBook-Hashedv1.xlsm"
        workbook = "WorkBook-FakeData.xlsm"
        worksheet = "SimilarityScoreIdentities"
        dims = 3
        CsvData()
        # excelData(workbook, worksheet, dims, "Username", "Identity")
