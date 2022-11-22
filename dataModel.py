import os
from datetime import datetime
from glob import glob
from hashlib import sha256

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object as hash_pd

from utilities import *

# NOTE this has been set to turn off unnecessary warnings regarding a df modification in the
# plotMDS function
pd.options.mode.chained_assignment = None

# https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html


class DataModel(object):

    """
    This is the base object which takes ingested data in the below described formats and provides the abilities to
    process, write and display results/data.
    """

    def __init__(self):

        """
        [attribute]                 [object type]       [description]
        rawPermissionData:          (np.array),         Columns of identities/attributes and rows of permissions as boolean
        privilegedData              (pd.DataFrame),     Columns of permission data and their associated level of privilege where 1 = baseline permission and 3 = system critical
        identityData:               (pd.DataFrame),     Columns of identity attriutes and rows of identities which are unique (by some key)
        mdsResults:                 (pd.DataFrame),     The dimensionally reduced data with corresponding attributes (if applicable)
        categories:                 (list),             List of the individual attribute values use in the calculations
                                                            NOTE this should correpsond to the same position as the relevant data in the *Data attributes
        categoriesHeader:           (list),             List of the attribute types used in the calculations
                                                            NOTE this should correpsond to the same position as the relevant data in the *Data attributes
        processingTypes:            (string),           Specified as either Identity or Attribute (determines calculation and plotting properties)

        dir:                        (dict),             Dictionary containing the directories and paths used
        joinKeys:                   (dict),             Dictionary containing the keys used to join the identity, permission and role data together
        permissionValue:            (str),              The attribute value in the permission dataframe which contains the permission data to model/report on
        mdsSavedResults:            (str),              Path of the current MDS results saved output

        [method]                    [output]            [description]
        plotMDS                     Plotly dashboard    Inputs positional data from the Multi-Dimensional Scaling calculation/s and outputs the dashboard
        reportData:                 Excel document      Inputs relevant raw data attributes of the DataModel and outputs the relevant excel reports
        reportMDS                   Excel document      Inputs the MDS outputs and outputs the relevant excel reports
        calculateMDS                MDS calculation     Inputs the self.*Data and outputs the MDS representation

        """

        # NOTE Rename these to better reflect what data goes into them
        self.categories = None
        self.categoriesHeader = None

        self.rawRawPermissionData = pd.DataFrame(None)
        self.rawPrivilegedData = pd.DataFrame(None)
        self.rawRoleData = pd.DataFrame(None)
        self.rawIdentityData = pd.DataFrame(None)

        self.rawPermissionData = pd.DataFrame(None)
        self.privilegedData = pd.DataFrame(None)
        self.roleData = pd.DataFrame(None)
        self.identityData = pd.DataFrame(None)
        self.mdsResults = pd.DataFrame(None)
        self.processingType = "identity"
        self.joinKeys = {
            "identity": None,
            "permission": None,
            "rolefilekey": None,
            "roleidkey": None,
            "managerkey": None,
        }
        self.managerIDs = None
        self.permissionValue = None
        self.mdsSavedResults = None

        self.dir = {}
        self.dir["wd"] = f"{os.getcwd()}\\"
        self.dir["results"] = f"{self.dir['wd']}results\\"
        self.dir["data"] = f"{self.dir['wd']}data\\"

        if not os.path.exists(self.dir["results"]):
            os.mkdir(self.dir["results"])
        if not os.path.exists(self.dir["data"]):
            os.mkdir(self.dir["results"])

    def plotMDS(self):

        """
        Take the data information and report the information in the dashboard
        """

        # launchApp(self)
        pass

    def processType(self, processName: str):

        """
        NOTE depreceated
        """

        if "ident" in processName.lower():
            processingType = "Identity"
        elif "attr" in processName.lower():
            processingType = "Attribute"
        else:
            processingType = None
            print("No valid processing type inputted")

        return processingType

    def processIdentities(self):

        self.identityData = self.rawIdentityData

    def processPermissions(self):

        # this is joining the identity and permission data as only the identities in the
        # identityData are INCLUDED in the permission data.
        # If the identity is present for any time extract in the identity data then it
        # will be included from all permission data
        """
        pStore = pStore[pStore[self.joinKeys["permission"]].isin(idSelect)]
        pStore[pivotSelect] = (
            pStore[self.joinKeys["permission"]] + f"_{date}"
        )  # this is only needed for the pivot table
        """
        # pStore["_DateTime"] = f"_{date}"

        # perform a pivot of the raw csv data to create a sparse matrix of occurence
        headers = list(self.rawPermissionData.columns.unique())
        headers.remove(self.joinKeys["permission"])
        headers.remove(self.permissionValue)
        pivot = self.rawPermissionData.pivot_table(
            values=headers[0],
            columns=self.permissionValue,
            index=[self.joinKeys["permission"], "_DateTime"],
            aggfunc="count",
            fill_value=0,
        )

        self.categoriesHeader = [self.joinKeys["permission"], "_DateTime"]
        self.categories = np.array([*pivot.index.to_numpy()])
        self.processingType = self.processType("Identity")
        self.permissionData = pivot

    def processRoles(self):

        """
        Take the role data and process into the necesary array for calculations
        """

        if len(self.rawRoleData) == 0:
            return

        modRoleData = self.rawRoleData
        modRoleData["_DateTime"] = -1

        pivot = modRoleData.pivot_table(
            values=self.rawRoleData.columns[0],
            columns="Value",
            index=[self.joinKeys["rolefilekey"], "_DateTime"],
            aggfunc="count",
            fill_value=0,
        )
        self.roleData = pivot

        # add the role information to the identity dataframe
        roledf = pd.DataFrame(None, columns=self.identityData.columns)
        roledf[self.joinKeys["roleidkey"]] = pivot.index.get_level_values(0)
        roledf[self.joinKeys["identity"]] = pivot.index.get_level_values(0)
        roledf["_DateTime"] = -1

        self.identityData = pd.concat([self.identityData, roledf])

        # add the role division information and fake datetime to the permission array (when it is processes)
        self.categories = np.r_[self.categories, np.array([*pivot.index.to_numpy()])]

    def processPrivilege(self):

        """
        Take the pivileged permission data and process it
        """

        if len(self.privilegedData) == 0:
            return

        self.privilegedData = self.privilegedData.set_index("Permission")

    def processManager(self):

        """
        Take the list of managers inputted and return only the identities which have managers
        """

        if self.managerIDs is not None:
            self.identityData = self.identityData[
                self.identityData[self.joinKeys["managerkey"]].isin(self.managerIDs)
            ]

    def hashData(self, len=8):

        """
        Create a hash value to represent the RAW data.

        Use raw not processed to allow for change in the processing of the information
        """

        self.hashValue = sha256(
            np.r_[
                # hash_pd(self.rawIdentityData, index=True).values,
                hash_pd(self.rawPermissionData, index=True).values,
                hash_pd(self.rawPrivilegedData, index=True).values,
                hash_pd(self.rawRoleData, index=True).values,
            ]
        ).hexdigest()[-len:]

    def processData(self, forceRecalculate=True, dims=3):

        """
        Calculate the MDS of the information provided

        Inputs
        -------
        dims : int, default = 3
            Dimensionality reduction of the data (must be either 2 or 3D)

        recalculate : boolean, default = True
            If True always calculate, else only calculate if there are no relevant files

        Outputs
        ------
        self.mdsResults : np.array
            Dimensionally reduced data positions for the dims specifications

        csvfile : None
            The dimensionally reduce data positions are saved in the self.dict['results'] folder

        """

        if dims != 2 and dims != 3:
            print("     Impossible dimensions inputted")
            return

        self.processPermissions()
        self.processIdentities()
        self.processPrivilege()
        self.processRoles()
        self.processManager()
        self.hashData()

        # attribute data requires the category information as well
        if "Attr" in self.processingType:
            joinedCategoryHeaders = "_".join(sorted(self.categoriesHeader))
            csvName = f"{self.processingType}{dims}D_{joinedCategoryHeaders}"
        # identity data requires the processing type and dimensionality
        else:
            csvName = f"{self.processingType}{dims}D_{self.hashValue}"

        # find any relevant files
        csvFiles = sorted(glob(f"{self.dir['results']}{csvName}*.csv"))

        # if either forced to recalculate or no relevant files found, recalculate the MDS,
        # else load the most recently created relevant file
        if forceRecalculate or len(csvFiles) == 0:

            print("     Recalculation beginning")
            mdsPositions = self.calculateMDS("isomap")
            dttime = int(datetime.utcnow().timestamp())
            mdsPositions.to_csv(f"{self.dir['results']}{csvName}_{dttime}.csv")

        else:
            csvPath = csvFiles[-1]
            print(f"     Using {csvPath} to load pre-calculated results")
            mdsPositions = pd.read_csv(csvPath, dtype=str)

        self.mergeIdentityData(mdsPositions, keep_misssing_ids=False)

    def calculateMDS(self, method="isomap"):

        print("     Calculating similarity matrix")
        # the rawPermissionData MUST be a pandas dataframe with the columns being the permissions
        # and the rows being the identities

        # prime the mds calculation?
        # mdsCalculation(self.permissionData[:10], verbose=0)

        # apply the impact of privileged permissions
        pos = mdsCalculation(
            self.permissionData, self.privilegedData, self.roleData, method=method
        )
        print(f"     Fitting complete")

        # get n-dimension labels
        dimNames = [f"Dim{n}" for n in range(len(pos[0]))]

        entitleExtract = pd.DataFrame(
            np.hstack([pos, self.categories]),
            columns=[*dimNames, *self.categoriesHeader],
        )

        # force the positions and datetime to be float/int
        entitleExtract[["__Dim0", "__Dim1", "__Dim2"]] = entitleExtract[
            ["__Dim0", "__Dim1", "__Dim2"]
        ].astype(float)
        entitleExtract["_DateTime"] = entitleExtract["_DateTime"].astype(int)

        entitleExtract["Sum of permissions"] = self.permissionData.sum(1).values

        return entitleExtract

    def mergeIdentityData(self, mdsPositions, keep_misssing_ids=False):

        """
        Combine the mds data with whatever identity data exists
        Add the number of permissions assigned to each identity at each time interval

        mdsPositions : pd.DataFrame
            Unique identifiers and their positions is 3D space

        keep_missing_ids : Boolean
            Check if there are ids in the permission data which do not exist in the identity data. If set to True, keep these data and add in "No ID data" to their attributes. If False (default) remove from further processing.
        """

        # merge all identity information if available, remove unnecessary columns
        if self.identityData is not None:
            selectColums = sorted(
                [
                    r
                    for r in list(self.identityData.columns)
                    if r.lower().find("unnamed") == -1
                ]
            )
            identityExtract = self.identityData[selectColums]

            # Join the data based on the available information
            if "_DateTime" in identityExtract.columns:
                mdsPositions["_DateTime"] = mdsPositions["_DateTime"].astype(int)
                identityExtract["_DateTime"] = identityExtract["_DateTime"].astype(int)
                mdsPositions = mdsPositions.sort_values("_DateTime")

                # add in the datetimes into a human readable format
                identityExtract = identityExtract.sort_values("_DateTime")
                identityExtract["Identity Datetime"] = identityExtract[
                    "_DateTime"
                ].apply(lambda x: create_datetime(x) if x > 0 else None)

                mdsPositions["Permission Datetime"] = mdsPositions["_DateTime"].apply(
                    lambda x: create_datetime(x) if x > 0 else None
                )

                # match for identity extracts with the closest in time to the entitlement extract. If there is no identity matched then the individual who is modelled will still be included (this is a left join), however they will have not associated identity data.
                # NOTE a tolerance of a week, tolerance = 604800
                posdf = pd.merge_asof(
                    mdsPositions,
                    identityExtract,
                    on="_DateTime",
                    left_by=self.joinKeys["permission"],
                    right_by=self.joinKeys["identity"],
                    direction="nearest",
                )

                # where the modelled position did not match to any explicilty stored identity, exclude the identities and permissions not explicitly joined
                if not keep_misssing_ids:
                    posdf = posdf[~posdf[self.joinKeys["identity"]].isna()]

                    self.permissionData = self.permissionData[
                        self.permissionData.index.isin(
                            self.identityData[self.joinKeys["identity"]][
                                self.identityData["_DateTime"] != -1
                            ].unique(),
                            0,
                        )
                    ]

                    # NOTE should include logging here which outputs which identities from each permission and identity data set are NOT included

                posdf.fillna("No ID data", inplace=True)
                # posdf = posdf.drop(self.joinKeys["permission"], axis = 1)     # Keep just the uid from the identity dataframe

            else:

                posdf = pd.merge(
                    mdsPositions,
                    identityExtract,
                    left_on=self.joinKeys["permission"],
                    right_on=self.joinKeys["identity"],
                ).fillna("No ID data")

        else:
            posdf = mdsPositions

        posdf[["__Dim0", "__Dim1", "__Dim2"]] = posdf[
            ["__Dim0", "__Dim1", "__Dim2"]
        ].astype(float)

        self.mdsResults = posdf
