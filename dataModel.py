import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import os
from dashboard import launchApp
from utilities import *
from hashlib import sha256
from pandas.util import hash_pandas_object as hash_pd

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
        joinKeys:                   (dict),             Dictionary containing the keys used to join the identity and permission data together
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

        self.rawPermissionData = pd.DataFrame(None)
        self.privilegedData = pd.DataFrame(None)
        self.roleData = pd.DataFrame(None)
        self.identityData = pd.DataFrame(None)
        self.mdsResults = pd.DataFrame(None)
        self.processingType = "identity"
        self.joinKeys = {"identity": None, "permission": None}
        self.permissionValue = None
        self.mdsSavedResults = None

        self.dir = {}
        self.dir["wd"] = f"{os.getcwd()}\\"
        self.dir["results"] = f"{self.dir['wd']}results\\"
        self.dir["data"] = f"{self.dir['wd']}data\\"

    def plotMDS(self):

        """
        Take the data information and report the information in the dashboard
        """

        launchApp(self)

    def processType(self, processName: str):

        if "ident" in processName.lower():
            processingType = "Identity"
        elif "attr" in processName.lower():
            processingType = "Attribute"
        else:
            processingType = None
            print("No valid processing type inputted")

        return processingType

    def processRoles(self):

        """
        Take the role data and process into the necesary array for calculations
        """

        self.roleData = self.roleData.pivot_table(
            values=self.roleData.columns[0],
            columns="Permission",
            index="RoleAssignment",
            aggfunc="count",
            fill_value=0,
        )

        # set the categories of the roles as -1. This can never be set by the timing because you cannot get a negative time!
        self.categories = np.r_[
            self.categories, [[r, "-1"] for r in self.roleData.index]
        ]

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

        if len(self.roleData) > 0:
            self.processRoles()

        # Use the hashed value of all data frame info
        # NOTE just using the last 8 values rather than the whole thing
        self.hashValue = sha256(
            np.r_[
                hash_pd(self.identityData, index=True).values,
                hash_pd(self.rawPermissionData, index=True).values,
                hash_pd(self.privilegedData, index=True).values,
                hash_pd(self.roleData, index=True).values,
            ]
        ).hexdigest()[-8:]

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
            self.calculateMDS()
            dttime = int(datetime.utcnow().timestamp())
            self.mdsResults.to_csv(f"{self.dir['results']}{csvName}_{dttime}.csv")

        else:
            csvPath = csvFiles[-1]
            print(f"     Using {csvPath} to load pre-calculated results")
            self.mdsResults = pd.read_csv(csvPath)

    def calculateMDS(self):

        print("     Calculating similarity matrix")
        # the rawPermissionData MUST be a pandas dataframe with the columns being the permissions
        # and the rows being the identities

        # apply the impact of privileged permissions
        pos = mdsCalculation(self.rawPermissionData, self.privilegedData, self.roleData)
        print(f"     Fitting complete")

        # get n-dimension labels
        dimNames = [f"Dim{n}" for n in range(len(pos[0]))]

        entitleExtract = pd.DataFrame(
            np.hstack([pos, self.categories]),
            columns=[*dimNames, *self.categoriesHeader],
        )

        # force the positions and datetime to be float/int
        entitleExtract[["Dim0", "Dim1", "Dim2"]] = entitleExtract[
            ["Dim0", "Dim1", "Dim2"]
        ].astype(float)
        entitleExtract["_DateTime"] = entitleExtract["_DateTime"].astype(int)

        # merge all identity information if available, remove unnecessary columns, standardise headings to have no spaces
        if self.identityData is not None:
            selectColums = [
                r
                for r in list(self.identityData.columns)
                if r.lower().find("unnamed") == -1
            ]
            posidentitiesSelect = self.identityData[selectColums]
            formatColumns0 = [
                r.replace(" ", "") for r in list(posidentitiesSelect.columns)
            ]
            identityExtract = posidentitiesSelect.set_axis(
                formatColumns0, axis=1, inplace=False
            )

            # Join the data based on the available information
            if "_DateTime" in identityExtract.columns:
                entitleExtract["_DateTime"] = entitleExtract["_DateTime"].astype(int)
                identityExtract["_DateTime"] = identityExtract["_DateTime"].astype(int)
                entitleExtract = entitleExtract.sort_values("_DateTime")
                identityExtract = identityExtract.sort_values("_DateTime")
                identityExtract["_IdentityDateTime"] = identityExtract[
                    "_DateTime"
                ].apply(
                    lambda x: datetime.fromtimestamp(int(x)).strftime(
                        "%m/%d/%Y, %H:%M:%S"
                    )
                )
                # match for identity extracts with the closest in time to the entitlement extract. If there is no identity matched then the individual who is modelled will still be included (this is a left join), however they will have not associated identity data.
                # NOTE a tolerance of a week, tolerance = 604800
                posdf = pd.merge_asof(
                    entitleExtract,
                    identityExtract,
                    on="_DateTime",
                    left_by=self.joinKeys["permission"],
                    right_by=self.joinKeys["identity"],
                    direction="nearest",
                )
                # posdf = posdf.drop(self.joinKeys["permission"])     # Keep just the uid from the identity dataframe
            else:

                posdf = pd.merge(
                    entitleExtract, identityExtract, left_on=self.joinKeys["permission"]
                )

        else:
            posdf = entitleExtract

        # update attribute
        self.mdsResults = posdf
