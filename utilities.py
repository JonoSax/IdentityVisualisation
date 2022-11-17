from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from openpyxl import Workbook
from plotly.colors import hex_to_rgb
from plotly.colors import qualitative as colours
from scipy.sparse import lil_matrix
from sklearn import manifold

# from numba import jit

# @jit


def create_datetime(t):

    """
    Convert a unix timestampe into a human readable time stamp
    """

    return datetime.fromtimestamp(int(t)).strftime("%d/%m/%Y")  # , %H:%M:%S")


def mdsCalculation(
    permissionData: pd.DataFrame,
    privilegedData=pd.DataFrame(None),
    roleData=pd.DataFrame(None),
    dims=3,
    verbose=2,
    method="mds",
    max_iter=500,
):

    """
    ### Description

    Perform the Muli-dimensional Scaling of the data

    ## Inputs

    **permissionData** (pd.DataFrame): A pandas dataframe containing the permission data where 1 indicates the identity has the permission and 0 indicates the id doesn't have it:

    |       |Perm0  |Perm1  |Permn  |...
    --------|-------|-------|-------|
    ID0     |0      |1      |0      |
    ID1     |1      |0      |1      |
    IDm     |0      |1      |1      |
        ...

    **privilegedData** (pd.DataFrame): A pandas dataframe containing two columns, the name of the permission and the relative "privilege" of this compared to a standard permissions.

    NOTE Relative privilege should be on a scale of 2-5 where 1 is the default
    value for a permission existing. Higher values can distort the scaling process

    Permissions |RelativePrivilege
    ------------|------------
    Perm1       |2
    Perm2       |4
    Permn       |x

    ## Outputs

    **pos** (np.array): Numpy array which describe the n-dimenionally reduced information. Each row corresponds to the position of the data described in the row/column of the dissimilarity matrix.

    """

    # apply the impact of privileged permissions
    # NOTE normalising the positions because a dotproduct for values > 1 is significantly slower than values <= 1
    if len(privilegedData) > 0:
        privliegePresent = permissionData.columns[
            permissionData.columns.isin(privilegedData.index)
        ]

        privilegeArray = (
            np.array(
                privilegedData.loc[privliegePresent]["RelativePrivilege"].astype(int)
            )
            / 10
        )

        permissionData.loc[:, privliegePresent] += privilegeArray[privliegePresent]
        permissionData /= permissionData.max().max()

    # Insert the role data
    allPermissionData = pd.concat([permissionData, roleData]).fillna(0)

    # compute the relative similarity of each data point
    """
    Use float32 as once again it balances performance and accuracy. 
    NOTE that for very large arrays (1000x1000+), using int or float16 often don't complete in a
    reasonable time. 
    See https://discourse.julialang.org/t/massive-performance-penalty-for-float16-compared-to-float32/6864/12 
    for a reasonably sensible explanation. 

    For a 500x500 array calcuation of np.dot(n, n.T)/np.sum(n, 1)
    int8 = 0.07399086952209473 sec average for 10 iterations
    int16 = 0.07402501106262208
    int32 = 0.0616971492767334
    int64 = 0.0726935863494873
    f16 = 1.084699773788452
    f32 = 0.0034945011138916016
    f64 = 0.0037222862243652343


    For a 5000x5000 array calculation:
    int8 = ~75 seconds (didn't continue, took too long)
    int16 - 64, based on the int8 time didn't attempt
    f16 = didn't complete a single iteration within 5 minutes
    f32 = 0.5625820875167846
    f64 = 1.3568515539169312

    Summary: f32 is a 20-300+x speed up on other data types and for larger multiplications, 
    is faster than f64. Given the negligible accuracy change f32 is used.
    """
    x = allPermissionData.to_numpy().astype(np.float32)
    similarityPermissionData = np.dot(x, x.T) / np.sum(x, 1)

    dissimilarity = 1 - similarityPermissionData * similarityPermissionData.transpose()

    # perform dimensionality reduction
    print(f"     Starting mds fit with {method}")

    """
    Enforce the dissimilarities to float32 as a compromise of accuracy and speed
    For a 350 x 350 matrix:
    d64 = 7.942158341407776 sec per iteration
    d32  = 5.387955260276795 sec
    d16 = 6.188236093521118 sec
    """

    if method == "mds":

        mds = manifold.MDS(
            n_components=dims,
            max_iter=max_iter,
            eps=1,
            random_state=np.random.RandomState(seed=3),
            dissimilarity="precomputed",
            n_jobs=4,
            verbose=verbose,
            metric=True,
        )

        # NOTE memory issue on windows for large array size: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
        pos = mds.fit_transform(dissimilarity)

    elif method == "isomap":

        isomap = manifold.Isomap(
            n_neighbors=50,  # 10 if len(dissimilarity) > 10 else len(dissimilarity) - 1,
            n_components=3,
            max_iter=max_iter,
        )
        start = time()
        pos = isomap.fit_transform(lil_matrix(dissimilarity))
        print(f"Isomap complete in {np.round(time()-start,0)} sec")

    return pos


def clusterData(df, uidAttr, attribute, sliderRoundValue, dictRules=None):

    """

    ### Description

    Cluster the data based on all identities position environment and an atribue


    Input

    df : pd.DataFrame
        Dataframe of interest which contains information

    uidAttr : str
        Unique identifier of the identities

    attribute : str
        Attribute of interest to investigate. Must be in the columns of the dataframe

    sliderRoundValue : float
        The spatial resolution to round to

    dictRules : dict
        Dictionary of rules corresponding to the columns in the df to process the final aggregated dataframe
    """

    # perform the spatial clustering and tranform the positional information
    sliderRoundValue = np.clip(sliderRoundValue, 0.001, 100)
    dfMod = df.copy()
    dfMod[["Dim0r", "Dim1r", "Dim2r"]] = dfMod[["Dim0", "Dim1", "Dim2"]].apply(
        lambda x: np.round(sliderRoundValue * np.round(x / sliderRoundValue), 2)
    )
    dfMod["_Count"] = dfMod.groupby(["Dim0r", "Dim1r", "Dim2r", attribute])[
        [uidAttr]
    ].transform("count")
    dfUniqID = dfMod[dfMod["_Count"] == 1]

    # Create the aggregation dictionary
    aggDict = {
        "Dim0": "median",
        "Dim1": "median",
        "Dim2": "median",
        "_Count": "median",
    }

    if dictRules is not None:
        aggDict.update(dictRules)

    dfCluster = (
        dfMod[dfMod["_Count"] > 1]
        .groupby(["Dim0r", "Dim1r", "Dim2r", attribute])
        .agg(aggDict)
        .reset_index()
    )
    dfPos = pd.concat([dfUniqID, dfCluster])
    dfPos = dfPos.sort_values(attribute).reset_index(drop=True)

    # add cluster names based on the size of the cluster and the attribute name
    clusterID = (
        dfPos[["_Count", attribute]]
        .sort_values(by=["_Count", attribute], ascending=[False, True])
        .reset_index()
    )

    cStore = []
    for c, _ in clusterID.sort_values("index").iterrows():
        c = str(c)
        while len(c) < len(str(clusterID.__len__())):
            c = "0" + str(c)
        cStore.append(f"Cluster {c}")

    dfPos["_ClusterID"] = cStore
    dfPos["_Count"] = dfPos["_Count"].astype(int)

    # if the aggregation rule does not create data (ie it is just a simple string) then replace the uid value with the clusterID
    # NOTE I think this can actaully be done by the aggregation rule...
    if np.all([type(d) == str for d in dfPos.loc[dfPos["_Count"] > 1, uidAttr]]):
        dfPos.loc[dfPos["_Count"] > 1, uidAttr] = dfPos.loc[
            dfPos["_Count"] > 1, "_ClusterID"
        ]

    return dfPos


def addToReport(ws, content, rowNo=None, colNo=None, colStart=1, rowStart=1):

    """
    Add a rowm or columns to an openpyxl worksheet

    If you specify a rowNo then it will automatically write along columns (from colStart)
    If you specify a colNo then it will automatically write down rows (from rowStart)

    Remember this is excel not python so indexs start at 1 not 0!!!
    """

    if rowNo is not None:
        for n, con in enumerate(content, colStart):
            ws.cell(row=int(rowNo), column=n).value = con
        rowNo += 1

    elif colNo is not None:
        for n, con in enumerate(content, rowStart):
            ws.cell(row=n, column=int(colNo)).value = con
        colNo += 1


def getClusterLimit(dfPos, attribute, sliderClusterValue):

    """
    Select clusters with only a threshold number of identities based on the maximum
    size of the clusters for any given element per attribute
    """

    dfPosSelect = None
    dfPosUnselect = None
    for attr in dfPos[attribute].unique():
        dfTemp = dfPos[dfPos[attribute] == attr]
        minClusteringSize = np.ceil(sliderClusterValue * dfTemp["_Count"].max()) - 1
        dfTempIncl = dfTemp[dfTemp["_Count"] > minClusteringSize]
        dfTempExcl = dfTemp[dfTemp["_Count"] <= minClusteringSize]
        if dfPosSelect is None:
            dfPosSelect = dfTempIncl
            dfPosUnselect = dfTempExcl
        else:
            dfPosSelect = pd.concat([dfPosSelect, dfTempIncl])
            dfPosUnselect = pd.concat([dfPosUnselect, dfTempExcl])

    return dfPosSelect, dfPosUnselect


def filterIdentityDataFrame(dfID, uid, includeInfo=[], excludeInfo=[]):

    """
    From an identity data frame, filter based on multiple columns and specific conditions

    Inputs
    ------
    dfID : pd.DataFrame
        Contains columns of information in a pandas dataframe

    includeInfo/excludeInfo : list of lists
        Contains the column to search for the information and the specific value within that column

    Priorities:

        If there is a conflict between the include/excludes the include will take priority

    """

    # include data dictionary
    includeDict = {}
    for attr, ele in includeInfo:
        if includeDict.get(attr) is None:
            includeDict[attr] = [ele]
        else:
            includeDict[attr].append(ele)

    # exclude data dictionary
    excludeDict = {}
    for attr, ele in excludeInfo:
        if excludeDict.get(attr) is None:
            excludeDict[attr] = [ele]
        else:
            excludeDict[attr].append(ele)

    if includeInfo != [] and excludeInfo != []:
        dfLogic = ~[
            np.all([~dfID[attr].isin(elems) for attr, elems in includeDict.items()], 0)
            | np.all([dfID[attr].isin(elems) for attr, elems in excludeDict.items()], 0)
        ][0]

    elif includeInfo != []:
        dfLogic = np.all(
            [dfID[attr].isin(elems) for attr, elems in includeDict.items()], 0
        )

    elif excludeInfo != []:
        dfLogic = np.all(
            [~dfID[attr].isin(elems) for attr, elems in excludeDict.items()], 0
        )

    else:
        dfLogic = np.array([True] * len(dfID))

    include = dfID[dfLogic]
    exclude = dfID[~dfLogic]

    return include, exclude


def create_colour_dict(
    *dfs,
    coltype="hex",
    transparency=1,
):

    """
    Create the colour dictionary used to annotate the plots

    Dont assume each dataframe has the same attributes
    """

    # create the colour dictionary to be used for all visualisation
    colour_dict = {}
    if coltype == "hex":
        colFunc = lambda x: colours.Plotly[x % len(colours.Plotly)]

    elif coltype == "rgb":
        colFunc = (
            lambda x: f"rgba{tuple(np.append(hex_to_rgb(colours.Plotly[x % len(colours.Plotly)]), transparency))}"
        )

    for n_c, c in enumerate(
        sorted(np.unique(np.concatenate([df.unique().astype(str) for df in dfs])))
    ):

        colour_dict[str(c)] = colFunc(n_c)

    return colour_dict
