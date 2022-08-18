import numpy as np
from sklearn import manifold
import pandas as pd

def mdsCalculation(permissionData: pd.DataFrame, privilegedData = None, dims = 3):

    '''
    Perform the Muli-dimensional Scaling of the data

    Input
    ---- 
    
    permissionData : pd.DataFrame
        A pandas dataframe containing the permission data where 1 indicates the 
        identity has the permission and 0 indicates the id doesn't have it:
                Perm0   Perm1   Permn   ...
        ID0     0       1       0
        ID1     1       0       1
        IDm     0       1       1
        ...

    privilegedData : pd.DataFrame
        A pandas dataframe containing two columns, the name of the permission and 
        the relative "privilege" of this compared to a standard permissions. 

        NOTE Relative privilege should be on a scale of 2-5 where 1 is the default
        value for a permission existing. Higher values can distort the scaling process

        Permissions     RelativePrivilege
        Perm1           2
        Perm2           4
        Permn           x

    '''

    # apply the impact of privileged permissions
    if privilegedData is not None:
        permissionData[privilegedData["Permission"]] *= np.array(privilegedData["RelativePrivilege"].astype(int))
    
    # compute the relative similarity of each data point
    x = permissionData.to_numpy().astype(int)
    similarityPermissionData = np.dot(x, x.T)/np.sum(x, 1)

    dissimilarity = 1 - similarityPermissionData * similarityPermissionData.transpose()

    # perform dimensionality reduction
    print("     Starting mds fit")

    mds = manifold.MDS( 
        n_components=dims, 
        # max_iter=1,
        eps=1e-6,
        random_state=np.random.RandomState(seed=3),
        dissimilarity="precomputed",
        n_jobs=1, 
        verbose=2, 
        metric = True
    )
    pos = mds.fit(dissimilarity).embedding_

    return(pos)