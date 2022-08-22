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
    # NOTE applying privileged permissions significantly increases the computational cost of the similarityPermissionData
    # calculation
    if privilegedData is not None:
        permissionData[privilegedData["Permission"]] *= np.array(privilegedData["RelativePrivilege"].astype(int))
    
    # compute the relative similarity of each data point
    '''
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
    '''
    x = permissionData.to_numpy().astype(np.float32)
    similarityPermissionData = np.dot(x, x.T)/np.sum(x, 1)

    dissimilarity = 1 - similarityPermissionData * similarityPermissionData.transpose()

    # perform dimensionality reduction
    print("     Starting mds fit")

    mds = manifold.MDS( 
        n_components=dims, 
        max_iter=1000,
        eps=1e-9,
        random_state=np.random.RandomState(seed=3),
        dissimilarity="precomputed",
        n_jobs=4, 
        verbose=2, 
        metric = True
    )

    isomap = manifold.Isomap(
        n_neighbors = 10,
        n_components = 3,
    )

    '''
    Enforce the dissimilarities to float32 as a compromise of accuracy and speed
    For a 350 x 350 matrix:
    d64 = 7.942158341407776 sec per iteration
    d32  = 5.387955260276795 sec
    d16 = 6.188236093521118 sec
    '''
    # results = mds.fit(dissimilarity.astype(np.float32), )
    # pos = results.embedding_
    # NOTE memory issue on windows for large array size: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
    pos = mds.fit_transform(dissimilarity)
    # pos = isomap.fit(dissimilarity).embedding_

    return(pos)