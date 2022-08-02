import numpy as np
import pandas as pd
import numpy as np
from sklearn import manifold
from glob import glob
from datetime import datetime
import os
from sklearn.metrics import euclidean_distances
from dashboard import launchApp

# https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html 

'''
TODO:
    Create proper logging
    Integrate better into excel etc methods
    Have the data read from here WRITE into excel for the appropriate reporting
    INVESTIGATE the differences in results for using rawPermissionData vs the excel calculated one
'''

class DataModel(object):

    '''
    This is the base object which takes ingested data in the below described formats and provides the abilities to 
    process, write and display results/data.
    '''

    def __init__(self):

        '''
        [attribute]                 [object type]       [description]
        rawPermissionData:          (np.array),         Columns of identities/attributes and rows of permissions as boolean
        similarityPermissionData:   (np.array),         The relative similarity of every data point relative to each other
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
        trackHistorical:            (boolean),          If True load relevant data and include in traces, otherwise just plot the existing data
        mdsSavedResults:            (str),              Path of the current MDS results saved output

        [method]                    [output]            [description]
        plotMDS                     Plotly dashboard    Inputs positional data from the Multi-Dimensional Scaling calculation/s and outputs the dashboard
        reportData:                 Excel document      Inputs relevant raw data attributes of the DataModel and outputs the relevant excel reports
        reportMDS                   Excel document      Inputs the MDS outputs and outputs the relevant excel reports
        calculateMDS                MDS calculation     Inputs the self.*Data and outputs the MDS representation

        '''

        # NOTE Rename these to better reflect what data goes into them
        self.categories = None
        self.categoriesHeader = None

        self.rawPermissionData = None
        self.similarityPermissionData = None
        self.identityData = None
        self.mdsResults = None
        self.processingType = None
        self.joinKeys = {"identity": None, "permission": None}
        self.permissionValue = None
        self.trackHistorical = None
        self.mdsSavedResults = None

        self.dir = {}
        self.dir['wd'] = f"{os.getcwd()}\\"
        self.dir['results'] = f"{self.dir['wd']}results\\"
        self.dir['data'] = f"{self.dir['wd']}data\\"

    def plotMDS(self):

        '''
        Take the data information and report the information in the dashboard
        '''

        if self.trackHistorical is not None:
        # if False:
            # NOTE the sorting here ensures that the data is added chronologically
            files = sorted(glob(f"{self.mdsSavedResults.split('_')[0]}*"))

            for n, f in enumerate(files[-5:]):
                timeUnix = int(f.split("_")[-1].split(".")[0])
                timeString = datetime.fromtimestamp(timeUnix).strftime("%m/%d/%Y, %H:%M:%S")
                df = pd.read_csv(f) 
                df.insert(0, "UnixTime", timeUnix)
                df.insert(0, "DateTime", timeString)
                if type(self.trackHistorical) is bool:
                    self.trackHistorical = df
                else:
                    # add some random noise for testing purposes
                    df[["Dim0", "Dim1", "Dim2"]] = df[["Dim0", "Dim1", "Dim2"]] + [n/10, (n/10)**2, np.random.random() * 0.1]
                    self.trackHistorical = pd.concat([self.trackHistorical, df], axis = 0)

                # self.trackHistorical[["Dim0", "Dim1", "Dim0"]] = self.trackHistorical[["Dim0", "Dim1", "Dim0"]] + np.random.random() + n

        launchApp(self)

    def reportData(self):

        '''
        Take the data information read and generate excel reports related to entitlement anlaysis
        '''

        pass

    def reportMDS(self):

        '''
        Take the MDS results and generate excel reports
        '''

    def processType(self, processName : str):

        if "ident" in processName.lower():
            processingType = "Identity"
        elif "attr" in processName.lower():
            processingType = "Attribute"
        else:
            processingType = None
            print("No valid processing type inputted")

        return processingType

    def calculateMDS(self, dims = 3, recalculate = True):

        '''
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

        '''

        if dims != 2 and dims != 3:
            print("     Impossible dimensions inputted")
            return

        # attribute data requires the category information as well
        if "Attr" in self.processingType: 
            joinedCategoryHeaders =  '_'.join(sorted(self.categoriesHeader))
            csvName = f"{self.processingType}{dims}D_{joinedCategoryHeaders}"
        # identity data requires the processing type and dimensionality 
        else:
            csvName = f"{self.processingType}{dims}D"

        # find any relevant files
        csvFiles = sorted(glob(f"{self.dir['results']}{csvName}*.csv"))
        self.trackHistorical = len(csvFiles) > 2
        if not recalculate:
            recalculate = len(csvFiles) == 0

        # if either forced to recalculate or no relevant files found, recalculate the MDS, else load the relevant file
        if recalculate:

            # if only raw data provided then perform the similarities calculation
            if self.rawPermissionData is not None:

                print("     Calculating similarity matrix")
                
                # NOTE this is the exact same similarity calculation currently performed in 
                # the excel worksheet, how can this be improved? 
                '''
                wid, _ = self.rawPermissionData.shape
                similarities = np.ones((wid, wid))
                for m in range(wid):
                    var0 = self.rawPermissionData[m, :]
                    for n in range(wid):
                        var1 = self.rawPermissionData[n, :]
                        if n >= m: 
                            continue
                        else:
                            try:        similarities[n, m] = (np.dot(var0, var1) * np.dot(var1, var0)) / (np.dot(var0, var0) * np.dot(var1, var1))
                            except:     similarities[n, m] = 0
                self.similarityPermissionData = similarities
                '''
                x = self.rawPermissionData
                self.similarityPermissionData = np.dot(x, x.T)/np.sum(x, 1)

            # create the pre-computed similartiies matrix
            if self.similarityPermissionData is not None:
                dissimilarity = 1 - self.similarityPermissionData * self.similarityPermissionData.transpose()

            # perform dimensionality reduction
            print("     Starting mds fit")

            mds = manifold.MDS( 
                n_components=dims, 
                # max_iter=1,
                eps=1e-3,
                random_state=np.random.RandomState(seed=3),
                dissimilarity="precomputed",
                n_jobs=1, 
                verbose=1
            )
            pos = mds.fit(dissimilarity).embedding_

            print(f"     Fitting complete")

            # get n-dimension labels
            dimNames = [f"Dim{n}" for n in range(len(pos[0]))]

            posdf = pd.DataFrame(np.hstack([pos, self.categories]), columns = [*dimNames, *self.categoriesHeader])

            # merge all identity information if available, remove unnecessary columns, standardise headings to have no spaces
            if self.identityData is not None: 
                selectColums = [r for r in list(self.identityData.columns) if r.lower().find("unnamed") == -1]
                posidentitiesSelect = self.identityData[selectColums]
                formatColumns0 = [r.replace(" ", "") for r in list(posidentitiesSelect.columns)]
                posidentities = posidentitiesSelect.set_axis(formatColumns0, axis=1, inplace=False)
                
                # NOTE maybe use how='inner' as the joining point to remove non-unique identities?
                posdf = posdf.merge(posidentities, left_on=self.joinKeys["permission"], right_on=self.joinKeys["identity"], how='left')

            # update attribute
            self.mdsResults = posdf

            # save the positional data results
            dt_string = int(datetime.utcnow().timestamp())
            csvPath = f"{self.dir['results']}{csvName}_{dt_string}.csv"
            posdf.to_csv(csvPath)

        else:
            csvPath = csvFiles[-1]
            self.mdsResults = pd.read_csv(csvPath)

        # save the path
        self.mdsSavedResults = csvPath
