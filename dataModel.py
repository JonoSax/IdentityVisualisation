import numpy as np
import pandas as pd
from sklearn import manifold
from glob import glob
from datetime import datetime
import os
from dashboard import launchApp

# NOTE this has been set to turn off unnecessary warnings regarding a df modification in the 
# plotMDS function
pd.options.mode.chained_assignment = None

# https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html 

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
        self.mdsSavedResults = None

        self.dir = {}
        self.dir['wd'] = f"{os.getcwd()}\\"
        self.dir['results'] = f"{self.dir['wd']}results\\"
        self.dir['data'] = f"{self.dir['wd']}data\\"

    def plotMDS(self):

        '''
        Take the data information and report the information in the dashboard
        '''

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

    def processData(self, dims = 3, recalculate = True):

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
        
        # evaluation if we need to recalculate or if a file can just be loaded
        if not recalculate:
            recalculate = len(csvFiles) == 0

        # if either forced to recalculate or no relevant files found, recalculate the MDS, else load the relevant file
        if recalculate:

            print("     Recalculation beginning")
            self.calculateMDS(dims)
            dttime = int(datetime.utcnow().timestamp())
            self.mdsResults.to_csv(f"{self.dir['results']}{csvName}_{dttime}.csv")
        
        else:
            csvPath = csvFiles[-1]
            print(f"     Using {csvPath} to load pre-calculated results")
            self.mdsResults = pd.read_csv(csvPath)

    def calculateMDS(self, dims):

        # if only raw data provided then perform the similarities calculation
        if self.rawPermissionData is not None:

            print("     Calculating similarity matrix")

            x = self.rawPermissionData
            self.similarityPermissionData = np.dot(x, x.T)/np.sum(x, 1)

        dissimilarity = 1 - self.similarityPermissionData * self.similarityPermissionData.transpose()

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

