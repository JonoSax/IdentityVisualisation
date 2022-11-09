PoSE: Permissions of the System Environment
Identity Visualiser (doesn't sound as wanky)

We want the clients PoSE so we visualise, explain and recommend actions to impove their system integrity.

This environment contains: 

* MultiDimAttributeAnalysis.py: This is the script which takes in inputs from the excel file (whatever is used for entitlement analysis), caluculates the similarity matrix and visualises the results.

* requirements.txt: contains the libraries required for this to be installed.

* runme.sh: this shell script will create the virtual environment, install the libraries and compile the script into an executable which can then be moved to the same directory as the excel file.

* compile.sh: this shell script exclusively compiles the py script into a single file executable.

Data inputs

Entitlements and identities need to be ingested at the same time and their time stamp attached. The timestamps should be set on the file name in UTC Linux time. If there is identity data for corresponding entitlement information missing then the previous identity information will be used (and it will be made clear on the data)

This logically how it works: 

    --- Data processing ---

    1 - Read in the relevant worksheet from the specified workbook
    2 - Extract the attribute headings in the data 
    3 - Assess whether the information contained in this workbook has been calculated before or if it is new
        a - Identities, attribute is present anywhere 
        b - Attributes, the exact combination of attributes are present
    
    If new data:
        4 - Extract the data, calculate the multidimensional scaling and save the results as a csv with the naming convention
            {Identity/Attribute sheet}_{Dimensional analysis}_{Attributes present in sheet}_{Time stamp (yymmdd_hhmmss)}
    
    Else:
        4 - return the relevant link to the pre-calculated mds positional values
    
    --- Visualisation ---

    5 - Check if a plot from the mds positional values exists

    If new plot:
        6a - plot the data according to the inputted variables (dimensions, attribute clustering)
        6b - save the plot as an html file for re-loading and distribution

    Else:
        6 - Load the relevant plot

TODO:

    - Create unit testing and validation 

    - Enable the ability to track identities with multiple roles

    - Create reports which explicilty address ISO standards

    - From identified roles, model the roles which minimise errors for specific individuals
        - Recommend the entitlements which would minimise the number of entitlements needed to be exceptional (ADDING only)

    - Implement a Shepard plot to report on the actual distance vs fitted distance for validation purposes

    - Make the reports display in html rather than as an excel doc (makes it nicer to view and more interactive)

    - Get the excel writing/reporting capabilities
        - RBAC: 
            -Recommend clusterings of the data
                - have a k-mean clustering algorithm apply for multiple different numbers
                - find the clustering numbers which 
                    - minimises errors/captures all the identities irrespective of the complexity of the clustering
                    - attempts to enforce the 80/20 rule as best as possible 
                    - attempts to find a clustering pattern which can be made with at most 2 different attributes
                - Pull the identities out that are being targetted
                - Pull the permissions out which have been modelled
                - Of the permissions, attempt to create 2-4 layers of Enterprise, Operational, Functional roles to cover all the newly suggested roles

            - BAU report (similar to outlier report) but instead of linking to the identity spreads relative to their peers, calculate relative to the role position
                - Provide recommendations on permissions that should be added/taken away from the roles/individuals relative to the "ideal role" and the current permissions of the indentities

        - Regular Reporting: 
            - [DONE] Report on any identities which have deviated significantly from other identiies with similar permissions as them
                - Compare each identities position in relation to other identities with similar attribute information. If their position has changed by some significant amount, include them in a report for possible over/under provisioning

            - Report on any identities which have suddenly gained significant privileged permissions
            
            - Report on how various departments/jobs positions/other attributes permission spread is looking and if that is changing

        - Insights:
            - How to incorporate BloodHound (or similar) to identify privileged escaltation and travel in data? 

    - Create new test data sets 
        - Create for a small testing set and a LARGE validation/demonstration set
        - Single identity gaining lots of new permissions
        - RBAC implementation, visualise the reduction in permission explosion
        - Merging departments together
        - Individuals are changing department 
        - Privledged access monitoring

    - Create a new optional privledged permission list and visualise that
        - KIND OF DONE: just input a privileged entitlement list file with weighted values

    - Performance improvements:
        - https://scikit-learn.org/stable/developers/performance.html?highlight=gesd

    - Create the entitlement anlaysis worksheet from this processed data

    - Create proper logging (non-repudiatable?)

    - Add formatting so it looks like PwC product
		    
    - Monitor the level of risk in the permission space, a 2D graph just tracking overall spread and when it reaches a certain threshold it will be obvious

        - Maybe don't monitor everything, just key roles?

    - 
        
