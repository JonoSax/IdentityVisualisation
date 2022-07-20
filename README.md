This environment contains: 

* MultiDimAttributeAnalysis.py: This is the script which takes in inputs from the excel file (whatever is used for entitlement analysis), caluculates the similarity matrix and visualises the results.

* requirements.txt: contains the libraries required for this to be installed.

* runme.sh: this shell script will create the virtual environment, install the libraries and compile the script into an executable which can then be moved to the same directory as the excel file.

* compile.sh: this shell script exclusively compiles the py script into a single file executable.

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