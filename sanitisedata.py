import pandas as pd
import os
import hashlib


def cleandata(file, dest, joiningKey):

    """
    For each column of data, get the column header name, all the unique values in the columns
    and then replace them with numbered versions of the column name header
    """

    fileName = os.path.basename(file)

    m = hashlib.sha256()

    data = pd.read_csv(file, dtype=str, on_bad_lines="skip")

    for c in data.columns:
        values = data[c]
        uniqValues = values.unique()
        if c == joiningKey:
            uniqNames = [hashlib.sha256(v.encode()).hexdigest()[:8] for v in uniqValues]
        else:
            uniqNames = [f"{c} {n}" for n in range(uniqValues.__len__())]
        valueNameDict = {k: v for k, v in zip(uniqValues, uniqNames)}

        data[c] = data[c].apply(lambda x: valueNameDict.get(x))

    data.to_csv(f"{dest}Sanitised_{fileName}.csv", index=False)


if __name__ == "__main__":

    files = ["dataairnz\Identities_EM.csv", "dataairnz\Permissions_EM.csv"]
    keys = ["Username", "Identity"]
    dest = "data\\"

    for f, k in zip(files, keys):
        cleandata(f, dest, k)
