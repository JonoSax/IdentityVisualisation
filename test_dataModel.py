"""
Unit testing for the datamodel object
"""

import json
import unittest

from dataLoader import CSVData


class TestDataModel(unittest.TestCase):

    """
    Test the processing of information through the DataModel object
    """

    @classmethod
    def setUpClass(self):

        print("Setting up test case")

        """
        Identiy in identity, permission, privileged and role data. The CSVData object serves as the baseline for all processing. As the methods in DataModel change they should ALWAYS be able to process the CSV data which serves as the most "naive" format
        """
        self.DataModel = CSVData()
        self.DataModel.getData(
            identityPath="data/unittestdata/Identities*.csv",
            permissionPath="data/unittestdata/Permissions*.csv",
            privilegedPath=None,  # "data/unittestdata/PrivilegedData*.csv",
            rolePath=None,  # "data/unittestdata/RoleData*.csv",
            identityKey="Username",  # Unique identifier column name in the identity data
            permissionKey="Identity",  # Unique identitifer column name in the permission data
            permissionValue="Value",  # Permission information columns name in the permission data
            roleIDKey="Role",
            roleFileKey="Role",
            managerKey="Manager",
            managerIDs=None,  # ["Manager0", "Manager1", "Test"],
            limitData=None,
        )

        self.DataModel.processData(False)

        print("DataModel loaded")

    def test_processIdentities(self):
        """
        Testing the processIdentities method of dataModel.DataModel

        At the moment there isn't any processing (just passes the variable through)
        """

        # Assert the identity dataframe identifier is in the table
        self.assertEqual("Username", self.DataModel.joinKeys["identity"])
        self.assertTrue("Username" in self.DataModel.identityData.columns.to_list())

        # Assert the _DateTime column is correctly inserted and formatted into the identity df
        self.assertTrue("Username" in self.DataModel.identityData.columns.to_list())
        self.assertTrue(
            all(
                [
                    type(t) is int
                    for t in self.DataModel.identityData.index.get_level_values(1)
                ]
            )
        )

        # Assert the permissions dataframe identifier is not in the identity table
        self.assertTrue("Identity" not in self.DataModel.identityData.columns.to_list())

    def test_processPermissions(self):
        """
        Testing the processPermissions method of dataModel.DataModel

        Ensuring that the dataframe has been constructed correctly
        """

        # Assert the permission dataframe identifier is in the table
        self.assertEqual("Identity", self.DataModel.joinKeys["permission"])
        self.assertTrue("Identity" in self.DataModel.permissionData.columns.to_list())

        # Assert the identity dataframe identifier is not in the permission table
        self.assertTrue(
            "Username" not in self.DataModel.permissionData.columns.to_list()
        )

        # Assert that every permissions has at least one identity assigned to it and there are no identities which are double counted on the permissions
        self.assertTrue((self.DataModel.permissionData.max() == 1).all())

        # Assert that every identity at every time point has at least one permission assigned to it and there are no permissions which are double counted on the identity
        self.assertTrue((self.DataModel.permissionData.max(1) == 1).all())

        # Assert that the index names of the permission data are the unique identifier for the permissions and _DateTime
        self.assertEqual(
            self.DataModel.permissionData.index.names, ["Identity", "_DateTime"]
        )

        # Assert that permission names are in the columns
        self.assertTrue(
            all(["Access" in a for a in self.DataModel.permissionData.columns])
        )

        # Assert that the identity names are in the first level index
        self.assertTrue(
            all(
                [
                    "User" in u
                    for u in self.DataModel.permissionData.index.get_level_values(0)
                ]
            )
        )

        # Assert the datatime values are integer in the second level index
        self.assertTrue(
            all(
                [
                    type(t) is int
                    for t in self.DataModel.permissionData.index.get_level_values(1)
                ]
            )
        )

    def test_processRoles(self):
        """
        Testing the processRoles method of dataModel.DataModel
        """

        testObj = self.DataModel

    def test_processPrivilege(self):
        """
        Testing the processPrivilege method of dataModel.DataModel
        """

        testObj = self.DataModel

    def test_processManager(self):
        """
        Testing the processManager method of dataModel.DataModel
        """

        testObj = self.DataModel

    def test_hashData(self):
        """
        Testing the hashData method of dataModel.DataModel
        """

        testObj = self.DataModel

    def test_processData(self):
        """
        Testing the processData method of dataModel.DataModel
        """

        testObj = self.DataModel

    def test_calculateMDS(self):
        """
        Testing the calculateMDS method of dataModel.DataModel
        """

        testObj = self.DataModel

    def test_mergeIdentityData(self):
        """
        Testing the mergeIdentityData method of dataModel.DataModel
        """

        testObj = self.DataModel


if __name__ == "__main__":

    unittest.main()
