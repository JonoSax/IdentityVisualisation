"""
Unit testing for the datamodel object
"""

import json
import unittest

from dataLoader import CSVData
from dataModel import DataModel


class TestDataModel(unittest.TestCase):

    """
    Test the processing of information through the DataModel object
    """

    def setUpClass(cls):

        print("Setting up test case")

        """
        Identiy in identity, permission, privileged and role data. The CSVData object serves as the baseline for all processing. As the methods in DataModel change they should ALWAYS be able to process the CSV data which serves as the most "naive" format
        """
        cls.DataModel = CSVData.getData(
            identityPath="data/unittestdata/Identities*.csv",
            permissionPath="data/unittestdata/Permissions*.csv",
            privilegedPath="data/unittestdata/PrivilegedData*.csv",
            rolePath="data/unittestdata/RoleData*.csv",
            identityKey="Username",
            permissionKey="Identity",
            permissionValue="Value",
            roleIDKey="Role",
            roleFileKey="Role",
            managerKey="Manager",
            managerIDs=None,  # ["Manager0", "Manager1", "Test"],
            limitData=None,
        )

    def test_processIdentities(self):
        """
        Testing the processIdentities method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_processPermissions(self):
        """
        Testing the processPermissions method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_processRoles(self):
        """
        Testing the processRoles method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_processPrivilege(self):
        """
        Testing the processPrivilege method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_processManager(self):
        """
        Testing the processManager method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_hashData(self):
        """
        Testing the hashData method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_processData(self):
        """
        Testing the processData method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_calculateMDS(self):
        """
        Testing the calculateMDS method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()

    def test_mergeIdentityData(self):
        """
        Testing the mergeIdentityData method of dataModel.DataModel
        """

        testObj = self.DataModel.copy()
