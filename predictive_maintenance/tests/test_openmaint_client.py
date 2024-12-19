# tests/test_openmaint_client.py

import unittest
from unittest.mock import patch, MagicMock
from IntegrationWithExistingSystems.OpenMaintClient import OpenMaintClient

class TestOpenMaintClient(unittest.TestCase):

    def setUp(self):
        self.api_url = 'http://fake-api-url'
        self.username = 'username'
        self.password = 'password'
        # Do not instantiate OpenMaintClient here
        # self.client = OpenMaintClient(self.api_url, self.username, self.password)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_logout(self, mock_post):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        # Test logout
        client.logout()
        self.assertIsNone(client.session)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_site_id(self, mock_get, mock_post):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        # Mock the get response for site lookup
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': [{'_id': 789, 'Code': 'SiteCode123'}]}
        )
        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        site_id = client.get_site_id('SiteCode123')
        self.assertEqual(site_id, 789)
        mock_get.assert_called_once()
        # Optionally, assert that the GET request was made to the correct URL with correct params
        mock_get.assert_called_with(
            f"{self.api_url}/classes/Site/cards",
            params={'CQL': "Code = 'SiteCode123'"}
        )

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_ci_id_not_found(self, mock_get, mock_post):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        # Mock the get response with empty data
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': []}
        )
        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        ci_id = client.get_ci_id('nonexistent_code')
        self.assertIsNone(ci_id)
        client.logger.warning.assert_called_with("No CI found with code 'nonexistent_code'.")

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_lookup_id_api_failure(self, mock_get, mock_post):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        # Mock API failure
        mock_get.return_value = MagicMock(
            status_code=500,
            json=lambda: {'success': False, 'error': 'Internal Server Error'}
        )
        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        lookup_id = client.get_lookup_id('COMMON - Priority', '3')
        self.assertIsNone(lookup_id)
        client.logger.error.assert_called()

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_login_success(self, mock_post):
        # Mock the login response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'success': True,
            'data': {'_id': 'fake-session-id'}
        }
        mock_post.return_value = mock_response

        # Instantiate the client after the patch is in effect
        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()  # Mock the logger

        self.assertIn('CMDBuild-Authorization', client.session.headers)
        self.assertEqual(client.session.headers['CMDBuild-Authorization'], 'fake-session-id')

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_login_failure(self, mock_post):
        # Mock the login failure response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {'success': False, 'error': 'Invalid credentials'}
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError):
            client = OpenMaintClient(self.api_url, self.username, self.password)
            client.logger = MagicMock()

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_get_lookup_id(self, mock_post, mock_get):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        # Mock the get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'success': True,
            'data': [{'code': '3', '_id': 123}]
        }
        mock_get.return_value = mock_response

        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        lookup_id = client.get_lookup_id('COMMON - Priority', '3')
        self.assertEqual(lookup_id, 123)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_create_work_order(self, mock_post):
        # Mock the login response
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}),
            MagicMock(status_code=200, json=lambda: {'success': True, 'data': {'_id': 'work_order_id'}})
        ]

        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        work_order_data = {'some': 'data'}
        work_order_id = client.create_work_order(work_order_data)

        self.assertEqual(work_order_id, 'work_order_id')
        self.assertEqual(mock_post.call_count, 2)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_get_work_order_details(self, mock_post, mock_get):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        # Mock the get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'success': True,
            'data': {'_id': 'work_order_id', 'details': 'some details'}
        }
        mock_get.return_value = mock_response

        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        work_order_details = client.get_work_order_details('work_order_id')

        self.assertEqual(work_order_details['_id'], 'work_order_id')
        mock_get.assert_called_once()

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_get_employee_id(self, mock_post, mock_get):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        # Mock the get response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'success': True,
            'data': [{'_id': 456, 'Description': 'John Doe'}]
        }
        mock_get.return_value = mock_response

        client = OpenMaintClient(self.api_url, self.username, self.password)
        client.logger = MagicMock()

        employee_id = client.get_employee_id('John Doe')

        self.assertEqual(employee_id, 456)

if __name__ == '__main__':
    unittest.main()
