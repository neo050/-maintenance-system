# tests/test_openmaint_client.py
import unittest
from unittest.mock import patch, MagicMock
from IntegrationWithExistingSystems.OpenMaintClient import OpenMaintClient

class TestOpenMaintClient(unittest.TestCase):

    def setUp(self):
        self.api_url = 'http://fake-api-url'
        self.username = 'username'
        self.password = 'password'

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_logout(self, mock_post):
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        client.logout()
        self.assertIsNone(client.session)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_site_id(self, mock_get, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': [{'_id': 789, 'Code': 'SiteCode123'}]}
        )

        site_id = client.get_site_id('SiteCode123')
        self.assertEqual(site_id, 789)
            f"{self.api_url}/classes/Site/cards",
            params={'CQL': "Code = 'SiteCode123'"}
        )

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_ci_id_not_found(self, mock_get, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': []}
        )
        client.logger = MagicMock()

        ci_id = client.get_ci_id('nonexistent_code')
        self.assertIsNone(ci_id)
        client.logger.warning.assert_called_with("No CI found with code 'nonexistent_code'.")

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_lookup_id_api_failure(self, mock_get, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        mock_get.return_value = MagicMock(
            status_code=500,
            json=lambda: {'success': False, 'error': 'Internal Server Error'}
        )
        client.logger = MagicMock()

        lookup_id = client.get_lookup_id('COMMON - Priority', '3')
        self.assertIsNone(lookup_id)
        client.logger.error.assert_called()

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_login_success(self, mock_post):

        self.assertIn('CMDBuild-Authorization', client.session.headers)
        self.assertEqual(client.session.headers['CMDBuild-Authorization'], 'fake-session-id')

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')

        with self.assertRaises(ValueError):

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        lookup_id = client.get_lookup_id('COMMON - Priority', '3')
        self.assertEqual(lookup_id, 123)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_create_work_order(self, mock_post):
        mock_post.side_effect = [
        ]

        work_order_data = {'some': 'data'}

        self.assertEqual(mock_post.call_count, 2)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_get_work_order_details(self, mock_post, mock_get):
        mock_post.return_value = MagicMock(
            status_code=200,
        )


    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

                'success': True,
                'data': [{'_id': 456, 'Description': 'John Doe'}]
            }




if __name__ == '__main__':
    unittest.main()
