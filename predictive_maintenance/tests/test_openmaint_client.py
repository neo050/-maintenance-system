# tests/test_openmaint_client.py
import unittest
from unittest.mock import patch, MagicMock
from IntegrationWithExistingSystems.OpenMaintClient import OpenMaintClient

class TestOpenMaintClient(unittest.TestCase):

    def setUp(self):
        self.api_url = 'http://fake-api-url'
        self.username = 'username'
        self.password = 'password'
        # We won't instantiate the client here, as each test might need to patch differently.

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_logout(self, mock_post):
        """
        Test that logout properly closes the session.
        """
        # Mock the login response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )
        self.assertIsNotNone(client.session)

        client.logout()
        self.assertIsNone(client.session)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_site_id(self, mock_get, mock_post):
        """
        Test retrieving a site ID.
        """
        # Mock login
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        # Mock GET for site
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': [{'_id': 789, 'Code': 'SiteCode123'}]}
        )

        client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )
        site_id = client.get_site_id('SiteCode123')

        self.assertEqual(site_id, 789)
        mock_get.assert_called_once_with(
            f"{self.api_url}/classes/Site/cards",
            params={'CQL': "Code = 'SiteCode123'"}
        )

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_ci_id_not_found(self, mock_get, mock_post):
        """
        Test that get_ci_id returns None and logs a warning if no matching CI is found.
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': []}
        )

        client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )
        client.logger = MagicMock()

        ci_id = client.get_ci_id('nonexistent_code')
        self.assertIsNone(ci_id)
        client.logger.warning.assert_called_with("No CI found with code 'nonexistent_code'.")

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_lookup_id_api_failure(self, mock_get, mock_post):
        """
        Test that get_lookup_id returns None and logs an error if the API call fails.
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        mock_get.return_value = MagicMock(
            status_code=500,
            json=lambda: {'success': False, 'error': 'Internal Server Error'}
        )

        client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )

        client.logger = MagicMock()

        lookup_id = client.get_lookup_id('COMMON - Priority', '3')
        self.assertIsNone(lookup_id)
        client.logger.error.assert_called()

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_login_success(self, mock_post):
        """
        Test successful login sets session headers.
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        client =   client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )

        self.assertIn('CMDBuild-Authorization', client.session.headers)
        self.assertEqual(client.session.headers['CMDBuild-Authorization'], 'fake-session-id')

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.OpenMaintClient.wait_for_api')  # Add this patch
    def test_login_failure(self, mock_wait_for_api, mock_post):
        """
        Test that login raises ValueError if the API call indicates a failure.
        """
        mock_post.return_value = MagicMock(
            status_code=401,
            json=lambda: {'success': False, 'error': 'Invalid credentials'}
        )
        mock_wait_for_api.return_value = None  # Ensure it's mocked

        with self.assertRaises(ValueError):
            OpenMaintClient(self.api_url, self.username, self.password, skip_wait=True, skip_docker=True)

        mock_wait_for_api.assert_not_called()  # Since skip_wait=True

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    def test_get_lookup_id(self, mock_get, mock_post):
        """
        Test that get_lookup_id returns the correct ID if found.
        """
        # Mock login
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )
        # Mock get
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': [{'code': '3', '_id': 123}]}
        )

        client =   client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )
        lookup_id = client.get_lookup_id('COMMON - Priority', '3')
        self.assertEqual(lookup_id, 123)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_create_work_order(self, mock_post):
        """
        Test creation of a work order returns the new work order ID.
        """
        # First post is login, second post is create_work_order
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: {'success': True, 'data': {'_id': 'session-id'}}),
            MagicMock(status_code=200, json=lambda: {'success': True, 'data': {'_id': 'wo-123'}})
        ]

        client = client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )
        work_order_data = {'some': 'data'}
        wo_id = client.create_work_order(work_order_data)
        self.assertEqual(wo_id, 'wo-123')

        self.assertEqual(mock_post.call_count, 2)

    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_get_work_order_details(self, mock_post, mock_get):
        """
        Test that get_work_order_details retrieves the correct details dict.
        """
        # login
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'session-id'}}
        )
        # get for the details
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'work_order_id', 'details': 'some details'}}
        )

        client =   client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,
            skip_docker=True
        )
        details = client.get_work_order_details('work_order_id')
        self.assertEqual(details['_id'], 'work_order_id')

    @patch('IntegrationWithExistingSystems.OpenMaintClient.OpenMaintClient.wait_for_api')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.subprocess.run')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.time.sleep')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.get')
    @patch('IntegrationWithExistingSystems.OpenMaintClient.requests.Session.post')
    def test_get_employee_id(
            self,
            mock_post,
            mock_get,
            mock_time_sleep,
            mock_subprocess_run,
            mock_wait_for_api
    ):
        # 1) Mock the Docker calls
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        # 2) Prevent any real sleeping
        mock_time_sleep.return_value = None

        # 3) Since skip_wait=True, wait_for_api should not be called
        # No need to set mock_wait_for_api.return_value

        # 4) Mock the POST for login
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {'success': True, 'data': {'_id': 'fake-session-id'}}
        )

        # 5) Mock the GET for get_employee_id
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                'success': True,
                'data': [{'_id': 456, 'Description': 'John Doe'}]
            }
        )

        # 6) Instantiate the client with skip_wait=True to skip wait_for_api
        client = OpenMaintClient(
            api_url='http://fake-api-url',
            username='fakeuser',
            password='fakepass',
            skip_wait=True,  # <-- Change here
            skip_docker=True
        )

        # 7) Test get_employee_id
        emp_id = client.get_employee_id('John Doe')
        self.assertEqual(emp_id, 456)

        # 8) Verify mocks
        mock_wait_for_api.assert_not_called()
        mock_time_sleep.assert_not_called()
        mock_get.assert_called_with(
            'http://fake-api-url/classes/Employee/cards',
            params={'CQL': "Description = 'John Doe'"}
        )


if __name__ == '__main__':
    unittest.main()
