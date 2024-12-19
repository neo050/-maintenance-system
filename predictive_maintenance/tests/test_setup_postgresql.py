# tests/test_setup_postgresql.py

import unittest
from unittest.mock import patch, MagicMock
from IntegrationWithExistingSystems import setup_postgresql
import pandas as pd

class TestSetupPostgreSQL(unittest.TestCase):

    @patch('IntegrationWithExistingSystems.setup_postgresql.create_engine')
    @patch('IntegrationWithExistingSystems.setup_postgresql.yaml.safe_load')
    @patch('IntegrationWithExistingSystems.builtins.open')
    def test_get_db_connection(self, mock_open, mock_yaml_load, mock_create_engine):
        # Mock configuration
        mock_yaml_load.return_value = {'database': {'url': 'postgresql://user:pass@localhost/db'}}

        engine = MagicMock()
        mock_create_engine.return_value = engine

        config = setup_postgresql.load_config('fake_config.yaml')
        connection = setup_postgresql.get_db_connection(config)

        mock_create_engine.assert_called_once_with('postgresql://user:pass@localhost/db')
        self.assertIsNotNone(connection)

    @patch('IntegrationWithExistingSystems.setup_postgresql.pd.read_sql_query')
    def test_fetch_assets(self, mock_read_sql):
        engine = MagicMock()
        df = pd.DataFrame({'asset_id': [1], 'asset_code': ['A1']})
        mock_read_sql.return_value = df

        result = setup_postgresql.fetch_assets(engine)

        mock_read_sql.assert_called_once()
        self.assertFalse(result.empty)

    @patch('IntegrationWithExistingSystems.setup_postgresql.os.makedirs')
    @patch('IntegrationWithExistingSystems.setup_postgresql.pd.DataFrame.to_csv')
    @patch('IntegrationWithExistingSystems.setup_postgresql.fetch_assets')
    @patch('IntegrationWithExistingSystems.setup_postgresql.fetch_maintenance_records')
    @patch('IntegrationWithExistingSystems.setup_postgresql.fetch_preventive_maintenance_records')
    @patch('IntegrationWithExistingSystems.setup_postgresql.get_db_connection')
    @patch('IntegrationWithExistingSystems.setup_postgresql.load_config')
    def test_main(self, mock_load_config, mock_get_db_conn, mock_fetch_preventive,
                  mock_fetch_maintenance, mock_fetch_assets, mock_to_csv, mock_makedirs):
        mock_load_config.return_value = {'database': {'url': 'postgresql://user:pass@localhost/db'}}
        engine = MagicMock()
        mock_get_db_conn.return_value = engine
        df_assets = pd.DataFrame({'asset_id': [1], 'asset_code': ['A1']})
        df_maintenance = pd.DataFrame({'maintenance_id': [1], 'maintenance_code': ['M1']})
        df_preventive = pd.DataFrame({'maintenance_id': [2], 'maintenance_code': ['PM1']})
        mock_fetch_assets.return_value = df_assets
        mock_fetch_maintenance.return_value = df_maintenance
        mock_fetch_preventive.return_value = df_preventive

        setup_postgresql.main()

        mock_fetch_assets.assert_called_once_with(engine)
        mock_fetch_maintenance.assert_called_once_with(engine)
        mock_fetch_preventive.assert_called_once_with(engine)
        self.assertEqual(mock_to_csv.call_count, 3)
        engine.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
