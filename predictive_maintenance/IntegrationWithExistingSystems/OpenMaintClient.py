#!/usr/bin/python3
import os
import sys
import requests
import json
import logging
from datetime import datetime
import urllib.parse

class OpenMaintClient:
    def __init__(self, api_url, username, password, log_file_path='../logs/OpenMaintClient.log'):
        self.api_url = api_url
        self.username = username
        self.password = password
        self.session = None
        self.logger = self.setup_logger(log_file_path)
        self.login()

    def setup_logger(self, log_file_path):
        # Remove existing handlers
        #for handler in logging.root.handlers[:]:
            #logging.root.removeHandler(handler)

        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Capture all logs

        # Create handlers
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # File handler for logging to a file (DEBUG level and above)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Stream handler for logging to the console (INFO level and above)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        return logger

    def login(self):
        """Authenticate with the API and initialize the session."""
        auth_url = f"{self.api_url}/sessions?scope=service&returnId=true"
        payload = {'username': self.username, 'password': self.password}
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        self.session = requests.Session()
        response = self.session.post(auth_url, json=payload, headers=headers)

        if response.status_code == 200 and response.json().get("success"):
            session_id = response.json().get("data", {}).get("_id")
            if session_id:
                self.session.headers.update({
                    "CMDBuild-Authorization": session_id,
                    "Content-Type": 'application/json',
                    "Accept": 'application/json'
                })
                self.logger.info("Login successful, session initialized.")
            else:
                self.logger.error("Failed to retrieve session ID.")
                raise ValueError("Session ID retrieval failed.")
        else:
            self.logger.error(f"Login failed: {response.text}")
            raise ValueError(f"Login failed: {response.text}")

    def logout(self):
        """Close the session."""
        if self.session:
            self.session.close()
            self.logger.info("Session closed.")

    # Helper Methods
    def get_lookup_id(self, lookup_type, code):
        """Retrieve the ID of a lookup value given its type and code or description."""
        try:
            encoded_lookup_type = urllib.parse.quote(lookup_type, safe='')
            response = self.session.get(f"{self.api_url}/lookup_types/{encoded_lookup_type}/values")
            if response.status_code == 200 and response.json().get('success'):
                for item in response.json().get('data', []):
                    if item.get('code') == code or item.get('description') == code:
                        return item.get('_id')
                self.logger.warning(f"Lookup code '{code}' not found for '{lookup_type}'.")
            else:
                self.logger.error(f"Failed to retrieve lookup values for '{lookup_type}': {response.text}")
        except Exception as e:
            self.logger.error(f"Error in get_lookup_id: {e}")
        return None

    def get_employee_id(self, employee_name, class_name='Employee'):
        """Retrieve the ID of an employee given their name."""
        try:
            response = self.session.get(
                f"{self.api_url}/classes/{class_name}/cards",
                params={'CQL': f"Description = '{employee_name}'"}
            )
            if response.status_code == 200 and response.json().get('success'):
                data = response.json().get('data', [])
                if data:
                    employee_id = data[0].get('_id')
                    return employee_id
                else:
                    self.logger.warning(f"Employee '{employee_name}' not found in class '{class_name}'.")
            else:
                self.logger.error(f"Failed to retrieve employee '{employee_name}' from class '{class_name}': {response.text}")
        except Exception as e:
            self.logger.error(f"Error in get_employee_id: {e}")
        return None

    def get_site_id(self, site_code):
        """Retrieve the ID of a site given its code."""
        try:
            response = self.session.get(
                f"{self.api_url}/classes/Site/cards",
                params={'CQL': f"Code = '{site_code}'"}
            )
            if response.status_code == 200 and response.json().get('success'):
                data = response.json().get('data', [])
                if data:
                    site_id = data[0].get('_id')
                    return site_id
                else:
                    self.logger.warning(f"Site '{site_code}' not found.")
            else:
                self.logger.error(f"Failed to retrieve site '{site_code}': {response.text}")
        except Exception as e:
            self.logger.error(f"Error in get_site_id: {e}")
        return None

    def get_ci_id(self, asset_code, class_name="Equipment"):
        """Retrieve the CI ID based on asset code and class name."""
        try:
            cards_url = f"{self.api_url}/classes/{class_name}/cards"
            cql_query = f'Code = "{asset_code}"'
            response = self.session.get(cards_url, params={'CQL': cql_query})
            if response.status_code == 200 and response.json().get('success'):
                data = response.json().get('data', [])
                if data:
                    ci_id = data[0].get('_id')
                    self.logger.info(f"CI found: {ci_id}")
                    return ci_id
                self.logger.warning(f"No CI found with code '{asset_code}'.")
            else:
                self.logger.error(f"Error retrieving CI: {response.text}")
        except Exception as e:
            self.logger.error(f"Error in get_ci_id: {e}")
        return None

    # Core Functionality
    def create_work_order(self, work_order_data):
        """Create a work order."""
        wo_url = f"{self.api_url}/processes/CorrectiveMaint/instances"
        response = self.session.post(wo_url, json=work_order_data)
        if response.status_code == 200 and response.json().get("success"):
            work_order_id = response.json().get("data", {}).get("_id")
            self.logger.info(f"Work order created with ID: {work_order_id}")
            return work_order_id
        else:
            self.logger.error(f"Failed to create work order: {response.text}")
            raise ValueError(f"Failed to create work order: {response.text}")

    def get_work_order_details(self, work_order_id):
        """Retrieve and return details of a specific work order."""
        wo_detail_url = f"{self.api_url}/processes/CorrectiveMaint/instances/{work_order_id}"
        response = self.session.get(wo_detail_url)
        if response.status_code == 200 and response.json().get("success"):
            work_order = response.json().get("data", {})
            self.logger.info(f"Work order details retrieved for ID: {work_order_id}")
            return work_order
        else:
            self.logger.error(f"Failed to retrieve work order details: {response.text}")
            raise ValueError(f"Failed to retrieve work order details: {response.text}")

    def get_current_activity(self, process_class, process_id):
        """Retrieve the current open activity for a process instance."""
        activities_url = f"{self.api_url}/processes/{process_class}/instances/{process_id}/activities"
        response = self.session.get(activities_url)
        if response.status_code == 200 and response.json().get("success"):
            activities = response.json().get("data", [])
            if not activities:
                self.logger.error("No activities found.")
                return None
            for activity in activities:
                activity_status = activity.get("status")
                if activity_status in ["open", "running"]:
                    return activity
            self.logger.error("No open activity found.")
            return None
        else:
            self.logger.error(f"Failed to retrieve activities: {response.text}")
            raise ValueError(f"Failed to retrieve activities: {response.text}")

    def get_available_transitions(self, process_class, process_id, activity_id):
        """Retrieve available transitions for a given activity."""
        transitions_url = f"{self.api_url}/processes/{process_class}/instances/{process_id}/activities/{activity_id}/transitions"
        response = self.session.get(transitions_url)
        if response.status_code == 200 and response.json().get("success"):
            transitions = response.json().get("data", [])
            return transitions
        else:
            self.logger.error(f"Failed to retrieve transitions: {response.text}")
            raise ValueError(f"Failed to retrieve transitions: {response.text}")

    def execute_transition(self, process_class, process_id, activity_id, transition_id, data={}):
        """Execute a transition for a given activity."""
        execute_url = f"{self.api_url}/processes/{process_class}/instances/{process_id}/activities/{activity_id}"
        payload = {
            "_activity": {
                "_id": activity_id
            },
            "_transition": {
                "_id": transition_id
            }
        }
        payload.update(data)  # Include any additional data required by the transition
        response = self.session.post(execute_url, json=payload)
        if response.status_code == 200 and response.json().get("success"):
            self.logger.info("Transition executed successfully.")
        else:
            self.logger.error(f"Failed to execute transition: {response.text}")
            raise ValueError(f"Failed to execute transition: {response.text}")

    def get_all_employees(self):
        """Retrieve all Employees in the system."""
        try:
            employees = []
            offset = 0
            limit = 50  # Adjust the limit as needed
            more_data = True
            while more_data:
                params = {'limit': limit, 'offset': offset}
                response = self.session.get(f"{self.api_url}/classes/Employee/cards", params=params)
                if response.status_code == 200 and response.json().get('success'):
                    data = response.json().get('data', [])
                    if not data:
                        more_data = False  # No more data to fetch
                    else:
                        employees.extend(data)
                        offset += limit
                else:
                    self.logger.error(f"Failed to retrieve employees: {response.text}")
                    more_data = False
            self.logger.info(f"Total employees retrieved: {len(employees)}")
            return employees
        except Exception as e:
            self.logger.error(f"Error in get_all_employees: {e}")
            return []

    # You can add more methods as needed
