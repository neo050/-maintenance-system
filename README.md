Prerequisites
Before setting up and running the Predictive Maintenance System, ensure that your system meets the following prerequisites:

Operating System:

Windows
Python:

Python 3.12 is required. You can download it from the official Python website. https://www.python.org/downloads/.
Docker:

Docker is essential for managing containerized services like Kafka openmaint and PostgreSQL.
Installation:
Windows :
Download and install Docker Desktop from the official Docker website https://www.docker.com/.

Post-Installation:
Ensure Docker is running. You can verify by running:


docker --version

You should see output similar to Docker version 20.10.7, build f0df350.

Docker Compose:
Docker Compose typically comes bundled with Docker Desktop. Verify its installation:

docker-compose --version
Expected output: docker-compose version 1.29.2, build 5becea4c.



Setup
1. Clone the Repository
Begin by cloning the repository to your local machine using Git.


git clone https://github.com/neo050/-maintenance-system.git
cd -maintenance-system
2. Create a Virtual Environment
It's recommended to use a virtual environment to manage your project's dependencies and avoid conflicts with other projects.


python -m venv venv
3. Activate the Virtual Environment
Activate the virtual environment you just created.

Windows

venv\Scripts\activate

After activation, your command prompt should reflect the active virtual environment, e.g., (venv) C:\path\to\project>.

4. Install Dependencies
Install the required Python packages using pip and the provided requirements.txt file.


pip install -r requirements.txt
Note: If you encounter any issues during installation, ensure that your pip is up to date:


pip install --upgrade pip
5. Run the Application
Once all dependencies are installed, you can run the main application script.


python run.py


