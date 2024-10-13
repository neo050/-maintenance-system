# SensorDataSimulator.py

import logging
import subprocess
import os
import sys
import time

from RealTimeProcessing.SensorDataSimulatorClient import SensorDataSimulator

def main():
    simulator = None
    try:
        simulator = SensorDataSimulator()
        simulator.start_simulation()  # Assuming this method starts the simulation
    except Exception as e:
        if simulator:
            simulator.logger.error(f"Simulation encountered an error: {e} end of simulation")
            simulator.cleanup()
        else:
            # Initialize a temporary logger
            logging.basicConfig(level=logging.ERROR)
            temp_logger = logging.getLogger(__name__)
            temp_logger.error(f"Simulation encountered an error before simulator initialization: {e}")
    finally:
        if simulator:
            simulator.cleanup()

if __name__ == "__main__":
    main()