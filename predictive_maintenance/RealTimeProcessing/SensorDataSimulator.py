# simulate_real_data_runner.py
from SensorDataSimulatorClient import SensorDataSimulator

def main():
    # Specify the path to your Kafka installation directory
    kafka_dir = r"C:\kafka\kafka_2.13-3.7.0"

    # Instantiate the simulator
    simulator = SensorDataSimulator(kafka_dir=kafka_dir)

    # Start the simulation
    try:
        simulator.start_simulation()
    except Exception as e:
        simulator.logger.error(f"Simulation encountered an error: {e}")
    finally:
        simulator.cleanup()

if __name__ == "__main__":
    main()
