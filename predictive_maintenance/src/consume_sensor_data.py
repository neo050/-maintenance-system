from kafka import KafkaConsumer
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

consumer = KafkaConsumer('sensor_data',
                         bootstrap_servers='localhost:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

temperatures = []

def animate(i):
    for message in consumer:
        data = message.value
        temperatures.append(data['temperature'])
        if len(temperatures) > 100:
            temperatures.pop(0)
        plt.cla()
        plt.plot(temperatures)
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title('Real-Time Temperature Data')

ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout()
plt.show()
