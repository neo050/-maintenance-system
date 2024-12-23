import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

# Load data for visualization
data = pd.read_csv('../data/raw/ai4i2020.csv')

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Predictive Maintenance Dashboard'),

    dcc.Graph(
        id='sensor-data',
        figure={
            'data': [
                go.Scatter(
                    x=data['timestamp'],
                    y=data['sensor_value'],
                    mode='lines',
                    name='Sensor Value'
                ),
                go.Scatter(
                    x=data['timestamp'],
                    y=data['predictions'],
                    mode='lines',
                    name='Predictions'
                )
            ],
            'layout': {
                'title': 'Sensor Data and Predictions'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
