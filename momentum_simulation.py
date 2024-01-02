
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html



def objective_function(x, y, z):
    term1 = np.sin(np.sin(x) + np.cos(y**2) + np.exp(-(x**2 + y**2)) + 1 / (1 + np.exp(-z)))
    term2 = np.log(1 + np.exp(x + y + z))
    term3 = np.tan(x) / np.cos(y)

    return term1 + term2 + term3

def gradient(x, y, z):
    df_dx = np.cos(x) * (np.cos(x) + 1) * np.exp(-(x**2 + y**2)) + np.exp(x + y + z) / (1 + np.exp(x + y + z)) + (1 / (np.cos(y)**2)) * (1 + np.tan(x)**2)
    df_dy = 2 * y * np.cos(y**2) * np.exp(-(x**2 + y**2)) - np.exp(x + y + z) / (1 + np.exp(x + y + z)) + (np.tan(x) * np.sin(y)) / np.cos(y)**2
    df_dz = np.exp(-z) / ((1 + np.exp(-z))**2) - np.exp(x + y + z) / (1 + np.exp(x + y + z))

    return np.array([df_dx, df_dy, df_dz])

def objective_function(x, y, z):
    term1 = np.sin(x) + np.cos(y)
    term2 = np.exp(z) + x**2 + 2*y**2 + 4*z**2
    term3 = 0.1*x*y - 0.5*y*z + 3*x*z
    term4 = np.sin(2*x*y) + np.cos(3*y*z)
    term5 = np.tanh(x) * np.sin(y) * np.arctan(z)

    return term1 + term2 + term3 + term4 + term5

def gradient(x, y, z):
    df_dx = np.cos(x) + 2*x + 0.1*y + 3*z + 2*y*np.cos(2*x*y) + np.tanh(x) * np.arctan(z)
    df_dy = -np.sin(y) + 4*y + 0.1*x - 0.5*z - 3*z*np.sin(2*x*y) + np.tanh(x) * np.arctan(z)
    df_dz = np.exp(z) + 8*z - 0.5*y + 3*x - 2*x*np.sin(2*x*y) - 3*y*np.cos(3*y*z) + np.tanh(x) * np.sin(y) / (1 + z**2)

    return np.array([df_dx, df_dy, df_dz])

def gradient_descent_with_momentum(initial_params, learning_rate, momentum, num_iterations):
    params = np.array(initial_params)
    velocity = np.zeros_like(params)
    path = [params]

    for _ in range(num_iterations):
        grad = gradient(*params)
        velocity = momentum * velocity - learning_rate * grad
        params = params + velocity
        path.append(params)

    return np.array(path)

def gradient_descent_without_momentum(initial_params, learning_rate, num_iterations):
    params = np.array(initial_params)
    path = [params]

    for _ in range(num_iterations):
        grad = gradient(*params)
        params = params - learning_rate * grad
        path.append(params)

    return np.array(path)

initial_params = [1.0, 1.0, 1.0]
learning_rate = 0.05
momentum = 0.9
num_iterations = 300

# Create initial surface and contour plots
fig_surface_with_momentum = go.Figure()
fig_surface_without_momentum = go.Figure()
fig_contour_with_momentum = go.Figure()
fig_contour_without_momentum = go.Figure()
fig_objective_function = go.Figure()

app = dash.Dash(__name__)
server = app.server()

app.layout = html.Div([
    html.Label("Initial Parameters:"),
    dcc.Slider(id='slider_x', min=-1, max=1, step=0.1, value=initial_params[0]),
    dcc.Slider(id='slider_y', min=-1, max=1, step=0.1, value=initial_params[1]),
    dcc.Slider(id='slider_z', min=-1, max=1, step=0.1, value=initial_params[2]),

    html.Label("Learning Rate:"),
    dcc.Slider(id='slider_learning_rate', min=0.02, max=0.2, step=0.01, value=learning_rate),

    html.Label("Momentum:"),
    dcc.Slider(id='slider_momentum', min=0, max=1, step=0.1, value=momentum),

    html.Label("Iterations:"),
    dcc.Slider(id='slider_iterations', min=50, max=500, step=50, value=num_iterations),

    dcc.Graph(id='graph_surface_with_momentum', figure=fig_surface_with_momentum),
    dcc.Graph(id='graph_contour_with_momentum', figure=fig_contour_with_momentum),
    dcc.Graph(id='graph_surface_without_momentum', figure=fig_surface_without_momentum),
    dcc.Graph(id='graph_contour_without_momentum', figure=fig_contour_without_momentum),
    dcc.Graph(id='graph_objective_function', figure=fig_objective_function),
])

@app.callback(
    [dash.Output('graph_surface_with_momentum', 'figure'),
     dash.Output('graph_contour_with_momentum', 'figure'),
     dash.Output('graph_surface_without_momentum', 'figure'),
     dash.Output('graph_contour_without_momentum', 'figure'),
     dash.Output('graph_objective_function', 'figure')],
    [dash.Input('slider_x', 'value'),
     dash.Input('slider_y', 'value'),
     dash.Input('slider_z', 'value'),
     dash.Input('slider_learning_rate', 'value'),
     dash.Input('slider_momentum', 'value'),
     dash.Input('slider_iterations', 'value')]
)

def update_graphs(x, y, z, learning_rate, momentum, num_iterations):
    # Update initial parameters and hyperparameters
    initial_params = [x, y, z]

    # Run gradient descent with and without momentum
    path_with_momentum = gradient_descent_with_momentum(initial_params, learning_rate, momentum, num_iterations)
    path_without_momentum = gradient_descent_without_momentum(initial_params, learning_rate, num_iterations)

    # Update surface plots
    fig_surface_with_momentum = go.Figure()
    fig_surface_with_momentum.add_trace(go.Surface(x=path_with_momentum[:, 0], y=path_with_momentum[:, 1], z=path_with_momentum[:, 2], colorscale='Viridis'))
    fig_surface_with_momentum.add_trace(go.Scatter3d(x=path_with_momentum[:, 0], y=path_with_momentum[:, 1], z=path_with_momentum[:, 2], mode='markers+lines', marker=dict(size=5, color='red')))
    fig_surface_with_momentum.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                                            title='Gradient Descent with Momentum - Surface Plot')

    fig_surface_without_momentum = go.Figure()
    fig_surface_without_momentum.add_trace(go.Surface(x=path_without_momentum[:, 0], y=path_without_momentum[:, 1], z=path_without_momentum[:, 2], colorscale='Viridis'))
    fig_surface_without_momentum.add_trace(go.Scatter3d(x=path_without_momentum[:, 0], y=path_without_momentum[:, 1], z=path_without_momentum[:, 2], mode='markers+lines', marker=dict(size=5, color='red')))
    fig_surface_without_momentum.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                                               title='Gradient Descent without Momentum - Surface Plot')

    # Update contour plots
    fig_contour_with_momentum = go.Figure()
    fig_contour_with_momentum.add_trace(go.Contour(x=path_with_momentum[:, 0], y=path_with_momentum[:, 1], z=path_with_momentum[:, 2]))
    fig_contour_with_momentum.update_layout(title='Gradient Descent with Momentum - Contour Plot')

    fig_contour_without_momentum = go.Figure()
    fig_contour_without_momentum.add_trace(go.Contour(x=path_without_momentum[:, 0], y=path_without_momentum[:, 1], z=path_without_momentum[:, 2]))
    fig_contour_without_momentum.update_layout(title='Gradient Descent without Momentum - Contour Plot')

    # Update objective function plot
    objective_function_values_with_momentum = [objective_function(*params) for params in path_with_momentum]
    objective_function_values_without_momentum = [objective_function(*params) for params in path_without_momentum]

    fig_objective_function = go.Figure()
    fig_objective_function.add_trace(go.Scatter(x=np.arange(len(objective_function_values_with_momentum)), y=objective_function_values_with_momentum, mode='lines', name='With Momentum'))
    fig_objective_function.add_trace(go.Scatter(x=np.arange(len(objective_function_values_without_momentum)), y=objective_function_values_without_momentum, mode='lines', name='Without Momentum'))
    fig_objective_function.update_layout(title='Objective Function Value over Iterations',
                                         xaxis_title='Iteration',
                                         yaxis_title='Objective Function Value')

    return fig_surface_with_momentum, fig_contour_with_momentum, fig_surface_without_momentum, fig_contour_without_momentum, fig_objective_function



app.run_server(debug=True)
