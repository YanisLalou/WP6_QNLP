import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
from statistics import mean, stdev
from plotly.subplots import make_subplots
import numpy as np

def plot_experiment_results(json_file):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract relevant data from the JSON
    experiment_count = data['input_args']['runs']
    iterations = data['input_args']['iterations']

    # Create empty lists to store the accuracy and loss values
    val_accuracies = []
    val_losses = []
    train_accuracies = []
    train_losses = []

    times = []

    # Extract the accuracy and loss values for each runs
    if data.get('val_acc') is not None:
        for i in range(experiment_count):
            val_acc = data['val_acc'][i]
            val_accuracies.append(val_acc)
        val_accuracies = np.array(val_accuracies)

        mean_val_accuracies = np.mean(val_accuracies, axis=0)
        std_val_accuracies = np.std(val_accuracies, axis=0)
       
       
    if data.get('val_loss') is not None:
        for i in range(experiment_count):
            val_loss = data['val_loss'][i]
            val_losses.append(val_loss)
        val_losses = np.array(val_losses)

        mean_val_losses = np.mean(val_losses, axis=0)
        std_val_losses = np.std(val_losses, axis=0)

    if data.get('train_acc') is not None:
        for i in range(experiment_count):
            train_acc = data['train_acc'][i]
            train_accuracies.append(train_acc)
        train_accuracies = np.array(train_accuracies)

        mean_train_accuracies = np.mean(train_accuracies, axis=0)
        std_train_accuracies = np.std(train_accuracies, axis=0)

    if data.get('train_loss') is not None:
        for i in range(experiment_count):
            train_loss = data['train_loss'][i]
            train_losses.append(train_loss)
        train_losses = np.array(train_losses)

        mean_train_losses = np.mean(train_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)

    # Extract the mean time
    if isinstance(data['time'], list):
        # We have a list of values
        for i in range(experiment_count):
            times.append(data['time'][i])
    else:
        # We have a single value
        times.append(data['time'])


    print(data['input_args'])

    # For the alpha models
    if 'alpha_original' in json_file:
        rgba_line_color = "rgba(0, 123, 255, 1)"
        rgba_fill_color = "rgba(0, 123, 255, 0.2)"
    elif 'alpha_lambeq' in json_file:
        rgba_line_color = "rgba(58, 250, 200, 1)"
        rgba_fill_color = "rgba(58, 250, 200, 0.2)"
    elif 'alpha_pennylane' in json_file:
        rgba_line_color = "rgba(140, 82, 255, 1)"
        rgba_fill_color = "rgba(140, 82, 255, 0.2)"

    dash = 'solid'


    if 'counterpart' in data['input_args']:
        # We have a counterpart result
        if data['input_args']['counterpart'] == 1:
            rgba_line_color = "rgba(0, 88, 179, 1)"
            rgba_fill_color = "rgba(0, 88, 179, 0.2)"
        elif data['input_args']['counterpart'] == 2:
            rgba_line_color = "rgba(40, 196, 161, 1)"
            rgba_fill_color = "rgba(40, 196, 161, 0.2)"
        elif data['input_args']['counterpart'] == 3:
            rgba_line_color = "rgba(106, 56, 213, 1)"
            rgba_fill_color = "rgba(106, 56, 213, 0.2)"
        dash = "dash" #To have dashlines

       


    print(len(mean_train_losses))
    print(len(list(range(1, iterations + 1)) ))
    # We have 3 case scenarios:
    # 1. We have all the data
    # 2. We have only the accuracy data
    # 3. We have only the loss data

    if (data.get('val_acc') is None or data.get('train_acc') is None) or (data.get('val_loss') is None or data.get('train_loss') is None):
        print("Accuracy or Loss missing from JSON file.")
         # Initialize figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.5, 0.5],
            row_heights=[0.5, 0.5],
            specs=[[{"type": "scatter"}, None],
                [{"type": "scatter"}, None]])
    else:
        # We have all the data
        # Initialize figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.5, 0.5],
            row_heights=[0.5, 0.5],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]])


    x_values = list(range(1, iterations + 1))  # Convert range to a list

    col_index = 1

    if data.get('train_loss') is not None and data.get('val_loss') is not None:
        # Add scatter training loss trace
        interactive_plotly_plot(x_values, mean_train_losses, std_train_losses, 1, col_index, "Training Loss", fig, rgba_line_color, rgba_fill_color, dash)
        
        # Add scatter validation loss trace
        interactive_plotly_plot(x_values, mean_val_losses, std_val_losses, 2, col_index, "Validation Loss", fig, rgba_line_color, rgba_fill_color, dash)
        col_index += 1


    if data.get('train_acc') is not None and data.get('val_acc') is not None:
        # Add scatter training accuracy trace
        interactive_plotly_plot(x_values, mean_train_accuracies, std_train_accuracies, 1, col_index, "Training Accuracy", fig, rgba_line_color, rgba_fill_color, dash)

        # Add scatter validation accuracy trace
        interactive_plotly_plot(x_values, mean_val_accuracies, std_val_accuracies, 2, col_index, "Validation Accuracy", fig, rgba_line_color, rgba_fill_color, dash)

        
    
    if data.get('val_acc') is None or data.get('train_acc') is None:
        # 2. We have only the accuracy data
        fig.update_layout(
                        legend=dict(x=0, y=1),
                        xaxis=dict(range=[0, 3]),
                        hovermode="x",
                        yaxis1_title = "Training Loss",
                        yaxis2_title = "Validation Loss",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
        )

    elif data.get('val_loss') is None or data.get('train_loss') is None:
        # 3. We have only the loss data
        fig.update_layout(
                        legend=dict(x=0, y=1),
                        xaxis=dict(range=[0, 3]),
                        hovermode="x",
                        yaxis1_title = "Training Accuracy",
                        yaxis2_title = "Validation Accuracy",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
        )
    else:
        # 1. We have all the data
        fig.update_layout(
                        legend=dict(x=0, y=1),
                        xaxis=dict(range=[0, 3]),
                        hovermode="x",
                        yaxis1_title = "Training Loss",
                        yaxis2_title = "Training Accuracy",
                        yaxis3_title = "Validation Loss",
                        yaxis4_title = "Validation Accuracy",
                        xaxis1_title = "Iterations",
                        xaxis2_title = "Iterations",
                        xaxis3_title = "Iterations",
                        xaxis4_title = "Iterations",
        )

        grid_color = 'rgba(128, 128, 128, 0.2)'  # Grey with 20% opacity


        fig.update_layout(yaxis1 = dict(range=[0, 1], gridcolor=grid_color, tickfont=dict(size=18)))    # loss
        fig.update_layout(yaxis2 = dict(range=[0.4, 1], gridcolor=grid_color, tickfont=dict(size=18)))    # acc
        fig.update_layout(yaxis3 = dict(range=[0, 1], gridcolor=grid_color, tickfont=dict(size=18))) # loss
        fig.update_layout(yaxis4 = dict(range=[0.4, 1], gridcolor=grid_color, tickfont=dict(size=18))) # acc

        fig.update_layout(xaxis1 = dict(gridcolor=grid_color, tickfont=dict(size=18)))
        fig.update_layout(xaxis2 = dict(gridcolor=grid_color, tickfont=dict(size=18)))
        fig.update_layout(xaxis3 = dict(gridcolor=grid_color, tickfont=dict(size=18)))
        fig.update_layout(xaxis4 = dict(gridcolor=grid_color, tickfont=dict(size=18)))

        fig.update_layout(
            plot_bgcolor='white',  # Set the background color to white
            paper_bgcolor='white',  # Set the paper color to white (for the entire plot area)
        )

        fig.update_layout(
        width=1100,  # Set your desired width in pixels
        height=1100,  # Set your desired height in pixels
        font=dict(
            size=18  # Set your desired font size for text elements
        )
        )

    
    fig['layout']['xaxis'].update(autorange = True)

    fig.show()


def interactive_plotly_plot(x_values, y_values, y_std, row_index, col_index, name, fig, rgba_line_color, rgba_fill_color, dash):

    fig.add_trace(
        go.Scatter(x=x_values, y=y_values, mode='lines', showlegend=False, name=name, line=dict(width= 3, dash = dash),
                    line_color=rgba_line_color),
        row=row_index, col=col_index
    )
    

    fig.add_trace(
        go.Scatter(x=x_values, y=y_values + y_std, mode='lines', line={'width': 0},
                   fillcolor=rgba_fill_color, line_color=rgba_line_color, showlegend=False, name='Upper Bound'),
        row=row_index, col=col_index
    )
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values - y_std, mode='lines', line={'width': 0}, fill='tonexty',
                   fillcolor=rgba_fill_color, line_color=rgba_line_color, showlegend=False, name='Lower Bound'),
        row=row_index, col=col_index
    )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot experiment results from a JSON file.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file with experiment results')

    args = parser.parse_args()

    plot_experiment_results(args.json_file)
