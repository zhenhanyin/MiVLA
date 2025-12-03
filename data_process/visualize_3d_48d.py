'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Script for visualizing skeletal data in 3D. 
Adapted from https://github.com/RogerQi/human-policy/blob/main/data/plot_keypoints.py.
'''

import plotly.graph_objects as go
import numpy as np
import h5py
from utils.skeleton_tfs import *

# by default, plot arms + neck + hands. change as desired
QUERY_TFS = LEFT_ARM + LEFT_FINGERS + RIGHT_ARM + RIGHT_FINGERS + NECK

def main(input_file):
    root = h5py.File(input_file)
    
    # Prepare data for animation
    frames = []
    num_frames = root['/transforms/camera'].shape[0]

    # extract position data 
    for i in range(num_frames):
        tfs = np.zeros([len(QUERY_TFS), 3]) 
        for j, tf_name in enumerate(QUERY_TFS):
            tfs[j] = root['/transforms/' + tf_name][i][:3, 3]
        frames.append(tfs.copy())
    
    # Create figure
    fig = go.Figure()
    
    def add_hand_keypoints(fingers, color):
        # Add finger keypoints
        fig.add_trace(go.Scatter3d(
            x=fingers[:, 0], y=fingers[:, 1], z=fingers[:, 2],
            mode='markers',
            name=f"joints",
            marker=dict(size=4, color=color, opacity=0.7)
        ))
    

    # Add initial hand keypoints
    add_hand_keypoints(frames[0], 
                      'green')
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title="3D Visualization",
        showlegend=True,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'type': 'buttons',
            'direction': 'left',
            'showactive': True
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'steps': [{
                'args': [[str(i)], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': str(i),
                'method': 'animate'
            } for i in range(len(frames))]
        }]
    )
    
    # Create animation frames
    fig_frames = []
    for i, frame in enumerate(frames):
        frame_traces = []
        
        # Add finger keypoints to each frame
        frame_traces.append(go.Scatter3d(
            x=frame[:, 0],
            y=frame[:, 1],
            z=frame[:, 2],
            mode='markers',
            marker=dict(size=4, color='green', opacity=0.7)
        ))
        
        fig_frames.append(go.Frame(data=frame_traces, name=str(i)))
    
    fig.frames = fig_frames
    fig.show()
    return frames

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot 3D data from a HDF5 file')
    parser.add_argument('--file', '-f', type=str, 
                        help='Path to some .hdf5 file, e.g., test/add_remove_lid/0.hdf5')
    args = parser.parse_args()

    main(args.file)