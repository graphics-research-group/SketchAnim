import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio

from PIL import Image
import sys
sys.path.append('/folder/path/SketchAnim/')
from co_tracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
from co_tracker.cotracker.predictor import CoTrackerPredictor

'''
Track points across frames using Co-Tracker.
Input:
    1) `video_frames` - Numpy array of shape (num_frames, height, width, 3) containing the video frames
    2) `input_frames` - Numpy array of shape (N,2) containing N 2D coordinates
    3) `edges` (optional) - Numpy array of shape (M,2) with M edges. This is only used when visualising a skeleton or triangulation.
    4) `device` (default='cpu')
    5) `cotracker_checkpoint` - Filepath to the weights to be used by the CoTracker model.
    6) `output_visualise_filepath` (optional) - Filepath to save  the gif visualising the results.
'''
def PointTracking(video_frames,
                  input_points,
                  edges=[],
                  device='cpu',
                  cotracker_checkpoint='./co_tracker/checkpoints/cotracker_stride_8_wind_16.pth', # use shape-of-motion for better tracking
                  output_visualise_filepath=None,
                  output_vertices_filepath=None,
                  verbose=False):
    
    video = torch.from_numpy(video_frames).permute(0, 3, 1, 2)[None].float()
    model = CoTrackerPredictor(checkpoint=cotracker_checkpoint)
    model = model.to(device)
    video = video.to(device)

    queries = []
    for vertex in input_points:
        queries.append([0.0, vertex[0], vertex[1]])
    queries = torch.tensor(queries)

    queries = queries.reshape(1, queries.shape[0], queries.shape[1])
    queries = queries.to(device)

    pred_tracks, pred_visibility = model(
        video,
        queries = queries,
        grid_query_frame=0,
    )
    if verbose:
        print("computed")
    
    tracked_vertices = np.array(pred_tracks.squeeze().to('cpu'))

    if output_visualise_filepath is not None:
        plot_skeleton_across_frames(video, tracked_vertices, edges, output_visualise_filepath) 
    #     video = video.squeeze().to('cpu')
    #     video_frames = np.array(video.permute(0, 2, 3, 1))

    #     skeleton_frames = []

    #     for i in range(len(video_frames)):
    #         frame = video_frames[i]
    #         # print(frame.shape)
    #         skeleton_coordinates = tracked_vertices[i]
    #         skeleton_visualise = np.copy(frame)
    #         for bone in edges:
    #             point0 = (int(skeleton_coordinates[bone[0]][0]), int(skeleton_coordinates[bone[0]][1]))
    #             point1 = (int(skeleton_coordinates[bone[1]][0]), int(skeleton_coordinates[bone[1]][1]))
    #             cv2.line(skeleton_visualise, point0, point1, (0, 255, 0), 2)
    #         for point in skeleton_coordinates:
    #             cv2.circle(skeleton_visualise, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
    #         skeleton_frames.append(np.array(skeleton_visualise).astype(np.uint8))
    #     imageio.mimsave(os.path.join(output_visualise_filepath, 'tracking.gif'), skeleton_frames, loop=0)

    if output_vertices_filepath is not None:
        vertices_across_frames = pred_tracks.squeeze().to('cpu')
        save_tracked_vertices(vertices_across_frames, output_vertices_filepath)
        # vertices_across_frames = vertices_across_frames.reshape(
        #     (vertices_across_frames.shape[0], vertices_across_frames.shape[1]*vertices_across_frames.shape[2]))
        # np.savetxt(os.path.join(output_vertices_filepath, 'skeleton_vertices_across_frames.txt'), vertices_across_frames)

    return tracked_vertices

def save_tracked_vertices(vertices_across_frames, output_vertices_filepath):
    # vertices_across_frames = pred_tracks.squeeze().to('cpu')
    vertices_across_frames = vertices_across_frames.reshape(
        (vertices_across_frames.shape[0], vertices_across_frames.shape[1]*vertices_across_frames.shape[2]))
    np.savetxt(os.path.join(output_vertices_filepath, 'skeleton_vertices_across_frames.txt'), vertices_across_frames)

def plot_skeleton_across_frames(video, tracked_vertices, edges, output_visualise_filepath):
    video = video.squeeze().to('cpu')
    video_frames = np.array(video.permute(0, 2, 3, 1))

    skeleton_frames = []

    for i in range(len(video_frames)):
        frame = video_frames[i]
        # print(frame.shape)
        skeleton_coordinates = tracked_vertices[i]
        skeleton_visualise = np.copy(frame)
        for bone in edges:
            point0 = (int(skeleton_coordinates[bone[0]][0]), int(skeleton_coordinates[bone[0]][1]))
            point1 = (int(skeleton_coordinates[bone[1]][0]), int(skeleton_coordinates[bone[1]][1]))
            cv2.line(skeleton_visualise, point0, point1, (0, 255, 0), 2)
        for point in skeleton_coordinates:
            cv2.circle(skeleton_visualise, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
        skeleton_frames.append(np.array(skeleton_visualise).astype(np.uint8))
    imageio.mimsave(os.path.join(output_visualise_filepath, 'tracking.gif'), skeleton_frames, loop=0)