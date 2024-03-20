import numpy as np
import torch

def frame_encoder(frame):
    transforms = torch.nn.Sequential(
        torch.conv2d()
    )
    frame_mat = frame
    return frame_mat

def events_encoder(events):
    events_mat = np.zeros((width, height))
    return events_mat

def build_condition_mat(frame, events):
    np.concatenate([frame_encoder(frame), events_encoder(events)], axis=0)