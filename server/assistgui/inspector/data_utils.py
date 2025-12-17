import os
from io import BytesIO
import re
import cv2
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, ConvertImageDtype
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import pandas as pd
import json
import random
import requests
import decord
from decord import VideoReader
from decord import cpu
import clip
from urllib.parse import urlparse
from PIL import Image, ImageDraw


def get_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps


def load_video_decord(video_path, desired_fps=0.5, minimum_frames=32, image_size=224):
    # Create a VideoReader object to access the video
    decord.bridge.set_bridge('torch')
    video_reader = VideoReader(video_path, ctx=cpu(0)) # width=image_size, height=image_size,

    # Get the video's FPS (frames per second)
    original_fps = video_reader.get_avg_fps()
    total_frames = len(video_reader)

    # Calculate the step size based on the desired FPS
    step_size = int(original_fps / desired_fps)

    # Determine the number of frames that would be sampled with the current step size
    sampled_frames = total_frames // step_size

    # If the number of sampled frames is less than minimum_frames, adjust the desired FPS
    if sampled_frames < minimum_frames:
        desired_fps = minimum_frames / total_frames * original_fps
        step_size = int(original_fps / desired_fps)

    # If the video has less than minimum_frames, return all frames
    frames = []
    if total_frames <= minimum_frames:
        for i in range(total_frames):
            frame = video_reader[i]
            frames.append(frame)
    else:
        # Iterate through the video frames and extract them at the desired FPS
        for i in range(0, total_frames, step_size):
            frame = video_reader[i]
            frames.append(frame)

    frames = torch.stack(frames)
    return frames


def sample_frames(num_frames, vlen, sample='rand', fix_start=None, begin_end_frame=None):
    acc_samples = min(num_frames, vlen)
    begin, end = begin_end_frame if begin_end_frame else (0, vlen)
    intervals = np.linspace(start=begin, stop=end, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def load_image_pil(image_source):
    if get_video_source_type(image_source) == 'url':
        response = requests.get(image_source['url'])
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    return img


def get_video_source_type(input_string):
    try:
        result = urlparse(input_string)
        if all([result.scheme, result.netloc]):
            return 'url'
        else:
            raise ValueError
    except ValueError:
        if os.path.exists(input_string):
            return 'path'
        else:
            return None


def check_video_or_image(path):
    if path.endswith('.mp4'):
        return 'video'
    else:
        return 'image'


def load_video_decord_deprecated(video_source, num_frames, mode='uniform', fix_start=None, image_size=(512, 512),
                      begin_end_time=None):
    # print("video path: {}".format(video_path))
    source_type = get_video_source_type(video_source)
    if source_type == 'url':
        video_reader = decord.VideoReader(video_source['url'], ctx=cpu(0))
    elif source_type == 'path':
        if image_size:
            width, height = image_size
            video_reader = decord.VideoReader(video_source, width=width, height=height, num_threads=1, ctx=cpu(0))
        else:
            video_reader = decord.VideoReader(video_source, num_threads=1, ctx=cpu(0))
    else:
        raise Exception("invalid video source: {}".format(video_source))

    decord.bridge.set_bridge('torch')
    vlen = len(video_reader)

    # segment video with given begin_end_time
    if begin_end_time:
        fps = get_fps(video_source) if source_type == 'path' else 30  # TODO: get fps from url
        begin_end_frame = [min(t * fps, vlen) for t in begin_end_time]
    else:
        begin_end_frame = None

    frame_idxs = sample_frames(num_frames, vlen, sample=mode,
                               fix_start=fix_start, begin_end_frame=begin_end_frame)  # sample: "random" or "uniform"
    frames = video_reader.get_batch(frame_idxs).byte()  # frames rgb
    #     frames = frames.permute(0, 3, 1, 2)  # [t, channel, , ]
    return frames, frame_idxs, vlen


def load_video_moviepy(video, sample_rate=0.5, num_frames=32):
    # Sample N frames evenly spaced throughout the video
    num_frames = num_frames if video.duration * sample_rate < num_frames else int(video.duration * sample_rate)
    frames = sample_frames_with_timestamps(video, num_frames)

    # Convert frames (numpy arrays) to PyTorch tensors
    torch_frames = [torch.from_numpy(np.transpose(frame, (2, 0, 1))).float() / 255.0 for frame in frames]
    return torch.stack(torch_frames, dim=0)


def sample_frames_with_timestamps(video, num_frames):
    timestamps = np.linspace(0, video.duration, num=num_frames, endpoint=False)
    frames = [video.get_frame(t) for t in timestamps]
    return frames


def draw_bbox(image, bbox):
    draw = ImageDraw.Draw(image)

    # Draw the box on the image with a specified color and width
    box_color = (255, 0, 0)  # Red, in RGB format
    box_width = 2
    draw.rectangle(bbox, outline=box_color, width=box_width)

    # # Save the image with the box drawn on it
    # output_image_path = "path/to/your/output_image.jpg"
    # image.save(output_image_path)
    return image


def is_url_or_filepath(input_string):
    # Check if input_string is a URL
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    if url_pattern.match(input_string):
        return 'URL'

    # Check if input_string is a file path
    file_path = os.path.abspath(input_string)
    if os.path.exists(file_path):
        return 'File path'

    return 'Invalid'
