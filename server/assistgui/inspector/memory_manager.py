import os.path
import datetime

import os
import magic
from assistgui.inspector import ImageMemoryCache, VideoMemoryCache


class MemoryManager:
    def __init__(self, save_previous=False, query="QA", image_summarizer="qwen-vl-chat"):
        self.visual = []
        self.cache = []
        self.save_previous = save_previous
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.cache_dir = f"cache/{query}-{current_time}/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def append_visual_input(self, visual_input_path, **kwargs):
        cache_path = os.path.join(self.cache_dir, f"visual-{len(self.visual)}")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        if self.file_type(visual_input_path) == 'video':
            self.append_video(cache_dir=cache_path, video_path=visual_input_path, **kwargs)
        elif self.file_type(visual_input_path) == 'image':
            self.append_image(cache_dir=cache_path, image_path=visual_input_path, **kwargs)
        else:
            raise Exception(f"File type {self.file_type(visual_input_path)} not supported.")

    def append_image(self, **kwargs):
        """input should be a PIL image"""
        image_dict = {}
        image_dict.update(kwargs)
        image = ImageMemoryCache(**image_dict)
        self.visual.append(image)

    def append_video(self, **kwargs):
        "input should be a video path or a moviepy video object"
        video_dict = {}
        video_dict.update(kwargs)
        video = VideoMemoryCache(**video_dict)
        self.visual.append(video)

    def update_visual_input(self, index, **kwargs):
        self.visual[index].update_attributes(kwargs)

    def get_description(self, index=None):
        if index is None:
            indices = range(len(self.visual))
        elif index < 0:
            indices = [len(self.visual) + index]
        else:
            indices = [index]

        summary = "The visual inputs are as follows:\n" if index is None else ""
        for ix in indices:
            summary += f"visual-{ix}: {self.visual[ix].get_description()}\n"
        return summary

    def get_visual_input(self, index):
        '''
        Get the visual input from the inspector
        :param index:
        :return: two dicts, one for image, one for video
        '''
        if not index:
            return None, None

        if type(index) is int:
            index = [index]
        elif type(index) is list:
            pass
        else:
            raise Exception("Index should be an integer or a list of integers.")

        image, video = [], []
        for ix in index:
            item = self.visual[ix]
            if item.type == 'image':
                image.append(item)
            elif item.type == 'video':
                video.append(item)

        return image, video

    def reset(self):
        if self.save_previous:
            self.cache.append(self.visual)
        self.visual = []

    @staticmethod
    def file_type(file_path):
        mime = magic.Magic(mime=True)
        file_mime = mime.from_file(file_path)
        file_type = file_mime.split('/')[0]
        return file_type

