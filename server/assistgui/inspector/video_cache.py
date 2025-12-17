import os.path
import json
from moviepy.editor import VideoFileClip
from PIL import Image

import datetime
from assistgui.inspector.data_utils import load_video_decord
from assistgui.model_zoo.shared_model import SharedModel

from assistgui.model.utils import run_llm, get_video_title


class VideoMemoryCache:
    def __init__(self, caption_model_name="qwen-vl-chat", **kwargs):
        self.user_comment = None
        self.type = "video"
        self.video_path = None
        self.video = None  # moviepy video object
        self.duration = None
        self.llm = "gpt-3.5-turbo"

        self.tensor = None
        self.audio = None
        self.audio_path = None
        self.subtitle = None
        self.subtitle_state = None
        self.subtitle_path = None
        self.narration = None

        # keyframe
        self.pil = None
        self.image_path = None
        self.ocr = None
        self.summary = None
        self.boxes = None

        self.cache_dir = "."
        self.language = 'en-US'

        self.if_save_audio = True

        self.update_attributes(kwargs)

        self.asr_model = SharedModel("whisper").model_dict["model"]
        self.caption_model_name = caption_model_name
        # self.caption_model = SharedModel(caption_model_name)

        if not self.video_path:
            raise Exception("A video path should be provided.")

        if not self.video:
            self.video = VideoFileClip(self.video_path)

        if self.video.duration < 1:
            print("Automatically Loading video tensor when video length less than 1 minutes...")
            self.tensor = load_video_decord(self.video_path, 0.5, 32)

        self.pil, self.image_path = self.get_keyframe(self.video, pre_fix=self.video_path.split("/")[-1].split(".")[0])
        self.duration = self.video.duration

        self.audio = self.video.audio
        if self.if_save_audio:
            self.save_audio()
            if not self.subtitle:
                print("transcribing audio...")
                self.get_subtitle()

        self.get_summary()

    def get_keyframe(self, video, keyframe_selector="middle", pre_fix=""):
        # Get the timestamp of the keyframe
        # TODO: add more sophisticated keyframe selection methods
        if keyframe_selector == "middle":
            video_duration = video.duration
            frame_time = video_duration / 2
        else:
            raise NotImplementedError

        # Get the frame
        frame = video.get_frame(frame_time)

        # Turn to PIL image
        frame_pil = Image.fromarray(frame.astype('uint8'), 'RGB')
        frame_path = f"{self.cache_dir}/{pre_fix}_keyframe.jpg"
        frame_pil.save(frame_path, "JPEG")
        return frame_pil, frame_path

    def save_audio(self):
        self.audio_path = self.video_path.replace(".mp4", ".mp3")
        if not os.path.exists(self.audio_path):
            self.audio.write_audiofile(self.audio_path)

    def update_attributes(self, attributes_dict):
        for key, value in attributes_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute '{key}' not found in VideoMemoryCache class.")

    @staticmethod
    def generate_cache_name(prefix, suffix=".mp4"):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = prefix + current_time + suffix
        return file_name

    def get_description(self):
        return f"a {self.duration:.2f} seconds {self.type}, {self.subtitle_state}, {self.user_comment}, {self.summary}\n"

    def get_summary(self):
        self.summary = None
        return self.summary

    def get_subtitle(self, raw_data=False):
        self.subtitle_path = self.video_path.replace(".mp4", ".json")
        if not self.subtitle:
            if os.path.exists(self.subtitle_path):
                self.subtitle = json.load(open(self.subtitle_path, "r"))
            else:
                self.subtitle = self.subtitle_to_text(self.asr_model.transcribe(self.audio_path)['segments'])
                with open(self.subtitle_path, "w") as f:
                    json.dump(self.subtitle, f)

        return self.subtitle

    @staticmethod
    def subtitle_to_text(input_data):
        output = ""
        for entry in input_data:
            output += f"{entry['start']:.2f} - {entry['end']:.2f}\n{entry['text'].strip()}\n"
        return output

    def check_subtitle(self):
        text = self.subtitle[:min(100, len(self.subtitle))]
        prompt = f"The following sentences are subtitles parsed by an ASR module. Help me check if the obtained subtitle is meaningful text. " \
                 f"You output should be one of [dense subtitle, sparse subtitle, no meaningful subtitle]" \
                 f"\nSubtitle:{text}\nOutput:"
        response = run_llm(prompt=prompt, llm="gpt-3.5-turbo")
        self.subtitle_state = response

