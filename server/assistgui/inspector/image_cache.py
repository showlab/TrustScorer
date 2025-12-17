import os
import datetime
from PIL import Image
from assistgui.model_zoo.shared_model import SharedModel


class ImageMemoryCache:
    def __init__(self, caption_model_name='qwen-vl-chat',  **kwargs):
        self.user_comment = None
        self.type = "image"
        self.image_path = None
        self.pil = None
        self.tensor = None
        self.ocr = None
        self.summary = None
        self.cache_dir = ""
        self.boxes = None

        self.update_attributes(kwargs)

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if not self.image_path:
            self.image_path = self.generate_cache_name(prefix=f"{self.cache_dir}/{self.user_comment}", suffix=".jpg")
            if self.pil is not None:
                self.pil.save(self.image_path, "JPEG")
            else:
                raise Exception("An image path or a PIL object should be provided.")

        if not self.pil:
            self.pil = Image.open(self.image_path)

        self.caption_model_name = caption_model_name
        # self.caption_model = SharedModel(caption_model_name)
        self.get_summary()

    def update_attributes(self, attributes_dict):
        for key, value in attributes_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute '{key}' not found in ImageMemoryCache class.")

    @staticmethod
    def generate_cache_name(prefix, suffix=".mp4"):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = prefix + current_time + suffix
        return file_name

    def get_summary(self):
        self.summary = "None"
        return self.summary

    def get_description(self):
        return f"an {self.type}, {self.user_comment}, main content is {self.summary}\n"

