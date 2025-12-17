import io
import os
import imageio
from PIL import Image
from google.cloud import vision
from assistgui.model.utils import run_llm
from assistgui.model.base_module import BaseModule


class BaseNarrator(BaseModule):
    name = "video_narrate"
    description = (
        '''
This tool can narrate the video by watching frame by frame.
Invoke command: video_narrate(query, visual[i])
:param query -> str, a question indicate what information to narrate.
''')

    def __init__(self):
        super(BaseNarrator, self).__init__()
        self.client = vision.ImageAnnotatorClient()

    def _run(self, query, input_image, input_video, history) -> dict:
        """Use the tool."""
        return {}

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("")

    @staticmethod
    def segment_video(input_video, clip_length):
        '''
        :param input_video:
        :param clip_length: seconds per video clip
        :return:
        '''
        # Load the video
        video_name = input_video.image_path.split('/')[-1].split('.')[0]
        video = input_video.video
        video_duration = video.duration

        # Calculate the number of clips
        num_clips = int(video_duration // clip_length)

        # Create the output folder if it doesn't exist
        if not os.path.exists(input_video.cache_dir):
            os.makedirs(input_video.cache_dir)

        # Loop through each clip
        frame_paths = []
        for i in range(num_clips):
            start_time = i * clip_length
            end_time = start_time + clip_length

            # Extract one frame (the middle frame)
            frame_time = (start_time + end_time) / 2
            frame = video.get_frame(frame_time)

            # Save the frame as a JPG image
            output_path = os.path.join(input_video.cache_dir, f'{video_name}_seg-{i}.jpg')
            frame_paths.append(output_path)
            imageio.imwrite(output_path, frame)

        frame_pils = [Image.open(frame_path) for frame_path in frame_paths]
        return frame_paths, frame_pils

    def merge_narration_and_subtitle(self, narration, subtitle, query):
        """Merge the narration and subtitle."""
        print("Merging subtitle and narration...")
        # subtitle = self.subtitle_to_text(subtitle)
        prompt = f"Here is a subtitle and narration of a video. You should merge them into one narration related to the query: {query}.\n" \
                    "Subtitle:\n" + subtitle + "\nNarration:\n" + narration + "\nMerged narration with proper time stamps:"
        response = run_llm(prompt, llm=self.llm, max_tokens=500, temperature=0, stop=['\n\n'])
        return response

    @staticmethod
    def subtitle_to_text(input_data):
        output = ""
        for entry in input_data:
            output += f"{entry['start']:.2f} - {entry['end']:.2f}\n{entry['text'].strip()}\n"
        return output

    def get_ocr(self, image_path):
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Perform OCR using the API
        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        # Extract the full text from the response
        if texts:
            full_text = texts[0].description
            return f" OCR on the frame ({full_text})"
        else:
            return ""

    def process_query(self, query):
        return query

