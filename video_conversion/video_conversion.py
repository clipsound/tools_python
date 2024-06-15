import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def find_videos_in_folder(folder_path, extensions=['.mp4', '.avi', '.mov', '.mkv', '.flv']):
    """
    Find all video files in the specified folder with the given extensions.

    Parameters:
    folder_path (str): Path to the folder where to search for video files.
    extensions (list): List of file extensions to search for.

    Returns:
    list: List of file paths to the found video files.
    """
    video_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))
    return video_files

def resize_frame(frame, width=None, height=None):
    """
    Resize the frame to the specified width and height.
    """
    if width and height:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return frame

def crop_frame(frame, x1, y1, x2, y2):
    """
    Crop the frame to the specified coordinates.
    """
    return frame[y1:y2, x1:x2]

def add_watermark_to_frame(frame, watermark_text, pos=('center', 'center'), fontsize=24, color=(255, 255, 255)):
    """
    Add a watermark text to the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = fontsize / 30  # Adjust the scale to match the fontsize
    thickness = 1  # You can adjust the thickness if needed

    text_size = cv2.getTextSize(watermark_text, font, font_scale, thickness)[0]
    text_x, text_y = 0, 0

    if pos[0] == 'center':
        text_x = (frame.shape[1] - text_size[0]) // 2
    elif pos[0] == 'left':
        text_x = 0
    elif pos[0] == 'right':
        text_x = frame.shape[1] - text_size[0]

    if pos[1] == 'center':
        text_y = (frame.shape[0] + text_size[1]) // 2
    elif pos[1] == 'top':
        text_y = text_size[1]
    elif pos[1] == 'bottom':
        text_y = frame.shape[0] - text_size[1]

    cv2.putText(frame, watermark_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def process_video(input_file, output_format, resize_opts=None, crop_opts=None, watermark_opts=None):
    """
    Process and convert the video to a specified format and apply optional transformations.

    Parameters:
    input_file (str): Path to the input video file.
    output_format (str): Desired output format (e.g., 'mp4', 'avi', 'gif').
    resize_opts (dict): Options for resizing the video, e.g., {'width': 640, 'height': 480}.
    crop_opts (dict): Options for cropping the video, e.g., {'x1': 50, 'y1': 50, 'x2': 500, 'y2': 300}.
    watermark_opts (dict): Options for adding a watermark, e.g., {'text': 'Sample', 'pos': 'bottom-right'}.
    """
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error opening video file {input_file}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID') if output_format != 'gif' else None
    output_file = input_file.rsplit('.', 1)[0] + '.' + output_format
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if output_format != 'gif':
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if resize_opts:
            frame = resize_frame(frame, **resize_opts)
        if crop_opts:
            frame = crop_frame(frame, **crop_opts)
        if watermark_opts:
            frame = add_watermark_to_frame(frame, **watermark_opts)

        if output_format == 'gif':
            frames.append(frame)
        else:
            out.write(frame)

    cap.release()
    if output_format != 'gif':
        out.release()
    else:
        gif_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        pil_frames = [Image.fromarray(frame) for frame in gif_frames]
        pil_frames[0].save(output_file, save_all=True, append_images=pil_frames[1:], loop=0, duration=int(1000 / fps))

    print(f"Video processed and saved as {output_file}")

# Example usage
folder_path = '/Users/pasquale/Desktop/last_apai/AbC_products_output'
video_files = find_videos_in_folder(folder_path)
print("Found video files:", video_files)

resize_options = {'width': 640, 'height': 480}
crop_options = {'x1': 50, 'y1': 50, 'x2': 500, 'y2': 300}
watermark_options = {'watermark_text': 'Sample Watermark', 'pos': ('center', 'bottom'), 'fontsize': 30, 'color': (255, 255, 255)}

for video_file in video_files:
    process_video(video_file, 'gif', crop_opts=crop_options, resize_opts=resize_options, watermark_opts=watermark_options)
