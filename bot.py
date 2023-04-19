import argparse
import numpy as np
from scipy.special import softmax
import cv2 as cv
import telebot
import io
from threading import Lock
import time

parser = argparse.ArgumentParser()
parser.add_argument('--token', help='Telegram bot token', required=True)
parser.add_argument('--max_frames', help='Limit maximum number of frames in video per request', default=300)
parser.add_argument('--delay', help='Limit time in seconds for interactions', type=int, default=0)
args = parser.parse_args()

bot = telebot.TeleBot(args.token)
mutex = Lock()


class HairSegmentation(object):
    def __init__(self):
        self.net = cv.dnn.readNet("hair_segmentation.tflite")


    def _mix_prev_mask(self, prev_mask, new_mask):
        combine_with_prev_ratio = 0.9
        eps = 1e-3
        uncertainty_alpha = 1.0 + (new_mask * np.log(new_mask + eps) + (1.0 - new_mask) * np.log(1.0 - new_mask + eps)) / np.log(2.0)
        uncertainty_alpha = np.clip(uncertainty_alpha, 0, 1)
        uncertainty_alpha *= 2.0 - uncertainty_alpha

        mixed_mask = new_mask * uncertainty_alpha + prev_mask * (1.0 - uncertainty_alpha)
        return mixed_mask * combine_with_prev_ratio + (1.0 - combine_with_prev_ratio) * new_mask


    def process_image(self, frame, color, num_runs=2):
        prev_mask = np.zeros((512, 512), dtype=np.float32)
        color = np.ones(frame.shape, dtype=np.uint8) * color

        # Prepare input
        blob = cv.dnn.blobFromImage(frame, 1.0 / 255, (512, 512), swapRB=True)
        blob = np.concatenate((blob, prev_mask.reshape(1, 1, 512, 512)), axis=1)

        for i in range(num_runs):
            # Copy previous frame mask to a new tensor
            blob[0, 3] = prev_mask

            # Run network
            self.net.setInput(blob)
            out = self.net.forward()

            out = softmax(out, axis=1)
            mask = out[0, 1]

            prev_mask = self._mix_prev_mask(prev_mask, mask)

        mask = cv.resize(prev_mask, (frame.shape[1], frame.shape[0]))
        lum = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) / 255
        mask *= lum

        mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        result = (mask * (color.astype(np.float32) - frame) + frame).astype(np.uint8)

        return result


    # def process_video(self, cap, out_cap):
    #     prev_mask = np.zeros((512, 512), dtype=np.float32)
    #     color = np.zeros((384, 384, 3), dtype=np.uint8)
    #     color[:, :, 0] = 255

    #     num_frames = 0
    #     while cap.isOpened() and num_frames < args.max_frames:
    #         has_frame, frame = cap.read()
    #         if not has_frame:
    #             break
    #         num_frames += 1

    #         # Prepare input
    #         blob = cv.dnn.blobFromImage(frame, 1.0 / 255, (512, 512), swapRB=True)

    #         # Copy previous frame mask to a new tensor
    #         blob = np.concatenate((blob, prev_mask.reshape(1, 1, 512, 512)), axis=1)

    #         # Run network
    #         self.net.setInput(blob)
    #         out = self.net.forward()

    #         out = softmax(out, axis=1)
    #         mask = out[0, 1]

    #         prev_mask = self._mix_prev_mask(prev_mask, mask)

    #         mask = cv.resize(prev_mask, (frame.shape[1], frame.shape[0]))
    #         lum = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) / 255
    #         mask *= lum

    #         mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    #         frame = (mask * (color.astype(np.float32) - frame) + frame).astype(np.uint8)

    #         out_cap.write(frame)


model = HairSegmentation()
colors = {}
timestamps = {}

# def process_video(inp_file_name, out_file_name):
#     cap = cv.VideoCapture(inp_file_name)
#     out = cv.VideoWriter(out_file_name, cv.VideoWriter_fourcc(*'mp4v'), 30, (384, 384))
#     model.process(cap, out)


# def send_video(message, video_path):
#     with open(video_path, 'rb') as f:
#         bot.send_video(chat_id=message.chat.id, video=f)


def get_image(message):
    fileID = message.photo[-1].file_id
    file = bot.get_file(fileID)
    data = bot.download_file(file.file_path)
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv.imdecode(buf, cv.IMREAD_COLOR)

def send_image(message, img):
    _, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 90])
    outputbuf = io.BytesIO(buf)
    bot.send_photo(message.chat.id, outputbuf)


# @bot.message_handler(content_types=['video_note'])
# def process_image(message):
#     f = bot.get_file(message.video_note.file_id)
#     data = bot.download_file(f.file_path)

#     inp_file_name = 'tmp_in.mp4'
#     out_file_name = 'tmp_out.mp4'
#     with open(inp_file_name, 'wb') as f:
#         f.write(data)

#     process_video(inp_file_name, out_file_name)
#     send_video(message, out_file_name)


@bot.message_handler(content_types=['photo'])
def process_image(message):
    chat_id = message.chat.id

    now = time.time()
    timestamp = timestamps.get(chat_id, 0)
    if now - timestamp < args.delay:
        bot.send_message(chat_id, f"Try again after {args.delay - int(now - timestamp)} seconds")
        return

    timestamps[chat_id] = now

    mutex.acquire()

    color = colors.get(chat_id, [255, 0, 0])

    img = get_image(message)
    stylized = model.process_image(img, color)
    send_image(message, stylized)
    mutex.release()


@bot.message_handler(commands=['color'])
def process_image(message):
    color_hex = message.text.split(' ')[1]
    if not color_hex.startswith('#') or len(color_hex) != 7:
        return
    color_hex = color_hex[1:]

    color_bgr = [int(color_hex[i:i + 2], 16) for i in (4, 2, 0)]
    colors[message.chat.id] = color_bgr


bot.polling()
