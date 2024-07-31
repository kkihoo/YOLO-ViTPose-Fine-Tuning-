import os
import cv2
import easy_ViTPose
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
from easy_ViTPose.easy_ViTPose import VitInference


MODEL_SIZE = "l"  # @param ['s', 'b', 'l', 'h']
YOLO_SIZE = "n"  # @param ['s', 'n']
DATASET = "multi-coco"  # @param ['multi-coco','coco_25', 'coco', 'wholebody', 'mpii', 'aic', 'ap10k', 'apt36k']
ext = ".pth"
ext_yolo = ".pt"

MODEL_TYPE = "torch"
YOLO_TYPE = "torch"
REPO_ID = "JunkyByte/easy_ViTPose"
FILENAME = (
    os.path.join(MODEL_TYPE, f"{DATASET}/vitpose-" + MODEL_SIZE + f"-{DATASET}") + ext
)
FILENAME_YOLO = "yolov8/yolov8" + YOLO_SIZE + ext_yolo

# model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
model_path = "/Users/ki/.cache/huggingface/hub/models--JunkyByte--easy_ViTPose/snapshots/2757e82adcccda02f9f7fef66e5a115b7be439fe/torch/multi-coco/vitpose-l-multi-coco.pth"
yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)

model = VitInference(
    model_path, yolo_path, model_name="l", yolo_size=320, is_video=False, device="mps"
)


def process_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_keypoints = model.inference(frame)
        
        result_frame = model.draw(show_yolo=True)

        out.write(result_frame)

        cv2.imshow("Processing", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if not os.path.exists("output"):
    os.makedirs("output")
    
input_video_path = (
    "datasets/test/normal/C_2_2_57_BU_DYB_10-20_13-59-11_CC_RGB_DF1_M4_M4.mp4"
)
input_filename = os.path.basename(input_video_path)
output_video_path = os.path.join("output/", input_filename)


process_video(input_video_path, output_video_path, model)

print(f"처리된 비디오가 {output_video_path}에 저장되었습니다.")
