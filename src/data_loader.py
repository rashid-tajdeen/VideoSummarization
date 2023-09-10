import cv2
import numpy as np
import json
import os
import tensorflow as tf
import random


def get_uniform_frames(video_path,
                       frame_size,
                       num_of_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_summary = []
    for idx in range(num_of_frames):
        frame_idx = int(idx * ((total_frames - 1) / (num_of_frames - 1)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame = cv2.resize(frame, frame_size)
        # Normalize pixel values
        frame = frame / 255.0
        video_summary.append(frame)

    return video_summary


def get_random_frames(video_path,
                      frame_size,
                      num_of_frames):
    print("No method defined for random frames")


def load_dataset(labels_path='../dataset/qv_pipe_dataset/qv_pipe_train.json',
                 dataset_directory='../dataset/qv_pipe_dataset/track1_raw_video/',
                 summary_directory='../dataset/qv_pipe_dataset/summary/',
                 frame_size=(480, 480),
                 frames_per_video=5,
                 summary_type="uniform"):
    summary_data_path = summary_directory + summary_type + "_data_tensor.txt"
    summary_label_path = summary_directory + summary_type + "_label_tensor.txt"

    if not (os.path.isfile(summary_data_path) and
            os.path.isfile(summary_label_path)):
        prepare_dataset(labels_path,
                        dataset_directory,
                        summary_directory,
                        frame_size,
                        frames_per_video,
                        summary_type)

    # encoded_data = tf.io.read_file(summary_data_path)
    # encoded_label = tf.io.read_file(summary_label_path)
    #
    # loaded_data = tf.io.parse_tensor(tf.io.decode_base64(encoded_data), out_type=tf.float32)
    # loaded_label = tf.io.parse_tensor(tf.io.decode_base64(encoded_label), out_type=tf.int32)
    #
    # return loaded_data, loaded_label


def prepare_dataset(labels_path,
                    dataset_directory,
                    summary_directory,
                    frame_size,
                    frames_per_video,
                    summary_type):
    data = []
    label = []
    no_of_classes = 17  # 1 normal class and 16 defect classes

    # Opening the json file containing the labels for each video
    file_json = open(labels_path)
    data_label = json.load(file_json)

    if summary_type == "random":
        get_frames = get_random_frames
    else:  # default uniform frames
        get_frames = get_uniform_frames

    processed = 0
    for data_key in data_label:
        # Getting a video summary using custom function
        # This gives out limited frames from the video
        # Use a different function to use a different method to select frames:
        # -> get_uniform_frames
        video_summary = get_frames(dataset_directory + data_key,
                                   frame_size,
                                   frames_per_video)
        # Append the current video and label to the data
        data.append(video_summary)
        label.append([1 if i in data_label[data_key]
                      else 0
                      for i in range(no_of_classes)])

        processed = processed + 1
        print("Processed :", processed,
              ", Remaining :", len(data_label) - processed,
              ", filename :", data_key)

        if processed == 5:
            break   

    # data = tf.convert_to_tensor(data, np.float32)
    # label = tf.convert_to_tensor(label, np.int32)
    #
    # tf.io.write_file(summary_directory + summary_type + "_data_tensor.txt",
    #                  tf.io.encode_base64(tf.io.serialize_tensor(data)))
    # tf.io.write_file(summary_directory + summary_type + "_label_tensor.txt",
    #                  tf.io.encode_base64(tf.io.serialize_tensor(label)))


load_dataset()
