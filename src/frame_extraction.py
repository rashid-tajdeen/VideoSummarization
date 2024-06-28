import os.path
import time
import subprocess
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import json
import random
from pathlib import Path
import imreg_dft as ird
from scipy.stats import rankdata

# Used for k-means
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Used for histogram
import matplotlib.pyplot as plt


class FrameExtraction:
    def __init__(self, methods=None, dataset_root="../dataset/qv_pipe_dataset/", num_frames=5):

        # ThreadPool
        self.max_threads = 10
        self.threadPool = ThreadPoolExecutor(max_workers=self.max_threads)
        self.threads = []
        self.running_threads = 0

        self.data_directory = dataset_root + "track1_raw_data/"
        self.video_directory = dataset_root + "track1_raw_video/"

        needs_prep = []
        if methods is None:
            # prepare all methods
            for method in needs_prep:
                self.prepare(method, dataset_root, num_frames)
        else:
            for method in methods:
                if method in needs_prep:
                    self.prepare(method, dataset_root, num_frames)

    def prepare(self, method, dataset_root, num_frames):
        data = {}
        # Check if data is already prepared

        # if os.path.isfile(data_file):
        #     print("Data for " + method + " frame extraction already exists")
        #     # Read the data from the JSON file
        #     with open(data_file, 'r') as file:
        #         data = json.load(file)

        # Get video names
        annotation_path = dataset_root + "qv_pipe_train.json"
        with open(annotation_path) as annotation_file:
            video_names = json.load(annotation_file).keys()

        # Defining method
        prepare_method = getattr(self, "_prepare_" + method)

        # Prepare data
        print("Preparing " + method + " frame data...")
        count = 0
        total_count = len(video_names)
        for video_name in video_names:
            print(video_name)
            json_data_file = self.data_directory + method + "/" + Path(video_name).stem + ".json"

            # json_data = {}
            #if not os.path.isfile(json_data_file):
            # with open(json_data_file, 'r') as file:
            #     json_data = json.load(file)

            # if method in json_data.keys():
            #     print("Exists")
            #     continue

            # Open video
            video_path = self.video_directory + video_name

            # Prepare data
            prepare_method(video_path, json_data_file)
            # json_data[method] = prepared_data

            # Progress printing
            count += 1
            print("(" + str(count) + "/" + str(total_count) + ")" + " done")

    #def _prepare_uniform(self, cap, num_frames):
    #    selected_frame_idx = []
    #    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #    for i in range(num_frames):
    #        frame_idx = int(i * ((total_frames - 1) / (num_frames - 1)))
    #        selected_frame_idx.append(frame_idx)
    #    return selected_frame_idx

    def _prepare_random(self, cap, num_frames):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_frame_idx = random.sample(range(total_frames), num_frames)
        return selected_frame_idx

    def _load_random(self, video_path, num_frames):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video at :", video_path)
            exit()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_frame_idx = random.sample(range(total_frames), num_frames)

        cap.release()

        selected_frames = self._get_frames(video_path, selected_frame_idx)

        return selected_frames

    def _prepare_motion(self, video_path, json_file):

        def compute_frame_weightage():
            data = self._load_data_from_file(json_file)

            #if not "frame_weightage" in data.keys():

            #frame_weightage = np.zeros(len(data["x"]))
            #for parameter in data.keys():

            # Normalize the data neglecting the first frame as it has no frame to compare
            norm_translations_x = data["x"][1:] / np.max(np.abs(data["x"][1:]))
            norm_translations_y = data["y"][1:] / np.max(np.abs(data["y"][1:]))
            norm_scales = np.abs(np.array(data["scale"][1:]) - 1) / np.max(np.abs(np.array(data["scale"][1:]) - 1))
            norm_rotations = np.abs(data["rotation"][1:]) / np.max(np.abs(data["rotation"][1:]))

            wts = [1, 1, 1]

            motion = wts[0] * (norm_translations_x + norm_translations_y) + wts[1] * norm_scales + wts[2] * norm_rotations
            
            motion = np.concatenate([np.array([1000]), motion])


            sorted_indices = sorted(range(len(motion)), key=lambda k: motion[k], reverse=True)
            frame_weightage = np.array([sorted_indices.index(i) + 1 for i in range(len(sorted_indices))])
            

            #frame_weightage = frame_weightage/len(data.keys())
            data["frame_weightage"] = frame_weightage.tolist()

            self._save_data_to_file(json_file, data)

        # Skip preparation if file already exists
        if not os.path.isfile(json_file):
            def thread(v_path, j_path):
                subprocess.run(["../imreg_fmt/build/video_main", v_path, j_path])
                compute_frame_weightage()
            if self.running_threads < self.max_threads:
                self.threads.append(self.threadPool.submit(thread, video_path, json_file))
                self.running_threads += 1
            else:
                concurrent.futures.wait(self.threads, return_when=concurrent.futures.FIRST_COMPLETED).done.pop()
                self.threads = [t for t in self.threads if not t.done()]
                self.threads.append(self.threadPool.submit(thread, video_path, json_file))
                self.running_threads = len(self.threads) + 1
        else:
            compute_frame_weightage()


    def _load_motion(self, video_path, num_frames):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video at :", video_path)
            exit()

        # Open the data file
        _, file = os.path.split(video_path)
        json_file = self.data_directory + "motion/" + os.path.splitext(file)[0] + ".json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Leave out 1st frame as it does not have prev frame
        rotation_data = json_data["rotation"][1:]
        scale_data = json_data["scale"][1:]
        x_data = json_data["x"][1:]
        y_data = json_data["y"][1:]

        rotation_average = sum(np.absolute(rotation_data)) / len(rotation_data)
        scale_average = sum(np.absolute(scale_data)) / len(scale_data)
        x_average = sum(np.absolute(x_data)) / len(x_data)
        y_average = sum(np.absolute(y_data)) / len(y_data)

        valid_frame_idx = []
        for idx in range(len(json_data["x"])):
            if abs(json_data["rotation"][idx]) < rotation_average and \
                    abs(json_data["scale"][idx]) < scale_average and \
                    abs(json_data["x"][idx]) < x_average and \
                    abs(json_data["y"][idx]) < y_average:
                valid_frame_idx.append(idx)

        if len(valid_frame_idx) < num_frames:
            required = num_frames - len(valid_frame_idx)
            not_yet_selected = [i for i in range(len(json_data["x"])) if i not in valid_frame_idx]
            valid_frame_idx += random.sample(not_yet_selected, required)

        selected_frame_idx = random.sample(valid_frame_idx, num_frames)
        # Select frames
        selected_frames = self._get_frames(cap, selected_frame_idx)

        cap.release()

        return selected_frames

    def _prepare_less_blur(self, video_path, json_file):

        # Skip preparation if file already exists
        if os.path.isfile(json_file):
            data = self._load_data_from_file(json_file)
        else:
            data = {"notes": "More laplacian variance means less blurry"}


        change_flag = False
        cap = self._open_video(video_path)

        if not "laplacian_variance" in data.keys():
            laplacian_variance = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Calculate blurriness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # if variance is less, image is more blurry
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                laplacian_variance.append(lap_var)

            data["laplacian_variance"] = laplacian_variance
            change_flag = True

        if not "frame_weightage" in data.keys():
            sorted_indices = sorted(range(len(data["laplacian_variance"])), key=lambda k: data["laplacian_variance"][k])
            data["frame_weightage"] = [sorted_indices.index(i) + 1 for i in range(len(sorted_indices))]
            change_flag = True


        if change_flag:
            self._verify_data_count(data, cap)
            self._save_data_to_file(json_file, data)

        # Close video
        cap.release()

    def _load_less_blur(self, video_path, num_frames):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video at :", video_path)
            exit()

        # Open the data file
        _, file = os.path.split(video_path)
        json_file = self.data_directory + "less_blur/" + os.path.splitext(file)[0] + ".json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Select frame indices
        data = json_data["laplacian_variance"]
        average = sum(data) / len(data)
        print("Average laplacian variance is", average)
        valid_frame_idx = [index for index, value in enumerate(data) if value > average]
        selected_frame_idx = random.sample(valid_frame_idx, num_frames)

        # Select frames
        selected_frames = self._get_frames(cap, selected_frame_idx)

        cap.release()

        return selected_frames

    def _prepare_k_means(self, video_path, json_file):

        cluster_nums = [5, 1]
        clusters_to_do = []

        # Skip preparation if file already exists
        if os.path.isfile(json_file):
            all_data = self._load_data_from_file(json_file)
            for k in cluster_nums:
                print(k)
                if str(k) + "_keyframes" not in all_data.keys():
                    clusters_to_do.append(k)

            if len(clusters_to_do) == 0:
                return
        else:
            all_data = {}
            clusters_to_do = cluster_nums

        # Function to extract ResNet features from a frame
        def extract_resnet_features(fr, model):
            resized_frame = cv2.resize(fr, (224, 224))
            preprocessed_frame = preprocess_input(resized_frame)
            feat = model.predict(np.expand_dims(preprocessed_frame, axis=0))
            return feat.flatten()

        # Load pre-trained ResNet model
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        # Extract ResNet features from each frame
        features_array = []
        frame_number = 0
        cap = self._open_video(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            print("Frame number : ", frame_number)
            features = extract_resnet_features(frame, resnet_model)
            features_array.append(np.array(features).tolist())
        # Convert features to numpy array
        features_array = np.array(features_array)

        for num_frames in clusters_to_do:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_frames, random_state=33)
            kmeans.fit(features_array)

            cluster_labels = []
            distance_to_centroid = []

            # Assign frames to their corresponding clusters
            for frame_idx, frame_features in enumerate(features_array):
                cluster_label = kmeans.labels_[frame_idx]
                distance = np.linalg.norm(frame_features - kmeans.cluster_centers_[cluster_label])

                cluster_labels.append(int(cluster_label))
                distance_to_centroid.append(float(distance))

            data = {"cluster_labels": cluster_labels,
                    "distance_to_centroid": distance_to_centroid}

            sorted_indices = sorted(range(len(data["distance_to_centroid"])), key=lambda k: data["distance_to_centroid"][k], reverse=True)
            data["frame_weightage"] = [sorted_indices.index(i) + 1 for i in range(len(sorted_indices))]

            all_data[str(num_frames) + "_keyframes"] = data

        #print(data)
        #print("####################################")
        #print(all_data)

        self._save_data_to_file(json_file, all_data)

        # Close video
        cap.release()

    def _load_k_means(self, video_path, num_frames):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video at :", video_path)
            exit()

        # Open the data file
        _, file = os.path.split(video_path)
        json_file = self.data_directory + "k_means/" + os.path.splitext(file)[0] + ".json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        if str(num_frames) + "_keyframes" not in json_data.keys():
            print("Data not prepared for the specified number of frames")
            exit(1)
        selected_frame_idx = json_data[str(num_frames) + "_keyframes"]
        # Select frames
        selected_frames = self._get_frames(cap, selected_frame_idx)

        cap.release()

        return selected_frames

    def _prepare_histogram(self, video_path, json_file):
        # Skip preparation if file already exists
        if os.path.isfile(json_file):
            return

        def calculate_histogram(curr_frame):
            # hist_b = cv2.calcHist([curr_frame], [0], None, [256], [0, 256])
            # hist_g = cv2.calcHist([curr_frame], [1], None, [256], [0, 256])
            # hist_r = cv2.calcHist([curr_frame], [2], None, [256], [0, 256])
            #
            # return hist_b, hist_g, hist_r
            #
            # chans = cv2.split(curr_frame)
            # colors = ("b", "g", "r")
            # plt.figure()
            # plt.title("'Flattened' Color Histogram")
            # plt.xlabel("Bins")
            # plt.ylabel("# of Pixels")
            # # loop over the image channels
            # for (chan, color) in zip(chans, colors):
            #     # create a histogram for the current channel and plot it
            #     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            #     plt.plot(hist, color=color)
            #     plt.xlim([0, 256])
            # plt.show()

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            # Calculate histogram
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            return hist

        def calculate_frame_difference(hist1, hist2):
            # # Compute histogram differences for each channel
            # diff_b = cv2.compareHist(hist1[0], hist2[0], cv2.HISTCMP_CHISQR)
            # diff_g = cv2.compareHist(hist1[1], hist2[1], cv2.HISTCMP_CHISQR)
            # diff_r = cv2.compareHist(hist1[2], hist2[2], cv2.HISTCMP_CHISQR)
            #
            # # Optionally, combine differences from individual channels
            # combined_diff = diff_b + diff_g + diff_r
            #
            # return combined_diff

            # Calculate histogram differences using chi-square distance
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            return diff

        cap = self._open_video(video_path)

        data = {"histogram_diff": []}

        # Read the first frame and calculate its histogram
        ret, prev_frame = cap.read()
        prev_hist = calculate_histogram(prev_frame)
        #data["histogram"].append(prev_hist)
        data["histogram_diff"].append(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate histogram for the current frame
            curr_hist = calculate_histogram(frame)
            #data["histogram"].append(curr_hist)

            # Calculate histogram difference between current and previous frames
            curr_diff = calculate_frame_difference(prev_hist, curr_hist)

            data["histogram_diff"].append(curr_diff)
            # Update previous histogram
            prev_hist = curr_hist

        sorted_indices = sorted(range(len(data["histogram_diff"])), key=lambda k: data["histogram_diff"][k])
        data["frame_weightage"] = [sorted_indices.index(i) + 1 for i in range(len(sorted_indices))]

        # plt.plot(histogram_diff)
        # plt.show()
        # plt.plot(np.cumsum(histogram_diff))
        # plt.show()

        self._verify_data_count(data, cap)
        self._save_data_to_file(json_file, data)

        # Close video
        cap.release()

    def _prepare_entropy(self, video_path, json_file):
        # Skip preparation if file already exists
        if os.path.isfile(json_file):
            return

        def calculate_entropy(frame):
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Compute the histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            # Normalize the histogram
            hist = hist / hist.sum()
            # Compute the entropy
            entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
            return entropy

        cap = cv2.VideoCapture(video_path)

        data = {"entropy": []}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the entropy of the current frame
            entropy = calculate_entropy(frame)
            data["entropy"].append(float(entropy))

        sorted_indices = sorted(range(len(data["entropy"])), key=lambda k: data["entropy"][k])
        data["frame_weightage"] = [sorted_indices.index(i) + 1 for i in range(len(sorted_indices))]

        self._verify_data_count(data, cap)
        self._save_data_to_file(json_file, data)

        # Close video
        cap.release()


    def load_frames(self, video_path, methods, num_frames, meth_prio):

        if methods == ["random"]:
            return self._load_random(video_path, num_frames)
        #elif method == "all":
        #    method = ["less_blur", "motion", "histogram"]
        #else:
        #    method = [method]

        frame_weightage = np.array([])

        for meth, multiplier in zip(methods, meth_prio):
            # Open the data file
            _, file = os.path.split(video_path)
            json_file = self.data_directory + meth + "/" + os.path.splitext(file)[0] + ".json"
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            # Special handling for k-means as it depends on frame count
            if meth == "k_means":
                json_data = json_data[str(num_frames) + "_keyframes"]
            # Initialise for the firse time
            if len(frame_weightage) == 0:
                frame_weightage = np.zeros(len(json_data["frame_weightage"]))
# Approach 1 #############################################################################################
        #    # Compute frame weightage
        #    frame_weightage += np.array(json_data["frame_weightage"]) * multiplier

        ## Get frame indices in descending order of frame_weightage
        #frames_by_priority = np.argsort(frame_weightage)[::-1]
        ## Select 5 times the required number of frames for randomness
        #eligible_frame_idx = frames_by_priority[:(num_frames*5)] if len(frames_by_priority)>(num_frames*5) else frames_by_priority
        ## Select the required number of frames randomly
        #selected_frame_idx = random.sample(eligible_frame_idx.tolist(), num_frames)
##########################################################################################################

# Approach 2 #############################################################################################

            # Compute frame weightage
            #frame_weightage += np.square(np.array(json_data["frame_weightage"])) * multiplier

        #selected_frame_idx = []
        #while(len(selected_frame_idx) != num_frames):
        #    # Randomly select with distribution
        #    selected_frame_idx += random.choices(range(len(frame_weightage)),
        #                                         weights=frame_weightage.tolist(),
        #                                         k=num_frames-len(selected_frame_idx))
        #    # Delete duplicates
        #    selected_frame_idx = list(set(selected_frame_idx))
        #
        #    if(len(selected_frame_idx) > num_frames):
        #        exit(1)
##########################################################################################################

# Approach 3 #############################################################################################
#            # Compute frame weightage
#            frame_weightage += np.array(json_data["frame_weightage"]) * multiplier
#
#        if (len(frame_weightage) % num_frames !=0):
#            frame_weightage = np.concatenate((frame_weightage, np.zeros(num_frames - (len(frame_weightage) % num_frames))))
#            if (len(frame_weightage) % num_frames !=0):
#                exit(1)
#        weight_batches = np.array_split(frame_weightage, num_frames)
#
#        selected_frame_idx = []
#        for idx, b in enumerate(weight_batches):
#            batch_len = len(b)
#            b = b[b!=0]
#            batch_priority = np.argsort(b)[::-1] + (idx * batch_len)
#
#            eligible_from_batch = batch_priority[:(num_frames)] if len(batch_priority)>(num_frames) else batch_priority
#            selected_from_batch = np.random.choice(eligible_from_batch)
#            selected_frame_idx.append(selected_from_batch)
##########################################################################################################

# Approach 4 #############################################################################################
            # Compute frame weightage
            frame_weightage += np.array(json_data["frame_weightage"]) * multiplier

        if (len(frame_weightage) % num_frames !=0):
            frame_weightage = np.concatenate((frame_weightage, np.zeros(num_frames - (len(frame_weightage) % num_frames))))
            if (len(frame_weightage) % num_frames !=0):
                exit(1)
        weight_batches = np.array_split(frame_weightage, num_frames)

        selected_frame_idx = []
        for idx, b in enumerate(weight_batches):
            selected_from_batch = random.choices(range(len(b)),
                                                 weights=b.tolist(),
                                                 k=1) + (idx * len(b))
            selected_frame_idx.append(selected_from_batch)
##########################################################################################################


        # Get selected frames
        selected_frames = self._get_frames(video_path, selected_frame_idx)
        return selected_frames

    def _get_frames(self, video_path, selected_frame_idx):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video at :", video_path)
            exit()

        selected_frames = []
        for frame_idx in selected_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            frame = frame[:, :, ::-1].copy()
            selected_frames.append(frame)

        selected_frames = np.array(selected_frames)

        # Close video
        cap.release()

        return selected_frames

    def _verify_data_count(self, data, cap):
        for key in data.keys():
            if (len(data[key]) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) and (key != "notes"):
                print("Data length does not match frame count")
                exit(1)

    def _open_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open video at :", video_path)
            exit()
        return cap

    def _load_data_from_file(self, file):
        # Load the data from a JSON file
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    def _save_data_to_file(self, file, data):
        # Save the data in a JSON file
        with open(file, 'w') as f:
            json.dump(data, f)


def main():
    frame_extraction = FrameExtraction()

    # # For testing the frame selection
    #
    # from torchvision import transforms
    # import matplotlib.pyplot as plt
    # from qvpipe_dataset import QVPipeDataset
    #
    # transform = transforms.Compose([
    #     transforms.Resize((240, 240)),
    #     # transforms.RandomAdjustSharpness(1.5),
    #     # transforms.RandomAutocontrast(),
    #     # transforms.RandomHorizontalFlip(),
    #     # transforms.RandomVerticalFlip(),
    #     # transforms.RandomErasing(),
    #     # transforms.GaussianBlur(kernel_size=3),
    #     # transforms.RandomRotation(30),
    #     # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.25)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #
    # uniform_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 1, 5, "../dataset/qv_pipe_dataset/train_keys.json",
    #                                 transform, 'uniform')
    # random_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 1, 5, "../dataset/qv_pipe_dataset/train_keys.json",
    #                                transform, 'random')
    # less_motion_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 1, 5,
    #                                     "../dataset/qv_pipe_dataset/train_keys.json", transform, 'less_motion')
    # less_blur_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 1, 5, "../dataset/qv_pipe_dataset/train_keys.json",
    #                                   transform, 'less_blur')
    #
    # for idd in [0, 10]:
    #     uniform_video, _ = uniform_dataset[idd]
    #     uniform_video = uniform_video.permute(0, 2, 3, 1).reshape(5 * 240, 240, 3).numpy()
    #
    #     random_video, _ = random_dataset[idd]
    #     random_video = random_video.permute(0, 2, 3, 1).reshape(5 * 240, 240, 3).numpy()
    #
    #     less_motion_video, _ = less_motion_dataset[idd]
    #     less_motion_video = less_motion_video.permute(0, 2, 3, 1).reshape(5 * 240, 240, 3).numpy()
    #
    #     less_blur_video, _ = less_blur_dataset[idd]
    #     less_blur_video = less_blur_video.permute(0, 2, 3, 1).reshape(5 * 240, 240, 3).numpy()
    #
    #     types = 4
    #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, types, figsize=(3 * types, 8))
    #
    #     ax1.set_title('Uniform Sampled Frames')
    #     ax1.imshow(uniform_video)
    #     ax2.set_title('Random Sampled Frames')
    #     ax2.imshow(random_video)
    #     ax3.set_title('Less Motion Frames')
    #     ax3.imshow(less_motion_video)
    #     ax4.set_title('Less Blur Frames')
    #     ax4.imshow(less_blur_video)
    #
    #     plt.show()


if __name__ == '__main__':
    main()
