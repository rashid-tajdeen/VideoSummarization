import os.path
import cv2
import numpy as np
import json
import random
import imreg_dft as ird

# Used for k-means
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Used for histogram
import matplotlib.pyplot as plt


class FrameExtraction:
    def __init__(self, method=None, dataset_root="../dataset/qv_pipe_dataset/", num_frames=5):
        needs_prep = ["histogram", "k_means", "less_motion", "less_blur"]
        if method is None:
            # prepare all methods
            methods = needs_prep
            for method in methods:
                self.prepare(method, dataset_root, num_frames)
        elif method in needs_prep:
            self.prepare(method, dataset_root, num_frames)

    def prepare(self, method, dataset_root, num_frames):
        data = {}
        # Check if data is already prepared
        data_file = "../frame_extraction/" + method + "_" + str(num_frames) + "frames.json"
        if os.path.isfile(data_file):
            print("Data for " + method + " frame extraction already exists")
            # Read the data from the JSON file
            with open(data_file, 'r') as file:
                data = json.load(file)

        # Get video names
        annotation_path = dataset_root + "qv_pipe_train.json"
        with open(annotation_path) as annotation_file:
            video_names = json.load(annotation_file).keys()

        # Defining method
        prepare_method = getattr(self, "_prepare_" + method)

        # Prepare data
        print("Preparing " + method + " frame data...")
        video_directory = dataset_root + "track1_raw_video/"
        count = 0
        total_count = len(video_names)
        for video_name in video_names:
            print(video_name)
            # Run only if data doesn't exist
            if video_name not in data.keys():
                video_path = video_directory + video_name

                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("Cannot open video at :", video_path)
                    exit()

                # Prepare data
                prepared_data = prepare_method(cap, num_frames)
                data[video_name] = prepared_data
                # Save the data in a JSON file
                with open(data_file, 'w') as file:
                    json.dump(data, file)

                # Close video
                cap.release()

            # Progress printing
            count += 1
            print("(" + str(count) + "/" + str(total_count) + ")" + " done")

    def _prepare_uniform(self, cap, num_frames):
        selected_frame_idx = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            frame_idx = int(i * ((total_frames - 1) / (num_frames - 1)))
            selected_frame_idx.append(frame_idx)
        return selected_frame_idx

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
        selected_frames = self._get_frames(cap, selected_frame_idx)

        cap.release()

        return selected_frames

    def _prepare_less_motion(self, cap, num_frames):
        motion_magnitude = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute motion by calculating the absolute difference between current and previous frames
            if prev_frame is not None:
                result = ird.translation(prev_frame, current_frame)
                motion = np.hypot(result["tvec"][0], result["tvec"][1])
            else:
                # High value to ignore
                motion = 1000
            motion_magnitude.append(motion)

            # Update the previous frame
            prev_frame = current_frame.copy()

        # Find index of frames with min motion
        selected_frame_idx = np.argsort(motion_magnitude)[0:num_frames]
        selected_frame_idx = selected_frame_idx.tolist()

        return selected_frame_idx

    def _prepare_less_blur(self, cap, num_frames):
        laplace_var_magnitude = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate blurriness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if variance is less, image is more blurry
            laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            laplace_var_magnitude.append(laplacian_variance)

        # Arrange frame index in increasing order of blurriness
        selected_frame_idx = np.argsort(laplace_var_magnitude)
        # Choose only the number of frames we need
        selected_frame_idx = selected_frame_idx[0:num_frames]
        selected_frame_idx = selected_frame_idx.tolist()

        return selected_frame_idx

    def _prepare_k_means(self, cap, num_frames):

        # Function to extract ResNet features from a frame
        def extract_resnet_features(frame, model):
            resized_frame = cv2.resize(frame, (224, 224))
            preprocessed_frame = preprocess_input(resized_frame)
            features = model.predict(np.expand_dims(preprocessed_frame, axis=0))
            return features.flatten()

        # Load pre-trained ResNet model
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        # Extract ResNet features from each frame
        features_list = []

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            print("Frame number : ", frame_number)
            features = extract_resnet_features(frame, resnet_model)
            features_list.append(features)

        # Convert features to numpy array
        features_array = np.array(features_list)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_frames, random_state=42)
        kmeans.fit(features_array)

        # # Select keyframes as centroids of clusters
        # selected_frame_idx = kmeans.cluster_centers_.argsort()[:, -1]

        # Select keyframes as centroids of clusters
        selected_frame_idx = []
        for cluster_center in kmeans.cluster_centers_:
            distances = np.linalg.norm(features_array - cluster_center, axis=1)
            closest_frame_index = np.argmin(distances)
            selected_frame_idx.append(closest_frame_index)

        selected_frame_idx = np.array(selected_frame_idx)
        selected_frame_idx = np.sort(selected_frame_idx)
        selected_frame_idx = selected_frame_idx.tolist()
        print("selected_frame_idx : ", selected_frame_idx)

        return selected_frame_idx

    def _prepare_histogram(self, cap, num_frames):

        def calculate_histogram(curr_frame):

            hist_b = cv2.calcHist([curr_frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([curr_frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([curr_frame], [2], None, [256], [0, 256])

            return hist_b, hist_g, hist_r

            chans = cv2.split(curr_frame)
            colors = ("b", "g", "r")
            plt.figure()
            plt.title("'Flattened' Color Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            # loop over the image channels
            for (chan, color) in zip(chans, colors):
                # create a histogram for the current channel and plot it
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
            plt.show()

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            # Calculate histogram
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            return hist

        def calculate_frame_difference(hist1, hist2):
            # Compute histogram differences for each channel
            diff_b = cv2.compareHist(hist1[0], hist2[0], cv2.HISTCMP_CHISQR)
            diff_g = cv2.compareHist(hist1[1], hist2[1], cv2.HISTCMP_CHISQR)
            diff_r = cv2.compareHist(hist1[2], hist2[2], cv2.HISTCMP_CHISQR)

            # Optionally, combine differences from individual channels
            combined_diff = diff_b + diff_g + diff_r

            return combined_diff

            # # Calculate histogram differences using chi-square distance
            # diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
            # return diff

        hist_differences = []

        # Read the first frame and calculate its histogram
        ret, prev_frame = cap.read()
        prev_hist = calculate_histogram(prev_frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate histogram for the current frame
            curr_hist = calculate_histogram(frame)
            # Calculate histogram difference between current and previous frames
            curr_diff = calculate_frame_difference(prev_hist, curr_hist)

            hist_differences.append(curr_diff)
            # Update previous histogram
            prev_hist = curr_hist

        plt.plot(hist_differences)
        plt.show()

        plt.plot(np.cumsum(hist_differences))
        plt.show()

        # exit(0)

    def load_frames(self, video_path, method, num_frames):
        # data_file = "../frame_extraction/" + method + "_" + str(num_frames) + "frames.json"
        # with open(data_file) as df:
        #     data = json.load(df)
        # video_name = os.path.split(video_path)[1]
        # selected_frame_idx = data[video_name]

        # Defining method
        load_method = getattr(self, "_load_" + method)
        selected_frames = load_method(video_path, num_frames)

        return selected_frames

    def _get_frames(self, cap, selected_frame_idx):
        selected_frames = []
        for frame_idx in selected_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            frame = frame[:, :, ::-1].copy()
            selected_frames.append(frame)

        selected_frames = np.array(selected_frames)
        return selected_frames


def main():
    frame_extraction = FrameExtraction("random")

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
