import os.path
import cv2
import numpy as np
import json
import random
import imreg_dft as ird


class FrameExtraction:
    def __init__(self, method=None, dataset_root="../dataset/qv_pipe_dataset/", num_frames=5):

        if method is None:
            # prepare all methods
            method = ["uniform", "random", "less_motion", "less_blur"]
            for meth in method:
                self.prepare(meth, dataset_root, num_frames)
        else:
            self.prepare(method, dataset_root, num_frames)

    def prepare(self, method, dataset_root, num_frames):
        # Check if data is already prepared
        data_file = "../frame_extraction/" + method + "_" + str(num_frames) + "frames.json"
        if os.path.isfile(data_file):
            print("Data for " + method + " frame extraction already exists")
            return

        # Get video names
        annotation_path = dataset_root + "qv_pipe_train.json"
        with open(annotation_path) as annotation_file:
            video_names = json.load(annotation_file).keys()

        # Defining method
        get_frame_index = getattr(self, "frame_index_" + method)

        # Prepare data
        print("Preparing " + method + " frame data...")
        data = {}
        video_directory = dataset_root + "track1_raw_video/"
        count = 0
        total_count = len(video_names)
        for video_name in video_names:
            video_path = video_directory + video_name

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Cannot open video at :", video_path)
                exit()

            # Select frame index
            selected_frame_idx = get_frame_index(cap, num_frames)
            data[video_name] = selected_frame_idx

            # Close video
            cap.release()

            # Progress printing
            count += 1
            print("(" + str(count) + "/" + str(total_count) + ")" + " done")

        # Save the data in a JSON file
        with open(data_file, 'w') as file:
            json.dump(data, file)

    def frame_index_uniform(self, cap, num_frames):
        selected_frame_idx = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            frame_idx = int(i * ((total_frames - 1) / (num_frames - 1)))
            selected_frame_idx.append(frame_idx)
        return selected_frame_idx

    def frame_index_random(self, cap, num_frames):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_frame_idx = random.sample(range(total_frames), num_frames)
        return selected_frame_idx

    def frame_index_less_motion(self, cap, num_frames):
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

        return selected_frame_idx

    def frame_index_less_blur(self, cap, num_frames):
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

        return selected_frame_idx

    def load_frames(self, video_path, method, num_frames):
        data_file = "../frame_extraction/" + method + "_" + num_frames + "frames.json"
        with open(data_file) as df:
            data = json.load(df)
        video_name = os.path.split(video_path)[1]
        selected_frame_idx = data[video_name]

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

        cap.release()

        selected_frames = np.array(selected_frames)
        return selected_frames


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
