import cv2
import numpy as np
import random


def load_frames_uniform(video_path, num_key_frames):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video at :", video_path)
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_frames = []
    selected_frame_idx = []
    for i in range(num_key_frames):
        frame_idx = int(i * ((total_frames - 1) / (num_key_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame = frame[:, :, ::-1].copy()
        selected_frames.append(frame)
        selected_frame_idx.append(frame_idx)

    cap.release()

    selected_frames = np.array(selected_frames)
    return selected_frames


def load_frames_random(video_path, num_key_frames):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video at :", video_path)
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frame_idx = random.sample(range(total_frames), num_key_frames)

    selected_frames = []
    for frame_idx in selected_frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame = frame[:, :, ::-1].copy()
        selected_frames.append(frame)

    cap.release()

    selected_frames = np.array(selected_frames)
    return selected_frames


def load_frames_less_motion(video_path, num_key_frames):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video at :", video_path)
        exit()

    motion_magnitude = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute motion by calculating the absolute difference between current and previous frames
        if prev_frame is not None:
            frame_diff = cv2.absdiff(current_frame, prev_frame)
            motion = cv2.sumElems(frame_diff)[0] / current_frame.size
        else:
            motion = 0

        motion_magnitude.append(motion)

        # Update the previous frame
        prev_frame = current_frame.copy()

    # Find index of frames with min motion
    selected_frame_idx = np.argsort(motion_magnitude)[0:num_key_frames]

    selected_frames = []
    for frame_idx in selected_frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame = frame[:, :, ::-1].copy()
        selected_frames.append(frame)

    cap.release()

    selected_frames = np.array(selected_frames)
    return selected_frames


def load_frames_less_blur(video_path, num_key_frames):
    cap = cv2.VideoCapture(video_path)

    blur_magnitude = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate blurriness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurriness = cv2.Laplacian(gray, cv2.CV_64F).var()

        blur_magnitude.append(blurriness)

    # Find index of frames with min blur
    selected_frame_idx = np.argsort(blur_magnitude)[0:num_key_frames]

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
    # For testing the frame selection

    from torchvision import transforms
    import matplotlib.pyplot as plt
    from qvpipe_dataset import QVPipeDataset

    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        # transforms.RandomAdjustSharpness(1.5),
        # transforms.RandomAutocontrast(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomErasing(),
        # transforms.GaussianBlur(kernel_size=3),
        # transforms.RandomRotation(30),
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.25)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    uniform_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 17, 5, transform, 'train', 'uniform')
    random_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 17, 5, transform, 'train', 'random')
    less_motion_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 17, 5, transform, 'train', 'less_motion')
    less_blur_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 17, 5, transform, 'train', 'less_blur')

    for idd in [0, 10]:
        uniform_video, _ = uniform_dataset[idd]
        uniform_video = uniform_video.permute(0, 2, 3, 1).reshape(5*240, 240, 3).numpy()

        random_video, _ = random_dataset[idd]
        random_video = random_video.permute(0, 2, 3, 1).reshape(5 * 240, 240, 3).numpy()

        less_motion_video, _ = less_motion_dataset[idd]
        less_motion_video = less_motion_video.permute(0, 2, 3, 1).reshape(5 * 240, 240, 3).numpy()

        less_blur_video, _ = less_blur_dataset[idd]
        less_blur_video = less_blur_video.permute(0, 2, 3, 1).reshape(5 * 240, 240, 3).numpy()

        types = 4
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, types, figsize=(3 * types, 8))

        ax1.set_title('Uniform Sampled Frames')
        ax1.imshow(uniform_video)
        ax2.set_title('Random Sampled Frames')
        ax2.imshow(random_video)
        ax3.set_title('Less Motion Frames')
        ax3.imshow(less_motion_video)
        ax4.set_title('Less Blur Frames')
        ax4.imshow(less_blur_video)

        plt.show()


if __name__ == '__main__':
    main()
