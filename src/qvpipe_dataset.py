import torch
from torch.utils.data import DataLoader
import json
import random
import frame_extraction


# Custom video dataset class that loads video frames and their labels.
class QVPipeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_root="../dataset/qv_pipe_dataset/",
                 no_of_classes=17,
                 num_key_frames=5,
                 transform=None,
                 frame_selection='uniform'):

        self.num_key_frames = num_key_frames
        self.transform = transform

        video_directory = dataset_root + "track1_raw_video/"
        annotation_path = dataset_root + "qv_pipe_train.json"

        file_json = open(annotation_path)
        annotations = json.load(file_json)

        video_files = list(annotations.keys())
        random.seed(10)
        random.shuffle(video_files)

        self.video_paths = []
        self.labels = []
        # _video_count = 0
        for video_name in video_files:
            self.video_paths.append(video_directory + video_name)
            self.labels.append([1 if i in annotations[video_name]
                                else 0
                                for i in range(no_of_classes)])
            # _video_count = _video_count + 1
            # if _video_count == 10:
            #     break

        # Set frame selection method
        self.load_frames = getattr(frame_extraction, 'load_frames_' + frame_selection)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_summary = self.load_frames(video_path, self.num_key_frames)
        label = torch.Tensor(self.labels[idx])
        if self.transform is not None:
            video_summary = torch.from_numpy(video_summary)
            video_summary = video_summary.permute((0, 3, 1, 2)).contiguous()
            video_summary = video_summary.to(dtype=torch.get_default_dtype())
            video_summary = video_summary.div(255)
            video_summary = self.transform(video_summary)
        return video_summary, label

    def __len__(self):
        return len(self.video_paths)


def main():
    # For testing the QVPipeDataset class

    import numpy as np
    from torchvision import transforms
    import matplotlib.pyplot as plt

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
    video_dataset = QVPipeDataset("../dataset/qv_pipe_dataset/", 17, 5, transform, 'train', 'uniform')

    all_ids = np.array(list(range(len(video_dataset))))
    ids = all_ids[np.random.choice(len(all_ids), size=3, replace=False)]
    for idd in ids:
        video, label = video_dataset[idd]
        video = video.permute(0, 2, 3, 1).reshape(5*240, 240, 3).numpy()
        plt.imshow(video)
        plt.show()


if __name__ == '__main__':
    main()
