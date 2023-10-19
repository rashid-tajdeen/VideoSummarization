import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from qvpipe_dataset import QVPipeDataset
from model import TModel2, Custom3DModel
from torchvision import transforms
import neptune
import argparse
import progressbar


def main():
    # Cuda specific code
    device = torch.device(f'cuda:{torch.cuda.current_device()}'
                          if torch.cuda.is_available()
                          else 'cpu')

    # Parse the command-line arguments
    args = parse_arguments()

    logger = neptune_logger()

    params = {"frame_selection": args.frame_selection,
              "dataset_root": args.dataset_path,
              "learning_rate": args.learning_rate,
              "num_key_frames": args.num_frames,
              "num_epochs": args.epochs,
              "batch_size": args.batch_size,
              "no_of_classes": 17,
              "resize_x": 240,
              "resize_y": 240,
              "channels": 3,
              "model_path": '../models/' +
                            args.frame_selection + '_' +
                            str(args.num_frames) + 'frames_' +
                            str(args.epochs) + 'epochs' +
                            '.pth',
              "train": args.train,
              "valid": args.valid,
              "early_stopping": args.early_stopping
              }
    logger["parameters"] = params

    train_loader, valid_loader = load_dataset(params)

    # model = TModel2(params["num_key_frames"], params["no_of_classes"])
    expected_input_shape = (params["batch_size"],
                            params["num_key_frames"],
                            params["channels"],
                            params["resize_x"],
                            params["resize_x"])
    model = Custom3DModel(expected_input_shape, params["no_of_classes"])
    # model = model.to(device)

    if args.train:
        logger["sys/tags"].add("train")
        train_step(params, train_loader, model, device, logger)
    if args.valid:
        logger["sys/tags"].add("valid")
        valid_step(params, valid_loader, model, device, logger)

    logger.stop()


def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Command line arguments that might be helpful to run the code')
    # Add arguments
    parser.add_argument('--frame_selection', type=str,
                        choices=['uniform', 'random', 'less_motion', 'less_blur'],
                        default='uniform',
                        help='Frame selection method to be chosen from the provided options')
    parser.add_argument('--dataset_path', type=str,
                        default='../dataset/qv_pipe_dataset/',
                        help='Path to the dataset directory')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--num_frames', '-n', type=int,
                        default=5,
                        help='Number of key frames to be selected for each video')
    parser.add_argument('--epochs', '-e', type=int,
                        default=20,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=16,
                        help='Batch size for training')
    parser.add_argument('--train', action='store_true',
                        help='Flag to enable training')
    parser.add_argument('--valid', action='store_true',
                        help='Flag to enable validation')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Flag to enable early stopping')

    return parser.parse_args()


def neptune_logger():
    # Credentials
    logger = neptune.init_run(
        project="rashid.deutschland/QVPipe",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZDhmYjkzZC0xMGUwLTRkZjAtOGFjMC1kNGU3NzA3YmQ3ZjkifQ==",
    )
    return logger


def load_dataset(params):
    transform = transforms.Compose([
        transforms.Resize((params["resize_x"], params["resize_y"])),
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

    # Load train dataset
    if params["train"]:
        train_keys_path = params["dataset_root"] + "train_keys.json"
        # Load your custom video dataset using DataLoader
        train_dataset = QVPipeDataset(params["dataset_root"],
                                      params["no_of_classes"],
                                      params["num_key_frames"],
                                      train_keys_path,
                                      transform=transform,
                                      frame_selection=params["frame_selection"])
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    else:
        train_loader = None

    # Load valid dataset
    if params["valid"]:
        valid_keys_path = params["dataset_root"] + "val_keys.json"
        # Load your custom video dataset using DataLoader
        valid_dataset = QVPipeDataset(params["dataset_root"],
                                      params["no_of_classes"],
                                      params["num_key_frames"],
                                      valid_keys_path,
                                      transform=transform,
                                      frame_selection=params["frame_selection"])
        valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"])
    else:
        valid_loader = None

    return train_loader, valid_loader


def train_step(params, train_loader, model, device, logger):
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Early stoping requirements
    prev_loss = 1  # Initially giving the maximum possible value
    trigger_times = 0
    patience = 3

    # Training loop
    for epoch in range(params["num_epochs"]):
        print("==========================================")
        print(f"Epoch {epoch + 1}/{params['num_epochs']}")
        print("==========================================")

        print("Training...")

        # Add progress bar
        bar = progressbar.ProgressBar(maxval=len(train_loader.dataset),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()

        model.train()
        running_loss = 0.0

        batch_done = 0
        for inputs, labels in train_loader:
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            curr_batch_len = len(inputs)

            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            # print("Getting outputs from model...")
            outputs = model(inputs)
            # print("Output from model : ", outputs)
            # print("Actual output : ", labels)
            loss = criterion(outputs, labels)
            # print("Back Propagating to update weights...")
            loss.backward()
            # print("Weights updated!")
            optimizer.step()

            running_loss += loss.item()

            # Update progress
            bar.update((batch_done * params["batch_size"]) + curr_batch_len)
            batch_done += 1

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch Loss: {epoch_loss:.4f}")
        logger["train_loss"].append(epoch_loss)

        # Close progress bar
        bar.finish()

        if params["early_stopping"]:
            current_loss = epoch_loss
            if current_loss > prev_loss:
                trigger_times += 1
                print('Loss increasing... Trigger Times :', trigger_times)
                if trigger_times >= patience:
                    print(f"Loss has been increasing for last {trigger_times} epochs")
                    print('Early stopping!!!')
                    break
            else:
                trigger_times = 0
                torch.save(model.state_dict(), params["model_path"])
                print("Model saved to ", params["model_path"])
            prev_loss = current_loss
        else:
            torch.save(model.state_dict(), params["model_path"])
            print("Model saved to ", params["model_path"])

    print("Training complete!")


def valid_step(params, valid_loader, model, device, logger):
    # Load trained model for evaluation
    model.load_state_dict(torch.load(params["model_path"]))

    # Change model to evaluation mode
    model.eval()

    # Validation loop
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.BCELoss()
    with torch.no_grad():
        # Add progress bar
        bar = progressbar.ProgressBar(maxval=len(valid_loader.dataset),
                                      widgets=[progressbar.Bar('=', 'Validating...[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()
        batch_done = 0

        for inputs, labels in valid_loader:
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Predict labels
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0) * labels.size(1)

            # Update progress
            curr_batch_len = len(inputs)
            bar.update((batch_done * params["batch_size"]) + curr_batch_len)
            batch_done += 1

            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

        # Close progress bar
        bar.finish()

    average_loss = total_loss / len(valid_loader)
    accuracy = (correct_predictions / total_samples) * 100

    print(f"Validation Loss: {average_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
