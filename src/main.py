import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from qvpipe_dataset import QVPipeDataset
from model import TModel2, Custom3DModel
import torchvision
from torchvision import transforms
import torchnet
import neptune
import argparse
import progressbar
from datetime import datetime

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, multilabel_confusion_matrix


def main():
    # Cuda specific code
    device = torch.device(f'cuda:{torch.cuda.current_device()}'
                          if torch.cuda.is_available()
                          else 'cpu')

    # Parse the command-line arguments
    args = parse_arguments()

    run_id = str(args.num_frames) + 'frame_' + '_'.join(args.frame_selection) + '_' + datetime.today().strftime('%Y_%m_%d_%H:%M')
    if args.model_name == None:
        model_path = '../models/' + run_id + '.pth'
    else:
        model_path = '../models/' + args.model_name

    params = {"frame_selection": args.frame_selection,
              "dataset_root": args.dataset_path,
              "learning_rate": args.learning_rate,
              "num_key_frames": args.num_frames,
              "num_epochs": args.epochs,
              "batch_size": args.batch_size,
              "num_classes": args.classes,
              "resize_x": 240,
              "resize_y": 240,
              "channels": 3,
              "model_path": model_path,
              "train": args.train,
              "valid": args.valid,
              "early_stopping": args.early_stopping
              }

    log_params = params.copy()
    log_params["frame_selection"] = '_'.join(log_params["frame_selection"])

    logger = neptune_logger(run_id)
    logger["parameters"] = log_params

    train_loader, valid_loader = load_dataset(params)
    epoch_valid_loader = get_subset(valid_loader)

    expected_input_shape = (params["batch_size"],
                            params["num_key_frames"],
                            params["channels"],
                            params["resize_x"],
                            params["resize_x"])
    # model = Custom3DModel(expected_input_shape, params["num_classes"])
    model = TModel2(params["num_key_frames"], params["num_classes"])
    #train_loader.dataset.update_method_priorities(model.method_priorities, logger)
    #print([float(model.method_priorities[0]), float(model.method_priorities[1]), float(model.method_priorities[2])])
    # model = model.to(device)

    # Start to train on existing model
    if os.path.isfile(params["model_path"]):
        model.load_state_dict(torch.load(params["model_path"]))
        print(f'Starting from existing model in "{params["model_path"]}"')

    if args.train:
        logger["sys/tags"].add("train")
        train_step(params, train_loader, epoch_valid_loader, model, device, logger)
    if args.valid:
        logger["sys/tags"].add("valid")
        valid_step(params, valid_loader, model, device, logger)

    logger.stop()


def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Command line arguments that might be helpful to run the code')
    # Add arguments
    #parser.add_argument('--frame_selection', type=str,
    #                    choices=['random', 'less_blur', 'motion', "histogram", "k_means", "all"],
    #                    default='uniform',
    #                    help='Frame selection method to be chosen from the provided options')
    parser.add_argument('--frame_selection', nargs='+',
                        required=True,
                        help='Frame selection method to be chosen from the provided options')
    parser.add_argument('--dataset_path', type=str,
                        default='../dataset/qv_pipe_dataset/',
                        help='Path to the dataset directory')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.01,
                        help='Learning rate for training')
    parser.add_argument('--num_frames', '-n', type=int,
                        default=5,
                        help='Number of key frames to be selected for each video')
    parser.add_argument('--epochs', '-e', type=int,
                        default=20,
                        help='Number of epochs for training')
    parser.add_argument('--classes', '-c', type=int,
                        default=17, choices=range(1, 18),
                        help='Number of classes to train on (value must range from 1 to 17)')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=32,
                        help='Batch size for training')
    parser.add_argument('--train', action='store_true',
                        help='Flag to enable training')
    parser.add_argument('--valid', action='store_true',
                        help='Flag to enable validation')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Flag to enable early stopping')
    parser.add_argument('--model_name', type=str,
                        default=None,
                        help='Model name inside "model/"')

    return parser.parse_args()


def neptune_logger(run_id):
    # Credentials
    logger = neptune.init_run(
        custom_run_id=run_id,
        project="rashid.deutschland/QVPipe",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyZDhmYjkzZC0xMGUwLTRkZjAtOGFjMC1kNGU3NzA3YmQ3ZjkifQ==",
    )
    return logger


def load_dataset(params):
    # Load train dataset
    train_transform = transforms.Compose([
        transforms.Resize((params["resize_x"], params["resize_y"])),
        transforms.RandomAdjustSharpness(1.5),
        transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.25)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_keys_path = params["dataset_root"] + "train_keys.json"
    # Load your custom video dataset using DataLoader
    train_dataset = QVPipeDataset(params["dataset_root"],
                                  params["num_classes"],
                                  params["num_key_frames"],
                                  train_keys_path,
                                  transform=train_transform,
                                  frame_selection_method=params["frame_selection"])
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

    # Load valid dataset
    valid_transform = transforms.Compose([
        transforms.Resize((params["resize_x"], params["resize_y"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_keys_path = params["dataset_root"] + "val_keys.json"
    # Load your custom video dataset using DataLoader
    valid_dataset = QVPipeDataset(params["dataset_root"],
                                  params["num_classes"],
                                  params["num_key_frames"],
                                  valid_keys_path,
                                  transform=valid_transform,
                                  frame_selection_method=params["frame_selection"])
    valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"])

    return train_loader, valid_loader


def get_subset(data_loader, shrink_size=0.2, random_seed=33):
    dataset = data_loader.dataset
    selected_size = int(shrink_size * len(dataset))
    rejected_size = len(dataset) - selected_size

    generator = torch.Generator().manual_seed(random_seed)
    selected_set, rejected_set = random_split(dataset, [selected_size, rejected_size], generator)
    return DataLoader(selected_set, batch_size=data_loader.batch_size)


def train_step(params, train_data_loader, valid_data_loader, model, device, logger):

    #print("Inside train step", [float(model.method_priorities[0]), float(model.method_priorities[1])])
    # Define loss function and optimizer
    # optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params["learning_rate"],
    #                                                 steps_per_epoch=train_size, epochs=params["num_epochs"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

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
        bar = progressbar.ProgressBar(maxval=len(train_data_loader.dataset),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()

        model.train()
        running_loss = []

        batch_done = 0

        #print([float(model.method_priorities[0]), float(model.method_priorities[1]), float(model.method_priorities[2])])

        #print(model.method_priorities[0])
        #print(model.method_priorities[1])

        for inputs, labels in train_data_loader:
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            curr_batch_len = len(inputs)

            # inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            # print("Getting outputs from model...")
            outputs = model(inputs)
            # print("Output from model : ", outputs)
            # print("Actual output : ", labels)
            loss = loss_function(outputs, labels)
            running_loss.append(loss.item())

            # print("Back Propagating to update weights...")
            loss.backward()
            # print("Weights updated!")
            optimizer.step()

            # Update progress
            bar.update((batch_done * params["batch_size"]) + curr_batch_len)
            batch_done += 1

        epoch_loss = sum(running_loss) / len(running_loss)
        print(f"Epoch Train Loss: {epoch_loss:.4f}")
        logger["epoch/train_loss"].append(epoch_loss)

        #print("After epoch", [float(model.method_priorities[0]), float(model.method_priorities[1])])

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
                #train_data_loader.dataset.update_method_priorities(model.method_priorities, logger)
                torch.save(model.state_dict(), params["model_path"])
                print("Model saved to ", params["model_path"])
            prev_loss = current_loss
        else:
            #train_data_loader.dataset.update_method_priorities(model.method_priorities, logger)
            torch.save(model.state_dict(), params["model_path"])
            print("Model saved to ", params["model_path"])

        #valid_data_loader.dataset.dataset.update_method_priorities(model.method_priorities, logger)
        epoch_valid_loss = valid_step_on_epoch(params, valid_data_loader, model, device, logger)

        scheduler.step(epoch_valid_loss)

    print("Training complete!")


def valid_step_on_epoch(params, valid_loader, model, device, logger):
    model.eval()
    total_loss = []
    val_loss = 0.0

    print("Validating...")

    # Validation loop
    with torch.no_grad():
        # Add progress bar
        bar = progressbar.ProgressBar(maxval=len(valid_loader.dataset),
                                      widgets=[progressbar.Bar('=', 'Validating...[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()

        meter = torchnet.meter.mAPMeter()  # keep track of mean average precision
        cls_meter = torchnet.meter.APMeter()  # keep track of class-wise average precision

        batch_done = 0

        for inputs, labels in valid_loader:
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Predict labels
            outputs = model(inputs)
            meter.add(outputs, labels)
            cls_meter.add(outputs, labels)

            # Calculate loss
            loss = loss_function(outputs, labels)
            total_loss.append(loss.item())
            val_loss += loss.item() * inputs.size(0)

            # Update progress
            curr_batch_len = len(inputs)
            bar.update((batch_done * params["batch_size"]) + curr_batch_len)
            batch_done += 1

        # epoch_loss = sum(total_loss) / len(total_loss)
        val_loss /= len(valid_loader.dataset)
        print(f"Epoch Valid Loss: {val_loss:.4f}")
        logger["epoch/valid_loss"].append(val_loss)

        # Close progress bar
        bar.finish()

    mean_average_precision = meter.value()
    print(f"Epoch Valid mAP: {mean_average_precision:.4f}")
    logger["epoch/valid_mAP"].append(mean_average_precision.item())

    average_precision = cls_meter.value()
    for idx, cls_ap in enumerate(average_precision):
        print("Epoch Valid AP class %02d:" % idx, cls_ap)
        logger["epoch/val_AP_class_" + str(idx)].append(cls_ap.item())

    return val_loss


def valid_step(params, valid_loader, model, device, logger):
    # Load trained model for evaluation
    model.load_state_dict(torch.load(params["model_path"]))

    # Change model to evaluation mode
    model.eval()

    # Validation loop
    total_loss = []
    #confusion_matrix = {"true_positives": 0,
    #                    "false_positives": 0,
    #                    "false_negatives": 0,
    #                    "true_negatives": 0}
    with torch.no_grad():
        # Add progress bar
        bar = progressbar.ProgressBar(maxval=len(valid_loader.dataset),
                                      widgets=[progressbar.Bar('=', 'Validating...[', ']'), ' ',
                                               progressbar.Percentage()])
        bar.start()

        meter = torchnet.meter.mAPMeter()  # keep track of mean average precision
        cls_meter = torchnet.meter.APMeter()  # keep track of class-wise average precision

        batch_done = 0

        confusion_matrix_17 = np.zeros((17, 17), dtype=int)

        for inputs, labels in valid_loader:
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Predict labels
            outputs = model(inputs)
            meter.add(outputs, labels)
            cls_meter.add(outputs, labels)

            loss = loss_function(outputs, labels)
            logger["batch_loss"].append(loss.item())
            total_loss.append(loss.item())

            #predicted = output_to_binary(outputs)
            predicted = (outputs >= 0.5).float()
            
            #confusion_matrix = update_confusion_matrix(confusion_matrix, predicted, labels)
            confusion_matrix_17 = update_confusion_matrix_17(confusion_matrix_17, predicted, labels)

            # Update progress
            curr_batch_len = len(inputs)
            bar.update((batch_done * params["batch_size"]) + curr_batch_len)
            batch_done += 1

        # Close progress bar
        bar.finish()

    average_loss = sum(total_loss) / len(total_loss)

    #correct_predictions = confusion_matrix["true_positives"] + confusion_matrix["true_negatives"]
    #total_samples = (confusion_matrix["true_positives"] + confusion_matrix["false_positives"] +
    #                 confusion_matrix["false_negatives"] + confusion_matrix["true_negatives"])
    #accuracy = (correct_predictions / total_samples) * 100

    mean_average_precision = meter.value()
    average_precision = cls_meter.value()

    print("Validation Complete!\n--------------------")
    print(f"Validation Loss: {average_loss:.4f}")
    #print(f"Validation Accuracy: {accuracy:.2f}%")
    #print("Confusion Matrix:", confusion_matrix)
    logger["average_loss"] = average_loss
    #logger["accuracy"] = accuracy
    #logger["confusion_matrix"] = confusion_matrix

    print("\nmAP Metrics\n--------------------")
    print("val_mAP", mean_average_precision)
    logger["val_mAP"] = mean_average_precision.item()
    for idx, cls_ap in enumerate(average_precision):
        print("val_AP_%02d" % idx, cls_ap)
        logger["val_AP"].append(cls_ap.item())
    
    confusion_matrix_2 = conv_conf_17_t0_2(confusion_matrix_17)

    # Convert the array to a pandas DataFrame
    df_17 = pd.DataFrame(confusion_matrix_17)
    df_17.to_csv('conf_mat_17.csv', index=True)
    logger['conf_mat_17'].upload('conf_mat_17.csv')
    df_2 = pd.DataFrame(confusion_matrix_2)
    df_2.to_csv('conf_mat_2.csv', index=True)
    logger['conf_mat_2'].upload('conf_mat_2.csv')



def loss_function(out, label):
    # return torchvision.ops.sigmoid_focal_loss(out, label, reduction='mean')
    return binary_cross_entropy_with_logits(out, label.float())

def output_to_binary(out):
    # Find the maximum value in each row
    max_values, max_indices = out.max(dim=1)

    # Initialize the binary tensor with zeros
    binary_tensor = torch.zeros_like(out, dtype=torch.int32)

    # Condition 1: Max value is in the first column
    condition1 = max_indices == 0

    # Set the first column to 1 where the condition is met
    binary_tensor[condition1, 0] = 1

    # Condition 2: Max value is not in the first column
    condition2 = max_indices != 0

    # For rows where the max value is not in the first column,
    # set columns with values greater than the first column value to 1
    greater_than_first_column = out > out[:, [0]]
    binary_tensor[condition2] = greater_than_first_column[condition2].int()

    print("Original Tensor:")
    print(out)
    print("\nBinary Tensor:")
    print(binary_tensor)


def update_confusion_matrix_17(conf_mat_17, predicted, true_labels):
    
    # Convert tensors to numpy arrays
    y_true = true_labels.numpy()
    y_pred = predicted.numpy()

    print(y_pred)
    print(y_true)

    # Populate the confusion matrix
    for actual, predicted in zip(y_true, y_pred):
        for i in range(17):
            conf_mat_17[i, np.where(predicted == 1)[0]] += (actual[i] == 1)

    print(conf_mat_17)

    ## Compute the presence of any defect (excluding the first column which is 'no defect')
    #actual_defect_presence = np.any(y_true[:, 1:], axis=1).astype(int)
    #predicted_defect_presence = np.any(y_pred[:, 1:], axis=1).astype(int)
    ## Compute the 2x2 confusion matrix for no defect vs defect
    #confusion_matrix_2x2 = multilabel_confusion_matrix(actual_defect_presence.reshape(-1, 1), 
    #                                                   predicted_defect_presence.reshape(-1, 1))

    ## Extract and print the 2x2 confusion matrix
    #conf_mat_2 += confusion_matrix_2x2[0]

    #print(conf_mat_2)
    
    #conv_conf_17_t0_2(conf_mat_17)

    return conf_mat_17


def update_confusion_matrix(confusion_matrix, predicted, true_labels):
    confusion_matrix["true_positives"] += torch.sum((true_labels == 1) & (predicted == 1))
    confusion_matrix["false_positives"] += torch.sum((true_labels == 0) & (predicted == 1))
    confusion_matrix["false_negatives"] += torch.sum((true_labels == 1) & (predicted == 0))
    confusion_matrix["true_negatives"] += torch.sum((true_labels == 0) & (predicted == 0))
    return confusion_matrix

def conv_conf_17_t0_2(conf_mat):
    return np.array([[np.sum(conf_mat[0:1, 0:1]), np.sum(conf_mat[0:1, 1:])],
                    [np.sum(conf_mat[1:, 0:1]), np.sum(conf_mat[1:, 1:])]])


if __name__ == '__main__':
    main()
