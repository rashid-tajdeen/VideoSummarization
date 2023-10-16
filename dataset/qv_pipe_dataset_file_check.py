import json
import os.path

file_json = open('qv_pipe_dataset/qv_pipe_train.json')
all_data = json.load(file_json)

files_verified = []
files_missing = []

for i in all_data:
    file_check = os.path.isfile("qv_pipe_dataset/track1_raw_video/" + i)
    # print(i, ":", "File Exists" if file_check else "File Does Not Exist")
    if file_check:
        files_verified += [i]
    else:
        files_missing += [i]

print("\n--------------------------------")
if len(files_missing):
    print("No.of Files Verified :", len(files_verified))
    print("No.of Files Missing :", len(files_missing))
    print("Files Missing :", files_missing)
else:
    print("No.of Files Verified :", len(files_verified))
    print("No.of Files Missing :", len(files_missing))
print("--------------------------------")

file_json.close()

with open('qv_pipe_dataset/train_keys.json') as f:
    data = json.load(f)
    train_keys_count = 0
    for i in data:
        train_keys_count += 1
    print("train_keys_count :", train_keys_count)

with open('qv_pipe_dataset/val_keys.json') as f:
    data = json.load(f)
    val_keys_count = 0
    for i in data:
        val_keys_count += 1
    print("val_keys_count :", val_keys_count)
