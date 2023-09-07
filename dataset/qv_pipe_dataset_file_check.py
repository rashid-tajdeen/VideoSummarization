import json
import os.path

file_json = open('qv_pipe_dataset/qv_pipe_train.json')
data = json.load(file_json)

files_verified = []
files_missing = []

for i in data:
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
