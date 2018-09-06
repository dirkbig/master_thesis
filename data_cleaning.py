import os
from functions.function_file import *

N_same_length = 10
# sudo find /Users/dirkvandenbiggelaar/Desktop/DATA/PRODUCTION -name ".DS_Store" -depth -exec rm {} \; removes all DS_Store files
sudoPassword = 'youwouldliketoknowrighttttt?'
command = 'sudo find /Users/dirkvandenbiggelaar/Desktop/DATA/PRODUCTION -name ".DS_Store" -depth -exec rm {} \;'
os.system('echo %s|sudo -S %s' % (sudoPassword, command))

path_to_dirs = "/Users/dirkvandenbiggelaar/Desktop/DATA/PRODUCTION"

lengths = []
good_lengths = []

with open('/Users/dirkvandenbiggelaar/Desktop/DATA/list_of_prod_folders.csv', 'w', newline='') as list_of_prod_folders:
    dir_writer = csv.writer(list_of_prod_folders)
    for dirs in os.walk(path_to_dirs):
        for dir_num in range(len(dirs[1])):
            print(dirs[1][dir_num])
            path_to_files = path_to_dirs + '/' + dirs[1][dir_num]
            for file in os.walk(path_to_files):
                num_good_lengths = 0
                for file_num in range(len(file[2])):
                    path_to_file = path_to_files + '/' + file[2][file_num]
                    length = check_file(path_to_files + "/" + file[2][file_num])
                    lengths.append(length)
                    if length == 991:
                        num_good_lengths += 1
                        good_lengths.append(path_to_file)
                if num_good_lengths > N_same_length:
                    dir_writer.writerow([dirs[1][dir_num]])
                    # print("production file %s has %d entries" % (file[2][file_num], length))

# print(len(good_lengths))
# print(min(lengths))

with open('/Users/dirkvandenbiggelaar/Desktop/DATA/useable_data.csv', 'w', newline='') as useable_data:
    writer = csv.writer(useable_data)
    for i in range(len(good_lengths)):
        writer.writerow([good_lengths[i]])

