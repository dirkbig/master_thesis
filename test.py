rootdir = 'C:/Users/sid/Desktop/test'


for data_folder in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA"):
    if data_folder == 'LOAD':
        print("this is load")
    if data_folder == 'PRODUCTION':
        for production_files in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA/PRODUCTION"):
            print("this is production")
