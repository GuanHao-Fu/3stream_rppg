import h5py
import math
import random
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset K-fold partition process")
    
    # Adding the arguments with default values
    parser.add_argument('--kfold', type=int, default=5, 
                        help='The number of fold in k-fold prptocol')
    
    parser.add_argument('--dataset_name', type=str, nargs='+', default="PURE", 
                        choices=["MMSE", "MAHNOB_HCI", "PURE", "UBFC"])
    
    parser.add_argument('--root_dir', type=str, default='3stream_rppg/Dataset', 
                        help='The direction is to saved the your reprocessed dataset')

    parser.add_argument('--save_dir', type=str, default='3stream_rppg/Dataset/D_kfold', 
                        help='The direction to save your dataset splitted by kfold protocol')

    parser.add_argument('--mode', type=str, nargs='+',default='Random', 
                        choices=["Random", "Sequential"],
                        help='Choose the method for splitting the dataset: "Random" for random split, or "Sequential" for sequential split.')  

    return parser.parse_args()

def main(args):
    k = args.kfold
    dataset_name = args.dataset_name
    root_path = args.root_dir       
    save_path = args.save_dir
    mode = args.mode
    file = h5py.File(root_path + dataset_name + ".hdf5", "r")
    print(f"file.keys() = {file.keys()}")
    print(f"Total_len={len(file.keys())}")
    #------------------------------------------------------------------------------------#
    if dataset_name == "PURE":
        #---PURE Dataset---#
        videos = list(file.keys())
        extracted_numbers = []
        extracted_numbers_set = []
        # ---Extract the first two numbers from each key and add them to the list---#
        for key in videos:
            numbers = key.split('-')[0]
            extracted_numbers.append(''.join(numbers))
        if mode=='Random':
            extracted_numbers_set = list(set(extracted_numbers))     
        else:
            extracted_numbers_set = sorted(set(extracted_numbers))
        # print(f"extracted_numbers_set = {extracted_numbers_set}")
        segment = len(extracted_numbers_set) // k
        # print(f"segment = {segment}")
        unseen_videos = [[] for _ in range(k)]
        for index, i in enumerate(range(0, len(extracted_numbers_set), segment)):
            for key in videos:
                numbers = key.split('-')[0]
                for j in range(0,segment):
                    if numbers == extracted_numbers_set[i+j]:
                        unseen_videos[index].append(key)
    #------------------------------------------------------------------------------------#
    elif dataset_name == "MMSE":
        #---MMSE Dataset---#
        videos = list(file.keys())
        extracted_numbers = []
        extracted_numbers_set = []
        #---Extract the first two numbers from each key and add them to the list---#
        for key in videos:
            numbers = key.split('_')[0]
            extracted_numbers.append(''.join(numbers))
        if mode=='Random':
            extracted_numbers_set = list(set(extracted_numbers))     
        else:
            extracted_numbers_set = sorted(set(extracted_numbers))
        # print(f"extracted_numbers_set = {extracted_numbers_set}")
        # print(f"the length of extracted_numbers_set = {len(extracted_numbers_set)}")
        segment = round(len(extracted_numbers_set) / k)
        # print(f"segment = {segment}")
        unseen_videos = [[] for _ in range(k)]
        for index, i in enumerate(range(0, len(extracted_numbers_set), segment)):
            for key in videos:
                numbers = key.split('_')[0]
                for j in range(0,segment):
                    if i+j < len(extracted_numbers_set) and numbers == extracted_numbers_set[i+j]:
                        unseen_videos[index].append(key)
    #------------------------------------------------------------------------------------#
    elif dataset_name == "UBFC":
        #---UBFC Dataset---#
        videos = list(file.keys())
        extracted_numbers = []
        extracted_numbers_set = []
        #---Extract the first two numbers from each key and add them to the list---#
        for key in videos:
            numbers = key.split('ubject')[1]
            extracted_numbers.append(''.join(numbers))
        if mode=='Random':
            extracted_numbers_set = list(set(extracted_numbers))     
        else:
            extracted_numbers_set = sorted(set(extracted_numbers))
        # print(f"extracted_numbers_set = {extracted_numbers_set}")
        # print(f"the length of extracted_numbers_set = {len(extracted_numbers_set)}")
        segment = round(len(extracted_numbers_set) / k)
        # print(f"segment = {segment}")
        unseen_videos = [[] for _ in range(k)]
        for index, i in enumerate(range(0, len(extracted_numbers_set), segment)):
            for key in videos:
                numbers = key.split('ubject')[1]
                for j in range(0,segment):
                    if i+j < len(extracted_numbers_set) and numbers == extracted_numbers_set[i+j]:
                        unseen_videos[index].append(key)
    #------------------------------------------------------------------------------------#
    elif dataset_name == "MAHNOB_HCI":
        #---MAHNOB_HCI Dataset---#
        videos = list(file.keys())
        random.shuffle(videos)
        #---Extract the first two numbers from each key and add them to the list---#
        if mode=='Random':
            extracted_numbers_set = list(set(extracted_numbers))     
        else:
            extracted_numbers_set = sorted(set(extracted_numbers))
        # print(f"videos = {videos}")
        # print(f"the length of extracted_numbers_set = {len(videos)}")
        segment = round(len(videos) / k)
        # print(f"segment = {segment}")
        unseen_videos = [[] for _ in range(k)]
        for index, i in enumerate(range(0, len(videos), segment)):
            for key in videos:
                for j in range(0,segment):
                    if i+j < len(videos) and key == videos[i+j]:
                        unseen_videos[index].append(key)

    else:
        print("can't identify this kind of dataset")


    for i in range(1, k+1):
        print('----------------------------------')
        print(f"FOLD: {i}")
        if i==k:
            test_ids = unseen_videos[i-1]
            train_ids = unseen_videos[0:(i-1)]
        else:
            test_ids = unseen_videos[i-1]
            train_ids = unseen_videos[0:i-1] + unseen_videos[i:]
        train_ids_reshaped = [item for sublist in train_ids for item in sublist]
        #------------------------------------------------------------------------------------#
        # Test
        print(len(train_ids), "Train_ids: ", train_ids)
        print(len(test_ids), "Test_ids: ", test_ids)
        #------------------------------------------------------------------------------------#
        # Writing the dataset (Be careful to use following code because it might overwrite your dataset)
        # train_file = h5py.File(save_path + dataset_name + f"_train_{i}.hdf5", "w")
        # for data_path in train_ids_reshaped:
        #     file.copy(file[data_path], train_file, data_path)
        # train_file.close()

        # test_file = h5py.File(save_path + dataset_name + f"_test_{i}.hdf5", "w")
        # for data_path in test_ids:
        #     file.copy(file[data_path], test_file, data_path)
        # test_file.close()
        #------------------------------------------------------------------------------------#
    file.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)