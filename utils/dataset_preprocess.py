import gc
import multiprocessing
import os
import h5py
import random
import time
import datetime
import natsort
# natsorted() identifies numbers anywhere in a string and sorts them naturally
from log import log_info_time
from image_preprocess import preprocess_Video_RGB_only
# from image_preprocess_Hao import preprocess_Video_RGB_only_Hao
from text_preprocess import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preprocess")
    
    parser.add_argument('--dataset_name', type=str, nargs='+', default="PURE", 
                        choices=["MMSE", "MAHNOB_HCI", "PURE", "UBFC"])
    
    parser.add_argument('--root_dir', type=str, default='3stream_rppg/BaseDataset', 
                        help='Root_direction is saved your original datasets')

    parser.add_argument('--save_dir', type=str, default='3stream_rppg/Dataset', 
                        help='The direction to saved the your reprocessed dataset')

    parser.add_argument('--cv_ratio', type=int, default=0.8, 
                        help='Modify the value to change your training and testing dataset ratio')  
    
    parser.add_argument('--roi_size', type=int, default=36, 
                        help='The parameter for face ROI size')      

    return parser.parse_args()


def preprocessing(args):
    
    dataset_name = args.dataset_name
    save_root_path = args.save_dir
    data_root_path = args.root_dir
    cv_ratio = args.cv_ratio
    img_size = args.roi_size

    manager = multiprocessing.Manager() #Multiprocessing Manager provides a way of creating centralized Python objects that can be shared safely among processes.

    if dataset_name == "PURE":
        dataset_root_path = data_root_path + '/' + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path))]
    elif dataset_name == "c":
        dataset_root_path = data_root_path + '/' + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path)) if data.__contains__("subject")]
        random.shuffle(data_list)
    elif dataset_name == "VIPL":
        dataset_root_path = data_root_path + '/' + dataset_name + '/data'
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path))]
        random.shuffle(data_list)
    elif dataset_name == "MMSE":
        dataset_root_path = data_root_path + '/' + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path))]
        random.shuffle(data_list)
    elif dataset_name == "cohface" or "MAHNOB_HCI":
        dataset_root_path = data_root_path + '/' + dataset_name
        data_list = [data for data in natsort.natsorted(os.listdir(dataset_root_path)) if data.isdigit()]
        random.shuffle(data_list)
    else:
        print('Not supported dataset')
        return
    # chunk_shape = (200, img_size, img_size, 3)

    threads = 6
    for i in np.arange(0, len(data_list), threads):
        if i + threads > len(data_list):
            threads = len(data_list) - i
        process = []
        return_dict = manager.dict()
        for data_path in data_list[i:i+threads]:
            proc = multiprocessing.Process(target=preprocess_dataset, args=(dataset_root_path + "/" + data_path, True,
                                                                            dataset_name, return_dict, img_size))
            process.append(proc)
            proc.start()
        for proc in process:
            proc.join()
            proc.terminate()

        file = h5py.File(save_root_path + dataset_name + ".hdf5", "a")
        for data_path in return_dict.keys():
            dset = file.create_group(data_path)
            video_data = return_dict[data_path]['video']
            label_data = return_dict[data_path]['label']
            video_shape = video_data.shape
            label_shape = label_data.shape
            dset.create_dataset('video', video_shape, np.uint8, video_data, chunks=video_shape)
            dset.create_dataset('label', label_shape, np.float32, label_data, chunks=label_shape)
            # dset['video'] = return_dict[data_path]['video']
            # dset['label'] = return_dict[data_path]['label']
        file.close()
        #
        del process, return_dict
        gc.collect()

    # read the fully datasets
    file = h5py.File(save_root_path + dataset_name + ".hdf5", "r")  
    train_length = int(len(file.keys()) * cv_ratio)
    # create a empty file for saved training sets
    train_file = h5py.File(save_root_path + dataset_name + "_train.hdf5", "w")
    for data_path in list(file.keys())[:train_length]:
        file.copy(file[data_path], train_file, data_path)
    train_file.close()

    # create a empty file for saved test sets
    test_file = h5py.File(save_root_path + dataset_name + "_test.hdf5", "w")
    for data_path in list(file.keys())[train_length:]:
        file.copy(file[data_path], test_file, data_path)
    test_file.close()
    file.close()

    log_info_time("Data Processing Time \t: ", datetime.timedelta(seconds=time.time() - start_time))


def preprocess_dataset(path, flag, dataset_name, return_dict, img_size):
    if dataset_name == 'UBFC':
        rst, video = preprocess_Video_RGB_only(path + "/vid.avi", flag, vid_res=img_size)
        if not rst:
            return
        else:
            label = UBFC_preprocess_Label(path + "/ground_truth.txt", video.shape[0])
            if label is None:
                return
            else:
                return_dict[path.split("/")[-1]] = {'video': video, 'label': label}

    elif dataset_name == 'cohface':
        for i in os.listdir(path):
            rst, video = preprocess_Video_RGB_only(path + '/' + i + '/data.avi', flag, vid_res=img_size)
            if not rst:
                return
            else:
                label = MTTS_cohface_Label(path + '/' + i + '/data.hdf5', video.shape[0])
                if label is None:
                    return
                else:
                    return_dict[path.split("/")[-1] + '_' + i] = {'video': video, 'label': label}

    elif dataset_name == 'VIPL':
        for v in os.listdir(path):
            for source in os.listdir(path + '/' + v):
                if source != 'source4':
                    rst, video = preprocess_Video_RGB_only(path + '/' + v + '/' + source + '/video.avi', flag, vid_res=img_size)
                    if not rst:
                        return
                    else:
                        label = MTTS_VIPL_Label(path + '/' + v + '/' + source + '/wave.csv', video.shape[0])
                        if label is None:
                            return
                        else:
                            return_dict[path.split("/")[-1] + '_' + v + '_' + source] = {'video': video, 'label': label}

    elif dataset_name == 'MMSE':
        for v in os.listdir(path):
            rst, video = preprocess_Video_RGB_only(path + '/' + v + '/video.avi', flag, vid_res=img_size)
            if not rst:
                return
            else:
                label = MTTS_MMSE_Label(path + '/' + v + '/BP_mmHg.txt', video.shape[0])
                if label is None:
                    return
                else:
                    return_dict[path.split("/")[-1] + '_' + v] = {'video': video, 'label': label}

    elif dataset_name == 'MAHNOB_HCI':
        rst, video = preprocess_Video_RGB_only(path + "/vid.avi", flag, vid_res=img_size)
        if not rst:
            return
        else:
            label = MTTS_MANHOB_Label(path + "/ground_truth.txt", video.shape[0])
            if label is None:
                return
            else:
                return_dict[path.split("/")[-1]] = {'video': video, 'label': label}

    if dataset_name == 'PURE':
        rst, video = preprocess_Video_RGB_only(path + "/" + str(path.split("/")[-1]) + ".avi", flag, vid_res=img_size)
        if not rst:
            return
        else:
            label = PURE_preprocess_label(path + "/" + str(path.split("/")[-1]) + ".json", video.shape[0])
            if label is None:
                return
            else:
                return_dict[path.split("/")[-1]] = {'video': video, 'label': label}

    del video, label, rst
    gc.collect()


if __name__ == '__main__':
    start_time = time.time()
    print("Hello")
    multiprocessing.set_start_method('forkserver')
    args = parse_args()
    preprocessing(args)