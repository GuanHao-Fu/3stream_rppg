from dataset.MTTSDataset import MTTSDataset
from dataset.TSDANDataset import TSDANDataset
from dataset.HSlowFast_FD_Dataset import SlowFast_FD_Dataset
from dataset.SlowFast_AM_Dataset import SlowFast_AM_Dataset
from dataset.DeepPhysDataset import DeepPhysDataset
import h5py

# Anh
def dataset_loader(train, save_root_path, model_name, dataset_name, window_length, fold = None):
    if fold is not None:
        train_file_path = save_root_path + dataset_name + f"_train_{fold}.hdf5"
        # train_file_path = save_root_path + "MANHOB_HCI.hdf5"
        valid_file_path = save_root_path + dataset_name + f"_test_{fold}.hdf5"
    else:
        train_file_path = save_root_path + dataset_name + "_train.hdf5"
        # train_file_path = save_root_path + "MANHOB_HCI.hdf5"
        valid_file_path = save_root_path + dataset_name + "_test.hdf5"
    test_file_path = save_root_path + dataset_name + "_test.hdf5"

    # test_file_path = save_root_path + "MMSE.hdf5"
    train_file = h5py.File(train_file_path, 'r')
    valid_file = h5py.File(valid_file_path, 'r')
    test_file = h5py.File(test_file_path, 'r') # all the dataset for cross-dataset testing  -  Anh

    print("train_file", train_file)

    if train == 0 or train == 1:
        train = True
        if model_name in ['MTTS', 'MTTS_CSTM', 'TSDAN']:
            train_set = MTTSDataset(train_file, dataset_name, window_length, False)
            valid_set = MTTSDataset(valid_file, dataset_name, window_length, True)
        elif model_name == "SlowFast_FD":
            train_set = SlowFast_FD_Dataset(train_file, dataset_name, window_length, True)
            valid_set = SlowFast_FD_Dataset(valid_file, dataset_name, window_length, False)
        elif model_name == "SlowFast_AM":
            train_set = SlowFast_AM_Dataset(train_file, dataset_name, window_length, True)
            valid_set = SlowFast_AM_Dataset(valid_file, dataset_name, window_length, False)
        elif model_name == "DeepPhys":
            train_set = DeepPhysDataset(train_file, window_length)
            valid_set = DeepPhysDataset(valid_file, window_length)
        else:
            raise Exception("Model name is not correct or model is not supported!")
        return train_set, valid_set
    elif train == 2:
        if model_name in ['MTTS', 'MTTS_CSTM', 'TSDAN']:
            test_set = MTTSDataset(test_file, dataset_name, window_length, True)
        elif model_name == "SlowFast_FD":
            test_set = SlowFast_FD_Dataset(test_file, dataset_name, window_length, False)
        elif model_name == "DeepPhys":
            test_set = DeepPhysDataset(test_file, window_length)
        else:
            raise Exception("Model name is not correct or model is not supported!")
        return test_set