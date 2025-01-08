from dataset.MTTSDataset import MTTSDataset
from dataset.MTTSDataset_Hao import MTTSDataset_Hao
from dataset.TSDANDataset import TSDANDataset
from dataset.SlowFast_FD_Dataset_TSW import SlowFast_FD_Dataset
from dataset.DeepPhysDataset import DeepPhysDataset
import h5py


def dataset_loader(train, save_root_path, model_name, dataset_name, window_length, fold = None, sliding_window=None,test_sliding_window = None):
    if type(dataset_name) == list:
        train_file_paths = []  # Anh was here
        valid_file_paths = []
        
        if fold is not None:
            if train==0 or train==1:
                for i in dataset_name:
                    train_file_path = save_root_path + '/' + i + f"_train_{fold}.hdf5"
                    train_file_paths.append(train_file_path)
                    
                    valid_file_path = save_root_path + '/' + i + f"_test_{fold}.hdf5"
                    valid_file_paths.append(valid_file_path)
            else:
                test_file_path = save_root_path + '/' + dataset_name[0] + "_test_" + str(fold) + ".hdf5"     # 5fold test
                print(f"test_file_path={test_file_path}")
        else:
            for i in dataset_name:
                train_file_path = save_root_path + i + "_train.hdf5"
                train_file_paths.append(train_file_path)

                valid_file_path = save_root_path + i + "_test.hdf5"
                valid_file_paths.append(valid_file_path)

        train_files = []
        valid_files = []
        test_files = []

        for i in train_file_paths:
            train_file = h5py.File(i, 'r')
            train_files.append(train_file)
        for i in valid_file_paths:
            valid_file = h5py.File(i, 'r')
            valid_files.append(valid_file)


        # overlapping_Anh  # if you wanna try Test_overlapping please uncomment this program
        if test_sliding_window is not None:
            test_file = h5py.File(test_file_path, 'r')  
            test_files.append(test_file)
        print("train_file", train_files)

    else:
        if fold is None:
            all_file_path = save_root_path + dataset_name + "_test.hdf5"             # cross dataset
            all_file = h5py.File(all_file_path, 'r')
            print("test_file", all_file)
        else:
            if train==0 or train==1:
                train_file_path = save_root_path + dataset_name + f"_train_{fold}.hdf5"
                train_file = h5py.File(train_file_path, 'r')
                valid_file_path = save_root_path + dataset_name + f"_test_{fold}.hdf5"
                valid_file = h5py.File(valid_file_path, 'r')
            else:
                test_file_path = save_root_path + dataset_name + "_test_" + str(fold) + ".hdf5"  #nonoverlapping_Dao
                test_file = h5py.File(test_file_path, 'r')
                # print("test_file", test_file)


        print("test_file_path = {}".format(train_file_path))
    
    if train == 0 or train == 1:
        train = True
        if model_name in ['MTTS', 'MTTS_CSTM', 'TSDAN']:
            train_set = MTTSDataset_Hao(train_files, dataset_name, window_length, False)   # overlapping_Anh
            valid_set = MTTSDataset_Hao(valid_files, dataset_name, window_length, True)
            # train_set = MTTSDataset_Hao(train_file, dataset_name, window_length, True)   # non_overlapping_Dao
            # valid_set = MTTSDataset_Hao(valid_file, dataset_name, window_length, False)
        elif model_name == "SlowFast_FD":
            train_set = SlowFast_FD_Dataset(train_files, dataset_name, window_length, True, sliding_window)
            valid_set = SlowFast_FD_Dataset(valid_files, dataset_name, window_length, False,sliding_window)
            # train_set = SlowFast_FD_Dataset(train_files, dataset_name, window_length, True)
            # valid_set = SlowFast_FD_Dataset(valid_files, dataset_name, window_length, False)
        elif model_name == "DeepPhys":
            train_set = DeepPhysDataset(train_files, window_length)
            valid_set = DeepPhysDataset(valid_files, window_length)
        else:
            raise Exception("Model name is not correct or model is not supported!")
        return train_set, valid_set
    elif train == 2:
        if model_name in ['MTTS', 'MTTS_CSTM', 'TSDAN']:
            # test_set = MTTSDataset(all_file, dataset_name, window_length, True)           
            if type(dataset_name) == list:
                if fold is None:
                    test_set = MTTSDataset_Hao(valid_files, dataset_name, window_length, True)    #  overlapping_Hao non_FiveFold      
                else:
                    test_set = MTTSDataset_TSW(test_files, dataset_name, window_length, True, sliding_window)    # overlapping_Hao   FiveFold   
            else:
                test_set = MTTSDataset_Hao(test_file, dataset_name, window_length, True)   # nonoverlapping_Dao  FiveFold 
            
        elif model_name == "SlowFast_FD":
            # test_set = SlowFast_FD_Dataset(all_file, dataset_name, window_length, False)    
            # test_set = SlowFast_FD_Dataset(test_file, dataset_name, window_length, False)   # nonoverlapping
            test_set = SlowFast_FD_Dataset(test_files, dataset_name, window_length, False, test_sliding_window)       
            # test_set = SlowFast_FD_Dataset(test_files, dataset_name, window_length, False)   # overlapping              
        elif model_name == "DeepPhys":
            # test_set = DeepPhysDataset(all_file, window_length)
            test_set = DeepPhysDataset(test_file, window_length)
        else:
            raise Exception("Model name is not correct or model is not supported!")
        return test_set
