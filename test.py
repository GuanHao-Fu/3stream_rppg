import copy
import datetime
import time
import numpy as np
import h5py
import math
import torch
import optim
from utils.loss import loss_fn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from log import log_info_time
from torch.optim import lr_scheduler
from utils.funcs import plot_graph, plot_loss_graph, BPF_dict, normalize
from utils.eval_metrics import *
import matplotlib.pyplot as plt
from torchvision import utils
from statistics import mean
import argparse


# ------------model for different input: 36*36 or 72*72 ------------ #

from models36 import is_model_support, get_model
from dataset.dataset_loader36 import dataset_loader
# from models72 import is_model_support, get_model
# from dataset.dataset_loader72 import dataset_loader

# ------------------------------------------------------------------ #



def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))

def parse_args():
    parser = argparse.ArgumentParser(description="Inference your trained model")
    
    # Adding the arguments with default values
    parser.add_argument('--Dataset_path', type=str, default="/Dataset/")

    parser.add_argument('--checkpoint_path', type=str, default="/checkpoint/")

    parser.add_argument('--model_name', type=str, default="SlowFast_FD",
                        help="MTTS, TSDAN, MTTS_CSTM, SlowFast_FD, SlowFast_AM")
    
    parser.add_argument('--checkpoint_name', type=str, default="")

    parser.add_argument('--dataset_name', type=list, nargs='+', default=["PURE"], 
                        choices=[["MMSE"], ["MAHNOB_HCI"], ["PURE"], ["UBFC"]])
    
    parser.add_argument('--fs', type=int, nargs='+', default=30, 
                        choices=[25,30,61],help =" MMSE:25 PURE,UBFC:30 MANHOB:61")
    
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--loss_metric', type=str, default="combined_loss", 
                        help='combined_loss, snr, mse')

    parser.add_argument('--optimi', type=str, default="ada_delta")

    parser.add_argument('--learning_rate', type=float, default=0.05)

    parser.add_argument('--lr_step', type=int, default=5, 
                        help='Step size for learning rate decay')
    
    parser.add_argument('--lr_decay_rate', type=float, default=0.7, 
                        help='Learning rate decay rate ')
    
    parser.add_argument('--epochs', type=int, default=25, 
                        help='Total number of epochs')
    
    parser.add_argument('--skip_connection', type=bool, default=True, 
                        help='Whether to use skip connections (default: True)')
    
    parser.add_argument('--new_group_tsm', type=bool, default=False, 
                        help='Whether to use a new group for TSM (default: False)')
    
    parser.add_argument('--shift_factor', type=float, default=0.5, 
                        help='Shift factor for temporal shift modules (default: 0.5)')
    
    parser.add_argument('--window_length', type=int, default=10, 
                        help='Window length of dataset, Noted that MAHNOB is 20, others are 10')
    
    parser.add_argument('--k_fold', type=int, default=5, 
                        help='Number of folds for cross-validation (default: 5)')
    
    parser.add_argument('--Test_SW', type=int, default=1, 
                        help='Flag to control if training with Sliding Windows (default: 1)')
    
    parser.add_argument('--dataset_dir', type=str, default='3stream_rppg/Dataset/D_kfold', 
                        help='The direction saved your kfold dataset')
    
    parser.add_argument('--checkpoint_dir', type=str, default='3stream_rppg/checkpoint', 
                        help='The direction saved your model')
    
    return parser.parse_args()

def main(args,TT):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    '''Setting up'''
    __TIME__ = True

    model_name = args.model_name
    dataset_name = args.dataset_name
    fs = args.fs
    batch_size = args.batch_size
    loss_metric = args.loss_metric
    optimizer = args.optimi
    learning_rate = args.learning_rate
    tot_epochs = args.epochs
    window_length = args.window_length 

    skip_connection = args.skip_connection
    new_group_tsm = args.new_group_tsm
    shift_factor = args.shift_factor
    learning_step = args.lr_step
    lr_decay_rate = args.lr_decay_rate
    Test_SW = args.Test_SW

    model_list = ["MTTS", "TSDAN", "MTTS_CSTM", "SlowFast_FD", "SlowFast_AM"]

    checkpoint_name = model_name+"_" + dataset_name[0] + "_T_"+ str(window_length) + "_shift_"+str(shift_factor)+"_" + loss_metric +"lrstep_"+str(learning_step)+"_"+str(lr_decay_rate)+ "_best_model_" + str(TT) + ".pth" 


    Dataset_path = args.dataset_dir
    checkpoint_path = args.checkpoint_dir

    if __TIME__:
        start_time = time.time()

    # train_dataset, valid_dataset = dataset_loader(train, save_root_path, model_name, dataset_name, window_length)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                           num_workers=6, pin_memory=True, drop_last=False)

    # validation_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=SequentialSampler(valid_dataset),
    #                                num_workers=6, pin_memory=True, drop_last=False)
    
    test_dataset = dataset_loader(2, Dataset_path, model_name, dataset_name, window_length, fold=TT, test_sliding_window=Test_SW)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                   num_workers=6, pin_memory=True, drop_last=False)

    app_mean = []
    app_std = []
    motion_mean = []
    motion_std = []

    # with tqdm(total=len(train_dataset) + len(valid_dataset), position=0, leave=True,
    #           desc='Calculating population statistics') as pbar:
    #     for data in train_loader:
    #         if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
    #             data = data[0]  # -> (Batch, 2, T, H, W, 3)
    #             motion_data, app_data = torch.tensor_split(data, 2, dim=1)
    #             B, one, T, C, H, W = motion_data.shape

    #             motion_data = motion_data.view(B*one, T, C, H, W)
    #             app_data = app_data.view(B*one, T, C, H, W)
    #             motion_data = motion_data.reshape(B*T, C, H, W)
    #             app_data = app_data.reshape(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name == 'SlowFast_FD':
    #             motion_data = data[0][0]
    #             app_data = data[0][2]
    #             B, T, C, H, W = motion_data.shape
    #             motion_data = motion_data.view(B*T, C, H, W)
    #             app_data = app_data.view(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name in ['STM_Phys', 'New']:
    #             data = data[0].numpy()  # B, T+1, H, W, C
    #             if window_length == 10:
    #                 data = data[:, :-1, :, :, :]
    #             else:
    #                 data = data[:, :-2, :, :, :]
    #             B, T, C, H, W = data.shape
    #             data = np.reshape(data, (B*T, C, H, W))
    #             batch_app_mean = np.mean(data, axis=(0, 2, 3))
    #             batch_app_std = np.std(data, axis=(0, 2, 3))
    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)

    #         pbar.update(B)

    #     for i, data in enumerate(validation_loader):
    #         if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
    #             data = data[0]  # shape (Batch, T+1, H, W, 6)
    #             motion_data, app_data = torch.tensor_split(data, 2, dim=1)
    #             B, one, T, C, H, W = motion_data.shape

    #             motion_data = motion_data.view(B*one, T, C, H, W)
    #             app_data = app_data.view(B*one, T, C, H, W)
    #             motion_data = motion_data.reshape(B*T, C, H, W)
    #             app_data = app_data.reshape(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name == 'SlowFast_FD':
    #             motion_data = data[0][0]
    #             app_data = data[0][2]
    #             B, T, C, H, W = motion_data.shape
    #             motion_data = motion_data.view(B*T, C, H, W)
    #             app_data = app_data.view(B*T, C, H, W)

    #             batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
    #             batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
    #             batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)
    #             motion_mean.append(batch_motion_mean)
    #             motion_std.append(batch_motion_std)

    #         elif model_name in ['STM_Phys', 'New']:
    #             data = data[0].numpy()  # B, T+1, H, W, C
    #             if window_length == 10:
    #                 data = data[:, :-1, :, :, :]
    #             else:
    #                 data = data[:, :-2, :, :, :]
    #             B, T, C, H, W = data.shape
    #             data = np.reshape(data, (B * T, C, H, W))

    #             batch_app_mean = np.mean(data, axis=(0, 2, 3))
    #             batch_app_std = np.std(data, axis=(0, 2, 3))

    #             app_mean.append(batch_app_mean)
    #             app_std.append(batch_app_std)

    #         pbar.update(B)
    #     pbar.close()

    with tqdm(total=len(test_dataset), position=0, leave=True,
              desc='Calculating population statistics') as pbar:

        for data in test_loader:
            if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM']:
                data = data[0]  # -> (Batch, 2, T, H, W, 3)
                motion_data, app_data = torch.tensor_split(data, 2, dim=1)
                B, one, T, C, H, W = motion_data.shape

                motion_data = motion_data.view(B*one, T, C, H, W)
                app_data = app_data.view(B*one, T, C, H, W)
                motion_data = motion_data.reshape(B*T, C, H, W)
                app_data = app_data.reshape(B*T, C, H, W)

                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                app_mean.append(batch_app_mean)
                app_std.append(batch_app_std)
                motion_mean.append(batch_motion_mean)
                motion_std.append(batch_motion_std)

            elif model_name == 'SlowFast_FD':
                motion_data = data[0][0]
                app_data = data[0][2]
                B, T, C, H, W = motion_data.shape
                motion_data = motion_data.view(B*T, C, H, W)
                app_data = app_data.view(B*T, C, H, W)

                batch_motion_mean = torch.mean(motion_data, dim=(0, 2, 3)).tolist()
                batch_motion_std = torch.std(motion_data, dim=(0, 2, 3)).tolist()
                batch_app_mean = torch.mean(app_data, dim=(0, 2, 3)).tolist()
                batch_app_std = torch.std(app_data, dim=(0, 2, 3)).tolist()

                app_mean.append(batch_app_mean)
                app_std.append(batch_app_std)
                motion_mean.append(batch_motion_mean)
                motion_std.append(batch_motion_std)

            elif model_name in ['STM_Phys', 'New']:
                data = data[0].numpy()  # B, T+1, H, W, C
                if window_length == 10:
                    data = data[:, :-1, :, :, :]
                else:
                    data = data[:, :-2, :, :, :]
                B, T, C, H, W = data.shape
                data = np.reshape(data, (B*T, C, H, W))
                batch_app_mean = np.mean(data, axis=(0, 2, 3))
                batch_app_std = np.std(data, axis=(0, 2, 3))
                app_mean.append(batch_app_mean)
                app_std.append(batch_app_std)

            pbar.update(B)

        pbar.close()

    if model_name in ['TSDAN', 'MTTS', 'MTTS_CSTM', 'SlowFast_FD']:
        # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
        app_mean = np.array(app_mean).mean(axis=0) / 255
        app_std = np.array(app_std).mean(axis=0) / 255
        motion_mean = np.array(motion_mean).mean(axis=0) / 255
        motion_std = np.array(motion_std).mean(axis=0) / 255
        pop_mean = np.stack((app_mean, motion_mean))  # 0 is app, 1 is motion
        pop_std = np.stack((app_std, motion_std))

    elif model_name in ['STM_Phys', 'New']:
        pop_mean = np.array(app_mean).mean(axis=0) / 255
        pop_std = np.array(app_std).mean(axis=0) / 255

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_model_support(model_name, model_list)

    model = get_model(model_name, pop_mean, pop_std, frame_depth=window_length, skip=skip_connection,
                      shift_factor=shift_factor, group_on=new_group_tsm)
    model.to(device)

    criterion = loss_fn(loss_metric)
    optimizer = optim.optimizer(model.parameters(), learning_rate, optimizer)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, lr_decay=0.5)

    min_val_loss = 10000
    min_val_loss_model = None
    torch.backends.cudnn.benchmark = True

    train_loss = []
    valid_loss = []

    if __TIME__:
        log_info_time("Preprocessing time \t: ", datetime.timedelta(seconds=time.time() - start_time))

    if __TIME__:
        start_time = time.time()
    
        
    checkpoint = torch.load(checkpoint_path + "/" + checkpoint_name)
    model.load_state_dict(checkpoint["model"])
    epoch = tot_epochs
    optimizer.load_state_dict(checkpoint["optimizer"])
    # scheduler.load_state_dict(checkpoint['scheduler'])
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    epoch =checkpoint["epoch"]
    print(f"the min epoch:{epoch}")
    min_val_loss = valid_loss[-1]
    min_val_loss_model = copy.deepcopy(model)

    with tqdm(test_loader, desc="Validation ", total=len(test_loader), colour='green') as tepoch:
        model.eval()
        running_loss = 0.0

        inference_array = []
        target_array = []
        inference_array_avg=[]
        target_array_avg=[]

        with torch.no_grad():
            # count=0
            # total_length = len(test_loader)
            # targets_10, outputs_10 = {}, {}
            for inputs, target ,frame_fs in tepoch:             # because overlapping
                tepoch.set_description(f"Test")
                if torch.isnan(target).any():
                    print('A')
                    return
                if torch.isinf(target).any():
                    print('B')
                    return

                if model_name == 'SlowFast_FD':
                    inputs = [_inputs.cuda() for _inputs in inputs]
                else:
                    inputs = inputs.to(device)
                target = target.to(device)

                outputs = model(inputs)
                if torch.isnan(outputs).any():
                    print('A')
                    return
                if torch.isinf(outputs).any():
                    print('B')
                    return

                if loss_metric =="snr":
                    loss = criterion(outputs, target, fs) #   SNR
                else:
                    loss = criterion(outputs, target) #   combined_loss

                running_loss += loss.item() * target.size(0) * target.size(1)
                tepoch.set_postfix(loss='%.6f' % (running_loss / len(test_loader) / window_length /batch_size))
                inference_array = np.append(inference_array, np.reshape(outputs.cpu().detach().numpy(), (1, -1)))
                target_array = np.append(target_array, np.reshape(target.cpu().detach().numpy(), (1, -1)))

                # targets_10[count]=target
                # outputs_10[count]=outputs
                # count+=1
                # if count==10:
                #     target=mean(targets_10)
                #     outputs = mean(outputs_10)
                #     inference_array = np.append(inference_array, np.reshape(outputs.cpu().detach().numpy(), (1, -1)))
                #     target_array = np.append(target_array, np.reshape(target.cpu().detach().numpy(), (1, -1)))
                #     count=0
                #     if total_length-tepoch < 10:
                #         tepoch+=total_length-tepoch
            valid_loss.append(running_loss / len(test_loader) / window_length /batch_size)
            
    Step = window_length//Test_SW
    inference_len = len(inference_array)
    WL =window_length
    # print(f"len(target_array)={len(target_array)}")
    # for i in range(0,150,10):
    #     # print(f"{i} ~ {i+10} of inference array is {target_array[i:i+10]}")
    #     print(f"{i} ~ {i+10} of target array is {target_array[-150+i:-140+i]}")

    # print(f"inference_len={inference_len}")

    # the begining 
    for i in range(0, WL, 1):
        sum_value_inf = []
        sum_value_tar = []
        # print(f"i={i}")
        for index in range(0,i//Test_SW+1,1):
            sum_value_inf.append(inference_array[(index*(WL-Test_SW))+i])
            sum_value_tar.append(target_array[(index*(WL-Test_SW))+i])
        inference_array_avg.append(np.mean(sum_value_inf))
        target_array_avg.append(np.mean(sum_value_tar))                
        # for index, j in enumerate(range(i,-1,-1)):
        #     # print(f"index_j={index},j={j}, index={(index*10)+j}")
        #     sum_value_inf = sum_value_inf + inference_array[(index*10)+j]
        #     sum_value_tar = sum_value_tar + target_array[(index*10)+j]
        # inference_array_avg.append(sum_value_inf/(i+1))
        # target_array_avg.append(sum_value_tar/(i+1))

    # the middle 
    # print(f"inference_len in beginning={len(inference_array_avg)}")
    for i in range((2*WL)-Test_SW, inference_len-((Step-1)*WL), WL):
        for j in range(0, Test_SW, 1):
            sum_value_inf = 0
            sum_value_tar = 0
            for index in range(0,Step,1):
                sum_value_inf = sum_value_inf + inference_array[(index*(WL-Test_SW))+i+j]
                # sum_value_inf = sum_value_inf + inference_array[(index*10)+j]
                sum_value_tar = sum_value_tar + target_array[(index*(WL-Test_SW))+i+j]
            inference_array_avg.append(sum_value_inf/Step)
            target_array_avg.append(sum_value_tar/Step)

    # the end
    for i in range(-WL+Test_SW, 0):
            sum_value_inf = 0
            sum_value_tar = 0
            # print(f"i={i}")
            # print(f"target_array[{i}] = {target_array[i]}")
            avg_number = math.ceil(abs(i)/Test_SW)
            for index in range(0,avg_number,1):
                # print(f"target_array[i-(index*(10-Test_SW))] = {target_array[i-(index*(10-Test_SW))]}")
                position = i-(index*(WL-Test_SW))
                sum_value_inf = sum_value_inf + inference_array[position]
                sum_value_tar = sum_value_tar + target_array[position]              
            inference_array_avg.append(sum_value_inf/avg_number)
            target_array_avg.append(sum_value_tar/avg_number)    

            # for index,i in enumerate(range(inference_len-100, inference_len, 10)):
            #     sum_value_inf = 0
            #     sum_value_tar = 0
            #     for j in range(0,10-index,1):
            #         sum_value_inf = sum_value_inf + inference_array[(index*9)+i]
            #         # sum_value_inf = sum_value_inf + inference_array[(index*10)+j]
            #         sum_value_tar = sum_value_tar + target_array[(index*9)+i]
            #     inference_array_avg.append(sum_value_inf/(10-index))
            #     target_array_avg.append(sum_value_tar/(10-index))


            # print(f"inference_len in the end={len(inference_array_avg)}")
            # # print(f"the last inference_array: {inference_array_avg[-20:-1]}")
            # print(f"The first of Original target_array: {target_array[0: 20]}")
            # print(f"the first target_array: {target_array_avg[0: 20]}")
            # print(f"The last of Original target_array: {target_array[inference_len-20: inference_len]}")
            # print(f"the last target_array: {target_array_avg[inference_len-20: inference_len]}")
    
    result = {}
    groundtruth = {}
    start_idx = 0
    n_frames_per_video = test_dataset.n_frames_per_video   # show the total frame of each video 
    # n_label_per_video = test_dataset.label_per_video     # show the label of each video
    print(f"n_frames_per_video={n_frames_per_video}")
    # print(f"n_label_per_video={n_label_per_video}")
    for i, value in enumerate(n_frames_per_video):
        # if dataset_name == 'PURE':
        #     result[i] = inference_array[start_idx:start_idx + value]
        # else:
        result[i] = normalize(inference_array_avg[start_idx:start_idx + value])
        groundtruth[i] = target_array_avg[start_idx:start_idx + value]
        start_idx += value

    # plot_loss_graph(train_loss, valid_loss)
    # plot_graph(0, 500, groundtruth[3], result[3])
    result = BPF_dict(result, fs)
    groundtruth = BPF_dict(groundtruth, fs)
    # print(len(groundtruth[3]))
    # plot_graph(1400, 1920, groundtruth[3], result[3])
    # plot_graph(0, 500, groundtruth[1], result[1])
    # np.savetxt('over.txt',result[3],delimiter=',')
    # np.savetxt('gover.txt',groundtruth[3],delimiter=',')


    # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
    # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
    # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

    mae10, rmse10, acc3, acc5, acc10 = HR_Metric(groundtruth, result, fs, 10, 1)
    print('MAE 10s: ' + str(round(mae10, 3)))
    print('RMSE 10s: ' + str(round(rmse10, 3)))
    
    # Comment following code by Hao

    mae20, rmse20, acc3, acc5, acc10c = HR_Metric(groundtruth, result, fs, 20, 1)
    print('MAE 20s: ' + str(round(mae20, 3)))
    print('RMSE 20s: ' + str(round(rmse20, 3)))

    # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
    # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
    # print('Accuracy 10 30s: ' + str(round(acc10, 3)))

    mae30, rmse30, acc3, acc5, acc10 = HR_Metric(groundtruth, result, fs, 30, 1)   # for pure/ubfc
    # mae, rmse, acc3, acc5, acc10 = HR_Metric(groundtruth, result, fs, 20, 1)  # for mmse
    pearson = Pearson_Corr(groundtruth, result)
    print('MAE 30s: ' + str(round(mae30, 3)))
    print('RMSE 30s: ' + str(round(rmse30, 3)))
    print('Pearson 30s: ' + str(round(pearson, 3)))
    
    # print('Accuracy 3 30s: ' + str(round(acc3, 3)))
    # print('Accuracy 5 30s: ' + str(round(acc5, 3)))
    # print('Accuracy 10 30s: ' + str(round(acc10, 3)))
    if __TIME__:
        log_info_time("Total time \t: ", datetime.timedelta(seconds=time.time() - start_time))
    return [mae10,rmse10,mae20,rmse20,mae30,rmse30,pearson]

if __name__ == '__main__':
    args = parse_args()
    print("***********************************************************************************************************")
    result = []
    
    for i in range(1,6):
        fold_result = main(args,i)
        result.append(fold_result)
    k_fold_avg = np.mean(np.transpose(result), axis=1)
    print(k_fold_avg)
    print("fold result:")
    print('MAE 10s: ' + str(round(k_fold_avg[0], 3)))
    print('RMSE 10s: ' + str(round(k_fold_avg[1], 3)))
    print('MAE 20s: ' + str(round(k_fold_avg[2], 3)))
    print('RMSE 20s: ' + str(round(k_fold_avg[3], 3)))
    print('MAE 30s: ' + str(round(k_fold_avg[4], 3)))
    print('RMSE 30s: ' + str(round(k_fold_avg[5], 3)))
    print('Pearson 30s: ' + str(round(k_fold_avg[6], 3)))


    print("***********************************************************************************************************")