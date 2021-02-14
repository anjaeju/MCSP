import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import Data_loader as D
from Model import Base
from Custom import CustomDataset

from tqdm import tqdm
from collections import Counter

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='type arguments for experiments.')
    parser.add_argument('--load_path', type=str, help='path for loading image')
    parser.add_argument('--save_path', type=str, help='path for saving the results')
    parser.add_argument('--total_exp', type=int, default=30, help='the number of training we want to conduct')
    parser.add_argument('--num_epochs', type=int, default=500, help='epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--num_classes', type=int, default=9, help='the number of classes for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
    
    args = parser.parse_args()
    
    # Hyper params
    load_path   = args.load_path
    save_path   = args.save_path    # 'res/1gram_baseline'
    BATCH_SIZE  = args.batch_size  
    TOTAL       = args.total_exp   
    EPOCH       = args.num_epochs  
    NUM_CLASS   = args.num_classes
    LR          = args.learning_rate 
    SEED        = [s for s in range(TOTAL)]
    INPUT_NODES = 768
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    temp = [[],[]]
    
    Loader = D.File_loader()
    data_a, label_a = Loader.read_files(load_path, interp = False)
    idx = np.argsort(label_a)
    sorted_data = data_a[idx].reshape(10736, -1)
    sorted_label = sorted(label_a)
        
    # creating data indices for spliting
    full_dataset = CustomDataset(sorted_data, sorted_label)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # spliting
    torch.manual_seed(10)
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = False)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    loss_total = []
    acc_total = []
    pred_total = []
    true_total = []
    
    softmax = nn.Softmax()
    for i in tqdm(range(TOTAL)):
        
        torch.manual_seed(SEED[i])
        net = Base(INPUT_NODES, NUM_CLASS)           
        net.to(device)
        print(net)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum = 0.1)
        
        loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        pred_temp = []
        true_temp = []
        
        for epoch in range(EPOCH):
            net.train()
            running_loss = 0
            total = train_size
            correct = 0 
            
            for step, image_and_label in enumerate(train_loader):
                inputs, labels = image_and_label            
                inputs, labels = inputs.type(torch.FloatTensor).to(device), labels.type(torch.LongTensor).to(device)
                
                outputs = net(inputs)
                
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                _, pred = torch.max(outputs, dim=1)
                correct += (pred == labels).sum().item()
                
            train_acc = correct/total
            loss_list.append(running_loss)
            train_acc_list.append(train_acc)
            print('{}th- epoch: {}, train_loss = {}, train_acc = {}'.format(i+1, epoch, running_loss, train_acc))
            
            with torch.no_grad():
                net.eval()
                correct = 0
                total = test_size
                pt, tt = [], []
                
                for step_t, image_and_label_t in enumerate(test_loader):
                    inputs_t, labels_t = image_and_label_t            
                    inputs_t, labels_t = inputs_t.type(torch.FloatTensor).to(device), labels_t.type(torch.LongTensor).to(device)
                    
                    outputs_t = net(inputs_t)
                    outputs_t = softmax(outputs_t)
                    
                    # test accuracy
                    _, pred_t = torch.max(outputs_t, dim = 1)
                    
                    pt.append(pred_t)
                    tt.append(labels_t)
                    
                    correct += (pred_t == labels_t).sum().item()
                    
                pred_temp.append(torch.cat(pt))
                true_temp.append(torch.cat(tt))
                
                test_acc = correct/total
                test_acc_list.append(test_acc)
                
                print('test Acc {}:'.format(test_acc))
                
        best_result_index = np.argmax(np.array(test_acc_list))
        loss_total.append(loss_list[best_result_index])
        acc_total.append(test_acc_list[best_result_index])
        pred_total.append(pred_temp[best_result_index].tolist())
        true_total.append(true_temp[best_result_index].tolist())
        
    file_name = 'res/1gram_baseline'
    torch.save(net.state_dict(), file_name +'.pth')
    
    loss_DF = pd.DataFrame(loss_total)
    loss_DF.to_csv(file_name+" loss.csv")
    
    acc_DF = pd.DataFrame(acc_total)
    acc_DF.to_csv(file_name +" acc.csv")
    
    pred_DF = pd.DataFrame(pred_total)
    pred_DF.to_csv(file_name +" pred.csv")
    
    true_DF = pd.DataFrame(true_total)
    true_DF.to_csv(file_name +" true.csv")