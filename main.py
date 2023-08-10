from params import Params
import torch 
import torch.nn as nn
from model import BertMLPModel
from tqdm import tqdm
from dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import os
import numpy as np

np.seterr(invalid='ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_label_size(predict_label):
          
    predict_label = predict_label[:, :, -1]
    predict_label = predict_label[:,1:-1]
    return predict_label

def train(model,train_gen,p):
    model.to(DEVICE)
    model.train()
    
    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), p.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criti = nn.BCELoss()
    best_loss = 99.9
    best_model = False
    epochs = p.n_epochs  
    for epoch in range(epochs):
        running_loss = 0.0
        for data in tqdm(enumerate(train_gen),total=len(train_gen)):
            tokens_tensors = data[1][0].to(DEVICE)
            labels = data[1][1].to(DEVICE)
            segments_tensors = torch.zeros_like(tokens_tensors).to(DEVICE)
            masks_tensors = torch.ones_like(tokens_tensors).to(DEVICE)
            for bi,batch_data in enumerate(tokens_tensors):
                for wi,w in enumerate(batch_data):
                    if w == 0:
                        masks_tensors[bi][wi] = 0
            
            # print(tokens_tensors,labels)
            # 將參數梯度歸零
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)

            loss = criti(predict_label_size(outputs),labels.float())
            # backward
            loss.backward()
            optimizer.step()
            
            # 累計當前 batch loss
            running_loss += loss.item()
            
        running_loss = running_loss/len(train_gen)
        scheduler.step(running_loss)
        print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))

        if running_loss < best_loss:
            best_model = True
            best_loss = running_loss
        else:
            best_model = False
        #存模型
        save_models(p,model,optimizer,epoch,running_loss,p.if_fine_tune,best_model)
    del model

def test(model,test_gen,p):
    
    model.load_state_dict(torch.load(p.model_PATH))
    with torch.no_grad():
        model.eval()
        # print(model)
        # 使用 Adam Optim 更新整個分類模型的參數

        acc,sum,recall,precision,pacc = 0,0,0,0,0
        for data in tqdm(enumerate(test_gen),total=len(test_gen)):
            tokens_tensors = data[1][0]
            labels = data[1][1]
            segments_tensors = torch.zeros_like(tokens_tensors)
            masks_tensors = torch.ones_like(tokens_tensors)
            # for bi,batch_data in enumerate(tokens_tensors):
            #     for wi,w in enumerate(batch_data):
            #         if w == 0:
            #             masks_tensors[bi][wi] = 0
            # print(tokens_tensors,labels)
            # forward pass
            outputs_prob = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            #將預測機率轉成label
            for label_ind,labs in enumerate(labels):
                for index , lab in enumerate(labs):
                    if tokens_tensors[label_ind][index+1] != 102 or tokens_tensors[label_ind][index+1] != 0:
                        sum += 1
                        if outputs_prob[label_ind][index+1] >= 0.5 and lab == 1:
                            acc += 1
                            pacc+=1
                        elif outputs_prob[label_ind][index+1] < 0.5 and lab == 0:
                            acc += 1
                        if lab == 1:
                            recall += 1
                        if outputs_prob[label_ind][index+1] >= 0.5:
                            precision += 1
        del model
    print(labels.size(),outputs_prob.size())
    print("Test state : ")
    print("Accuracy : %.2f %%" % ((acc / sum *100) ,))
    print(acc , sum)
    r = (pacc / recall *100)
    p = (pacc / precision *100)
    print("F1-score : %.2f %%" % ((2*p*r/(p+r)),) )
    print("Recall : %.2f %%" % ((r),) )
    print("Precision : %.2f %%"  % ((p),) )

def create_mini_batch(samples):
    
    tokens_tensors = [s[0] for s in samples]
    label_ids = [s[1] for s in samples]  
    # # 測試集有 labels
    # if samples[0][1] is not None:
    #     label_ids = torch.stack([s[1] for s in samples])
    # else:
    #     label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)   
    label_ids = pad_sequence(label_ids, 
                                  batch_first=True)  
    
    return tokens_tensors, label_ids

def save_models(p, models,optimizer, epoch,loss,if_fine_tune,best_model):
    if not if_fine_tune:
        try:
            if best_model == True:
                torch.save(models.state_dict(), p.model_path_prefix+'bert_bilstm_best_model.pt')
            torch.save(models.state_dict(), p.model_path_prefix+'bert_bilstm_%.2f_epoch%d.pt'% (loss, (epoch+1)))
        except Exception as e:
            print("Model saving failed.")
    else:
        checkpoint = {
            "model" : model,
            "opt" : optimizer
        }
        for name, model in models.items():
            checkpoint[name] = model
        for name, opt in optimizer.items():
            checkpoint[name] = opt
        torch.save(checkpoint,p.model_path_prefix+'checkpoint_bert_bilstm_%.2f_%d.pt'% (loss, (epoch+1)))
        

if __name__ == "__main__":

    p = Params()
        
    dataset = Dataset(p.file_path)
    vocab = dataset.build_vocab()

    model = BertMLPModel(p.PRETRAINED_MODEL_NAME,num_labels=1)
    train_data, test_data = train_test_split(dataset,random_state = 27, train_size=0.8)
    # print(train_data[0],test_data[0])
    
    if p.status == "train":
        #清除checkpoint 裡的模型參數檔
        if p.del_checkpoint:
            for f in os.listdir(p.model_path_prefix):
                os.remove(os.path.join(p.model_path_prefix, f))

        train_gen = DataLoader(train_data,batch_size=p.batch_size,shuffle=True, pin_memory=True,collate_fn=create_mini_batch)
        train(model,train_gen,p)
    else:
        test_gen = DataLoader(test_data,batch_size=p.batch_size,shuffle=True, pin_memory=True,collate_fn=create_mini_batch)
        test(model,test_gen,p)