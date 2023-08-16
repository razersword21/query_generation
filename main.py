from params import Params
import torch 
import torch.nn as nn
from model import BertMLPModel
from tqdm import tqdm
from dataset import Dataset
from big_corpus_dataset import big_corpus_Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import os
import numpy as np

np.seterr(invalid='ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_label_size(p,predict_label):
          
    predict_label = predict_label[:, :, -1] #刪除最後一個column 若num labels = 2 則不用這行
    if p.is_bert_model:
        predict_label = predict_label[:,1:-1] #刪除其中cls和最後一個pad標籤 使形狀與labels一致
    return predict_label

def train(model,train_gen,p,vocab):
    model.to(DEVICE)
    model.train()
    
    # 使用 Adam Optim 更新整個分類模型的參數
    optimizer = torch.optim.Adam(model.parameters(), p.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2,verbose=True)
    
    criti = nn.BCELoss()
    best_loss = 99.9
    best_model = False
    epochs = p.n_epochs  
    for epoch in range(epochs):
        running_loss,batch_c = 0.0,0
        if p.is_bert_model:
            
            for data in tqdm(enumerate(train_gen),total=len(train_gen)):
                tokens_tensors = data[1][0].to(DEVICE)
                labels = data[1][1].to(DEVICE)
                segments_tensors = torch.zeros_like(tokens_tensors).to(DEVICE)
                masks_tensors = torch.ones_like(tokens_tensors).to(DEVICE)
                for bi,batch_data in enumerate(tokens_tensors):
                    for wi,w in enumerate(batch_data):
                        if w == 0:
                            masks_tensors[bi][wi] = 0
                # print(tokens_tensors[0])
                optimizer.zero_grad()
                # test predict prob
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors)
                loss = criti(predict_label_size(p,outputs),labels.float())
                # print(outputs.size())
                # loss = test_critition(predict_label_size(p,outputs),labels.float())
                # backward
                loss.backward()
                optimizer.step()
                
            # 累計當前 batch loss
                running_loss += loss.item()
            scheduler.step(running_loss)
                #將預測機率轉成label
            running_loss = running_loss/len(train_gen)
            scheduler.step(running_loss)
            print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))
            # print("lr(變動) : ",optimizer.state_dict()['param_groups'][0]['lr'])
            if running_loss < best_loss:
                best_model = True
                best_loss = running_loss
            else:
                best_model = False
            #存模型
            save_models(p,model,optimizer,epoch,running_loss,p.if_fine_tune,best_model)
                                            
        else:
            prog_bar = tqdm(range(1, 5000+1))
            for batch_count in prog_bar: 
                batch = next(train_gen)
                batch_c +=1
                input_tensor = batch.input_tensor.to(DEVICE)
                labels = batch.label_tensor.to(DEVICE)
                optimizer.zero_grad()
                masks_tensors = torch.ones_like(input_tensor).to(DEVICE)
                for bi,batch_data in enumerate(input_tensor):
                        for wi,w in enumerate(batch_data):
                            if w == 0:
                                masks_tensors[bi][wi] = 0
                outputs = model(input_ids=input_tensor, 
                                    attention_mask=masks_tensors)

                loss = criti(predict_label_size(p,outputs),labels.float())
                # backward
                loss.backward()
                optimizer.step()
                
                # 累計當前 batch loss
                running_loss += loss.item()
            running_loss = running_loss/batch_c
            scheduler.step(running_loss)
            print('[epoch %d] loss: %.3f' %(epoch + 1, running_loss))
            print("lr(變動) : ",optimizer.state_dict()['param_groups'][0]['lr'])
            if running_loss < best_loss:
                best_model = True
                best_loss = running_loss
            else:
                best_model = False
            #存模型
            save_models(p,model,optimizer,epoch,running_loss,p.if_fine_tune,best_model)
        
    del model

def test(model,test_gen,p,vocab):
    # import os
    # print(os.getcwd())
    model.load_state_dict(torch.load(p.model_PATH), strict=False)
    model.to(DEVICE)
    with torch.no_grad():
        model.eval()
        # print(model)
        # 使用 Adam Optim 更新整個分類模型的參數
        
        acc,sum,recall,precision,pacc = 0,0,0,0,0
        if p.is_bert_model:
            print("Test "+p.model_PATH+" model ...")
            for data in tqdm(enumerate(test_gen),total=len(test_gen)):
                tokens_tensors = data[1][0].to(DEVICE)
                labels = data[1][1].to(DEVICE)
                segments_tensors = torch.zeros_like(tokens_tensors).to(DEVICE)
                masks_tensors = torch.ones_like(tokens_tensors).to(DEVICE)
                for bi,batch_data in enumerate(tokens_tensors):
                    for wi,w in enumerate(batch_data):
                        if w == 0 :
                            masks_tensors[bi][wi] = 0
                # print(tokens_tensors[0])

                # test predict prob
                outputs_prob = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors)
                
                #將預測機率轉成label
                outputs_prob = predict_label_size(p,outputs_prob)
                for label_ind,labs in enumerate(labels):
                    for index , lab in enumerate(labs):
                        #確保不是cls,sep,pad
                        if p.is_bert_model:
                            if tokens_tensors[label_ind][index] != 101 or tokens_tensors[label_ind][index] != 102 or tokens_tensors[label_ind][index] != 0:
                                sum += 1
                                if outputs_prob[label_ind][index] >= 0.5 :
                                    precision += 1
                                    if lab == 1:
                                        acc += 1
                                        pacc+=1
                                        
                                elif outputs_prob[label_ind][index] < 0.5 and lab == 0:
                                    acc += 1
                                if lab == 1:
                                    recall += 1 
                        else:
                            if input_tensor[label_ind][index] != 1 or input_tensor[label_ind][index] != 2 or input_tensor[label_ind][index] != 0:
                                sum += 1
                                if outputs_prob[label_ind][index] >= 0.5 :
                                    precision += 1
                                    if lab == 1:
                                        acc += 1
                                        pacc+=1
                                elif outputs_prob[label_ind][index] < 0.5 and lab == 0:
                                    acc += 1
                                if lab == 1:
                                    recall += 1
                                           
        else:
            print("Test Embedding+Bi-LSTM+MLP model ...")
            prog_bar = tqdm(range(1, 625+1))
            for batch_count in prog_bar: 
                batch = next(test_gen)
                
                input_tensor = batch.input_tensor.to(DEVICE)
                labels = batch.label_tensor.to(DEVICE)
                
                masks_tensors = torch.ones_like(input_tensor).to(DEVICE)
                for bi,batch_data in enumerate(input_tensor):
                        for wi,w in enumerate(batch_data):
                            if w == 0:
                                masks_tensors[bi][wi] = 0
                outputs_prob = model(input_ids=input_tensor, 
                                    attention_mask=masks_tensors)
            # print(input_tensor.size(),labels.size(),outputs_prob.size())
                outputs_prob = predict_label_size(p,outputs_prob)
                for label_ind,labs in enumerate(labels):
                    for index , lab in enumerate(labs):
                        #確保不是cls,sep,pad
                        if p.is_bert_model:
                            if tokens_tensors[label_ind][index] != 101 or tokens_tensors[label_ind][index] != 102 or tokens_tensors[label_ind][index] != 0:
                                sum += 1
                                if outputs_prob[label_ind][index] >= 0.5 :
                                    precision += 1
                                    if lab == 1:
                                        acc += 1
                                        pacc+=1
                                elif outputs_prob[label_ind][index] < 0.5 and lab == 0:
                                    acc += 1
                                if lab == 1:
                                    recall += 1 
                        else:
                            if input_tensor[label_ind][index] != 1 or input_tensor[label_ind][index] != 2 or input_tensor[label_ind][index] != 0:
                                sum += 1
                                if outputs_prob[label_ind][index] >= 0.5 :
                                    precision += 1
                                    if lab == 1:
                                        acc += 1
                                        pacc+=1
                                elif outputs_prob[label_ind][index] < 0.5 and lab == 0:
                                    acc += 1
                                if lab == 1:
                                    recall += 1
        del model
    
    print("Test state : ")
    print("Accuracy : %.2f %%" % ((acc / sum *100) ,))
    print(pacc , precision,acc)
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
    
    if p.is_bert_model:
        file_prefix = 'bert_'
        if p.if_multi:
            file_prefix += 'multi_'
        if p.bert_lstm_state:
            file_prefix += 'bilstm_'
        if p.is_attention_:
            file_prefix += 'atten_'
        if p.which_data_path:
            file_prefix += 'LEC'
        else:
            file_prefix += 'big_corpus'
    else:
        file_prefix = 'embedd_bilstm_'
        if p.is_attention_:
            file_prefix += 'atten_'
        if p.which_data_path:
            file_prefix += 'LEC'
        else:
            file_prefix += 'big_corpus'

    if not if_fine_tune:
        try:
            if best_model == True:#bert+bilstm_best_model
                torch.save(models.state_dict(), p.model_path_prefix+file_prefix+'_best_model.pt')
            # torch.save(models.state_dict(), p.model_path_prefix+file_prefix+'_%.2f_epoch%d.pt'% (loss, (epoch+1)))
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
        torch.save(checkpoint,p.model_path_prefix+'checkpoint_'+file_prefix+'_%.2f_%d.pt'% (loss, (epoch+1)))
        

if __name__ == "__main__":

    p = Params()
    
    if p.which_data_path:
        dataset = Dataset(p.file_path)
        vocab = dataset.build_vocab(p.vocab_size)
        train_data, test_data = train_test_split(dataset,random_state = 27, train_size=0.8)
        
    else:
        dataset = big_corpus_Dataset(p.file_path)
        vocab = dataset.build_vocab(p.vocab_size)
        train_data, test_data = train_test_split(dataset,random_state = 27, train_size=0.8)
    # print(train_data[0],test_data[0])

    model = BertMLPModel(p.PRETRAINED_MODEL_NAME,vocab,p.num_labels)

    if p.status == "train":
        #清除checkpoint 裡的模型參數檔
        if p.del_checkpoint:
            for f in os.listdir(p.model_path_prefix):
                os.remove(os.path.join(p.model_path_prefix, f))
        if p.is_bert_model:
            print("Train BERT(multi)+MLP model ...")
            train_gen = DataLoader(train_data,batch_size=p.batch_size,shuffle=True, pin_memory=True,collate_fn=create_mini_batch)
        else:
            print("Train embedding+Bi-LSTM+MLP model ...")
            train_gen = dataset.generator(train_data,p.batch_size, vocab)
        train(model,train_gen,p,vocab)
    else:
        # print(test_data)
        if p.is_bert_model:
            test_gen = DataLoader(test_data,batch_size=p.batch_size,shuffle=True, pin_memory=True,collate_fn=create_mini_batch)
        else:
            
            test_gen = dataset.generator(test_data,p.batch_size, vocab)
        
        test(model,test_gen,p,vocab)