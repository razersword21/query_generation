import csv
from transformers import BertTokenizer
import torch
from params import Params
from model import BertMLPModel
from tqdm import tqdm
import pandas as pd
import nltk
import re

def rad_csv(file_name):
    spenotation = "】【|(-/)）（"
    title_list = []
    query_list = []
    title_id = 0
    # 開啟 CSV 檔案
    with open(file_name,encoding="utf-8", newline='') as csvfile:

    # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)

    # 以迴圈輸出每一列
        for row in rows:
            if title_id == 0:
                title_id = 1
                continue
            for nota in spenotation:
                row[6] = row[6].replace(nota,'')
            title_list.append(row[6])
            query_list.append(row[12])
    csvfile.close()
    return title_list,query_list

def tokenizer(p,title_list):
   
    token_title_ids = []
    token_title = []
    tokenizer = BertTokenizer.from_pretrained(p.PRETRAINED_MODEL_NAME)
    # vocab = tokenizer.get_vocab()
    # for t in title_list:
    #     token = nltk.tokenize.word_tokenize(t)
    #     for nltk_t in token:
    #         if not '\u4e00' <= nltk_t <= '\u9fff':
    #             tokenizer.add_tokens(nltk_t)
    print("@ Prepare token title...")
    
    for title in tqdm(title_list):
        title_start = ["[CLS]"]
        token = tokenizer.tokenize(title)
        title_start += token + ["[SEP]"]
        token_id = tokenizer.convert_tokens_to_ids(title_start)
        token_title.append(title_start)
        token_title_ids.append(token_id)
    return token_title_ids,token_title

def query_generator(p,token_title_ids,title_list):
    ...
   
    temp_count = 0 #控制輸出一次機率值向量大小
    tokenizer = BertTokenizer.from_pretrained(p.PRETRAINED_MODEL_NAME)
    model = BertMLPModel(p.PRETRAINED_MODEL_NAME,num_labels=1)
    model.load_state_dict(torch.load(p.model_PATH))
    with torch.no_grad():
        model.eval()
        print("[Test state] Comperss product title into shorter ...")
        shorter_title_list = []

        for token_title_index,tensor_data in tqdm(enumerate(token_title_ids),total=len(token_title_ids)):
            tokens_tensors = torch.tensor(tensor_data).view(1,len(tensor_data))
            token_type_ids = torch.zeros_like(tokens_tensors).view(1,len(tensor_data))
            attention_mask = torch.ones_like(tokens_tensors).view(1,len(tensor_data))
            
            predict_prob = model(tokens_tensors,token_type_ids,attention_mask)

            # if temp_count == 0:
            #     print(tokens_tensors.size(),"predict_prob size",predict_prob.size())
            #     temp_count = 1
            
            shorter_title = ""
            
            predict_prob = predict_prob[-1,: ]

            #處理中英文            
            # title = tokenizer.tokenize(title_list[token_title_index])
            eng_index = 0
            eng_token_list = []
            eng_token = ""
            for char in title_list[token_title_index]:
                if char not in token_title[token_title_index]:
                    eng_token += char
                else:
                    eng_token_list.append(eng_token)
                    eng_token = ""
            # print(title_list[token_title_index],title)
            # print(len(predict_prob),len(token_title[token_title_index]),token_title[token_title_index])
            for index , prob in enumerate(predict_prob):
                if index == 0:
                    continue
                elif index == (len(predict_prob)-1):
                    continue
                else:
                    # print(index)
                    if token_title[token_title_index][index] != '[UNK]':
                        if prob >= 0.5:
                        
                            shorter_title += token_title[token_title_index][index]
                    else:
                        if prob >= 0.5:
                            shorter_title += eng_token_list[eng_index]
                        eng_index+=1
                            
            # print(shorter_title)
            shorter_title_list.append(shorter_title)
        del model
    return shorter_title_list

if __name__ == "__main__":

    p = Params()
    
    title_list,query_list = rad_csv(p.file_name)
    token_title_ids,token_title = tokenizer(p,title_list)
    shorter_title_list = query_generator(p,token_title_ids,title_list)
    columns=["商品原標題","query","compress result"]
    my_df = pd.DataFrame({
        "商品原標題" : title_list,
        "query" : query_list,
        "compress result" : shorter_title_list
    })
    my_df.to_csv('comperss result.csv', columns=columns,index=False, header=False,encoding="utf_8_sig")