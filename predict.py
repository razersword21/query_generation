import csv
from transformers import BertTokenizer
import torch
from params import Params
from model import BertMLPModel
from tqdm import tqdm
import pandas as pd
from dataset import Dataset
import jieba

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

def tokenizer(p,title_list,v):
   
    token_title_ids = []
    token_title = []
    print("@ Prepare token title...")
    if p.is_bert_model:
        tokenizer = BertTokenizer.from_pretrained(p.PRETRAINED_MODEL_NAME)
    # vocab = tokenizer.get_vocab()
    # for t in title_list:
    #     token = nltk.tokenize.word_tokenize(t)
    #     for nltk_t in token:
    #         if not '\u4e00' <= nltk_t <= '\u9fff':
    #             tokenizer.add_tokens(nltk_t)
        for title in tqdm(title_list):
            title_start = ["[CLS]"]
            token = tokenizer.tokenize(title)
            title_start += token + ["[SEP]"]
            token_id = tokenizer.convert_tokens_to_ids(title_start)
            token_title.append(title_start)
            token_title_ids.append(token_id)
    else:
        
        for title in tqdm(title_list):
            title_start = []
            tokentitle = jieba.lcut(title,cut_all=True)
            token_text = []
            token_id = []
            for tw in tokentitle:
                if tw == ' ' or tw == '':
                    continue
                title_start+=[tw]
                if tw in v:
                    token_id.append(v[tw])
                else:
                    token_id.append(vocab.UNK)
            
            token_title_ids.append(token_id)
            token_title.append(title_start)

    return token_title_ids,token_title

def query_generator(p,token_title_ids,v,title_list,token_title):
    ...
    if p.is_bert_model:
        model = BertMLPModel(p.PRETRAINED_MODEL_NAME,v,num_labels=1)
        model.load_state_dict(torch.load(p.model_PATH), strict=False)
    # print(model)
        with torch.no_grad():
            model.eval()
            print("[Test state (BERT)] Comperss product title into shorter ...")
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
                # print(title_list[token_title_index]) #聲寶 SAMPO ECSA05HT 聲寶 手持充電吸塵器 
                for char in title_list[token_title_index]:
                    if char == ' ':
                        if eng_token != "" and eng_token != " " and eng_token not in token_title[token_title_index]:
                            eng_token_list.append(eng_token)
                        eng_token = ""
                        
                    elif char not in token_title[token_title_index]: 
                        eng_token += char
                    else:
                        if eng_token != "" and eng_token != " " and eng_token not in token_title[token_title_index]:
                            eng_token_list.append(eng_token)
                        eng_token = ""
                if eng_token != "" and eng_token != " ":
                    eng_token_list.append(eng_token)    
                # print(title_list[token_title_index],title)
                # print(len(predict_prob),len(token_title[token_title_index]),token_title[token_title_index])
                # print(title_list[token_title_index])
                # print("**",eng_token_list,"**",eng_index,"**",token_title[token_title_index])
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
        
    else:
        model = BertMLPModel(p.PRETRAINED_MODEL_NAME,v,num_labels=1)
        model.load_state_dict(torch.load(p.model_PATH), strict=False) 
        with torch.no_grad():
            model.eval()
            print("[Test state (Embedding)] Comperss product title into shorter ...")
            shorter_title_list = []
            for token_title_index,tensor_data in tqdm(enumerate(token_title_ids),total=len(token_title_ids)):
                print(tensor_data,token_title[token_title_index],v['<EOS>'],v['<SOS>'],v['<UNK>'],v['<PAD>'],vocab.UNK,vocab.PAD,vocab.SOS,vocab.EOS)
                tokens_tensors = torch.tensor(tensor_data).view(1,len(tensor_data))
                attention_mask = torch.ones_like(tokens_tensors).view(1,len(tensor_data))
                
                predict_prob = model(input_ids=tokens_tensors,attention_mask=attention_mask)
                shorter_title = ""
                predict_prob = predict_prob[-1,: ]

                unk_index = 0
                unk_token_list = []
                unk_token = ""
                # print(title_list[token_title_index]) #聲寶 SAMPO ECSA05HT 聲寶 手持充電吸塵器 
                
                for index , prob in enumerate(predict_prob):
                    if index == 0:
                        continue
                    elif index == (len(predict_prob)-1):
                        continue
                    else:
                        # print(index)
                        if tensor_data[index] != vocab.UNK:
                            if prob >= 0.5:
                                shorter_title += token_title[token_title_index][index]
                        else:
                            if prob >= 0.5:
                                shorter_title += token_title[token_title_index][index]
                                
                # print(shorter_title)
                shorter_title_list.append(shorter_title)    

    del model
    return shorter_title_list

if __name__ == "__main__":

    p = Params()
    dataset = Dataset(p.file_path)
    vocab = dataset.build_vocab(p.vocab_size)
    title_list,query_list = rad_csv(p.file_name)
    
    token_title_ids,token_title = tokenizer(p,title_list,vocab)

    shorter_title_list = query_generator(p,token_title_ids,vocab,title_list,token_title)

    columns=["商品原標題","query","compress result"]
    my_df = pd.DataFrame({
        "商品原標題" : title_list,
        "query" : query_list,
        "compress result" : shorter_title_list
    })
    my_df.to_csv('comperss result.csv', columns=columns,index=False, header=False,encoding="utf_8_sig")