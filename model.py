from transformers import BertModel
import torch.nn as nn
from params import Params
import torch

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        out, (ht,ct) = self.lstm(x)
        # out, _ = self.lstm(x, (h0, c0))

        return out

class BertMLPModel(nn.Module):
    
    def __init__(self, PRETRAINED_MODEL_NAM , num_labels=1):
        super(BertMLPModel, self).__init__()
        self.p = Params()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAM)  # 載入預訓練 BERT
        self.bilstm = BiLSTM(self.p.hidden_size,self.p.lstm_hidden,self.p.num_layers)
        self.dropout = nn.Dropout(self.p.hidden_dropout_prob)
        # 簡單 linear 層
        self.mlp = nn.Linear(self.p.mlp_hidden, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # BERT 輸入就是 tokens
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        hidden_state = outputs.last_hidden_state
        
        # print("BERT model output size...",hidden_state.size())#[batch size,seq len,bert embedd(768)]
        lstm_out = self.bilstm(hidden_state)
        # 轉成字要不要留下來的機率值
        
        # print("Bi-LSTM model output size...",lstm_out.size())#[batch size,seq len,2*lstm embedd]
        logits = self.mlp(lstm_out)
        predict_label = self.sigmoid(logits)
        
        # print("predict_label prob output size...",predict_label.size(),labels.size())#[batch size,seq len,label]
        # 回傳各個字的 predict_label 機率值
        loss = BertMLPModel.label_loss(self,input_ids,predict_label,labels)
        
        return predict_label
    
    def label_loss(self,input_ids,predict_label,labels):
        
        
        predict_label = predict_label[:, :, -1]
        predict_label = predict_label[:,1:-1]
        # print("predict_label prob output size...",predict_label.size(),type(predict_label))
        # print("labels prob output size...",labels.size(),labels)
        # print(predict_label,labels)
        loss = self.loss(predict_label,labels.float())
        # print(loss)
        return loss
    
    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        # BERT 輸入就是 tokens
        # print("(Test state) BERT model input size...",input_ids.size())
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        hidden_state = outputs.last_hidden_state
        
        # print("(Test state) BERT model output size...",hidden_state.size())
        lstm_out = self.bilstm(hidden_state)
        # 轉成字要不要留下來的機率值
     
        # print("(Test state) Bi-LSTM model output size...",lstm_out.size())
        logits = self.mlp(lstm_out)
        predict_label = self.sigmoid(logits)

        return predict_label

