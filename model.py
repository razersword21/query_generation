from transformers import BertModel
import torch.nn as nn
from params import Params
import torch
import random

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,dropout=0.1
        )

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        out, (ht,ct) = self.lstm(x)
        # out, _ = self.lstm(x, (h0, c0))

        return out,ht,ct
class selfattention(nn.Module):
    def __init__(self):
        super(selfattention, self).__init__()
        self.p = Params()
        self.q_w = nn.Linear(self.p.mlp_hidden,self.p.attention_hidden)
        self.k_w = nn.Linear(self.p.mlp_hidden,self.p.attention_hidden)
        self.v_w = nn.Linear(self.p.mlp_hidden,self.p.attention_hidden)
        self.softmax = nn.Softmax(-1)

    def forward(self,input,attention_mask):
        
        Q = self.q_w(input)
        K = self.k_w(input) 
        V = self.v_w(input)
        score = torch.matmul(Q,torch.transpose(K,1,2))
        
        attention_mask = attention_mask.unsqueeze(1).repeat(1,score.shape[1],1)
        score = score*attention_mask
        score = score - torch.where(attention_mask > 0, torch.zeros_like(score), torch.ones_like(score) * float('inf')) # apply mask to softmax for thoese value is `0`
        atten_prob = self.softmax(score)
        atten_score = torch.matmul(atten_prob,V)

        return atten_score,atten_prob
        ...
class BertMLPModel(nn.Module):
    
    def __init__(self, PRETRAINED_MODEL_NAM ,vocab, num_labels=1):
        super(BertMLPModel, self).__init__()
        self.p = Params()
        self.num_labels = num_labels
        self.vocab = vocab
        self.vocab_size = len(vocab)
        if self.p.is_bert_model:
            self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAM)  # 載入預訓練 BERT
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.p.hidden_size, padding_idx=vocab.PAD)
        self.bilstm = BiLSTM(self.p.hidden_size,self.p.lstm_hidden,self.p.num_layers)
        self.dropout = nn.Dropout(self.p.hidden_dropout_prob)
        
        self.selfattention = selfattention()
        
        # 簡單 linear 層
        self.mlp = nn.Linear(self.p.mlp_hidden, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # BERT 輸入就是 tokens
        if self.p.is_bert_model:
            outputs = self.bert(input_ids, token_type_ids, attention_mask)
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = self.embedding(input_ids)
        # print("BERT model output size...",hidden_state.size())#[batch size,seq len,bert embedd(768)]
        lstm_out,_,_ = self.bilstm(hidden_state)
        # 轉成字要不要留下來的機率值
        
        # print("Bi-LSTM model output size...",lstm_out.size())#[batch size,seq len,2*lstm embedd]
        if self.p.is_attention_:
            lstm_out,_ = self.selfattention(lstm_out,attention_mask)

        logits = self.mlp(lstm_out)
        drop_out = self.dropout(logits)
        predict_label = self.sigmoid(drop_out)
        
        # print("predict_label prob output size...",predict_label.size(),labels.size())#[batch size,seq len,label]
        # 回傳各個字的 predict_label 機率值
        # loss = BertMLPModel.label_loss(self,input_ids,predict_label,labels)
        
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
    
class MutitaskModel(nn.Module):
    def __init__(self, PRETRAINED_MODEL_NAME, num_labels=1,lstm_layers=1):
        super(MutitaskModel, self).__init__()
        self.p = Params()
        self.encoder = MultitaskEncoder(PRETRAINED_MODEL_NAME)

        self.tcdecoder = TitlecompressionDecoder(num_labels,lstm_layers)
        self.qgdecoder = QuerygenerationDecoder(num_labels,lstm_layers)
        self.selfattention = selfattention()
        self.mlp = nn.Linear(self.p.mlp_hidden, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_ids,token_type_ids, attention_mask,tclabel,qglabel,teacher_forcing_ratio = 0.5):
        
        bilstm_out,_ = self.encoder(input_ids,token_type_ids, attention_mask)

        tcoutputs = torch.zeros(tclabel.shape)
        qgoutputs = torch.zeros(qglabel.shape)
        tc_len = tclabel.shape[0]
        qg_len = qglabel.shape[0]

        tc_decoder_input = bilstm_out
        qg_decoder_input = bilstm_out

        for i in range(tc_len):
            # run decode for one time step
            
            tcoutput, tchidden, tccell = self.tcdecoder(tc_decoder_input)

            # place predictions in a tensor holding predictions for each time step
            tcoutputs[i] = tcoutput

            # decide if we are going to use teacher forcing or not
            tc_teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            tc_decoder_input = tclabel[i] if tc_teacher_forcing else tcoutput

        for i in range(qg_len):
            # run decode for one time step
            qgoutput, qghidden, qgcell = self.tcdecoder(qg_decoder_input)

            # place predictions in a tensor holding predictions for each time step
            qgoutputs[i] = qgoutput

            # decide if we are going to use teacher forcing or not
            qg_teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            qg_decoder_input = qglabel[i] if qg_teacher_forcing else qgoutput
        
        return tcoutputs , qgoutputs
class MultitaskEncoder(nn.Module):
    def __init__(self, PRETRAINED_MODEL_NAME):
        super(MultitaskEncoder, self).__init__()
        self.p = Params()
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)  # 載入預訓練 BERT
        self.bilstm = BiLSTM(self.p.hidden_size,self.p.lstm_hidden,self.p.num_layers)

    def forward(self,input_ids,token_type_ids, attention_mask):
        
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        hidden_state = outputs.last_hidden_state
        # print("BERT model output size...",hidden_state.size())#[batch size,seq len,bert embedd(768)]
        lstm_out,(hidden,_) = self.bilstm(hidden_state)
        print("lstm model output size...",hidden.size()) 
        
        return lstm_out,hidden

class TitlecompressionDecoder(nn.Module):
    def __init__(self, num_labels=1,lstm_layers=1):
        super(TitlecompressionDecoder, self).__init__()
        self.p = Params()
        
        self.lstm = nn.LSTM(
            2*self.p.lstm_hidden, self.p.lstm_hidden, lstm_layers, batch_first=True, bidirectional=False,dropout=0.1
        )
        
        self.linear = nn.Linear(self.p.mlp_hidden, num_labels)
        self.dropout = nn.Dropout(self.p.hidden_dropout_prob)
        self.softmax = nn.Softmax(0)
        self.relu = nn.ReLU()

    def forward(self,tclabel,encoder_states, hidden, cell):

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        decoder_init_state,(hidden, cell) = self.lstm(input[:,[input.shape[1]-1],:],(hidden, cell))
        prediction = self.linear(decoder_init_state.squeeze(0))

        return prediction, hidden, cell

class QuerygenerationDecoder(nn.Module):
    def __init__(self, num_labels=1,lstm_layers=1):
        super(QuerygenerationDecoder, self).__init__()
        self.p = Params()
        self.dropout = nn.Dropout(self.p.hidden_dropout_prob)
        self.lstm = nn.LSTM(
            2*self.p.lstm_hidden, self.p.lstm_hidden, lstm_layers, batch_first=True, bidirectional=False,dropout=0.1
        )
    def forward(self,input,hidden, cell):

        decoder_init_state,(hidden, cell) = self.lstm(input[:,[input.shape[1]-1],:],(hidden, cell))
        prediction = self.linear(decoder_init_state.squeeze(0))

        return prediction, hidden, cell
