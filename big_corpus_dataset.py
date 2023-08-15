from transformers import BertTokenizer
from params import Params
import os 
from collections import Counter
from typing import NamedTuple, List, Callable, Dict, Tuple, Optional
import torch
from random import shuffle
from tqdm import tqdm
import pickle
import numpy as np
from functools import lru_cache
import jieba
from zh_wiki import zh2Hant, zh2Hans
from langconv import Converter
import re

word_detector = re.compile('\w')

def convertor(text):
    temp_text = ""
    for item in text:
      convtext = Converter('zh-hant').convert(item) 
      temp_text += convtext
    return temp_text
class Vocab(object):
  PAD = 0
  SOS = 1
  EOS = 2
  UNK = 3
  def __init__(self):
    self.word2index = {}
    self.word2count = Counter()
    self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    self.index2word = self.reserved[:]
  def add_words(self, words: List[str]):
    for word in words:
      if word not in self.word2index:
        self.word2index[word] = len(self.index2word)
        self.index2word.append(word)
    self.word2count.update(words)
    
  def trim(self, *, vocab_size: int=None, min_freq: int=1):
    if min_freq <= 1 and (vocab_size is None or vocab_size >= len(self.word2index)):
      return
    ordered_words = sorted(((c, w) for (w, c) in self.word2count.items()), reverse=True)
    if vocab_size:
      ordered_words = ordered_words[:vocab_size]
    self.word2index = {}
    self.word2count = Counter()
    self.index2word = self.reserved[:]
    for count, word in ordered_words:
      if count < min_freq: break
      self.word2index[word] = len(self.index2word)
      self.word2count[word] = count
      self.index2word.append(word)
  def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
    num_embeddings = 0
    vocab_size = len(self)
    with open(file_path, 'rb') as f:
      for line in f:
        line = line.split()
        word = line[0].decode('utf-8')
        idx = self.word2index.get(word)
        if idx is not None:
          vec = np.array(line[1:], dtype=dtype)
          if self.embeddings is None:
            n_dims = len(vec)
            self.embeddings = np.random.normal(np.zeros((vocab_size, n_dims))).astype(dtype)
            self.embeddings[self.PAD] = np.zeros(n_dims)
          self.embeddings[idx] = vec
          num_embeddings += 1
    return num_embeddings

  def __getitem__(self, item):
    if type(item) is int:
      return self.index2word[item]
    return self.word2index.get(item, self.UNK)

  def __len__(self):
    return len(self.index2word)

  @lru_cache(maxsize=None)
  def is_word(self, token_id: int) -> bool:
    """Return whether the token at `token_id` is a word; False for punctuations."""
    if token_id < 4: return False
    if token_id >= len(self): return True  # OOV is assumed to be words
    token_str = self.index2word[token_id]
    if not word_detector.search(token_str) or token_str == '<P>':
      return False
    return True
    
class OOVDict(object):

  def __init__(self, base_oov_idx):
    self.word2index = {}  # type: Dict[Tuple[int, str], int]
    self.index2word = {}  # type: Dict[Tuple[int, int], str]
    self.next_index = {}  # type: Dict[int, int]
    self.base_oov_idx = base_oov_idx
    self.ext_vocab_size = base_oov_idx

  def add_word(self, idx_in_batch, word) -> int:
    key = (idx_in_batch, word)
    index = self.word2index.get(key)
    if index is not None: return index
    index = self.next_index.get(idx_in_batch, self.base_oov_idx)
    self.next_index[idx_in_batch] = index + 1
    self.word2index[key] = index
    self.index2word[(idx_in_batch, index)] = word
    self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
    return index

class Example(NamedTuple):
  src: List[str]
  tgt: List[str]
  src_len: int  # inclusive of EOS, so that it corresponds to tensor shape
  tgt_len: int 

class Batch(NamedTuple):
  examples: List[Example]
  input_tensor: Optional[torch.Tensor]
  label_tensor: Optional[torch.Tensor]
  input_lengths: Optional[List[int]]
 
class big_corpus_Dataset(object):
  def __init__(self, filename: str):
    super(big_corpus_Dataset, self).__init__()
    self.p = Params()
    self.tokenizer = BertTokenizer.from_pretrained(self.p.PRETRAINED_MODEL_NAME)
    
    self.src_len = 0
    self.tgt_len = 0
    self.pairs = []
    
    self.filename = filename#f = open('text.txt', 'r')

    print("Reading dataset %s..." % filename, end=' ', flush=True)
    print("")
    num_lines = sum(1 for line in open(filename,'r',encoding="utf-8"))
    with open(filename,encoding="utf-8") as f:
      
      for i, line in tqdm(enumerate(f),total= num_lines):
        if i == 500000:
          break
        word_pieces = ["[CLS]"]
        
        temp_list = line.split("\t")#strip()
        delblank = []
        for i in temp_list:
          if i != '':
            delblank.append(i)
        
        title = delblank[0].strip()
        label = delblank[1].strip()
        
        # blank_list = title.split(" ")
        # label = label.split(" ")
        # label = list(map(float, label))
      
        # for index , bi_word in enumerate(blank_list):
        #   for t in bi_word:
        #     if not '\u4e00' <= t <= '\u9fff':
        #       self.tokenizer.add_tokens(t)           
        # print("delblank",delblank)
        if self.p.is_bert_model:
          new_label = []
          title = convertor(title)
          label = convertor(label)
          title = self.tokenizer.tokenize(title)
          label = self.tokenizer.tokenize(label)
        #   print("title",title,label)
          word_pieces += title + ["[SEP]"]
          title_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
          title_ids = torch.tensor(title_ids)
          # new_label.append(0)

          for index , wordphrase in enumerate(title):
            if wordphrase in label:
              new_label.append(1)
            else:
              new_label.append(0)
          # new_label.append(0)
          newlabel = torch.tensor(new_label)
          # print(word_pieces,label)
          src_len = len(title_ids)  # +sep or +eos
          tgt_len = len(newlabel)
          self.src_len = max(self.src_len, src_len)
          self.tgt_len = max(self.tgt_len, tgt_len)
          
          self.pairs.append(Example(title_ids, newlabel, src_len, tgt_len))
        else:
          sos = []
          product_title = convertor(title)
          label = convertor(label)
          # print(product_title)
          titlejiebatoken = jieba.lcut(product_title,cut_all=True)
          labeljiebatoken = jieba.lcut(label ,cut_all=True)
          for tw in titlejiebatoken:
            tw = tw.strip()
            sos += [tw]
          new_label = []
          for ttw in sos:
            if ttw in labeljiebatoken:
                new_label.append(1)
            else:
              new_label.append(0)
        
          label = torch.tensor(new_label)
          src_len = len(sos)  # +sep or +eos
          tgt_len = len(label)
          self.src_len = max(self.src_len, src_len)
          self.tgt_len = max(self.tgt_len, tgt_len)      
          self.pairs.append(Example(sos, label, src_len, tgt_len))
        
    f.close()
    print("Dataset size %d pairs." % len(self.pairs))

  def build_vocab(self,vocab_size):
    filename, _ = os.path.splitext(self.filename)
    
    if self.p.is_bert_model:
      if self.p.if_multi:
        filename += '_multi.vocab'
      else:
        filename += '.vocab'
      if os.path.isfile(filename):
        with open(filename, 'rb') as f:
          vocab = pickle.load(f)
        print(filename," Vocabulary loaded..., %d words." % len(vocab))
        f.close()
      else:
        vocab = self.tokenizer.get_vocab()
        print("Use/Save BERT model vocab ... , ", filename)
        with open(filename, 'wb') as f:
          pickle.dump(vocab, f)
        f.close()
      return vocab
    else:
      filename, _ = os.path.splitext(self.filename)
      if vocab_size:
        filename += ".%d" % vocab_size
      filename += '.vocab'
      if os.path.isfile(filename):
        vocab = torch.load(filename)
        print("Vocabulary loaded, %d words." % len(vocab))
      else:
        print("Building vocabulary...", end=' ', flush=True)
        vocab = Vocab()
        for example in self.pairs:
          vocab.add_words(example.src)
        vocab.trim(vocab_size=vocab_size)
        print("%d words." % len(vocab))
        print(vocab)
        torch.save(vocab, filename)
    
      return vocab
  
  def __len__(self):
    return len(self.pairs)
  
  def __getitem__(self, index):
    # if self.p.is_bert_model:
      return self.pairs[index][0], self.pairs[index][1]
    # else:
    #   src_tensor = torch.zeros(len(self.pairs[index][0]), dtype=torch.long)
    #   for iw,w in enumerate(self.pairs[index][0]):
    #     idx = v[w]
    #     src_tensor[iw] = idx
    #     # print(w,idx)
    #   return src_tensor, self.pairs[index][1]
    
  def generator(self,data, batch_size: int,src_vocab):
    ptr = len(data)
    
    while True:
      if ptr + batch_size > len(data):
        shuffle(data)  # shuffle inplace to save memory
        ptr = 0
      examples = data[ptr:ptr + batch_size]
      ptr += batch_size
      src_tensor= None
      lengths = None
        # initialize tensors
      if src_vocab:
        lengths = [len(x[0]) for x in examples]
        max_src_len = max(lengths)
        src_tensor = torch.zeros(batch_size, max_src_len, dtype=torch.long)
        tgt_tensor = torch.zeros(batch_size, max_src_len, dtype=torch.float)
      # fill up tensors by word indices
      
      for i, example in enumerate(examples):
        for j, word in enumerate(example[0]):
          if word not in src_vocab:
            src_tensor[i,j] = src_vocab.UNK
          else:
            idx = src_vocab[word]
            src_tensor[i,j] = idx
      for t, exam in enumerate(examples):
        # print(len(exam[0]),len(exam[1]))
        for k, label in enumerate(exam[1]):
          tgt_tensor[t,k] = label
      yield Batch(examples, src_tensor ,tgt_tensor , lengths)

# if __name__ == "__main__":
#    p = Params()
#    big_corpus_Dataset(p.big_corpus_filepath)
