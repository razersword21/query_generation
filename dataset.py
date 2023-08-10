from transformers import BertTokenizer
from params import Params
import os 
from collections import Counter
from typing import NamedTuple, List, Callable, Dict, Tuple, Optional
import torch
from random import shuffle
from tqdm import tqdm
import pickle
import nltk

# class Vocab(object):
#   PAD = 0
#   SOS = 1
#   EOS = 2
#   UNK = 3
#   def __init__(self):
#     self.word2index = {}
#     self.word2count = Counter()
#     self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
#     self.index2word = self.reserved[:]
    

# class OOVDict(object):

#   def __init__(self, base_oov_idx):
#     self.word2index = {}  # type: Dict[Tuple[int, str], int]
#     self.index2word = {}  # type: Dict[Tuple[int, int], str]
#     self.next_index = {}  # type: Dict[int, int]
#     self.base_oov_idx = base_oov_idx
#     self.ext_vocab_size = base_oov_idx

#   def add_word(self, idx_in_batch, word) -> int:
#     key = (idx_in_batch, word)
#     index = self.word2index.get(key)
#     if index is not None: return index
#     index = self.next_index.get(idx_in_batch, self.base_oov_idx)
#     self.next_index[idx_in_batch] = index + 1
#     self.word2index[key] = index
#     self.index2word[(idx_in_batch, index)] = word
#     self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
#     return index


class Example(NamedTuple):
  src: List[str]
  tgt: List[str]
  src_len: int  # inclusive of EOS, so that it corresponds to tensor shape
  tgt_len: int 

class Batch(NamedTuple):
  examples: List[Example]
  input_tensor: Optional[torch.Tensor]
  attention_mask : Optional[torch.Tensor]
  label_tensor: Optional[torch.Tensor]
  input_lengths: Optional[List[int]]

class Dataset(object):
  def __init__(self, filename: str):
    super(Dataset, self).__init__()
    self.p = Params()
    self.tokenizer = BertTokenizer.from_pretrained(self.p.PRETRAINED_MODEL_NAME)
    self.src_len = 0
    self.tgt_len = 0
    self.pairs = []
    
    self.filename = filename
    print("Reading dataset %s..." % filename, end=' ', flush=True)
    print("")
    num_lines = sum(1 for line in open(filename,'r',encoding="utf-8"))
    with open(filename,encoding="utf-8") as f:
      
      for i, line in tqdm(enumerate(f),total= num_lines):
        if i == 1000:
          break
        word_pieces = ["[CLS]"]
        temp_list = line.split("\t")#strip()
        title = temp_list[1]
        
        # print(title)
        label = temp_list[-1]
        new_label = []
        blank_list = title.split(" ")
        label = label.split(" ")
        label = list(map(float, label))
        

        # for index , bi_word in enumerate(blank_list):
        #   for t in bi_word:
        #     if not '\u4e00' <= t <= '\u9fff':
        #       self.tokenizer.add_tokens(t)

        for index , wordphrase in enumerate(blank_list):
          for w in self.tokenizer.tokenize(wordphrase):
            if label[index] > 0:
              new_label.append(1)
            else:
              new_label.append(0)
        
        title = self.tokenizer.tokenize(title)

        
        word_pieces += title + ["[SEP]"]
        title_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        title_ids = torch.tensor(title_ids)

        label = torch.tensor(new_label)
        
        src_len = len(title_ids)  # EOS
        tgt_len = len(label)   # EOS
        self.src_len = max(self.src_len, src_len)
        self.tgt_len = max(self.tgt_len, tgt_len)

        self.pairs.append(Example(title_ids, label, src_len, tgt_len))
    f.close()
    print("Dataset size %d pairs." % len(self.pairs))

  def build_vocab(self):
    filename, _ = os.path.splitext(self.filename)
    filename += '.vocab'
    # print(filename)
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
  
  def __len__(self):
        return len(self.pairs)
  
  def __getitem__(self, index):
        return self.pairs[index][0], self.pairs[index][1]
  # def generator(self, batch_size: int):
  #   ptr = len(self.pairs)  # make sure to shuffle at first run
  #   # if ext_vocab:
  #   #   assert vocab is not None
  #   #   base_oov_idx = len(vocab)
  #   while True:
  #     if ptr + batch_size > len(self.pairs):
  #       shuffle(self.pairs)  # shuffle inplace to save memory
  #       ptr = 0
  #     examples = self.pairs[ptr:ptr + batch_size]
  #     ptr += batch_size
  #     src_tensor, label_tensor = None, None
  #     lengths = [x.src_len for x in examples]
  #     print(examples)
  #     src_tensor = torch.tensor(examples.src)
  #     label_tensor = torch.tensor(examples.tgt)
      
  #     yield Batch(examples, src_tensor , label_tensor, lengths)

# if __name__ == "__main__":
#    p = Params()
#    Dataset(p.file_path)


