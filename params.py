from typing import Optional

class Params:

    which_data_path: bool = False
    if which_data_path:
        file_path: str = "LESD4EC_L/LESD4EC_L"
    else:
        big_corpus_filepath : str = 'ProductTitleSummarizationCorpus/big_corpus/big_corpus.txt'
    file_name: str = 'product_export_tw_2023-08-01.csv'
    
    is_bert_model : bool = True
    if is_bert_model:
        hidden_size : int = 768 # fit bert
        mlp_hidden : int = 768
        model_PATH:str = 'checkpoints/bert_tozhant_best_model.pt'
    else:
        hidden_size : int = 128 # fit bert
        mlp_hidden : int = 256
        model_PATH:str = 'checkpoints/w2v+bilstm_best_model.pt'
    status:str = "train"

    if_multi:bool = True
    if if_multi:
        PRETRAINED_MODEL_NAME : str = "bert-base-multilingual-cased"
    else:
        PRETRAINED_MODEL_NAME : str = "bert-base-chinese"

    if_fine_tune: bool = False
    del_checkpoint: bool = False
    is_attention_: bool = True
    bert_lstm_state:bool = True

    num_labels : int = 1
    vocab_size : int = 30000
    batch_size: int = 32
    n_epochs: int = 20

    lr: float = 0.0001
    hidden_dropout_prob : float = 0.5
    lstm_hidden : int = 128
    attention_hidden: int = 256
    num_layers : int = 2
    
    model_path_prefix: Optional[str] = 'checkpoints/'
    