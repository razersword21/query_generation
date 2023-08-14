from typing import Optional

class Params:

    file_path: str = "LESD4EC_L/LESD4EC_L"
    file_name: str = 'product_export_tw_2023-08-01.csv'
    is_bert_model : bool = True
    if is_bert_model:
        hidden_size : int = 768 # fit bert
        model_PATH:str = 'checkpoints/bert+bilstm_best_model.pt'
    else:
        hidden_size : int = 256 # fit bert
        model_PATH:str = 'checkpoints/w2v+bi-lstm_best_model.pt'
    status:str = "train"

    vocab_size : int = 30000
    batch_size: int = 32
    n_epochs: int = 20

    lr: float = 0.0001
    hidden_dropout_prob : float = 0.2
    
    lstm_hidden : int = 128
    attention_hidden: int = 256
    mlp_hidden : int = 768
    num_layers : int = 2
    
    PRETRAINED_MODEL_NAME : str = "bert-base-chinese"
    # max_title_len : int = 20
    # max_short_title_len: int = 15

    if_fine_tune: bool = False
    del_checkpoint: bool = True
    is_attention_: bool = False
    
    model_path_prefix: Optional[str] = 'checkpoints/'
    