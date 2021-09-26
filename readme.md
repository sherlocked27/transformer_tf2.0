# Transformer (Tensorflow 2.0 implementation)




Implementation of the ***Transformer*** model from the paper:

> Ashish Vaswani, et al. ["Attention is all you need."](https://arxiv.org/pdf/1706.03762.pdf) 



## Architecture




![Transformer model](./images/attention.png)



Image credit - https://github.com/lilianweng/transformer-tensorflow

## To Run

Simply go to train.py and make changes to hyperparameters and inputs such as  
  * D_MODEL = 200  
  * VOCAB_SIZE = 32000
  * BATCH_SIZE = 64
  * DATA_LIMIT = 40000
  * D_FF = 400
  * DROPOUT = 0.1
  * ENCODER_COUNT = 5
  * DECODER_COUNT = 5
  * N_H = 5  # (number of heads, keep it divisible by D_MODEL)
  * EPOCHS = 3
  * DATA_DIR = "./data"

To change data refer to data_loader.py
