from numpy.lib.utils import source
import tensorflow as tf
from urllib.request import urlretrieve
import os
from icecream import ic
import sentencepiece
from sklearn.model_selection import train_test_split


class Data_Loader:
    CONFIG = {
        'wmt14/en-de': {
            'source_lang': 'en',
            'target_lang': 'de',
            'base_url': 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/',
            'train_files': ['train.en', 'train.de'],
            'vocab_files': ['vocab.50K.en', 'vocab.50K.de'],
            'dictionary_files': ['dict.en-de'],
            'test_files': [
                'newstest2012.en', 'newstest2012.de',
                'newstest2013.en', 'newstest2013.de',
                'newstest2014.en', 'newstest2014.de',
                'newstest2015.en', 'newstest2015.de',
            ]
        }
    }
    PATHS = {
        'source_data': None,
        'train_data': None
    }
    BPE = {
        "source": None,
        "target": None
    }
    MAX_SEQ_LENGTH = 100
    DATA_LIMIT = None
    BATCH_SIZE = None

    def __init__(self, dir, dataset_name, data_limit, batch_size) -> None:
        self.DIR = dir
        self.DATASET = dataset_name

        self.PATHS['source_data'] = os.path.join(
            self.DIR, self.CONFIG[dataset_name]['train_files'][0])
        self.PATHS['target_data'] = os.path.join(
            self.DIR, self.CONFIG[dataset_name]['train_files'][1])
        self.BPE_VOCAB_SIZE = 32000
        self.DATA_LIMIT = data_limit
        self.BATCH_SIZE = batch_size

    def load(self):
        ic('#1 download data')
        self.download_dataset()

        ic('#2 parse data')
        source_data = self.parse_data_and_save(self.PATHS['source_data'])
        target_data = self.parse_data_and_save(self.PATHS['target_data'])

        ic(source_data[0])
        ic(target_data[0])

        ic('#3 train bpe')
        self.train_bpe(self.PATHS['source_data'],
                       self.PATHS['source_data']+".bpe")
        self.train_bpe(self.PATHS['target_data'],
                       self.PATHS['target_data']+".bpe")

        if(self.DATA_LIMIT is not None):
            source_data = source_data[:self.DATA_LIMIT]
            target_data = target_data[:self.DATA_LIMIT]

        ic("#4 tokenise and load")
        source_encoded_data = self.encode_data(
            source_data, self.PATHS['source_data']+".bpe")
        target_encoded_data = self.encode_data(
            target_data, self.PATHS['target_data']+".bpe")

        print(source_encoded_data[0])
        print(target_encoded_data[0])

        ic("#5 test-train split")
        source_sequence_trained, source_sequence_val = train_test_split(
            source_encoded_data, train_size=0.85)
        target_sequence_trained, target_sequence_val = train_test_split(
            target_encoded_data, train_size=0.85)

        ic('#6 create dataset')
        train_dataset = self.create_dataset(
            source_sequence_trained,
            source_sequence_val
        )

        val_dataset = self.create_dataset(
            target_sequence_trained,
            target_sequence_val
        )
        return train_dataset, val_dataset

    def download_dataset(self):
        for file in (self.CONFIG[self.DATASET]['train_files']
                     + self.CONFIG[self.DATASET]['vocab_files']
                     + self.CONFIG[self.DATASET]['dictionary_files']
                     + self.CONFIG[self.DATASET]['test_files']):
            self._download("{}{}".format(
                self.CONFIG[self.DATASET]['base_url'], file))

    def _download(self, url):
        path = os.path.join(self.DIR, url.split('/')[-1])
        if not os.path.exists(path):
            urlretrieve(url, path)

    def parse_data_and_save(self, path):
        ic("opening the file and loading data from ", path)
        with open(path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        ic("saving the file")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return lines

    def train_bpe(self, data_path, model_prefix):
        vocab_path = model_prefix + ".vocab"
        model_path = model_prefix + ".model"

        if not(os.path.exists(vocab_path) and os.path.exists(model_path)):
            ic("bpe model don't exist, training one")
            train_param = "--input={} \
                --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 \
                --model_prefix={}\
                --vocab_size={}  \
                --model_type=bpe ".format(
                data_path,
                model_prefix,
                self.BPE_VOCAB_SIZE
            )
            sentencepiece.SentencePieceTrainer.Train(train_param)
        else:
            ic("bpe model exists")

    def encode_data(self, data_list, model_prefix):
        model_path = model_prefix + ".model"

        sp_bpe = sentencepiece.SentencePieceProcessor()
        sp_bpe.Load(
            model_path)
        sequences = []

        for sent in data_list:
            # adding start of sentence and end of sentence token to the encoded ids
            seq = [2] + sp_bpe.encode_as_ids(sent) + [3]
            sequences.append(seq)

        # padding the sequences

        return sequences

    def create_dataset(self, source_sequences, target_sequences):
        new_source_sequences = []
        new_target_sequences = []
        i = 0
        for source_sent, target_sent in zip(source_sequences, target_sequences):
            # Not include sentences that are above MAX SEQ length
            if(len(source_sent) > self.MAX_SEQ_LENGTH or len(target_sent) > self.MAX_SEQ_LENGTH):
                i += 1
                continue
            new_source_sequences.append(source_sent)
            new_target_sequences.append(target_sent)
        print(f"{i} sentences above seq_length")
        source_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(
            new_source_sequences, padding='post', maxlen=self.MAX_SEQ_LENGTH)
        target_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(
            new_target_sequences, padding='post', maxlen=self.MAX_SEQ_LENGTH)

        dataset = tf.data.Dataset.from_tensor_slices(
            (source_sequences_padded, target_sequences_padded))

        dataset = dataset.padded_batch(self.BATCH_SIZE).prefetch(
            tf.data.experimental.AUTOTUNE)

        return dataset


a = Data_Loader(os.getcwd() + '/data', 'wmt14/en-de',
                data_limit=10000, batch_size=16)
a.load()
