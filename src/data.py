import os
import re
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

CLS = "[CLS]"
SEP = "[SEP]"
UKN = "[UKN]"
LANG = ['Tamil', 'English', 'Sweedish', 
        'Malayalam', 'Russian', 'Arabic',
        'Kannada', 'Greek', 'Dutch', 
        'Turkish', 'German', 'Italian', 
        'Danish', 'Portugeese', 'French', 
        'Spanish', 'Hindi']
LANG_DICT = {x: i for i, x in enumerate(LANG)}
LANG_LOOKUP = {v:k for k,v in LANG_DICT.items()}

class SentenceLanguageDataset(Dataset):
    # Wrapper to handle pd.DataFrame as torch.Dataset 
    def __init__(self, dataframe):
        self.text = dataframe["Text"]
        self.language = dataframe["Language"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sentence = self.text.iloc[idx]
        language = self.language.iloc[idx]
        return sentence, language


def info(df):
    print(df.iloc[0]["Text"])
    print("Total Data: ", len(df))
    print("Vocab Size: ", len(get_vocab(df["Text"])))
    print("Total sentence by Language:\n", df["Language"].value_counts())
    print("Mean sentence length: ", df["Text"].str.len().mean())
    print("Max  sentence length: ", df["Text"].str.len().max())
    print("Min  sentence length: ", df["Text"].str.len().min())
    print()


def get_vocab(text_column):
    # create vocabulary from text dataframe column
    d = {vocab: i+3 for i, vocab in enumerate(set([x for y in text_column for x in y]))}
    d[CLS] = 0
    d[SEP] = 1
    d[UKN] = 2
    return d


def create_data_split(data, test_perc=0.1, val_perc=0.2):
    num_test = round(len(data) * test_perc)
    num_val = round(len(data) * val_perc)
    # init mask with 0 (False)
    train_mask = torch.zeros(len(data), dtype=torch.bool)
    val_mask = torch.zeros(len(data), dtype=torch.bool)
    test_mask = torch.zeros(len(data), dtype=torch.bool)
    # index random permutation
    perm = torch.randperm(len(data))
    t = int(len(data)) - (num_val + num_test)
    train_mask[perm[:t]] = True
    val_mask[perm[t:t + num_test]] = True
    test_mask[perm[t + num_test:]] = True
    return train_mask, val_mask, test_mask


def collate_batch(batch):
    # custom collate function to generate offsets in iter DataLoader
    label_list, text_list, offsets = [], [], [0]
    for (text, label) in batch:
         label_list.append(label)
         text_list.append(text)
         offsets.append(text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # compute offset for each element in batch as cumulative sum
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # concatenate all sentence in the batch
    text_list = torch.cat(text_list)
    return text_list, label_list, offsets


def get_dataloader(df, batch_size=8, shuffle=True):
    # generate torch.DataLoader from pd.DataFrame
    train_mask, val_mask, test_mask = create_data_split(df)
    train = df[train_mask.tolist()]
    valid = df[val_mask.tolist()]
    test = df[test_mask.tolist()]
    return DataLoader(SentenceLanguageDataset(train), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch) \
          ,DataLoader(SentenceLanguageDataset(valid), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch) \
          ,DataLoader(SentenceLanguageDataset(test), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch) 

def clean_function(text):
    # Remove all special character and struct (http, @gmail, whitespaces etc)
    text = re.sub(r'[\([{})\]!@#$,"%^*?.:;~`0-9]', " ", text)
    text = text.lower()
    text = re.sub("http\S+\s*", " ", text)
    text = re.sub("RT|cc", " ", text)
    text = re.sub("#\S+", " ", text)
    text = re.sub("@\S+", " ", text)
    text = re.sub("\s+", " ", text)
    text = text.replace("'", "")  
    text = text.replace("\u200b", "")  
    return text


def divide_chunks(l, n):
    # split a list in sublist of len=n
    for i in range(0, len(l), n):
        yield (l[i:i + n])


def expand(df, max_len):
    # truncate sentence where len>max_len and return extended df
    rows = df["Text"].apply(lambda x: list(divide_chunks(x, max_len)))
    new = []
    for x, y, z in zip(rows, df["Language"], df["Text"]):
        for split in x:
            new.append({"Text": [CLS] + split + [SEP], "Language": y})
    return pd.DataFrame.from_records(new)


def less_frequent_word(words, threshold = 1):
    # retrieve all the words with frequency < threshold 
    bow = set(words)
    counter = dict((word,0) for word in bow)
    for word in words:
        counter[word]+=1
    return [k for k,v in counter.items() if v <= threshold]


def preprocessing(df, input_dim=16):
    # Drop Duplicates
    df.drop(df[df.duplicated()].index, axis=0, inplace=True)
    df.dropna(inplace=True)
    df_clean = df.copy()
    # Remove symbols, puntaction, etc
    df_clean["cleaned_Text"] = df_clean["Text"].apply(lambda x: clean_function(x))
    df_clean['Text'] = df_clean["cleaned_Text"]
    df_clean.drop(columns=["cleaned_Text"], inplace = True)
    # Sentence to list of words
    df_clean["Text"] = df_clean["Text"].str.split()
    # Store vocab
    vocab = get_vocab(df_clean["Text"])
    torch.save(vocab, "vocab.pt")
    # Add UKN to avoid Out of Dictionary during inference
    words = [x for y in df_clean["Text"] for x in y]
    mask = {value: (value if random.uniform(0, 1) > 0.5 else UKN) for value in less_frequent_word(words)}
    for word in words:
        mask.setdefault(word, word)
    df_clean["Text"] = df_clean["Text"].apply(lambda sentence: [mask[item] for item in sentence])
    # Categorical Language column to 
    df_clean["Language"] = df_clean["Language"].map(LANG_DICT)
    # Extract senteces that exceed max_len (-2 due CLS and SEP)
    max_len = input_dim - 2
    df_exceed = df_clean.loc[np.array(list(map(len,df_clean["Text"].values)))>max_len]
    # Trim long sentences and add rows from them
    df_new = expand(df_exceed, max_len)
    # Remove long sentences from source dataframe
    df_clean = df_clean.drop(df_exceed.index)
    # Add special tokens
    df_clean["Text"] = df_clean["Text"].apply(lambda sentence: [CLS] + sentence + [SEP])
    # Add new padded long sentences to source dataframe
    df_clean = pd.concat([df_clean, df_new])
    # Words to IDs using vocab
    df_clean["Text"] = df_clean["Text"].apply(lambda sentence: [vocab[x] for x in sentence])
    # IDs list to tensor
    df_clean["Text"] = df_clean["Text"].apply(lambda sentence: torch.tensor(sentence))
    assert len(df) - len(df_exceed) + len(df_new) == len(df_clean)
    return df_clean, vocab


def tokenize(sentences, vocab):
    # create tokenized sentence based on vocab
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = clean_function(sentence)
        sentence = sentence.split()
        sentence = [CLS]+sentence+[SEP]
        sentence = [vocab[word] if word in vocab else vocab[UKN] for word in sentence]
        tokenized_sentences.append(sentence)
    return torch.tensor(tokenized_sentences)