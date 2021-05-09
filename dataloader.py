import os
import pandas as pd 
import spacy

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Target : text ---> numerical values

#Vocabulary to map each word to a index
#freq_threshold : min number of times a word should be repeated, to gain attention

class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy.eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


#Setup a PyTorch dataset to load the data
class FlickrDataset(Dataset):
    def __init__(self,root_dir,captions_file,transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        #Get imgs and caption columns from csv file
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #build Vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        caption = self.captions[index]
        img_id =  self.imgs[index]

        img = Image.open(os.path.join(self.root_dir,img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # We have the text but we can't directly use it
        
        # We have to convert it into numericalized version
        numericalized_caption = [self.vocab.stoi("<SOS>")]  #Start of Sentence
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi("<EOS>")) #End of sentence

        #now convert numericalized_caption to a tensor
        return img,torch.tensor(numericalized_caption)
    
 #Padding
 class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs,targets
    
     
