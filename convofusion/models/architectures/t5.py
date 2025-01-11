import torch
import os

from typing import List, Union
from torch import nn, Tensor



class T5TextEncoder(nn.Module):
    def __init__(self,
                 modelpath: str,
                 finetune: bool = False,
                 last_hidden_state: bool = False, # TODO: remove or use this
                 latent_dim: list = [1, 256],
                 **kwargs) -> None:

        super().__init__()

        from transformers import AutoTokenizer, T5EncoderModel
        from transformers import logging

        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.latent_dim = latent_dim #[-1]

        self.text_max_length = 200
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, model_max_length=self.text_max_length, use_fast=True)
        # breakpoint()
        self.tokenizer.add_special_tokens({'eos_token': '<eos>', 'bos_token': '<bos>', 'pad_token': '<pad>', 'unk_token': '<unk>'})
        # Text model
        self.text_model = T5EncoderModel.from_pretrained(modelpath)
        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        
        # self.text_encoded_dim = self.text_model.config.hidden_size
        self.text_encoded_dim = self.latent_dim  # enable projection
        # self.save_hyperparameters(logger=False)

        encoded_dim = self.text_model.config.hidden_size

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, self.latent_dim))

    def forward(self, texts: List[str], return_map: bool = False):
        # breakpoint()
        text_encoded, mask, token2word_map = self.get_last_hidden_state(texts,
                                                        return_map=return_map)
        
        # text_encoded = self.get_cache_or_embedding(texts)
        text_emb = self.projection(text_encoded)

        return text_emb, mask, token2word_map

    def get_cache_or_embedding(self, texts: List[str]):
        # define cache directory according to your setup
        cache_dir = './experiments/t5_text_cache/'
        encodings = []
        for text in texts:
            _hash = str(hash(text)) + '.pt'
            if os.path.exists(os.path.join(cache_dir, _hash)):
                text_encoded = torch.load(os.path.join(cache_dir, _hash))
            else:
                text_encoded, mask = self.get_last_hidden_state(text,
                                                        return_mask=True)
                torch.save(text_encoded, os.path.join(cache_dir, _hash))
            encodings.append(text_encoded)

        return torch.stack(encodings)
    
    def token_to_word_list(self, texts, token2word_map):
        # breakpoint()
        word_map_batch = []
        for idx, txt in enumerate(texts):
            txt_split = txt.split()
            txt_tokens = token2word_map[idx]
            word_map = [txt_split[j] if j is not None else '' for j in txt_tokens]
            word_map_batch.append(word_map)

        return word_map_batch

    def get_last_hidden_state(self,
                              texts: List[str],
                              return_map: bool = False
                              ):  #-> Union[Tensor, tuple[Tensor, Tensor]]:
        # breakpoint()
        texts = [f"<bos> {text} <eos>" if text != '-'*10 else text for text in texts ]
        encoded_inputs = self.tokenizer(texts,
                                        return_tensors="pt",
                                        padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        # if not return_mask:
        #     return output.last_hidden_state
        if not return_map:
            return output.last_hidden_state, encoded_inputs.attention_mask.to(
            dtype=bool), None
        
        token2word_map = [encoded_inputs.word_ids(i) for i in range(len(texts))]
        word_map = self.token_to_word_list(texts, token2word_map)
        
        
        return output.last_hidden_state, encoded_inputs.attention_mask.to(
            dtype=bool), word_map


if __name__ == "__main__":
    model = T5TextEncoder(modelpath='t5-base')
    
    a = model(['hello', 'world'])