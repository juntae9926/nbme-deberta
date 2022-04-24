import os
import torch
import json
import torch.nn as nn
from transformers import GPT2LMHeadModel

class BioMegatron(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = GPT2LMHeadModel.from_pretrained('nvidia/biomegatron')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.fc = nn.Linear(len(self.tokenizer), 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0)
            if module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output