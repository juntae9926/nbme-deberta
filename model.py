import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel

class Network(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        transformers.logging.set_verbosity_error() # ignore warning 

        self.config = AutoConfig.from_pretrained(model_name, output_hidden_stats=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.fc_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.config.hidden_size, 1)
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
        output = self.fc(self.fc_dropout(feature))
        return output