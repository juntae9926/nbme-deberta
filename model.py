import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel

class Network(nn.Module):
    def __init__(self, Config, config_file=None, pretrained=False):
        super().__init__()
        self.config = Config
        transformers.logging.set_verbosity_error() # warning 무시

        if config_file is None:
            self.config = AutoConfig.from_pretrained(Config.model, output_hidden_stats=True)
        else:
            self.config = torch.load(config_file)
        
        if pretrained == True:
            self.model = AutoModel.from_pretrained(Config.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        
        self.fc_dropout = nn.Dropout(Config.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
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