import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

R_tokenizer = XLMRobertaTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')
premise = "Jupiter's Biggest Moons Started as Tiny Grains of Hail"
hypothesis = "This text is about space & cosmos"

input_ids = R_tokenizer.encode(premise, hypothesis, return_tensors='pt', max_length=256, truncation=True, padding='max_length')

mask = input_ids != 1
mask = mask.long()

class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self):
        super(PyTorch_to_TorchScript, self).__init__()
        self.model = XLMRobertaForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
    def forward(self, data, attention_mask=None):
        return self.model(data, attention_mask)
    
pt_model = PyTorch_to_TorchScript().eval()
traced_script_module = torch.jit.trace(pt_model, (input_ids, mask), strict=False)
traced_script_module.save("model.pt")