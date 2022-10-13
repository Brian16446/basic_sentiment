import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import triton_python_backend_utils as pb_utils

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_class_id = logits.argmax().item()
# print(model.config.id2label[predicted_class_id])


class TritonPythonModel:

    # @staticmethod
    # def auto_complete_config(auto_complete_model_config):
    #     # OPTIONAL
    #     pass

    def initialize(self, args):
        # OPTIONAL. Called once when model is loaded.
        print("initialised")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def execute(self, requests):
        # Must be implemented. Returns a list of pb_utils.InferenceRequest
        responses = []
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()]
            tokens = self.tokenizer(text=query[0], return_tensors="pt", return_attention_mask=False)
            print("tokens generated")
            with torch.no_grad():
                logits = self.model(**tokens).logits
                predicted_class_id = logits.argmax().item()
                responses.append(pb_utils.InferenceResponse(predicted_class_id))

        return responses

    # def finalize(self):
    #     # OPTIONAL. Runs on exit. Clean up.
    #     print("Cleaning up")
