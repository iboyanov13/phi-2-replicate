from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class Predictor:
    def __init__(self):
        base_model_id = "microsoft/phi-2"
        lora_path = "https://huggingface.co/pointserv/fanniemae-phi-2-lora"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
