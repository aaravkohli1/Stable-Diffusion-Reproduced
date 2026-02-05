from typing import List, Union 

class CLIPTokenizerWrapper: 
    def __init__(self, hf_tokenizer, max_length=77):
        self.tok = hf_tokenizer
        self.max_length = max_length
    
    @classmethod
    def from_pretrained(cls, model_name, max_length): 
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        return cls(tokenizer, max_length=max_length)
    
    def __call__(self, texts: Union[str, List[str]], device=None):
        if isinstance(texts, str):
            texts = [texts]
        out = self.tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            out = {k: v.to(device) for k, v in out.items()}
        return out