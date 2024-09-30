from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding, BitsAndBytesConfig, GenerationConfig
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union

class Chat:
  def __init__(self, model_name: str, device:str='cpu', quantized:bool=True) -> None:
    assert isinstance(model_name, str), "model_name must be a string"
    assert isinstance(device, str), "device must be a string"
    
    self.model_name = model_name
    self.device = device
    token = "" # INSERT YOUR TOKEN
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # to avoid an error
    self.generation_config = GenerationConfig.from_pretrained(model_name, token=token)
    if quantized:
      quantization_config = BitsAndBytesConfig(load_in_8bit=quantized)
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config, device_map='auto', token=token)
    else:
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', token=token)
    
  def ask(self, prompt:str | list, max_new_tokens:int=10) -> Union[str|list, torch.Tensor]:
    ## Return
    #  - response: string that contains the generated text
    #  - output.scores: list containing the scores of every token in the vucabulary for each model generation
    assert isinstance(prompt, str) or isinstance(prompt, list), "prompt must be a string or a list"
    assert isinstance(max_new_tokens, int), "max_new_tokens must be an int"
    
    with torch.no_grad():
      encoded_input = self.tokenizer(prompt, padding=True, return_tensors='pt').to(self.device)
      output = self.model.generate(
        **encoded_input,
        # generation_config=self.generation_config,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=self.tokenizer.eos_token_id,
      )
      responses = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
      output.scores = torch.stack(output.scores)    # generated_tokens x batch_size x scores
      probs = nn.Softmax(dim=2)(output.scores)
        
    return responses, probs
  
  def tokenize(self, string:str) -> BatchEncoding:
    assert isinstance(string, str), "string must be a string"
  
    return self.tokenizer.encode(string, return_tensors='pt', add_special_tokens=False)
  
  def detokenize(self, ids: Tensor | list) -> str:
    assert isinstance(ids, Tensor) or isinstance(ids, list), "ids must be a Tensor or a list"

    return self.tokenizer.batch_decode(ids, skip_special_tokens=False)
