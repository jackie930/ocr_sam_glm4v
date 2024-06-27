import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from urllib.parse import urlparse
from djl_python import Input, Output

from io import BytesIO
import base64, json
from PIL import Image
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.system("pip install --upgrade pip")
os.system("pip install -i https://pypi.org/simple/ bitsandbytes")
os.system("pip install flash-attn==2.5.8 --no-build-isolation")

model_dict = None


def load_model(properties):
    model_id = properties["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    ).eval()
    
    return {"tokenizer": tokenizer, "model": model}

def handle(inputs: Input):
    global model_dict
    if model_dict is None:
        model_dict = load_model(inputs.get_properties())
        
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    
    if inputs.is_empty():
        return None
    
    data = inputs.get_as_json()
    
    image = data["image"]
    role = data["role"]
    query = data["query"]
    image = Image.open(BytesIO(base64.b64decode(image)))
    glm4v_input = [{"role": role, "image": image, "content":query}]
    glm4v_input = tokenizer.apply_chat_template(
        glm4v_input,
        add_generation_prompt=True, 
        tokenize=True, 
        return_tensors="pt",
        return_dict=True)
    glm4v_input = glm4v_input.to("cuda:0")
    gen_kwargs = data["gen_kwargs"]
    with torch.no_grad():
        outputs = model.generate(**glm4v_input, **gen_kwargs)
        outputs = outputs[:, glm4v_input['input_ids'].shape[1]:]
        
    output = tokenizer.decode(outputs[0])

    return Output().add(output.strip())
    
    