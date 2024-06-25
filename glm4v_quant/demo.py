import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

from io import BytesIO
import base64, json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)

query = '识别图中文本'
image = Image.open("data/handwritten.jpeg").convert('RGB')

buffer = BytesIO()
image.save(buffer, format="JPEG")
im_bytes = buffer.getvalue()
im_b64 = base64.b64encode(im_bytes).decode('utf-8')

tmp = json.dumps({"image": im_b64})
_tmp = json.loads(tmp)
image = _tmp["image"]
image = Image.open(BytesIO(base64.b64decode(image)))

inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # chat mode

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4v-9b",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='auto'
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
