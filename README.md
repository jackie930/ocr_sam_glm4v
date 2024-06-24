# ocr_sam_glm4v
Build ocr process via sam+glm4v, running on sagemaker

## 部署流程

* 部署sam和glm4v成sagemaker endpoint
  * inference SAM on g5.xlarge, use huggingface container: Run [notebook.ipynb](/SAM/notebook.ipynb)
  * inference glm-4v-9b on g5.12xlarge, use djl container: Run [deploy-v2.ipynb](/glm4v/deploy-v2.ipynb)
  
## 运行

todo: fix this as a lambda program
```bash
python main.py --image_path IMAGE_PATH
```

