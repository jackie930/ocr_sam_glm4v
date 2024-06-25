# ocr_sam_glm4v
Build ocr process via sam+glm4v, running on sagemaker

## 部署流程

* 部署sam和glm4v成sagemaker endpoint
  * inference SAM on g5.xlarge, use huggingface container: Run [notebook.ipynb](/SAM/notebook.ipynb)
  * inference glm-4v-9b on g5.12xlarge, use djl container: Run [deploy-v2.ipynb](/glm4v/deploy-v2.ipynb)
  * inference glm-4v-9b-4bit on g5.2xlarge, use djl container: Run [deploy-quant.ipynb](/glm4v_quant/deploy-quant.ipynb)
  
## 运行

todo: fix this as a lambda program
```bash
python main.py --image_path IMAGE_PATH \
--glm4v_endpoint_name GLM4V_ENDPOINT_NAME \
--sam_endpoint_name SAM_ENDPOINT_NAME \
--output_folder FOLDER_NAME
```

## 推理成本:

期望成本: 1000张图10刀 * 3000 =30k (3kw张图)

现有优化方案成本: - 36*3000 = 108k

org: 
1000张图调用sam的成本 2.8s * 1000 = 47min, cost ~ $1.1 (47/60 * $1.408) , 调用glm4v的成本 2000min, cost ~ $236 (2000/60 * $7.09)

优化1: 从上往下拼接20张图调用glm4-v
1000张图调用sam的成本 2.8s * 1000 = 47min, cost ~ $1.1 (47/60 * $1.408) , 调用glm4v的成本 304min, cost ~ ¥$36 (304/60 * $7.09)

优化2: todo: 从上往下拼接20张图调用glm4-v + 量化模型

