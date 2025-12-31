# Dual-Domain Semantic Decoupling with Multi-Scale Feature Interaction for Remote Sensing Image Change Captioning
<img width="1978" height="694" alt="主干架构5" src="https://github.com/user-attachments/assets/9f720a61-ef85-43b9-9424-7f7547781f2b" />

# Set up
 conda create -n DDD_Net python=3.9
 pip install -r requirements.txt
# Dataset
[LEVIR_CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)
[Dubai_CC](https://disi.unitn.it/~melgani/datasets.html)
The data preprocessing procedure follows the same settings as those adopted in the [RSICCformer](https://github.com/Chen-Yang-Liu/RSICC) project.
# Training
The model can be trained using the script below:
 python train.py
This script handles data loading, preprocessing, model training, and checkpoint saving.
# Test
To evaluate the trained model, run:
 eval.py
This script loads the trained model and reports the evaluation results on the test dataset.
# Models_checkpoint
BEST_checkpoint_dinov2_vitl14_DualDomainTransformer_trans.pth.tar
https://pan.baidu.com/s/1im3u6F34KneWMhf5ifoyig?pwd=n5yz code: n5yz
# experiment results
<img width="1556" height="801" alt="可视化效果3" src="https://github.com/user-attachments/assets/29122f81-c777-4c0a-b746-277a5adf159b" />
<img width="1483" height="787" alt="可视化描述" src="https://github.com/user-attachments/assets/749e7bac-bdd4-4c16-a0e4-938abb0cf025" />
<img width="1455" height="495" alt="image" src="https://github.com/user-attachments/assets/2b3efb3a-dc94-4766-965a-c6ee5d581245" />

# Acknowledgements
We would like to sincerely thank the authors of [RSICCformer](https://github.com/Chen-Yang-Liu/RSICC) for their valuable contribution in releasing their code to the public.
