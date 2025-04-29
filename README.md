# Masked Image Modeling: A Survey
We surveyed recent studies on masked image modeling (MIM), an approach that emerged as a powerful self-supervised learning technique in computer vision. The MIM task involves masking some information, e.g. pixels, patches, or even latent representations, and training a model, usually an autoencoder, to predicting the missing information by using the context available in the visible part of the input. This repository categorizes the papers about masked image modeling, according to their contribution. The classifcation is based on our survey [Masked Image Modeling: A Survey](https://arxiv.org/abs/2408.06687).

## Summary

#### Single Level Contribution
1. [Masking Strategy](#1)
2. [Downstream Task](#2)
3. [Target Features](#3)
4. [Objective Function](#4)
5. [Model Architecture](#5)
6. [Theoretical Analysis](#6) 
___ 
#### Two Level Contribution
7. [Target Features and Objective Function](#7)
8. [Model Architecture and Objective Function](#8)
9. [Masking Strategy and Target Features](#9)
10. [Objective Function and Theoretical Analysis](#10)
11. [Downstream Task and Theoretical Analysis](#11)
12. [Masking Strategy and Objective Function](#12)
13. [Masking Strategy and Downstream Task](#13)
14. [Masking Strategy and Model Architecture](#14)
15. [Target Features and Downstream Task](#15)
16. [Model Architecture and Target Features](#16)
___ 
#### Multi-Level Contribution
17. [Masking Strategy, Model Architecture and Objective Function](#17)
18. [Masking Strategy, Target Features and Objective Function](#18)
19. [Masking Strategy, Model Architecture, Downstream Task and Objective Function](#19)

## Content

### Masking Strategy <a name="1"></a>
  1. [MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://papers.nips.cc/paper_files/paper/2022/file/944618542d80a63bbec16dfbd2bd689a-Paper-Conference.pdf) \
  [Link to code](https://github.com/voletiv/mcvd-pytorch)
  2. [Masked image training for generalizable deep image denoising](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Masked_Image_Training_for_Generalizable_Deep_Image_Denoising_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/haoyuc/MaskedDenoising)
  3. [Masked images are counterfactual samples for robust fine-tuning](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_Masked_Images_Are_Counterfactual_Samples_for_Robust_Fine-Tuning_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/Coxy7/robust-finetuning)
  4. [Hard patches mining for masked image modeling](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Hard_Patches_Mining_for_Masked_Image_Modeling_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/Haochen-Wang409/HPM)
  5. [What to hide from your students: Attention-guided masked image modeling](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900299.pdf) \
  [Link to code](https://github.com/gkakogeorgiou/attmask)
  6. [SMAUG: Sparse Masked Autoencoder for Efficient Video-Language Pre-training](https://openaccess.thecvf.com/content/ICCV2023/papers/Lin_SMAUG_Sparse_Masked_Autoencoder_for_Efficient_Video-Language_Pre-Training_ICCV_2023_paper.pdf)
  7. [SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders](https://proceedings.neurips.cc/paper_files/paper/2022/file/5c186016d0844767209dc36e9e61441b-Paper-Conference.pdf9) \
  [Link to code](https://github.com/ucasligang/SemMAE)
  8. [MedIM: Boost Medical Image Representation via Radiology Report-guided Masking](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_2) \
  [Link to code](https://github.com/YtongXie/MedIM)
  9. [autoSMIM: Automatic Superpixel-Based Masked Image Modeling for Skin Lesion Segmentation](https://pubmed.ncbi.nlm.nih.gov/37379178/) \
  [Link to code](https://github.com/Wzhjerry/autoSMIM)
  10. [Self-supervised learning with masked image modeling for teeth numbering, detection of dental restorations, and instance segmentation in dental panoramic radiographs](https://openaccess.thecvf.com/content/WACV2023/papers/Almalki_Self-Supervised_Learning_With_Masked_Image_Modeling_for_Teeth_Numbering_Detection_WACV_2023_paper.pdf) \
  [Link to code](https://github.com/AmaniHAlmalki/DentalMIM)
  11. [Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation](https://ojs.aaai.org/index.php/AAAI/article/view/25674/25446) \
  [Link to code](https://github.com/NSSL-SJTU/YaTC)
  12. [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/OpenGVLab/VideoMAEv2)
  13. [MRM: Masked Relation Modeling for Medical Image Pre-Training with Genetics](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_MRM_Masked_Relation_Modeling_for_Medical_Image_Pre-Training_with_Genetics_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/CityU-AIM-Group/MRM)
  14. [MA2CL: Masked attentive contrastive learning for multi-agent reinforcement learning](https://www.ijcai.org/proceedings/2023/0470.pdf) \
  [Link to code](https://github.com/song-hl/MA2CL)
  15. [Learning audio-visual speech representation by masked multimodal cluster prediction](https://openreview.net/pdf?id=Z1Qlm11uOM) \
  [Link to code](https://github.com/facebookresearch/av_hubert)
  16. [OmniMAE: Single Model Masked Pretraining on Images and Videos](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_OmniMAE_Single_Model_Masked_Pretraining_on_Images_and_Videos_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/facebookresearch/omnivore)
  17. [Good helper is around you: Attention-driven masked image modeling](https://ojs.aaai.org/index.php/AAAI/article/view/25269/25041)
  18. [MM-3DScene: 3D Scene Understanding by Customizing Masked Modeling with Informative-Preserved Reconstruction and Self-Distilled Consistency](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_MM-3DScene_3D_Scene_Understanding_by_Customizing_Masked_Modeling_With_Informative-Preserved_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/MingyeXu/mm-3dscene)
  19. [Masked autoencoders for point cloud self-supervised learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620591.pdf) \
  [Link to code](https://github.com/Pang-Yatian/Point-MAE)
  20. [Masked generative adversarial networks are data-efficient generation learners](https://proceedings.neurips.cc/paper_files/paper/2022/file/0efcb1885b8534109f95ca82a5319d25-Paper-Conference.pdf)
  21. [SimMIM: A Simple Framework for Masked Image Modeling](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.pdf) \
  [Link to code](https://github.com/microsoft/SimMIM)
  22. [Masked autoencoders are scalable vision learners](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf)
  23. [DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_DropMAE_Masked_Autoencoders_With_Spatial-Attention_Dropout_for_Tracking_Tasks_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/jimmy-dq/DropMAE.git)
  24. [MGMAE: Motion Guided Masking for Video Masked Autoencoding](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_MGMAE_Motion_Guided_Masking_for_Video_Masked_Autoencoding_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/MCG-NJU/MGMAE)
  25. [Multi-modal masked pre-training for monocular panoramic depth completion](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610372.pdf) \
  [Link to code](https://github.com/yanzq95/MMMPT)
  26. [Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training](https://arxiv.org/pdf/2209.07098) \
  [Link to code](https://github.com/zhjohnchan/M3AE)
  27. [Masked generative distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710053.pdf) \
  [Link to code](https://github.com/yzd-v/MGD)
  28. [Masked motion predictors are strong 3D action representation learners](https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/maoyunyao/MAMP)
  29. [Siamese masked autoencoders](https://proceedings.neurips.cc/paper_files/paper/2023/file/7ffb9f1b57628932518505b532301603-Paper-Conference.pdf) \
  [Link to project](https://siam-mae-video.github.io/)
  30. [MixMAE: Mixed and Masked Autoencoder for Efficient Pretraining of Hierarchical Vision Transformers](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_MixMAE_Mixed_and_Masked_Autoencoder_for_Efficient_Pretraining_of_Hierarchical_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/Sense-X/MixMIM)
  31. [Masked frequency modeling for self-supervised visual pre-training](https://openreview.net/pdf?id=9-umxtNPx5E) \
  [Link to code](https://github.com/Jiahao000/MFM)
### Downstream Task <a name="2"></a>
  1. [MaskGIT: Masked Generative Image Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf) \
  [Link to code](https://github.com/google-research/maskgit)
  2. [MAESTER: Masked Autoencoder Guided Segmentation at Pixel Resolution for Accurate, Self-Supervised Subcellular Structure Recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_MAESTER_Masked_Autoencoder_Guided_Segmentation_at_Pixel_Resolution_for_Accurate_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/bowang-lab/MAESTER)
  3. [GeoMIM: Towards Better 3D Knowledge Transfer via Masked Image Modeling for Multi-view 3D Understanding](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_GeoMIM_Towards_Better_3D_Knowledge_Transfer_via_Masked_Image_Modeling_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/Sense-X/GeoMIM)
  4. [AMAE: Adaptation of Pre-Trained Masked Autoencoder for Dual-Distribution Anomaly Detection in Chest X-Rays](https://arxiv.org/pdf/2307.12721)
  5. [Delving into masked autoencoders for multi-label thorax disease classification](https://openaccess.thecvf.com/content/WACV2023/papers/Xiao_Delving_Into_Masked_Autoencoders_for_Multi-Label_Thorax_Disease_Classification_WACV_2023_paper.pdf) \
  [Link to code](https://github.com/lambert-x/Medical_MAE)
  6. [Seeing beyond the brain: Conditional diffusion model with sparse masked modeling for vision decoding](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Seeing_Beyond_the_Brain_Conditional_Diffusion_Model_With_Sparse_Masked_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/zjc062/mind-vis)
  7. [Test-Time Training with Masked Autoencoders](https://papers.neurips.cc/paper_files/paper/2022/file/bcdec1c2d60f94a93b6e36f937aa0530-Paper-Conference.pdf) \
  [Link to code](https://github.com/yossigandelsman/test_time_training_mae)
  8. [Masked Image Modeling Advances 3D Medical Image Analysis](https://openaccess.thecvf.com/content/WACV2023/papers/Chen_Masked_Image_Modeling_Advances_3D_Medical_Image_Analysis_WACV_2023_paper.pdf) \
  [Link to code](https://github.com/ZEKAICHEN/MIM-Med3D)
  9. [Graph Masked Autoencoder Enhanced Predictor for Neural Architecture Search](https://www.ijcai.org/proceedings/2022/0432.pdf) \
  [Link to code](https://github.com/kunjing96/GMAENAS.git)
  10. [MAGVIT: Masked Generative Video Transformer](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/google-research/magvit)
  11. [Advancing Radiograph Representation Learning with Masked Record Modeling](https://openreview.net/pdf/fc3034053cc948a50f2822650fee36b83e7cb54b.pdf) \
  [Link to code](https://github.com/RL4M/MRM-pytorch)
  12. [Layer Grafted Pre-training: Bridging Contrastive Learning And Masked Image Modeling For Label-Efficient Representations](https://openreview.net/pdf?id=jwdqNwyREyh) \
  [Link to code](https://github.com/VITA-Group/layerGraftedPretraining_ICLR23.git)
  13. [Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology](https://openaccess.thecvf.com/content/CVPR2024/papers/Kraus_Masked_Autoencoders_for_Microscopy_are_Scalable_Learners_of_Cellular_Biology_CVPR_2024_paper.pdf) \
  [Link to code](https://github.com/recursionpharma/maes_microscopy)
  14. [Unleashing vanilla vision transformer with masked image modeling for object detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_Unleashing_Vanilla_Vision_Transformer_with_Masked_Image_Modeling_for_Object_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/hustvl/MIMDet)
  15. [MEGA: Masked Generative Autoencoder for Human Mesh Recovery](https://arxiv.org/pdf/2405.18839) \
  [Link to code](https://github.com/g-fiche/MEGA)
### Target Features <a name="3"></a>
  1. [Masked motion encoding for self-supervised video representation learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Sun_Masked_Motion_Encoding_for_Self-Supervised_Video_Representation_Learning_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/XinyuSun/MME)
  2. [MIMT: Masked Image Modeling Transformer for Video Compression](https://openreview.net/pdf?id=j9m-mVnndbm)
  3. [RILS: Masked Visual Reconstruction in Language Semantic Space](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_RILS_Masked_Visual_Reconstruction_in_Language_Semantic_Space_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/hustvl/RILS)
  4. [SdAE: Self-distillated Masked Autoencoder](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900107.pdf) \
  [Link to code](https://github.com/AbrahamYabo/SdAE)
  5. [Masked Autoencoders Enable Efficient Knowledge Distillers](https://openaccess.thecvf.com/content/CVPR2023/papers/Bai_Masked_Autoencoders_Enable_Efficient_Knowledge_Distillers_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/UCSC-VLAA/DMAE)
  6. [Mask3D: Pre-training 2D Vision Transformers by Learning Masked 3D Priors](https://openaccess.thecvf.com/content/CVPR2023/papers/Hou_Mask3D_Pre-Training_2D_Vision_Transformers_by_Learning_Masked_3D_Priors_CVPR_2023_paper.pdf)
  7. [EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_EVA_Exploring_the_Limits_of_Masked_Visual_Representation_Learning_at_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/baaivision/EVA/tree/master/EVA-01)
  8. [MeshMAE: Masked Autoencoders for 3D Mesh Data Analysis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630038.pdf) \
  [Link to code](https://github.com/liang3588/MeshMAE)
  9. [GeoMAE: Masked Geometric Target Prediction for Self-supervised Point Cloud Pre-Training](https://openaccess.thecvf.com/content/CVPR2023/papers/Tian_GeoMAE_Masked_Geometric_Target_Prediction_for_Self-Supervised_Point_Cloud_Pre-Training_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/Tsinghua-MARS-Lab/GeoMAE)
  10. [Masked Siamese networks for label-efficient learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910442.pdf) \
  [Link to code](https://github.com/facebookresearch/msn)
  11. [Point cloud domain adaptation via masked local 3D structure prediction](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630159.pdf) \
  [Link to code](https://github.com/VITA-Group/MLSP)
  12. [An empirical study of end-to-end video-language transformers with masked visual modeling](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_An_Empirical_Study_of_End-to-End_Video-Language_Transformers_With_Masked_Visual_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/tsujuifu/pytorch_empirical-mvm)
  13. [Masked Jigsaw Puzzle: A Versatile Position Embedding for Vision Transformers](https://openaccess.thecvf.com/content/CVPR2023/papers/Ren_Masked_Jigsaw_Puzzle_A_Versatile_Position_Embedding_for_Vision_Transformers_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/yhlleo/MJP)
### Objective Function <a name="4"></a>
  1. [MATE: Masked Autoencoders are Online 3D Test-Time Learners](https://openaccess.thecvf.com/content/ICCV2023/papers/Mirza_MATE_Masked_Autoencoders_are_Online_3D_Test-Time_Learners_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/jmiemirza/MATE)
  2. [MIMEx: Intrinsic Rewards from Masked Input Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/file/6fe10a4c0d680609f0560920bd9ade4a-Paper-Conference.pdf) \
  [Link to code](https://github.com/ToruOwO/mimex)
  3. [Exploring the role of mean teachers in self-supervised masked auto-encoders](https://openreview.net/pdf?id=7sn6Vxp92xV) \
  [Link to code](https://github.com/youngwanLEE/rc-mae)
  4. [Masked autoencoding does not help natural language supervision at scale](https://openaccess.thecvf.com/content/CVPR2023/papers/Weers_Masked_Autoencoding_Does_Not_Help_Natural_Language_Supervision_at_Scale_CVPR_2023_paper.pdf)
  5. [Masked Discrimination for Self-Supervised Learning on Point Clouds](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620645.pdf) \
  [Link to code](https://github.com/haotian-liu/MaskPoint)
  6. [Masked Frequency Consistency for Domain-Adaptive Semantic Segmentation of Laparoscopic Images](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_63) \
  [Link to code](https://github.com/MoriLabNU/MFC)
  7. [Masked unsupervised self-training for label-free image classification](https://openreview.net/pdf?id=ZAKkiVxiAM9) \
  [Link to code](https://github.com/salesforce/MUST)
  8. [Contrastive masked autoencoders are stronger vision learners](https://arxiv.org/pdf/2207.13532) \
  [Link to code](https://github.com/ZhichengHuang/CMAE)
### Model Architecture <a name="5"></a>
  1. [MCMAE: Masked Convolution Meets Masked Autoencoders](https://proceedings.neurips.cc/paper_files/paper/2022/file/e7938ede51225b490bb69f7b361a9259-Paper-Conference.pdf) \
  [Link to code](https://github.com/Alpha-VL/ConvMAE)
  2. [Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling](https://openreview.net/pdf?id=NRxydtWup1S) \
  [Link to code](https://github.com/keyu-tian/SparK)
  3. [Task-Customized Masked Autoencoder via Mixture of Cluster-Conditional Experts](https://openreview.net/pdf?id=j8IiQUM33s)
  4. [Self-Supervised Masked Convolutional Transformer Block for Anomaly Detection](https://arxiv.org/pdf/2209.12148) \
  [Link to code](https://github.com/ristea/ssmctb)
  5. [SparseMAE: Sparse Training Meets Masked Autoencoders](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_SparseMAE_Sparse_Training_Meets_Masked_Autoencoders_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/aojunzz/SparseMAE)
  6. [Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Ristea_Self-Supervised_Predictive_Convolutional_Attentive_Block_for_Anomaly_Detection_CVPR_2022_paper.pdf) \
  [Link to code](https://github.com/ristea/sspcab)
  7. [RevColV2: Exploring disentangled representations in masked image modeling](https://proceedings.neurips.cc/paper_files/paper/2023/file/5d56e69c317429945785ede86c00b44e-Paper-Conference.pdf) \
  [Link to code](https://github.com/megvii-research/RevCol)
### Theoretical Analysis <a name="6"></a>
  1. [Towards Understanding Why Mask Reconstruction Pretraining Helps in Downstream Tasks](https://openreview.net/pdf?id=PaEUQiY40Dk)
  2. [Rethinking Out-of-distribution (OOD) Detection: Masked Image Modeling is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Out-of-Distribution_OOD_Detection_Masked_Image_Modeling_Is_All_You_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/lijingyao20010602/MOOD)
  3. [Understanding Masked Autoencoders via Hierarchical Latent Variable Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Understanding_Masked_Autoencoders_via_Hierarchical_Latent_Variable_Models_CVPR_2023_paper.pdf)
  4. [Masked prediction: A parameter identifiability view](https://proceedings.neurips.cc/paper_files/paper/2022/file/85dd09d356ca561169b2c03e43cf305e-Paper-Conference.pdf)
  5. [On masked pre-training and the marginal likelihood](https://proceedings.neurips.cc/paper_files/paper/2023/file/fc0e3f908a2116ba529ad0a1530a3675-Paper-Conference.pdf) \
  [Link to code](https://github.com/pmorenoz/MPT-LML)
  6. [On Data Scaling in Masked Image Modeling](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_On_Data_Scaling_in_Masked_Image_Modeling_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/microsoft/SimMIM)
  7. [Revealing the dark secrets of masked image modeling](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Revealing_the_Dark_Secrets_of_Masked_Image_Modeling_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/zdaxie/MIM-DarkSecrets)
  8. [Improving Adversarial Robustness of Masked Autoencoders via Test-time Frequency-domain Prompting](https://openaccess.thecvf.com/content/ICCV2023/papers/Huang_Improving_Adversarial_Robustness_of_Masked_Autoencoders_via_Test-time_Frequency-domain_Prompting_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/shikiw/RobustMAE)
### Target Features and Objective Function <a name="7"></a>
  1. [Self-supervised visual representations learning by contrastive mask prediction](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Self-Supervised_Visual_Representations_Learning_by_Contrastive_Mask_Prediction_ICCV_2021_paper.pdf)
  2. [Masked contrastive representation learning for reinforcement learning](https://arxiv.org/pdf/2010.07470) \
  [Link to code](https://github.com/teslacool/m-curl)
  3. [Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder](https://openreview.net/pdf?id=RZGtK2nDDJ) \
  [Link to code](https://github.com/alinlab/MetaMAE)
  4. [MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining](https://openaccess.thecvf.com/content/CVPR2023/papers/Dong_MaskCLIP_Masked_Self-Distillation_Advances_Contrastive_Language-Image_Pretraining_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/LightDXY/MaskCLIP)
  5. [Generic-to-specific distillation of masked autoencoders](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Generic-to-Specific_Distillation_of_Masked_Autoencoders_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/pengzhiliang/G2SD)
  6. [Understanding masked image modeling via learning occlusion invariant feature](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Understanding_Masked_Image_Modeling_via_Learning_Occlusion_Invariant_Feature_CVPR_2023_paper.pdf)
  7. [Contrastive Masked Image-Text Modeling for Medical Visual Representation Learning](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_48) \
  [Link to code](https://github.com/cchen-cc/CMITM)
  8. [Position-Aware Masked Autoencoder for Histopathology WSI Representation Learning](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_69) \
  [Link to code](https://github.com/WkEEn/PAMA)
  9. [Masked image modeling with local multi-scale reconstruction](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Masked_Image_Modeling_With_Local_Multi-Scale_Reconstruction_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/huawei-noah/Efficient-Computing/tree/master/Self-supervised/LocalMIM)
### Model Architecture and Objective Function <a name="8"></a>
  1. [SupMAE: Supervised Masked Autoencoders Are Efficient Vision Learners](https://arxiv.org/pdf/2205.14540) \
  [Link to code](https://github.com/enyac-group/supmae)
  2. [Masked auto-encoders meet generative adversarial networks and beyond](https://openaccess.thecvf.com/content/CVPR2023/papers/Fei_Masked_Auto-Encoders_Meet_Generative_Adversarial_Networks_and_Beyond_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/parthagrawal02/MAE_GAN)
  3. [Stare at What You See: Masked Image Modeling without Reconstruction](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_Stare_at_What_You_See_Masked_Image_Modeling_Without_Reconstruction_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/OpenDriveLab/maskalign)
  4. [CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf) \
  [Link to code](https://github.com/antofuller/CROMA)
  5. [SwinMM: Masked Multi-view with Swin Transformers for 3D Medical Image Segmentation](https://arxiv.org/pdf/2307.12591) \
  [Link to code](https://github.com/UCSC-VLAA/SwinMM/)
  6. [Contrastive audio-visual masked autoencoder](https://openreview.net/pdf?id=QPtMRyk5rb) \
  [Link to code](https://github.com/yuangongnd/cav-mae)
  7. [VideoMAC: Video Masked Autoencoders Meet ConvNets](https://openaccess.thecvf.com/content/CVPR2024/papers/Pei_VideoMAC_Video_Masked_Autoencoders_Meet_ConvNets_CVPR_2024_paper.pdf) \
  [Link to code](https://github.com/NUST-Machine-Intelligence-Laboratory/VideoMAC)
### Masking Strategy and Target Features <a name="9"></a>
  1. [Masked retraining teacher-student framework for domain adaptive object detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/JeremyZhao1998/MRT-release)
  2. [Mx2M: masked cross-modality modeling in domain adaptation for 3D semantic segmentation](https://arxiv.org/pdf/2307.04231)
  3. [Diffusion models as masked autoencoders](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Diffusion_Models_as_Masked_Autoencoders_ICCV_2023_paper.pdf) \
  [Link to project](https://weichen582.github.io/diffmae.html)
  4. [Representation learning for visual object tracking by masked appearance transfer](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Representation_Learning_for_Visual_Object_Tracking_by_Masked_Appearance_Transfer_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/difhnp/MAT)
  5. [Joint-MAE: 2D-3D joint masked autoencoders for 3D point cloud pre-training](https://www.ijcai.org/proceedings/2023/0088.pdf)
  6. [Learning 3D representations from 2D pre-trained models via image-to-point masked autoencoders](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Learning_3D_Representations_From_2D_Pre-Trained_Models_via_Image-to-Point_Masked_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/ZrrSkywalker/I2P-MAE)
  7. [Compact transformer tracker with correlative masked modeling](https://ojs.aaai.org/index.php/AAAI/article/view/25327/25099) \
  [Link to code](https://github.com/HUSTDML/CTTrack)
  8. [Masked Embedding Modeling With Rapid Domain Adjustment for Few-Shot Image Classification](https://ieeexplore.ieee.org/document/10230026/)
  9. [Supervised masked knowledge distillation for few-shot transformers](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Supervised_Masked_Knowledge_Distillation_for_Few-Shot_Transformers_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/HL-hanlin/SMKD)
  10. [SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](https://arxiv.org/pdf/2504.00527) \ 
  [Link to code](https://github.com/fmthoker/SMILE)
### Objective Function and Theoretical Analysis <a name="10"></a>
  1. [How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders](https://papers.neurips.cc/paper_files/paper/2022/file/adb2075b6dd31cb18dfa727240d2887e-Paper-Conference.pdf) \
  [Link to code](https://github.com/zhangq327/U-MAE)
###  Downstream Task and Theoretical Analysis <a name="11"></a>
  1. [MaskSketch: Unpaired Structure-guided Masked Image Generation](https://openaccess.thecvf.com/content/CVPR2023/papers/Bashkirova_MaskSketch_Unpaired_Structure-Guided_Masked_Image_Generation_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/google-research/masksketch)
### Masking Strategy and Objective Function <a name="12"></a>
  1. [Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Masked_Scene_Contrast_A_Scalable_Framework_for_Unsupervised_3D_Representation_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/Pointcept/Pointcept)
  2. [Contextual image masking modeling via synergized contrasting without view augmentation for faster and better visual pretraining](https://openreview.net/pdf?id=A3sgyt4HWp) \
  [Link to code](https://github.com/Sherrylone/ccMIM)
  3. [HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception](https://proceedings.neurips.cc/paper_files/paper/2023/file/9ed1c94a6c87276f25ebb65231c86c3e-Paper-Conference.pdf) \
  [Link to code](https://github.com/junkunyuan/HAP)
  4. [MST: Masked Self-Supervised Transformer for Visual Representation](https://proceedings.neurips.cc/paper/2021/file/6dbbe6abe5f14af882ff977fc3f35501-Paper.pdf)
  5. [MAST: Masked Augmentation Subspace Training for Generalizable Self-Supervised Priors](https://openreview.net/pdf/a6ee04305991a9efa073850bebbec9317ac4de6d.pdf) \
  [Link to project](https://machinelearning.apple.com/research/mast)
  6. [Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.pdf) \
  [Link to code](https://github.com/lulutang0608/Point-BERT)
  7. [Contrastive masked autoencoders for self-supervised video hashing](https://arxiv.org/pdf/2211.11210) \
  [Link to code](https://github.com/huangmozhi9527/ConMH)
  8. [Masked image modeling with denoising contrast](https://openreview.net/pdf?id=1fZd4owfJP6) \
  [Link to code](https://github.com/TencentARC/ConMIM)
  9. [Denoising Masked AutoEncoders Help Robust Classification](https://openreview.net/pdf?id=zDjtZZBZtqK) \
  [Link to code](https://github.com/quanlin-wu/dmae)
### Masking Strategy and Downstream Task <a name="13"></a>
  1. [Masked Autoencoders As Spatiotemporal Learners](https://proceedings.neurips.cc/paper_files/paper/2022/file/e97d1081481a4017df96b51be31001d3-Paper-Conference.pdf) \
  [Link to code](https://github.com/facebookresearch/mae_st)
  2. [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf) \
  [Link to code](https://github.com/MCG-NJU/VideoMAE)
  3. [Tell me what happened: Unifying text-guided video completion via multimodal masked video generation](https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Tell_Me_What_Happened_Unifying_Text-Guided_Video_Completion_via_Multimodal_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/tsujuifu/pytorch_tvc)
  4. [Masked Spatio-Temporal Structure Prediction for Self-Supervised Learning on Point Cloud Videos](https://openaccess.thecvf.com/content/ICCV2023/papers/Shen_Masked_Spatio-Temporal_Structure_Prediction_for_Self-supervised_Learning_on_Point_Cloud_ICCV_2023_paper.pdf) \
  [Link to code]( https://github.com/JohnsonSign/MaST-Pre)
  5. [Global k-Space Interpolation for Dynamic MRI Reconstruction Using Masked Image Modeling](https://arxiv.org/pdf/2307.12672) \
  [Link to code](https://github.com/JZPeterPan/k-gin)
  6. [FocusMAE: Gallbladder Cancer Detection from Ultrasound Videos with Focused Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2024/papers/Basu_FocusMAE_Gallbladder_Cancer_Detection_from_Ultrasound_Videos_with_Focused_Masked_CVPR_2024_paper.pdf) \
  [Link to code](https://github.com/sbasu276/FocusMAE)
  7. [SMAE: Few-shot Learning for HDR Deghosting with Saturation-Aware Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_SMAE_Few-Shot_Learning_for_HDR_Deghosting_With_Saturation-Aware_Masked_Autoencoders_CVPR_2023_paper.pdf)
  8. [LEMaRT: Label-Efficient Masked Region Transform for Image Harmonization](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_LEMaRT_Label-Efficient_Masked_Region_Transform_for_Image_Harmonization_CVPR_2023_paper.pdf) \
  [Link to project](https://www.amazon.science/publications/lemart-label-efficient-masked-region-transform-for-image-harmonization)
  9. [MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_MAGE_MAsked_Generative_Encoder_To_Unify_Representation_Learning_and_Image_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/LTH14/mage)
  10. [MARLIN: Masked Autoencoder for facial video Representation LearnINg](https://openaccess.thecvf.com/content/CVPR2023/papers/Cai_MARLIN_Masked_Autoencoder_for_Facial_Video_Representation_LearnINg_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/ControlNet/MARLIN)
  11. [Multiple instance learning framework with masked hard instance mining for whole slide image classification](https://openaccess.thecvf.com/content/ICCV2023/papers/Tang_Multiple_Instance_Learning_Framework_with_Masked_Hard_Instance_Mining_for_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/DearCaat/MHIM-MIL)
  12. [PMatch: Paired Masked Image Modeling for Dense Geometric Matching](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_PMatch_Paired_Masked_Image_Modeling_for_Dense_Geometric_Matching_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/ShngJZ/PMatch)
  13. [Masked Autoencoders are Efficient Class Incremental Learners](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_Masked_Autoencoders_are_Efficient_Class_Incremental_Learners_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/scok30/MAE-CIL)
### Masking Strategy and Model Architecture <a name="14"></a>
  1. [Green hierarchical vision transformer for masked image modeling](https://papers.neurips.cc/paper_files/paper/2022/file/7e487c72fce6e45879a78ee0872d991d-Paper-Conference.pdf) \
  [Link to code](https://github.com/LayneH/GreenMIM)
  2. [CL-MAE: Curriculum-Learned Masked Autoencoders](https://openaccess.thecvf.com/content/WACV2024/papers/Madan_CL-MAE_Curriculum-Learned_Masked_Autoencoders_WACV_2024_paper.pdf) \
  [Link to code](https://github.com/ristea/cl-mae)
  3. [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/facebookresearch/ConvNeXt-V2)
  4. [4M: Massively Multimodal Masked Modeling](https://openreview.net/pdf?id=TegmlsD8oQ) \
  [Link to code](https://github.com/apple/ml-4m/)
  5. [Multi-modal Pathological Pre-training via Masked Autoencoders for Breast Cancer Diagnosis](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_44)
  6. [Unmasked Teacher: Towards Training-Efficient Video Foundation Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Unmasked_Teacher_Towards_Training-Efficient_Video_Foundation_Models_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/OpenGVLab/unmasked_teacher)
  7. [PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_PiMAE_Point_Cloud_and_Image_Interactive_Masked_Autoencoders_for_3D_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/BLVLab/PiMAE)
  8. [Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning](https://openaccess.thecvf.com/content/ICCV2023/papers/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/bair-climate-initiative/scale-mae)
  9. [MultiMAE: Multi-modal Multi-task Masked Autoencoders](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970341.pdf) \
  [Link to code](https://github.com/EPFL-VILAB/MultiMAE)
  10. [Multi-view masked world models for visual robotic manipulation](https://proceedings.mlr.press/v202/seo23a/seo23a.pdf) \
  [Link to code](https://github.com/younggyoseo/MV-MWM)
### Target Features and Downstream Task <a name="15"></a>
  1. [MAGVLT: Masked Generative Vision-and-Language Transformer](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_MAGVLT_Masked_Generative_Vision-and-Language_Transformer_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/kakaobrain/magvlt)
  2. [Uni4Eye: Unified 2D and 3D Self-supervised Pre-training via Masked Image Modeling Transformer for Ophthalmic Image Classification](https://arxiv.org/pdf/2203.04614) \
  [Link to code](https://github.com/Davidczy/Uni4Eye)
  3. [Deblurring masked autoencoder is better recipe for ultrasound image recognition](https://arxiv.org/pdf/2306.08249) \
  [Link to code](https://github.com/MembrAI/DeblurringMIM)
  4. [Masked Video Distillation: Rethinking Masked Feature Modeling for Self-Supervised Video Representation Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Masked_Video_Distillation_Rethinking_Masked_Feature_Modeling_for_Self-Supervised_Video_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/ruiwang2021/mvd)
  5. [T4P: Test-Time Training of Trajectory Prediction via Masked Autoencoder and Actor-Specific Token Memory](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_T4P_Test-Time_Training_of_Trajectory_Prediction_via_Masked_Autoencoder_and_CVPR_2024_paper.pdf) \
  [Link to code](https://github.com/daeheepark/T4P)
  6. [MAPSeg: Unified Unsupervised Domain Adaptation for Heterogeneous Medical Image Segmentation Based on 3D Masked Autoencoding and Pseudo-Labeling](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_MAPSeg_Unified_Unsupervised_Domain_Adaptation_for_Heterogeneous_Medical_Image_Segmentation_CVPR_2024_paper.pdf) \
  [Link to code](https://github.com/XuzheZ/MAPSeg/)
### Model Architecture and Target Features <a name="16"></a>
  1. [Masked Autoencoders Are Stronger Knowledge Distillers](https://openaccess.thecvf.com/content/ICCV2023/papers/Lao_Masked_Autoencoders_Are_Stronger_Knowledge_Distillers_ICCV_2023_paper.pdf)
  2. [Bootstrapped Masked Autoencoders for Vision BERT Pretraining](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900246.pdf) \
  [Link to code](https://github.com/LightDXY/BootMAE)
  3. [Cycle-consistent masked autoencoder for unsupervised domain generalization](https://openreview.net/pdf/80923f306be26ea8c160f03fb85027b691b34dbb.pdf)
  4. [Self-supervised Pre-training with Masked Shape Prediction for 3D Scene Understanding](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_Self-Supervised_Pre-Training_With_Masked_Shape_Prediction_for_3D_Scene_Understanding_CVPR_2023_paper.pdf)
  5. [Masked feature prediction for self-supervised visual pre-training](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf)
  6. [Masked Image Residual Learning for Scaling Deeper Vision Transformers](https://proceedings.neurips.cc/paper_files/paper/2023/file/b3bac97f3227c52c0179a6d967480867-Paper-Conference.pdf) \
  [Link to code](https://github.com/russellllaputa/MIRL)
  7. [Self-supervised 3D anatomy segmentation using self-distilled masked image transformer (SMIT)](https://arxiv.org/pdf/2205.10342) \
  [Link to code](https://github.com/The-Veeraraghavan-Lab/SMIT)
### Masking Strategy, Model Architecture and Objective Function <a name="17"></a>
  1. [AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders](https://openaccess.thecvf.com/content/CVPR2023/papers/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.pdf) \
  [Link to code](https://github.com/wgcban/adamae.git)
  2. [StrucTexTv2: Masked Visual-Textual Prediction for Document Image Pre-training](https://arxiv.org/pdf/2303.00289) \
  [Link to code](https://github.com/PaddlePaddle/VIMER/tree/main/StrucTexT)
  3. [MaskViT: Masked Visual Pre-Training for Video Prediction](https://openreview.net/pdf/519ede840ab88f1eeeef20446b915f73429dec70.pdf) \
  [Link to code](https://github.com/agrimgupta92/maskvit)
  4. [Masked AutoDecoder is Effective Multi-Task Vision Generalist](https://openaccess.thecvf.com/content/CVPR2024/papers/Qiu_Masked_AutoDecoder_is_Effective_Multi-Task_Vision_Generalist_CVPR_2024_paper.pdf) \
  [Link to code](https://github.com/hanqiu-hq/MAD)
  5. [Towards Latent Masked Image Modeling for Self-Supervised Visual Representation Learning](https://arxiv.org/pdf/2407.15837) \
  [Link to code](https://github.com/yibingwei-1/LatentMIM)
### Masking Strategy, Target Features and Objective Function <a name="18"></a>
  1. [One-for-All: Proposal Masked Cross-Class Anomaly Detection](https://openreview.net/pdf/c970527dd8148f2fe07c822e2d4f7c1f723e50f2.pdf) \
  [Link to code](https://github.com/xcyao00/PMAD)
  2. [Masked vision and language modeling for multi-modal representation learning](https://openreview.net/pdf?id=ZhuXksSJYWn)
  3. [Architecture-Agnostic Masked Image Modeling--From ViT back to CNN](https://proceedings.mlr.press/v202/li23af/li23af.pdf) \
  [Link to code](https://github.com/Westlake-AI/A2MIM)
  4. [Masked Feature Generation Network for Few-Shot Learning](https://www.ijcai.org/proceedings/2022/0513.pdf)
### Masking Strategy, Model Architecture, Downstream Task and Objective Function <a name="19"></a>
  1. [MAViL: Masked Audio-Video Learners](https://proceedings.neurips.cc/paper_files/paper/2023/file/40b60852a4abdaa696b5a1a78da34635-Paper-Conference.pdf) \
  [Link to code](https://github.com/facebookresearch/MAViL)
  2. [Improved masked image generation with token-critic](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830070.pdf)
  3. [Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors](https://openaccess.thecvf.com/content/CVPR2024/papers/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.pdf) \
  [Link to code](https://github.com/ristea/aed-mae)
  4. [Audiovisual masked autoencoders](https://openaccess.thecvf.com/content/ICCV2023/papers/Georgescu_Audiovisual_Masked_Autoencoders_ICCV_2023_paper.pdf) \
  [Link to code](https://github.com/google-research/scenic/tree/main/scenic/projects/av_mae)
