# DL-Lab Project : Diabetic Retinopathy detection

# Team04
- Name: Mohammed Jaseel Kunnathodika (st191717)
- Name: Yueyang Jiang (st186731)

# How to run the code
1. Run the resize.py 
2. Run the dataclass.py
3. Run the data_balance.py
4. Run the main.py file. with FLAGS.train = True
5. Run the main.py file with FLAGS.train = False
6. Update ensemble = True in main.py  for ensembled results.
7. Run the wandb_sweep.py for tuning the hyperparameters. (Please add comments to train_model.unfrz_layer, grad_cam_visualization.output_path and grad_cam_visualization.img_path in config.gin, before running this file or it will raise errors)


# Results
1. After running the resize.py, you can find the resized images in the path of "/home/RUS_CIP/st186731/revized_images/train" and "/home/RUS_CIP/st186731/revized_images/test".
2. After running the dataclass.py, you can find the output images in the path of '/home/RUS_CIP/st186731/revized_images/test/binary'.
3. After running the data_balance.py, you can find the output images in the path of '/home/RUS_CIP/st186731/augmented_images/train'.
4. After running the main.py file. with FLAGS.train = True, you can get the training accuracy, test(validation) accuracy and checkpoints for three models(mobilenet_like, vgg_like and inception_v2_like). After that, you can click the link of wandb to check the charts.
5. After running the main.py file with FLAGS.train = False, you can get the evaluation accuracy, sensitivity, specificity, precision, f1_score, confusion_matrix and the result of deep visualization (in the path of /home/RUS_CIP/st186731/dl-lab-24w-team04/output_grad_cam).
6. After updating ensemble = True in main.py, you can get the evaluation accuracy, sensitivity, specificity, precision, f1_score and confusion_matrix for ensembled results and the result of deep visualization  (in the path of /home/RUS_CIP/st186731/dl-lab-24w-team04/output_grad_cam).
7. After running the wandb_sweep.py, you can get the training accuracy, test(validation) accuracy and checkpoints for three models(mobilenet_like, vgg_like and inception_v2_like). After that, you can click the link of wandb in the result to get the hyperparameters you need and check the charts. 
# HAR
