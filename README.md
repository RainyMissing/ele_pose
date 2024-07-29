# ele_pose
端子排检测


## Dependencies

* Python == 3.11.8
* Pytorch == 2.2.2
* ultralytics == 8.1.46


We export our conda virtual environment as env_yolo.yaml. You can use the following command to create the environment.

```bash
conda env create -f env_yolo.yaml
```

## Dataset

You can find the dataset we used in our paper in the `./data` folder. This dataset is sufficient for training.




## Demo



If you want to test the performance, the model file `./test.pt` is already in the project root directory. You can use the following command to restore the test images:

```bash
python pred.py
```


## Training

If you want to re-train the model or use your own data, you need to first put the training set into the data_of_you/ folder, change the ta


and use the following command:

```bash
python train.py
```