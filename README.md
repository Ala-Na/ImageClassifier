# ImageClassifier

## Technical exercise

Code, corrected dataset, best model and report are present in ```Technical Exercise``` folder.


A ```requirements.txt``` is present but not cleaned (a lot of package may be unnecessary).
Please, unzip ```corrected_dataset.zip``` and rename it ```dataset.zip```.

Images of data visualization prior / after training and losses evolution are available under ```png``` files.


To launch data analysis of dataset use the command:
```python3 data_analysis.py```


To launch training and evaluation of classifier use the command:
```python3 cnn_solution.py```

For cnn_solution, additionnal parameters are available from command line:

- --seed: fixed seed for random initialization (default=42)
- --image_size: size use for image resizing such as (image_size, image_size) (default=224)
- --n_epochs: number of epochs performed for training (default=40)
- --lr: learning rate for Adam optimizer (default=3e-4)
- --batch_size: size of batch (defaut=32)
- --filename: name of the experiment/to save model under (default="classifier")
- --force: to force retraining of the classifier even if --filename is already saved (default=False)

## Reading exercise

Article and report are present in ```Reading Exercise``` folder.