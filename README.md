# ECE-5831 - Final Project
## Using Online Dataset:
In the report and presentation (linked at the end of this README), I mention two approaches. The first one is the Kaggle dataset approach which failed. I created a separate repository for that one since it would make this repository too big and disorganized. The link to that repository is [here](https://github.com/hosman-1/ece5831-project-failed)

## Using Custom Dataset:
The repository here is for the second approach, which is by creating my own custom dataset.

### Explanation of Files
Here I will explain what each file is:
1. __.gitignore:__ This is to ignore images, videos, and numpy arrays saved so it is easier to work with version control in vscode.
2. __generate_own_dataset.ipynb:__ Python notebook where I ran functions repeatedly whenever I wanted to add more data. It has a lot of lines so a cleaner version with no outputs will be uploaded.
3. __Pre_Processing_Data.ipynb:__ This is where I show all the work I did for pre-processing before I moved all the methods to a separate file to make it into a class.
4. __pre_process_asl.py:__ Script with a class to pre-process videos created using generate_own_dataset.ipynb.
5. __training_asl.ipynb:__ Notebook where I show all the work I did to train the model. I also do some predictions in this notebook. First I trained a model for 3 words to make sure it works, then 6 words, and finally for 11 words using a class from another script.
6. __training.py:__ This script has a class to train and run prediction on one video.
7. __live_asl_test.ipynb:__ In this python notebook, I was testing out the code to start a webcam video capture and do live predictions by loading the model created when training.
8. __live_asl.py:__ After testing the live demo code in the live_asl_test.ipynb file, I moved the code to a separate script. When this script is called, a webcam will start, and it will begin collecting frames until there is enough for a prediction. Then this prediction will be shown, as well as a sentence that consists of all predicted words so far. To stop the script, press q.
9. __words.txt:__ This text file has all the word IDs and the corresponding word in csv format. This is loaded when the live demo code is run.
10. __asl_words_3words_model_87acc.keras:__ This is the model I trained when I tested the training code with 3 words. It achieved 87% accuracy so I knew my custom data was working.
11. __asl_words_6words_2L2D_91acc.keras:__ This is the model I trained when I tested the training code with 6 words. It is 2 LSTM layers and 2 Dense layers. It achieved 91% accuracy.
12. __asl_11words_model_4L2D_93acc.keras:__ This is the model I trained when I tested the training code with 11 words. It is 4 LSTM layers and 2 Dense layers. It achieved 93% accuracy.
13. __final-project.ipynb:__ This is a notebook with cells that represent this approach in the project. It covers pre-processing, training, evaluating, and finally predicting words using the model on a live video feed.
14. __final-project-notebook-model.keras:__ This is the model generated using the final-project.ipynb notebook.

## External Links:
In this section are links to the presentation video, presentation, report, custom dataset (for kaggle one, go to the other repository. It is also in the same google drive folder as this dataset), and demo video.
1. __Presentation Video:__ [Link](https://youtu.be/y3EGdE80u74)
2. __Presentation:__ [Link](https://docs.google.com/presentation/d/1QyfIuLRMbldTGwg5HuSDN_agdpczvv_djvZfGYrBI6w/edit?usp=sharing)
3. __Report:__ [Link](https://drive.google.com/file/d/1RItNzhd6pOQZ_gZghmhS16FXc6LV-2Ev/view?usp=sharing)
4. __Custom Dataset:__ [Link](https://drive.google.com/file/d/1xNfAH_7lifdN-dyBdPI3FScjSefWJvk0/view?usp=sharing)
5. __Comprehensive Demo Video:__ [Link](https://youtu.be/X_iXuet0y6E)
6. __Google Drive Folder:__ [Link](https://drive.google.com/drive/folders/1qWAlCqedrBtXhsuGGBNwXs2KApk9t1WF?usp=sharing)