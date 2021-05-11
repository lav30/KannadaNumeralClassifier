## Kannada Handwritten Digit Recognizer

![Alt Text](kannadadigits.gif)


This project has focused on training a CNN model on handwritten Kannada numerals and also on making predictions in real time using a GUI developed by [Gradio](https://www.gradio.app). Live predictions are made by drawing digits 0 to 9 in the Kannada language on the drawing pad and the trained model is utilised to make accurate predictions in real time.

Owing to the dataset size and limitations of web deployment , Heroku deployment is currently unavailable. However, local deployment is done using Gradio. 


## Project Description


The dataset can be accessed on [Kaggle](https://www.kaggle.com/c/Kannada-MNIST/data)

### <h3 align="center" id="Multilabel Classification : High Level View of the Model">Multilabel Classification : High Level View of the Model</h3>
![Alt Text](CNNML.png)

[Image created using NN-SVG](https://alexlenail.me/NN-SVG/)


## Results 

Several performance metrics have been utilised to determine the model generalization capability on the test set. Accuracy, precision and recall are a few well known metrics used for image data. Other metrics such as the ROC curve can also be used to determine how well the model generalizes. 


## Citation 

1. Prabhu, V. (2019, August 03). Kannada-MNIST: A new handwritten DIGITS dataset for the Kannada language. Retrieved April 19, 2021, from https://arxiv.org/abs/1908.01242v1


2. [NN-SVG rendering of the model in AlexNet style](https://alexlenail.me/NN-SVG/) 

## License

MIT 
