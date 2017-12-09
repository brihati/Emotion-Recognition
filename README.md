# Emotion-Recognition
This repository deals with the recognition of emotions by extracting the facial expressions. I used Convolutional Neural Network for the same using the tflearn library. In total, I detected seven degrees of emotions: Angry, Disgusted, Happy, Fearful, Neutral, Sad and Surprised.
My model achived an accuracy of 75% using the <a href="http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main"> FER 2013 Dataset.</a>

<h2>Preprocessing</h2>
In the preprocessing step, I scaled the images using standard deviation which was caculated from the sample data. Also, I zero sampled the images using the mean calculated from the sample data. These pre-processing methods were used both during the training and testing time.

<h2> CNN Model</h2>

<p>self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1],
                              data_preprocessing = img_preProcess)</p>
<p>self.network = conv_2d(self.network, 64, 5, activation = 'relu')</p>
<p>self.network = max_pool_2d(self.network, 3, strides = 2)</p>
<p>self.network = conv_2d(self.network, 64, 5, activation = 'relu')</p>
<p>self.network = max_pool_2d(self.network, 3, strides = 2)</p>
<p>self.network = conv_2d(self.network, 128, 4, activation = 'relu')</p>
<p>self.network = dropout(self.network, 0.3)</p>
<p>self.network = fully_connected(self.network, 3072, activation = 'relu')</p>
<p>self.network = fully_connected(self.network, len(EMOTIONS), activation ='softmax')</p>
<p>self.network = regression(self.network,
  optimizer = 'momentum',
  loss = 'categorical_crossentropy',learning_rate=0.001)</p>

<h2>Usage</h2>
<p>1) Install python3, tensorflow, anaconda and pandas in your system.</p>
<p>2) The default configuration used for the training can be seen in emotion_cnn.py</p>
<p>3) The dataset has 11179 images in total in which 20% is validation images and rest 80 validation images</p>
<p>4) To run the code: run the below command by going into emotion_recognition folder</p>
   python -c "import model.emotion_cnn"
<h2>Results</h2>
The accuracy of the model is 75%. I implemented a live application too through which you can detect emotions through a live video feed by using webcam
