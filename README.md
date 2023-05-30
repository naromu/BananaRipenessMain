# BananaRipenessEnsembleFlask


The following document provides a brief overview of a project involving the classification of banana maturity using artificial intelligence. The aim of this project is to predict the stage of maturity of a banana categorizing it as overripe, ripe, or unripe. The project utilizes various technologies, including Flask, Keras, Kotlin, AWS, GitHub, Mongo Atlas, and Google Colab.

Google Colab serves as the platform for training the models and can be accessed through the following URL: [Open Google Colab](https://colab.research.google.com/drive/1HmUNt9PbDQ4hqSq-m32hn3UtQ48FqBix?usp=sharing). Within the Google Colab notebook, the models employed for classification include Densenet121, Mobilenetv2, Resnet50v2, and Xception.

The project comprises three repositories hosted on GitHub:

1. Application for Android: The Android application, developed using Kotlin, can be found at the following URL: [Open BananaRipenessApp](https://github.com/Davix002/BananaRipenessApp.git). This application serves as the user interface for interacting with the banana maturity classification system.

2. Database Repository: The database repository contains the necessary data for managing the dataset used in training and testing. You can access the database repository at the following URL: [Open BananaRipenessDataset](https://github.com/Davix002/BananaRipenessDataset.git). 

3. Flask Server with Trained Models: this is The Flask server repository houses the server-side implementation responsible for hosting the trained models. It can be accessed at the following URL: [Open BananaRipenessEnsembleFlask](https://github.com/naromu/BananaRipenessEnsembleFlask.git). This server interacts with the Android application, receiving requests for banana maturity classification and providing the corresponding predictions.

Lastly, the cloud server used in this project is accessible via the following URL: [Open ServerBananaRipeness](http://ec2-18-205-229-14.compute-1.amazonaws.com:5000/). This cloud server hosts the necessary infrastructure for deploying and running the Flask server and its associated models.
