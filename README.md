# Custom Components

This repository contains a collection of machine-learning components for various tasks.

## Components and Tasks

1. **Vector Assembler**

   - This component allows you to easily combine multiple feature columns into a single feature vector.

2. **Split Data**

   - This component allows you to specify the percentage of data allocated for training and testing.

3. **Train Linear Regression**

   - This component is designed to train your linear regression model.
   - You can specify various model parameters such as:
     - Target column
     - Maximum iterations
     - Regularization parameter
     - Elastic net mixing parameter
   - The trained model will be saved to a location specified by `modelPath`. For example: `modelPath="dbfs:/FileStore/lr_model"`

4. **Predictions Linear Regression**

   - This component is designed to make predictions based on loading a pre-trained model and applying it to new data for prediction.

5. **Evaluations Linear Regression**

   - This component is designed to evaluate the performance of the linear regression model by providing the Root Mean Squared Error (RMSE).

6. **Load Data from Hugging Face**

   - This component is designed to load datasets from the Hugging Face's datasets library and prepare data for Natural Language Processing (NLP).

7. **Convert Word to Vector**

   - This component is designed to generate word embeddings from text data.
   - You can configure word2vector parameters such as:
     - Input Column
     - Output Column
     - Vector Size
     - Minimum Count
     - Number of partitions
     - Step size
     - Maximum Iterations
     - Window Size
     - Max Sentence Length
     - Model path
   - Once word embeddings are created, you can save the trained model to a specified location for later use. For example: `modelPath="dbfs:/FileStore/trainedModel"`
