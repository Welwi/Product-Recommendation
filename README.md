# Santander Product Recommendation
By: Edna Fernandes

## Project Goals/Overview
In 2016, Santander Bank was concerned with the fact that some customers got a lot of products recommended while others rarely got any recommendations. Therefore, Santander wanted to improve their product recommendation system, and only recommend the top 7 products that the customer would most likely purchase. This would help increase the bank's profits, while at the same time improving the customer's experience.

The final presentation can be found in the file 'Santander_Bank_Product_Recommendation'.

## README Summary
In this README I will present my data cleaning process, some insights drawn form my data analysis and the modeling results.

### Data Cleaning
The file with the training and test sets can be downlowaded from this link https://www.kaggle.com/c/santander-product-recommendation/data

The data cleaning process can be found on the utilities.py file.
The dataset was too large, and any little operation to it would take quite some time to run. Therefore, the first step of my cleaning process was to reduce the memory of the dataset. This can be found in the function DfLowMemory. 
Most of the missing values were replaced with the mode, and others, due to my understanding of the data, I was able to do some imputation that I thought were more appropriate. All of the steps and the reasoning are going to be described in more detail in the utilities.py file.


### Model
For modeling, instead of using the entire dataset, I selected only used the months of May2015, June2015 and May2016, as described by BreakfastPirate. This helped reduce the size of my dataset, and helped with the model training time.

There were 24 different products, and so I trained a different model for each of the products.
I treated the problem as a classification problem, and for every product, I predicted the probability of the customer to buy or not the product. 

I did my train test split and used AUC as my metric. For some products, the model does really well, with a score of 99% and others the model does just as good as guessing, with 50%. Either way, I am considering this a good starting point.
My current model can be founf on FinalModel.


### Next Steps:
Rank the products for each customer by their probabilities.

Suggest the top 7 products with highest probabilities that the customer does not have.

Seeing how much better neural networks would be for solving this problem.

