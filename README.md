# IMDb-dataset-sentiment-analysis-
Movie ratings prediction and sentiment analysis 


•	Implemented a Naïve Bayesian classifier to perform sentimental analysis on a collated dataset of 10,000 movies collected from multiple sources.

•	Modelled and implemented DNN, Support Vector Regressor, and Random Forest Regressor models to evaluate and predict the performance of a movie based on viewer’s sentiment.

•	Effected the prediction by increasing the accuracy by 15% than existing prediction systems.


# Repository Structure:
<ol>
<li>`IMDB data collection` - The directory 'IMDB data collection' contains the code used to fetch additional attributes from the IMDB API. It also contains a directory called 'review_data' which contains one csv file for each movie, and each csv file contains all reviews for that movie. Due to limitations by GitHub, only 1000 files are visible, but the total number of files are 9716.</li>
  
  <ol>
    <li> `collect_data.py` - This file contains the code for API calls to the IMdB API for attributes like actors, directors, box office, budget and reviews. The reviews are stored in a different folder named 'review_data' with movie id as the name for every file.</li> 
    <li> `trainset_2.xls' - This is our final dataset used by our regression models. This was converted from .csv to .xls to meet Github's max size limit </li>
  </ol>
  <br>
<li>`IMDB movie review sentiments` - This directory contains the Bag-of-Words Naive Bayes Classifier and the 'IMDB 50k reviews' dataset. The files included are:</li>
  <ul>
  <li>`NaiveBayesClassifier.py` - This file contains the actual Naive Bayes class. This classifier is made entirely using standard python libraries.</li>
  <li>`positive_rate.py` - This file calls the predict method of the Naive Bayes classifier for each movie in the dataset and creates a new attribute for the positive review percentage.</li>
  <li>`train.py` - This file was used to plot all the graphs and tables used in the report for the Bag-of-Words model.</li>
  </ul>
  <br>
   <li> `Data Pre-processing` - The directory contains the ipython notebook here we merge the movielns datasets and use the merged dataset for preprocessing. It also has the final    dataset we need in CSV format.</li>
   <br>
<li>`Notebooks` - This directory contains the implementations of various models. The files included here are:</li><br>
<ul>
   <li>`preprocessing.ipynb` - Contains data preprocessing part where we standardized and encoded the data to obtained our final dataset.</li>
   <li>`DNN.ipynb` - Contains DNN implementation using Pytorch.</li>
   <li>`SVR_RandomForest.ipynb` - Contains implementations of Support Vector Regressor and Random Forest Regressor for our dataset.</li>
 </ul>
 <br>
<li>`Reports` - The directory contains reports for milestone 1 and 2 i.e Project Proposal and Progress Report respectively, along with the final report.</li>
</ol>

