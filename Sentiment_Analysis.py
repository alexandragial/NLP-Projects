import nltk
from nltk.corpus import movie_reviews  #Import Dataset
from nltk.corpus import stopwords
import string
from random import shuffle
from nltk import classify
from nltk import NaiveBayesClassifier
from sklearn.linear_model import LogisticRegression

"""Download packages"""

nltk.download('movie_reviews')
nltk.download('stopwords')

"""Create dictionary of clean words"""

def createDictWords(reviews):
  wordsDict = dict()
  for word in reviews :
    wordsDict[word] = True

  return wordsDict
  
"""Data Cleaning function"""

def dataCleaning(allWords):
  cleanReviews = list()
  stopWords = stopwords.words('english')

  for word in allWords:
    if word not in stopWords and word not in string.punctuation and word[0].isalpha():
      cleanReviews.append(word.lower())

  cleanReviews = createDictWords(cleanReviews)

  return cleanReviews
  
"""Tag reviews as positive or negative"""

def tagReviews():
  posReviews = []
  for fileid in movie_reviews.fileids('pos'):
      words = movie_reviews.words(fileid)
      posReviews.append(words)

  negReviews = []
  for fileid in movie_reviews.fileids('neg'):
      words = movie_reviews.words(fileid)
      negReviews.append(words)

  return posReviews, negReviews
  
"""Feature Extraction"""

def extractFeatures():

  posReviews, negReviews = tagReviews()

  posReviewSet = []
  for words in posReviews:
      posReviewSet.append((dataCleaning(words), 'positive'))

  negReviewSet = []
  for words in negReviews:
      negReviewSet.append((dataCleaning(words), 'negative'))

  return posReviewSet, negReviewSet

def main():
  # All words from the reviews
  reviewsWords = movie_reviews.words()
  # Extract Features
  posFeatures, negFeatures = extractFeatures()

  # Split into training & test sets
  trainData = posFeatures[:750] + negFeatures[:750]
  testData = posFeatures[750:] + negFeatures[750:]

  # Shuffle positive & negative results
  shuffle(trainData)
  shuffle(testData)

  # Classify reviews
  nb = NaiveBayesClassifier.train(trainData)
  print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(nb, testData))*100)
  nb.show_most_informative_features(15)

  LogisticRegression_classifier = classify.SklearnClassifier(LogisticRegression())
  LogisticRegression_classifier.train(trainData)
  print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testData))*100)

if __name__=="__main__":
  main()  