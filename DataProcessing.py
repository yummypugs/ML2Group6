import nltk
from nltk.corpus import reuters
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataProcessing:    
    def __init__(self, threshold = 30, SVDComponents = 2500):
        self.threshold = threshold
        self.SVDComponents = SVDComponents
        self.documents = reuters.fileids()
        self.list_filteredCategory = 0
        self.variance_ratio = 0
        self.finalReuters_df = None
        self.lsa_data_raw = None
        self.finalLSA_df = None
        self.counts_tfidf = None


    def initDataframe(self):
        """
        Initializes a DataFrame from the Reuters corpus.

        Returns:
            pandas.DataFrame: A DataFrame containing news articles categorized by 'category', 'subject', and 'content'.
        """
        #this creates a dataframe that splits it into 'category', 'subject' and 'content'
        data = []

        # Loop over each news article in the Reuters corpus
        for article_id in self.documents:
            # Get the categories, title, and text of the article
            categories = reuters.categories(article_id)
            subject, body = reuters.raw(article_id).split('\n', maxsplit=1)
            if len(categories)>1:
                continue
            # Add a new row for each category
            for category in categories:
                # Store the data in a dictionary
                data.append({'category': category, 'subject': subject, 'content': body})

        # Create a DataFrame from the data
        reuters_df = pd.DataFrame(data)
        return reuters_df
    
    def clean_text(self,text):    
        """
        Cleans the given text by replacing HTML symbols, removing escape sequences, punctuation,
        numbers, decimal places, and stopwords.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.

        """
        # Define a dictionary of html symbols and their replacements
        html_symbols = { '&lt;': '<', '&gt;': '>', '&amp;': '&', '&apos;': '\'', '&quot;': '\"' }
        # Replace html symbols with their corresponding characters
        for symbol, char in html_symbols.items():
            text = text.replace(symbol, char)
        
        # Remove escape sequences from the text
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        # Replace hyphens with spaces to preserve words
        text = text.replace('-', ' ')

        # Remove punctuation, numbers, and decimal places from the text
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ''.join(char for char in text if not char.isdigit() and char != '.')

        # split the text into words
        words = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in words if word.lower() not in stop_words]
        
        # join the words back into a string
        return ' '.join(words)
    
    def filterCategoryByThreshold(self, dataframe, threshold):
        """
        Filters the given DataFrame based on the specified threshold for category counts.

        Args:
            dataframe (pandas.DataFrame): The DataFrame to be filtered.
            threshold (int): The minimum count threshold for each category.

        Returns:
            pandas.DataFrame: The filtered DataFrame.

        """
        self.list_filteredCategory = dataframe["category"].value_counts()[dataframe["category"].value_counts()>=threshold].index.tolist()
        dataframe = dataframe.groupby('category').filter(lambda x: len(x) >= threshold)
        dataframe = dataframe.reset_index(drop=True)

        return dataframe
    
    def vectorize(self, dataframe):
        """
        Vectorizes the content of the given DataFrame using TF-IDF (Term Frequency-Inverse Document Frequency).

        Args:
            dataframe (pandas.DataFrame): The DataFrame containing the text content.

        Returns:
            scipy.sparse.csr_matrix: The TF-IDF matrix of the content.

        """
        vectorizer = TfidfVectorizer(max_df=0.3, # drop words that occur in more than X percent of documents
                             min_df=8, # only use words that appear at least X times
                             stop_words='english', 
                             lowercase=True, #convert everything to lower case 
                             use_idf=True,#we definitely want to use inverse document frequencies in our weighting
                             norm=u'l2', #Applies a correction factor so that longer paragraphs and shorter paragraphs get treated equally
                             smooth_idf=True #Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                            )

        #Pass pandas series to our vectorizer model
        self.counts_tfidf = vectorizer.fit_transform(dataframe.content)
        return self.counts_tfidf
    
    def runSVD(self, dataframe):
        """
        Runs Singular Value Decomposition (SVD) on the TF-IDF matrix.

        Args:
            dataframe (pandas.DataFrame): The DataFrame containing the text content.

        Returns:
            sklearn.decomposition.TruncatedSVD: The fitted SVD model.

        """
        self.vectorize(dataframe)
        svd = TruncatedSVD(self.SVDComponents)
        svd.fit(self.counts_tfidf)
        self.variance_ratio = svd.explained_variance_ratio_.sum()

        return svd
    
    def runLSA(self, dataframe, svd):
        """
        Runs Latent Semantic Analysis (LSA) on the TF-IDF matrix.

        Args:
            dataframe (pandas.DataFrame): The DataFrame containing the text content.
            svd (sklearn.decomposition.TruncatedSVD): The fitted SVD model.

        Returns:
            pandas.DataFrame: The LSA-transformed data with category information.

        """
        lsa = make_pipeline(svd, Normalizer(copy=False)) # LSA is normalizing
        lsa_data = lsa.fit_transform(self.counts_tfidf)
        self.lsa_data_raw = pd.DataFrame(lsa_data)
        df = dataframe["category"].copy()
        lsa_category = pd.concat([df, self.lsa_data_raw], axis=1)
        return lsa_category
    
    def createCSV(self, dataframe, filename):
        """
        Creates a CSV file from the given DataFrame.

        Args:
            dataframe (pandas.DataFrame): The DataFrame to be saved as CSV.
            filename (str): The name of the CSV file to be created.

        """
        dataframe.to_csv(filename)

    def fullGenerate(self):
        """
        Performs the full generation process, including initializing the DataFrame, cleaning text,
        filtering by category threshold, running SVD, and running LSA.

        Returns:
            pandas.DataFrame: The LSA-transformed data with category information.

        """
        dataframe = self.initDataframe()
        
        dataframe['content'] = dataframe['content'].apply(self.clean_text)
        dataframe['subject'] = dataframe['subject'].apply(self.clean_text)

        self.finalReuters_df = self.filterCategoryByThreshold(dataframe, self.threshold)
        svd = self.runSVD(self.finalReuters_df)
        self.finalLSA_df = self.runLSA(self.finalReuters_df, svd)

        return self.finalLSA_df

    def splitDataFrame(self, dataframe):
        """
        Splits the given DataFrame into features (X) and target (y) for training and testing purposes.

        Args:
            dataframe (pandas.DataFrame): The DataFrame to be split.

        Returns:
            pandas.DataFrame: The features (X) DataFrame.
            pandas.Series: The target (y) Series.
            pandas.DataFrame: The training features (X_train) DataFrame.
            pandas.DataFrame: The testing features (X_test) DataFrame.
            pandas.Series: The training target (y_train) Series.
            pandas.Series: The testing target (y_test) Series.

        """
        y = dataframe['category']
        X = dataframe.drop(columns=['category'])

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=0,
                                                            stratify=y)
        return X, y, X_train, X_test, y_train, y_test
    
    def splitDataFrameEncoded(self, dataframe):
        """
        Splits the given DataFrame into encoded features (X) and target (y) for training and testing purposes.

        Args:
            dataframe (pandas.DataFrame): The DataFrame to be split.

        Returns:
            pandas.DataFrame: The encoded features (X) DataFrame.
            numpy.ndarray: The encoded target (y) array.
            pandas.DataFrame: The training features (X_train) DataFrame.
            pandas.DataFrame: The testing features (X_test) DataFrame.
            numpy.ndarray: The training target (y_train) array.
            numpy.ndarray: The testing target (y_test) array.

        """
        y = dataframe['category']
        le = LabelEncoder()
        y = le.fit_transform(y)
        X = dataframe.drop(columns=['category'])

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=0,
                                                            stratify=y)
        return X, y, X_train, X_test, y_train, y_test


