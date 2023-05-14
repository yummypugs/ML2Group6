import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import umap
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore")

class Visualization:

    def __init__(self):
        self.export_data_path = "Exported_Data/"

    def plot_bar(self,X_val, Y_val, X_label, Y_label, Title, filename = ''):
        """
        Creates a bar chart for the given data.

        Args:
            X_val (array-like): The values for the x-axis.
            Y_val (array-like): The values for the y-axis.
            X_label (str): The label for the x-axis.
            Y_label (str): The label for the y-axis.
            Title (str): The title of the chart.
            filename (str, optional): The filename to save the chart as. Defaults to an empty string.

        """        
        # Create a bar chart for accuracy and nmi
        fig, ax = plt.subplots()
        ax.bar(X_val, Y_val)
        ax.set_xlabel(X_label)
        ax.set_ylabel(Y_label)
        ax.set_title(Title)     
        
        if len(filename) > 0:
            plt.savefig(f"{self.export_data_path}{filename}.png", dpi=600)

        plt.show()
    
    def plot_confusion_matrix(self,cm,  title='Confusion matrix', cmap=plt.cm.Blues, fsize=10, filename = ''):
        """
        Plots a confusion matrix.

        Args:
            cm (array-like): The confusion matrix.
            title (str, optional): The title of the plot. Defaults to 'Confusion matrix'.
            cmap (matplotlib.colors.Colormap, optional): The colormap to be used. Defaults to plt.cm.Blues.
            fsize (int, optional): The size of the figure. Defaults to 10.
            filename (str, optional): The filename to save the plot as. Defaults to an empty string.

        """
        fig, ax = plt.subplots(figsize=(fsize, fsize))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=1, vmax=cm.max()))

        tick_marks = np.arange(len(cm))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        # Add annotations
        thresh = cm.max() / 2.
        for i in range(len(cm)):
            for j in range(len(cm)):
                value = '{:.0f}'.format(cm[i, j])
                ax.text(j, i, value, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

        # Add axis labels and title
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(title)

        # Adjust layout
        fig.tight_layout()

        if len(filename) > 0:
            plt.savefig(f"{self.export_data_path}{filename}.png", dpi=600)

        # Show plot
        plt.show()

    def plot_explained_variance_ratio(self, counts_tfidf, maxComponents, filename = ''):
        """
        Plots the explained variance ratio vs. the number of components in a line plot.

        Args:
            counts_tfidf (scipy.sparse.csr_matrix): The TF-IDF matrix.
            maxComponents (int): The maximum number of components.
            filename (str, optional): The filename to save the plot as. Defaults to an empty string.

        """
        # Initialize SVD with different n_components values
        n_components_range = range(1, maxComponents, 500)
        explained_variance_ratios = []
        for n_components in n_components_range:
            svd = TruncatedSVD(n_components)
            svd.fit(counts_tfidf)
            explained_variance_ratios.append(svd.explained_variance_ratio_.sum())

        # create a line plot
        df = pd.DataFrame({'n_components': n_components_range, 'explained_variance_ratio': explained_variance_ratios})
        fig = px.line(df, x='n_components', y='explained_variance_ratio', title='Explained variance ratio vs. n_components')
        
        if len(filename) > 0:
            plt.savefig(f"{self.export_data_path}{filename}.png", dpi=600)

        fig.show()

    def plot_UMAP_3D(self, LSA_df, raw_df):
        """
        Performs dimensionality reduction using UMAP and plots a 3D visualization.

        Args:
            LSA_df (pandas.DataFrame): The LSA-transformed data.
            raw_df (pandas.DataFrame): The original data.

        """
        # perform dimensionality reduction using UMAP
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean', n_components=3)
        X_umap = reducer.fit_transform(LSA_df)
        le = LabelEncoder()
        categories = raw_df['category'].unique()
        categories = pd.DataFrame(categories)
        categories['label_encoded'] = le.fit_transform(categories.iloc[:,0])
        data = []
        for category in categories[0]:
            mask = raw_df['category'] == category
            data.append(go.Scatter3d(x=X_umap[mask, 0], y=X_umap[mask, 1], z=X_umap[mask, 2], 
                                     mode='markers', name=str(category), marker=dict(color=le.transform([category])[0], 
                                     colorscale='Viridis', opacity=0.8, size=2)))
        fig = go.Figure(data=data)
        fig.update_layout(title='UMAP 3D Visualization', 
                          scene=dict(xaxis_title='UMAP Dimension 1', 
                                     yaxis_title='UMAP Dimension 2', 
                                     zaxis_title='UMAP Dimension 3'))
        fig.show()

    def plot_TSNE_3D(self, LSA_df, raw_df):        
        """
        Performs dimensionality reduction using t-SNE and plots a 3D visualization.

        Args:
            LSA_df (pandas.DataFrame): The LSA-transformed data.
            raw_df (pandas.DataFrame): The original data.

        """
        tsne = TSNE(n_components=3, random_state=0, init='random')
        vectors_tsne = tsne.fit_transform(LSA_df)
        valid_indices = (vectors_tsne[:, 0] >= -100) & (vectors_tsne[:, 0] <= 100) & \
                        (vectors_tsne[:, 1] >= -100) & (vectors_tsne[:, 1] <= 100) & \
                        (vectors_tsne[:, 2] >= -100) & (vectors_tsne[:, 2] <= 100)
        vectors_tsne = vectors_tsne[valid_indices]
        raw_df = raw_df.iloc[valid_indices]
        # Generate a list of colors for the categories
        color_list = px.colors.qualitative.Plotly

        # Create plotly figure
        fig = px.scatter_3d(raw_df, x=vectors_tsne[:, 0], y=vectors_tsne[:, 1], z=vectors_tsne[:, 2], 
                            color='category', color_discrete_sequence=color_list)

        # Set the size of the dots
        dot_size = 2
        fig.update_traces(marker=dict(size=dot_size, opacity=0.8))

        # Set the title and axis labels
        fig.update_layout(title='LSA 3D Visualization', 
                          scene=dict(xaxis_title='Dim 1', 
                                     yaxis_title='Dim 2', 
                                     zaxis_title='Dim 3')) 
                                     #xaxis=dict(range=[-60, 60]),
                                     #yaxis=dict(range=[-60, 60]),
                                     #zaxis=dict(range=[-60, 60]))) 

        # Show plot
        fig.write_html("TSNE_LSA.html")
        fig.show()

    def plotMultiBar(self, modelResults, filename=''):
        """
        Plots a grouped bar chart to visualize performance metrics by model.

        Args:
            modelResults (dict): A dictionary containing model results.
                The keys are the model labels, and the values are dictionaries
                containing the performance metrics (accuracy, NMI, precision, recall).
            filename (str, optional): The filename to save the plot as. Defaults to an empty string.

        """
        # Create a list of the labels
        labels = list(modelResults.keys())

        # Create a list of the acc, nmi, and prec values for each label
        acc_values = [modelResults[label]['acc'] for label in labels]
        nmi_values = [modelResults[label]['nmi'] for label in labels]
        prec_values = [modelResults[label]['prec'] for label in labels]
        rec_values = [modelResults[label]['rec'] for label in labels]

        # Create dataframe
        data = {'Categories': labels,
                'Accuracy': acc_values,
                'NMI': nmi_values,
                'Precision': prec_values,
                'Recall': rec_values}
        df = pd.DataFrame(data)

        # Melt dataframe
        melted_df = df.melt(id_vars='Categories', var_name='Metrics', value_name='Values')

        # Plot bar chart
        sns.set_style('whitegrid')
        sns.set_context("talk")
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Categories', y='Values', hue='Metrics', data=melted_df, palette='husl')
        plt.ylim(0, 1)
        plt.title('Performance Metrics by Model')
        plt.xlabel('Model')
        plt.ylabel('Score')

        if len(filename) > 0:
            plt.savefig(f"{self.export_data_path}{filename}.png", dpi=600)

        # Show the plot
        plt.show()
