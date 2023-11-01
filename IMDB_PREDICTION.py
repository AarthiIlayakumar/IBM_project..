
#DATA PROCESSING AND CLEANING:
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
movies = pd.read_csv('/content/imdb_dataset.csv')
movies
# Write your code for inspection here
movies.dtypes
#describe
movie.describe()
# Write your code for column-wise null count here
column_null = movies.isna().sum()
column_null
# Write your code for row-wise null count here
row_wise = movies.isnull().sum(axis=1)
row_wise
# Write your code for column-wise null percentages here
percent_missing = movies.isnull().sum() * 100 / len(movies)
percent_missing
movies.info()
#drop rows with missing values
movies=movies.dropna()
movies.info()
#checking null values
movies.isnull().values.any()
#shape
movies.shape
# Display the top N movies with the highest scores (e.g., top 10)
top_n = 10
top_movies = movies_sorted.head(top_n)
# Print the top movies
print("Movies with the Highest IMDb Scores:")
print(top_movies)
# Assuming the dataset has a 'Genre' column (adjust column name as needed)
genre_column = 'Genre'
# Count the occurrences of each genre
genre_counts = movies[genre_column].value_counts()
# Find the most popular genre
most_popular_genre = genre_counts.idxmax()
# Display the most popular genre and its count
print("Most Popular Genre:", most_popular_genre)
print("Count:", genre_counts.max())

#VISUALIZATION

#SCATTERPLOT
import matplotlib.pyplot as plt
# Choose the two variables for the scatter plot
x_variable = 'Runtime'  # Replace with the actual column name
y_variable = 'IMDB Score'     # Replace with the actual column name

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(movies[x_variable], movies[y_variable], alpha=0.5)
plt.title(f'Scatter Plot of {x_variable} vs. {y_variable}')
plt.xlabel(x_variable)
plt.ylabel(y_variable)
 #HISTOGRAM
import seaborn as sns
# Set the style for Seaborn plots (optional but makes plots more visually appealing)
sns.set(style="whitegrid")

# Create a histogram of IMDb scores
plt.figure(figsize=(10, 6))
sns.histplot(movies['IMDB Score'], kde=True, color='skyblue')
plt.title('Distribution of IMDb Scores')
plt.xlabel('IMDb Score')
plt.ylabel('Frequency')
plt.show()


# Assuming the dataset has a 'Language' column (adjust column name as needed)
language_column = 'Language'

# Filter the dataset to include only English-language movies
english_movies = movies[movies[language_column].str.contains('English', case=False)]

# Display the DataFrame with English-language movies
print("English-language Movies:")
print(english_movies)
# Find the movie with the highest runtime
highest_runtime_movie = movies[movies['Runtime'] == movies['Runtime'].max()]

# Find the movie with the lowest runtime
lowest_runtime_movie = movies[movies['Runtime'] == movies['Runtime'].min()]

# Print the movies with the highest and lowest runtimes
print("Movie with the Highest Runtime:")
print(highest_runtime_movie)

print("\nMovie with the Lowest Runtime:")
print(lowest_runtime_movie)
plt.grid(True)
plt.show()

#LINEAR REGRESSION :
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv('/content/imdb_dataset.csv')

# Select the features and target variable
X = data[['Runtime']]
y = data['IMDB Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
r2 = r2_score(y_test, y_pred)
print('R² score:', r2)

#EVALUTION USING LINEAR REGRESSION

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Assuming you have predictions (y_pred) and actual IMDb scores (y_test) from your model
# Replace 'y_pred' and 'y_test' with your actual variable names

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:",r2)
