# AI Movie Recommendation System

## Project Description

This project is a simple movie recommendation system made using Python and Machine Learning.

It recommends movies that are similar to a movie entered by the user. The system uses movie information such as the overview, genres, keywords, cast, and director.

The project uses the **TMDB 5000 Movie Dataset**. Movie details are combined and converted into numerical data using `CountVectorizer`. **Cosine similarity** is then used to find movies that are most similar to the selected movie.

The final output shows the top 5 recommended movies.

## Instructions

### 1. Requirements

Make sure Python is installed on your computer.

Install the required libraries using:

```bash
pip install pandas scikit-learn
```

### 2. Dataset

Download the following files from the TMDB 5000 Movie Dataset:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

Keep both CSV files in the same folder as the notebook.

### 3. Open the Project

Open `movie_recommender.ipynb` using Jupyter Notebook or JupyterLab.

### 4. Run the Notebook

Run the cells from top to bottom.

Make sure the two CSV files are in the correct location before running the notebook.

## Usage

After running all the cells, the program will ask you to enter a movie name.

Example:

```text
Enter a movie name (or 'quit' to exit): Inception
```

The system will then display the **top 5 movies similar to Inception**.

You can enter another movie to get new recommendations.

To stop the program, enter:

```text
quit
```

### Example

```text
Enter a movie name (or 'quit' to exit): Inception

Recommended movies:
1. ...
2. ...
3. ...
4. ...
5. ...
```

If the entered movie is not available in the dataset, the program will show a message that the movie was not found.
