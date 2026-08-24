# %%
import csv
import re
import polars as pl
from rapidfuzz import process, fuzz
from collections import defaultdict
import random
import re
import lzma
import numpy as np
from scipy.sparse import csr_matrix

# %%
ratings = (
    pl.read_csv("../week1/movieLense-100k/ratings.csv")
    .select(["userId", "movieId", "rating"])
    .with_columns(
        pl.col("userId").cast(pl.Int32),
        pl.col("movieId").cast(pl.Int32),
        pl.col("rating").cast(pl.Float32),
    )
)

users = ratings["userId"].to_numpy()
movies = ratings["movieId"].to_numpy()
stars = ratings["rating"].to_numpy()

# %%
print(movies)

# %%
#movieId,title,genres

class movie(object):
    def __init__(self,movie_id,title,generes):
        self.movie_id = movie_id
        self.title = title
        self.generes = generes
        self.plot_summary = None
        # can extend this into the tags as well


# %%
# enriched MovieLens-32M
df = pl.read_parquet("hf://datasets/krishnakamath/movielens-32m-movies-enriched/data/train-00000-of-00001.parquet")


# %%
def get_year(title):
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else None


def clean_title(title):
    title = re.sub(r"\(\d{4}\)\s*$", "", title)
    title = re.sub(r"\(a\.k\.a\..*?\)", "", title, flags=re.I)
    title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return " ".join(title.split())

# %%
exact_lookup = dict(zip(df["title"].to_list(), df["plot_summary"].to_list()))

year_lookup = {}

for title, summary in zip(df["title"].to_list(), df["plot_summary"].to_list()):
    if summary is None:
        continue

    year = get_year(title)

    if year is not None:
        year_lookup.setdefault(year, {})
        year_lookup[year][clean_title(title)] = summary

# %%
movies_information = []

with open("../week1/movieLense-100k/movies.csv", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        new_movie = movie(row["movieId"], row["title"], row["genres"])
        new_movie.plot_summary = exact_lookup.get(new_movie.title)
        movies_information.append(new_movie)

# %%
exact_lookup = dict(zip(df["title"].to_list(), df["plot_summary"].to_list()))

for m in movies_information:
    m.plot_summary = exact_lookup.get(m.title)

# %%
matched = sum(m.plot_summary is not None for m in movies_information)

print(f"Matched: {matched}/{len(movies_information)}")
print(f"Coverage: {matched / len(movies_information):.2%}")

# %%
users = ratings["userId"].to_numpy()
movies = ratings["movieId"].to_numpy()
stars = ratings["rating"].to_numpy()

R = csr_matrix(
    (stars, (users, movies)),
    shape=(users.max() + 1, movies.max() + 1),
    dtype=np.float32,
)

R_user = R
R_movie = R.tocsc()

# %%
def enclosing_subgraph(u, v, h, R_user, R_movie):
    U = {u}
    V = {v}

    U_fringe = {u}
    V_fringe = {v}

    for i in range(h):

        U_new = set()

        for movie in V_fringe:
            start = R_movie.indptr[movie]
            end = R_movie.indptr[movie + 1]

            U_new.update(R_movie.indices[start:end])

        U_new -= U

        V_new = set()

        for user in U_fringe:
            start = R_user.indptr[user]
            end = R_user.indptr[user + 1]

            V_new.update(R_user.indices[start:end])

        V_new -= V

        U_fringe = U_new
        V_fringe = V_new

        U |= U_fringe
        V |= V_fringe


    edges = []

    for user in U:
        start = R_user.indptr[user]
        end = R_user.indptr[user + 1]

        user_movies = R_user.indices[start:end]
        user_ratings = R_user.data[start:end]

        for movie, rating in zip(user_movies, user_ratings):

            movie = int(movie)

            if movie not in V:
                continue

            if user == u and movie == v:
                continue

            edges.append((user, movie, float(rating)))


    return {
        "users": U,
        "movies": V,
        "edges": edges,
    }

# %%
G = enclosing_subgraph(
    u=1,
    v=296,
    h=1,
    R_user=R_user,
    R_movie=R_movie,
)

# %%
print("Users:", G["users"])
print("Movies:", G["movies"])
print("Edges:", G["edges"][:20])

# %%
print(R_user)

# %%
# user's movies:
def users_movies(user):
    # user here being a number
    return list(zip(R_user.indices[R_user.indptr[user]:R_user.indptr[user+1]], R_user.data[R_user.indptr[user]:R_user.indptr[user+1]]))

# %%
user_1_movies = users_movies(1)

# %%
def get_user_movie_bins(u, R_user, movie_df):
    start, end = R_user.indptr[u], R_user.indptr[u + 1]

    user_ratings = pl.DataFrame({
        "movie_id": R_user.indices[start:end],
        "rating": R_user.data[start:end],
    }).with_columns(
        pl.when(pl.col("rating") >= 4).then(pl.lit("5-4"))
        .when(pl.col("rating") >= 3).then(pl.lit("4-3"))
        .when(pl.col("rating") >= 2).then(pl.lit("3-2"))
        .when(pl.col("rating") >= 1).then(pl.lit("2-1"))
        .otherwise(pl.lit("below-1"))
        .alias("rating_bin")
    )

    return user_ratings.join(movie_df, on="movie_id", how="left").sort("rating", descending=True)

# %%
new_rating = get_user_movie_bins(1, R_user, df)

# %%
new_rating = new_rating.filter(pl.col("movie_id") != 70)

# %%
def concat_movie_text(user_movies):
    text_cols = [c for c, dtype in zip(user_movies.columns, user_movies.dtypes) if dtype == pl.String]
    return "\n".join(" ".join(str(v) for v in row if v is not None) for row in user_movies.select(text_cols).iter_rows())

# %%
texts = {
    bin_name: concat_movie_text(new_rating.filter(pl.col("rating_bin") == bin_name))
    for bin_name in ["5-4", "4-3", "3-2", "2-1"]
}

# %%
print(texts)

# %%
encoded_data_5_4 = texts["5-4"].encode("utf-8")
raw_compress_5_4 = lzma.compress(encoded_data_5_4)
print(raw_compress_5_4.__sizeof__())

encoded_data_4_3 = texts["4-3"].encode("utf-8")
raw_compress_4_3 = lzma.compress(encoded_data_4_3)
print(raw_compress_4_3.__sizeof__())

encoded_data_3_2 = texts["3-2"].encode("utf-8")
raw_compress_3_2 = lzma.compress(encoded_data_3_2)
print(raw_compress_3_2.__sizeof__())

encoded_data_2_1 = texts["2-1"].encode("utf-8")
raw_compress_2_1 = lzma.compress(encoded_data_2_1)
print(raw_compress_2_1.__sizeof__())

# %%
to_test = movies_information[69]
new_info = f"{to_test.genres} + {to_test.plot_summary} + {to_test.title}"
print(new_info)

# %%
# so if 3B1B is right user 1, to movie 70 should be the most correct to be 3 stars.
new_encoded_data_5_4 = f"{texts['5-4']} {new_info}".encode("utf-8")
new_encoded_data_5_4 = lzma.compress(new_encoded_data_5_4)
print(new_encoded_data_5_4.__sizeof__()/raw_compress_5_4.__sizeof__())

new_encoded_data_4_3 = f"{texts['4-3']} {new_info}".encode("utf-8")
new_compressed_data_4_3 = lzma.compress(new_encoded_data_4_3)
print(new_compressed_data_4_3.__sizeof__()/raw_compress_4_3.__sizeof__())

new_encoded_data_3_2 = f"{texts['3-2']} {new_info}".encode("utf-8")
new_compressed_data_3_2 = lzma.compress(new_encoded_data_3_2)
print(new_compressed_data_3_2.__sizeof__()/raw_compress_3_2.__sizeof__())

new_encoded_data_2_1 = f"{texts['2-1']} {new_info}".encode("utf-8")
new_compressed_data_2_1 = lzma.compress(new_encoded_data_2_1)
print(new_compressed_data_2_1.__sizeof__()/raw_compress_2_1.__sizeof__())

# %%



