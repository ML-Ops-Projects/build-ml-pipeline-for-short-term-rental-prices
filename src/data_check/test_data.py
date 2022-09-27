import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    actual_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(actual_columns)


def test_neighborhood_names(data):

    expected_neighborhoods = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    actual_neighborhoods = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(expected_neighborhoods) == set(actual_neighborhoods)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    is_outside_nyc = ~data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)
    assert np.sum(is_outside_nyc) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    
    Nota: 
    KL (Kullback–Leibler) divergence is a measure of similarity between 2 distributions
    
    """
    dist_1 = data['neighbourhood_group'].value_counts().sort_index()
    dist_2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    kl_divergence = scipy.stats.entropy(dist_1, dist_2, base=2)
    
    similar_distributions = kl_divergence < kl_threshold
    
    assert similar_distributions


########################################################
# Implement here test_row_count and test_price_range   #
########################################################
