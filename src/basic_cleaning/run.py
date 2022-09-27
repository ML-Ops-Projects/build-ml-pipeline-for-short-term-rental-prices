#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. 
    logger.info('Downloading input artifact')
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Load data
    logger.info('Loading data into pandas DataFrame')
    rentals_data = pd.read_csv(artifact_local_path, parse_dates=['last_review'])
    
    # Data Cleaning
    logger.info('START: Data Cleaning')

    # converting variables
    logger.info('Data Cleaning: converting variables to expected types')
    rentals_data['id'] = rentals_data['id'].astype(str)
    rentals_data['host_id'] = rentals_data['host_id'].astype(str)
    
    # Filtering price outliers 
    logger.info('Data Cleaning: Filtering outliers in "price"')
    custom_percentiles = [.01, .05, .1, .15, .25, .5, .75, .85, .9, .95, .99]
    
    min_price = args.min_price
    max_price = args.max_price
    is_not_outlier_price = rentals_data['price'].between(min_price, max_price)
    n_rows_before = rentals_data.shape[0]
    rentals_data = rentals_data[is_not_outlier_price]
    n_rows_after = rentals_data.shape[0]
    rows_dropped = n_rows_before - n_rows_after
    pct_rows_dropped = round((rows_dropped / n_rows_before) * 100, 2)

    logger.info(f"{rows_dropped=}, i.e {pct_rows_dropped=}%")
    
    # Filtering minimum_nights outliers
    logger.info('Data Cleaning: Filtering outliers in "minimum_nights"')
    minimun_nights_stats = rentals_data['minimum_nights'].describe(percentiles=custom_percentiles)
    outlier_str_locator = f"{round(args.outliers_percentile_threshold)}%"
    minimum_nights_max = minimun_nights_stats[outlier_str_locator]
    is_not_outlier_minimum_nights = rentals_data['minimum_nights'] <= minimum_nights_max

    n_rows_before = rentals_data.shape[0]
    rentals_data = rentals_data[is_not_outlier_minimum_nights]
    n_rows_after = rentals_data.shape[0]
    rows_dropped = n_rows_before - n_rows_after
    pct_rows_dropped = round((rows_dropped / n_rows_before) * 100, 2)  
    logger.info(f"{rows_dropped=}, i.e {pct_rows_dropped=}%")

    logger.info("END: Data Cleaning")
    
    logger.info("Dumping clean data to csv")
    rentals_data.to_csv(args.output_artifact, index=False)
    
    logger.info("Logging the csv as Artifact on W&B")
    artifact = wandb.Artifact(name=args.output_artifact,
                              type=args.output_type,
                              description=args.output_description)
    
    artifact.add_file('clean_sample.csv')
    
    run.log_artifact(artifact)

    run.finish()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument("input_artifact", type=str,
                        help="the name, i.e path of the input artifact, with ':latest' if using W&B",
                        )


    parser.add_argument(
        "output_artifact", 
        type=str,
        help="the name, i.e path of the output articat",
    )

    parser.add_argument(
        "output_type", 
        type= str,
        help=" The type of the ouput",
    )

    parser.add_argument(
        "output_description", 
        type=str,
        help="The description of the output",
    )

    parser.add_argument(
        "min_price", 
        type=float,
        help="The minimum price",
    )

    parser.add_argument(
        "max_price", 
        type=float,
        help="The maximum price)",
    )

    parser.add_argument(
        "outliers_percentile_threshold", 
        type=float,
        help="The percentile above which values minimum_nights will be discarded as outliers",
    )
    

    args = parser.parse_args()

    go(args)

