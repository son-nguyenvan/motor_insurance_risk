import ast
import os

import pandas as pd
import numpy as np

from motor_insurance_risk.data_processor.processor import DataProcessor
from motor_insurance_risk.database.connection import DatabaseConnection
from motor_insurance_risk.embeddings.utils import calculate_total_embeddings_cost
from motor_insurance_risk.utils.logger import logger

def process_csv_to_embeddings(input_file, output_file):
    """
    Process the input CSV to generate embeddings and save to output file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the processed CSV with embeddings.
    """
    logger.info(f"Processing CSV file for embeddings: {input_file}")
    
    processor = DataProcessor()
    
    df = pd.read_csv(input_file)

    total_cost = calculate_total_embeddings_cost(df)
    logger.info(f"Total cost for all the content transformed into embeddings = $ {total_cost}")
    
    processed_df = processor.process_dataframe(df)
    
    processor.save_to_file(processed_df, output_file)
    
    logger.info(f"CSV processing completed: {output_file}")

def batch_insert_embeddings_from_file(file_path):
    """
    Process and insert data with embeddings into the database.

    Args:
        file_path (str): Path to the CSV file with embeddings.
    """
    df = pd.read_csv(file_path)
    db = DatabaseConnection()
    db.connect()
    
    try:
        logger.info(f"Processing batch insert from file: {file_path}")
        
        data_list = [
            (
                row['document'],
                int(row['driver_id']),
                int(row['vehicle_id']),
                int(row['policy_id']),
                row['underwriting_decision'],
                row['risk_class'],
                row['reason_for_decline'],
                row['content'],
                int(row['tokens']),
                np.array(ast.literal_eval(row['embeddings']))  # Ensure proper conversion of embeddings
            )
            for _, row in df.iterrows()
        ]
        
        db.batch_insert_embeddings(data_list)
        logger.info("Batch insert completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during batch insert: {str(e)}", exc_info=True)
    finally:
        db.close()

def main():
    """
    Main function to process motor insurance data and generate embeddings.
    """
    filename = 'motor_insurance_hk_data_non_pii_103624oct2024.csv'
    logger.info(f"Starting process for motor insurance data: {filename}")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(root_dir, 'data', filename)
    output_file = os.path.join(root_dir, 'data', 'motor_insurance_hk_data_non_pii_103624oct2024_embeddings.csv')
    
    process_csv_to_embeddings(input_file, output_file)
    
    batch_insert_embeddings_from_file(output_file)

if __name__ == "__main__":
    main()
