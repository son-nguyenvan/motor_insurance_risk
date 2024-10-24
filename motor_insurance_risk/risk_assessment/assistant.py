from motor_insurance_risk.risk_assessment.processor import RiskAssessmentProcessor


def main():
    # Create an instance of RiskAssessmentProcessor
    risk_processor = RiskAssessmentProcessor()

    # Example user input (This can be made dynamic by taking input from the user, command line, or other sources)
    user_input = "Motor insurance risk assessment for a 45-year-old male driver with 10 years of driving experience, 3 accidents, and a McLaren Speedtail. Additionally, provide a confidence score between 1 and 10 for your assessment."

    # Define the top_k to retrieve the top 5 similar documents
    top_k = 5

    # Process the risk assessment for the given input
    print("Processing risk assessment...")
    response = risk_processor.process_risk_assessment(user_input, top_k)

    # Print out the response from the processor
    print("Risk Assessment Response:")
    print(response)

if __name__ == "__main__":
    main()
