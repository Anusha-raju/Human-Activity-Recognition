import sys
import os

# Add the parent directory to the system path for importing modules from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing functions for training, prediction, and video analysis from respective components
from component.train import train_and_evaluate
from component.predict import predict_video_class
from component.post_training import video_analysis
import argparse

def main():
    """
    Main function to handle the command-line interface (CLI) and execute the chosen operation.
    
    The user can choose between training a model, predicting the class of a video, 
    or performing post-training video analysis.
    
    CLI Arguments:
        --opt (str)   : Required argument to specify the operation ('train', 'predict', 'analysis')
        --path (str)  : Path to the model or video file, required for 'predict' and 'analysis'
        --method (str): Method of analysis (e.g., 'heatmap' or 'activity_labels'), required for 'analysis'
    """
    
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Choose to train or predict or analysis")
    
    # Adding the optional arguments to the parser
    parser.add_argument('--opt', required=True, choices=['train', 'predict', 'analysis'],
                        help="Option to run: 'train' or 'predict' or 'analysis'")
    parser.add_argument('--path', type=str, help="Path to the model for prediction")
    parser.add_argument('--method', type=str, help="Method of analysis (e.g., 'heatmap' or 'activity_labels')")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Perform the selected operation
    if args.opt == 'train':
        # Call the function to train and evaluate the model
        train_and_evaluate()

    elif args.opt == 'predict':
        # Prediction mode: Ensure that --path is provided, and make predictions
        if args.path:
            predicted_class = predict_video_class(args.path)
            print(f"\nThe Predicted Class: {predicted_class}\n")
        else:
            print("Error: --path is required when using --opt predict")
            exit(1)

    elif args.opt == 'analysis':
        # Analysis mode: Ensure that --path and --method are provided, then perform the analysis
        if args.path:
            if args.method == 'heatmap':
                # Perform video analysis with heatmap option
                video_analysis(args.path, option="heatmap")
            elif args.method == 'activity_labels':
                # Perform video analysis with activity labels option
                video_analysis(args.path, option="activity_labels")
            else:
                # Handle invalid method
                print("Error: invalid method!!")
                exit(1)
        else:
            print("Error: --path is required when using --opt analysis")
            exit(1)

# Check if this script is executed directly, and run the main function
if __name__ == "__main__":
    main()
