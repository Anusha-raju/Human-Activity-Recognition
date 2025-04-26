import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from component.train import train_and_evaluate
from component.predict import predict_video_class
import argparse

def main():
    parser = argparse.ArgumentParser(description="Choose to train or predict")
    parser.add_argument('--opt', required=True, choices=['train', 'predict'],
                        help="Option to run: 'train' or 'predict'")
    parser.add_argument('--path', type=str, help="Path to the model for prediction")
    args = parser.parse_args()

    if args.opt == 'train':
        train_and_evaluate()
    elif args.opt == 'predict':
        if args.path:
            predicted_class = predict_video_class(args.path)
            print(f"\nThe Predicted Class: {predicted_class}\n")
        else:
            print("Error: --path is required when using --opt predict")
            exit(1)

if __name__ == "__main__":
    main()
