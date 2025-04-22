from preprocess import get_data
from sklearn.metrics import mean_absolute_error
from models.lstm import LSTM
from models.cnn import CNN
from models.transformer import Transformer
from models.rnn import RNN
from models.mlp import MLP
import sys

# Dictionary mapping command line arguemnts to models
model_map = {
    "mlp": MLP,
    "cnn": CNN,
    "rnn": RNN,
    "lstm": LSTM,
    "transformer": Transformer
}

def run_model(model_name, X_train, y_train, X_test, y_test, target_scaler, start_prices, show_output=True):
    """
    Runs a given model and prints statistics to the terminal

    Args:
        X_train, y_train: Training inputs and targets.
        X_test, y_test: Test inputs and targets.
        target_scaler: Scaler used to normalize and inverse-transform predictions and targets.
        start_prices: Unscaled starting prices to print with statistics

    Returns:
        float: Mean absolute error
    """
    print(f"\nModel: {model_name.upper()}")

    # Get model class and instantiate
    model_class = model_map[model_name]
    model = model_class()

    # Build model with correct input shape
    model.build(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train the model
    model.train(X_train, y_train)

    # Make predictions on test set
    predictions = model.predict(X_test)

    # Unscale prices for comparisons
    predictions_final = target_scaler.inverse_transform(predictions)
    y_test_final = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Compute MAE in original scale
    mae = mean_absolute_error(y_test_final, predictions_final)

    # Print sample predictions if enabled using custom formatting
    if show_output:
        print("Start Price | True Future Price | Predicted Future Price")
        for i in range(5):
            print(f"{start_prices[i]:>11.2f} | {y_test_final[i][0]:>18.2f} | {predictions_final[i][0]:>23.2f}")
        print(f"\nMean Absolute Error (MAE): {mae:.3f}")

    return mae

def main():
    # Process command line arguments
    if len(sys.argv) < 2 or sys.argv[1].lower() not in model_map and sys.argv[1].lower() != "all":
        print("COMMAND LINE INSTRUCTIONS: python main.py <mlp/cnn/rnn/lstm/transformer/all>")
        exit()

    model_name = sys.argv[1].lower()

    # Load training and testing data
    X_train, y_train, _, _ = get_data(testing=False)
    X_test, y_test, target_scaler, start_prices = get_data(testing=True)

    # Run a single model or all models
    if model_name == "all":
        results = {}
        for name in model_map:
            mae = run_model(name, X_train, y_train, X_test, y_test, target_scaler, start_prices)
            results[name] = mae

        # Print all MAEs sorted from best to worst
        print("\n--- Model Mean Absolute Error Comparison ---")
        for name, mae in sorted(results.items(), key=lambda x: x[1]):
            print(f"{name.upper()}: MAE = {mae:.3f}")

    else:
        # Run only the specified model
        run_model(model_name, X_train, y_train, X_test, y_test, target_scaler, start_prices)

if __name__ == '__main__':
    main()