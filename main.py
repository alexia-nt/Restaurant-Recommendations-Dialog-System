import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_FILE = 'data/dialog_acts.dat'

def create_dataframe(data_file):
    """
    Creates a Pandas DataFrame from the given data file.

    Args:
        data_file: The path of the data file.

    Returns:
        df: A DataFrame with two columns: 'label' and 'utterance'.
    """

    labels = []
    utterances = []

    with open(data_file, 'r') as file:
        for line in file:
            label, utterance = line.split(maxsplit=1)
            labels.append(label.lower())
            utterances.append(utterance.lower())

    data_dict = {'label': labels, 'utterance': utterances}
    df = pd.DataFrame(data_dict)

    return df

def return_majority_label(df):
    """
    Returns the most frequent label in the DataFrame.

    Args:
        df: A Pandas DataFrame with 'label' and 'utterance' columns.

    Returns:
        majority_label: The most frequent label in the DataFrame.
    """

    # Count occurrences of each label
    label_counts = df['label'].value_counts()

    # Get the most frequent label
    majority_label = label_counts.index[0]

    # print("Percentage of most most frequent label:", label_counts[0]/len(df))

    return majority_label

def majority_label_model_predict(df, X_test):
    """
    Predicts the majority label for all utterances in the test set.

    Args:
        df: A Pandas DataFrame containing the training data.
        X_test: A list or Series of test utterances.

    Returns:
        y_pred: A list of predicted labels (the majority label).
    """
    y_pred = []
    majority_label = return_majority_label(df)
    for _ in X_test: # for every utterance in X_test
        y_pred.append(majority_label)
    return y_pred

def rule_based_model_get_label(utterance):
    """
    Predicts the label based on keyword matching rules using a dictionary.

    Args:
        utterance: The input utterance.

    Returns:
        predicted_label: The predicted label.
    """

    # The labels in the dictionary are ordered based on the frequency of the labels in the dataset.
    # This will prioritize checking the more frequent labels first, which could lead to better performance.

    rules = {
        'inform': ['restaurant', 'food', 'place', 'serves', 'i need', 'looking for', 'preference', 'information', 'moderate price', 'any part of town', 'south part of town'],
        'request': ['what is', 'where is', 'can you tell', 'post code', 'address', 'location', 'phone number', 'could i get', 'request', 'find', 'search'],
        'thankyou': ['thank', 'thanks', 'appreciate', 'grateful', 'thank you', 'many thanks', 'thankful'],
        'reqalts': ['how about', 'alternative', 'other', 'other options', 'alternatives', 'another', 'another suggestion', 'different', 'else', 'instead'],
        'null': ['cough', 'noise', 'laugh', 'uh', 'um', 'background', 'incoherent'],
        'affirm': ['yes', 'yeah', 'right', 'correct', 'sure', 'absolutely', 'definitely', 'of course', 'affirm', 'indeed'],
        'negate': ['no', 'not', 'don\'t', 'nope', 'none', 'wrong', 'never', 'incorrect'],
        'bye': ['goodbye', 'bye', 'see you', 'farewell', 'later', 'take care', 'talk to you soon'],
        'confirm': ['confirm', 'is it', 'check if', 'right?', 'verify', 'am i correct', 'confirming', 'confirmation', 'can you confirm'],
        'hello': ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy'],
        'repeat': ['repeat', 'again', 'say again', 'can you repeat', 'could you repeat', 'pardon'],
        'ack': ['okay', 'um', 'uh-huh', 'got it', 'alright', 'fine', 'ok', 'acknowledge'],
        'deny': ['dont want', 'don\'t want', 'reject', 'not that', 'no thanks', 'not interested', 'decline', 'denying'],
        'restart': ['restart', 'start over', 'begin again', 'reset', 'new', 'begin anew'],
        'reqmore': ['more', 'additional', 'extra', 'any more', 'other', 'further']
    }

    for label, keywords in rules.items():
        if any(keyword in utterance for keyword in keywords):
            return label

    # Default label if no keyword matches
    return 'inform'

def rule_based_model_predict(X_test):
    """
    Predicts labels for the test set using a rule-based model.

    Args:
        X_test: A list of test utterances.

    Returns:
        y_pred: A list of predicted labels corresponding to the test utterances.
    """
    y_pred = []
    for utterance in X_test:
        y_pred.append(rule_based_model_get_label(utterance))
    return y_pred

def save_confusion_matrix(y_test, y_pred, model_name):
    """
    Creates and saves a confusion matrix for a given test set labels and
    predicted labels of a specific model.

    Args:
        y_test: A list of the actual labels of the test set.
        y_pred: A list of the predicted labels for the test set.
        model_name: A string with the name of the model for evaluation.
    """
    classes = y_test.unique()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()

    # Add title with the model name
    plt.title(f"Confusion Matrix - {model_name}")

    # Create the 'figures' directory if it doesn't exist
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the plot in the 'figures' directory
    plot_filename = os.path.join(figures_dir, f"{model_name} Confusion Matrix.png")
    plt.savefig(plot_filename)


def print_metrics(y_test, y_pred, model_name):
    """
    Prints classification metrics for a given test set labels and predicted labels
    of a specific model.

    Args:
        y_test: A list of the actual labels of the test set.
        y_pred: A list of the predicted labels for the test set.
        model_name: A string with the name of the model for evaluation.
    """

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    save_confusion_matrix(y_test, y_pred, model_name)

    # Calculate and print the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def print_metrics_for_each_model(df, X_test, y_test):
    """Prints evaluation metrics for both the majority-label model and the rule-based model.

    Args:
        df: The DataFrame containing the training data.
        X_test: A list of test utterances.
        y_test: A list of the actual labels for the test utterances.
    """
    # Make predictions and print metrics for the majority label model
    print("\nMajority-Label Model - Evaluation Metrics:")
    y_pred = majority_label_model_predict(df, X_test)
    print_metrics(y_test, y_pred, "Majority-Label Model")

    # Make predictions and print metrics for the rule based model
    print("\nRule-Based Model - Evaluation Metrics:")
    y_pred = rule_based_model_predict(X_test)
    print_metrics(y_test, y_pred, "Rule-Based Model")

def print_menu():
    """Prints the available menu options."""
    print("\nOptions:")
    print("1. Print evaluation metrics")
    print("2. Give an utterance for prediction")
    print("3. Exit")

def main():
    # Load the data into a DataFrame
    df = create_dataframe(DATA_FILE)

    # Split the data into features (utterances) and labels
    X = df['utterance']
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Make predictions on the test set
    # y_pred = majority_label_model_predict(df, X_test)
    y_pred = rule_based_model_predict(X_test)

    # # Print evaluation metrics
    # print_metrics(y_test, y_pred)

    while True:
        # Display the menu
        print_menu()

        # Get input from the user
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '1':
            # Print evaluation metrics
            print_metrics_for_each_model(df, X_test, y_test)

        elif choice == '2':
            while True:
                # Get input from the user
                user_utterance = input("\nEnter an utterance (or type '0' to go to menu): ").lower()

                # Check if the user wants to exit
                if user_utterance == '0':
                    print("Exiting...")
                    break
                predicted_label = rule_based_model_get_label(user_utterance)
                print(f"Predicted label: {predicted_label}")

        elif choice == '3':
            # Exit the program
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please select a valid option.")

# Run the main function
main()
