import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

def create_dataframe_no_dup(df):

    # remove \n to avoid newlines in no dup file
    df['utterance'] =  df['utterance'].str.replace('\n', '')

    # drop duplicates from df utterances
    df_no_dup = df.drop_duplicates(subset=['utterance'], keep='first')

    # write df without dup. to new dat file
    df_no_dup.to_csv("data/dialog_acts_no_dup.dat", sep=' ', header=False, index=False, quoting=3, escapechar=' ')

    return df_no_dup

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

def train_DT_LR(X_train, y_train):
    vectorizer = CountVectorizer()
    X_bow_train = vectorizer.fit_transform(X_train)
    # X_bow_test = vectorizer.transform(X_test)
    
    # Train Decision-Tree model
    DT_model = DecisionTreeClassifier()
    DT_model.fit(X_bow_train, y_train)

    # Train Logistic Regression model
    LR_model = LogisticRegression(max_iter=400)
    LR_model.fit(X_bow_train, y_train)

    return DT_model, LR_model, vectorizer

def predict_DT_LR(DT_model, LR_model, vectorizer, X_test):

    X_bow_test = vectorizer.transform(X_test)
    Y_pred_DT = DT_model.predict(X_bow_test)
    Y_pred_LR = LR_model.predict(X_bow_test)

    return Y_pred_DT, Y_pred_LR

def save_confusion_matrix(y_test, y_pred, model_name):
    """
    Creates and saves a confusion matrix for a given test set labels and
    predicted labels of a specific model.

    Args:
        y_test: A list of the actual labels of the test set.
        y_pred: A list of the predicted labels for the test set.
        model_name: A string with the name of the model for evaluation.
    """
    # labels = ["ack", "affirm", "bye", "confirm", "deny", "hello", "inform", "negate", "null", "repeat", "reqalts", "require", "request", "restart", "thankyou"]
    
    # Get the unique labels from the test set and sort to keep labels in the correct order
    labels = sorted(list(set(y_test)))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot()
    disp.plot(xticks_rotation=45)  # Rotate x-axis labels by 45 degrees

    # Add title with the model name
    plt.title(f"Confusion Matrix - {model_name}")

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

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
    print(classification_report(y_test, y_pred, zero_division=0))

    save_confusion_matrix(y_test, y_pred, model_name)

    # Calculate and print the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def print_metrics_for_each_model(df, X1_test, y1_test, y2_test, y1_pred_DT, y1_pred_LR, y2_pred_DT, y2_pred_LR):
    """
    Prints evaluation metrics for both the majority-label model and the rule-based model.

    Args:
        df: The DataFrame containing the training data.
        X_test: A list of test utterances.
        y_test: A list of the actual labels for the test utterances.
    """
    # Make predictions and print metrics for the majority label model
    # print("\nMajority-Label Model - Evaluation Metrics:")
    # y_pred_ml = majority_label_model_predict(df, X1_test)
    # print_metrics(y1_test, y_pred_ml, "Majority-Label Model")

    # # Make predictions and print metrics for the rule based model
    # print("\nRule-Based Model - Evaluation Metrics:")
    # y_pred_rb = rule_based_model_predict(X1_test)
    # print_metrics(y1_test, y_pred_rb, "Rule-Based Model")

    print("\nDecision-Tree model - Evaluation Metrics:")
    print("\nOriginal data:")
    print_metrics(y1_test, y1_pred_DT, "DT-model original data")
    print("\nData without duplications:")
    print_metrics(y2_test, y2_pred_DT, "DT-model no-dup data")

    print("\nLogistic-Regression model - Evaluation Metrics:")
    print("\nOriginal data:")
    print_metrics(y1_test, y1_pred_LR, "LR-model original data")
    print("\nData without duplications:")
    print_metrics(y2_test, y2_pred_LR, "LR-model no-dup data")


def print_menu():
    """Prints the available menu options."""
    print("\nOptions:")
    print("1. Print evaluation metrics")
    print("2. Give an utterance for prediction")
    print("3. Exit")

def main():
    # Load the data into a DataFrame
    df = create_dataframe(DATA_FILE)

    # Create dataframe without duplicates
    df_no_dup = create_dataframe_no_dup(df)

    # Split the data into features (utterances) and labels
    X_1 = df['utterance']
    y_1 = df['label']
    X_2 = df_no_dup['utterance']
    y_2 = df_no_dup['label']

    # Split the data into training and testing sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X_1, y_1, test_size=0.15, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_2, y_2, test_size=0.15, random_state=42)

    # Make predictions on original data for DT and LR
    DT_model, LR_model, vectorizer = train_DT_LR(X1_train, y1_train)
    y1_pred_DT, y1_pred_LR = predict_DT_LR(DT_model, LR_model, vectorizer, X1_test)

    # Make predictions on no dup test set for DT and LR
    DT_model_no_dup, LR_model_no_dup, vectorizer_no_dup = train_DT_LR(X2_train, y2_train)
    y2_pred_DT, y2_pred_LR = predict_DT_LR(DT_model_no_dup, LR_model_no_dup, vectorizer_no_dup, X2_test)
   

    while True:
        # Display the menu
        print_menu()

        # Get input from the user
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '1':
            # Print evaluation metrics
            print_metrics_for_each_model(df, X1_test, y1_test, y2_test, y1_pred_DT, y1_pred_LR, y2_pred_DT, y2_pred_LR)


        elif choice == '2':
            while True:
                # Get input from the user
                user_utterance = input("\nEnter an utterance (or type '0' to go to menu): ").lower()

                # Check if the user wants to exit
                if user_utterance == '0':
                    print("Exiting...")
                    break

                user_utterance_vec = vectorizer.transform([user_utterance])

                # Predict labels using DT and LR models
                y_pred_DT = DT_model.predict(user_utterance_vec)[0]
                y_pred_LR = LR_model.predict(user_utterance_vec)[0]

                # Print predicted labels
                print(f"Predicted Label (Decision Tree): {y_pred_DT}")
                print(f"Predicted Label (Logistic Regression): {y_pred_LR}")
                

        elif choice == '3':
            # Exit the program
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please select a valid option.")

# Run the main function
main()
