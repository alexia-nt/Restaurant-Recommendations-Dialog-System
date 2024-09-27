import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_FILE = 'data/dialog_acts.dat'

RULES = {
    'inform': ['restaurant', 'food', 'place', 'serves', 'i need', 'looking for', 'preference', 'information', 'moderate price', 'any part of town', 'south part of town'],        'request': ['what is', 'where is', 'can you tell', 'post code', 'address', 'location', 'phone number', 'could i get', 'request', 'find', 'search', 'detail'],
    'request': ['what is', 'where is', 'can you tell', 'post code', 'address', 'location', 'phone number', 'could i get', 'request', 'find', 'search', 'detail'],
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
    """
    Removes duplicate utterances from a DataFrame and writes the result to a new file.

    Args:
        df: A Pandas DataFrame with 'label' and 'utterance' columns.

    Returns:
        df_no_dup: A new DataFrame with duplicate utterances removed.
    """
    # remove \n to avoid newlines in no dup file
    df['utterance'] =  df['utterance'].str.replace('\n', '')

    # drop duplicates from df utterances
    df_no_dup = df.drop_duplicates(subset=['utterance'], keep='first')

    # write df without dup. to new dat file
    df_no_dup.to_csv("data/dialog_acts_no_dup.dat", sep=' ', header=False, index=False, quoting=3, escapechar=' ')

    return df_no_dup

def generate_barplots(df):
    """
    Generates a bar plot visualizing the distribution of utterance lengths (number of words) in the given DataFrame.

    Args:
        df: A Pandas DataFrame with 'label' and 'utterance' columns.
    """
    count = df['utterance'].str.split().str.len()
    count.index = count.index.astype(str) + ' words:'
    count.value_counts().plot(logy=True, kind='bar', color='green', title='Distribution of # words per utterance of original dataset')
    
    plt.xlabel('# words in utterance', labelpad=12)
    plt.ylabel('Count')

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Create the 'figures' directory if it doesn't exist
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the plot in the 'figures' directory
    plot_filename = os.path.join(figures_dir, "Number of words per utterance.png")
    plt.savefig(plot_filename)

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
    Predicts the label of an utterance based on keyword matching rules using a dictionary.

    Args:
        utterance: The input utterance.

    Returns:
        predicted_label: The predicted label.
    """
    for label, keywords in RULES.items():
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
    """
    Trains Decision Tree and Logistic Regression models on the given training data.

    Args:
        X_train: A list or array containing the training utterances (text data).
        y_train: A list or array containing the corresponding labels for the training utterances.

    Returns:
        DT_model: The trained Decision Tree model.
        LR_model: The trained Logistic Regression model.
        vectorizer: The fitted CountVectorizer object.
    """
    vectorizer = CountVectorizer()
    X_bow_train = vectorizer.fit_transform(X_train)
    
    # Train Decision-Tree model
    DT_model = DecisionTreeClassifier()
    DT_model.fit(X_bow_train, y_train)

    # Train Logistic Regression model
    LR_model = LogisticRegression(max_iter=400)
    LR_model.fit(X_bow_train, y_train)

    return DT_model, LR_model, vectorizer

def predict_DT_LR(DT_model, LR_model, vectorizer, X_test):
    """
    Predicts labels for the test data using the trained Decision Tree and Logistic Regression models.

    Args:
        DT_model: The trained Decision Tree model.
        LR_model: The trained Logistic Regression model.
        vectorizer: The fitted CountVectorizer object.
        X_test: A list or array containing the new utterances for prediction.

    Returns:
        Y_pred_DT: A list or array containing the predicted labels for the test data using the Decision Tree model.
        Y_pred_LR: A list or array containing the predicted labels for the test data using the Logistic Regression model.
    """
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

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed
    disp.plot(ax=ax, xticks_rotation=45)  # Rotate x-axis labels by 45 degrees

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

def print_metrics_for_baseline_models(df, X_test, y_test):
    """
    Prints evaluation metrics for the baseline models (majority-label model and rule-based model).

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

def print_metrics_for_DT_LR(y_dup_test, y_dup_pred_DT, y_dup_pred_LR, y_no_dup_test, y_no_dup_pred_DT, y_no_dup_pred_LR):
    """
    Prints evaluation metrics for the ML models (Decision Tree and Logistic Regression)
    on both the original and duplicate-free data.

    Args:
        y_dup_test: The true labels for the test set with duplicate utterances.
        y_dup_pred_DT: The predicted labels for the test set with duplicate utterances using the Decision Tree model.
        y_dup_pred_LR: The predicted labels for the test set with duplicate utterances using the Logistic Regression model.

        y_no_dup_test: The true labels for the test set without duplicate utterances.
        y_no_dup_pred_DT: The predicted labels for the test set without duplicate utterances using the Decision Tree model.
        y_no_dup_pred_LR: The predicted labels for the test set without duplicate utterances using the Logistic Regression model.
    """
    print("\nDecision-Tree model - Evaluation Metrics:")
    print("\nOriginal data:")
    print_metrics(y_dup_test, y_dup_pred_DT, "DT-model original data")
    print("\nData without duplications:")
    print_metrics(y_no_dup_test, y_no_dup_pred_DT, "DT-model no-dup data")

    print("\nLogistic-Regression model - Evaluation Metrics:")
    print("\nOriginal data:")
    print_metrics(y_dup_test, y_dup_pred_LR, "LR-model original data")
    print("\nData without duplications:")
    print_metrics(y_no_dup_test, y_no_dup_pred_LR, "LR-model no-dup data")

def print_menu():
    """Prints the available menu options."""
    print("\nOptions:")
    print("1. Print evaluation metrics")
    print("2. Give an utterance for prediction")
    print("3. Exit")

if __name__ == "__main__":
    # Load the data into a DataFrame
    df = create_dataframe(DATA_FILE)

    # generate_barplots(df)

    # Create dataframe without duplicates
    df_no_dup = create_dataframe_no_dup(df)

    # Split the data into features (utterances) and labels
    X_dup = df['utterance']
    y_dup = df['label']
    X_no_dup = df_no_dup['utterance']
    y_no_dup = df_no_dup['label']

    # Split the data into training and testing sets
    X_dup_train, X_dup_test, y_dup_train, y_dup_test = train_test_split(X_dup, y_dup, test_size=0.15, random_state=42)
    X_no_dup_train, X_no_dup_test, y_no_dup_train, y_no_dup_test = train_test_split(X_no_dup, y_no_dup, test_size=0.15, random_state=42)

    # Train and make predictions on original data for DT and LR
    DT_model, LR_model, vectorizer = train_DT_LR(X_dup_train, y_dup_train)
    y_dup_pred_DT, y_dup_pred_LR = predict_DT_LR(DT_model, LR_model, vectorizer, X_dup_test)

    # Train and make predictions on data without duplicated utterances for DT and LR
    DT_model_no_dup, LR_model_no_dup, vectorizer_no_dup = train_DT_LR(X_no_dup_train, y_no_dup_train)
    y_no_dup_pred_DT, y_no_dup_pred_LR = predict_DT_LR(DT_model_no_dup, LR_model_no_dup, vectorizer_no_dup, X_no_dup_test)

    while True:
        # Display the menu
        print_menu()

        # Get input from the user
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '1': 
            # Print evaluation metrics for baseline models using original data (with duplicated utterances)  
            print_metrics_for_baseline_models(df, X_dup_test, y_dup_test)

            # Print evaluation metrics for baseline models using both original data and data without duplicated utterances
            print_metrics_for_DT_LR(y_dup_test, y_dup_pred_DT, y_dup_pred_LR, y_no_dup_test, y_no_dup_pred_DT, y_no_dup_pred_LR)

        elif choice == '2':
            while True:
                # Get input from the user
                user_utterance = input("\nEnter an utterance (or type '0' to go to menu): ").lower()

                # Check if the user wants to exit
                if user_utterance == '0':
                    print("Exiting...")
                    break

                # Transform the user utterance with the vectorizer
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


