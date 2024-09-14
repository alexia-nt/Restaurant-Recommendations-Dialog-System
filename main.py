import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

DATA_FILE = 'data/dialog_acts.dat'

def create_dataframe(data_file):
    """Creates a Pandas DataFrame from the given data file.

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

def return_majority_label(data_file):
    """Returns the most frequent label in the data file.

    Args:
        data_file: The path of the data file.

    Returns:
        majority_label: The most frequent label in the data file.
    """

    label_list = []
    with open(data_file, 'r') as file:
        for line in file:
            label_list.append(line.split()[0])  # Get the first word of each line as the label

    # make a dictionary with unique labels as keys and their frequency in the label list as values
    label_counter = {}
    for label in label_list:
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1

    # get the most frequent label
    majority_label = max(label_counter, key=label_counter.get)

    # print("Percentage of most most frequent label:", label_counter[majority_label]/len(label_list))

    # return the most frequent label
    return majority_label

def return_majority_label_from_df(df):
    """Returns the most frequent label in the DataFrame.

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

def baseline_model_majority_label_predict(df, X_test):
    """Predicts the majority label for all utterances in the test set.

    Args:
        df: A Pandas DataFrame containing the training data.
        X_test: A list or Series of test utterances.

    Returns:
        y_pred: A list of predicted labels (the majority label).
    """
    y_pred = []
    majority_label = return_majority_label_from_df(df)
    for _ in X_test: # for every utterance in X_test
        y_pred.append(majority_label)
    return y_pred

def rule_based_get_label(utterance):
    """Predicts the label based on keyword matching rules using a dictionary.

    Args:
        utterance: The input utterance.

    Returns:
        predicted_label: The predicted label.
    """

    # The labels in the dictionary are ordered based on the frequency of the labels in the dataset.
    # This will prioritize checking the more frequent labels first, which could lead to better performance as well.

    rules = {
        'inform': ['restaurant', 'food', 'place', 'serves', 'i need', 'looking for', 'preference', 'information', 'find'],
        'request': ['what is', 'where is', 'can you tell', 'post code', 'address', 'location'],
        'thankyou': ['thank', 'thanks', 'appreciate it', 'grateful'],
        'reqalts': ['how about', 'alternative', 'other options', 'alternatives'],
        'null': ['cough', 'noise', 'laugh', 'uh'],
        'affirm': ['yes', 'right', 'correct', 'sure', 'absolutely', 'definitely'],
        'negate': ['no', 'not', 'don\'t', 'nope'],
        'bye': ['goodbye', 'bye', 'see you', 'farewell', 'later'],
        'confirm': ['confirm', 'is it', 'check if', 'correct?', 'right?'],
        'hello': ['hi', 'hello', 'hey', 'greetings'],
        'repeat': ['repeat', 'say again', 'can you repeat'],
        'ack': ['okay', 'um', 'uh-huh'],
        'deny': ['dont want', 'don\'t want', 'reject', 'no', 'not that', 'not this'],
        'restart': ['restart', 'start over', 'begin again', 'reset'],
        'reqmore': ['more', 'additional', 'other', 'else'],
    }

    # rules = {
    #     'inform': ['restaurant', 'food', 'place'],
    #     'request': ['what is', 'where is', 'can you tell'],
    #     'thankyou': ['thank', 'thanks'],
    #     'reqalts': ['how about', 'alternative'],
    #     'null': ['cough', 'noise'],
    #     'affirm': ['yes', 'right', 'correct', 'sure'],
    #     'negate': ['no', 'not'],
    #     'bye': ['goodbye', 'bye', 'see you'],
    #     'confirm': ['confirm', 'is it', 'check if'],
    #     'hello': ['hi', 'hello', 'hey'],
    #     'repeat': ['repeat', 'say again'],
    #     'ack': ['okay', 'um', 'uh-huh'],
    #     'deny': ['dont want', 'reject', "don't want"],
    #     'restart': ['restart', 'start over'],
    #     'reqmore': ['more']
    # }

    for label, keywords in rules.items():
        if any(keyword in utterance for keyword in keywords):
            return label

    # Default label if no keyword matches
    return 'inform'

def rule_based_model_predict(X_test):
    """Predicts labels for the test set using a rule-based model.

    Args:
        X_test: A list or Pandas Series of test utterances.

    Returns:
        y_pred: A list of predicted labels corresponding to the test utterances.
    """
    y_pred = []
    for utterance in X_test:
        y_pred.append(rule_based_get_label(utterance))
    return y_pred

def print_metrics(y_test, y_pred):
    """Prints classification metrics for a given test set labels and predicted labels.

    Args:
        y_test: The actual labels of the test set.
        y_pred: The predicted labels for the test set.
    """

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate and print the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

def main():
    # Load the data into a DataFrame
    df = create_dataframe(DATA_FILE)

    # Split the data into features (utterances) and labels
    X = df['utterance']
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Make predictions on the test set
    # y_pred = baseline_model_majority_label_predict(df, X_test)
    y_pred = rule_based_model_predict(X_test)

    print_metrics(y_test, y_pred)

# Run the main function
main()
