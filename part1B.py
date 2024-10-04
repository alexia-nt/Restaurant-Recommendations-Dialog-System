import pandas as pd
import Levenshtein

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

LR_MODEL_PREFERENCE = 1
DT_MODEL_PREFERENCE = 2
RULE_BASED_MODEL_PREFERENCE = 3

class RestaurantRecommendationSystem:
    """
    Implementation of a recommender system
    that provides information about restaurants
    based on the user's preferences by 
    utilizing Rule-Based and Machine Learning
    (Decision Tree and Logistic Regression)
    classification.

    Attributes
    ----------
    - model_preference (int): reflects model choice of user:
        1 for Logistic Regression model
        2 for Decision Tree model
        3 for Rule-Based model

    Constants
    ---------
    DATA_FILE: path to CSV file restaurant data
    DT_FILE: path to trained model file Decision Tree
    LR_FILE: path to trained model file Logistic Regression
    VECTORIZER_FILE: path to trained BoW vectorizer model

    *_STATE: constants representing states in conversation flow (see state diagram)
    DONT_CARE_KEYWORDS: keywords representing user's absence of preference
    RULE_BASED_MODEL_RULES: rules representing Rule-Based model from assign. 1A
    """

    DATA_FILE = "data/restaurant_info.csv"
    DIALOG_ACTS_FILE = "data/dialog_acts.dat"
    DT_FILE = "models/dt_model.pkl"
    LR_FILE = "models/lr_model.pkl"
    VECTORIZER_FILE = "models/vectorizer.pkl"

    WELCOME_STATE = 1
    ASK_INITIAL_PREFERENCES_STATE = 2
    ASK_FOOD_STATE = 3
    ASK_PRICE_STATE = 4
    ASK_AREA_STATE = 5
    SEARCH_STATE = 6
    NO_MORE_RECOMMENDATIONS_STATE = 7
    GIVE_DETAILS_STATE = 8
    SEARCH_MORE_STATE = 9
    END_STATE = 10

    DONT_CARE_KEYWORDS = [
        "do not care", "don't care", "dont care",
        "do not mind", "don't mind", "dont mind",
        "does not matter", "doesn't matter", "doesnt matter",
        "any", "anywhere", "whatever"
    ]

    RULE_BASED_MODEL_RULES = {
        'request': ['what is', 'where is', 'can you tell', 'post code', 'address', 'location', 'phone', 'number', 'could i get', 'request', 'find', 'search', 'detail'],
        'reqmore': ['more', 'additional', 'extra', 'any more', 'other', 'further', 'another'],
        'reqalts': ['how about', 'what about', 'change', 'alternative', 'other options', 'different', 'else', 'instead', 'maybe'],

        'bye': ['goodbye', 'bye', 'see you', 'farewell', 'later', 'take care', 'talk to you soon'],
        'negate': ['no', 'not', 'don\'t', 'nope', 'none', 'wrong', 'never', 'incorrect'],
        'deny': ['dont want', 'don\'t want', 'reject', 'not that', 'no thanks', 'not interested', 'decline', 'denying'],

        'ack': ['okay', 'um', 'uh-huh', 'got it', 'alright', 'fine', 'ok', 'acknowledge'],
        'confirm': ['confirm', 'is it', 'check if', 'right?', 'verify', 'am i correct', 'confirming', 'confirmation', 'can you confirm'],
        'affirm': ['yes', 'yeah', 'right', 'correct', 'sure', 'absolutely', 'definitely', 'of course', 'affirm', 'indeed'],
        'thankyou': ['thank', 'thanks', 'appreciate', 'grateful', 'thank you', 'many thanks', 'thankful'],

        'hello': ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy'],
        'inform': ['restaurant', 'food', 'place', 'serves', 'i need', 'looking for', 'preference', 'information', 'moderate price', 'any part of town', 'south part of town'],

        'repeat': ['repeat', 'again', 'say again', 'can you repeat', 'could you repeat', 'pardon'],
        'restart': ['restart', 'start over', 'begin again', 'reset', 'new', 'begin anew'],

        'null': ['cough', 'noise', 'laugh', 'uh', 'um', 'background', 'incoherent']
    }

    def __init__(self, model_preference):
        """
        Parameters
        ----------
        - model_preference (int): reflects model choice of user:
            1 for Logistic Regression model
            2 for Decision Tree model
            3 for Rule-Based model
        """
        
        # Read restaurant data
        self.df = pd.read_csv(self.DATA_FILE)

        # Extract keywords for preferences
        self.food_keywords, self.price_keywords, self.area_keywords = self.get_keywords()
        # Add "center" in area keywords, so it is also included in the
        # area_keywords list and not only the "centre" area keyword
        self.area_keywords.append("center")

        # Initialize variables
        self.user_input = ""
        self.food_preference = None
        self.price_preference = None
        self.area_preference = None

        self.possible_restaurants = []
        self.levenshtein_threshold = 1

        self.model_preference = model_preference
        
        if(self.model_preference == LR_MODEL_PREFERENCE):
            self.LR_model, self.vectorizer = self.train_LR()

        elif(self.model_preference == DT_MODEL_PREFERENCE):
            self.DT_model, self.vectorizer = self.train_DT()

    def get_keywords(self):
        """
        Extracts unique keywords for food, price, 
        and area from the restaurant dataset.

        Returns
        -------
        - food_keywords (list): List of unique food keywords
        - price_keywords (list): List of unique price keywords
        - area_keywords (list): List of unique area keywords
        """

        food_keywords = list(self.df["food"].dropna().astype(str).unique())
        price_keywords = list(self.df["pricerange"].dropna().astype(str).unique())
        area_keywords = list(self.df["area"].dropna().astype(str).unique())
        return food_keywords, price_keywords, area_keywords

    def create_dataframe(self, data_file):
        """
        Creates a Pandas DataFrame from the given data file.

        Parameters
        ----------
        - data_file: The path of the data file.

        Returns
        -------
        - df: A DataFrame with two columns: 'label' and 'utterance'.
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

    def train_LR(self):
        """
        Trains a Logistic Regression model on dialog act classification data.

        Returns
        -------
        - LR_model: The trained Logistic Regression model.
        - vectorizer: The CountVectorizer used to transform text data into numerical features.
        """

        df_dialog_acts = self.create_dataframe(self.DIALOG_ACTS_FILE)

        X = df_dialog_acts['utterance']
        y = df_dialog_acts['label']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15, random_state=42)

        vectorizer = CountVectorizer()
        X_bow_train = vectorizer.fit_transform(X_train)

        # Train Logistic Regression model
        LR_model = LogisticRegression(max_iter=400)
        LR_model.fit(X_bow_train, y_train)

        return LR_model, vectorizer
    
    def train_DT(self):
        """
        Trains a Decision Tree model on dialog act classification data.

        Returns
        -------
        - DT_model: The trained Decision Tree model.
        - vectorizer: The CountVectorizer used to transform text data into numerical features.
        """

        df_dialog_acts = self.create_dataframe(self.DIALOG_ACTS_FILE)

        X = df_dialog_acts['utterance']
        y = df_dialog_acts['label']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.15, random_state=42)

        vectorizer = CountVectorizer()
        X_bow_train = vectorizer.fit_transform(X_train)
        
        # Train Decision-Tree model
        DT_model = DecisionTreeClassifier()
        DT_model.fit(X_bow_train, y_train)

        return DT_model, vectorizer
    
    def extract_specific_preferences(self, utterance, keywords):
        """
        Extracts user preferences other than 'any' from 
        an utterance through matching it with provided 
        keywords. 
        If no match is found, Levenshtein distance is
        used to find the best match.

        Parameters
        -----------
        - utterance (str): string of user input 
        - keywords (list): list of keywords that are relevant to the requested preference category (e.g., food types, price ranges) 
        
        Returns
        --------
        - (None, str): matched keyword or None if no match was found
        """

        for keyword in keywords:
            if keyword in utterance:
                if keyword == "center":
                    return "centre"
                return keyword
        
        utterance_words = utterance.lower().split()
        
        for word in utterance_words:
            # If the word from user utterance is "want", do not check levenshtein distance,
            # because it would be matched as "west" area preference for threshold >=2
            if word == "want" or word == "yes":
                continue 
            for keyword in keywords:
                # Check if any word from the utterance matches part of a multi-word keyword
                # e.g. if the user want "asian" return "asian oriental" which is an are keyword
                if word in keyword.split():
                    return keyword
                distance = Levenshtein.distance(word.lower(), keyword.lower())
                if distance <= self.levenshtein_threshold:  # Adjust threshold as needed
                    return keyword
        return None
        
    def extract_initial_preferences(self, utterance, keywords, preference_type):
        """
        Extracts initial user preferences at the start of the session,
        allowing preferences to be stated in a single utterance only.
        
        Parameters
        -----------
        - utterance (str): user's input
        - keywords (list): list of keywords that are relevant to the requested preference category (e.g., food types, price ranges)
        - preference_type (str): The type of preference being requested (e.g., "food", "price", "area").
        
        Returns
        --------
        - (None, str): matched keyword or None if no match was found
        """
        
        if f"any {preference_type}" in utterance:
            return "any"
        
        return self.extract_specific_preferences(utterance, keywords)

    def extract_preferences(self, utterance, keywords):
        """
        Extracts user preferences from an utterance, checking if the user
        has any "don't care" preferences or if a specific keyword matches
        their preferences.

        Parameters
        -----------
        - utterance (str): string of user input 
        - keywords (list): list of keywords that are relevant to the requested preference category (e.g., food types, price ranges)
        
        Returns
        --------
        - (None, str): matched keyword or None if no match was found
        """

        for keyword in self.DONT_CARE_KEYWORDS:
            if keyword in utterance:
                return "any"
            
        return self.extract_specific_preferences(utterance, keywords)

    def get_matching_restaurants(self):
        """
        Filters the restaurant dataset based on user preferences and
        returns list with restaurants that match those preferences.
        In case no matches are found, an empty list is returned. 

        Returns
        -------- 
        - (list) list of dictionaries of matched restaurants
        """

        filtered_df = self.df.copy()

        if self.food_preference != "any":
            filtered_df = filtered_df[filtered_df["food"] == self.food_preference]
        
        if self.price_preference != "any":
            filtered_df = filtered_df[filtered_df["pricerange"] == self.price_preference]
        
        if self.area_preference != "any":
            filtered_df = filtered_df[filtered_df["area"] == self.area_preference]

        if not filtered_df.empty:
            self.possible_restaurants = filtered_df.to_dict(orient='records')
            return self.possible_restaurants
        else:
            return []
        
    def print_details(self):
        """
        Prints the available details for the recommended restaurant.
        If the user has not specified what details they want, system
        asks user until a valid answer is given and then prints details.
        """
        gave_details = False

        # Check if the user asks for the phone number
        if ("phone" in self.user_input) or ("number" in self.user_input):
            gave_details = True
            if(pd.isna(self.possible_restaurants[0]['phone'])):
                print("I don't have an available phone number.")
            else:
                print(f"The phone number is {self.possible_restaurants[0]['phone']}.")
        
        # Check if the user asks for the address
        if "address" in self.user_input:
            gave_details = True
            if(pd.isna(self.possible_restaurants[0]['addr'])):
                print("I don't have an available address.")
            else:
                print(f"The address is {self.possible_restaurants[0]['addr']}.")
        
        # Check if the user asks for the post code
        if ("post" in self.user_input) or ("code" in self.user_input):
            gave_details = True
            if(pd.isna(self.possible_restaurants[0]['postcode'])):
                print("I don't have an available post code.")
            else:
                print(f"The post code is {self.possible_restaurants[0]['postcode']}.")
        
        if gave_details == True:
            return
        else:
            print("What details do you want?")
            self.user_input = input(">>").lower() 
            while not any(word in self.user_input for word in ["phone", "number", "address", "post", "code"]):
                print("Please give a valid detail (phone number, address, post code).")
                self.user_input = input(">>").lower()
            
            # Recursive call to print details if no details were printed
            self.print_details()
        
    def rule_based_model_get_label(self):
        """
        Determines the dialog act label using the Rule-Based model
        and returns dialog act label when a match is found between the 
        user input and the model. When no match is found, the majority
        label 'inform' is returned. 
        
        Returns
        --------
        - (label (str) or 'inform'): label of the matched dialog act, 
        'inform' if no match is found.
        """

        for label, keywords in self.RULE_BASED_MODEL_RULES.items():
            if any(keyword in self.user_input for keyword in keywords):
                return label

        # Default label if no keyword matches
        return 'inform'
        
    def dialog_act_prediction(self):
        """
        Predicts label of dialog act by using the LR, DT or Rule-Based model, 
        based on the model preference of the user.

        Returns
        --------
        - (str) predicted label of the user input by the chosen model by the user
        """

        if(self.model_preference == LR_MODEL_PREFERENCE):
            return self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
        elif(self.model_preference == DT_MODEL_PREFERENCE):
            return self.DT_model.predict(self.vectorizer.transform([self.user_input]))[0]
        else:
            return self.rule_based_model_get_label()

    # ------------------------ State Handlers ------------------------

    def welcome_state_handler(self):
        """
        Handles the first stage of the conversation with the user.
        Ask the user for input, leads the user to the prediction 
        of the dialog act in the user input. 
        """
        print("Hello, welcome to the restaurant recommendations dialogue system! You can ask for restaurants by area, price range, or food type.")
        self.user_input = input(">>").lower()

    def ask_initial_preferences_handler(self):
        """
        Handles the extraction process of the initial preferences
        of the user at the start of the session. 
        Checks if preferences are given by user and if not, returns
        corresponding states to ask for specific preferences.

        Returns
        -------
        - (int): next state based on extracted/missing preferences (ASK_FOOD, ASK_AREA, ASK_PRICE)
        """

        self.food_preference = self.extract_initial_preferences(self.user_input, self.food_keywords, "food")
        self.price_preference = self.extract_initial_preferences(self.user_input, self.price_keywords, "price")
        self.area_preference = self.extract_initial_preferences(self.user_input, self.area_keywords, "area")

        if self.food_preference is None:
            return self.ASK_FOOD_STATE
        
        if self.price_preference is None:
            return self.ASK_PRICE_STATE
        
        if self.area_preference is None:
            return self.ASK_AREA_STATE

        return self.SEARCH_STATE

    def ask_food_handler(self):
        """
        Handles asking the user for their food preference
        by prompting repeatedly until a preference is given.
        """

        if self.food_preference is not None:
            return

        print("What kind of food would you like?")
        while self.food_preference is None:
            self.user_input = input(">>").lower()            

            self.food_preference = self.extract_preferences(self.user_input, self.food_keywords)

            if self.food_preference is None:
                print("Please give a valid food preference.")
        return

    def ask_price_handler(self):
        """
        Handles asking the user for their price preference
        by prompting repeatedly until a preference is given.
        """

        if self.price_preference is not None:
            return

        print("What price range do you want?")
        while self.price_preference is None:
            self.user_input = input(">>").lower()

            self.price_preference = self.extract_preferences(self.user_input, self.price_keywords)

            if self.price_preference is None:
                print("Please give a valid price preference.")
        return

    def ask_area_handler(self):
        """
        Handles asking the user for their area preference
        by prompting repeatedly until a preference is given.
        """

        if self.area_preference is not None:
            return

        print("What part of town do you have in mind?")
        while self.area_preference is None:
            self.user_input = input(">>").lower()

            self.area_preference = self.extract_preferences(self.user_input, self.area_keywords)

            if self.area_preference is None:
                print("Please give a valid area preference.")
        return

    def found_restaurant_for_recommendation(self):
        """
        Determines the next state after a restaurant recommendation is made:
        end conversation, givedetails about restaurant or recommend more restaurants.

        Returns
        --------
        - (int): next state (END, GIVE_DETAILS, RECOMMEND_MORE)
        """

        dialog_act = self.dialog_act_prediction()

        # If dialog act is bye, go to the "end" state
        if dialog_act in ("bye"):
            return self.END_STATE
        
        # If the user requests details of the recommended restaurant
        elif dialog_act in ("request", "ack", "affirm"):
            return self.GIVE_DETAILS_STATE
        
        # Default case, continue recommending more restaurants
        else:
            return self.SEARCH_MORE_STATE
        
    def search_handler(self):
        """
        Handles the restaurant recommendation process. Returns first restaurant of
        list of matched restaurants with user input. If no matches are found, it
        leads the user to a handler for cases with no recommendations.

        Returns
        --------
        - (int): The next state (NO_MORE_RECOMMENDATIONS, END, GIVE_DETAILS, RECOMMEND_MORE)
        """

        print(f"Ok, I am searching for a restaurant based on the following preferences: {self.food_preference} restaurant at {self.price_preference} price in {self.area_preference} area...")
        
        self.possible_restaurants = self.get_matching_restaurants()

        # If there are no matching restaurants
        if (len(self.possible_restaurants)) == 0:
            print(f"I found no restaurant based on your preferences. Do you want something else?")
            self.user_input = input(">>").lower()
            return self.NO_MORE_RECOMMENDATIONS_STATE
            
        # If there are matching restaurants
        else:
            if (len(self.possible_restaurants)) == 1:
                print("I found ", len(self.possible_restaurants), " restaurant based on your preferences.")
            else:
                print("I found ", len(self.possible_restaurants), " restaurants based on your preferences.")

            print(f"{self.possible_restaurants[0]['restaurantname']} is a nice restaurant serving {self.possible_restaurants[0]['food']} food.")
            print("Do you want details for this restaurant?")
            self.user_input = input(">>").lower()

            return self.found_restaurant_for_recommendation()
            
    def no_more_recommendations_handler(self):
        """
        Handles scenario where no more restaurants match the user's preferences. 
        Asks the user if they want to adjust their preferences or end the conversation.

        Returns
        --------
        - (int): The next state (END or ASK_INITIAL_PREFERENCES_STATE)
        """

        # Predict dialog act
        dialog_act = self.dialog_act_prediction()

        # If dialog act is bye, negate or deny, go to "end" state
        if dialog_act in ("bye","negate","deny"):
            return self.END_STATE
        
        # If the user does not want to end the conversation, go to "ask initial preferences" state
        else:
            # Forget previous user preferences by setting the preferences to None
            self.food_preference = None
            self.price_preference = None
            self.area_preference = None
            return self.ASK_INITIAL_PREFERENCES_STATE
                        
    def search_more_handler(self):
        """
        Handles process of recommending additional restaurants if multiple matching 
        restaurants were found. 
        Removes the previously recommended restaurant from the list and recommends
        a new one if there is one.

        Returns
        --------
        - (int): The next state (END, GIVE_DETAILS, RECOMMEND_MORE, 
        NO_MORE_RECOMMENDATIONS)
        """

        # If there are more restaurants to recommend
        if len(self.possible_restaurants) > 1:
            # Remove the previously recommended restaurant from the list
            self.possible_restaurants.pop(0)

            # Recommend the first restaurant in the list
            print(f"{self.possible_restaurants[0]['restaurantname']} is another nice restaurant serving {self.possible_restaurants[0]['food']} food.")
            print("Do you want details for this restaurant?")
            self.user_input = input(">>").lower()

            return self.found_restaurant_for_recommendation()
        
        # If there are no more restaurants to recommend
        else:
            print("There are no more restaurants to recommend. Do you want something else?")
            self.user_input = input(">>").lower()
            return self.NO_MORE_RECOMMENDATIONS_STATE

    def give_details_handler(self):
        """
        Provides details about a matched restaurant, first item from the
        list of all matched restaurants. Returns next state based on 
        user request for details about more restaurants.

        Returns
        --------
        - (int): The next state (GIVE_DETAILS or END or SEARCH_MORE)
        """

        self.print_details()

        self.user_input = input(">>").lower()
        dialog_act = self.dialog_act_prediction()
        
        # If dialog act is bye, go to the "end" state
        if dialog_act in ("bye"):
            return self.END_STATE
        
        # If dialog act is request, go to the "print details" state
        if dialog_act in ("request"):
            return self.GIVE_DETAILS_STATE
        
        # Default case, continue recommending more restaurants
        else:
            return self.SEARCH_MORE_STATE
        
    # ------------------------ State Transition ------------------------

    def handle_transition(self, state):
        """
        Handles the transitions between different conversational states.
        Calls handler functions based on current state and updates states after.
        """

        if state == self.END_STATE:
            print("I hope I was helpful, goodbye!")
            return
        
        elif state == self.WELCOME_STATE:
            self.welcome_state_handler()
            return self.handle_transition(self.ASK_INITIAL_PREFERENCES_STATE)

        elif state == self.ASK_INITIAL_PREFERENCES_STATE:
            state = self.ask_initial_preferences_handler()
            return self.handle_transition(state)

        elif state == self.ASK_FOOD_STATE:
            self.ask_food_handler()
            return self.handle_transition(self.ASK_PRICE_STATE)

        elif state == self.ASK_PRICE_STATE:
            self.ask_price_handler()
            return self.handle_transition(self.ASK_AREA_STATE)

        elif state == self.ASK_AREA_STATE:
            self.ask_area_handler()
            return self.handle_transition(self.SEARCH_STATE)

        elif state == self.SEARCH_STATE:
            state = self.search_handler()
            return self.handle_transition(state)
        
        elif state == self.NO_MORE_RECOMMENDATIONS_STATE:
            state = self.no_more_recommendations_handler()
            return self.handle_transition(state)

        elif state == self.SEARCH_MORE_STATE:
            state = self.search_more_handler()
            return self.handle_transition(state)

        elif state == self.GIVE_DETAILS_STATE:
            state = self.give_details_handler()
            return self.handle_transition(state)

def print_models_menu():
    """
    Prints menu with the three classification models (RB, RL, DT)  
    so the user can choose a model for dialog act classification.
    """
    print("\nOptions:")
    print("1. Linear Regression Model")
    print("2. Decision Tree Model")
    print("3. Rule Based Model")

def get_model_preference():
    """
    Prompts the user to select a dialog act classification model
    from the available options (Linear Regression, Decision Tree, Rule-Based).
    
    Returns
    --------
    - model_preference (str): The selected model preference constant (LR_MODEL_PREFERENCE, 
                              DT_MODEL_PREFERENCE, or RULE_BASED_MODEL_PREFERENCE).
    """
    # Display the models menu
    print_models_menu()

    # Get input from the user
    choice = input("\nEnter your model choice (1/2/3).\n>>").strip()

    while True:
        
        if(choice == '1'):
            model_preference = LR_MODEL_PREFERENCE
            break
        elif(choice == '2'):
            model_preference = DT_MODEL_PREFERENCE
            break
        elif(choice == '3'):
            model_preference = RULE_BASED_MODEL_PREFERENCE
            break
        
        else:
            # Display the menu
            print_models_menu()

            # Get input from the user
            choice = input("\nInvalid choice. Please select a valid option (1/2/3).\n>>")
    
    return model_preference

# Create an instance and run the system
if __name__ == "__main__":
    # Get model preference from user
    model_preference = get_model_preference()

    print("\nStarting conversation...\n")
    system = RestaurantRecommendationSystem(model_preference)
    system.handle_transition(1) # 1 == WELCOME_STATE
