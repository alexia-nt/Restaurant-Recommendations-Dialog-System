import pandas as pd
import Levenshtein
import pickle

class RestaurantRecommendationSystem:
    DATA_FILE = "data/restaurant_info.csv"
    DT_FILE = "models/dt_model.pkl"
    LR_FILE = "models/lr_model.pkl"
    VECTORIZER_FILE = "models/vectorizer.pkl"

    WELCOME_STATE = 1
    ASK_INITIAL_PREFERENCES_STATE = 2
    ASK_FOOD_STATE = 3
    ASK_PRICE_STATE = 4
    ASK_AREA_STATE = 5
    RECOMMEND_STATE = 6
    NO_MORE_RECOMMENDATIONS_STATE = 7
    GIVE_DETAILS_STATE = 8
    RECOMMEND_MORE_STATE = 9
    END_STATE = 10

    # levenshtein_threshold = 1
    # possible_restaurants = []

    DONT_CARE_KEYWORDS = [
        "do not care", "don't care", "dont care",
        "do not mind", "don't mind", "dont mind",
        "does not matter", "doesn't matter", "doesnt matter",
        "any", "anywhere", "whatever"
    ]

    def __init__(self):
        # Read restaurant data
        self.df = pd.read_csv(self.DATA_FILE)

        # Extract keywords for preferences
        self.food_keywords, self.price_keywords, self.area_keywords = self.get_keywords()
        # Add "center" in area keywords, so it is also included in the
        # area_keywords list and not only the "centre" area keyword
        self.area_keywords.append("center")

        # Initialize state
        self.state = self.WELCOME_STATE

        # Initialize variables
        self.user_input = ""
        self.food_preference = None
        self.price_preference = None
        self.area_preference = None

        self.possible_restaurants = []
        self.levenshtein_threshold = 1

        
        with open(self.LR_FILE,'rb') as file:
            self.LR_model = pickle.load(file)
        with open(self.VECTORIZER_FILE,'rb') as file:
            self.vectorizer = pickle.load(file)

    def get_keywords(self):
        food_keywords = list(self.df["food"].dropna().astype(str).unique())
        price_keywords = list(self.df["pricerange"].dropna().astype(str).unique())
        area_keywords = list(self.df["area"].dropna().astype(str).unique())
        return food_keywords, price_keywords, area_keywords

    # Come to this function either from extract_initial_preferences or extract_preferences,
    # after they have checked for "any" preferences
    def extract_specific_preferences(self, utterance, keywords):
        for keyword in keywords:
            if keyword in utterance:
                return keyword
        
        utterance_words = utterance.lower().split()
        
        for word in utterance_words:
            # If the word from user utterance is "want", do not check levenshtein distance,
            # because it would be matched as "west" area preference for threshold >=2
            if word == "want":
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
        if f"any {preference_type}" in utterance:
            return "any"
        
        return self.extract_specific_preferences(utterance, keywords)

    def extract_preferences(self, utterance, keywords):
        for keyword in self.DONT_CARE_KEYWORDS:
            if keyword in utterance:
                return "any"
            
        return self.extract_specific_preferences(utterance, keywords)

    def get_matching_restaurants(self):
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

    # ------------------------ State Handlers ------------------------

    def welcome_state_handler(self):
        print("Hello, welcome to the restaurant recommendations dialogue system! You can ask for restaurants by area, price range, or food type.")
        self.user_input = input(">>").lower()
        dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
        print(f"Dialog act: {dialog_act}")
        return

    def ask_initial_preferences_handler(self):
        self.food_preference = self.extract_initial_preferences(self.user_input, self.food_keywords, "food")
        self.price_preference = self.extract_initial_preferences(self.user_input, self.price_keywords, "price")
        self.area_preference = self.extract_initial_preferences(self.user_input, self.area_keywords, "area")

        if self.food_preference is None:
            return self.ASK_FOOD_STATE
        
        if self.price_preference is None:
            return self.ASK_PRICE_STATE
        
        if self.area_preference is None:
            return self.ASK_AREA_STATE

        return self.RECOMMEND_STATE

    def ask_food_handler(self):
        if self.food_preference is not None:
            return

        print("What kind of food would you like?")
        while self.food_preference is None:
            self.user_input = input(">>").lower()

            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print(f"Dialog act: {dialog_act}")

            self.food_preference = self.extract_preferences(self.user_input, self.food_keywords)

            if self.food_preference is None:
                print("Please give a valid food preference.")
        return

    def ask_price_handler(self):
        if self.price_preference is not None:
            return

        print("What price range do you want?")
        while self.price_preference is None:
            self.user_input = input(">>").lower()

            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print(f"Dialog act: {dialog_act}")

            self.price_preference = self.extract_preferences(self.user_input, self.price_keywords)

            if self.price_preference is None:
                print("Please give a valid price preference.")
        return

    def ask_area_handler(self):
        if self.area_preference is not None:
            return

        print("What part of town do you have in mind?")
        while self.area_preference is None:
            self.user_input = input(">>").lower()

            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print(f"Dialog act: {dialog_act}")

            self.area_preference = self.extract_preferences(self.user_input, self.area_keywords)

            if self.area_preference is None:
                print("Please give a valid area preference.")
        return

    def recommend_handler(self):
        # Grounding
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
            
            self.user_input = input(">>").lower()
            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print("dialog act: ", dialog_act)

            # If dialog act is bye, ack or affirm, go to "end" state 
            if dialog_act in ("bye", "ack", "affirm"):
                return self.END_STATE
            
            # If dialog act is reqmore or reqalts, go to "recommend more restaurants" state (reqalts might be wrong here ??)
            elif dialog_act in ("reqmore","reqalts"):
                return self.RECOMMEND_MORE_STATE
            
            # If dialog act is request, user wants the details of the recommended restaurant
            elif dialog_act == "request":
                return self.GIVE_DETAILS_STATE
             
            # Default case, continue recommending more restaurants
            else:
                return self.RECOMMEND_MORE_STATE
            
    def no_more_recommendations_handler(self):
        # Predict dialog act
        dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
        print("dialog act: ", dialog_act)

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
                        
    def recommend_more_handler(self):
        # If there are more restaurants to recommend
        if len(self.possible_restaurants) > 1:
            # Remove the previously recommended restaurant from the list
            self.possible_restaurants.pop(0)

            # Recommend the first restaurant in the list
            print(f"{self.possible_restaurants[0]['restaurantname']} is another nice restaurant serving {self.possible_restaurants[0]['food']} food.")

            # Get user input and predict dialog act
            self.user_input = input(">>").lower()
            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print("dialog act: ", dialog_act)

            # If dialog act is bye, ack, or affirm, go to the "end" state
            if dialog_act in ("bye", "ack", "affirm"):
                return self.END_STATE
            
            # If the user requests more recommendations or alternatives
            elif dialog_act in ("reqmore", "reqalts"):
                return self.RECOMMEND_MORE_STATE
            
            # If the user requests details of the recommended restaurant
            elif dialog_act == "request":
                return self.GIVE_DETAILS_STATE
            
            # Default case, continue recommending more restaurants
            else:
                return self.RECOMMEND_MORE_STATE
        
        # If there are no more restaurants to recommend
        else:
            print("There are no more restaurants to recommend. Do you want something else?")
            self.user_input = input(">>").lower()
            return self.NO_MORE_RECOMMENDATIONS_STATE

    def give_details_handler(self):
        print(f"I have the following details for {self.possible_restaurants[0]['restaurantname']} restaurant. Phone number: {self.possible_restaurants[0]['phone']}. Address: {self.possible_restaurants[0]['addr']}. Postcode: {self.possible_restaurants[0]['postcode']}.")
        self.user_input = input(">>").lower()
        dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
        print("dialog act: ", dialog_act)

        # If the dialog act is reqmore or reqalts, go to "recommend more restaurants" state (reqalts might be wrong here ??)
        if dialog_act in ("reqmore","reqalts"):
            return self.RECOMMEND_MORE_STATE
        
        # maybe add the choice to go to "ask initial preferences" state ??

        # Else, go to end state
        else:
            return self.END_STATE
        
    # ------------------------ State Transition ------------------------

    def handle_transition(self):
        if self.state == self.WELCOME_STATE:
            self.welcome_state_handler()
            self.state = self.ASK_INITIAL_PREFERENCES_STATE
            return

        elif self.state == self.ASK_INITIAL_PREFERENCES_STATE:
            self.state = self.ask_initial_preferences_handler()
            return

        elif self.state == self.ASK_FOOD_STATE:
            self.ask_food_handler()
            self.state = self.ASK_PRICE_STATE
            return

        elif self.state == self.ASK_PRICE_STATE:
            self.ask_price_handler()
            self.state = self.ASK_AREA_STATE
            return

        elif self.state == self.ASK_AREA_STATE:
            self.ask_area_handler()
            self.state = self.RECOMMEND_STATE
            return

        elif self.state == self.RECOMMEND_STATE:
            self.state = self.recommend_handler()
            return 
        
        elif self.state == self.NO_MORE_RECOMMENDATIONS_STATE:
            self.state = self.no_more_recommendations_handler()
            return

        elif self.state == self.RECOMMEND_MORE_STATE:
            self.state = self.recommend_more_handler()
            return

        elif self.state == self.GIVE_DETAILS_STATE:
            self.state = self.give_details_handler()
            return

    def run(self):
        while self.state != self.END_STATE:
            self.handle_transition()

            print("State: ", self.state)
        
        print("Goodbye, I hope I was helpful!")

# Create an instance and run the system
if __name__ == "__main__":
    system = RestaurantRecommendationSystem()
    system.run()
