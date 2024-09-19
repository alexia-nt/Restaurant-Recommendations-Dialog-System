import pandas as pd
import Levenshtein
import pickle

class RestaurantRecommendationSystem:
    DATA_FILE = "data/restaurant_info.csv"
    DT_FILE = "models/dt_model.pkl"
    LR_FILE = "models/lr_model.pkl"
    VECTORIZER_FILE = "models/vectorizer.pkl"

    with open(LR_FILE,'rb') as file:
        LR_model = pickle.load(file)
    with open(VECTORIZER_FILE,'rb') as file:
        vectorizer = pickle.load(file)
    #y_pred_LR = LR_model.predict(vectorizer.transform(["hi"]))[0]

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

    levenshtein_threshold = 1

    possible_restaurants = [] 
    #I'm using this variable to keep track of which restaurants have not been shown to the user yet
    #Maybe there's a better way?

    DONT_CARE_KEYWORDS = [
        "do not care", "don't care", "dont care",
        "do not mind", "don't mind", "dont mind",
        "does not matter", "doesn't matter", "doesnt matter",
        "any", "anywhere", "whatever"
    ]

    def __init__(self):
        self.df = pd.read_csv(self.DATA_FILE)
        self.food_keywords, self.price_keywords, self.area_keywords = self.get_keywords()
        self.state = self.WELCOME_STATE
        self.user_input = ""
        self.food_preference = None
        self.price_preference = None
        self.area_preference = None

    def get_keywords(self):
        food_keywords = list(self.df["food"].dropna().astype(str).unique())
        price_keywords = list(self.df["pricerange"].dropna().astype(str).unique())
        area_keywords = list(self.df["area"].dropna().astype(str).unique())
        return food_keywords, price_keywords, area_keywords

    def extract_initial_preferences(self, utterance, keywords, preference_type):
        if f"any {preference_type}" in utterance:
            return "any"
        
        for keyword in keywords:
            if keyword in utterance:
                return keyword
        
        utterance_words = utterance.lower().split()
        
        for word in utterance_words:
            for keyword in keywords:
                distance = Levenshtein.distance(word.lower(), keyword.lower())
                if distance <= self.levenshtein_threshold:  # Adjust threshold as needed
                    return keyword
        return None

    def extract_preferences(self, utterance, keywords):
        for keyword in self.DONT_CARE_KEYWORDS:
            if keyword in utterance:
                return "any"
        
        for keyword in keywords:
            if keyword in utterance:
                return keyword
        
        utterance_words = utterance.lower().split()
        
        for word in utterance_words:
            for keyword in keywords:
                distance = Levenshtein.distance(word.lower(), keyword.lower())
                if distance <= self.levenshtein_threshold:  # Adjust threshold as needed
                    return keyword
        return None

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

        while self.food_preference is None:
            print("What kind of food would you like?")
            self.user_input = input(">>").lower()
            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print(f"Dialog act: {dialog_act}")
            self.food_preference = self.extract_preferences(self.user_input, self.food_keywords)

        return

    def ask_price_handler(self):
        if self.price_preference is not None:
            return

        while self.price_preference is None:
            print("What price range do you want?")
            self.user_input = input(">>").lower()
            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print(f"Dialog act: {dialog_act}")
            self.price_preference = self.extract_preferences(self.user_input, self.price_keywords)

        return

    def ask_area_handler(self):
        if self.area_preference is not None:
            return

        while self.area_preference is None:
            print("What part of town do you have in mind?")
            self.user_input = input(">>").lower()
            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            print(f"Dialog act: {dialog_act}")
            self.area_preference = self.extract_preferences(self.user_input, self.area_keywords)

        return

    def recommend_handler(self):
        restaurants = self.get_matching_restaurants()
        if (len(restaurants)) == 0:
            print(f"I found no restaurant based on your preferences. Do you want something else?")
            self.user_input = input(">>").lower()
            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            if dialog_act in ("bye","negate","deny"):
                return self.END_STATE
            else:
                self.food_preference = None
                self.price_preference = None
                self.area_preference = None
                self.state = self.ASK_INITIAL_PREFERENCES_STATE
                print(dialog_act)
                return
        else:
            print(f"{restaurants[0]['restaurantname']} is a nice restaurant serving {restaurants[0]['food']} food.")
            self.user_input = input(">>").lower()
            dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
            if dialog_act in ("bye", "ack", "affirm"):
                return self.END_STATE
            elif dialog_act in ("reqmore","reqalts"):
                return self.RECOMMEND_MORE_STATE
            elif dialog_act == "request":
                return self.GIVE_DETAILS_STATE
            else:
                return self.RECOMMEND_MORE_STATE
            
    def no_more_recommendation_handler(self):
        return
                        
    def recommend_more_handler(self):
        #TypeError: 'dict' object cannot be interpreted as an integer
        # Handler to recommend more restaurants
        self.possible_restaurants.pop(next(iter(self.possible_restaurants)))
        
        print(f"{self.possible_restaurants[0]['restaurantname']} is another nice restaurant serving {self.possible_restaurants[0]['food']} food.")
        self.user_input = input(">>").lower()
        dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
        if dialog_act in ("bye", "ack", "affirm"):
            return self.END_STATE
        elif dialog_act in ("reqmore","reqalts"):
            if (len(self.possible_restaurants) > 1):
                return self.RECOMMEND_MORE_STATE
            else:
                print("There are no other restaurants for your preferences.")
                return self.NO_MORE_RECOMMENDATIONS_STATE
        elif dialog_act == "request":
            self.state = self.GIVE_DETAILS_STATE
        else:
            print(f"{self.possible_restaurants[0]['restaurantname']} is another nice restaurant serving {self.possible_restaurants[0]['food']} food.")


    def give_details_handler(self):
        print(f"{self.possible_restaurants[0]['restaurantname']} is a {self.possible_restaurants[0]['pricerange']}ly priced {self.possible_restaurants[0]['food']} restaurant in the {self.possible_restaurants[0]['area']} of town. Phone number: {self.possible_restaurants[0]['phone']}. Address: {self.possible_restaurants[0]['addr']}. Postcode: {self.possible_restaurants[0]['postcode']}.")
        self.user_input = input(">>").lower()
        dialog_act = self.LR_model.predict(self.vectorizer.transform([self.user_input]))[0]
        if dialog_act in ("reqmore","reqalts"):
            return self.RECOMMEND_MORE_STATE
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
            self.state = self.END_STATE
            self.recommend_more_handler()
            return

        elif self.state == self.GIVE_DETAILS_STATE:
            self.state = self.give_details_handler()
            return
            


    def run(self):
        while self.state != self.END_STATE:
            self.handle_transition()

            print("State: ", self.state)
            print("User input: ", self.user_input)
            print("food_type: ", self.food_preference)
            print("price_range: ", self.price_preference)
            print("area: ", self.area_preference)

# Create an instance and run the system
if __name__ == "__main__":
    system = RestaurantRecommendationSystem()
    system.run()

# branch another test
