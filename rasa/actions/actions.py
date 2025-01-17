# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
from fuzzywuzzy import process

# Load your dataset
DATA_PATH = "../global_health.csv"

try:
    data = pd.read_csv(DATA_PATH)
    print(data.head())
except FileNotFoundError:
    print(f"File not found!")

# class ActionMeanLifeExpectancy(Action):
#     def name(self) -> Text:
#         return "action_mean_life_expectancy"
#
#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         mean_life_expectancy = data["Life_Expectancy"].mean()
#         dispatcher.utter_message(text=f"The mean life expectancy across all countries is {mean_life_expectancy:.2f} years.")
#         return []
class ActionGetCountryData(Action):
    def name(self) -> str:
        return "action_country_specific"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        geo_country = tracker.get_slot("GPE")

        if not geo_country:
            dispatcher.utter_message(
                text="I didn't understand the country name. Can you please specify the country that you meant?")
            return []

        dispatcher.utter_message(text=f"The selected country is {geo_country}")
        return [SlotSet("GPE", geo_country)]


class ActionGetAverage(Action):
    def name(self) -> str:
        return "action_calculate_column_stat"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain) -> list:
        # Get the column name from the slot
        column_name = tracker.get_slot("column_name")
        value_action = tracker.get_slot("value_action")

        if not value_action:
            dispatcher.utter_message(
                text="I need to know whether you want the 'min', 'max', or 'mean' value. Can you specify?")
            return []

        columns = data.columns.tolist()
        match, score = process.extractOne(column_name, columns)

        if value_action == "mean":
            result = data[match].mean()
        elif value_action == "min":
            result = data[match].min()
        elif value_action == "max":
            result = data[match].max()
        else:
            dispatcher.utter_message(
                text=f"I can only calculate 'min', 'max', or 'mean'. You requested '{value_action}'.")
            return []

        if not column_name:
            dispatcher.utter_message(text="I couldn't understand the column name. Can you clarify?")
            return []

        # Simulate processing the column name
        dispatcher.utter_message(text=f"The {value_action} for {match} is {result}")
        return [SlotSet("column_name", column_name), SlotSet("value_action", value_action)]