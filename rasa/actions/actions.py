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
import pandas as pd

# Load dataset
data = pd.read_csv("global_health_data.csv")

class ActionGetObesityStats(Action):
    def name(self) -> Text:
        return "action_get_obesity_stats"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        country = tracker.get_slot("country")
        if country:
            country_data = data[data['Country'] == country]
            obesity = country_data['Obesity_Rate_Percent'].values[0]
            overweight = country_data['Overweight_Rate_Percent'].values[0]
            dispatcher.utter_message(text=f"In {country}, the obesity rate is {obesity}% and the overweight rate is {overweight}%.")
        else:
            mean_obesity = data['Obesity_Rate_Percent'].mean()
            mean_overweight = data['Overweight_Rate_Percent'].mean()
            dispatcher.utter_message(text=f"The mean obesity rate is {mean_obesity:.2f}% and the mean overweight rate is {mean_overweight:.2f}%.")
        return []
