import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class EmotionalIntelligence:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_emotion(self, text):
        sentiment = self.sia.polarity_scores(text)
        return sentiment

    def resonate_empathy(self, emotion):
        if emotion['compound'] >= 0.05:
            return "You seem happy, what's bringing you joy?"
        elif emotion['compound'] <= -0.05:
            return "You seem sad, would you like to talk about it?"
        else:
            return "You seem neutral, how can I help?"
