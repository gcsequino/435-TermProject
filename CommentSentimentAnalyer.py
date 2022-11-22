from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup

class CommentSentimentAnalyer():

    def __init__(self):
        self.sid = SentimentIntensityAnalyzer("/s/bach/a/class/cs435/cs435a/nltk_data/sentiment/vader_lexicon/vader_lexicon.txt")

    def get_sentiment_polarity(self, comment):
        cleaned_comment = self.remove_html_tags(comment)
        return self.sid.polarity_scores(cleaned_comment)

    def remove_html_tags(self, comment):
        return BeautifulSoup(comment, "html.parser").text

if __name__ == "__main__":
    csa = CommentSentimentAnalyer()
    x = input("Enter text to analyze\n")
    while x != "":
        print(csa.get_sentiment_polarity(x))
        x = input("Enter text to analyze\n")
