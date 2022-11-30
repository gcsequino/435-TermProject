from nltk.sentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup

class CommentSentimentAnalyzer():

    def __init__(self):
        self.sid = SentimentIntensityAnalyzer("/s/bach/a/class/cs435/cs435a/nltk_data/sentiment/vader_lexicon/vader_lexicon.txt")

    def get_sentiment_polarity(self, comment):
        cleaned_comment = self.remove_html_tags(comment)
        return self.sid.polarity_scores(cleaned_comment)

    def remove_html_tags(self, comment):
        try:
            return BeautifulSoup(comment, "html.parser").text
        except:
            return ""


if __name__ == "__main__":
    csa = CommentSentimentAnalyzer()
    x = input("Enter text to analyze\n")
    while x != "":
        print(csa.get_sentiment_polarity(x))
        x = input("Enter text to analyze\n")

