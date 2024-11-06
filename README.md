# nlppra
# Install required libraries
!pip install transformers torch

# Import libraries
from transformers import pipeline

# Load the sentiment analysis pipeline for a multilingual model
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define the Hindi NASA text
hindi_text = """
2010 में, नासा की ग्रह रक्षा टीम ने पृथ्वी के पास 1 किलोमीटर चौड़े 90 प्रतिशत क्षुद्रग्रहों की पहचान की और उन्हें सूचीबद्ध किया।
ये 'पृथ्वी के निकट वस्तुएं' या NEO के रूप में जानी जाती हैं, जिनके आकार पहाड़ों जैसे हैं और इनमें पृथ्वी की कक्षा में 50 मिलियन किलोमीटर के भीतर की कोई भी वस्तु शामिल है।
"""

# Perform sentiment analysis
result = sentiment_pipeline(hindi_text)
print("Sentiment Analysis Result:", result)