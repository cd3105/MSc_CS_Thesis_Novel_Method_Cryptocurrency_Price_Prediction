# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "Bitcoin (BTC) touches $29k, Ethereum (ETH) Set To Explode, RenQ Finance (RENQ) Crosses Massive Milestone"
result = classifier(text)
print(result)
