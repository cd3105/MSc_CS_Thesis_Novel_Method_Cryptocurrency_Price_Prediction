from pytrends.request import TrendReq
import pandas as pd
from time import sleep
import matplotlib.pyplot as plt

# Initialize a Google Trends session
pytrends = TrendReq(hl='en-US')

sleep(60)

# Define search terms
keywords = ["Python Programming", "Data Science", "Machine Learning"]
# Build payload
pytrends.build_payload(kw_list=keywords, timeframe='today 12-m', geo='US')

# Fetch interest over time
interest_over_time_df = pytrends.interest_over_time(sleep=60)
# Display the data
print(interest_over_time_df.head())


related_queries = pytrends.related_queries()
# Display related queries for each term
for key, value in related_queries.items():
    print(f"Related queries for {key}:")
    print(value['top'])

# Fetch interest by region
interest_by_region_df = pytrends.interest_by_region(resolution='COUNTRY')
# Display interest by region
print(interest_by_region_df.head())

# Plotting a bar chart for top countries
interest_by_region_df.sort_values(by='Python Programming', ascending=False).head(10).plot(kind='bar', figsize=(10, 6))
plt.title('Top 10 Countries Interested in Python Programming')
plt.xlabel('Country')
plt.ylabel('Interest Level')
plt.grid()
plt.show()

# Building payload with a category filter (e.g., 'Computer & Electronics')
pytrends.build_payload(kw_list=["Python"], cat=5, timeframe='today 3-m', geo='US')
# Extracting and exporting data to a CSV file
interest_over_time = pytrends.interest_over_time()
interest_over_time.to_csv('google_trends_data.csv')
