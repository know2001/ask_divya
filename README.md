# ask_divya
Move over, Emma

Python scripts run in the following order:

Within folder 'scraper_embeddings':
1_scraper.ipynb -> produces 2_scraped_sections.csv
3_text_preprocessing_and_embedding.ipynb -> produces 4_clean_text.csv and 5_embeddings.csv

Within folder 'app':
app.py -> runs the streamlit app using an OpenAI API key

To run the app:
>>streamlit run app.py

Make sure the OpenAI API key has been inserted in the .streamlit/secrets.toml folder before running.

