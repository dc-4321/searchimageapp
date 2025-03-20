import pandas as pd
# -----------------------------------------ww.py-streamlit dashboard--------------
import google.generativeai as genai
import json
import streamlit as st
import requests
from PIL import Image
import io
import os
import math
from dotenv import load_dotenv
import re
from datetime import date
 
# Configure Google Gemini API
load_dotenv()
api_key = os.getenv('API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
 
# Path to the CSV file
CSV_FILE_PATH = r"C:\Users\Admin\Downloads\data.csv"
 
# df = pd.read_csv(CSV_FILE_PATH, encoding="latin1")
# df.to_csv("data.csv", encoding="utf-8", index=False)
 
def parse_query_with_gemini(user_query):
    prompt = f"""
You are an assistant for a sports analytics platform. Your task is to **extract only explicitly mentioned information** from the query and convert it into structured parameters.
   
    ---
    ### **STRICT RULES:**
    1. **DO NOT infer, assume, or interpret meanings beyond the exact words present in the query.**
    2. **ONLY use the predefined categories below**—if the query does not contain an exact match, return `NULL`.
    3. **DO NOT assume an 'event_type', 'mood', or 'sublocation' based on context.** Extract them **only if they appear as-is** in the query.
    4. **DO NOT translate synonyms or implied meanings.** Example:
    - ❌ "receiving an award" does **NOT** mean `"event_type": "award ceremony"`. It should **only** be `"action": "receiving an award"`.
    - ❌ "on the field" does **NOT** mean `"sublocation": "field"`. Ignore it unless "field" is a standalone phrase.
    - ❌ "practice nets" does **NOT** mean `"event_type": "practice"`.
    - ❌ "airport" does **NOT** mean `"action": "travelling"`,`"location": "airport"` . It should **only** be `"sublocation": "airport"`.
    5. When parsing the user input into the json format, make sure to return the values in lower case except for the "Player Name" and "Photographer" fields.
    6. **If a category is missing in the query, set it to NULL.**
    ---
### **User Query:**  
"{user_query}"
 
---
### **Output Format (JSON)**
{{
    "Player Name": [List of player names] (optional),
    "Photographer": NULL (unless explicitly mentioned),
    "Date": NULL (unless explicitly mentioned),
    "TimeOfDay": NULL (unless explicitly mentioned as 'morning', 'afternoon', 'evening', or 'night'),
    "Focus": NULL (unless explicitly mentioned as 'solo' or 'group'),
    "Shot Type": NULL (unless explicitly mentioned as 'close' or 'wide'),
    "event_type": NULL (unless explicitly mentioned as 'practice', 'match', 'fan engagement', 'award ceremony', 'press meet', 'promotional event', 'others'),
    "mood": NULL (unless explicitly mentioned as 'casual', 'celebratory', 'formal'),
    "action": NULL (unless explicitly mentioned as 'stretching', 'training', 'listening', 'talking', 'resting', 'batting', 'bowling', 'fielding',
                     'wicketkeeping', 'running', 'jogging', 'sprinting', 'appealing', 'celebrating', 'catching', 'walking',
                     'strategizing', 'signing', 'waving', 'posing', 'greeting', 'receiving an award', 'sitting', 'travelling', 'reacting'),
    "apparel": NULL (unless explicitly mentioned as 'practice jersey', 'official match jersey', 'protective gear', 'casual attire'),
    "sublocation": NULL (unless explicitly mentioned as 'field', 'practice nets', 'hotel', 'stage', 'stadium', 'airport'),
    "Results": NULL (unless explicitly mentioned)
}}
 
---
### **Examples to Follow (STRICT MATCHING ONLY)**:
✅ **Correct Parsing:**  
**Query:** "JP King receiving an award on the field"  
**Output:**  
{{
    "Player Name": ["JP King"],
    "action": "receiving an award",
    "event_type": NULL,  # 'award ceremony' was NOT explicitly stated
    "mood": NULL,  # 'celebratory' was NOT explicitly stated
    "sublocation": NULL  # 'field' was NOT explicitly stated as a standalone term
}}
 
❌ **Incorrect Parsing (DO NOT DO THIS):**  
**Query:** "JP King receiving an award on the field"  
**Wrong Output:**  
{{
    "Player Name": ["JP King"],
    "action": "receiving an award",
    "event_type": "award ceremony",  # ❌ INCORRECT—was not in the query
    "mood": "celebratory",  # ❌ INCORRECT—was not in the query
    "sublocation": "field"  # ❌ INCORRECT—was inferred, not explicitly stated
}}
 
---
### **Final Reminder:**
- **DO NOT** make assumptions based on context.
- **DO NOT** return values unless there is an **exact match** in the query.
- **DO NOT** interpret words based on meaning—use only predefined options.
- **If a category is missing in the query, set it to NULL.**
"""
 
    response = model.generate_content(prompt)
    return response.text
 
# import streamlit as st
 
# Set page configuration
st.set_page_config(layout="wide", page_title="Image Search")
 
# Custom HTML banner
custom_html = """
    <style>
        .banner {
            width: 100%;
            height: 100px;  /* Increased height for better spacing */
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #FFD700;
            text-align: center;
        }
        .banner img {
            max-height: 100px; /* Adjust size */
        }
    </style>
    <div class="banner">
        <img src="https://logotyp.us/file/super-kings.svg" width="200" height="100">
    </div>
    """
 
# Display banner
st.markdown(custom_html, unsafe_allow_html=True)
 
# Title and description
st.title("Image Search")
 
 
 
tab2, tab1 = st.tabs(["Filter Search", "LLM Search"])
 
df = pd.read_csv(CSV_FILE_PATH)
def filter_images_by_tags(df, player_name=None, photographer=None, date=None, time_of_day=None, focus=None,
                           shot_type=None, event_type=None, mood=None, action=None, apparel=None, sublocation=None):
    grouped = df.groupby("File Name").agg(
        {"Player Name": lambda x: set(x), "Photographer": "first", "Date": "first", "TimeOfDay": "first",
         "Focus": "first", "Shot Type": "first", "event_type": "first", "mood": "first", "action": lambda x: set(x),
         "apparel": "first", "sublocation": "first", "URL": "first"})
 
    # Start with all rows
    result = grouped
 
    # Filtering logic
    generic_terms = {"players", "person", "people"}
    if player_name and not generic_terms.intersection(set(map(str.lower, player_name))):
            player_name_lower = {name.lower() for name in player_name}  # Convert user input to lowercase
            result = result[result["Player Name"].apply(lambda x: any(name.lower() in player_name_lower for name in x))]
    
 
    if photographer:
        result = result[result["Photographer"].str.contains(photographer, case=False, na=False)]
    if date:
        result = result[result["Date"].str.contains(date, case=False, na=False)]
    if time_of_day:
        result = result[result["TimeOfDay"].str.contains(time_of_day, case=False, na=False)]
    if focus:
        result = result[result["Focus"].str.contains(focus, case=False, na=False)]
    if shot_type:
        result = result[result["Shot Type"].str.contains(shot_type, case=False, na=False)]
    if event_type:
        result = result[result["event_type"].str.contains(event_type, case=False, na=False)]
    if mood:
        result = result[result["mood"].str.contains(mood, case=False, na=False)]
   
    if action:
        action_set = set(action) if isinstance(action, list) else {action}
        result = result[result["action"].apply(lambda x: any(act.lower() in {a.lower() for a in x} for act in action_set))]
 
    if apparel:
        result = result[result["apparel"].str.contains(apparel, case=False, na=False)]
    if sublocation:
        result = result[result["sublocation"].str.contains(sublocation, case=False, na=False)]
 
    return result["URL"].tolist()
 
 
# Function to extract view URL and direct image URL from Google Drive URL
# Function to extract view URL and direct image URL from Google Drive URL
def get_drive_view_url_and_direct_link(url):
    file_id = url.split("/")[-2]  # Extract file ID from Google Drive URL
    view_link = f"https://drive.google.com/file/d/{file_id}/view"
    direct_link = f"https://drive.google.com/uc?id={file_id}"
    return view_link, direct_link
 
# Function to fetch the image with retry logic
# Function to fetch the image with retry logic
def fetch_image_with_retry(direct_link, retries=3):
    for _ in range(retries):
        try:
            response = requests.get(direct_link)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            print(f"Error fetching image: {e}")
    return None
 
def display_results():
    current_page = st.session_state.current_page
    num_results = st.session_state.num_results
    result_urls = st.session_state.result_urls
    df = pd.read_csv(CSV_FILE_PATH)  # Load the CSV file to retrieve captions
    start_idx = current_page * num_results
    end_idx = start_idx + num_results
    st.write(f"Displaying results {start_idx + 1} to {min(end_idx, len(result_urls))}:")
    current_results = result_urls[start_idx:end_idx]
 
    # Create 2 rows of results
    rows = 2
    cols = 3
    num_cells = rows * cols
    current_results = current_results[:num_cells]
 
    for row_idx in range(rows):
        cols_in_row = st.columns(cols)
        start_idx_in_row = row_idx * cols
        end_idx_in_row = start_idx_in_row + cols
        for col_idx, (url, col) in enumerate(zip(current_results[start_idx_in_row:end_idx_in_row], cols_in_row)):
            with col:
                if "drive.google.com" in url:
                    view_link, direct_link = get_drive_view_url_and_direct_link(url)
                    image_content = fetch_image_with_retry(direct_link)
                    if image_content:
                        image = Image.open(io.BytesIO(image_content))
                        col.image(image, use_container_width=True)
 
                        # Fetch the caption for the current URL
                        caption_row = df[df['URL'] == url]
                        caption = caption_row["caption"].iloc[0] if not caption_row.empty and "caption" in df.columns else "No caption available"
                       
                        # Display the caption
                        col.write(f"**{caption}**")
                       
                        # Add a link to open in Google Drive
                        col.markdown(f'<a href="{view_link}" target="_blank" style="color: blue;">Open in Google Drive</a>', unsafe_allow_html=True)
                    else:
                        col.write("Error fetching image.")
                else:
                    col.write(f"Invalid URL: {url}")
 
    if end_idx >= len(result_urls):
        st.write("No more results to display.")
 
def app():
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    if "query_submitted" not in st.session_state:
        st.session_state.query_submitted = False
    if "result_urls" not in st.session_state:
        st.session_state.result_urls = []
    if "num_results" not in st.session_state:
        st.session_state.num_results = 6
 
    # 🔍 Tab 1: LLM Search
    with tab1:
        st.subheader("LLM Search")
        st.markdown("""
            This application uses an image repository to query image links based on player names, actions, and other parameters.
           
            **Sample Input:** The inputs can be like "Give me images of Gerald Coetzee  bowling","Images of players travelling".
        """)        
        user_query = st.text_input("Enter query")
 
        if st.button("Submit Query⏩"):
            st.session_state.current_page = 0
            st.session_state.query_submitted = True
            if user_query:
                try:
                    df = pd.read_csv(CSV_FILE_PATH)
                    # df = pd.read_csv(CSV_FILE_PATH, encoding="ISO-8859-1")
                    # df.to_csv(CSV_FILE_PATH, encoding="utf-8", index=False)
                    # Convert specific columns to lowercase
                    columns_to_lowercase = ["Shot Type", "event_type", "mood", "action", "apparel","sublocation"]
                    # Convert player names to lowercase
                    # df["Player Name"] = df["Player Name"].str.lower()
                    
                    for col in columns_to_lowercase:
                        if col in df.columns:  # Ensure column exists to avoid errors
                            df[col] = df[col].astype(str).str.lower()
                    #st.subheader("Lowercase Converted DataFrame")
                    #st.dataframe(df)
                    parsed_query_raw = parse_query_with_gemini(user_query)
                    cleaned_response = parsed_query_raw.strip("```").strip()
                    start_idx = cleaned_response.find('{')
                    end_idx = cleaned_response.rfind('}') + 1
                    valid_json = cleaned_response[start_idx:end_idx]
                    if valid_json:
                        try:
                            parsed_query = json.loads(valid_json)
                            # Write the parsed query on the Streamlit screen
                            #st.write("### Parsed Query:")
                            # st.json(parsed_query)  # Display as a formatted JSON object
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing JSON: {e}")
                            return
                    else:
                        st.error("Valid JSON not found in the response.")
                        return
                    query_player_name = parsed_query.get("Player Name", [])
                    query_action = parsed_query.get("action", None)
                    query_time_of_day = parsed_query.get("TimeOfDay", None)
                    query_focus = parsed_query.get("Focus", None)
                    query_photographer = parsed_query.get("Photographer", None)
                    query_date = parsed_query.get("Date", None)
                    query_shot_type = parsed_query.get("Shot Type", None)
                    query_event_type = parsed_query.get("event_type", None)
                    query_mood = parsed_query.get("mood", None)
                    query_apparel = parsed_query.get("apparel", None)
                    query_sublocation = parsed_query.get("sublocation", None)
                    match = re.search(r"\b(\d+)\s*images?\b", user_query)
                    num_results = int(match.group(1)) if match else 6
    
                    result_urls = filter_images_by_tags(
                    df, query_player_name, query_photographer, query_date, query_time_of_day, query_focus, query_shot_type,
                    query_event_type, query_mood, query_action, query_apparel, query_sublocation
                )
                    st.session_state.result_urls = result_urls
                    st.session_state.num_results = num_results
                    if result_urls:
                        display_results()
                    else:
                        st.write("No matching images found.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
        # Pagination controls inside the app
        if st.session_state.query_submitted and st.session_state.result_urls:
            # "Back" button
            if st.button("Back"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
                    display_results()
    
            # "Next" button
            if st.button("Next"):
                if (st.session_state.current_page + 1) * st.session_state.num_results < len(st.session_state.result_urls):
                    st.session_state.current_page += 1
                    display_results()
with tab2:
    # st.subheader("Apply Filters")
 
    # Create a row with three columns
    cols = st.columns(3)
 
    # Player names filter
    player_names = df["Player Name"].dropna().unique().tolist()
    with cols[0]:
        selected_players = st.multiselect("Select Player Names", player_names, default=[])
 
    # Date range filter (removed as requested)
    with cols[1]:
        start_date = st.date_input("From Date", date.today())
 
    with cols[2]:
        end_date = st.date_input("To Date", date.today())
 
    # Other filters
    filter_columns = {
        "Focus": df["Focus"].dropna().unique().tolist(),
        "Shot Type": df["Shot Type"].dropna().unique().tolist(),
        "Event Type": df["event_type"].dropna().unique().tolist(),
        "Mood": df["mood"].dropna().unique().tolist(),
        "Action": df["action"].dropna().unique().tolist(),
        # "Brands": df["brands_and_logos"].dropna().unique().tolist(),
        "Sub-Location": df["sublocation"].dropna().unique().tolist(),
        "TimeOfDay": df["TimeOfDay"].dropna().unique().tolist()
    }
 
    filters = {}
    cols = st.columns(3)
    for i, (col, options) in enumerate(filter_columns.items()):
        with cols[i % 3]:
            selected_values = st.multiselect(f"Select {col}", options, default=[])
            if selected_values:
                filters[col] = selected_values
 
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
 
    # Apply filters to the DataFrame
    filtered_df = df.copy()
 
    # Player filter
    if selected_players:
        filtered_df = filtered_df[filtered_df["Player Name"].isin(selected_players)]
 
    # Other filters
    for col, selected_values in filters.items():
        column_mapping = {
            "Event Type": "event_type",
            "Mood": "mood",
            "Action": "action",
            # "Brands": "brands_and_logos",
            "Sub-Location": "sublocation"
        }
        col_name = column_mapping.get(col, col)
        if col_name in df.columns:
            filtered_df = filtered_df[filtered_df[col_name].isin(selected_values)]
 
    # Date range filter (if the start and end dates are selected, apply them)
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= pd.to_datetime(start_date)) & (filtered_df["Date"] <= pd.to_datetime(end_date))
        ]

 
    if st.button("Search Images⏩"):
        st.subheader("Filtered Images")
 
 
        if filtered_df.empty:
            st.warning("⚠ No images available for the selected filters.")
        else:
            num_images = len(filtered_df)
            images_per_row = 3
            num_rows = math.ceil(num_images / images_per_row)
            for row in range(num_rows):
                cols = st.columns(images_per_row)
                for col_idx in range(images_per_row):
                    img_idx = row * images_per_row + col_idx
                    if img_idx < num_images:
                        row_data = filtered_df.iloc[img_idx]
                        player_name = row_data["Player Name"]
                        img_url = row_data["URL"]
                        with cols[col_idx]:
                            if isinstance(img_url, str) and "drive.google.com" in img_url:
                                view_link, direct_link = get_drive_view_url_and_direct_link(img_url)
                                image_content = fetch_image_with_retry(direct_link)
                                if image_content:
                                    image = Image.open(io.BytesIO(image_content))
                                    st.image(image, caption=f"👤 {player_name}", use_container_width=True)
                                    st.markdown(f"[🔗 View on Google Drive]({view_link})", unsafe_allow_html=True)
                                else:
                                    st.warning(f"⚠ Error fetching image for {player_name}")
                            else:
                                st.warning(f"⚠ Invalid image URL for {player_name}.")  
if __name__ == "__main__":
    app()