import requests  # We import the requests library to make HTTP requests to the OpenWeather API.
import psycopg2  # We import psycopg2 to connect to our PostgreSQL database.
from datetime import datetime  # We import datetime to convert Unix timestamps to readable format

API_KEY = 'd1fcb4f6d3a4a69679e237c0723f1918' # We define the API_KEY using Valid OpenWeather API key
CITIES = ['Kigali', 'Accra', 'Nairobi', 'Cairo', 'Lagos', 'Johannesburg', 
          'Dakar', 'Addis Ababa', 'Kinshasa', 'Algiers'] #  cities to fetch data for.
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'  # We set the BASE_URL to the Current Weather API endpoint.


DB_HOST = '127.0.0.1' # We configure the database host using the localhost IP.
DB_NAME = 'project2' # We specify the database name.
DB_USER = 'postgres'  # We set the database user.
DB_PASSWORD = 'Samad@0787933080'  # # We define the database password in PostgreSQL

# We define a function to fetch weather data for a given city.
def fetch_weather(city):
    """Fetch current weather data from OpenWeather API for a given city."""
    params = {'q': city, 'appid': API_KEY}   # We create a params dictionary with city and API key for the request.
    url = f"{BASE_URL}?q={city}&appid={API_KEY}"  # We build the full URL for debugging purposes.
    print(f"URL: {url}")   # We print the URL to verify what we’re requesting.

    # We use a try-except to handle potential API errors.
    try:
        response = requests.get(BASE_URL, params=params)  # We send a GET request to the API with our parameters.
        print(f"status code: {response.status_code}")  # We print the status code to confirm the request’s success.

        # We check for a 401 error and print details if it occurs.
        if response.status_code == 401:
            print(f"Error details: {response.json()}")
            return None

        response.raise_for_status()  # We raise an exception for any HTTP errors.
        data = response.json()  # We parse the JSON response into a dictionary.

        # We return a dictionary with the extracted weather data.
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'longitude': data['coord']['lon'],
            'latitude': data['coord']['lat'],
            'weather_description': data['weather'][0]['description'],
            'temperature': data['main']['temp'] - 273.15,
            'humidity': data['main']['humidity'],
            'timestamp': datetime.fromtimestamp(data['dt'])
        }

    # We catch HTTP errors and print them for debugging.
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for {city}: {e}")
        return None

    # We catch any other request exceptions and print them for debugging.
    except requests.exceptions.RequestException as e:
        print(f"Error in fetching data for {city}: {e}")
        return None

# We attempt to connect to our PostgreSQL database.
try:
    # We establish a connection using our database credential s.
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()  # We create a cursor to execute SQL queries.
    print("Connected to PostgreSQL database.")  # We print a confirmation of successful connection.

    # We create the weather_data table. This table contains many columns, 
    # but we sample a few of them that are needed for answering Part 4 – Analysis Queries.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            country VARCHAR(50),
            longitude FLOAT,
            latitude FLOAT,
            weather_description TEXT,
            temperature FLOAT,
            humidity INTEGER,
            timestamp TIMESTAMP NOT NULL
        )
    """)
    conn.commit()   # We commit the table creation.
    print("weather_data table is created") # We print confirmation of table creation.

# We handle any database connection errors.
except psycopg2.Error as e:
    print(f"Database connection failed: {e}")
    exit()

success = True  # We initialize a flag to track overall success.
# We loop through each city to fetch and insert weather data.
for city in CITIES:
    weather = fetch_weather(city)  # We fetch weather data for the current city.

    # We proceed only if we successfully fetched data.
    if weather:
        try:
            # We check if the city and timestamp combination already exists to avoid duplicates.
            cursor.execute("SELECT 1 FROM weather_data WHERE city = %s AND timestamp = %s", 
                          (weather['city'], weather['timestamp']))
            if cursor.fetchone() is None:
                # We define the INSERT query with placeholders for security.
                insert_query = """
                INSERT INTO weather_data (city, country, longitude, latitude, weather_description, temperature, humidity, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                # We execute the query with the weather data.
                cursor.execute(insert_query, (
                    weather['city'],
                    weather['country'],
                    weather['longitude'],
                    weather['latitude'],
                    weather['weather_description'],
                    weather['temperature'],
                    weather['humidity'],
                    weather['timestamp']
                ))
                print(f"Inserted data for {city}")   # We print a success message for each inserted city.
            else:
                # We print a message if data already exists for the city at that timestamp.
                print(f"Data for {city} at {weather['timestamp']} already exists, skipping insertion.")

        # We catch any database insertion errors and rollback.
        except psycopg2.Error as e:
            print(f"Error inserting data for {city}: {e}")
            conn.rollback()
            success = False
    else:
        print(f"Skipping insertion for {city} due to failed data fetch.")  # We print a message if data fetch failed, skipping insertion.
        success = False

conn.commit() # We commit any successful changes.

# We close the cursor and connection to free resources.
cursor.close()
conn.close()
print("ETL process completed successfully")