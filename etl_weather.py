
#                 OpenWeather ETL Project – AIMS Intern 
#              =====================================================

#   Project Title:  ETL Pipeline for Weather Data using OpenWeather API & PostgreSQL
   
#   Prepared by:  Damas NIYONKURU

#   Date:  25 August 2025

#         ---------------------------------------------------------------
#   Project Overview:
#   This project fetches live weather data from the OpenWeather API
#   and loads it into a PostgreSQL database. The pipeline simulates
#   an ETL (Extract, Transform, Load) workflow for real-time data.

# ===============================================================

import requests  # We import the requests library to make HTTP requests to the OpenWeather API.
import psycopg2  # We import psycopg2 to connect to our PostgreSQL database.
from datetime import datetime  # We import datetime to convert Unix timestamps to readable format

API_KEY = 'd1fcb4f6d3a4a69679e237c0723f1918' # # We define the API_KEY using Valid OpenWeather API key
CITIES = ['Kigali', 'Accra', 'Nairobi', 'Cairo', 'Lagos', 'Johannesburg',
          'Dakar', 'Addis Ababa', 'Kinshasa', 'Algiers']  #  cities to fetch data for.
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'  # We set the BASE_URL to the Current Weather API.


DB_HOST = '127.0.0.1' # We configure the database host using the localhost IP.
DB_NAME = 'project2'  # We specify the database name.
DB_USER = 'postgres'  # We set the database user.
DB_PASSWORD = 'Samad@0787933080' # # We define the database password in PostgreSQL

# Fetch weather data for a given city
def fetch_weather(city):
    params = {'q': city, 'appid': API_KEY}  # We create a params dictionary with city and API key for the request.
    try:
        response = requests.get(BASE_URL, params=params) # We send a GET request to the API with our parameters.
        response.raise_for_status()  # We raise an exception for any HTTP errors.
        data = response.json()  # We parse the JSON response into a dictionary.
        
         # We return a dictionary with the extracted weather data.
        return {
            'city': data.get('name'),
            'country': data['sys'].get('country'),
            'longitude': data['coord'].get('lon'),
            'latitude': data['coord'].get('lat'),
            'weather_id': data['weather'][0].get('id'),
            'weather_main': data['weather'][0].get('main'),
            'weather_description': data['weather'][0].get('description'),
            'weather_icon': data['weather'][0].get('icon'),
            'base': data.get('base'),
            'temperature': data['main'].get('temp'),
            'feels_like': data['main'].get('feels_like'),
            'temp_min': data['main'].get('temp_min'),
            'temp_max': data['main'].get('temp_max'),
            'pressure': data['main'].get('pressure'),
            'humidity': data['main'].get('humidity'),
            'sea_level': data['main'].get('sea_level'),
            'grnd_level': data['main'].get('grnd_level'),
            'visibility': data.get('visibility'),
            'wind_speed': data['wind'].get('speed'),
            'wind_deg': data['wind'].get('deg'),
            'wind_gust': data['wind'].get('gust'),
            'rain_1h': data.get('rain', {}).get('1h'),
            'snow_1h': data.get('snow', {}).get('1h'),
            'clouds_all': data['clouds'].get('all'),
            'data_time': datetime.fromtimestamp(data.get('dt')),
            'sys_type': data['sys'].get('type'),
            'sys_id': data['sys'].get('id'),
            'sunrise': datetime.fromtimestamp(data['sys'].get('sunrise')),
            'sunset': datetime.fromtimestamp(data['sys'].get('sunset')),
            'timezone': data.get('timezone'),
            'city_id': data.get('id'),
            'cod': data.get('cod')
        }

    except Exception as e:
        print(f"Error fetching data for {city}: {e}")
        return None

# We Connect to PostgreSQL using our database credentials.
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor() # We create a cursor to execute SQL queries.
    print("Connected to PostgreSQL database.")

    # Create table from OpenWeather response
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100),
            country VARCHAR(50),
            longitude FLOAT,
            latitude FLOAT,
            weather_id INTEGER,
            weather_main VARCHAR(50),
            weather_description TEXT,
            weather_icon VARCHAR(10),
            base VARCHAR(50),
            temperature FLOAT,
            feels_like FLOAT,
            temp_min FLOAT,
            temp_max FLOAT,
            pressure INTEGER,
            humidity INTEGER,
            sea_level INTEGER,
            grnd_level INTEGER,
            visibility INTEGER,
            wind_speed FLOAT,
            wind_deg INTEGER,
            wind_gust FLOAT,
            rain_1h FLOAT,
            snow_1h FLOAT,
            clouds_all INTEGER,
            data_time TIMESTAMP,
            sys_type INTEGER,
            sys_id INTEGER,
            sunrise TIMESTAMP,
            sunset TIMESTAMP,
            timezone INTEGER,
            city_id INTEGER,
            cod INTEGER
        )
    """)
    conn.commit()  # We commit the table creation.
    print("weather_data table created.")

# We handle any database connection errors.
except psycopg2.Error as e:
    print(f"Database connection failed: {e}")
    exit()

# Insert weather data into PostgreSQL
# We loop through each city to fetch
for city in CITIES:
    weather = fetch_weather(city)  # We fetch weather data for the current city.

    if weather:
        try:
            # We def ine the INSERT query with placeholders for security.
            insert_query = """
                INSERT INTO weather_data (
                    city, country, longitude, latitude, weather_id, weather_main, weather_description, weather_icon,
                    base, temperature, feels_like, temp_min, temp_max, pressure, humidity, sea_level, grnd_level,
                    visibility, wind_speed, wind_deg, wind_gust, rain_1h, snow_1h, clouds_all, data_time,
                    sys_type, sys_id, sunrise, sunset, timezone, city_id, cod
                )
                VALUES (%(city)s, %(country)s, %(longitude)s, %(latitude)s, %(weather_id)s, %(weather_main)s,
                        %(weather_description)s, %(weather_icon)s, %(base)s, %(temperature)s, %(feels_like)s,
                        %(temp_min)s, %(temp_max)s, %(pressure)s, %(humidity)s, %(sea_level)s, %(grnd_level)s,
                        %(visibility)s, %(wind_speed)s, %(wind_deg)s, %(wind_gust)s, %(rain_1h)s, %(snow_1h)s,
                        %(clouds_all)s, %(data_time)s, %(sys_type)s, %(sys_id)s, %(sunrise)s, %(sunset)s,
                        %(timezone)s, %(city_id)s, %(cod)s
                )
            """
            cursor.execute(insert_query, weather)
            print(f"Inserted data for {city}")
        except psycopg2.Error as e:
            print(f"Error inserting data for {city}: {e}")
            conn.rollback()

conn.commit()  # We commit any successful changes.
cursor.close()  # We close the cursor and connection to free resources.
conn.close()
print("ETL process completed successfully.")
