"""
Weather Tool — Current Weather via OpenWeatherMap
==================================================
Uses the OpenWeatherMap free API to fetch current weather conditions.

Free tier: 60 calls/minute, 1,000,000 calls/month
Sign up:   https://openweathermap.org/api  (free account)

How agents pass parameters to tools:
  LangChain tools receive a SINGLE string input from the agent.  The agent's
  LLM formats the user's request into that string.  For example:
    User: "What's the weather like in Tokyo?"
    Agent thinks: "I need to call the weather tool"
    Agent passes: "Tokyo" as the input string to this tool

  If your tool needs multiple parameters (e.g., city AND units), you have
  two options:
    1. Parse them from a single string (what we do here)
    2. Use a StructuredTool with a Pydantic schema (more advanced)
"""

import os
import requests
from langchain.tools import Tool


def create_weather_tool(api_key: str = None):
    """
    Create a LangChain Tool that fetches weather from OpenWeatherMap.

    Args:
        api_key: Your OpenWeatherMap API key.  Falls back to the environment
                 variable OPENWEATHERMAP_API_KEY if not provided.

    Returns:
        A LangChain Tool the agent can invoke by name.
    """

    def _get_weather(query: str) -> str:
        """
        Fetch current weather for a city name.

        Args:
            query: A city name like "London", "New York", "Tokyo".
        """
        # Resolve API key: explicit param → environment variable → missing
        effective_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        if not effective_key:
            return (
                "Weather data is unavailable because no API key is configured. "
                "Set OPENWEATHERMAP_API_KEY in your .env file. "
                "Get a free key at https://openweathermap.org/api"
            )

        city = query.strip()
        if not city:
            return "Please provide a city name (e.g., 'London', 'New York')."

        try:
            # --- Call OpenWeatherMap Current Weather API ---
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": effective_key,
                "units": "metric",  # Celsius; use "imperial" for Fahrenheit
            }
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 404:
                return f"City '{city}' not found. Try a different spelling or add the country code (e.g., 'London,UK')."

            if response.status_code == 401:
                return "Invalid OpenWeatherMap API key. Please check your configuration."

            response.raise_for_status()
            data = response.json()

            # --- Parse the response ---
            weather_desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            city_name = data["name"]
            country = data["sys"]["country"]

            return (
                f"🌤  Weather in {city_name}, {country}:\n"
                f"  Conditions  : {weather_desc}\n"
                f"  Temperature : {temp}°C (feels like {feels_like}°C)\n"
                f"  Humidity    : {humidity}%\n"
                f"  Wind Speed  : {wind_speed} m/s"
            )

        except requests.exceptions.Timeout:
            return "Weather API request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "Could not connect to the weather service. Check your internet connection."
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"

    tool = Tool(
        name="weather",
        func=_get_weather,
        description=(
            "Use this to get current weather conditions for a city. "
            "Input should be a city name (e.g., 'London', 'New York', 'Tokyo'). "
            "Returns temperature, conditions, humidity, and wind speed. "
            "For weather forecasts or historical data, use web_search instead."
        ),
    )

    return tool
