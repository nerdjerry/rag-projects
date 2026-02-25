"""
src/tools/weather_tool.py

Fetches current weather data from the OpenWeatherMap API.

HOW THE AGENT PASSES PARAMETERS:
    The agent's LLM reads the tool description, extracts the relevant entity
    from the user's question (e.g. "London" from "What's the weather in London?"),
    and passes it as the `city` string to this tool.  The tool's job is simply to
    accept that string and return a human-readable result.

UNITS:
    We use metric (°C, km/h) by default because it is universally understood.
    If your users are in the US you can change `units=metric` to `units=imperial`
    in the API URL to receive °F and mph.
"""

import requests
from langchain.tools import Tool


def create_weather_tool(openweathermap_api_key: str) -> Tool:
    """
    Build a LangChain Tool that returns current weather from OpenWeatherMap.

    Free tier: 60 API calls/minute, no credit card required.
    Sign up at https://openweathermap.org/api

    Args:
        openweathermap_api_key: A valid OpenWeatherMap API key.

    Returns:
        A configured LangChain Tool for weather lookups.
    """

    def get_weather(city: str) -> str:
        """
        Call the OpenWeatherMap "current weather" endpoint for a given city.

        The agent passes the city name exactly as it understands it from the
        user's question — it may be "London", "New York, US", "Paris, FR", etc.
        OpenWeatherMap accepts most common city formats.

        Args:
            city: City name string provided by the agent.

        Returns:
            A single human-readable weather summary string.
        """
        city = city.strip()
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={openweathermap_api_key}&units=metric"
        )

        try:
            response = requests.get(url, timeout=10)

            # OpenWeatherMap returns 404 when the city name isn't recognised.
            if response.status_code == 404:
                return (
                    f"Could not find weather for '{city}'. "
                    "Please check the city name (try adding the country code, "
                    "e.g. 'Paris, FR')."
                )

            response.raise_for_status()
            data = response.json()

            # Extract fields — the API always returns these keys on success.
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            description = data["weather"][0]["description"].capitalize()
            wind_speed_ms = data["wind"]["speed"]
            # Convert m/s → km/h for a more intuitive display.
            wind_kmh = wind_speed_ms * 3.6
            city_name = data.get("name", city)
            country = data.get("sys", {}).get("country", "")
            location = f"{city_name}, {country}" if country else city_name

            return (
                f"Weather in {location}: "
                f"{temp:.1f}°C (feels like {feels_like:.1f}°C), "
                f"{description}, "
                f"Humidity: {humidity}%, "
                f"Wind: {wind_kmh:.1f} km/h"
            )

        except requests.exceptions.Timeout:
            return f"Weather service timed out for '{city}'. Please try again."
        except Exception as exc:
            return f"Could not retrieve weather for '{city}'. Error: {exc}"

    return Tool(
        name="get_weather",
        func=get_weather,
        description=(
            "Get current weather and forecast for any city. "
            "Input: city name (e.g., 'London' or 'New York, US'). "
            "Returns temperature, weather conditions, humidity, and wind speed."
        ),
    )


def create_mock_weather_tool() -> Tool:
    """
    Return a mock weather tool used when no OpenWeatherMap API key is set.

    The mock returns plausible-looking data so that the full agent pipeline can
    be tested without any API keys.  The response clearly labels itself as mock
    data so it is never confused with real weather information.
    """

    def mock_weather(city: str) -> str:
        city = city.strip()
        return (
            f"[MOCK DATA — configure OPENWEATHERMAP_API_KEY for real weather] "
            f"Weather in {city}: 18.0°C (feels like 17.0°C), "
            f"Partly cloudy, Humidity: 65%, Wind: 14.0 km/h"
        )

    return Tool(
        name="get_weather",
        func=mock_weather,
        description=(
            "Get current weather and forecast for any city. "
            "NOTE: Running in mock mode (no API key configured). "
            "Input: city name (e.g., 'London' or 'New York, US')."
        ),
    )
