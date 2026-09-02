from __future__ import annotations


HF_LOCATIONS = {
    "Birmingham, United Kingdom": {
        "name": "Birmingham, United Kingdom", "lat": 52.4862, "lon": -1.8904, "type": "city",
    },
    "Dubai, United Arab Emirates": {
        "name": "Dubai, United Arab Emirates", "lat": 25.2048, "lon": 55.2708, "type": "city",
    },
    "Gander, Canada": {
        "name": "Gander, Canada", "lat": 48.9569, "lon": -54.6089, "type": "city",
    },
    "London, United Kingdom": {
        "name": "London, United Kingdom", "lat": 51.5074, "lon": -0.1278, "type": "city",
    },
    "Madrid, Spain": {
        "name": "Madrid, Spain", "lat": 40.4168, "lon": -3.7038, "type": "city",
    },
    "New York, United States": {
        "name": "New York, United States", "lat": 40.7128, "lon": -74.0060, "type": "city",
    },
    "North Atlantic corridor": {
        "name": "North Atlantic corridor", "lat": 51.0, "lon": -32.0, "type": "corridor",
    },
    "Reykjavik, Iceland": {
        "name": "Reykjavik, Iceland", "lat": 64.1466, "lon": -21.9426, "type": "city",
    },
    "Shannon, Ireland": {
        "name": "Shannon, Ireland", "lat": 52.7038, "lon": -8.8641, "type": "city",
    },
    "Singapore": {
        "name": "Singapore", "lat": 1.3521, "lon": 103.8198, "type": "city",
    },
    "Tokyo, Japan": {
        "name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503, "type": "city",
    },
    "Toronto, Canada": {
        "name": "Toronto, Canada", "lat": 43.6532, "lon": -79.3832, "type": "city",
    },
}

DEFAULT_HF_ROUTE_SCENARIO = "Birmingham → New York"

HF_ROUTE_SCENARIOS = {
    DEFAULT_HF_ROUTE_SCENARIO: (
        "Birmingham, United Kingdom", "New York, United States",
    ),
    "London → New York": ("London, United Kingdom", "New York, United States"),
    "Birmingham → North Atlantic corridor": (
        "Birmingham, United Kingdom", "North Atlantic corridor",
    ),
    "London → Toronto": ("London, United Kingdom", "Toronto, Canada"),
    "London → Reykjavik": ("London, United Kingdom", "Reykjavik, Iceland"),
    "London → Dubai": ("London, United Kingdom", "Dubai, United Arab Emirates"),
    "London → Singapore": ("London, United Kingdom", "Singapore"),
    "London → Tokyo": ("London, United Kingdom", "Tokyo, Japan"),
}


def location_names() -> list[str]:
    return sorted(HF_LOCATIONS)


def resolve_route_scenario(name: str) -> tuple[dict, dict]:
    origin_name, target_name = HF_ROUTE_SCENARIOS[name]
    return dict(HF_LOCATIONS[origin_name]), dict(HF_LOCATIONS[target_name])


def resolve_location(name: str) -> dict:
    return dict(HF_LOCATIONS[name])
