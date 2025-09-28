# genai_utils.py
from google import genai
import os

client = genai.Client()

def explain_grid_instability(county_name: str, state_name: str, predicted_customers: float, duration: float, event_type: str) -> str:
    """
    Generate a textual explanation for why a power grid in a county may become unstable.
    """
    prompt = (
        f"Explain the potential causes of power grid instability in {county_name}, {state_name}.\n"
        f"Predicted affected customers: {predicted_customers}\n"
        f"Outage duration: {duration} minutes\n"
        f"Event type: {event_type}\n"
        "Provide a clear, concise explanation suitable for end users."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text
