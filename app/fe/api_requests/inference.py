import requests
from core.config import settings

def make_inference_request(payload: dict, invoke_type: str = "indirect") -> dict:
    """
    Make a request to the inference endpoint.
    Args:
        payload (dict): The input data for inference.
        invoke_type (str): The type of invocation, default is "indirect".

    Returns:
        Response from the inference endpoint.
    """
    if invoke_type == "indirect":
        endpoint = settings.inference_endpoint
    elif invoke_type == "direct":
        endpoint = settings.direct_inference_endpoint
    else:
        raise ValueError("Invalid invoke_type. Use 'indirect' or 'direct'.")

    response = requests.post(endpoint, json=payload)
    
    if response.status_code == 200:
        response_json = response.json()
        response = response_json["response"].strip().lower()
        return response
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
