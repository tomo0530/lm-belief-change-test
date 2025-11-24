# auth_dev.py — minimal local auth, no FastAPI/Supabase required
import os
from langgraph_sdk import Auth
from langgraph_sdk.auth.exceptions import HTTPException

auth = Auth()  # <-- instance required by the server


@auth.authenticate
async def authenticate(request):
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        # SDK-native exception, no FastAPI
        raise HTTPException(403, {"message": "Authorization header missing"})
    token = header.split(" ", 1)[1].strip()
    expected = os.getenv("AUTH_TOKEN", "dev")
    if token != expected:
        raise HTTPException(403, {"message": "Invalid token"})
    # What you return becomes config["configurable"]["langgraph_auth_user"]
    return {"user_id": "local-dev"}
