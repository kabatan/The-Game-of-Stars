import move_ver2
import os
import json

params_json = os.environ.get("PLAYER2_PARAMS")

if params_json:
    try:
        params = json.loads(params_json)
        move_ver2.set_parameters(params)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error setting parameters for player2: {e}")


def move2(players, stars):
    return move_ver2.move_ver2(players, stars, 1)
