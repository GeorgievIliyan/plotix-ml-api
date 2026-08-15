import pickle
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()
key = os.getenv('SECURE_KEY')

MODEL_PATH = Path(__file__).parent.parent / "model" / "model.pkl"


def authorize(x_api_key: str = Header(...)) -> None:
    if x_api_key is None or x_api_key != key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
district_medians = bundle["district_medians"]
city_medians = bundle["city_medians"]
global_median = bundle["global_median"]
cat_cols = bundle["cat_cols"]
features = bundle["features"]
category_values = bundle["category_values"]

BUILDING_ERA_CATS = ["стар", "среден", "скорошен", "нов"]

ROOM_MAP = {
    "1 room": 1, "2 room": 2, "3 room": 3, "multi-room": 4,
    "house": 5, "villa": 4, "land/plot": 1, "commercial": 2, "garage/parking": 1, "other": 2
}

class PredictRequest(BaseModel):
    oblast: str
    city: str
    district: str
    type: str
    area: float
    floor: int = 1
    total_floors: int = 1
    construction: str = "тухла"
    izlozhenie: str = "юг"
    elevator: int | None = None
    access_control: int | None = None
    parking: int | None = None
    ac: int | None = None
    furnished: int | None = None
    tec: int | None = None
    act_16: int | None = None
    is_renovated: int | None = None
    is_bds: int | None = None
    is_lux: int | None = None
    seller_type: int = 1
    building_age: int = 20
    building_era: str = "среден"


@app.get("/api")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest, _=Depends(authorize)):
    if req.area <= 0:
        raise HTTPException(status_code=400, detail="area must be greater than 0")

    is_ground = 1 if req.floor == 1 else 0
    is_top = 1 if req.floor == req.total_floors else 0
    is_middle = 1 if (not is_ground and not is_top) else 0

    dist_base = district_medians.get(req.district, global_median)
    city_base = city_medians.get(req.city, global_median)
    estimated_rooms = ROOM_MAP.get(req.type, 2)

    # без негативен ефект | no negative effect
    def to_nan(v): return 1.0 if v else np.nan

    row = {
        "district_baseline": dist_base,
        "city_baseline": city_base,
        "oblast": req.oblast,
        "city": req.city,
        "district": req.district,
        "type": req.type,
        "area": req.area,
        "floor": req.floor,
        "total_floors": req.total_floors,
        "floor_ratio": req.floor / max(req.total_floors, 1),
        "sqm_per_room": req.area / estimated_rooms,
        "is_ground": is_ground,
        "is_top": is_top,
        "is_middle": is_middle,
        "construction": req.construction if req.construction in category_values["construction"] else "тухла",
        "изложение": req.izlozhenie if req.izlozhenie in category_values["изложение"] else "юг",
        "elevator":       to_nan(req.elevator),
        "access_control": to_nan(req.access_control),
        "parking":        to_nan(req.parking),
        "ac":             to_nan(req.ac),
        "furnished":      to_nan(req.furnished),
        "тец":            to_nan(req.tec),
        "act_16":         to_nan(req.act_16),
        "is_renovated":   to_nan(req.is_renovated),
        "is_bds":         to_nan(req.is_bds),
        "is_lux":         to_nan(req.is_lux),
        "seller_type": req.seller_type,
        "building_age": req.building_age,
        "building_era": req.building_era if req.building_era in BUILDING_ERA_CATS else "среден",
        "renovated_x_lux":       1.0 if (req.is_renovated and req.is_lux)      else np.nan,
        "furnished_x_renovated": 1.0 if (req.furnished    and req.is_renovated) else np.nan,
    }

    input_df = pd.DataFrame([row])
    for col in cat_cols:
        if col == "building_era":
            input_df[col] = pd.Categorical(input_df[col], categories=BUILDING_ERA_CATS, ordered=True)
        else:
            input_df[col] = pd.Categorical(input_df[col], categories=category_values[col])

    pred_log = model.predict(input_df[features])[0]
    pred_price_sqm = float(np.exp(pred_log))
    total_price = pred_price_sqm * req.area

    return {
        "price_per_sqm": round(pred_price_sqm, 2),
        "total_price": round(total_price, 2),
        "district_baseline": round(float(dist_base), 2),
        "city_baseline": round(float(city_base), 2)
    }