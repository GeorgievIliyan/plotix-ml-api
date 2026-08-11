import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

MODEL_PATH = Path(__file__).parent.parent / "model" / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
district_medians = bundle["district_medians"]
city_medians = bundle["city_medians"]
global_median = bundle["global_median"]
cat_cols = bundle["cat_cols"]
features = bundle["features"]
category_values = bundle["category_values"]

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
    elevator: int = 0
    access_control: int = 0
    parking: int = 0
    ac: int = 0
    furnished: int = 0
    tec: int = 0
    act_16: int = 0
    is_renovated: int = 0
    is_bds: int = 0
    is_lux: int = 0
    seller_type: int = 1


@app.get("/api")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.area <= 0:
        raise HTTPException(status_code=400, detail="area must be greater than 0")

    is_ground = 1 if req.floor == 1 else 0
    is_top = 1 if req.floor == req.total_floors else 0
    is_middle = 1 if (not is_ground and not is_top) else 0

    dist_base = district_medians.get(req.district, global_median)
    city_base = city_medians.get(req.city, global_median)
    estimated_rooms = ROOM_MAP.get(req.type, 2)

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
        "elevator": req.elevator,
        "access_control": req.access_control,
        "parking": req.parking,
        "ac": req.ac,
        "furnished": req.furnished,
        "тец": req.tec,
        "act_16": req.act_16,
        "is_renovated": req.is_renovated,
        "is_bds": req.is_bds,
        "is_lux": req.is_lux,
        "seller_type": req.seller_type
    }

    input_df = pd.DataFrame([row])
    for col in cat_cols:
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