import pickle
import json
import logging
import traceback
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os
from google.cloud import firestore
from upstash_redis import Redis
from datetime import datetime, timezone

env_path = Path(__file__).resolve().parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    env_path = Path(__file__).resolve().parent / '.env.local'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("plotix_predict")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
KEY_PATH = BASE_DIR / ".secret" / "serviceAccount.json"

try:
    if os.getenv("GOOGLE_CREDENTIALS"):
        credentials = json.loads(os.environ["GOOGLE_CREDENTIALS"])
        db = firestore.Client.from_service_account_info(credentials)
    else:
        if not KEY_PATH.exists():
            logger.error(f"Service account key not found: {KEY_PATH}")
            raise FileNotFoundError(f"Service account key not found: {KEY_PATH}")
        db = firestore.Client.from_service_account_json(KEY_PATH)
except Exception as e:
    logger.error(f"Failed to initialize Firestore: {e}")
    raise

redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

if not redis_url or not redis_token:
    logger.error("Redis credentials not configured")
    raise ValueError("Redis credentials not configured")

try:
    redis = Redis(url=redis_url, token=redis_token)
except Exception as e:
    logger.error(f"Failed to initialize Redis: {e}")
    raise

app = FastAPI()

MODEL_DIR = Path(__file__).parent.parent / "model"

MODEL_PATHS = {
    "atlas": MODEL_DIR / "plotix_atlas.pkl",
    "northpearl": MODEL_DIR / "plotix_northpearl.pkl",
    "horizon": MODEL_DIR / "plotix_horizon.pkl"
}

CSV_PATH = MODEL_DIR / "listings.csv"

for name, path in MODEL_PATHS.items():
    if not path.exists():
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")

if not CSV_PATH.exists():
    logger.error(f"CSV file not found: {CSV_PATH}")
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

try:
    with open(MODEL_PATHS["atlas"], "rb") as f:
        atlas_bundle = pickle.load(f)
    with open(MODEL_PATHS["northpearl"], "rb") as f:
        northpearl_bundle = pickle.load(f)
    with open(MODEL_PATHS["horizon"], "rb") as f:
        horizon_bundle = pickle.load(f)
except Exception as e:
    logger.error(f"Failed to load pickle files: {e}")
    raise

try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    logger.error(f"Failed to read CSV: {e}")
    raise

cities_from_csv = set(
    df["city"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
)

CITY_MODELS = {
    "варна": ("NorthPearl", northpearl_bundle),
    "софия": ("Atlas", atlas_bundle),
    "пловдив": ("Atlas", atlas_bundle),
    "бургас": ("Atlas", atlas_bundle),
}

def authorize(x_api_key: str = Header(...)) -> None:
    try:
        docs = list(
            db.collection("keys")
            .where("key", "==", x_api_key)
            .stream()
        )
    except Exception as e:
        logger.error(f"Firestore query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authorization service unavailable"
        )

    if not docs:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-API-Key"
        )

    try:
        expires_at = docs[0].to_dict().get("expiresAt")
        if expires_at and expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=401,
                detail="API key has expired"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check expiration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authorization validation failed"
        )

    rate_limit_key = f"rate_limit:{x_api_key}"

    try:
        count = redis.incr(rate_limit_key)
        if count == 1:
            redis.expire(rate_limit_key, 60)
        if count > 10:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Redis rate limit failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Rate limiting service unavailable"
        )

BUILDING_ERA_CATS = [
    "стар",
    "среден",
    "скорошен",
    "нов"
]

ROOM_MAP = {
    "1 room": 1,
    "2 room": 2,
    "3 room": 3,
    "multi-room": 4,
    "house": 5,
    "villa": 4,
    "land/plot": 1,
    "commercial": 2,
    "garage/parking": 1,
    "other": 2
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
        raise HTTPException(
            status_code=400,
            detail="area must be greater than 0"
        )

    city_key = req.city.strip().lower()

    if city_key in CITY_MODELS:
        model_name, bundle = CITY_MODELS[city_key]
    elif city_key in cities_from_csv:
        model_name = "Horizon"
        bundle = horizon_bundle
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported city: {req.city}"
        )

    try:
        model = bundle.get("model")
        if model is None:
            raise ValueError("Model not found in bundle")
        
        district_medians = bundle.get("district_medians", {})
        city_medians = bundle.get("city_medians", {})
        global_median = bundle.get("global_median", 0)
        cat_cols = bundle.get("cat_cols", [])
        features = bundle.get("features", [])
        category_values = bundle.get("category_values", {})

        if not features:
            raise ValueError("Features list is empty")

        is_ground = 1 if req.floor == 1 else 0
        is_top = 1 if req.floor == req.total_floors else 0
        is_middle = 1 if (not is_ground and not is_top) else 0

        dist_base = district_medians.get(
            req.district,
            global_median
        )

        city_base = city_medians.get(
            req.city,
            global_median
        )

        estimated_rooms = ROOM_MAP.get(
            req.type,
            2
        )

        def to_nan(v):
            return 1.0 if v else np.nan

        safe_type = (
            req.type
            if "type" not in cat_cols or req.type in category_values.get("type", [])
            else "other"
        )

        construction_val = (
            req.construction
            if req.construction in category_values.get("construction", [])
            else "тухла"
        )

        izlozhenie_val = (
            req.izlozhenie
            if req.izlozhenie in category_values.get("изложение", [])
            else "юг"
        )

        building_era_val = (
            req.building_era
            if req.building_era in BUILDING_ERA_CATS
            else "среден"
        )

        row = {
            "district_baseline": dist_base,
            "city_baseline": city_base,
            "oblast": req.oblast,
            "city": req.city,
            "district": req.district,
            "type": safe_type,
            "area": req.area,
            "floor": req.floor,
            "total_floors": req.total_floors,
            "floor_ratio": req.floor / max(req.total_floors, 1),
            "sqm_per_room": req.area / max(estimated_rooms, 1),
            "is_ground": is_ground,
            "is_top": is_top,
            "is_middle": is_middle,
            "construction": construction_val,
            "изложение": izlozhenie_val,
            "elevator": to_nan(req.elevator),
            "access_control": to_nan(req.access_control),
            "parking": to_nan(req.parking),
            "ac": to_nan(req.ac),
            "furnished": to_nan(req.furnished),
            "тец": to_nan(req.tec),
            "act_16": to_nan(req.act_16),
            "is_renovated": to_nan(req.is_renovated),
            "is_bds": to_nan(req.is_bds),
            "is_lux": to_nan(req.is_lux),
            "seller_type": req.seller_type,
            "building_age": req.building_age,
            "building_era": building_era_val,
            "renovated_x_lux": (
                1.0
                if (req.is_renovated and req.is_lux)
                else np.nan
            ),
            "furnished_x_renovated": (
                1.0
                if (req.furnished and req.is_renovated)
                else np.nan
            ),
        }

        input_df = pd.DataFrame([row])

        for col in cat_cols:
            if col not in input_df.columns:
                continue
            if col == "building_era":
                input_df[col] = pd.Categorical(
                    input_df[col],
                    categories=BUILDING_ERA_CATS,
                    ordered=True
                )
            else:
                cats = category_values.get(col, [])
                if cats:
                    input_df[col] = pd.Categorical(
                        input_df[col],
                        categories=cats
                    )

        missing_features = [f for f in features if f not in input_df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        pred_log = model.predict(
            input_df[features]
        )[0]

        pred_price_sqm = float(np.exp(pred_log))
        total_price = pred_price_sqm * req.area

        return {
            "price_per_sqm": round(pred_price_sqm, 2),
            "total_price": round(total_price, 2),
            "district_baseline": round(float(dist_base), 2),
            "city_baseline": round(float(city_base), 2),
            "model": model_name
        }

    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Key error in predict: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Missing data field: {e}"
        )
    except ValueError as e:
        logger.error(f"Value error in predict: {e}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Predict failed for req=%s", req.model_dump())
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=422,
            detail=f"{type(e).__name__}: {str(e)}"
        )