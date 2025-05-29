from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import recommend
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging
import uvicorn
from data_loader import load_reaction_data, load_post_mappings
from prepare_data import prepare_rating_matrix
from model import create_interaction_matrix, create_account_dict, runMF, load_model, load_interactions, load_account_dict
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
interactions = None
account_dict = None
item_dict = load_post_mappings()

def train_model():
    global model, interactions, account_dict
    try:
        try:
            model = load_model()
            interactions = load_interactions()
            account_dict = load_account_dict()
            logger.info("Loaded model and data from disk.")
        except:
            logger.warning("Could not load saved model. Training from scratch...")
            df = load_reaction_data()
            df_rating = prepare_rating_matrix(df)
            interactions = create_interaction_matrix(df_rating, 'account_id', 'post_id', 'rating')
            account_dict = create_account_dict(interactions)
            model = runMF(interactions)
        
        logger.info(f"Model ready at {datetime.now()}")
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

# Chạy train model lần đầu khi khởi động app
train_model()

scheduler = BackgroundScheduler()
scheduler.add_job(train_model, 'interval', hours=1)
scheduler.start()

class RecommendRequest(BaseModel):
    account_id: int
    nrec_items: int = 10

@app.post("/recommend")
def recommender(data: RecommendRequest):
    try:
        recommendations = recommend(
            model, interactions, data.account_id, account_dict, item_dict,
            nrec_items=data.nrec_items, show=False
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"recommended_post_ids": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)