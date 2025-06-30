from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import recommend_collaborative_filtering
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging
import uvicorn
from data_loader import load_reaction_data, load_post_mappings, load_post_tags, load_tags
from prepare_data import prepare_rating_matrix
from model import create_interaction_matrix, create_account_dict, runMF, load_model, load_interactions, load_account_dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

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

@app.post("/recommend/collaborative-filtering")
def recommender_collaborative_filtering(data: RecommendRequest):
    try:
        recommendations = recommend_collaborative_filtering(
            model, interactions, data.account_id, account_dict, item_dict,
            nrec_items=data.nrec_items, show=False
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"recommended_post_ids": recommendations}

@app.post("/recommend/content-based-filtering")
def recommend_content_based(data: RecommendRequest):
    # Load dữ liệu từ DB
    post_tags = load_post_tags()      # post_id, tag_id
    tags = load_tags()                # id, name
    posts_df, _, _ = load_post_mappings()  # post_id, caption

    # Giữ lại các bài post đang hoạt động (ví dụ: status = 'active' nếu bạn thêm được)
    active_posts = posts_df  # Hoặc lọc theo status nếu có

    # Merge để gắn tag name vào post_tags
    merged = post_tags.merge(tags, left_on="tag_id", right_on="id")

    # Chỉ giữ các post_id hợp lệ (trong bảng posts)
    merged = merged[merged["post_id"].isin(active_posts["post_id"])]

    # One-hot encode tag_name cho từng post
    tag_df = pd.pivot_table(
        merged,
        index='post_id',
        columns='name',
        aggfunc=lambda x: 1,
        fill_value=0
    )
    tag_df.columns = tag_df.columns.get_level_values(0)  # Flatten MultiIndex nếu có

    # Tính score bằng độ dài vector
    tag_vector_df = tag_df.fillna(0)
    tag_df['score'] = np.linalg.norm(tag_vector_df.values, axis=1)

    # Chọn các post có score cao nhất
    top_posts = tag_df.sort_values('score', ascending=False).index[:data.nrec_items].tolist()

    return top_posts

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)