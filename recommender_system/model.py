# model.py

import numpy as np
import pandas as pd
from lightfm import LightFM
from scipy import sparse
import joblib

def create_interaction_matrix(df, account_col, item_col, rating_col, norm=False, threshold=None):
    """
    Tạo interaction matrix dạng account_id - post_id với giá trị rating.
    Nếu norm=True, thì chuyển thành binary 1/0 theo threshold.
    """
    interactions = df.groupby([account_col, item_col])[rating_col] \
        .sum().unstack().reset_index().fillna(0).set_index(account_col)
    if norm:
        if threshold is None:
            raise ValueError("threshold must be provided when norm=True")
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    joblib.dump(interactions, 'interactions.pkl')
    return interactions

def create_account_dict(interactions):
    """
    Tạo dictionary: account_id → row index trong ma trận tương tác.
    """
    account_dict = {account_id: idx for idx, account_id in enumerate(interactions.index)}
    joblib.dump(account_dict, 'account_dict.pkl')
    return account_dict

def create_item_dict(df, id_col, name_col):
    """
    Tạo dictionary: item_id → item_name.
    """
    item_dict = {df.loc[i, id_col]: df.loc[i, name_col] for i in range(df.shape[0])}
    joblib.dump(item_dict, 'item_dict.pkl')
    return item_dict

def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30, n_jobs=4):
    """
    Train mô hình LightFM dựa trên ma trận tương tác.
    """
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components=n_components, loss=loss, k=k)
    model.fit(x, epochs=epoch, num_threads=n_jobs)
    joblib.dump(model, 'lightfm_model.pkl')
    return model

# def sample_recommendation_account(model, interactions, account_id, account_dict, 
#                                   item_dict, threshold=0, nrec_items=10, show=True):
#     """
#     Trả về danh sách item gợi ý cho account_id.
#     """
#     n_accounts, n_items = interactions.shape
#     if account_id not in account_dict:
#         raise ValueError(f"Account ID {account_id} not in account_dict.")

#     account_x = account_dict[account_id]
#     scores = pd.Series(model.predict(account_x, np.arange(n_items)))
#     scores.index = interactions.columns
#     scores = list(pd.Series(scores.sort_values(ascending=False).index))

#     known_items = list(interactions.loc[account_id, :][interactions.loc[account_id, :] > threshold].index)
#     known_items = sorted(known_items, key=lambda x: interactions.loc[account_id, x], reverse=True)

#     scores = [x for x in scores if x not in known_items]
#     return_score_list = scores[:nrec_items]

#     known_titles = [item_dict.get(x, str(x)) for x in known_items]
#     recommend_titles = [item_dict.get(x, str(x)) for x in return_score_list]

#     if show:
#         print("Known Likes:")
#         for idx, title in enumerate(known_titles, start=1):
#             print(f"{idx}- {title}")
#         print("\nRecommended Items:")
#         for idx, title in enumerate(recommend_titles, start=1):
#             print(f"{idx}- {title}")

#     return return_score_list

def load_model():
    model = joblib.load('lightfm_model.pkl')
    return model

def load_account_dict():
    account_dict = joblib.load('account_dict.pkl')
    return account_dict

def load_item_dict():
    item_dict = joblib.load('item_dict.pkl')
    return item_dict

def load_interactions():
    interactions = joblib.load('interactions.pkl')
    return interactions

def recommend(model, interactions, account_id, account_dict, item_dict, nrec_items=10, show=True):
    """
    Trả về danh sách post_id của top-k bài viết được gợi ý cho account_id.
    Args:
        model: LightFM model đã train
        interactions: Ma trận tương tác
        account_id: ID của account cần gợi ý
        account_dict: Dictionary mapping account_id -> index
        item_dict: Dictionary mapping post_id -> title
        nrec_items: Số lượng bài viết gợi ý
        show: Có in kết quả ra console không
        
    Returns:
        List các post_id được gợi ý
    """
    n_accounts, n_items = interactions.shape
    
    # Kiểm tra account_id có tồn tại không
    if account_id not in account_dict:
        raise ValueError(f"Account ID {account_id} not in account_dict.")
        
    # Lấy index của account trong ma trận
    account_x = account_dict[account_id]
    
    # Dự đoán điểm cho tất cả items
    scores = pd.Series(model.predict(account_x, np.arange(n_items)))
    scores.index = interactions.columns
    
    # Sắp xếp theo điểm giảm dần và lấy top-k
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    return scores[:nrec_items]