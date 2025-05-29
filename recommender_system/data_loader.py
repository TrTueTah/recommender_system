# load_data.py

import pandas as pd
import psycopg2
from config import DB_CONFIG

def load_reaction_data():
    conn = psycopg2.connect(**DB_CONFIG)
    query = """
        SELECT account_id, post_id, viewed, liked, comments, to_timestamp(created_at) AS created_at
        FROM account_reaction_posts
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_post_mappings():
    """
    Tải dữ liệu bài viết từ bảng posts và tạo mapping:
    - normal_mapping: caption → post_id
    - reverse_mapping: post_id → caption
    """
    conn = psycopg2.connect(**DB_CONFIG)

    query = "SELECT id AS post_id, caption FROM posts"
    df_title = pd.read_sql_query(query, conn)

    conn.close()

    titles = df_title['caption'].tolist()
    post_ids = df_title['post_id'].tolist()

    normal_mapping = dict(zip(titles, post_ids))   # title -> post_id
    reverse_mapping = dict(zip(post_ids, titles))  # post_id -> title

    return df_title, normal_mapping, reverse_mapping

