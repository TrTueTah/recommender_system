# prepare_data.py

import pandas as pd

def prepare_rating_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển các cột boolean thành số nguyên và tính điểm rating cho mỗi (account_id, post_id)
    Rating = viewed + liked * 2 + comments * 3
    """
    # Chuyển boolean thành int
    df['viewed'] = df['viewed'].astype(int)
    df['liked'] = df['liked'].astype(int)
    df['comments'] = df['comments'].astype(int)

    # Tính điểm tương tác
    df['rating'] = df['viewed'] + df['liked'] * 2 + df['comments'] * 3

    # Gom nhóm để cộng dồn nếu có trùng lặp
    df_rating = df.groupby(['account_id', 'post_id'], as_index=False)['rating'].sum()

    return df_rating
