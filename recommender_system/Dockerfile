# Sử dụng base image chính thức cho FastAPI
FROM python:3.10

# Tạo thư mục làm việc
WORKDIR /app

# Copy các file cần thiết
COPY . /app

# Cài đặt các dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Expose port
EXPOSE 8000

# Chạy app bằng uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
