# Docker Setup - Hệ thống nhận dạng ngôn ngữ ký hiệu

Hướng dẫn sử dụng Docker để chạy hệ thống.

## Cấu trúc

- `dockerfile/dockerfile.ui`: Dockerfile cho UI service (Gradio)
- `docker-compose/docker-compose.yml`: Docker Compose configuration

## Cách sử dụng

### 1. Build và chạy với Docker Compose

```bash
cd docker/docker-compose
docker-compose up --build
```

### 2. Chạy ở background

```bash
docker-compose up -d --build
```

### 3. Xem logs

```bash
docker-compose logs -f ui
```

### 4. Dừng services

```bash
docker-compose down
```

### 5. Rebuild từ đầu

```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

## Truy cập ứng dụng

Sau khi chạy, truy cập UI tại: `http://localhost:7860`

## Development mode

Docker Compose đã được cấu hình với volume mounts để code changes được reflect ngay lập tức (hot reload).

## Health Check

Container có health check tự động kiểm tra mỗi 30 giây. Xem trạng thái:

```bash
docker-compose ps
```

## Troubleshooting

### Port đã được sử dụng

Nếu port 7860 đã được sử dụng, thay đổi trong `docker-compose.yml`:

```yaml
ports:
  - "7861:7860"  # Thay đổi port bên trái
```

### Lỗi build

Nếu gặp lỗi build, thử:

```bash
docker-compose build --no-cache --pull
```

### Xem logs chi tiết

```bash
docker-compose logs --tail=100 -f ui
```



