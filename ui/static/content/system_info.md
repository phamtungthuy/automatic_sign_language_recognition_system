## Về hệ thống

Hệ thống sử dụng kiến trúc **ConvNeXt-Tiny + Transformer** để nhận dạng ngôn ngữ ký hiệu Việt Nam.

### Thông số kỹ thuật

- **Model**: ConvNeXt-Tiny (pretrained ImageNet) + Transformer Encoder
- **Input**: Video 16 frames @ 224x224
- **Output**: 100 classes ngôn ngữ ký hiệu

### Chế độ nhận dạng

| Chế độ         | Mô tả                                |
| -------------- | ------------------------------------ |
| **Single**     | Nhận dạng 1 ký hiệu từ toàn bộ video |
| **Continuous** | Nhận dạng chuỗi ký hiệu từ video dài |

### API Endpoints

```
POST /api/v1/slr/predict       - Nhận dạng đơn
POST /api/v1/slr/predict/topk  - Top-k predictions
POST /api/v1/slr/predict/continuous - Nhận dạng chuỗi
GET  /api/v1/slr/health        - Health check
GET  /api/v1/slr/labels        - Danh sách labels
```

### Hướng dẫn

1. **Ánh sáng**: Đảm bảo đủ ánh sáng, tránh ngược sáng
2. **Vị trí**: Đặt tay trong khung hình, nền đơn giản
3. **Tốc độ**: Thực hiện ký hiệu với tốc độ bình thường
