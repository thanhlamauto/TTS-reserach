# SSD không có mặt trong đợt chuyển máy

Lúc inventory ngày22/10/2026, `/Volumes` chỉ có Macintosh HD; Extreme SSD không
được gắn. Các symlink dưới đây được ghi lại, không dereference và không xóa dữ
liệu đích. Vì vậy migration không khẳng định đã sao lưu nội dung chỉ nằm trên
SSD. Mang SSD sang máy mới hoặc kiểm tra bản HF sẵn có trước khi cần campaign đó.

- `static_bfs_graph_work/results/static-bfs-graph` → toàn bộ
  `/Volumes/Extreme SSD/TQK-research/static-bfs-graph`.
- `static_bfs_graph_work/arc-modulo-gpu/results/static-bfs-graph/`
  `n8-composition-diagnostic-v1`, `adaptive-edge-racing-v1`,
  `aer-budget-allocation-v1` → các thư mục cùng tên trên SSD.

Tìm bằng hai catalog HF. Nếu không có entry cho một file trên SSD, cần upload
và xác minh riêng; không suy ra archived chỉ từ việc thấy symlink. Các bản tar.gz
trên SSD và raw trên VM không bị sửa/xóa trong lần dọn local này.
