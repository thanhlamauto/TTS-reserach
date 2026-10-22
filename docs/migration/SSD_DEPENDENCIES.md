# SSD completion — bổ sung sau khi cắm Extreme SSD

Ngày22/10/2026, chủ dự án gắn lại SSD và yêu cầu upload nốt. Dữ liệu project
được inventory tại `/Volumes/Extreme SSD/TQK-research` (Static BFS Graph và
PointMaze), cùng4 protocol cụ thể đã được nhắc trong chat ở SSD Download.
Không upload thư mục Download không liên quan hoặc toàn bộ SSD của chủ máy.

Catalog mới trên public HF dataset LamTNguyen/TQK-research-artifacts:
`ssd_completion_20261022/SSD_CATALOG.json`. Chỉ coi backup hoàn tất khi
`status=COMPLETE`; receipt pin catalog ở `ssd_completion_20261022/CATALOG_RECEIPT.json`,
tóm tắt ở `ssd_completion_20261022/COMPLETE.json`.

Original paths được đổi sang prefix portable:

- `/Volumes/Extreme SSD/TQK-research/static-bfs-graph/...`
  → `ssd/TQK-research/static-bfs-graph/...`.
- `/Volumes/Extreme SSD/TQK-research/pointmaze_compiler_v1/...`
  → `ssd/TQK-research/pointmaze_compiler_v1/...`.
-4 protocol trên SSD Download → `ssd/handoff_inputs/...`.

Dùng restore_artifacts.py với `--prefix ssd/` hoặc một `--path ssd/...` để lấy
đúng file. Catalog SSD tự chứa receipts của payload reused, không cần tái tạo
symlink Mac hoặc tải toàn bộ workspace. Bản gốc trên SSD được giữ nguyên.

Một số kết quả được lưu sẵn trong `.tar.gz`; backup giữ nguyên byte của các gói
đó. Cần restore archive trước rồi giải nén riêng trên Windows/Colab. Nội dung
các archive gốc cũng được kiểm tra credential và có member hashes tại
`ssd_completion_20261022/ARCHIVE_MEMBER_HASHES.json`. Các file `._*` và
`.DS_Store` là metadata macOS được bỏ qua, không phải raw nghiên cứu.

Các symlink cũ trong workspace Mac chỉ là provenance; không phải điều kiện
để agent mới làm việc. Hướng dẫn SSD-vắng-mặt trước đây đã được supersede bởi
catalog completion này. Không suy ra dữ liệu trên VM TPU đã được backup chỉ từ
inventory SSD; TPU hiện không còn access và không được dùng để chạy tiếp.
