# SSD completion và đổi backend

Ngày22/10/2026, đã bổ sung vào public HF dataset
[LamTNguyen/TQK-research-artifacts](https://huggingface.co/datasets/LamTNguyen/TQK-research-artifacts):

-41,117 file trong TQK-research trên SSD, cùng4 protocol đã được nhắc ở Download.
-Tổng2,205,314,387bytes (~2.054GiB); không tính metadata `._*`/`.DS_Store`.
-40,557 unique payload mới;564 file dùng payload đã có hoặc trùng trong SSD.
-11 archive mới, tổng503,993,156bytes (~480.65MiB), đã verify member hashes,
  metadata immutable và full HTTP download SHA256. Không upload lại payload trùng.
-Catalog SSD tự chứa receipts; trạng thái COMPLETE,0 credential findings.
-Không xóa, sửa hay giải nén thành nhiều file trên SSD. Không truy cập TPU.

Raw catalog: `ssd_completion_20261022/SSD_CATALOG.json`.
Receipt: `ssd_completion_20261022/CATALOG_RECEIPT.json`.
Catalog revision: `23d7c8ae7efeb1cd7fa11c7d2c23102b069ffc27`.
Catalog SHA256: `23ae284fe7d9b4ac3378b298e5b77d2d70b8404fe51969c194d524ac895044f3`.
Các gzip archive cũ cũng được scan bên trong và lưu hash từng member.

`SSD_CLOUD_RESTORE_CHECK.json` ghi lần tải public hai file alias trùng payload,
phục hồi đủ cả hai tên/path và khớp checksum. Bộ test mới còn kiểm tra index
SSD độc lập, chặn catalog incomplete, không ghi đè và dọn scratch khi hash lỗi.
Không thực thi Windows native hoặc neural trên Colab trong lần backup này.

Chủ dự án xác nhận không còn access TPU. Backend mới: **Google Colab GPU**;
Windows quản lý code/artifact, không cần GPU local. Readme chính và prompt
`WINDOWS_COLAB_SETUP_PROMPT.md` đã chuyển theo lựa chọn này. Mọi comparison mới
phải thiết lập baseline/method trên cùng CUDA sampler; không trộn kết quả TPU
thành paired evidence với Colab. Các source/protocol frozen vẫn nguyên.
