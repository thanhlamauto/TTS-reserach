# Handoff nghiên cứu — 22/10/2026

Đây là chuyển máy, không phải một experiment mới. Không có neural compute nào
được launch trong migration. Lấy dữ liệu từ public dataset
`LamTNguyen/TQK-research-artifacts`; SHA/revision nằm trong hai catalog ở root
và `migration_windows_20261022/MIGRATION_CATALOG.json`.

## Những kết quả gần nhất

- Mean-gradient versus BFS, cùng 2400 NFE, 6 TRAIN prompts ×2 seeds:
  `artifacts/mean_noise_bfs_2400_20261017`. BFS hơn mean-gradient về best IR
  trung bình 0.18373; mean-gradient và allocation-uniform cùng top8. Bộ lọc gần
  như không hoạt động. Không phải TEST/confirmation.
- Covariance repair eta-correction1:
  `artifacts/covariance_repair_20261021`, frozen commit
  `f34fea3c159083e23fbe15e2d9fc00b9ecf32e80`. 48 root states, chỉ2 có direction
  active và đều thuộc prompt6; cap .25/.5 rQ chưa bind. Không nominate finalist.
- Eta-correction16:
  `artifacts/covariance_eta16_20261021`, frozen commit
  `aee278972467582b4e8b99c84328f014ddd1a39c`. Hai cap đã tạo correction khác nhau
  trên hai active parents; so matched random cùng scout và future tape.
  Whole-pool best/top8 không đổi 12/12 units. Không có bằng chứng thắng độc lập;
  chỉ một prompt có correction nonzero. Không tự sweep tiếp để tìm winner.

Raw eta16 phụ thuộc raw round1; `PARENT_CACHE_DEPENDENCY.json` ghi đầy đủ.
Raw round1: HF path `artifacts/covariance_repair_20261021/full_raw.tar.zst`,
revision `76bd6d7be11c777680a175c8b889e65c72b0318c`, SHA256
`3f5f5653a7e1b8a21472f0421a592cd55958cc19b663df126d70d4c0b2b189cd`.
Raw eta16: `artifacts/covariance_eta16_20261021/full_raw.tar.zst`, revision
`b2352570abf7ed83e5ee224e5e36d7b9ac820dc3`, SHA256
`1536c330c5504c6989d30df5da6e05bcd7705ec8b538009b592d9573f0616b5d`.

**Compute accounting:** eta16 cold budget2360 NFE/method/unit, gồm1760 scout
construction+rollout. Actual new work800 NFE gồm400 changed suffix +400 replay
(kể cả200 failed replay); cộng round1 là36720 physical NFE. README prereg của
experiment còn con số dự kiến600/36520; report/accounting sau chạy là evidence
thực tế. Không sửa lịch sử frozen để che failed replay.

## Những lỗi/tính chất cần giữ trong đầu

- ImageReward CPU có thể lệch khoảng1e-6 khi đổi batch shape dù pixels giống hệt.
  Replay eta16 đã sửa bằng giữ batch36 common/batch16 correction gốc; không nới
  tolerance. Không dùng batch1 để tuyên bố exact replay của batch36.
- DDIM stochastic eta luôn1; eta16 là strength correction, hai tham số khác nhau.
- Không xóa scout cost nhờ cache reuse khi so ngân sách với BFS.
- Worker DINOv2 và sampler Torch/XLA là hai môi trường riêng, không nâng cấp gộp.
- Không coi reference dense là optimum. Graph/local skip intervention có noise,
  mismatch full-policy và composition limitation; giữ các negative results.
- ORX project hiện đã export, không có in-flight run của project pilot tại lúc
  migration. Project khác trong ORX toàn cục không nằm trong phạm vi cleanup này.
- `OPENRESEARCH_STATE.json` chỉ gồm project evidence của workspace, không có
  chat credentials/backend auth/global database; cần đăng ký lại ORX ở máy mới.

## Những gì agent mới cần làm trước khi chạy thêm

1. Đọc yêu cầu mới của chủ dự án. Migration này không cấp một campaign mới.
2. Chọn đúng protocol/commit, restore raw hoặc report theo prefix, xác minh hash.
3. Xác minh SSH VM bằng key của máy mới, kiểm tra owner/job, disk/RAM và env.
4. Cấu hình project ORX tại đường dẫn mới; chạy exact replay gate nếu tiếp tục
   một method cũ. Không sửa branch đã trả lời hypothesis.
5. Giữ TRAIN/VALIDATION/TEST/confirmation split và frozen seed/RNG. Không dùng
   pool reserve. Không chuyển TPU→CUDA rồi so như cùng numerical backend.

Best-of-N notebooks nằm tại `notebooks/best_of_n/`. CUDA notebook là baseline
portable; TPU notebook giữ sampler hiện có và cần kernel trên VM.
Các file source chưa commit từ những campaign Static BFS cũ đã được đưa vào
branch chuyển máy; đó là bảo tồn code, không xác nhận chúng đã pass gate hay
hoàn tất. Đọc report/checklist từng campaign trong archive tương ứng.
