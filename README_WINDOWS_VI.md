# Windows + Google Colab — setup hiện tại

**Cập nhật22/10/2026:** chủ dự án không còn quyền truy cập TPU cũ. Từ bây giờ
Windows dùng để đọc code/quản lý kết quả; **Google Colab GPU chạy thí nghiệm**.
Không SSH vào host/alias TPU cũ, không chạy các shell runner TPU hoặc tự thuê Modal.
Hướng dẫn TPU trước đây được giữ làm lịch sử tại
`docs/migration/README_WINDOWS_TPU_ARCHIVE_20261022.md`.

## 1. Clone trên Windows

Cài Git for Windows và Python3.11/3.12, mở PowerShell:

```powershell
git config --global core.longpaths true
mkdir D:\TQK-research
cd D:\TQK-research
git -c core.autocrlf=false clone https://github.com/thanhlamauto/TTS-reserach.git
cd TTS-reserach
git config core.autocrlf false
py -3.11 -m venv .venv-controller
.\.venv-controller\Scripts\python.exe -m pip install -r tools/migration/requirements.txt
```

Nếu repo đã có, `git fetch origin` rồi đọc main mới; không ghi đè uncommitted work.
Không copy venv macOS và không cài Torch/XLA vào Windows controller. Repo giữ LF
qua `.gitattributes` để tránh đổi source hash. Chạy Colab qua trình duyệt Windows;
không cần có GPU NVIDIA trên máy Windows. WSL2 không bắt buộc cho controller.

## 2. Lấy đúng kết quả từ HF

[Public dataset](https://huggingface.co/datasets/LamTNguyen/TQK-research-artifacts).
Ba catalog giữ các giai đoạn khác nhau:

- `STORAGE_MANIFEST.json`: raw đã đưa lên HF trước migration.
- `migration_windows_20261022/MIGRATION_CATALOG.json`: workspace Mac, Git bundles,
  OpenResearch evidence,43 tài liệu gốc và dependency của các raw delta.
- `ssd_completion_20261022/SSD_CATALOG.json`: phần kết quả trên Extreme SSD được
  bổ sung sau đó, gồm Static BFS và PointMaze; index tự chứa receipt của các
  payload đã reuse từ các catalog trước.

```powershell
.\.venv-controller\Scripts\python.exe tools/migration/restore_artifacts.py list --prefix artifacts/covariance_eta16_20261021/
.\.venv-controller\Scripts\python.exe tools/migration/restore_artifacts.py list --prefix ssd/TQK-research/static-bfs-graph/n8-retuning-v1/
.\.venv-controller\Scripts\python.exe tools/migration/restore_artifacts.py restore --prefix ssd/TQK-research/static-bfs-graph/n8-retuning-v1/ --workspace D:\TQK-research
```

Dùng `--path` để lấy một file sau khi xem danh sách. Prefix `ssd/` tự chọn index
SSD, không cần tải index workspace lớn. File nén tạm và từng member đều được
kiểm tra SHA-256; chỉ member được chọn được phục hồi; file khác nội dung không
bị ghi đè. Không cần token để đọc public artifacts. Không tải toàn bộ dataset.

Một số campaign trên SSD vốn được giữ trong `.tar.gz`; chúng được backup nguyên
byte như một file. Restore `.tar.gz` đó rồi đọc/giải nén bằng Python `tarfile`
với `extractall(..., filter="data")` hoặc7-Zip vào một thư mục trống trên Windows. Không giải nén theo đường dẫn Mac.
Không có việc giải nén hàng nghìn file lên SSD trong lần upload này. File `._*`
và `.DS_Store` là metadata macOS được bỏ qua. Bản gốc SSD được giữ nguyên.
Các protocol ở SSD Download liên quan dự án nằm tại `ssd/handoff_inputs/`.

Sau khi dùng, có thể xóa bản raw vừa tải; kết quả mới phải được upload và xác
minh trước khi xóa. Credentials không nằm trong Git/HF. Khi ghi HF, lấy token
bằng Colab Secrets hoặc `getpass`, giữ trong RAM, không lưu vào cell output.

## 3. Mở Colab

Mở [Best-of-N SD1.5 GPU notebook](https://colab.research.google.com/github/thanhlamauto/TTS-reserach/blob/main/notebooks/best_of_n/best_of_n_sd15_gpu.ipynb),
lưu bản riêng vào Drive, chọn **Runtime → Change runtime type → GPU** và chạy
cell theo thứ tự. Notebook có cell cài dependencies; ghi GPU/CUDA/Torch/dtype
thực tế. Không giả định được cấp một GPU cố định. Colab có thể ngắt runtime,
nên lưu theo prompt-seed unit rồi đồng bộ HF/Drive định kỳ, không chờ cuối campaign.
Xem [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Notebook mặc định BoN N24,T100:2400 sample-NFE,4800 CFG branch evaluations,
không scout. Nó chưa được chạy neural trên Colab trong đợt migration. Agent mới
cần kiểm tra smoke thực tế và dependency compatibility trước campaign.
Nếu runtime không có CUDA, dừng neural và báo resource cụ thể; không âm thầm
chạy CPU/TPU hay thuê backend khác. Giữ Torch CUDA do runtime cung cấp nếu tương
thích; không thay bằng CPU wheel hoặc cài thêm Torch/XLA.

**Đây là backend mới.** Không kỳ vọng CUDA replay bitwise TPU hoặc dùng raw
latent TPU như cache hợp lệ cho CUDA. Giữ SD1.5 revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`,DDIM100 eta1,CFG7.5,512px; ghi protocol
amendment cho backend/dtype cần đổi. Đừng vừa đổi backend vừa retune phương pháp.
Các baseline và method phải chạy lại cùng Colab backend, cùng prompt-seed,
actual initial-noise/future-noise tensors và precision/batching đã khóa. Dùng
seed integer giống nhau không tự động tạo trajectory giống TPU.

## 4. Agent mới bắt đầu ở đâu

Copy [WINDOWS_COLAB_SETUP_PROMPT.md](WINDOWS_COLAB_SETUP_PROMPT.md) vào agent mới.
Đọc thêm `docs/migration/AGENT_HANDOFF_VI.md`, protocol/report đúng experiment và
các hash/dependency trên HF. Mục tiêu bước đầu là setup + correctness + một
engineering smoke, không mở lại mọi campaign hay tự tìm winner.

Cây OpenResearch/65 runs cũ là evidence export, không phải database import. ID
cũ không tự tồn tại trên Windows. Nếu cần tracked research, đăng ký project mới
ở đường dẫn mới và đọc help/skill phiên bản ORX cài trên máy. Không dùng SSH
backend ORX để điều khiển Colab free; làm việc qua notebook tương tác. Không
copy auth/database toàn cục từ Mac. Các automation cũ vẫn PAUSED.

GitHub giữ nhánh frozen; HF có git bundle của8 repos tại prefix
`artifact_storage/migration_windows_20261022/git_bundles/`. Restore bundle rồi
`git clone <bundle> <thu-muc-trong>` nếu cần lịch sử ngoài repo chính. Không
merge/rebase hoặc chỉnh các branch experiment đã hoàn tất để port CUDA.
