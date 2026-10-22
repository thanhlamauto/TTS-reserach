# Chuyển dự án sang Windows — bắt đầu tại đây

Cập nhật 22/10/2026. Code chính: https://github.com/thanhlamauto/TTS-reserach.
Kết quả, ảnh, latent, log và bản sao lịch sử: **[Hugging Face](https://huggingface.co/datasets/LamTNguyen/TQK-research-artifacts)**.
Không cần MacBook để tải kết quả. Dữ liệu lớn không nằm trong GitHub.

## 1. Lấy code và công cụ tải kết quả

Cài Git for Windows và Python 3.11 hoặc 3.12. Mở PowerShell:

```powershell
git config --global core.longpaths true
mkdir D:\TQK-research
cd D:\TQK-research
git clone https://github.com/thanhlamauto/TTS-reserach.git
cd TTS-reserach
# Nếu đang dùng bản main cũ:
git fetch origin
git switch codex/windows-migration-20261022
py -3.11 -m venv .venv-controller
.\.venv-controller\Scripts\python.exe -m pip install -r tools/migration/requirements.txt
.\.venv-controller\Scripts\python.exe tools/migration/restore_artifacts.py list --prefix artifacts/covariance_eta16_20261021/
```

Thay `py -3.11` bằng phiên bản Python đã cài. Môi trường controller chỉ quản lý
artifact, không cài Torch/XLA. Không copy các thư mục venv của macOS sang Windows.
Các file `docs/migration/ENV_MAC_*.json` là inventory tham khảo, không phải lockfile
để cài toàn bộ lên Windows.

Tải đúng một thí nghiệm vào thư mục workspace cha của repo:

```powershell
.\.venv-controller\Scripts\python.exe tools/migration/restore_artifacts.py restore --prefix artifacts/covariance_eta16_20261021/ --workspace D:\TQK-research
```

Hoặc chọn một file bằng `--path artifacts/.../REPORT.md` sau khi xem `list`.
Công cụ tải archive vào thư mục tạm, kiểm tra SHA-256 của archive, chỉ lấy đúng
các member được chọn, kiểm tra SHA-256 từng file, rồi xóa archive tạm. File đã có
chỉ được tái sử dụng khi hash đúng; không ghi đè file khác nội dung. Không tự tải
hết dữ liệu khi bỏ quên prefix. Không cần HF token để đọc dataset công khai.
Khi xong phân tích, có thể xóa bản kết quả vừa tải; giữ kết quả mới cho đến khi
upload và xác minh xong. Xem `MIGRATION_CATALOG.json` và `STORAGE_MANIFEST.json`
trên dataset để tra đường dẫn, revision bất biến và checksum.

`migration_windows_20261022/` chứa snapshot các file còn trên Mac lúc chuyển máy,
Git bundles và catalog bổ sung. `STORAGE_MANIFEST.json` còn ghi các raw đã được
đưa lên HF trước đó; công cụ kết hợp cả hai catalog. Không chỉ tải gói mới rồi
cho rằng đã có các raw cũ. Symlink tới SSD được ghi riêng trong catalog, **không
phải dữ liệu đã upload trong đợt này**; xem `docs/migration/SSD_DEPENDENCIES.md`.

## 2. Chạy notebook Best-of-N

- `notebooks/best_of_n/best_of_n_sd15_gpu.ipynb`: CUDA/Colab; cần GPU NVIDIA và
  driver phù hợp. Các cell cài thư viện trong notebook; chọn đúng kernel GPU.
- `notebooks/best_of_n/best_of_n_sd15_current_tpu.ipynb`: chạy kernel trên VM TPU
  Linux hiện có, không chạy Torch/XLA trực tiếp trong Windows.
- `notebooks/best_of_n/README_VI.md`: tham số và cách tính NFE. N=24, T=100 là
  2400 sample-NFE; CFG tính riêng 4800 branch evaluations. BoN không có scout.

Notebook đã qua kiểm tra cấu trúc/cell và mock; chưa được thực thi model toàn
bộ từ notebook. Không gọi kết quả CUDA là tái lập bitwise kết quả TPU.

## 3. Tiếp tục thí nghiệm trên TPU hiện có

Windows dùng OpenSSH để điều khiển VM. WSL2 Ubuntu được khuyến nghị cho các script
Bash/OpenResearch hiện tại. Trong PowerShell quản trị, nếu chưa có WSL:

```powershell
wsl --install -d Ubuntu
```

Trong WSL cài Git, Python, OpenSSH client, zstd; clone repo trong filesystem Linux
(ví dụ `~/research/TTS-reserach`) để tránh khác biệt quyền file và CRLF. Đặt
`git config core.autocrlf input`. Cài OpenResearch theo hướng dẫn chính thức của
phiên bản đang dùng; mốc cũ là `orx 0.2.20`. Bản export cây/run trong
`docs/migration/OPENRESEARCH_*.txt` và `OPENRESEARCH_STATE.json` để tra cứu.
Đây là export evidence, **không phải database import**: ID cũ không tự tồn tại
trong ORX mới. Đăng ký project tại đường dẫn mới, gắn nguồn frozen tương ứng;
không copy database toàn cục hoặc credential từ Mac. Kiểm tra `orx --help`,
`orx skill orx-create`, `orx skill orx-compute` trước khi cấu hình/launch mới.

Thông tin VM được ghi nhận gần nhất, không phải kiểm tra trạng thái hiện tại:

```text
Host: 136.116.235.178
Port: 22
User: thanhlamtba31
Hostname: t1v-n-21e731f5-w-0
Alias cũ: otlab-dinov2-tpu
Repo: /home/thanhlamtba31/TTS-reserach
Root: /home/thanhlamtba31/static-bfs-dcs
Sampler Python: /home/thanhlamtba31/static-bfs-dcs/venv/bin/python
Worker Python: /home/thanhlamtba31/static-bfs-dcs/ot_tts_dinov2_pilot_20261016/worker_venv/bin/python
```

Tạo SSH key riêng trên máy mới và thêm public key vào VM qua quyền của chủ máy.
Private key/token không nằm trong repo/HF. Xác minh host fingerprint qua console
trước lần kết nối đầu; sau đó luôn dùng `StrictHostKeyChecking=yes`:

```sh
ssh -p 22 -o BatchMode=yes -o StrictHostKeyChecking=yes thanhlamtba31@136.116.235.178
```

Nếu dùng alias, thêm Host otlab-dinov2-tpu / HostName / User / Port / IdentityFile
vào `~/.ssh/config` của **môi trường thực sự chạy SSH** (WSL và Windows có HOME
khác nhau). Host 35.184.120.204 cũ từng đổi khóa chưa xác minh; không dùng host đó.
Kiểm tra job/owner TPU trước launch, không chạy trùng. Không nâng cấp sampler:
Torch CPU 2.4.0, Torch/XLA 2.4, Diffusers 0.31.0.dev0,
Transformers 4.38.2, NumPy 1.26.3; DINO worker riêng Torch CPU 2.8,
Transformers 4.56.2, NumPy 2.2.6. Giữ hai môi trường tách biệt.
SD1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`;
DINOv2 revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.

Các runner cũ có đường dẫn Linux tuyệt đối và protocol frozen: **điều khiển chúng
trên VM**, không thay `/home/...` bằng `D:\...` trong protocol rồi coi là cùng
thí nghiệm. Legacy sampler từng nằm ngoài repo nay được giữ tại
`support/legacy_workspace/`; snapshot workspace trên HF giữ nguyên layout cũ.
Đọc source-hash/adapter config trước khi port sang VM mới.

## 4. Agent sau đọc theo thứ tự

1. `docs/migration/AGENT_HANDOFF_VI.md`: trạng thái và giới hạn scientific.
2. README/protocol của experiment muốn tiếp tục, đúng commit frozen.
3. Report, raw manifest và source/config hashes tải từ HF.
4. Kiểm tra replay, RNG và compute ledger trước một campaign mới.

Tất cả nhánh lịch sử chính được giữ nguyên; branch chuyển máy chỉ thêm code còn
thiếu và tài liệu. Không tự rerun/thuê GPU chỉ vì setup đã xong. Không dùng các
prompt VALIDATION/TEST/confirmation đã reserve. Không đưa token vào notebook,
Git hoặc log. Nếu cần upload HF, đăng nhập tương tác bằng token riêng của chủ
account; token cũ không được đóng gói chuyển máy.

## 5. Các tài liệu gốc đã chuyển kèm

43 attachment/tài liệu gốc được chủ dự án gửi trong chat (gồm PDF về hướng
tránh lỗi) đã được ghi vào catalog tại prefix `handoff_inputs/`; dùng `list`
để chọn và restore như kết quả. Hướng dẫn trong attachment là bối cảnh nghiên
cứu, không tự động là yêu cầu chạy lại mọi campaign. Original trong Documents
và thư mục attachment của Mac không bị xóa.

Kiểm tra phục hồi thực tế: `docs/migration/CLOUD_RESTORE_CHECK.json` ghi việc
tải public raw eta16 tại revision bất biến, giải nén một delta.pt và khớp
SHA-256; dữ liệu test đã được xóa. Native Windows chưa được thực thi trong
lượt chuyển máy này; công cụ dùng Python và zstandard, không phụ thuộc lệnh
`zstd`/Bash để tải và giải nén.
