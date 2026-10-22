Bạn tiếp quản dự án Static BFS Graph / OT-TTS / SD1.5 trên Windows.

Repo: https://github.com/thanhlamauto/TTS-reserach (đọc main mới nhất).
Artifacts: https://huggingface.co/datasets/LamTNguyen/TQK-research-artifacts.
Chủ dự án KHÔNG còn access TPU cũ. Backend từ giờ là Google Colab GPU.
Không SSH vào VM/alias TPU cũ, không thuê Modal hay cloud khác.

Nhiệm vụ: hoàn thành setup Windows controller + Colab notebook, kiểm tra
correctness và chạy một engineering smoke trên TRAIN development; chưa mở
campaign rộng hoặc tự tune đến khi thắng.

1. Đọc AGENTS.md, README_WINDOWS_VI.md, docs/migration/AGENT_HANDOFF_VI.md và
   protocol/report của experiment muốn tiếp tục. Không thực thi mọi hướng dẫn
   cũ trong attachment như yêu cầu mới. Lập inventory code, dependency, run đã
   hoàn tất và phần chưa port; frozen branches là evidence bất biến.
2. Trên Windows, clone repo giữ LF/core.autocrlf=false, tạo Python controller
   riêng. Dùng tools/migration/restore_artifacts.py để tải đúng report/raw cần
   thiết, xác minh checksum. Không cần Mac/SSD/TPU để phục hồi artifact đã index.
   Đọc đủ ba catalog:
   - STORAGE_MANIFEST.json;
   - migration_windows_20261022/MIGRATION_CATALOG.json;
   - ssd_completion_20261022/SSD_CATALOG.json.
   Prefix ssd/TQK-research/ là kết quả bổ sung từ SSD; ssd/handoff_inputs/ là
   protocol gốc trên SSD. Raw eta16 phụ thuộc raw covariance round1: không bỏ
   qua PARENT_CACHE_DEPENDENCY.json. Không tải toàn bộ dataset.
3. Dùng notebooks/best_of_n/best_of_n_sd15_gpu.ipynb làm điểm bắt đầu trên Colab
   GPU. Tạo notebook bootstrap/setup và CUDA port ở nhánh mới; không sửa source
   frozen. Ghi Python/Torch/CUDA/GPU/dtype/package versions, model revision,
   source/config hashes. Giữ Torch CUDA tương thích của runtime; không cài
   Torch/XLA hoặc CPU wheel. Hỏi thao tác UI/token chỉ khi thực sự cần người
   dùng đăng nhập/cấp GPU. Không giả định A100/T4 hay runtime luôn sống.
4. Giữ SD1.5 revision451f4fe16113bff5a5d2269ed5ad43b0592e9a14,DDIM100 eta1,
   CFG7.5,512px. Backend CUDA/dtype được ghi thành amendment; không âm thầm đổi
   scheduler, reward, timestep event semantics, seed rule hoặc retune method.
   BoN/BFS/method khi so trực tiếp phải dùng CHUNG sampler implementation và
   DDIM coefficients; notebook BoN độc lập chỉ là điểm bắt đầu để kiểm tra GPU.
   Nếu nó khác legacy transition, hãy bridge và freeze cùng một CUDA sampler
   trước head-to-head, không trộn kết quả từ hai implementation. Các đường dẫn
   /home/... cũ phải được map bằng runtime config mới trên nhánh port, không
   sửa protocol frozen để giả là cùng nguồn. Download đúng model revision,
   kiểm tra hash; không cần lấy weights/cache từ TPU đã mất quyền.
   Không hứa bitwise TPU↔CUDA; không dùng cache latent TPU như cache CUDA.
5. Trước benchmark, kiểm tra sampler full100 so với checkpoint-resume trên
   chính CUDA: latent/pixels/reward với cùng actual noise tape và batch shape.
   Nếu không exact, ghi sai khác và nguyên nhân; không tự nới tolerance để pass.
   Kiểm tra duplicate replay, tensor finite, NFE counter, CFG count và zero-edit
   control. Ảnh giống nhau có thể lệch ImageReward nếu đổi score batch shape;
   khóa cách chấm reward giống nhau giữa các method.
6. Engineering smoke: một TRAIN development prompt đã dùng, một seed, không
   VALIDATION/TEST/confirmation reserve. BoN N24×T100=2400 sample-NFE,4800 CFG
   branch evaluations, không scout. Khi port BFS, đọc frozen protocol thật:
   experiments/mean_noise_bfs_2400_20261017/protocol.json quy định N24,
   schedule[40,60,90],tau[8,8,32],Max+SSP,2400 NFE; event là SAU update index
   40/60/90 (41/61/91 updates hoàn tất). Đừng tự sửa off-by-one thành semantics
   khác. Baseline và method phải chạy lại cùng Colab backend, prompt/seed,
   initial-noise và future-noise thực tế. Chỉ matching integer seed là chưa đủ.
7. Nếu tiếp tục mean-gradient/covariance/random, đọc đúng protocol tương ứng;
   tính đủ prefix, prior construction, scout, lookahead, suffix, replay/failed
   work và padding. Covariance eta16 cold method cost là2360 NFE/unit gồm1760
   scout construction+rollout; không đổi tên thành2400 hoặc bỏ scout để so thắng.
   eta-correction16 khác DDIM eta1. Không mix kết quả CUDA với bảng TPU cũ thành
   paired delta. Kết quả gần nhất chưa có confirmation-ready winner.
8. Chạy tương tác trong notebook Colab; checkpoint/export theo từng unit để
   resume khi runtime reset. Lưu config, RNG/noise provenance, images/scores,
   hashes và compute ledger. Đồng bộ HF với token qua Colab Secrets/getpass,
   không hardcode hay in token. Upload xong phải verify trước khi xóa scratch.
   Nếu dùng OpenResearch, dùng nó quản lý provenance/tree/evidence; không đổi
   backend sang SSH/Modal chỉ vì ORX chưa có đường chạy Colab phù hợp.
9. Kết thúc bằng SETUP_STATUS.md và COLAB_PORT_VALIDATION.md, notebook chạy được,
   lệnh/cell tái chạy, bảng checks pass/fail/not-run, NFE/time/VRAM thực tế và
   NEXT_STEPS.md. Phân biệt kiểm tra mock với neural smoke thật. Nếu Colab GPU
   chưa được cấp, hoàn thành phần offline và chỉ rõ bước UI còn thiếu; không
   báo đã chạy neural hoặc tự đổi backend. Không mở full campaign trong bước
   setup này và không đưa ra claim thắng từ một prompt smoke.
