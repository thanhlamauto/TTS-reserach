# Notebook Best-of-N

- `best_of_n_sd15_current_tpu.ipynb`: dùng đúng sampler/source hashes/seed stream trên TPU VM hiện tại. Chọn kernel venv cũ; không cài lại package. N phải chia hết4.
- `best_of_n_sd15_gpu.ipynb`: độc lập trênCUDA/Colab, có cell cài package. Model revision được khóa; không coi kết quả bitwise tương đươngTPU.

Cả hai mặc địnhN=24,T=100,CFG7.5,DDIMη1,512px. Sinh24ảnh, chấm terminalImageReward, chọn điểm cao nhất. BoN=2400sampleNFE,4800CFGbranchsampleevaluations; khôngscout/resampling. CounterUNet kiểm tra actualNFE; xuấtallPNG, bestPNG, grid, scoresCSV vàmetadataJSON. Không token được nhúng, không chạy thuê compute, không launch neural để tạo notebook.

Đổi PROMPT, BASE_SEED vàN ởcelltham số; TPUcònPROMPT_ID phải khớp thí nghiệm paired. TPUoutputs RAMdisk cầnđượcbackuptrướcVMrestart. GPUdùngseedriêngtheocandidate; khôngcoisameintegerseedlàsameTPUtrajectory.

Validation ởVALIDATION.json: kiểmJSON/cúpháp, mockNFE/seed/batch cuối vàargmax/export; chưa chạy model từ notebook này. Mã không chứa kết quả giả.
