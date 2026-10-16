# Tích hợp vào sampler cũ

## Không thay backend chỉ để chạy bộ này

Bộ mã không có source repo/runner private của bạn. Không giả định các module lịch sử tồn tại ngoài máy bạn. Dùng file API thủ công hoặc nối `SamplerAdapter`. Có adapter DDIM CPU/CUDA tham chiếu trong `otlab/adapter.py`; chưa được kiểm chứng bitwise parity với XLA cũ.

## Ba hàm cần nối

```python
advance(xt, start, end, source_ids, noise_tape) -> xt_after
predict_clean(xt, index) -> x0_hat_fp32
score(clean, tag) -> (reward_fp32, rgb_uint8)
```

`advance`:
- trạng thái tại đầu `start`, chạy indices `[start,end)` của **lịch100 gốc**;
- không `set_timesteps(end-start)`;
- dùng `noise_tape.at(index, source_ids)` trả noise FP32 cho mỗi bước;
- nhiều proposal cùng một source có source_ids trùng nhau: chúng phải nhận cùng variance noise;
- giữ CFG float32/DDIM float32/state BF16 như sampler frozen nếu đó là protocol cũ;
- model output phải được tính lại sau edit, không dùng prediction từ latent trước edit;
- step counters tính theo số samples, không số batch calls.

`predict_clean`:
- evaluate tại state hiện tại/index hiện tại;
- không consume diffusion randomness;
- ghi riêng extra UNet NFE;
- không lấy terminal reward hay terminal image của source.

`score`:
- giữ VAE/preprocessing/reward version cũ;
- trả RGB uint8 đúng ảnh được chấm; P1 encoder dùng cùng preview pixels;
- `tag=proxy`, `terminal`, `replay` để đếm verifier riêng;
- không batch-softmax hoặc normalize reward theo các methods trong batch.

`examples/callback_adapter.py` cung cấp wrapper thật. Factory backend của bạn trả các callable và `collections.Counter`. Đây là phần glue duy nhất không thể ánh xạ chính xác khi chưa có source repo.

## Quy trình file thủ công cho P2/P3

1. Xuất checkpoint bằng `examples/export_checkpoint.py`.
2. Sửa paths trong `examples/request_p2.json`; `loss=sinkhorn` cho P2, `loss=sliced` cho P3, `loss=mean` cho control.
3. Chạy worker ở env riêng:

```bash
$WORKER_PY -m otlab.worker --config configs/worker.yaml --request request_p2.json
```

4. Đọc `entry_states` rồi chạy suffix từ `next_index=50` bằng sampler cũ, cùng tape với none.
5. Lưu terminal rewards/features để đánh giá. Không đưa chúng trở lại worker selection của chính run đó.

File API không yêu cầu chuyển UNet hoặc XLA runtime sang env mới. Chỉ chuyển tensor CPU ở checkpoint.

## Quy trình P1

1. Worker `op=p1-propose` nhận checkpoint, trả `entry_states[N,3,...]`, `active[N,3]`.
2. Chỉ chạy các proposals active. Flatten các cặp `(source,proposal)`; giữ original source ID cho noise pairing.
3. Lookahead4, từ index50 tới54. Evaluate clean preview tại54, decode/chấm reward.
4. Đóng `preview` bundle:

```python
{
 "meta": checkpoint["meta"], "bank_reward": checkpoint["bank_reward"],
 "preview": {
   "states": states_at_54,   # [N,K,C,H,W]
   "clean": predicted_clean_at_54,
   "rgb": actual_scored_rgb_uint8,  # [N,K,3,H,W]
   "proxy_reward": rewards, # [N,K]
   "active": active,
   "next_index": 54
 }
}
```

5. Worker `op=p1-select` trả selected states và indices. Các cấu hình selector/lambda dùng lại cùng bundle.
6. Tiếp tục từ54, không chạy lại50..53. Dùng cùng noise tape đã khóa.

No-op luôn có preview; không dùng score cached ở50 làm no-op baseline cho candidates được chấm ở54.

## Memory

P2/P3 worker dùng VAE FP32 surrogate và tensor preprocessing có gradient. Nó không phải VAE BF16 terminal evaluator cũ; đây là phần thiết kế surrogate, cần được ghi rõ trong provenance. VAE weights/scaling vẫn phải đúng ID.

Hai-pass VJP phải chạy encoder ở eval, không dropout/augment ngẫu nhiên. Không `torch.inference_mode()` bao ngoài proximal_edit. Không dùng PIL/numpy hoặc `.detach()` trên đường clean->VAE->encoder.

P1 có thể dùng forward trên CPU; P2/P3 CPU backward có thể chậm. Không có benchmark wall time GPU/TPU của model thật trong bản phát hành này.

## Metadata/noise/replay

- `next_index` là số completed updates, không training timestep.
- `clean_prediction_index` phải bằng next_index.
- `schedule_sha` nên gồm timesteps + alphas + prediction type + eta, không chỉ hash số100.
- Lưu model revision, VAE identity, bank/prefix IDs và package freeze của sampler/worker.
- Dùng actual FP32 tape từ protocol cũ khi so với report cũ. Helper new tape là một protocol thí nghiệm mới, cần chạy lại none.
- `replay` CLI kiểm tra lặp cùng input hai lần. Nó không tự chứng minh tương đương với runner lịch sử; phải đối chiếu trusted-none bên bạn.
- Không viết đè namespace đã freeze. Runner hash file input trước khi chạy.

## Phân tán

`--shard-count K --shard-index i` chia theo prompt, giữ các methods của cùng prompt với cùng cache. Mỗi process có một worker encoder. Khởi đầu K=1. Chạy song song quá nhiều encoder worker có thể hết host RAM dù TPU còn rảnh.

Không có orchestration SSH/cloud hoặc thay đổi trạng thái máy bên ngoài trong gói này.
