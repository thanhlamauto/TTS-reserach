# OT-TTS Lab v2 — P1 / P2 / P3

Bộ mã nghiên cứu để thử **nhiều ảnh đạt chất lượng và khác nhau hơn** bằng OT trong diffusion test-time scaling.

**Phạm vi:** worker thuật toán, solver, encoder, sweep, cache, adapter contract, adapter DDIM CPU/CUDA tham chiếu, replay, paired analysis và unit tests. **Không phải bản đã triển khai trên repo/TPU riêng của bạn.** Muốn giữ benchmark SD1.5/XLA cũ cần nối ba hàm sampler theo `docs/INTEGRATION_VI.md`. Không tự nâng môi trường cũ, không tự chạy SSH, không xóa cache.

## 1. Các nhánh đã có mã

| Nhánh | Mã | Đối chứng có sẵn |
|---|---|---|
| P1 — Selective semantic edit allocation | `otlab/p1.py` | Chọn độc lập; greedy trên cùng proposals/proxy |
| P2 — Feature-space Sinkhorn surrogate | `otlab/proximal.py`, `otlab/ot.py` | Feature-mean; shuffled target |
| P3 — Feature-space weighted Sliced OT | Cùng optimizer với P2 | Feature-mean; shuffled target |
| Stage 0 — Chọn output từ pool cố định | `otlab/pool_select.py` | Top reward; greedy MMR; group-quota allocation |

P1 là **discrete capacitated transport/min-cost allocation**, không giả vờ là gradient Sinkhorn. P2/P3 là **projected clean-latent surrogate optimization**; không phải exact Doob transform hoặc bảo đảm lấy mẫu đúng reward-tilted distribution.

Target tốt/xấu: mặc định lấy top50% reference, không dùng negative repulsion. Có sweep target fraction và mức cân bằng mass theo nhóm. Đây là thay đổi target có chủ đích, không giữ nguyên exponential tilting thuần túy.

## 2. Kiểm tra mã, chưa tải model

```bash
cd ot_tts_lab_v2
# Trong môi trường đã có torch, numpy, PyYAML, Pillow, pytest:
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m pytest -q
python -m otlab.cli demo --out /tmp/otlab_demo_new
```

Demo dùng model/encoder/toy reward tổng hợp, **không phải kết quả SD1.5/FLUX**. Nó kiểm tra orchestration P1/P2/P3, cache, noise pairing và analysis. Dùng thư mục mới mỗi lần; không tự overwrite kết quả cũ.

Tình trạng kiểm thử thực tế nằm ở `TEST_STATUS.json`. Không có weights, dữ liệu private hoặc font được đóng gói.

## 3. Hai môi trường, không phá sampler đang tái lập được

### Sampler cũ

Giữ nguyên torch/XLA/Diffusers/Transformers hiện tại. Chỉ thêm source code vào `PYTHONPATH`:

```bash
export OTLAB=/absolute/path/ot_tts_lab_v2
export PYTHONPATH="$OTLAB:$OTLAB/examples:$PYTHONPATH"
export SAMPLER_PY=/absolute/path/to/frozen/venv/bin/python
```

Core không import Transformers/Diffusers khi chạy plan/solver. Không chạy `pip install -U` trong venv cũ.

### Worker encoder riêng

```bash
python3 -m venv /absolute/path/ot-worker
export WORKER_PY=/absolute/path/ot-worker/bin/python
# Cài torch + torchvision phù hợp CPU/CUDA trong venv MỚI trước.
# Không cài XLA vào worker chỉ để chạy DINO/VAE CPU.
$WORKER_PY -m pip install -r "$OTLAB/requirements-worker.txt"
```

Worker dùng Transformers 4.56.2 để có DINOv3; sampler cũ không bị nâng theo. Pin build torch/torchvision theo máy rồi lưu `pip freeze` sau preflight. Danh sách version là môi trường **dự kiến để kiểm tra**, không phải tuyên bố đã chạy pretrained end-to-end trong môi trường này.

Trước model thật, sửa paths trong `examples/request_preflight.json` rồi chạy:

```bash
$WORKER_PY -m otlab.worker --config "$OTLAB/configs/worker.yaml" \
  --request /data/request_preflight.json
```

Preflight kiểm tra input gradient qua VAE/encoder, repeatability và directional finite difference. Không suy ra chất lượng ảnh từ preflight.

Sửa `configs/worker.yaml`:

- `vae.model_id`: đường dẫn snapshot SD1.5 **đang dùng**, không tải model thứ hai chỉ để lấy VAE.
- `vae.identity`: phải giống `checkpoint.meta.vae_id`.
- `encoder.model_id`: DINOv3 ViT-B/16; được phép dùng đường dẫn local.
- `encoder.revision`: khóa commit sau khi tải/cấp quyền hợp lệ.
- `local_files_only: true` là mặc định. DINOv3 có điều kiện truy cập/license; cần cấp quyền/cache hợp lệ trước.
- `device: cpu` để thử nhỏ; có CUDA riêng thì đổi sang `cuda`. **Không coi P2/P3 CPU là cấu hình sweep nhanh**: VAE/encoder backward có thể chi phối thời gian.
- `feature_microbatch: 1` giới hạn activation memory.

P1 cần forward encoder. P2/P3 cần backward qua VAE + encoder nhưng **không qua UNet**. Sliced OT giảm chi phí solver, không loại chi phí encoder/VAE backward. Chạy một worker trước; không nhân bốn worker chỉ vì có bốn TPU chip nếu host RAM/CPU không đủ.

## 4. Dữ liệu đầu vào và chống leakage

Xem hàm xuất thật trong `examples/export_checkpoint.py`.

Mỗi checkpoint `.pt` gồm:

```python
{
  "schema": "otlab.checkpoint.v2",
  "meta": {...},
  "source_xt": source_xt,         # [N,C,H,W], đúng dtype đưa vào sampler
  "source_clean": source_x0hat,   # [N,C,H,W], FP32, dự đoán tại next_index
  "bank_clean": bank_x0hat,       # [M,C,H,W], cùng prompt/checkpoint
  "bank_reward": bank_rewards,   # [M], terminal rewards của REFERENCE được phép dùng
  "signal_scale": a_t            # scalar của VP: sqrt(alpha_bar[timestep[next_index]])
}
```

`source_clean` và `bank_clean` phải được dự đoán ở **cùng state/noise level hiện tại**, không lấy predicted clean từ model evaluation bước trước. Metadata có `clean_prediction_index == next_index` để bắt lỗi index thông thường. Không thể kiểm chứng tính đúng của tensor chỉ bằng tên trường; exporter chịu trách nhiệm.

**Không đưa source terminal reward/image hoặc none reward vào input steering.** Validator từ chối các trường phổ biến này. Terminal bank rewards được phép dùng theo recipe bank16+source8.

Noise tape phải là FP32 thực, gồm `[steps,N,C,H,W]`, `start_index`, `source_ids`, `schedule_sha`. Mọi proposal của source i dùng cùng tape với no-op của source i. Không đưa method/proposal ID vào khóa noise. Có helper tạo tape **mới** trong `otlab/noise.py`, không khẳng định giống RNG lịch sử.

Nếu chưa có các tensor predicted-clean ở index40/60, **không thể sweep timing bằng cách sửa số index**. Cần xuất checkpoint đúng tương ứng; planner sẽ từ chối dữ liệu thiếu.

Mẫu manifest ở `examples/manifest.example.jsonl`. Path có thể tương đối với thư mục manifest; planner chuyển thành absolute path và khóa SHA.

## 5. Cách hoạt động từng nhánh

### P1

1. DINO encode predicted-clean images source/reference tại cùng checkpoint.
2. Mỗi source có tối đa ba lựa chọn: no-op, edit theo reference tốt gần nhất, edit theo reference tốt khác nhóm.
3. Edit ở subspace clean-latent tần số thấp, áp trust bound vào delta **x_t thực**.
4. Chạy lookahead cùng noise; chấm reward của chính edited state, không lấy reward reference làm gain.
5. Phân nhóm **preview sau edit**. Loại edits có proxy reward thấp hơn no-op; guard SigLIP tùy chọn, mặc định off để không bịa tolerance.
6. Giải hard allocation có soft group capacity, hoặc independent/greedy controls.
7. Tiếp tục từ selected lookahead states tại `next_index + lookahead`, không denoise lại prefix và không trung bình các proposals.

P1 objective:

`sum(selected standardized proxy gain) - lambda * sum_g max(occupancy_g-capacity_g, 0)`.

No-op luôn feasible, cũng được tính trong occupancy. DP giải đúng bài toán nhỏ; không có entropy cần tune. `lambda=0` tương đương independent max utility khi cùng filters. Soft quota không bảo đảm diversity đầu ra; cluster và preview reward đều là proxy.

### P2

`u0 = stopgrad(predicted_clean)`; optimize `u0 + lowpass_edit` thông qua VAE + DINO với Sinkhorn divergence.

- Full KL-regularized OT value; không chỉ dùng `<P,C>` rồi bỏ entropy khi đánh giá objective.
- Plan được giải FP64 log-domain, gradient cost lấy bằng envelope theorem.
- Self-cost giữ **cả hai** đường gradient; không detach một phía, không mask diagonal.
- Epsilon = kappa * std(double-centered cross-cost), tính **một lần**, detach khỏi graph. Centering chỉ để lấy scale, không thay cost thật.
- Finite empirical Sinkhorn self-batch là đúng với objective đang tối ưu, không phải estimator population W-Flow độc lập hai batch.
- Two-pass VJP: tính gradient theo features rồi recompute VAE/encoder từng microbatch. Không giữ graph UNet hoặc graph của cả batch ảnh.
- Projected normalized descent hai bước, không khẳng định là tích phân WGF chính xác hoặc loss luôn giảm.
- Trả `x_t_new = x_t + a_t*(u-u0)`, cast và đo lại RMS thực.

Control `feature_mean` dùng cùng feature encoder, target, số bước, subspace và trust region. Khác nhau giữa latent gradients không tự chứng minh source-specific OT vì encoder Jacobian đã có thể tạo khác biệt.

### P3

Giữ nguyên P2 nhưng objective là trung bình weighted W2 một chiều qua 32/64 projections cố định. Solver xử lý unequal masses và unequal atom counts bằng overlap quantile intervals, **không** sort hai batch rồi trừ theo index.

Không có epsilon; vẫn có projection count/seed, target và trust bound. Seed không tối ưu theo reward.

## 6. Sweep theo tầng

### Stage 0 — chỉ chọn output từ pool đã có

Input `pool.pt`: terminal reward, DINO features, tùy chọn DreamSim/PickScore/fidelity. Không dùng intermediate features làm terminal diversity.

```bash
$SAMPLER_PY -m otlab.pool_select \
  --input /data/calibration/pool.pt --out /runs/stage0_selectors.json
```

So top8 theo reward, greedy MMR và group-quota selection ở cùng pool. Đây là selector-only baseline, không phải advancement của generation.

### Stage 1 — smoke có control

```bash
$SAMPLER_PY -m otlab.cli plan \
  --manifest /data/manifest_dev.jsonl --defaults "$OTLAB/configs/defaults.yaml" \
  --preset smoke --limit-prompts 1 --replicates 0 --out /runs/ot_v2_smoke_plan
```

Preset smoke có **9 cấu hình**: none; P1 OT/independent/greedy; P2/feature-mean/P2-shuffle; P3/P3-shuffle. Chạy một prompt trước để kiểm tra pipeline, không chọn thắng thua từ một prompt.

### Replay trước khi sweep

```bash
$SAMPLER_PY -m otlab.cli replay \
  --jobs /runs/ot_v2_smoke_plan/jobs.jsonl \
  --adapter your_frozen_module:factory --adapter-config /data/frozen_adapter.yaml \
  --out /runs/ot_v2_replay.json
```

Factory cần map ba hàm: `advance`, `predict_clean`, `score`, và cung cấp counters. Xem `examples/callback_adapter.py` và `docs/INTEGRATION_VI.md`. Có adapter CPU/CUDA tham chiếu: `otlab.adapter:diffusers_factory`; không dùng nó thay thế XLA cũ rồi tuyên bố bitwise same benchmark.

Duplicate replay là điều kiện cần. Trước campaign thật còn đối chiếu none từ exporter với output trusted của sampler cũ trên cùng tape.

### Chạy jobs

```bash
$SAMPLER_PY -m otlab.cli run \
  --jobs /runs/ot_v2_smoke_plan/jobs.jsonl \
  --adapter your_frozen_module:factory --adapter-config /data/frozen_adapter.yaml \
  --worker-python "$WORKER_PY" --worker-config "$OTLAB/configs/worker.yaml" \
  --replay-proof /runs/ot_v2_replay.json \
  --out /runs/ot_v2_smoke --cache /runs/ot_v2_cache
```

`your_frozen_module:factory` là chỗ nối repo của bạn, không phải module được bịa là đã tồn tại. Bộ này không có source runner private nên không thể tự hoàn thành ánh xạ đó. Worker/solver/runner orchestration đã được cung cấp.

Feature worker là process lâu sống để không load DINO/VAE cho từng job. Cache khóa checkpoint SHA, noise SHA, config, worker config, execution identity và mã package. Job result dùng namespace immutable. Đổi model/revision/backend code phải dùng namespace mới và lưu provenance backend.

### Stage 2 — sweep riêng từng nhánh

```bash
for P in p1 p2 p3; do
  $SAMPLER_PY -m otlab.cli plan \
    --manifest /data/manifest_dev.jsonl --defaults "$OTLAB/configs/defaults.yaml" \
    --preset "$P" --limit-prompts 6 --replicates 0,1 --out "/runs/${P}_plan"
done
```

| Preset | Tham số | Số cấu hình gồm controls + none |
|---|---|---:|
| p1 | bound .025/.05; OT & greedy lambda .1/.3; independent lambda0 | 11 |
| p2 | bound .025/.05; kappa .25/1/4; feature-mean & shuffled | 11 |
| p3 | bound .025/.05; slices32/64; feature-mean & shuffled32 | 9 |

Sáu prompt, hai suffix replicates: P1/P2 mỗi plan132 jobs, P3 108 jobs. **Không cần chạy cả ba plan ngay.** Chọn một nhánh có tín hiệu ở smoke/dev nhỏ rồi mở plan của nhánh đó.

P1: selector/lambda dùng lại cùng proposals + lookahead cache. Những selected states hoàn toàn giống nhau còn tái sử dụng suffix cache. Không miễn phí các cache khi tính cold cost của từng phương pháp.

### Stage 3 — chỉ mở thêm trục sau khi khóa finalist

Xuất `winners.json` là list các `config` đã chọn từ `plan.json`, khóa trên development.

```bash
for P in geometry timing target lookahead; do
  $SAMPLER_PY -m otlab.cli plan \
    --manifest /data/manifest_dev_all_checkpoints.jsonl \
    --preset "$P" --winners /runs/locked_winners.json \
    --out "/runs/${P}_plan"
done
```

- geometry: DINO global vs 25% spatial 2x2 features.
- timing: completed updates40/50/60; phải có checkpoint đúng.
- target: P1 top50%/75%; P2/P3 group-mass balance alpha0/.5.
- lookahead: P1 2/4/8 bước liên tiếp lịch cũ.

Không sweep mọi trục cùng lúc. Không auto chọn cấu hình bằng held-out reward. Encoder checkpoint/seed/quality thresholds phải khóa trước held-out.

## 7. Đo và so sánh

```bash
$SAMPLER_PY -m otlab.analyze \
  --root /runs/ot_v2_smoke --out /runs/ot_v2_summary --bootstrap 20000 \
  --require-plan /runs/ot_v2_smoke_plan/jobs.jsonl
```

Outputs:
- `per_run.csv`: paired source gain, p10 loss, pool best/top8 gain, contribution vào output, DINO diversity suffix, cost.
- `summary.json`: trung bình bank/prefix/replicate trong từng prompt trước khi bootstrap prompt blocks.
- `paired_contrasts.json`: direct method-vs-method contrasts chỉ khi các blocks ghép đúng. Đây là all-pairs exploratory output, không multiplicity-adjusted.

Không gộp samples giữa methods, giữa sweep trials hoặc giữa suffix replicates để chọn best. Bank thắng ties để ổn định selection. Các threshold `N_good_distinct` phải chọn trên calibration độc lập, không điều chỉnh để một method thắng.

### Independent evaluation

```bash
$SAMPLER_PY -m otlab.evaluate manifest \
  --root /runs/ot_v2_smoke --out /runs/source_images.jsonl
# Bổ sung các ảnh bank TERMINAL bằng manifest có role=bank và bank_index đúng 0..M-1.
cat /runs/source_images.jsonl /data/bank_images.jsonl > /runs/all_eval_images.jsonl

# Chạy trong worker hoặc evaluation venv đã có dreamsim/PickScore/SigLIP và đủ storage.
$WORKER_PY -m otlab.evaluate score \
  --manifest /runs/all_eval_images.jsonl --out /runs/eval.jsonl \
  --models dreamsim,pickscore,siglip --device cpu --allow-download

$SAMPLER_PY -m otlab.analyze \
  --root /runs/ot_v2_smoke --out /runs/ot_v2_cross_eval \
  --external-eval /runs/eval.jsonl --thresholds /data/frozen_thresholds.json
```

DreamSim có dependencies bổ sung; cài trong evaluation venv riêng nếu xung đột với worker. `--allow-download` là opt-in rõ ràng; DreamSim loader có thể tải weights. Không dùng flag này trước khi kiểm tra quyền model/cache/disk. PickScore dùng raw scaled cosine, không batch-softmax; điểm không phụ thuộc batch chứa phương pháp nào.

Threshold file ví dụ schema, **không có số mặc định đã được xác nhận**:

```json
{"reward_field":"pickscore","reward_threshold":NUMBER_FROM_CALIBRATION,
 "dreamsim_distance_threshold":NUMBER_FROM_CALIBRATION,
 "fidelity_threshold":NUMBER_FROM_CALIBRATION_OR_NULL}
```

Không copy nguyên placeholder này thành JSON. Điền số được khóa từ calibration. Có thể đặt reward_field=image_reward để nối metric cũ, nhưng phải phân biệt với metric ngoài reward được tối ưu.

`N_good_distinct` được tính chính xác trên tám output bằng duyệt subsets. Pool-top8 diversity/good-distinct **chỉ được xuất khi có terminal bank embeddings**. Thiếu dữ liệu thì metric bị bỏ trống, không tráo bằng bank predicted-clean feature.

Giữ CLIP metric cũ trong evaluator hiện tại để nối report trước. Kit không tự gọi cosine DINO/DreamSim là “CLIP diversity” lịch sử.

## 8. Cost accounting

Recipe cũ bank16/source8/T100:2400 UNet sample-steps. P1 tối thiểu:

`2400 + N*(K-1)*lookahead + B_aux`.

Với N8,K3,lookahead4:2464+B_aux. Runner tham chiếu đánh giá clean preview riêng (24 model calls), không tái dùng prediction này cho bước sau. Nếu source-clean ban đầu tốn thêm8 và bank-clean đã cache trong full trajectories, cold count là2496, không phải2400. Nếu có proposal inactive, count giảm theo số proposal thực.

`cold_initial_aux_nfe` phải phản ánh recipe thật. Không biết thì để thiếu: runner trả cold_nfe=null và lower bound, không bịa exact budget.

P2/P3 không thêm UNet backward nhưng thêm decoder/encoder forward/backward. Ghi actual sample counters, verifier calls, worker operations, wall time. Cache campaign không chứng minh cold latency. Đối chiếu BoN ở đường cong compute gần đúng chỉ là phân tích phụ; không gọi cost-matched nếu chưa tính verifier/feature compute.

## 9. Điều kiện dừng / đi tiếp

- P1 phải hơn independent/greedy trên **cùng candidate pool**, không chỉ hơn none.
- P2/P3 phải hơn feature-mean và shuffled controls.
- Source gain phải chuyển thành nhiều output đáng giữ; chỉ tăng reward của ảnh bị loại là chưa đủ.
- Tăng diversity phải giữ quality/prompt fidelity; encoder distance cao không tự chứng minh đa dạng tốt.
- Finalist kiểm tra trên prompt mới + bank/source prefix mới. Thêm suffix noise chỉ đo độ ổn định conditional on bank cũ.
- Solver không hội tụ, gradient đứt, noise pairing sai, clean index lệch hoặc replay fail: sửa implementation trước khi đọc reward.

## 10. Tài liệu và giới hạn

`docs/MATH_AND_AUDIT.md`: mục tiêu, gradients, kiểm thử.
`docs/INTEGRATION_VI.md`: ba hooks sampler, file workflow, metadata và CPU/CUDA reference adapter.
`docs/SWEEP_PLAYBOOK.md`: lệnh chạy và quyết định tuning.
`docs/SOURCES.md`: nguồn primary cho encoder/APIs/OT. Các proposal kết hợp ở đây là thiết kế thử nghiệm của bộ mã, không phải claim kết quả của những paper đó.

Bản hiện tại ưu tiên **SD1.5 VP/DDIM** theo báo cáo của bạn. Không được dùng công thức VP hoặc adapter DDIM cho raw FLUX states. Muốn chuyển FLUX cần adapter flow-matching và kiểm chứng mapping clean correction riêng.
