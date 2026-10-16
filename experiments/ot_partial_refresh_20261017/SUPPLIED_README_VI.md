# Kế hoạch thí nghiệm OT tiếp theo

**Trạng thái:** đề xuất thí nghiệm, chưa chạy model, không phải một runner đã cài vào repo TPU. `study_plan.yaml` là bản khai báo protocol; cần ánh xạ vào backend thực sự đã chạy và bổ sung các input/threshold còn null. Không tự động đọc các preset này bằng CLI của bộ mã cũ.

## Dữ kiện đầu vào

Nguồn nội bộ: `REPORT.md`, `REPORT_VI (1).md`, `REPORT_VI.md` trong cuộc trò chuyện.

- Pilot DINOv2 là một bản tái dựng độc lập từ mô tả, không khẳng định cùng implementation với gói mã được đề xuất trước đó.
- P1 OT, independent và greedy nhận cùng preview và chọn cùng endpoint trên một prompt. Chưa có bằng chứng lợi ích riêng của OT.
- P2 chưa tạo endpoint: solver không đạt tolerance. P3 và các control sau đó chưa chạy.
- Không coi ancestor count, group count, OT cost hoặc gradient preflight là bằng chứng final reward/diversity tăng.

## E0: kiểm tra trần của proposal

1. Giữ nguyên raw outputs, code snapshot và hashes.
2. Trên cache, lưu valid mask, nhãn nhóm của từng proposal, no-op reason, utility và occupancy.
3. Tính fraction source có nhiều hơn một nhóm trong các proposal hợp lệ. Nếu mọi source chỉ có một nhóm, penalty chỉ phụ thuộc group count là hằng số dưới ràng buộc một output/source.
4. Hoàn thành mười edited proposals chưa được chọn. Chi phí 460 suffix sample-NFE chỉ đúng khi xác nhận resume state ở index54; các auxiliary evaluations phải đo riêng.
5. Oracle theo source: chọn terminal tốt nhất trong ba proposal. Oracle này chỉ chẩn đoán, không phải inference algorithm.
6. So với oracle bỏ ràng buộc một output/source để thấy chi phí của việc cố giữ mọi lineage.
7. Đo preview gain/terminal gain và group label stability. Không fit cluster riêng cho preview và terminal rồi so trực tiếp các số nhãn.
8. Chấm riêng điều kiện ngữ nghĩa khó trong prompt, bao gồm các ảnh bank. Không suy từ một điểm ImageReward rằng mọi thuộc tính đều đúng.

## R1: partial refresh với semi-relaxed OT

Population24, không có reference bank ngoài, trả8 ảnh từ pool24. Event đầu tiên ở index40.

- Tính UNet prediction và predicted-clean image đúng tại trạng thái hiện tại.
- Giữ18 state có proxy reward cao nhất, thay6 state thấp nhất.
- Donor pool chung cho các control là top12.
- Tạo tối đa4 medoid của các donor; merge các medoid quá giống nhau bằng ngưỡng khóa từ development. Không ép đủ4 mode.
- Đếm survivor trong từng nhóm; phần thiếu so với quota của population là demand cho sáu slot còn trống.
- Matrix cost có hàng là nhóm nội dung cần phủ, không phải danh tính source đã bỏ.

Với demand alpha, donor reference mass u đều, reward logit ell_i và distance chuẩn hóa D_gi:

    minimize_P  sum_gi P_gi (gamma D_gi - ell_i)
                + epsilon KL(P || alpha outer u)
    subject to P 1 = alpha; target marginal is FREE.

Nghiệm:

    P_gi = alpha_g softmax_i(log(u_i) + (ell_i - gamma D_gi)/epsilon)
    donor_weight_i = sum_g P_gi

Prototype đặt epsilon=1 và tune gamma; không cần iterative Sinkhorn. Gamma=0 phải khôi phục đúng reward-only weights.

Dùng residual resampling từ donor weights để chọn sáu donor. Copy nguyên state và đúng UNet prediction rồi chạy transition gốc với noise theo CHILD SLOT. Không lấy barycenter của state.

Tuyệt đối không double-center cost: target marginal đang tự do, nên column offsets thay đổi chính bài toán chọn donor.

### Controls bắt buộc

- No steering.
- Partial refresh, reward-only donor selection.
- Partial refresh, hard group-stratified selection.
- Partial refresh, semi-relaxed OT.

Chỉ sau khi OT có tác dụng: thêm shuffled geometry và reward-only ở ESS tương đương. Trước khi claim advancement: tune lại BFS, thêm reward-only/systematic resampling và chạy hybrid12 base +12 BFS cùng tổng budget. Tài liệu arXiv2604.06779 bản v2 hiện mang tên VASR (bản v1 từng mang tên FVD); cần pin và kiểm tra đúng phiên bản/code khi đưa baseline này vào.

### Chú ý phương pháp

Đây là heuristic reward-diversity search. Nó không bảo đảm sampling đúng exponential tilt. Solver đóng form không phải novelty; phần cần chứng minh là donor allocation tốt hơn các baseline đơn giản.

## R1 bổ sung: antithetic sibling innovations

Khi hai child cùng donor, dùng xi và -xi cho năm bước đầu sau fork, sau đó dùng noise độc lập. Child lẻ dùng noise độc lập.

- Không đổi dấu latent.
- Không tăng DDIM eta hay thêm noise ngoài scheduler.
- Ghép child trước khi sinh innovation; không chọn dấu bằng reward.
- Mỗi child giữ đúng marginal Gaussian của innovation trước các selection sau đó; điều này không bảo đảm joint reward/diversity tốt hơn.
- So factorial allocation reward-only/OT x noise iid/antithetic.
- Đo pair mean, pair best và diversity của các sibling đạt chất lượng. Không suy correlation âm bằng cách trừ chính pair mean rồi tính correlation.

## R2: archive terminal-feedback, fork tại index20/40

Recipe lạnh mẫu:

- 8 full roots x100 =800.
- 16 rollouts từ index40 x60 =960.
- 8 rollouts từ index20 x80 =640.
- Tổng2400 sample-NFE,32 endpoint; chưa tính auxiliary/cpu.

Có thể chia bốn batch: 4@index20,8@index40,4@index20,8@index40. Chỉ cập nhật archive sau khi hoàn thành batch, không dùng reward của một action chưa hoàn tất để chọn chính action đó.

Mỗi node lưu prefix_hash, timestep, direct-rollout count, rewards, semantic-quality flags, terminal features và group counts. Chỉ các rollout trực tiếp bằng base kernel mới dùng như mẫu của conditional base continuation tại node. Không tự gộp adaptive/steered descendants vào cùng estimator nếu không có correction.

So bốn selector trên cùng recipe:

1. Uniform node selection.
2. Terminal-success hoặc reward UCB, không OT.
3. Group-stratified selection.
4. Semi-relaxed action OT.

Ở biến thể OT, hàng là nhóm good output còn thiếu; cột là action tiếp tục từ một node. Cost có thể dùng -log xác suất action tạo good endpoint trong nhóm đó, ước lượng từ direct rollouts với pseudo-count cố định. Đây là conditional outcome model, không chỉ DINO distance của parent.

Giữ25% action slots làm exploration như nhau cho mọi control. Nếu chưa có good endpoint nào thì không dựng các good mode giả; dùng nhánh exploration/phân bổ không có coverage.

Baseline nghiên cứu gần: Diffusion Tree Sampling. Recipe này không tự kế thừa theorem hoặc tuyên bố là bản tái lập DTS.

## R3: transport từ weak Poisson equation

Với proxy cố định R(x), muốn tăng mass theo R-R_mean thay vì khớp một bank.

Chọn test functions phi_l khả vi và đặt velocity = sum_l a_l grad(phi_l).

    G_lm = mean_i <grad phi_l(x_i), grad phi_m(x_i)>
    b_l  = mean_i (R_i - mean(R)) phi_l(x_i)
    a    = solve(G + ridge I, b)

Không dùng reward gradient. Linear basis là control quan trọng vì nó chỉ tạo constant field. Nonlinear kernels phải hơn control linear và shuffled reward.

Các sanity tests:

- Gaussian prior, linear reward: so với mean của tilted Gaussian giải tích.
- Quadratic reward: so mean/covariance giải tích.
- Four-mode target: đo mass của từng mode và failure khi initial particles thiếu mode hiếm.
- N=8,24,64; luôn tính reward calls.

Chỉ chuyển sang image khi toy và numerical gates qua. Dùng một linear projection tần số thấp có transpose rõ ràng, một correction ở index40, max actual x_t RMS0.0125 hoặc0.025. Cần tính lại UNet sau edit.

Hạn chế: exp(lambda E[r|X_t]) nói chung không bằng E[exp(lambda r)|X_t]. Proxy Poisson flow không tự là exact terminal reward-tilted diffusion. Clip/projection cũng thay phương trình weak transport; phải log residual sau update.

## P2 repair là một nhánh kỹ thuật riêng

Save cross/source-self/target-self term gây lỗi, cost/masses/epsilon/history. Giữ objective, epsilon cuối và tolerance trong repair. Remove zero-mass support một cách tương đương toán học nếu có. Test log-domain, warm-start và epsilon continuation về đúng epsilon đích. Kiểm tra envelope gradients trên chính instance lỗi và đủ hai phía của self-term.

Không giữ TPU chờ CPU giải một matrix nhỏ. Job lỗi phải ghi FAILED; các job khác vẫn có thể chạy độc lập sau gate của chính chúng. Không coi preflight VAE-DINO là kiểm tra gradient của cả Sinkhorn objective.

## Thứ tự campaign

1. E0: cache audit + missing terminal endpoints.
2. Một prompt smoke R1: kỹ thuật, không claim chất lượng.
3. E1:6 development prompts x1 prefix mới x4 methods, nominal57600 sample-NFE trước auxiliary/warmup.
4. Chỉ nếu có tín hiệu, sweep geometry strength0.3/1/3 rồi replacement fraction0.25/0.5; không Cartesian product mọi encoder/timestep.
5. E2: factorial noise trên prefix mới thứ hai.
6. R2 chỉ ưu tiên khi proxy hoặc proposal trần thấp; R3 chạy toy trước.
7. Final: khóa finalist,24 prompt chưa xem x2 prefix seeds x4 methods =460800 nominal sample-NFE. Chỉ chạy sau quyết định riêng; không bắt đầu tự động.

## Metrics và bảo toàn kết luận

Giữ output top8 by ImageReward để nối report cũ. Thêm một selector quality-diversity giống hệt cho tất cả methods; evaluator ngoài geometry tối ưu là kiểm tra bổ sung. Đo number of good-and-distinct trong8 output bằng ngưỡng chất lượng/semantic/distance khóa trước. Phân biệt diversity output thực với group/ancestor counts.

BFS mạnh hiện có chỉ là vài điểm của một trade-off, chưa phải frontier đã tune. Phải so với BFS yếu hơn và hybrid trước khi quy diversity gain cho OT.

Mọi confidence interval dùng prompt làm đơn vị, average seeds trong prompt trước. Không gộp pool của các methods hoặc trials. Không dùng oracle candidate union như inference output. Cache campaign savings không làm mất cold cost, verifier time hoặc encoder work.

## Trạng thái file

`study_plan.yaml` đã được parse và kiểm tra các phép cộng budget bằng mã. Không có model experiment mới nào được chạy trong lúc viết kế hoạch. Không có adapter TPU mới được cài hoặc kiểm thử.
