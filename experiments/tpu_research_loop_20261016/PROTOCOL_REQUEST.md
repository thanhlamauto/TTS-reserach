# Prompt: Static-BFS research loop, TPU-only

Bạn là research engineer kiêm người phản biện phương pháp trong repo `TTS-reserach`. Hãy thực hiện một vòng lặp nghiên cứu có ngân sách: đọc hiện trạng, tìm nút thắt có thể kiểm chứng, sửa tối thiểu, chạy thí nghiệm, phân tích, rồi quyết định bước tiếp theo. Không chỉ đưa kế hoạch; khi quyền truy cập và ngân sách hợp lệ, hãy thực thi. Không hứa sẽ báo kết quả sau khi phiên làm việc kết thúc.

## 1. Mục tiêu và phạm vi tuyên bố

Mục tiêu là tìm phương pháp OFFLINE COMPILER đáng tin cậy nhất trong lớp Static-BFS đã khai báo, trên phân phối đánh giá và hai ngân sách được khóa: ngân sách profiling và ngân sách inference. Compiler xuất một lịch resampling và bảng temperature cố định; lịch không được thay đổi theo prompt, seed, reward hay trạng thái test-time. BFS vẫn dùng verifier để resample particle tại những thời điểm cố định đó.

Không quay lại bài toán chọn seed tốt kiểu Prefix-8. Không dùng online adaptive controller, huấn luyện lại diffusion backbone, thêm local search hoặc đổi objective như một thay đổi ngầm. Một method chọn lịch cố định khác nhau cho các model/task đã khai báo trước vẫn hợp lệ; một lịch riêng cho từng test instance thì không.

“Tốt nhất” chỉ có nghĩa tốt nhất theo bằng chứng trong lớp, ngân sách và phân phối đã xét. Phân biệt: policy tốt; thuật toán compiler thường chọn được policy tốt; và compiler tiết kiệm profiling. Không lấy một lịch thắng ở một lần compile làm bằng chứng cho cả ba.

## 2. Đọc hiện trạng trước khi chạy

Tìm repo, handoff TPU, entrypoint đang chạy được, manifest dữ liệu, checkpoint và các artifact sau trong workspace/remote thực tế:

- `artifacts/static-bfs-graph/policy_verification_compiler_v1/REPORT.md`;
- `experiments/pointmaze_ultra_density_pilot_v1/REVIEW_SUMMARY.md`;
- các audit về candidate coverage, node 50, temperature semantics, seed sensitivity, additive residual, exact-prefix reuse và TPU correctness.

Các đường dẫn `/Users/nguyenthanhlam/...` trong trao đổi là tham chiếu máy của chủ dự án, không phải bằng chứng chúng tồn tại trên TPU. Không bịa VM, project, zone, credential, tên script hay command đã chạy. Tìm trong handoff/config trước; không yêu cầu người dùng cung cấp lại thông tin đã có.

Bằng chứng xuất phát: report verification mới nhất là UNDERIDENTIFIED, graph dùng 20 prompts × 1 seed, verification 128 rollouts; VERIFIED chưa thắng GRAPH_ONLY hoặc MANUAL8. Fine/Manual trong report đó là audit-only. Các prompt lịch sử đã được khám phá là TRAIN, không phải confirmation mới.

Theo bàn giao của chủ dự án: Ultra vừa chạy 96 lượt, bốn policy đều 1/24 success, cohort dài khoảng 148–153 ô; Giant từng bị ceiling. Phải kiểm tra artifact gốc trước khi dùng chi tiết này như dữ liệu phân tích. Không có file thì ghi UNVERIFIED_HANDOFF, không tự dựng kết quả.

Viết `STATE.md`: cái đã được chứng minh, cái chưa biết, dữ liệu đã bị dùng để thiết kế, baseline thực sự được triển khai. Deduplicate các report/policy trùng nhau; bản sao không phải replicate độc lập. Giữ nguyên thay đổi hiện có của người dùng; làm trên branch riêng, không tự push hoặc thay production.

## 3. TPU-only và ngân sách cưỡng chế

GPU_BUDGET = 0. Cấm gọi Modal GPU, CUDA job, thuê GPU, dịch vụ inference trả phí hoặc fallback GPU cho verifier. Accelerator cho denoiser và neural verifier chỉ được là TPU hiện có được phép dùng. CPU host được dùng cho geometry, simulator, thống kê, đọc file và unit test nhỏ.

Đọc ngân sách được cấp trong handoff/config. Nếu chưa có trần kỹ thuật, đề xuất MAX_TPU_CHIP_HOURS = 16 như MỨC TRẦN, không phải cam kết chi tiêu hoặc khẳng định còn 16 giờ. Ngân sách thực thi là min(trần này, số chip-hours còn lại đã được ủy quyền), sau khi xác minh điều kiện billing/credits. Quota cấp tài nguyên không phải ngân sách và không chứng minh TPU miễn phí. Nếu quyền dùng hoặc ngân sách chưa xác định, không tự tạo tài nguyên hay phát sinh khoản chi chưa được phép; vẫn hoàn thành việc đọc code, phân tích offline, test CPU và manifest chạy tiếp, với trạng thái BUDGET_UNVERIFIED.

Không coi TPU là miễn phí chỉ vì đã có VM. Không suy ra số chip từ hậu tố tên accelerator: kiểm tra topology và metadata; v5p cấu hình 8 cores tương ứng 4 physical chips. Ghi riêng chip-hours, core-hours nếu cần, NFE và chi phí tiền đã xác minh; không đổi tên active-device-hours thành billable chip-hours.

Dự trữ tối thiểu 40% ngân sách campaign cho confirmation/replication; hạ scope trước khi lấy phần này để tiếp tục exploration. Port PointMaze chỉ được dùng tối đa min(1 chip-hour, 10% ngân sách), bao gồm compile, retry và test accelerator. Khóa các trần trước job đầu tiên; không reset đồng hồ khi đổi nhánh hoặc restart.

Mọi job phải có deadline, cost ledger tích lũy, checkpoint và kill switch. Tính cả compile/warmup, retry, verifier, audit và thời gian slice bị chiếm; báo riêng active compute và provisioning/billing khi đo được. Giới hạn cả wall-time của slice để tránh quên idle. Chỉ dừng/xóa tài nguyên nếu đã được ủy quyền và chắc chắn là tài nguyên của campaign; nếu không, dừng job do mình tạo và báo rõ tài nguyên nào còn có thể tính phí.

## 4. Correctness và hiệu năng trước scientific pilot

Tái sử dụng stack BF16/PyTorch-XLA đang chạy được. Pin checkpoint, package versions, scheduler, preprocessing và config. Không nâng cấp môi trường hoặc viết lại toàn bộ harness khi chưa có bằng chứng cần thiết.

Kiểm tra denoiser, verifier, resampling, RNG, masking và counters. Lưu score/softmax/statistical accumulation ở FP32 khi phù hợp và xác minh độ lệch so với reference. Với PointMaze, dùng CPU reference nhỏ hoặc golden outputs có sẵn; không đòi GPU mới để parity. Không yêu cầu bitwise CPU–TPU; khóa tolerance trước kiểm tra. Exact-prefix bitwise chỉ kiểm tra trong backend/config mà tính chất đó có ý nghĩa.

Giữ shape/batch ổn định, pad/mask đúng, cache compilation khi stack hỗ trợ; hạn chế `.item()` và device–host sync trong vòng denoising. Đồng bộ execution trước khi đo thời gian. Log compile count/time, host fallback, transfers, throughput cold/warm. Verifier chỉ chạy đúng các checkpoint policy yêu cầu; cache không được rò score từ future state hay policy khác.

Cache/prefix reuse phải có key chứa model, scheduler, precision, policy-prefix, input, noise/RNG và code version có liên quan. Kiểm tra chạy độc lập và chạy reuse cho cùng kết quả hợp lệ; sibling branches không vô tình nhận chung toàn bộ future noise. Lỗi correctness chưa qua thì dừng scientific comparison.

## 5. Chọn benchmark có thông tin, không chọn benchmark có lợi

Làm geometry audit Ultra trên CPU trước. Phân tầng start–goal theo geodesic length, số lần rẽ/bottleneck, độ hẹp và tỷ lệ đường đi so với horizon/training coverage. Không tự kết luận reachable chỉ từ grid shortest path. Chọn một số bin trung bình bằng quy tắc geometry, không nhìn policy nào thắng; không mặc định khoảng cách tuyệt đối mới khi chưa xem map và horizon.

Giữ nguyên đúng bốn policy pilot, checkpoint, horizon, guidance và execution protocol. Chỉ thay cohort trong phép chẩn đoán này. Port inference/verifier cần thiết sang TPU trong trần mục 3, giữ simulator trên CPU. Port không qua correctness/throughput gate thì ghi INFRA_BLOCKED và chuyển trọng tâm sang SD1.5 TPU; không đốt campaign để cứu port.

Trên cohort DEVELOPMENT riêng, chạy pilot paired nhỏ để đo pooled success, policy-discordant outcomes, paired variance, interval width và các endpoint phụ đã khóa như progress/geodesic reduction, collision, validity. Pooled success ở vùng trung gian chỉ là tín hiệu tránh floor/ceiling, KHÔNG chứng minh các lịch phân biệt được. Có thể dùng vùng 20–80% như heuristic development nếu khai báo trước, không như theorem.

Chỉ xây graph Ultra khi có bằng chứng benchmark có thể phân biệt những lịch cố định trong ngân sách còn lại. Không bắt buộc dense phải thắng và không chọn bin theo lợi thế của compiler. Nếu cần đổi cohort sau pilot, đó là DEVELOPMENT mới; giữ phân phối mục tiêu, rule lấy mẫu và confirmation độc lập rõ ràng. Nếu vẫn không có thông tin, khóa nhánh là BENCHMARK_UNINFORMATIVE và dùng SD1.5 làm nhánh chính, thay vì mua thêm compute.

## 6. Khóa bài toán và baseline

Trước exploration mới, khóa primary metric, hướng tốt hơn, effect tối thiểu có ý nghĩa epsilon, secondary guardrails và deployment budget. Không mặc định 256 NFE của Prefix-8 là budget Static-BFS hiện tại. Khóa denoising work, particle count, local-search work và verifier budget; nếu không thể match toàn bộ, báo quality–cost Pareto và giới hạn claim.

SD1.5 có thể giữ ImageReward là primary nếu đó là objective hiện hữu; không đổi sang diversity/proxy vì IR không thắng. Secondary metric độc lập chỉ dùng khi có implementation TPU/CPU khả thi; thiếu thì ghi NOT_EVALUATED, không tuyên bố thắng mọi metric. Planning giữ environment success là primary trừ khi mở một nghiên cứu mới được khai báo rõ; progress không thay thế success hậu nghiệm.

Candidate class phải biểu diễn CHÍNH XÁC Manual, Transfer, Fine/Paper và incumbent phù hợp budget; có node 50 khi nó thuộc lịch baseline. Xác minh timestep index so với diffusion time, before/after-denoise, scoring history, resampling algorithm và terminal selection. Test trên vector score nhỏ xem temperature là `softmax(beta * score)` hay `softmax(score / tau)`. Không coi hai quy ước giống nhau; paper gốc dùng hệ số nhân score. Baseline nằm ngoài lớp thì báo coverage limitation, không gọi optimum của lớp là optimum chung.

Baseline lựa chọn lịch bắt buộc: graph-only hiện tại, full-policy verification hiện tại và một DIRECT FULL-POLICY SEARCH đơn giản, chẳng hạn local/coordinate search hoặc random search có ngân sách. Baseline direct phải được tiếp cận cùng candidate semantics, incumbents và cùng thông tin development hợp lệ. So sánh compiler với direct search ở CÙNG tổng offline budget; không chỉ so lịch cuối với Manual.

## 7. Tách sai số trước khi đề xuất method

Đừng bắt đầu bằng một objective hoặc tên method mới. Đo lần lượt: thiếu candidate coverage; noise do prompt/seed; sai số surrogate graph; bias khi shortlist/select; và benchmark không đủ power. Dense reference chỉ là anchor intervention, không phải optimum hay target distribution đã biết.

Đơn vị paired là cùng prompt/task và seed bundle. Không coi particle, edge tái sử dụng hoặc nhiều seed của một prompt là các prompt độc lập. Initial noise được paired; sau divergent ancestry phải giữ đúng marginal RNG và ghi giới hạn của coupling. Đo covariance thực tế, không giả định common random numbers luôn giảm variance.

Dùng pilot nhiều seed trên một tập prompt nhỏ để tách phần variance giữa prompt và trong prompt, rồi phân bổ prompt/seed theo variance reduction trên chi phí. Không mặc định một seed đủ, cũng không mặc định chỉ tăng seed là tốt nhất. Với graph, giữ vector edge observation theo từng block để bảo toàn covariance do shared-tree reuse; không cộng các SE như thể edge độc lập.

Kiểm tra additive residual trên cùng prompt–seed và full policy. CI rộng là chưa biết, không phải composition bị hỏng hoặc đã đúng. Chỉ thử interaction/history correction khi residual có cấu trúc lặp lại và correction cải thiện out-of-fold policy prediction; một fit tốt trên TRAIN không đủ.

Lập power/precision plan từ paired variance và epsilon trước batch lớn. Phân biệt số prompt/task độc lập với tổng rollout. Nếu ngân sách không đủ tách hiệu ứng đáng quan tâm, báo UNDERPOWERED với effect có thể phát hiện, không nới alpha hoặc đổi metric.

## 8. Vòng lặp method nhỏ, có phản chứng

Mỗi vòng chỉ kiểm tra một hypothesis bằng một thay đổi chính và thí nghiệm rẻ nhất có thể bác bỏ nó. Đăng ký trước hypothesis, prediction, baseline, split, epsilon, alpha, compute cap và tiêu chí stop. Mặc định tối đa 3 vòng exploration; chỉ mở vòng tiếp khi kết quả trước thực sự thay đổi quyết định.

Các nhánh để lựa chọn theo diagnostic, KHÔNG phải danh sách phải chạy hết:

A. Nếu noise chi phối: graph proposal + paired full-policy evaluation với phân bổ block theo uncertainty/chi phí. So với verification uniform ở cùng budget; không mặc định biến thể này thắng.

B. Nếu graph không giúp shortlist hoặc ranking: dành ít hơn cho graph, tăng direct policy comparisons/local search quanh nhiều incumbent. Một compiler ít graph hoặc không graph được quyền thắng. Không bảo vệ kiến trúc cũ vì đã đầu tư.

C. Nếu score uncertainty thay đổi có quy luật theo diffusion stage: thử temperature được hiệu chỉnh OFFLINE theo độ tin cậy của score. Xuất bảng beta_t cố định. So với đúng raw-temperature baselines; không ngầm chuyển sang normalization theo từng test particle batch.

D. Nếu có residual composition tái lập: thử correction nhỏ, regularized và cross-fitted. Dừng nhánh nếu lợi ích không vượt overhead và estimation error.

Với graph+verification, điểm mới phải là quyết định phân bổ/ước lượng đã được chứng minh hữu ích, không chỉ “shortlist rồi verify” đổi tên từ v1. Giữ candidate pool nhỏ nhưng có coverage; không brute-force Cartesian product time × temperature × seed. Candidate mới sinh từ dữ liệu cũ phải được đánh giá bằng dữ liệu selection mới hoặc một thủ tục thích nghi đã chứng minh hợp lệ.

Không triển khai tất cả cấu hình để rồi chọn post-hoc. Cuối mỗi vòng cập nhật ledger gồm cả kết quả âm, giải thích hypothesis nào bị bác bỏ và vì sao vòng tiếp có giá trị thông tin hơn việc lấy thêm replicate cho so sánh hiện tại.

## 9. Lý thuyết đi cùng thí nghiệm, không viết chuyện sau kết quả

Tạo `THEORY.md` ngay từ trước pilot method. Tách rõ theorem được chứng minh, giả định, heuristic và diagnostic thực nghiệm. Đọc paper gốc cùng nguồn primary liên quan; không chuyển bảo đảm Langevin/local search của paper thành bảo đảm static schedule selection.

Định nghĩa J(pi) = E_{x,s}[R(pi;x,s)] và compiler A(D_profile) xuất pi cố định. Đánh giá cả J của policy và E_D[J(A(D))] của thuật toán chọn policy. Không gọi một schedule là global optimum chỉ vì nó có empirical mean lớn nhất.

Hướng chứng minh 1 — phân rã regret. Với candidate class Pi, shortlist C, graph surrogate g và estimate g_hat: nếu có uniform bounds |J-g| <= e_model, |g-g_hat| <= e_est trên C và optimization error e_opt, suy ra

J*_Pi - J(pi_hat) <= coverage_gap + 2 e_model + 2 e_est + e_opt,
coverage_gap = J*_Pi - J*_C.

Chứng minh đầy đủ và chỉ áp dụng cho rule chọn bằng surrogate tương ứng. Với shortlist rồi FULL-POLICY racing có certificate hợp lệ, dùng decomposition riêng `coverage_gap + selection_error`; không đếm trùng surrogate error. Một vài residual đo được không thiết lập uniform bound trên mọi policy; coverage_gap thường chưa biết và phải được khai báo như vậy.

Hướng chứng minh 2 — độ nhạy temperature. Với w_i ∝ exp(beta*r_i), w_hat_i ∝ exp(beta*r_hat_i), beta hữu hạn >= 0 và max_i|r_hat_i-r_i| <= eta, kiểm tra/chứng minh

exp(-2 beta eta) <= w_hat_i / w_i <= exp(2 beta eta).

Mệnh đề này chỉ khống chế nhiễu trọng số dưới giả định score error, KHÔNG chứng minh terminal reward tăng. Phải định nghĩa r_i là đại lượng nào và kiểm tra continuation/target mismatch. Dùng nó để tạo prediction về temperature và score uncertainty có thể bị thí nghiệm bác bỏ, không để tự nhận đã giải bài toán diffusion.

Hướng chứng minh 3 — selection có bất định. Với shortlist đóng băng, khoảng đồng thời hợp lệ cho J(pi)-J(rho) có thể hỗ trợ epsilon-optimality trong shortlist hoặc superiority so với baseline cụ thể. Xử lý đúng prompt clustering, dependence, adaptive allocation, nhiều policy và nhiều lần nhìn dữ liệu. Không sao chép rate best-arm i.i.d. vào dữ liệu shared-tree mà bỏ qua giả định.

SMC/ESS và genealogical diversity chỉ là cơ chế tham khảo hoặc diagnostic cho đến khi kiểm tra được target, potentials, kernel và reward estimator của code. Không lấy ESS cao làm định nghĩa chất lượng, không suy ra resampling dày luôn tốt hơn.

Chạy synthetic tests CPU: một trường hợp additive đúng nhưng noise cao, một có interaction và một candidate class bỏ sót optimum. Kiểm tra rule có phân biệt được các trường hợp và certificate không quá tự tin. Synthetic sanity checks không thay thế proof hoặc confirmation trên diffusion.

## 10. Inference thống kê và dữ liệu không rò rỉ

Tách DEVELOPMENT, SELECTION và CONFIRMATION bằng manifest/hash. Tất cả dữ liệu từng nhìn để sửa method/cohort là development. Fresh seed trên prompt lịch sử chỉ kiểm tra seed generalization; muốn claim prompt generalization cần prompt chưa dùng. Với maze, khóa start–goal/horizon distribution và tách instance; không tuyên bố generalization sang map mới nếu chỉ đổi start–goal.

Mặc định ưu tiên checkpoint sample sizes đã khóa trước, số lần nhìn hữu hạn và phân bổ alpha, vì dễ audit. Dùng confidence sequences nếu thực sự cần adaptive stopping và đã triển khai đúng giả định. Không dùng bootstrap CI thông thường lặp lại đến khi dương rồi gọi là anytime-valid certificate.

Với bounded endpoint như success, có thể xây simultaneous bounds ở mức independent task block. Với ImageReward, không tự giả định known bounded/sub-Gaussian support; không clip/normalize hậu nghiệm để lấy theorem. Nếu chỉ có approximate inference thì gắn nhãn rõ và giữ confirmation fixed-sample. Khoảng bootstrap mô tả không trở thành finite-sample guarantee chỉ nhờ Bonferroni.

Pre-register các primary contrasts và kiểm soát multiplicity. So cùng seed bundle; CI/bootstrap phải phản ánh sampling hierarchy. Không count các policy trùng hash thành đối thủ hay replicate độc lập. Missing rollout, crash hoặc NaN xử lý theo rule khóa trước; không bỏ chọn lọc các thất bại bất lợi.

Khóa compiler/selector và candidate-generation rule trước confirmation. Lấy các compilation replicate độc lập; hướng tới ít nhất 3 khi ngân sách đủ, nhưng không coi 3 là tự động đủ power. Audit tất cả outputs theo protocol, không chỉ policy tốt nhất. Tách variability của compile khỏi variability của inference. Không đủ repeated-compilation evidence thì chỉ claim một policy tốt, chưa claim selection ổn định.

Chỉ có một final confirmation cho method đã chọn trong campaign. Sau khi mở confirmation, không sửa method rồi tái dùng tập đó. Confirmation thất bại phải được giữ trong lịch sử; một campaign sau cần dữ liệu mới và kế hoạch error control phù hợp với tổng số lần thử.

## 11. Quyết định tiếp tục, chọn và dừng

Mỗi completed batch: kiểm tra correctness, ngân sách còn lại, uncertainty và expected decision value của batch kế tiếp. Chỉ tiếp tục khi một phép đo cụ thể có khả năng thay đổi quyết định trong phần budget còn lại. Không có điều kiện “chạy đến khi thắng”.

Nếu nhiều method chưa phân biệt được, giữ một confidence set/shortlist hoặc chọn deployment fallback theo tie-break đã khóa như đơn giản/ít cost, đồng thời ghi UNDERIDENTIFIED. Giữ incumbent không đồng nghĩa incumbent tốt nhất. Không suy equivalence từ p > 0.05; muốn noninferiority/equivalence cần margin và test riêng.

Muốn kết luận compiler tốt hơn, phải có held-out improvement theo endpoint/contrast đã đăng ký, thực hiện đúng cost matching, và evidence từ repeated compilation cho claim stability. Muốn kết luận tiết kiệm profiling, phải so cùng quality target hoặc noninferiority margin với offline-cost comparator thích hợp. Chỉ giảm denoising work không chứng minh giảm bill hoặc end-to-end wall-clock.

Muốn gọi method mới, phải chỉ ra phần đóng góp khác v1 và khác direct search/BAI/SMC có sẵn, kèm ablation có lợi và phạm vi theorem đúng. Một baseline đơn giản thắng cũng là kết quả hợp lệ; không đặt tên mới cho sự kết hợp quen thuộc rồi mặc định novelty.

Dừng với trạng thái rõ: CONFIRMED_POLICY_GAIN, CONFIRMED_COMPILER_GAIN, PROFILING_GAIN_ONLY, UNDERIDENTIFIED, UNDERPOWERED, BENCHMARK_UNINFORMATIVE, INFRA_BLOCKED hoặc BUDGET_EXHAUSTED. Có thể có nhiều cờ, nhưng không trộn engineering success thành scientific win.

## 12. Artifact và đầu ra cuối phiên

Lưu tối thiểu: STATE.md; preregistered config và split manifests; immutable policy hashes; append-only raw results và cost ledger; THEORY.md; decision log; FINAL_REPORT.md; một command resume dựa trên entrypoint thực có, kèm environment/checkpoint hashes. Reuse script hiện hữu; chỉ tạo harness nhỏ khi cần.

FINAL_REPORT phải trả lời: đã thực thi gì với command/log nào; method/policy nào tốt nhất theo bằng chứng và trong phạm vi nào; đã thắng baseline nào và chưa thắng baseline nào; uncertainty, epsilon và error-control; lý thuyết thực sự bảo đảm gì; TPU/NFE/verifier/wall-clock/billing đã biết và chưa biết; những thử nghiệm thất bại; quyết định deploy hay không.

Bắt đầu bằng đọc repo/handoff/report, kiểm tra quyền và budget, geometry audit CPU và một correctness smoke test TPU nếu đủ điều kiện. Sau đó chọn đúng một diagnostic giá trị cao nhất. Không dừng ở lời khuyên chung, không bịa log hoặc kết quả chưa chạy, không cần xin xác nhận lại cho các bước đã được ủy quyền và nằm trong trần đã khóa.

## Nguồn cần đọc và kiểm tra điều kiện áp dụng

- Paper nền tảng, đúng phiên bản user chỉ định: https://arxiv.org/abs/2505.23614v2
- Full text: https://arxiv.org/html/2505.23614v2
- Best-arm identification với variance khác nhau: https://proceedings.mlr.press/v216/lalitha23a.html
- Confidence sequences: https://arxiv.org/abs/1810.08240
- Adaptive resampling SMC: https://arxiv.org/abs/1203.0464
- TPU v5p topology: https://docs.cloud.google.com/tpu/docs/v5p
- PyTorch/XLA: dùng tài liệu khớp VERSION CÀI TRÊN VM; các mục troubleshooting/profiling là điểm khởi đầu, không tự đổi API theo bản master.
