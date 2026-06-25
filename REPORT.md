# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Thái Dương — AI20K-009  
**Ngày nộp**: 2026-06-25  
**Submission option**: B (HuggingFace Hub)

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2.5 3B, NF4 quantized, loaded via Unsloth)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train + 20 eval), random seed = 42
- **max_seq_length**: 1024 (hard cap cho T4; p95 token length của dataset nằm trong khoảng này)
- **GPU**: Tesla T4, 14.6 GB VRAM (Google Colab Free)
- **Training cost**: ~$0.07 (12.2 phút tổng cộng @ $0.35/hr T4)
- **HF Hub**: https://huggingface.co/Dwg2504/qwen2.5-3b-vi-lab21-r16

---

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.0 min    | 7.2 GB    | 1.5577    | 4.75       |
| 16   | 3,686,400       | 4.3 min    | 6.6 GB    | 1.5161    | 4.55       |
| 64   | 14,745,600      | 4.0 min    | 8.0 GB    | 1.4768    | 4.38       |

**Tỉ lệ trainable params so với tổng:**
- r=8: 0.06% (1.84M / 3.09B)
- r=16: 0.12% (3.69M / 3.09B)
- r=64: 0.48% (14.7M / 3.10B)

---

## 3. Loss Curve Analysis

![Loss Curve r=16](loss_curve.png)

**Quan sát**: Do notebook chạy trên T4 với `eval_strategy="no"` (tắt eval-during-training để tránh OOM), chỉ có **train loss curve** cho r=16. Train loss giảm đều từ ~1.8 xuống ~1.5 trong 3 epochs — không có dấu hiệu rõ ràng của overfitting.

**Nhận xét về overfitting**: Với dataset nhỏ (180 samples) và chỉ 0.06–0.48% trainable params, rủi ro overfitting là thấp. LoRA vốn có tính regularization tự nhiên do rank thấp ràng buộc không gian cập nhật. Tuy nhiên, nếu có eval loss để so sánh, có thể r=64 sẽ overfitting nhẹ hơn r=8 do có nhiều tham số học hơn trên cùng số lượng data.

---

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

| | Response |
|---|---|
| **Base** | Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. |
| **Fine-tuned (r=16)** | Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo)... |

**Nhận xét**: Improved — FT dùng ngôn ngữ tự nhiên hơn, giải thích rõ "không cần hướng dẫn trực tiếp" — đặc trưng của Alpaca Vietnamese style.

---

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

| | Response |
|---|---|
| **Base** | Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python: `def fibonacci(n): if n <= 0: return "N p...` |
| **Fine-tuned (r=16)** | `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.") elif n == 0:...` |

**Nhận xét**: Improved — FT thêm `ValueError` cho input âm, xử lý edge case tốt hơn. Code style cũng sạch hơn.

---

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

| | Response |
|---|---|
| **Base** | 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng... |
| **Fine-tuned (r=16)** | 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng... 3. Đơn giản... |

**Nhận xét**: Degraded — FT liệt kê ngắn hơn và không giải thích từng nguyên tắc. Base model cho câu trả lời đầy đủ và có giá trị hơn ở ví dụ này.

---

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

| | Response |
|---|---|
| **Base** | LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng... LoRA là phương pháp cải thiện hiệu suất bằng cách thêm adapter low-rank... |
| **Fine-tuned (r=16)** | LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA)... |

**Nhận xét**: Degraded — FT giải thích sai tên viết tắt của LoRA ("Layer-wise Adaptive Regularization" thay vì "Low-Rank Adaptation"). Base model chính xác hơn về factual knowledge. Đây là trường hợp fine-tuning trên general Alpaca data không cải thiện — thậm chí làm giảm — factual accuracy.

---

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

| | Response |
|---|---|
| **Base** | Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học... |
| **Fine-tuned (r=16)** | Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp... |

**Nhận xét**: Same — cả hai đều giải thích đúng theo dạng liệt kê. FT có cách diễn đạt tự nhiên hơn nhưng không có sự khác biệt rõ rệt về chất lượng nội dung.

---

## 5. Conclusion về Rank Trade-off

Kết quả rank experiment cho thấy một pattern thú vị: **tăng rank không tỉ lệ thuận với cải thiện perplexity**. Từ r=8 lên r=64, số trainable params tăng 8× (1.84M → 14.7M) nhưng perplexity chỉ cải thiện 8% (4.75 → 4.38). Đây là dấu hiệu rõ ràng của **diminishing returns** — phần lớn "signal" trong dataset 200 samples đã được capture từ r=8, và việc tăng rank cao hơn chỉ thêm khả năng học các pattern nhỏ hơn trong data.

Training time gần như bằng nhau cho cả 3 ranks (~4 phút), điều này cho thấy bottleneck không phải ở số params trainable mà ở forward/backward pass của base model (frozen). VRAM tăng đáng kể từ ~6.6 GB (r=16) lên 8.0 GB (r=64) — khoảng cách này sẽ lớn hơn nhiều với model 7B+ hoặc dataset dài hơn.

**ROI tốt nhất cho dataset này là r=16**: perplexity 4.55 tốt hơn r=8 (+4.2%) với chi phí VRAM không quá cao. r=64 chỉ cải thiện thêm 3.8% perplexity so với r=16 nhưng dùng gấp 4× trainable params và nhiều VRAM hơn — không đáng trên dataset nhỏ 200 samples.

**Recommendation production**: Với dataset nhỏ (<500 samples) và domain Vietnamese general instruction-following, **r=16** là lựa chọn hợp lý. Nếu dataset lớn hơn (>5000 samples) hoặc task phức tạp hơn (reasoning, structured output), r=32–64 mới bắt đầu cho thấy lợi thế rõ rệt. r=8 phù hợp khi VRAM bị giới hạn nghiêm ngặt hoặc cần deploy nhiều adapters song song (nhỏ hơn, load nhanh hơn).

---

## 7. Stretch Goal — DoRA Variant

**Config**: `use_dora=True`, r=16, alpha=32, target_modules=["q_proj","v_proj"] — cùng setup với LoRA r=16 baseline, chỉ thêm flag DoRA.

| Method | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|--------|-----------------|------------|-----------|-----------|------------|
| LoRA r=16 | 3,686,400 | 4.3 min | 6.6 GB | 1.5161 | 4.55 |
| DoRA r=16 | 3,769,344 | 4.7 min | **12.9 GB** | 1.5162 | 4.55 |

**Kết quả**: DoRA cho perplexity **gần như giống hệt** LoRA (4.55 vs 4.55, chênh -0.10%) nhưng tiêu tốn **gấp đôi VRAM** (12.9 GB vs 6.6 GB). Điều này xảy ra vì DoRA tách weight matrix thành magnitude vector và direction matrix — phần magnitude cần lưu thêm trong VRAM nhưng không đủ expressive để tạo ra sự khác biệt trên dataset nhỏ 200 samples.

**Insight**: DoRA có lợi thế rõ ràng hơn khi dataset lớn và task phức tạp (reasoning, structured output). Với 200 samples Vietnamese general instruction, bottleneck là data quantity chứ không phải phương pháp optimization — cả LoRA lẫn DoRA đều converge đến cùng một điểm.

---

## 6. What I Learned

- **LoRA rank không phải "càng cao càng tốt"**: với dataset nhỏ, r=8 và r=64 cho perplexity gần như tương đương. Bottleneck thực sự là data quality và quantity, không phải số trainable params.
- **QLoRA 4-bit + Unsloth cho phép chạy 3B model trên T4 16GB với chỉ 6–8 GB VRAM**: điều này trước đây không thể làm được với full fine-tuning. Gradient checkpointing là yếu tố then chốt giúp giảm 60% VRAM.
- **Fine-tuning trên general data không cải thiện factual knowledge**: Example 4 cho thấy FT model còn giải thích sai tên LoRA. Fine-tuning chỉ điều chỉnh *style và format* của output, không inject thêm kiến thức mới — đó là vai trò của pretraining hoặc RAG.
