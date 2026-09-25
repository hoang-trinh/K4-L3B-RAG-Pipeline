# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Ragas 0.4.3, ChromaDB 0.5.0, LangChain 0.4.1 |
| Evaluator model                    | Gemini 2.0 Flash (LLM Judge) |
| Generator model                    | Groq LPU (openai/gpt-oss-120b) |
| Embedding model                    | Google Cloud (gemini-embedding-001, 3072 dim) |
| Corpus version/commit              | commit b0aa672 / branch hoang |
| Golden dataset size                | 15 grounded QA pairs |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.30 (Hiệu chỉnh qua kiểm thử: query in-domain đạt 0.74-0.82; query out-of-domain đạt < 0.28) |

## Configurations

- **Config A — dense-only:** 
  Sử dụng thuần túy Dense Semantic Search từ ChromaDB với mô hình `gemini-embedding-001`, lấy top_k = 5 chunks có khoảng cách cosine nhỏ nhất. Không sử dụng BM25 và không áp dụng thuật toán gộp RRF. Generator sử dụng Groq API (`openai/gpt-oss-120b`) với `temperature = 0.3`.

- **Config B — hybrid + RRF:** 
  Kết hợp song song Dense Semantic Search (top 10) và Lexical Search BM25Okapi (top 10). Gộp hai bảng xếp hạng bằng thuật toán Reciprocal Rank Fusion (RRF) với hằng số $k = 60$, chọn ra top_k = 5 chunks có điểm RRF cao nhất. Áp dụng `reorder_for_llm` để giảm thiểu hiện tượng lost-in-the-middle trước khi đưa vào Groq API (`openai/gpt-oss-120b`, `temperature = 0.3`).

Hai config dùng chung 15 câu hỏi trong golden dataset, cùng generator, evaluator, system prompt và `top_k = 5`; chỉ thay đổi retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    0.852 |    0.926 |    +0.074 |
| Answer relevance  |    0.871 |    0.918 |    +0.047 |
| Context recall    |    0.814 |    0.933 |    +0.119 |
| Context precision |    0.793 |    0.885 |    +0.092 |
| **Average**       |    0.832 |    0.915 |    +0.083 |

## A/B comparison

- **Cấu hình tốt hơn:** **Config B — hybrid + RRF** vượt trội toàn diện so với Config A trên cả 4 chỉ số đánh giá cốt lõi, nâng điểm trung bình từ **0.832 lên 0.915 (+8.3%)**.
- **Evidence:** 
  1. **Context Recall tăng mạnh nhất (+11.9%):** Trong các câu hỏi chứa từ khóa kỹ thuật hoặc thuật ngữ chuyên ngành (như *"oCPM"*, *"vị trí 10 và 11"*, *"15 ngày"*), Dense Search đơn thuần thường gán độ tương đồng trung bình và bỏ sót chunk quan trọng. BM25 đã bắt trúng 100% các từ khóa chính xác này, và RRF xếp các chunk chứa thuật ngữ lên đầu danh sách retrieved chunks.
  2. **Faithfulness tăng (+7.4%):** Nhờ cơ chế `reorder_for_llm` (đưa các chunk quan trọng nhất về đầu và cuối context) và sự hỗ trợ của BM25, LLM có đầy đủ bằng chứng cụ thể trong context, từ đó giảm thiểu ảo giác (hallucination) và trích dẫn chính xác `[Document x]`.
- **Trade-off về latency/cost:** 
  - *Về Latency:* Config B bổ sung bước tính BM25 trên CPU (~8ms) và tính toán RRF (~2ms), tổng thời gian xử lý chỉ tăng thêm ~10ms trên mỗi truy vấn, hoàn toàn không đáng kể so với thời gian sinh từ của LLM (~600ms - 800ms trên Groq).
  - *Về Cost:* Cả hai cấu hình đều chỉ gọi 1 request embedding cho query và 1 request generation qua Groq API, chi phí tài nguyên API là tương đương nhau 100%.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Thời hạn tối đa để Người Mua gửi yêu cầu Trả hàng và Hoàn tiền trên Shopee là bao lâu kể từ khi đơn hàng giao thành công? | Config A | 0.720 | 0.810 | 0.667 | 0.650 | retrieval | Kích thước chunk cố định (500 ký tự) vô tình cắt đôi đoạn văn quy định thời hạn 15 ngày của Shopee Mall và thời hạn của Người bán thường, khiến dense search chỉ lấy được một nửa ngữ cảnh. |
|   2 | Chi phí của Dịch Vụ Hiển Thị Trang Chủ Shopee được tính như thế nào? | Config A | 0.780 | 0.830 | 0.750 | 0.710 | generation | Mô hình LLM giải thích thêm định nghĩa thuật ngữ viết tắt oCPM từ kiến thức tiền huấn luyện thay vì chỉ trích xuất ngắn gọn từ context cung cấp. |
|   3 | Người dùng Shopee có trách nhiệm gì đối với thông tin đăng nhập tài khoản của mình theo Điều khoản dịch vụ? | Config B | 0.860 | 0.880 | 0.800 | 0.740 | data | Văn bản Điều khoản Dịch vụ lặp lại nhiều lần cụm từ "bảo mật", "tài khoản" tại nhiều điều khoản khác nhau, làm loãng điểm xếp hạng và kéo theo các chunk thứ yếu vào top-5. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Áp dụng Semantic Chunking hoặc Header-based Chunking cho văn bản pháp lý | Failure Case 1 bị đứt đoạn điều khoản quan trọng do cắt cố định 500 ký tự | Tăng Context Recall thêm 5-8% trên các văn bản luật có cấu trúc điều/khoản | So sánh điểm Context Recall của bộ 5 câu hỏi pháp lý trước và sau khi đổi splitter |
|        2 | Thắt chặt System Prompt về việc cấm bổ sung định nghĩa ngoài context | Failure Case 2 bị LLM giải thích thêm thuật ngữ oCPM | Nâng Faithfulness đạt mức tuyệt đối (> 0.96) cho các câu hỏi chứa từ viết tắt | Chạy lại metric Faithfulness trên 3 câu hỏi liên quan đến thuật ngữ quảng cáo |
|        3 | Bổ sung Cross-Encoder Reranker chuyên dụng (BGE-Reranker-v2) sau RRF | Failure Case 3 bị nhiễu do nhiều chunk có từ khóa giống nhau nhưng không trả lời trọng tâm | Tăng Context Precision từ 0.885 lên trên 0.920 | Đo lường độ chính xác thứ tự xếp hạng của top-3 chunks sau rerank |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Chuyển đổi LLM Generator: Groq LPU (`openai/gpt-oss-120b`) so với Local LLM (`Ollama Qwen2.5-7B`) | Local LLM | Faithfulness +0.03, Answer Relevance +0.05 | Latency giảm từ 8.4s xuống 0.72s (-91.4% thời gian chờ); Cost: Miễn phí | Groq Cloud LPU mang lại trải nghiệm tương tác thời gian thực vượt trội hoàn toàn cho Chatbot mà vẫn đảm bảo tính bảo mật và tuân thủ định dạng citation nghiêm ngặt. |
| Đánh giá Embedding: Gemini Cloud 3072 chiều so với BGE-M3 1024 chiều | BGE-M3 (1024 dim) | Context Recall +0.04, Cosine Separation +0.12 | Tốc độ batching API đạt ~400 chunks trong 12s | Vector 3072 chiều của Gemini phân tách ngữ nghĩa các văn bản thương mại điện tử tiếng Việt rõ nét hơn, ngưỡng threshold phân định in-domain và out-of-domain đáng tin cậy hơn. |
