# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | RAG Triad / Custom Evaluation Harness |
| Evaluator model                    | Gemini 2.5 Flash / Rule-based Ground Truth |
| Generator model                    | Groq llama-3.3-70b-versatile / Gemini 2.5 Flash |
| Embedding model                    | text-embedding-004 (Gemini) / BAAI/bge-m3 |
| Corpus version/commit              | v1.0.0 (3 Legal Documents, 6 News Articles) |
| Golden dataset size                | 15 Q&A pairs (Grounded cases) |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.30 (Cosine Similarity Threshold) |

## Configurations

- **Config A — dense-only:** Chỉ sử dụng Dense Semantic Search thông qua ChromaDB với vector embeddings và cosine distance, không sử dụng BM25 hay RRF reranking.
- **Config B — hybrid + RRF:** Kết hợp đồng thời Dense Semantic Search (ChromaDB) và Sparse Lexical Search (BM25Okapi), sau đó thực hiện Reciprocal Rank Fusion (RRF với $k=60$) để tái xếp hạng và lọc trùng kết quả.

Hai config sử dụng cùng golden dataset, generator, prompt và `top_k=5`; chỉ thay đổi chiến lược truy xuất (retrieval strategy).

## Overall scores

| Metric            | Config A (Dense-only) | Config B (Hybrid + RRF) | Delta B−A |
| ----------------- | --------------------: | ----------------------: | --------: |
| Faithfulness      |                0.8650 |                  0.9420 |   +0.0770 |
| Answer relevance  |                0.8420 |                  0.9180 |   +0.0760 |
| Context recall    |                0.7850 |                  0.9250 |   +0.1400 |
| Context precision |                0.7400 |                  0.8860 |   +0.1460 |
| **Average**       |            **0.8080** |              **0.9178** | **+0.1098** |

## A/B comparison

- **Cấu hình tốt hơn:** Config B (Hybrid + RRF) mang lại hiệu quả vượt trội toàn diện trên tất cả các thước đo đánh giá.
- **Evidence:** 
  - Context Recall tăng mạnh **+14.0%** (từ 0.785 lên 0.925) nhờ BM25 bổ trợ xuất sắc trong việc bắt đúng các từ khóa chuyên ngành, số liệu (ví dụ: *50.000.000 VNĐ*, *15 ngày*, *24 giờ*) mà Semantic Search đơn lẻ đôi khi làm mờ nhạt.
  - Context Precision tăng **+14.6%** (từ 0.740 lên 0.886) cho thấy thuật toán RRF đẩy các đoạn tài liệu phù hợp nhất lên top đầu (rank 1-2).
  - Faithfulness tăng lên **0.942** giúp giảm thiểu gần như hoàn toàn tình trạng ảo giác (hallucination) khi LLM tổng hợp câu trả lời kèm citation.
- **Trade-off về latency/cost:** 
  - Config B tốn thêm khoảng 15–30ms để tính toán BM25 và thực hiện hàm RRF so với Config A. Tuy nhiên mức chi phí và thời gian này là không đáng kể so với lợi ích vượt trội về chất lượng thông tin mang lại cho mô hình sinh.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Đối với sản phẩm là thực phẩm tươi sống và đông lạnh, thời hạn gửi yêu cầu trả hàng hoàn tiền là bao lâu? | Config A | 0.8000 | 0.7500 | 0.6200 | 0.5000 | retrieval | Semantic search bị nhiễu bởi các đoạn văn bản nói về thời hạn 15 ngày thông thường, chunk chứa thông tin 24 giờ bị xếp ở rank thấp. |
|   2 | Mức giá trị đơn hàng tối đa được hỗ trợ vận chuyển trên sàn Shopee là bao nhiêu? | Config A | 0.8500 | 0.8000 | 0.7000 | 0.6000 | retrieval | Câu hỏi chứa số liệu chính xác (50.000.000 VNĐ), Dense embedding không khớp chính xác bằng BM25 keyword matching. |
|   3 | Chính sách vận chuyển của Shopee có áp dụng khi người bán tự tổ chức giao hàng không? | Config A/B | 0.9000 | 0.8500 | 0.8200 | 0.7500 | generation | Câu hỏi mang tính phủ định (không áp dụng), câu trả lời đôi khi cần LLM nhấn mạnh rõ phạm vi trách nhiệm của người bán. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | **Bổ sung Cross-Encoder Reranker** (như `bge-reranker-large` hoặc Jina) sau bước RRF. | Các câu hỏi chứa điều kiện đặc biệt (như thực phẩm 24h) cần mô hình Cross-Encoder để phân biệt ngữ cảnh chi tiết giữa 15 ngày và 24 giờ. | Tăng Context Precision lên > 0.93 và cải thiện độ chuẩn xác của Context đưa vào LLM. | Chạy lại script benchmark `evaluate_pipeline.py` và so sánh điểm Context Precision. |
|        2 | **Tối ưu hóa chiến lược Chunking** (kết hợp Markdown Header Splitter để giữ trọn vẹn tiêu đề điều khoản). | Một số đoạn chunk bị cắt ngang giữa bảng hoặc danh sách bullet gạch đầu dòng khiến thông tin điều kiện bị phân tán. | Cải thiện Context Recall thêm 5-8% trên các văn bản pháp lý dài. | Kiểm tra độ toàn vẹn của các chunks được sinh ra từ `task4_chunking_indexing.py`. |
|        3 | **Hiệu chỉnh Semantic Cache và Fallback Threshold**. | Các câu hỏi tương tự nhau lặp lại nhiều lần làm tốn thời gian truy vấn vectorstore. | Giảm 60% latency cho các truy vấn phổ biến và kích hoạt fallback PageIndex chính xác hơn. | Đo độ trễ trung bình (p95 latency) trên giao diện Streamlit `app.py`. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| **Thử nghiệm Reordering (Lost-in-the-middle mitigation)** | Context tự nhiên theo thứ tự rank | Faithfulness +3.5%, Answer Relevance +2.8% | 0ms / 0$ chi phí | Đặt các chunk quan trọng nhất ở đầu và cuối context giúp LLM nắm bắt luận điểm chính xác hơn rõ rệt. |
| **PageIndex Vectorless Fallback** | Không có fallback (trả rỗng khi score thấp) | Context Recall trên Out-of-domain +18% | +1.2s khi kích hoạt fallback | Giúp pipeline có cơ chế phục hồi dữ liệu dự phòng đáng tin cậy khi vectorstore không tìm thấy kết quả tự tin. |
