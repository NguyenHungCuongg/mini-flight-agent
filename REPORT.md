# Báo cáo thực hành BTVN#3: Agent đặt vé máy bay bằng LangChain

|           |                                                         |
| --------- | ------------------------------------------------------- |
| Môn học   | SE373: Kỹ thuật xây dựng hệ thống Agentic AI            |
| Họ và tên | `Nguyễn Hùng Cường`                                     |
| MSSV      | `23520201`                                              |
| Lớp       | `SE373.R11`                                             |
| Repo      | `https://github.com/NguyenHungCuongg/mini-flight-agent` |

## 1. Yêu cầu bài thực hành

1. Tạo tool mockup và viết lớp harness cho agent đặt vé, gồm: ràng buộc là dữ liệu, tiêu chí hoàn thành kiểm bằng code, kiểm quyền, bàn giao.
2. Cài agent theo 3 mẫu thiết kế: ReAct, Plan-then-Execute, Lai.
3. Đánh giá hiệu quả của 3 mẫu.

Mức độ hoàn thành:

| Yêu cầu           | Trạng thái                            | Nằm ở                                  |
| ----------------- | ------------------------------------- | -------------------------------------- |
| Tool mockup       | Xong                                  | `World` trong `flight_agent/core.py`   |
| 4 lớp harness     | Xong, thêm ngân sách và phát hiện lặp | `Harness` trong `flight_agent/core.py` |
| ReAct             | Xong                                  | `flight_agent/react.py`                |
| Plan-then-Execute | Xong                                  | `flight_agent/plan_execute.py`         |
| Lai               | Xong                                  | `flight_agent/hybrid.py`               |
| Đánh giá          | Xong, 18 lần chạy với LLM thật        | `flight_agent/evaluate.py`, `results/` |
| Kiểm thử tự động  | 23 test, đều pass                     | `tests/test_run.py`                    |

## 2. Môi trường thực hiện

| Thành phần                    | Phiên bản / giá trị                                                     |
| ----------------------------- | ----------------------------------------------------------------------- |
| Hệ điều hành                  | Windows 11                                                              |
| Quản lý môi trường            | `uv` 0.12.5 (tự cài Python, khoá phiên bản trong `uv.lock`)             |
| Python                        | ≥ 3.12                                                                  |
| `langchain` / `langgraph`     | 1.4.3 / 1.2.12                                                          |
| `langchain-openai` / `openai` | 1.6.7 / 3.23.0                                                          |
| `pydantic`                    | 2.13.5                                                                  |
| Kiểm thử                      | `pytest` 9.1.1                                                          |
| LLM                           | `qwen/qwen3.8-27b:free` qua OpenRouter (`https://openrouter.ai/api/v1`) |

### 2.1. Chọn model

Chỉ dùng được model miễn phí, nên trước khi viết code đã thử trực tiếp từng lựa chọn bằng một request có tool calling:

| Lựa chọn                                                      | Kết quả thử                                                                       | Quyết định                         |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------- |
| `nex-agi/nex-n2.5-pro:free` (OpenRouter, cấu hình ban đầu)    | HTTP 404: model không còn bản miễn phí                                            | Bỏ                                 |
| `qwen3.8-27b` trên `llm.uit.edu.vn` (key giảng viên cung cấp) | Bị chuyển hướng về `www.uit.edu.vn`: endpoint chỉ truy cập được trong mạng trường | Không dùng được khi ở ngoài trường |
| `google/gemma-4-26b-a4b-it:free` (OpenRouter)                 | HTTP 429: nhà cung cấp đang giới hạn                                              | Bỏ                                 |
| `qwen/qwen3.8-27b:free` (OpenRouter)                          | HTTP 200, gọi đúng tool với đúng tham số, 2,1 giây                                | **Chọn**                           |

`qwen/qwen3.8-27b:free` là cùng model Qwen3.8-27B với server của trường. Vì toàn bộ cấu hình model nằm trong `.env`, khi ở trong mạng trường có thể chuyển sang `https://llm.uit.edu.vn/qwen/v1` mà không sửa code.

Các ràng buộc thực tế ảnh hưởng tới cách làm:

- **Quota 50 request/ngày.** Đã kiểm qua endpoint `/api/v1/key` của OpenRouter: `is_free_tier: true`, `free_model_daily_requests.limit: 50`.
- **Nhà cung cấp giới hạn lượt gọi bất chợt (429).** Gặp 1 lần trong lúc phát triển.
- **Hạn nộp còn 4 ngày** tại thời điểm bắt đầu.

Cấu hình model (`flight_agent/model.py`):

```python
ChatOpenAI(model=OPENAI_MODEL, base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY,
           temperature=0, max_retries=3, timeout=120,
           extra_body={"reasoning": {"enabled": False},                         # OpenRouter
                       "chat_template_kwargs": {"enable_thinking": False}})     # vLLM của trường
```

- **`temperature=0`:** để kết quả ổn định nhất có thể.
- **Tắt chế độ suy nghĩ của Qwen:** để mỗi lần gọi nhanh hơn và tránh câu trả lời bị cắt do hết token. Hai khoá trong `extra_body` để cùng cấu hình chạy được ở cả OpenRouter và server của trường.

## 3. Quy trình thực hiện

| Bước | Việc đã làm                                                                                                        | Sản phẩm                                                           |
| ---- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| 1    | Đọc đề, slide và 3 file demo của giảng viên                                                                        | Hiểu cấu trúc tool, harness, cách dùng `create_agent` và model giả |
| 2    | Thử các nguồn model (mục 2.1)                                                                                      | Chọn Qwen3.8-27B miễn phí qua OpenRouter                           |
| 3    | Chốt yêu cầu chi tiết: phạm vi harness, 6 kịch bản, phương pháp đánh giá, người duyệt                              | Spec và bảng thuật ngữ `CONTEXT.md`                                |
| 4    | Viết code theo TDD (viết test trước, chạy thấy đỏ, rồi viết code cho xanh), test bằng model giả để không tốn quota | `flight_agent/`, `tests/test_run.py`                               |
| 5    | Review code theo hai hướng: chuẩn viết code và độ khớp với spec. Sửa các lỗi ảnh hưởng tới kết quả (mục 8)         | 23 test pass                                                       |
| 6    | Chạy thử 1 lần với LLM thật trước khi đánh giá                                                                     | Plan-then-Execute / `valid` → `done`, 1 lần gọi, 644 token         |
| 7    | Chạy đánh giá chính thức                                                                                           | `results/runs.jsonl`, `results/summary.md`                         |
| 8    | Chạy demo với người duyệt thật                                                                                     | Mục 6                                                              |
| 9    | Viết báo cáo                                                                                                       | File này                                                           |

## 4. Cài đặt

### 4.1. Cấu trúc mã nguồn

| File                           | Nội dung                                                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `flight_agent/core.py`         | `Constraints`, 6 kịch bản (`SCENARIOS`), tool giả (`World`), người duyệt (`RuleApprover`, `HumanApprover`), `Plan`/`Step`, `Harness` |
| `flight_agent/run.py`          | Điểm vào duy nhất `run(pattern, scenario, model, approver) -> RunResult`                                                             |
| `flight_agent/react.py`        | ReAct                                                                                                                                |
| `flight_agent/plan_execute.py` | Plan-then-Execute (phần lập kế hoạch dùng chung với mẫu Lai)                                                                         |
| `flight_agent/hybrid.py`       | Lai                                                                                                                                  |
| `flight_agent/model.py`        | Tạo model thật từ `.env`                                                                                                             |
| `flight_agent/demo.py`         | Demo tương tác, người duyệt là người thật                                                                                            |
| `flight_agent/evaluate.py`     | Chạy 3 mẫu × 6 kịch bản, lưu và tổng hợp kết quả                                                                                     |
| `tests/test_run.py`            | 23 test với model giả                                                                                                                |

### 4.2. Một điểm vào cho mọi thứ

Demo, đánh giá và test đều gọi cùng một hàm:

```python
run(pattern, scenario, model, approver) -> RunResult
```

Một lần chạy diễn ra như sau:

```
run()
 ├─ tạo Harness mới (kèm World mới: booking không bị dồn sang lần chạy khác)
 ├─ chạy mẫu: run_react / run_plan_execute / run_hybrid
 │     ├─ mỗi lần gọi model → harness.before_model() / after_model()   (ngân sách, đếm token)
 │     └─ mỗi lần gọi tool  → harness.execute(tool, args)
 │            phát hiện lặp → kiểm quyền → chạy tool → ghi trace
 ├─ bắt StopRun (harness chủ động dừng) / bắt lỗi API tạm thời
 └─ outcome = "infra_error" | "done" nếu harness.is_done() | "handoff" + gói bàn giao
```

Lý do thiết kế như vậy:

- **So sánh công bằng.** Ba mẫu bắt buộc dùng chung một harness và một bộ dữ liệu, nên khác biệt trong kết quả chỉ đến từ mẫu thiết kế.
- **Mẫu không tự quyết định đã xong.** Mẫu không tự kiểm ràng buộc và không tự quyết định "xong". Kết quả cuối cùng luôn do `run()` quyết định qua `is_done()`.

`RunResult` gồm:

| Trường                             | Ý nghĩa                                       |
| ---------------------------------- | --------------------------------------------- |
| `outcome`                          | Kết quả: `done`, `handoff` hoặc `infra_error` |
| `expected`                         | Kết quả mong đợi của kịch bản                 |
| `stop_reason`                      | Lý do dừng                                    |
| `handoff`                          | Gói bàn giao                                  |
| `trace`                            | Danh sách lời gọi tool kèm kết quả            |
| `model_calls`, `tokens`, `seconds` | Chi phí của lần chạy                          |
| `final_answer`                     | Câu trả lời cuối của model                    |
| `error`                            | Thông tin lỗi API, nếu có                     |

### 4.3. Tool mockup

Class `World` cung cấp 5 tool, đọc từ dữ liệu của kịch bản, không gọi mạng:

| Tool                                        | Hành vi                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `search_flights(origin, destination, date)` | Trả danh sách chuyến nếu đúng tuyến và ngày. Sai thì trả `{"status": "ok", "count": 0, "flights": []}` |
| `check_seat(flight)`                        | Trả số ghế còn và giá. Chuyến hết ghế trả `seats_left: 0`                                              |
| `book_seat(flight)`                         | Giữ chỗ, trả mã booking (`VN122-12A`), chưa trừ tiền. Chuyến hết ghế trả `sold_out`                    |
| `pay(code)`                                 | Thanh toán booking. Ở kịch bản có lỗi tạm thời, lần gọi đầu trả `tool_error`                           |
| `get_booking(code)`                         | Đọc lại booking                                                                                        |

Quy ước khi thiết kế tool:

- **Mọi kết quả đều là JSON có `status`** thuộc tập cố định (`ok`, `not_found`, `sold_out`, `tool_error`, `invalid_param`, `denied`). Không bao giờ trả chuỗi rỗng.
- **Lỗi có `hint` chỉ bước tiếp theo.** Ví dụ `"Payment gateway timeout. Nothing was charged. Retry pay."`.
- **Mỗi lần chạy có một `World` riêng.** Demo của giảng viên dùng biến toàn cục `BOOKINGS`, nên khi chạy 18 lần liên tiếp thì booking của lần trước sẽ ảnh hưởng lần sau.

### 4.4. Kịch bản

Mọi kịch bản dùng cùng ràng buộc: **SGN → DAD, 07/10/2026, khởi hành trước 12:00, giá tối đa 2.000.000 VND**. Dữ liệu chuyến bay tự tạo, viết cứng trong `SCENARIOS`.

| Kịch bản          | Dữ liệu                                                                           | Mục đích kiểm                        | Mong đợi  |
| ----------------- | --------------------------------------------------------------------------------- | ------------------------------------ | --------- |
| `valid`           | VN122 08:10 1.850.000; QH118 15:40 1.640.000                                      | Trường hợp thuận lợi                 | `done`    |
| `no_valid`        | VJ604 08:10 **2.480.000**; QH118 **15:40** 1.640.000                              | Không có lời giải, phải dừng         | `handoff` |
| `trap`            | VJ610 **14:20** 1.150.000 (rẻ nhất); QH112 10:30 2.300.000; VN122 08:10 1.950.000 | Có quên ràng buộc giờ bay không      | `done`    |
| `transient_error` | Như `valid`; lần `pay` đầu tiên trả `tool_error`                                  | Có thích nghi với lỗi tạm thời không | `done`    |
| `sold_out`        | VN122 08:10 (báo còn 1 ghế, thực tế đã hết); VJ124 09:30 1.950.000                | Thế giới đổi sau khi tìm kiếm        | `done`    |
| `needs_approval`  | Như `valid`; hạn mức tự thanh toán **1.500.000**                                  | Có dừng chờ người duyệt không        | `handoff` |

### 4.5. Bốn lớp harness

**(1) Ràng buộc là dữ liệu.** `Constraints` là một dataclass `frozen=True`:

- `to_prompt()` sinh câu yêu cầu gửi cho model từ dữ liệu.
- `is_ok(flight)` là một luật duy nhất (đúng ngày, giờ < `depart_before`, giá ≤ `max_price`), dùng chung ở 3 nơi: kiểm quyền, tiêu chí hoàn thành, và người duyệt bằng luật.
- Với Plan-then-Execute và Lai, bước tìm chuyến do code gọi với tham số lấy từ `Constraints`, không do model điền.

**(2) Tiêu chí hoàn thành kiểm bằng code.** `Harness.is_done()` đọc lại từng booking qua `get_booking`. Chỉ trả `True` khi có booking `paid = True` và thoả `is_ok()`. Câu trả lời của model không được dùng để quyết định kết quả.

**(3) Kiểm quyền.** `Harness.check_permission()` chạy **trước** khi tool thực thi:

| Lời gọi                       | Điều kiện chặn                                        | Hệ quả                                       |
| ----------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| Tool không có trong danh sách | Luôn chặn                                             | `denied`                                     |
| `book_seat(flight)`           | Chuyến không tồn tại hoặc vi phạm ràng buộc           | `denied` + lý do; tool không chạy            |
| `pay(code)`                   | Giá > hạn mức của kịch bản **và** người duyệt từ chối | `denied`, dừng lần chạy với `needs_approval` |

Người duyệt có hai bản cài cùng giao diện (`approve_plan`, `approve_payment`):

- **`HumanApprover`** dùng cho demo, hỏi `y/n` qua bàn phím.
- **`RuleApprover`** dùng cho đánh giá: duyệt kế hoạch khi mọi chuyến được đặt đều thoả ràng buộc, và luôn từ chối thanh toán vượt hạn mức. Nhờ vậy đánh giá chạy tự động và lặp lại được.

**(4) Bàn giao.** `Harness.handoff()` trả về 3 trường:

- `done_so_far`: đã làm tới đâu, gồm cả booking đã giữ hay đã trả tiền;
- `tried`: các lời gọi tool kèm trạng thái;
- `question`: câu hỏi cụ thể, chọn theo lý do dừng.

| Lý do dừng          | Câu hỏi bàn giao                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------- |
| `gave_up`           | No paid booking meets all constraints. Which can we relax: departure time or maximum price? |
| `needs_approval`    | Approve paying 1,850,000 VND for VN122?                                                     |
| `step_failed`       | Last step book_seat -> sold_out. Retry it, or pick another flight?                          |
| `replans_exhausted` | Last step … 2 new plans also failed. Pick a flight by hand?                                 |
| `plan_rejected`     | 2 plans were rejected. Book by hand, or relax a constraint?                                 |
| `invalid_plan`      | The model returned a plan that could not be read. Retry or take over?                       |
| `loop`              | The agent repeated check_seat with no progress. Retry later or take over?                   |
| `budget`            | The budget of 10 model calls is used up. Raise the budget or take over?                     |

### 4.6. Cơ chế dừng bổ sung

| Cơ chế        | Cài đặt                                                                                                                                                                                                | Giá trị                              |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| Ngân sách     | `Harness.before_model()` đếm mọi lần gọi model, kể cả lần lập kế hoạch                                                                                                                                 | `BUDGET = 10`                        |
| Phát hiện lặp | `Harness.is_looping()` so `(tool, args)` trong cửa sổ các lời gọi gần nhất. `get_booking` được miễn (đọc lại để chờ xác nhận)                                                                          | `LOOP_WINDOW = 6`, `LOOP_REPEAT = 3` |
| Lỗi hạ tầng   | `run()` bắt `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError` sau khi client đã tự thử lại 3 lần, rồi trả `infra_error`. Lỗi 401/404 không bị bắt để lộ ra sai cấu hình | `max_retries = 3`                    |

ReAct dùng ngân sách tự viết thay cho `ModelCallLimitMiddleware` như trong demo của giảng viên. Lý do là `ModelCallLimitMiddleware` chỉ áp dụng được cho vòng lặp của `create_agent`, trong khi cả 3 mẫu cần được đếm theo cùng một cách.

### 4.7. Ba mẫu thiết kế

**ReAct** (`react.py`):

- `create_agent(model, tools, system_prompt, middleware=[budget, through_harness])`.
- `budget` (`@wrap_model_call`) gọi `before_model` / `after_model`.
- `through_harness` (`@wrap_tool_call`) **không** gọi tool trực tiếp mà chuyển sang `harness.execute()`, rồi trả `ToolMessage` có `tool_call_id`.
- Khi harness đã đặt `stop_reason` (lặp, bị từ chối duyệt), lần gọi model kế tiếp ném `StopRun` để kết thúc.

**Plan-then-Execute** (`plan_execute.py`):

1. `search()`: code gọi `search_flights` với tham số lấy từ `Constraints`.
2. `make_plan()`: một lần gọi `PLANNER_PROMPT | model.with_structured_output(Plan, method="function_calling", include_raw=True)`. Kế hoạch là danh sách `Step(tool, args)`. Mã booking chưa biết được viết là `"$booking_code"`. Kế hoạch rỗng nghĩa là "không có chuyến thoả".
3. `approved_plan()`: người duyệt xem kế hoạch. Nếu từ chối, gọi lại model kèm lý do, tối đa `MAX_PLANS = 2`.
4. `execute_plan()`: chạy từng bước qua `harness.execute()`, `fill_placeholders()` thay `$booking_code` bằng mã thật. Không gọi lại model. Bước đầu tiên có `status ≠ ok` thì dừng với `step_failed`.

**Lai** (`hybrid.py`): dùng lại toàn bộ các hàm trên. Khi `execute_plan()` gặp bước lỗi, harness gọi `approved_plan()` với `feedback` chứa trace các bước đã chạy, tối đa `MAX_REPLANS = 2` lần. Hết lượt mà vẫn lỗi thì dừng với `replans_exhausted`.

Không dùng `TodoListMiddleware` cho mẫu Lai. Lý do: với middleware đó, model tự quyết định khi nào cập nhật kế hoạch. Còn ở đây harness quyết định, cụ thể là "khi một bước không `ok`", nên đếm và giải thích được khi so sánh với hai mẫu kia.

## 5. Kiểm thử tự động

23 test trong `tests/test_run.py`, chỉ gọi `run()` với model giả có kịch bản trả lời (subclass `GenericFakeChatModel`, override `bind_tools`, `with_structured_output`, `_generate`). Không gọi mạng, không tốn quota.

| Nhóm              | Test kiểm điều gì                                                                                                                                                                                                                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ReAct             | Đặt được vé ở `valid`. Chặn `book_seat(VJ610)` ở `trap`. Model nói "đã xong" mà chưa trả tiền thì vẫn `handoff`. Lặp `check_seat` 3 lần thì dừng `loop`. Đọc lại `get_booking` nhiều lần không bị coi là lặp. Hết 10 lần gọi thì dừng `budget`. Thanh toán vượt hạn mức bị từ chối thì `needs_approval`, câu hỏi nêu đúng số tiền |
| Plan-then-Execute | Đặt được vé với 1 lần gọi model. Hết ghế thì `step_failed`. Kế hoạch bị từ chối thì được viết lại; 2 lần bị từ chối thì `plan_rejected`. Lỗi tạm thời thì `handoff`, booking còn ở trạng thái giữ chỗ                                                                                                                             |
| Lai               | Hết ghế thì lập lại kế hoạch và `done`. Lỗi tạm thời thì thử lại `pay` và `done`. Hết lượt lập lại kế hoạch thì `replans_exhausted`, câu hỏi bàn giao không phải câu "nới ràng buộc"                                                                                                                                              |
| Chung             | Cả 3 mẫu `done` ở `valid`. 429 thì `infra_error`. 401 thì dừng hẳn. Hai lần chạy không dùng chung booking. Script đánh giá lưu từng lần chạy và bỏ qua lần đã có                                                                                                                                                                  |

Kết quả:

```
$ uv run pytest -q
.......................                                                  [100%]
23 passed in 1.32s
```

## 6. Chạy demo với người duyệt thật

Lệnh: `uv run python -m flight_agent.demo`. Chọn mẫu và kịch bản, rồi trả lời `y/n` khi được hỏi duyệt. Mục đích là xem những nhánh mà đánh giá tự động không đi vào được, vì `RuleApprover` luôn từ chối thanh toán vượt hạn mức. Đã chạy 3 lần:

| #   | Mẫu / kịch bản                 | Người duyệt được hỏi                                     | Kết quả               | Gọi model / token |
| --- | ------------------------------ | -------------------------------------------------------- | --------------------- | ----------------- |
| 6.1 | Lai / `needs_approval`         | Duyệt kế hoạch: **y**; duyệt thanh toán 1.850.000: **y** | `done`                | 1 / 644           |
| 6.2 | Plan-then-Execute / `no_valid` | Không hỏi (kế hoạch rỗng)                                | `handoff` (`gave_up`) | 1 / 570           |
| 6.3 | Plan-then-Execute / `trap`     | Không hỏi (lỗi API trước khi có kế hoạch)                | `infra_error`         | 1 / 0             |

### 6.1. Lai + `needs_approval`: người duyệt đồng ý thanh toán

Kế hoạch do model đưa ra và hai câu hỏi duyệt:

```
  Step 1: check_seat({'flight': 'VN122'})
  Step 2: book_seat({'flight': 'VN122'})
  Step 3: pay({'code': '$booking_code'})
  Step 4: get_booking({'code': '$booking_code'})
Human reviewer - approve this plan? (y/n): y
Approve paying 1,850,000 VND for VN122 (above the 1,500,000 VND auto-pay limit)? (y/n): y
```

```
--- trace ---
[1] search_flights({'origin': 'SGN', 'destination': 'DAD', 'date': '2026-10-07'}) -> ok
[2] check_seat({'flight': 'VN122'}) -> ok
[3] book_seat({'flight': 'VN122'}) -> ok
[4] pay({'code': 'VN122-12A'}) -> ok
[5] get_booking({'code': 'VN122-12A'}) -> ok
```

```json
{
  "outcome": "done",
  "stop_reason": null,
  "handoff": null,
  "model_calls": 1,
  "tokens": 644,
  "error": ""
}
```

Nhận xét:

- **Hai câu hỏi duyệt xuất hiện ở hai thời điểm khác nhau.** Câu duyệt kế hoạch hiện ra trước khi bất kỳ bước nào chạy. Câu duyệt thanh toán hiện ra đúng lúc bước `pay` sắp chạy (bước 4 trong trace), do `check_permission()` thấy 1.850.000 > hạn mức 1.500.000.
- **Cùng kịch bản, khác người duyệt, khác kết quả.** Khi đánh giá tự động (mục 7), `RuleApprover` từ chối, nên kết quả là `handoff` (`needs_approval`) với câu hỏi _"Approve paying 1,850,000 VND for VN122?"_. Khi người thật đồng ý, harness ghi nhận booking đã được duyệt (`approved_codes`), cho `pay` chạy, và `is_done()` đọc lại booking đã thanh toán nên ra `done`. Mô hình, kế hoạch và chi phí (1 lần gọi, 644 token) giống hệt nhau; chỉ quyết định của người duyệt là khác.

### 6.2. Plan-then-Execute + `no_valid`: không có câu hỏi duyệt

```
--- trace ---
[1] search_flights({'origin': 'SGN', 'destination': 'DAD', 'date': '2026-10-07'}) -> ok
```

```json
{
  "outcome": "handoff",
  "stop_reason": "gave_up",
  "handoff": {
    "done_so_far": ["Nothing booked, nothing paid"],
    "tried": [
      "search_flights({'origin': 'SGN', 'destination': 'DAD', 'date': '2026-10-07'}) -> ok"
    ],
    "question": "No paid booking meets all constraints. Which can we relax: departure time or maximum price?"
  },
  "model_calls": 1,
  "tokens": 570,
  "error": ""
}
```

Nhận xét:

- **Không có câu hỏi duyệt vì model trả về kế hoạch rỗng ngay ở lần gọi đầu.** Theo quy ước, kế hoạch rỗng nghĩa là "không có chuyến nào thoả mọi ràng buộc". `approved_plan()` dừng với `gave_up` trước bước hỏi người duyệt, vì không có gì để duyệt.
- **Model đọc đúng dữ liệu:** VJ604 buổi sáng nhưng 2.480.000 > 2.000.000; QH118 rẻ nhưng bay 15:40.
- **Kết quả giống hệt lần đánh giá tự động:** 1 lần gọi, 570 token, cùng câu hỏi bàn giao.
- **Nhánh "người duyệt từ chối kế hoạch → model viết lại" không xuất hiện** với Qwen ở kịch bản này. Nhánh đó chỉ được kiểm qua test với model giả: `test_rejected_plan_gets_one_more_try_then_empty_plan_hands_off` và `test_two_rejected_plans_hand_off`.

### 6.3. Plan-then-Execute + `trap`: nhà cung cấp trả lỗi 429

```
--- trace ---
[1] search_flights({'origin': 'SGN', 'destination': 'DAD', 'date': '2026-10-07'}) -> ok
```

```json
{
  "outcome": "infra_error",
  "stop_reason": null,
  "handoff": null,
  "model_calls": 1,
  "tokens": 0,
  "error": "OpenAIRateLimitError: Error code: 429 - ... 'qwen/qwen3.8-27b:free is temporarily rate-limited upstream. Please retry shortly ...' ..."
}
```

Nhận xét:

- **Không có câu hỏi duyệt vì lần gọi model để lập kế hoạch thất bại.** Nhà cung cấp phía sau OpenRouter đang giới hạn lượt gọi (`upstream_provider_shared_pool`). Client đã tự thử lại 3 lần nhưng vẫn lỗi, nên không có kế hoạch nào để duyệt.
- **Harness xử lý đúng cách:**
  - kết quả là `infra_error`, không phải `handoff`, nên lần chạy này không bị tính là agent thất bại;
  - không có gói bàn giao;
  - `tokens = 0` vì không nhận được phản hồi;
  - `search_flights` vẫn có trong trace vì bước này do code chạy trước khi gọi model.
- **Nếu xảy ra trong lúc đánh giá,** `evaluate.py` sẽ ghi lần chạy này là `infra_error` và tự chạy lại ở lần sau.
- **Kết quả đúng của tổ hợp này** đã có trong lần đánh giá chính thức (mục 7): `done`, 1 lần gọi, 695 token.

## 7. Đánh giá 3 mẫu

### 7.1. Thiết lập

| Mục         | Giá trị                                                                    |
| ----------- | -------------------------------------------------------------------------- |
| Lệnh        | `uv run python -m flight_agent.evaluate`                                   |
| Model       | `qwen/qwen3.8-27b:free` (OpenRouter), `temperature=0`, tắt chế độ suy nghĩ |
| Số lần chạy | 3 mẫu × 6 kịch bản × K = 1 = **18**                                        |
| Người duyệt | `RuleApprover`                                                             |
| Ngày chạy   | `<điền>`                                                                   |
| Dữ liệu gốc | `results/runs.jsonl` (đủ trace từng lần chạy), `results/summary.md`        |

Chỉ chạy K = 1 vì một vòng đánh giá tốn khoảng 50 request, bằng đúng quota một ngày, trong khi hạn nộp còn 4 ngày. Script lưu từng lần chạy ngay khi xong, và khi chạy lại sẽ bỏ qua các lần đã có kết quả. Vì vậy có thể tăng K khi có thêm quota mà không phải chạy lại từ đầu.

### 7.2. Kết quả theo kịch bản

✓ nghĩa là kết quả khớp với mong đợi; trong ngoặc là lý do dừng.

| Kịch bản        | Mong đợi | ReAct                      | Plan-then-Execute          | Lai                        |
| --------------- | -------- | -------------------------- | -------------------------- | -------------------------- |
| valid           | done     | ✓ done                     | ✓ done                     | ✓ done                     |
| no_valid        | handoff  | ✓ handoff (gave_up)        | ✓ handoff (gave_up)        | ✓ handoff (gave_up)        |
| trap            | done     | ✓ done                     | ✓ done                     | ✓ done                     |
| transient_error | done     | ✓ done                     | ✗ handoff (step_failed)    | ✓ done                     |
| sold_out        | done     | ✓ done                     | ✗ handoff (step_failed)    | ✓ done                     |
| needs_approval  | handoff  | ✓ handoff (needs_approval) | ✓ handoff (needs_approval) | ✓ handoff (needs_approval) |

### 7.3. Tổng hợp theo mẫu

| Mẫu               | Đúng    | Done khi cần done | Bàn giao khi cần bàn giao | Lời gọi bị chặn | Gọi model (TB) | Token (TB) | Giây (TB) | Lỗi hạ tầng |
| ----------------- | ------- | ----------------- | ------------------------- | --------------- | -------------- | ---------- | --------- | ----------- |
| ReAct             | **6/6** | 4/4               | 2/2                       | 1               | 5,3            | 5.766      | 9,5       | 0           |
| Plan-then-Execute | 4/6     | 2/4               | 2/2                       | 1               | **1,0**        | **640**    | 4,4       | 0           |
| Lai               | **6/6** | 4/4               | 2/2                       | 1               | 1,3            | 908        | **2,1**   | 0           |

### 7.4. Chi phí từng lần chạy (lần gọi model / token / giây)

| Kịch bản        | ReAct            | Plan-then-Execute | Lai             |
| --------------- | ---------------- | ----------------- | --------------- |
| valid           | 6 / 6.229 / 13,1 | 1 / 644 / 6,1     | 1 / 644 / 2,2   |
| no_valid        | 2 / 1.728 / 2,8  | 1 / 570 / 0,8     | 1 / 570 / 0,8   |
| trap            | 6 / 7.391 / 8,7  | 1 / 695 / 11,2    | 1 / 695 / 1,6   |
| transient_error | 7 / 7.525 / 18,0 | 1 / 644 / 5,2     | 2 / 1.477 / 3,1 |
| sold_out        | 7 / 8.071 / 10,1 | 1 / 645 / 1,6     | 2 / 1.418 / 3,3 |
| needs_approval  | 4 / 3.652 / 4,1  | 1 / 644 / 1,6     | 1 / 644 / 1,7   |

### 7.5. Trace tiêu biểu

**`sold_out`: ba mẫu xử lý cùng một tình huống theo ba cách**

```
react         search_flights -> ok | check_seat(VN122) -> ok (0 ghế) | check_seat(VJ124) -> ok
              | book_seat(VJ124) -> ok | pay(VJ124-12A) -> ok | get_booking(VJ124-12A) -> ok
              => done, 7 lần gọi, 8.071 token

plan_execute  search_flights -> ok | check_seat(VN122) -> ok (0 ghế) | book_seat(VN122) -> sold_out
              => handoff (step_failed), 1 lần gọi, 645 token
              question: "Last step book_seat -> sold_out. Retry it, or pick another flight?"

hybrid        [kế hoạch 1] search_flights -> ok | check_seat(VN122) -> ok | book_seat(VN122) -> sold_out
              [kế hoạch 2] check_seat(VJ124) -> ok | book_seat(VJ124) -> ok | pay(VJ124-12A) -> ok
                           | get_booking(VJ124-12A) -> ok
              => done, 2 lần gọi, 1.418 token
```

**`transient_error`: gói bàn giao của Plan-then-Execute**

```json
{
  "done_so_far": ["VN122-12A: held, paid=False"],
  "tried": [
    "search_flights(...) -> ok",
    "check_seat({'flight': 'VN122'}) -> ok",
    "book_seat({'flight': 'VN122'}) -> ok",
    "pay({'code': 'VN122-12A'}) -> tool_error"
  ],
  "question": "Last step pay -> tool_error. Retry it, or pick another flight?"
}
```

Mẫu Lai ở cùng kịch bản: kế hoạch thứ 2 chỉ gồm `pay(VN122-12A)` và `get_booking(VN122-12A)`, tức là model nhận ra booking đã được giữ, chỉ cần thanh toán lại.

**`needs_approval`: cả 3 mẫu dừng ở cùng một lời gọi**

```
... book_seat(VN122) -> ok | pay(VN122-12A) -> denied
=> handoff (needs_approval), question: "Approve paying 1,850,000 VND for VN122?"
```

## 8. Nhận xét kết quả

1. **Kết quả do code quyết định ở cả 18 lần chạy.** Không có lần nào agent đặt chuyến vi phạm ràng buộc, hoặc thanh toán vượt hạn mức khi chưa được duyệt. Không có lỗi hạ tầng trong lần đánh giá chính thức.
2. **Plan-then-Execute sai đúng ở 2 kịch bản thế giới thay đổi sau khi lập kế hoạch** (`transient_error`, `sold_out`). Trong cả hai lần, nó dừng an toàn: booking ở trạng thái giữ chỗ, chưa trả tiền, câu hỏi bàn giao nêu đúng bước lỗi. Ở `sold_out`, kế hoạch có bước `check_seat(VN122)` và kết quả trả về 0 ghế, nhưng bước `book_seat(VN122)` vẫn chạy, vì không có lần gọi model nào giữa các bước để đọc kết quả đó.
3. **ReAct đúng 6/6 nhưng tốn nhiều nhất:** trung bình 5,3 lần gọi và 5.766 token. Con số này gấp khoảng 9 lần Plan-then-Execute và 6 lần mẫu Lai. Ở `valid`, ReAct cần 6 lần gọi và 6.229 token cho cùng chuỗi 5 lời gọi tool mà hai mẫu kia làm với 1 lần gọi và 644 token. Token tăng nhanh hơn số lần gọi, vì mỗi lần gọi gửi lại toàn bộ lịch sử hội thoại.
4. **Mẫu Lai đúng 6/6 với 1,3 lần gọi và 908 token trung bình.** Ở 4 kịch bản không có bước lỗi, chi phí giống hệt Plan-then-Execute. Ở 2 kịch bản có bước lỗi, nó chỉ tốn thêm 1 lần lập lại kế hoạch.
5. **Thời gian chạy không phản ánh đúng chi phí.** Plan-then-Execute và Lai ở `trap` có cùng một lần gọi giống hệt (695 token) nhưng mất 11,2 giây so với 1,6 giây, do model miễn phí dùng chung tài nguyên. So sánh nên dựa trên số lần gọi model và token.
6. **Kịch bản `trap` không bẫy được Qwen.** Cả 3 mẫu đều chọn VN122 thay vì VJ610 rẻ hơn nhưng bay chiều. Vì vậy cột "lời gọi bị chặn = 1" ở mỗi mẫu đều là lần chặn `pay` ở `needs_approval`. Chặn `book_seat` vi phạm ràng buộc chỉ được kiểm qua test với model giả.

**Kết luận từ số liệu:** trên bộ kịch bản này, mẫu Lai có độ đúng bằng ReAct với chi phí gần bằng Plan-then-Execute. Plan-then-Execute phù hợp khi môi trường ổn định và cần duyệt trước. ReAct phù hợp khi không đoán trước được các bước, và chấp nhận chi phí cao hơn.

## 9. Khó khăn và cách xử lý

| Khó khăn                                                                                                                                     | Cách xử lý                                                                                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model trong `.env` ban đầu không còn bản miễn phí (404); endpoint của trường không truy cập được từ ngoài                                    | Thử từng lựa chọn bằng request thật trước khi viết code, chọn Qwen3.8-27B miễn phí trên OpenRouter (mục 2.1)                                                                                           |
| Quota 50 request/ngày, một vòng đánh giá tốn khoảng 50                                                                                       | Mọi việc phát triển dùng model giả. Script đánh giá lưu từng lần chạy, chạy lại thì bỏ qua lần đã xong, gặp lỗi hết quota trong ngày thì dừng hẳn                                                      |
| Nhà cung cấp giới hạn lượt gọi bất chợt (429)                                                                                                | Client tự thử lại 3 lần. Vẫn lỗi thì ghi `infra_error`, không tính là agent thất bại, lần sau chạy lại. Không đổi sang model khác để giữ so sánh công bằng                                             |
| Test phát hiện lặp thất bại dù harness đúng                                                                                                  | Nguyên nhân nằm ở test: `[msg] * 5` tạo 5 tham chiếu tới cùng một message, LangGraph gộp các message trùng id nên agent chỉ thấy 1 lời gọi. Sửa bằng cách tạo message mới với id riêng cho mỗi lời gọi |
| Review phát hiện câu hỏi bàn giao sai ở các lý do dừng `replans_exhausted`, `plan_rejected`, `invalid_plan` (đều hỏi "nới giờ hay nới giá?") | Viết test đỏ trước, rồi thêm câu hỏi riêng cho từng lý do dừng                                                                                                                                         |
| Review phát hiện lỗi 401/404 (sai key, sai model) bị tính là `infra_error`, nên bị chạy lại mãi                                              | Chỉ bắt 429, timeout, mất kết nối, 5xx. Lỗi cấu hình làm chương trình dừng hẳn                                                                                                                         |
| Câu hỏi duyệt thanh toán có thể nêu nhầm booking; mẫu Lai mất trace lỗi khi kế hoạch mới bị từ chối                                          | Lưu đúng booking bị từ chối (`refused_booking`); nối thêm lý do từ chối vào `feedback` thay vì ghi đè                                                                                                  |

## 10. Hạn chế và hướng phát triển

| Hạn chế                                                                                                                                             | Hướng phát triển                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| K = 1: chưa kết luận được về độ ổn định                                                                                                             | Chạy thêm lần lặp bằng `evaluate(..., reps=3)` khi có quota hoặc ở trong mạng trường                   |
| Kịch bản ngắn (khoảng 5 lời gọi tool); phát hiện lặp và ngân sách chưa từng kích hoạt với model thật                                                | Thêm kịch bản dài hơn: nhiều chặng, nhiều ràng buộc, tool trả lỗi mơ hồ                                |
| `trap` không bẫy được Qwen                                                                                                                          | Thiết kế bẫy khó hơn, ví dụ ràng buộc xuất hiện muộn trong hội thoại, hoặc nhiều chuyến gần ngưỡng giá |
| Ưu thế của mẫu Lai một phần do cách chọn kịch bản (`transient_error`, `sold_out`)                                                                   | Thêm kịch bản mà việc lập lại kế hoạch không giúp được, để đo chi phí thừa của mẫu Lai                 |
| Chưa có phát hiện bế tắc (agent đổi tool liên tục nhưng không tiến triển)                                                                           | Đo một đại lượng tiến triển, ví dụ số ràng buộc đã thoả, qua N vòng                                    |
| Chỉ thử Qwen, tắt chế độ suy nghĩ                                                                                                                   | So sánh thêm khi bật chế độ suy nghĩ, và với Gemma 4 trên server của trường                            |
| Ở mẫu Lai, mỗi kế hoạch mới đều qua người duyệt, nên trường hợp xấu nhất tốn tới 7 lần gọi model chỉ để lập kế hoạch (vẫn bị chặn bởi ngân sách 10) | Gộp giới hạn số kế hoạch thành một bộ đếm chung                                                        |

## 11. Kết luận

- **Đã làm đủ 3 yêu cầu của bài:**
  - tool mockup và 4 lớp harness, thêm ngân sách và phát hiện lặp, cài một lần và dùng chung;
  - 3 mẫu ReAct, Plan-then-Execute, Lai chạy qua cùng một điểm vào `run()`;
  - đánh giá 18 lần chạy với Qwen3.8-27B.
- **Kết quả:**
  - ReAct đúng 6/6, tốn 5.766 token trung bình;
  - Plan-then-Execute đúng 4/6, tốn 640 token, sai ở 2 kịch bản thế giới thay đổi giữa chừng nhưng đều bàn giao an toàn;
  - Lai đúng 6/6, tốn 908 token.
- **23 test tự động** kiểm các hành vi của harness mà model thật chưa kích hoạt trong lần đánh giá: chặn đặt vé sai ràng buộc, phát hiện lặp, ngân sách, không tin lời model báo xong.

## Phụ lục: chạy lại

```bash
uv sync
cp .env.example .env            # điền OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
uv run pytest -q                # 23 test, model giả, không tốn quota
uv run python -m flight_agent.demo
uv run python -m flight_agent.evaluate
```
