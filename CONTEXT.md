# Flight Booking Agent

Agent đặt vé máy bay (BTVN#3) dùng để so sánh ba mẫu thiết kế agent trên cùng một harness và cùng một bộ kịch bản.

## Language

### Agent và harness

**Agent**:
Model chọn bước tiếp theo lúc chạy, trong giới hạn của harness. `Agent = Model + Harness`.

**Harness**:
Phần code bao quanh model, quyết định một lời gọi tool có được phép chạy không và tác vụ đã thật sự xong chưa.
_Avoid_: framework, wrapper

**Ràng buộc** (Constraints):
Yêu cầu của người dùng (tuyến bay, ngày, giờ khởi hành tối đa, giá tối đa), lưu thành dữ liệu để prompt được sinh ra từ nó và code kiểm tra theo nó.
_Avoid_: yêu cầu, điều kiện

**Tiêu chí hoàn thành**:
Quy tắc kiểm bằng code, đọc lại booking từ hệ thống: có booking đã thanh toán và thoả mọi ràng buộc. Lời model nói "xong rồi" không bao giờ được tính.
_Avoid_: model báo xong

**Kiểm quyền**:
Bước harness chạy trước khi thực thi tool, chặn lời gọi vi phạm ràng buộc hoặc vượt thẩm quyền.

**Bàn giao** (Handoff):
Gói thông tin trao cho người khi agent dừng bất thường: đã làm tới đâu, đã thử gì, và một câu hỏi cụ thể người nhận trả lời được trong 30 giây.

**Người duyệt** (Approver):
Bên phê duyệt kế hoạch hoặc hành động vượt hạn mức. Trong demo là người thật, khi đánh giá là luật viết bằng code.

**Ngân sách**:
Giới hạn cứng số lần gọi model của một lần chạy.

**Phát hiện lặp**:
Dừng khi cùng một cặp (tool, args) lặp lại trong vài vòng gần nhất.

### Mẫu thiết kế

**ReAct**:
Model quyết định từng bước một, sau mỗi kết quả quan sát.

**Plan-then-Execute**:
Model viết toàn bộ kế hoạch trong một lần gọi, người duyệt, rồi code chạy tuần tự các bước mà không gọi lại model.

**Lai** (Hybrid):
Như Plan-then-Execute, nhưng khi một bước thất bại hoặc bị chặn thì harness gọi model lập lại kế hoạch kèm kết quả vừa quan sát, tối đa N lần.
_Avoid_: ReAct + Plan, replanning agent

**Lập lại kế hoạch** (Replan):
Một lần model viết kế hoạch mới trong mẫu Lai, do harness kích hoạt.

### Đánh giá

**Kịch bản** (Scenario):
Một bộ dữ liệu giả cố định (chuyến bay, ghế, lỗi tool) cùng ràng buộc và kết quả mong đợi (hoàn thành hoặc bàn giao).
_Avoid_: case, test case

**Lần chạy** (Run):
Một lần chạy một mẫu thiết kế trên một kịch bản, cho ra kết quả do tiêu chí hoàn thành quyết định, kèm số lần gọi model, token và thời gian.
