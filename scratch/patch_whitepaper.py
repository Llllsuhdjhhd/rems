import sys

path = r"c:\Users\40575\Desktop\project\rems\readme.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

# Normalize line endings for matching
normalized = content.replace("\r\n", "\n")

# ──────────────────────────────────────────────────────────────
# Edit 1: Replace §4.1
# ──────────────────────────────────────────────────────────────
old_41 = (
    "### 4.1 残影（Shadow）与未完成事件的精准剥离\n"
    "\n"
    "- **残影**：存留在系统缓冲区内、未经整理的原始文本素材。\n"
    "- **未完成事件库**：存储已开启但未闭环的逻辑对象。\n"
    "- **复杂输入的精准剥离案例**：\n"
    "  - 当系统接收到输入：*\u201c我去跑了步，然后就回家了。回家后我打扫了卫生，然后继续进行A项目的研发，大概1小时后，看了一会儿小说\u201d*。\n"
    "  - 判定与剥离逻辑：大模型扫描到\u201cA项目的研发\u201d时，通过检索命中未完成事件库中的历史挂起事件（例如几天前开启的 A 项目逻辑实体）。此时，系统必须精准划开边界：将\u201c打扫卫生\u201d与\u201c看小说\u201d剥离出来作为全新的基本事件封装；将\u201c继续研发A项目 1小时\u201d单独抽离，归入库中已存在的 A 项目未完成事件，更新其完成进度与状态。这种分离机制保证了宏观长期任务与微观即时琐事的物理隔离。"
)

new_41 = (
    "### 4.1 残影（Shadow）——统一代谢缓冲区\n"
    "\n"
    "- **残影（Shadow）**：系统**唯一**的代谢缓冲区，存储一切尚未封存入库的原始文本片段，不区分来源类型。其内容实际涵盖两类来源，但统一以连续文本形式存储：\n"
    "  1. **文本截断尾巴**：当前素材中语义尚未闭合的结尾部分（边界模型无法从此处切出完整事件）；\n"
    "  2. **逻辑未闭环片段**：已开始叙述但缺少关键结果或后续的内容（如进行中的任务、挂起的事项）。\n"
    "- **边界检测两路输出**：`BoundaryDetectionSkill` 仅输出两类结果——\n"
    "  - **completed_events**：语义上已闭环的片段，直接送入封存流程生成基本事件；\n"
    "  - **remaining_shadow**：一切未闭环内容（文本截断尾巴与逻辑挂起片段均归入此路），统一写回残影缓冲区，等待下一轮输入续写。\n"
    "- **剥离案例**：\n"
    "  - 输入：*\u201c我去跑了步，然后就回家了。回家后我打扫了卫生，然后继续进行A项目的研发，大概1小时后，看了一会儿小说\u201d*。\n"
    "  - 剥离逻辑：边界模型将\u201c打扫卫生\u201d与\u201c看小说\u201d识别为已闭环片段直接封存；\u201c继续研发A项目1小时\u201d因缺乏阶段性结果，归入残影缓冲区。下一轮若有新的 A 项目进展输入，残影中的挂起片段将参与合并，待语义闭环后封存为基本事件。"
)

# ──────────────────────────────────────────────────────────────
# Edit 2: BoundaryDetectionSkill step description
# ──────────────────────────────────────────────────────────────
old_bd = (
    "  1. `BoundaryDetectionSkill`：只做序号剥离、残影与未完成事件判定，**不输出**摘要或角色；"
)
new_bd = (
    "  1. `BoundaryDetectionSkill`：只做序号剥离与残影判定，输出 `completed_events`（已闭环）与 `remaining_shadow`（未闭环，含逻辑挂起片段）两路，**不输出**摘要或角色；"
)

# ──────────────────────────────────────────────────────────────
# Edit 3: Force-seal → physical redline
# ──────────────────────────────────────────────────────────────
old_force = (
    "则尝试悬置为未决片段在缓冲区挂起，不急于生成事件。**底线兜底触发**：若该模糊信息的积累长度已经越过红线界限（逾越 `msg_len` 的 1.2 倍），为防止内存与计算爆炸，系统不再等待，**强制立刻剥离并生成事件**。"
    "是否为该事件附带「可疑（Suspicious）」角色占位 **由下游角色抽取与 `RoleService` 仲裁决定**（当 `EventEnrichmentSkill` 无法产出合规角色或仲裁判定主体身份不确定时，注册 `is_suspicious=True` 的临时占位角色），而不在长度阈值触发点硬性置位——避免把「物理长度越线」误等同于「身份不明」。"
)
new_force = (
    "则将该片段归入残影缓冲区悬置，不急于生成事件。**物理红线兜底**：当残影总长突破 `physical_redline` 时，系统触发强制压缩——截取残影尾部至安全水位，并对仍超限的连续片段执行强制封存。"
    "是否为该事件附带「可疑（Suspicious）」角色占位 **由下游角色抽取与 `RoleService` 仲裁决定**（当 `EventEnrichmentSkill` 无法产出合规角色或仲裁判定主体身份不确定时，注册 `is_suspicious=True` 的临时占位角色）。"
)

# Apply
errors = []
if old_41 not in normalized:
    errors.append("§4.1 original text not found")
if old_bd not in normalized:
    errors.append("BoundaryDetectionSkill line not found")
if old_force not in normalized:
    errors.append("force-seal text not found")

if errors:
    print("ERRORS:", errors)
    sys.exit(1)

normalized = normalized.replace(old_41, new_41)
normalized = normalized.replace(old_bd, new_bd)
normalized = normalized.replace(old_force, new_force)

# Write back with original CRLF line endings
output = normalized.replace("\n", "\r\n")
with open(path, "w", encoding="utf-8") as f:
    f.write(output)

print("Done. All 3 edits applied successfully.")
