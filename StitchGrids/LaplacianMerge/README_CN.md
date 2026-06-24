# LaplacianMerge 工作流 — 合并海/陆 Bouguer 异常网格 (概要)

[English Version](./README.md)

该目录包含用于在 Laplacian 域合并海/陆重力网格的脚本（顺序按 0 -> 7 编号）。技术细节与理论依据见同目录下的 `Document2.md`。  
总体思路：先在原始网格上分别计算 Laplacian（避免缝合线在人为差异处产生奇异），按优先规则（海优先）合并 Laplacian，补洞（插值填充），再在频率域反演得到最终 Bouguer 场。

## 目录中脚本概览（执行顺序）
- 0_read_asc_header.py  
  - 作用：小工具 / 示例，读取 ESRI ASCII header 并打印基本信息（用于检查输入 .asc 文件格网与覆盖范围）。
  - 可选执行。

- 1_geometry_and_mask.py  (Step 1)  
  - 作用：对齐海网格到陆网格（使两者共用同一 global domain Ω），构建优先掩模 m(x,y)（海=1、仅陆=0、无数据=NaN），并生成初始复合场 g0(x,y)（仅用于可视化检查）。
  - 输出：
    - `PriorityMask.asc` — 掩模（mask）文件（用于后续决定 Laplacian 来源）。
    - `InitialComposite_g0.asc` — 初始合成场（可视化用，不作为后续计算输入）。
    - `Sea_Aligned_To_LandGrid.asc` — 将海数据对齐到陆格网后的文件（作为后续步骤输入）。

- 2_laplacian_computation.py  (Step 2)  
  - 作用：在各自原始网格上独立计算离散 Laplacian（5 点差分），并对 NaN 邻域处理（若模板中任一点为 NaN，则该 Laplacian 点设为 NaN）。
  - 输出：
    - `Land_Laplacian.asc` — 在陆网格上计算的 Laplacian（使用陆网格 header）。
    - `Sea_Laplacian_Original.asc` — 海网格原始 Laplacian（用海 header）。
    - `Sea_Laplacian_Aligned.asc` — 将海 Laplacian 对齐到陆网格的结果（用于合并）。

- 3_merge_and_interpolate_dual.py  (Step 3 & 4)  
  - 作用：使用优先掩模合并两个 Laplacian（Equation (15)），生成合并后源场 L0；识别因 stencil 边界产生的 gap（Ω_gap）；对 gap 区域做插值填充（支持 `INTERP_MODE` 的 local/global 两种策略）。
  - 中间/输出文件：
    - `L0_MergedLaplacian_BeforeInterp.asc` — 合并后但未插值的 L0（保留以便检查 gap）。  
    - `L_FilledLaplacian.asc` — 插值填充后的统一 Laplacian f(i,j)，供后续频域反演使用。

- 4_gap_check.py  
  - 作用：统一的 gap 检查脚本（多模式：`all`/`summary`/`quality`/`quick`），检查掩模、L_before、L_filled 的统计与质量（邻域平滑性等）。
  - 使用示例：
    - `python 4_gap_check.py --mode all`
    - `python 4_gap_check.py --mode quality`

- (4.5) 7_Laplacian_with_model_constraint.py （可选，建议在存在可靠大尺度模型时使用）
  - 作用：在合并并插值后的 Laplacian（`L_FilledLaplacian.asc`）上应用外部大尺度模型（例如 ICGEM 的 `BouguerModelled.asc`）作为低频/背景约束，从而在频域反演时保留观测数据的高频细节并使外部区域与模型一致。
  - 原理简述：在源域（Laplacian）上混合 model 与 observed Laplacian，识别核心区（保留观测）、过渡带（混合区）与外部区（以模型为主）；在过渡带使用高斯平滑权重进行混合，并在边界层估算并应用模型与观测间的偏移。
  - 使用与维护说明：该脚本支持命令行参数（详见脚本头部注释或运行 `python 7_Laplacian_with_model_constraint.py --help`）。README 中不包含完整长命令示例，建议直接查看脚本的帮助信息以获得最新使用方法与默认值。
  - 输出示例：`BouguerLaplacianConstrained.asc`（供 `5_frequency_inversion.py` 使用），脚本同时在运行时打印边界差异统计与权重范围以供检查。

- 5_frequency_inversion.py  (Step 5)  
  - 作用：将填充后的 Laplacian 在频率域反演得到最终 Bouguer 异常场（使用离散 Laplacian 的传递函数 H(kx,ky)）；支持直接除以 H（默认）或 Tikhonov 正则化。
  - 默认输入/输出（可通过命令行参数覆盖）：
    - 输入：`L_FilledLaplacian.asc` 或 受约束输出（`BouguerLaplacianConstrained.asc`） (`--input`)
    - header 用于尺寸与 cellsize：`Land_Laplacian.asc` (`--header`)
    - 输出（无正则）：`BouguerFinal.asc`
    - 输出（Tikhonov）：`BouguerFinal_Tikhonov.asc`
  - 使用示例：
    - 直接除法（默认）：  
      `python 5_frequency_inversion.py --input L_FilledLaplacian.asc --header Land_Laplacian.asc --output BouguerFinal.asc --no-tikhonov`
    - 使用 Tikhonov：  
      `python 5_frequency_inversion.py --tikhonov --alpha 1e-8 --input L_FilledLaplacian.asc --header Land_Laplacian.asc --output BouguerFinal_Tikhonov.asc`

- 6_validate_bouguer_merge.py  
  - 作用：在陆地仅区间比较原始陆地网格与合并后最终結果，計算差值統計並可選擇性地對 final 網格做系統性偏移校正（`--correct`）。
  - 使用示例：  
    `python 6_validate_bouguer_merge.py --land BouguerLandGrid.asc --sea BouguerSeaGrid.asc --final BouguerFinal.asc --correct --output BouguerFinal_Corrected.asc --threshold 1.0`

---

## 中间文件清单（按产生顺序）
（文件名 — 由哪个脚本生成 — 内容与用途）

1. `PriorityMask.asc` — 由 `1_geometry_and_mask.py` 生成  
   - 内容：掩模 m(x,y)：1=海优先、0=仅陆、NaN=无数据。用于决定合并 Laplacian 的来源。

2. `InitialComposite_g0.asc` — 由 `1_geometry_and_mask.py` 生成（可视化/检查）  
   - 内容：按掩模拼接的初始重力场 g0（并非合并结果用于反演，仅便于检查覆盖与断层）。

3. `Sea_Aligned_To_LandGrid.asc` — 由 `1_geometry_and_mask.py` 生成  
   - 内容：把海网格数据对齐到陆网格坐标系（后续海 Laplacian 或海数据对齐使用）。

4. `Land_Laplacian.asc` — 由 `2_laplacian_computation.py` 生成  
   - 内容：在陆原始网格上计算的离散 Laplacian，NaN 在 stencil 任何点为 NaN 时继承。

5. `Sea_Laplacian_Original.asc` — 由 `2_laplacian_computation.py` 生成  
   - 内容：在海原始网格上计算的 Laplacian（海的 header/尺寸）。

6. `Sea_Laplacian_Aligned.asc` — 由 `2_laplacian_computation.py` 生成  
   - 内容：将海 Laplacian 对齐到陆格网后的数组（用于合并）。

7. `L0_MergedLaplacian_BeforeInterp.asc` — 由 `3_merge_and_interpolate_dual.py` 生成（保存合并前状态）  
   - 内容：按掩模合并后的 Laplacian 源 L0，gap 处为 NaN。用于检查 gap 分布与后续插值。

8. `L_FilledLaplacian.asc` — 由 `3_merge_and_interpolate_dual.py` 生成  
   - 内容：对 L0 中的 gap 区域进行插值（global 或 local）并填充的统一 Laplacian f(i,j)。此文件作为频域反演的输入。

9. `BouguerLaplacianConstrained.asc` — 由 `7_Laplacian_with_model_constraint.py`（可选）生成  
   - 内容：在 Laplacian 源域对合并结果应用外部大尺度模型并混合后的受约束源格（用于 Step 5 的反演）。

10. `BouguerFinal.asc` / `BouguerFinal_Tikhonov.asc` — 由 `5_frequency_inversion.py` 生成  
    - 内容：由插值后的 Laplacian 在频率域反演得到的最终 Bouguer 异常网格（无正则 / Tikhonov 版本）。

11. `BouguerFinal_Corrected.asc`（可选） — 由 `6_validate_bouguer_merge.py --correct` 生成  
    - 内容：若在陆地仅区发现系统性偏移且要求校正，则将偏移加回 final 并保存校正后的网格。

---

## 运行要求（依赖）
- Python 3.x  
- numpy  
- scipy （scipy.signal.convolve2d、scipy.interpolate.griddata、scipy.ndimage 等）  
- 注意：脚本中有默认的硬编码目录 `C:/.../BouguerAnomaly/StitchGrids`，在执行前建议修改脚本顶部的 `data_dir` / `output_dir` 或将当前工作目录切换到包含 .asc 文件的目录，或优先使用脚本的 CLI 参数（若脚本支持）。

## 常见参数说明
- INTERP_MODE (`3_merge_and_interpolate_dual.py`)：'local'（推荐，大网格时较快）或 'global'（小网格或稀疏 gap）。  
- PADDING (`3_merge_and_interpolate_dual.py`)：local 模式时环绕 gap 的扩展邻域大小（网格点数）。  
- --tikhonov / --alpha (`5_frequency_inversion.py`)：是否使用 Tikhonov 正则化以及 alpha 值（若启用，alpha 典型取 1e-6 ~ 1e-8；脚本默认 alpha=1e-10）。

## 注意事项
- 路径硬编码：多数脚本在文件顶部使用硬编码路径，运行前请修改为你本地的数据路径或改为通过命令行传参。  
- 文件格式的一致性：各脚本对 nodata 的约定不同（有的用 -9999，有的根据 header nodata_value），运行前请确保输入 .asc 的 nodata 与脚本一致，或调整脚本以统一 nodata 处理。
- 已发现代码问题（建议修复）：

## 推荐的典型执行流程（最少人工干预）
1. 准备：把 `BouguerLandGrid.asc` 与 `BouguerSeaGrid.asc` 放到指定 data 目录，修改脚本中的路径或切换工作目录。  
2.（可选）检查 header：`python 0_read_asc_header.py` 或直接用 `1_geometry_and_mask.py` 的 header 读取功能。  
3. 生成掩模并对齐海格网：`python 1_geometry_and_mask.py`  
4. 在原始网格上分别计算 Laplacian：`python 2_laplacian_computation.py`  
5. 合并 Laplacian 并插值补洞（根据数据大小选择 local/global）：修改 `INTERP_MODE` 与 `PADDING` 后运行 `python 3_merge_and_interpolate_dual.py`  
6. 检查插值质量：`python 4_gap_check.py --mode all`（查看统计与平滑性）  
7. 可选（推荐在有大尺度模型时）：在 Laplacian 源域应用模型约束：`python 7_Laplacian_with_model_constraint.py --laplacian L_FilledLaplacian.asc --model BouguerModelled.asc --output BouguerLaplacianConstrained.asc`  
8. 频域反演回到 Bouguer 场（可选 Tikhonov）：
   - 无正则：`python 5_frequency_inversion.py --input L_FilledLaplacian.asc --header Land_Laplacian.asc --output BouguerFinal.asc --no-tikhonov`  
   - 带正则：`python 5_frequency_inversion.py --tikhonov --alpha 1e-8 --input L_FilledLaplacian.asc --header Land_Laplacian.asc --output BouguerFinal_Tikhonov.asc`  
9. 验证并（可选）根据陆地仅区执行偏移校正：`python 6_validate_bouguer_merge.py --land BouguerLandGrid.asc --sea BouguerSeaGrid.asc --final BouguerFinal.asc --correct --output BouguerFinal_Corrected.asc --threshold 1.0`  

---

## 参考
- 请参阅本目录下的 `Document2.md`：完整数学与实现细节（拉普拉斯算子、插值、频域反演与 Tikhonov 正则化）。

---
