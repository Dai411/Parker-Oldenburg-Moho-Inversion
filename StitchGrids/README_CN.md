## Why do we need to stitch?

[English Version](./README.md)

### 背景：三组数据来源

我们的研究涉及三组重力数据：

1. **Parker正演结果**：通过剥除沉积物层得到的"正确"重力异常（已经过地层修正，非自由空气异常）
2. **陆地重力异常数据**：地面测量的重力数据（自由空气异常）
3. **自由空气异常**：卫星/船测数据（自由空气异常）

注意：.asc 文件需要用相同的投影坐标系。

### 拼接的必要性

#### 1. **数据基准不一致**
   - 陆地和海洋数据的获取方法完全不同，理论上都参考自由空气异常基准
   - Parker正演结果已经过地层修正，需要与原始重力异常统一到同一基准

#### 2. **数据重叠区域的翘边问题**
   - 陆地和海洋数据在沿岸地带有重合区域
   - 直接拼合会产生不连续或步长跳跃（边界不匹配）
   - 需要平滑过渡或权重融合

#### 3. **数据范围和分辨率差异**
   - 陆地数据和海洋数据的空间范围和量纲可能存在差异
   - 分辨率不一致（陆地可能高分辨率，海洋可能相对粗糙，或反之）

#### 4. **反演区域需要规则化**
   - 陆海拼合后得到不规则形状的网格
   - 后续莫霍面反演需要规则的矩形网格（便于FFT运算和正则化约束）
   - **解决方案**：用下载的规则范围自由空气异常数据（如ICGEM等）填充边界和空缺区域

#### 5. **数据质量和分辨率优化**
   - 拼合后局部分辨率不一致的问题
   - 通过后续反演中的**低通滤波**和**Tikhonov正则化**等手段自动平衡

### 💡 核心原理梳理

**简言之**：拼接的目的是为反演创建一个**统一基准、规则范围、适当分辨率**的重力异常网格。
这不仅解决了多源数据的不兼容性，也为后续的 Moho 深度反演提供了稳定的输入。


---

### 新增：在 Laplacian 域使用模型约束（可选）

当存在可靠的大尺度模型（例如 ICGEM 的 BouguerModelled.asc）时，建议在合并并插值后的 Laplacian（L_FilledLaplacian.asc）上应用模型约束，然后再进行频率域反演。

- 原理：ICGEM 模型分辨率较低但能很好描述大尺度趋势。若直接将合并的 Laplacian 反演回 Bouguer 场，反演过程可能因频谱处理导致全场“平均化”——高值被拉低、低值被抬高。将模型约束应用在源域能钉住低频/背景分量，保留观测数据在局部（核心区）的高频信息。

- 建议流程位置：在 `3_merge_and_interpolate_dual.py`（生成 `L_FilledLaplacian.asc`）之后，`5_frequency_inversion.py` 之前运行模型约束脚本（`7_Laplacian_with_model_constraint.py`），该脚本输出 `BouguerLaplacianConstrained.asc`，随后将其传入 `5_frequency_inversion.py` 进行反演。

- 推荐示例命令：
  ```bash
  python 7_Laplacian_with_model_constraint.py \
    --laplacian L_FilledLaplacian.asc \
    --our L_FilledLaplacian.asc \
    --model BouguerModelled.asc \
    --output BouguerLaplacianConstrained.asc \
    --transition-sigma 20 --safety-padding 10 --boundary-width 5
  ```

- 参数建议（基于 cellsize=300m 的示例）：
  - `--transition-sigma`: 10–50（像素）；10≈3km，50≈15km，视模型分辨率与观测差异而定。
  - `--safety-padding`: 5–20（像素），定义核心区保持观测信息，不被模型覆盖。
  - `--boundary-width`: 3–10（像素），用于估计边界层的模型偏差。

- 可视化与检查项：
  - 边界层差异（model - laplacian）地图与统计（均值、标准差）。
  - 过渡带权重场与横断面剖面图，检查混合是否平滑。
  - 运行 `5_frequency_inversion.py` 前后比较未约束与约束后反演结果的差异（全场与局部）。

- 风险提示：
  - 若模型包含系统误差或参考面不同，过度依赖模型可能掩盖真实局部信号。建议保守选择过渡带宽及权重，并先检查 boundary_diff 统计值。

