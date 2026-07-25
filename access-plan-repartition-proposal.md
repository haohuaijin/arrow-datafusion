# ParquetAccessPlan 感知的 Repartitioning — Issue 与 PR 描述草稿

> 分支:`access-plan-repartition`(commits `35359dc42` + `1f692936a`)
> 提交上游前需翻译为英文;结构已按 DataFusion 的 issue/PR 模板组织。
> 建议流程:先发 Issue 征求设计反馈(cc 参与过 #14754 的 @alamb、@xudong963、@mertak-synnada),1-2 周后带基准提 PR。

---

## Issue 草稿

**标题**:`ParquetSource repartitioning ignores ParquetAccessPlan: partition skew and useless file opens for index-based scans`

### 问题描述(Is your feature request related to a problem?)

`ParquetAccessPlan` 是 DataFusion 官方推荐的外部索引接入方式(见博客 [Using External Indexes to Accelerate Queries on Apache Parquet](https://datafusion.apache.org/blog/2025/08/15/external-parquet-indexes/) 和 `advanced_parquet_index.rs` 示例):外部索引把命中行做成 per-file 的 access plan 挂在 `PartitionedFile.extensions` 上,scan 只解码选中的 row group / 行。

但 repartitioning 对 access plan **完全不感知**。`ParquetSource` 没有覆写 `FileSource::repartitioned`,走的是默认 `FileGroupPartitioner` 的字节等分。当 access plan 是稀疏的(索引场景的常态),这造成两个问题:

1. **分区倾斜,解码串行化**。选中行所在的 row group 在文件里往往是聚集的(典型:`SELECT * ... ORDER BY ts DESC LIMIT n` 在按时间排序的数据上,top-N 全在文件头部的少数 row group)。字节等分后这些热 row group 落进 1-2 个分区,其余分区无事可做——`target_partitions` 形同虚设。
2. **空转的文件打开**。分到的字节范围里全是被 plan 跳过的 row group 的分区,仍然会打开文件、解析 footer 和 page index,然后什么都不解码。

**实测(生产形态,OpenObserve 日志检索)**:`SELECT * FROM logs WHERE pod = '...' ORDER BY _timestamp DESC LIMIT 100`,tantivy 外部索引把 44 个文件剪到 2 个、精确选中 100 行(分布在 6 个 row group);884MB 的文件被字节等分成 11 个 84MB 分区后,40% 的解码量集中在同一个分区,8/12 分区空转,有效并行度 ~4。单线程解码 840ms,实际 wall 只压到 220ms。

### 期望的方案(Describe the solution you'd like)

`ParquetSource` 覆写 `FileSource::repartitioned`(这个 hook 正是 #14754 为"各格式自定义切分策略"引入的),当文件带有 `ParquetAccessPlan` 时**按被扫描的 row group 切分**:

- 每个被扫描的 row group 是一个工作项,权重使用估算扫描字节数:`Scan` 取文件平均 row-group bytes,`Selection` 取 selection 覆盖比例 × 平均 row-group bytes,再计入固定页面解码成本;
- 工作项按权重做 LPT 装箱到 ≤ `target_partitions` 个分区;被跳过的 row group 不进任何分区;
- 同一文件可出现在多个分区,每个分区携带一个子 plan(保持原 row group 总数,非本分区的 row group 置 `Skip`)——`ParquetOpener` 现有的 plan 校验与消费逻辑无需任何改动;
- 无 plan 的文件以文件 bytes 计权并按占比切成 byte ranges,与 planned row-group 工作项共享同一个分区预算,混合场景不丢并行度;
- `repartition_file_min_size` 保持公开配置语义:作用于全部输入文件的总 bytes;达到阈值后允许单个输出块小于该值;
- **排序保证**:声明了 output ordering 时,每个输出分区中会产出行的工作项只来自单一原始 group、按(文件顺序 × row group 升序)排列——有序流的子序列仍有序,分区级排序声明成立,`SortPreservingMergeExec` 语义不变;零输出的 validation-only plan 可能附加到分区但不影响行序;有效组数 ≥ `target_partitions` 时与 `FileGroupPartitioner` 一样放弃切分。

行为纯增量:没有 access plan 的 scan 走原路径,零变化。

### 已考虑的替代方案(Describe alternatives you've considered)

- **给热 row group 的字节范围做感知切分**:计划期不知道 row group 的字节偏移(需要读 footer),放弃;子 plan 方式不依赖字节偏移。
- **在应用层自己切分 FileScanConfig**:可行(我们目前就这么做),但所有 `ParquetAccessPlan` 用户(如 deletion vectors、外部索引)都会遇到同样的问题,应当在 `ParquetSource` 内解决。

---

## PR 描述草稿

**标题**:`feat: ParquetAccessPlan-aware repartitioning for parquet scans`

### Which issue does this PR close?

- Closes #XXXXX(上面的 issue)

### Rationale for this change

外部索引通过 `ParquetAccessPlan` 把 scan 剪到稀疏的行集合,但默认的字节等分 repartitioning 对 plan 不感知:热 row group 集中在少数分区(解码串行化),其余分区打开文件后发现自己的 row group 全被跳过(空转)。本 PR 让 `ParquetSource` 按 plan 实际扫描的 row group 切分,以估算扫描字节数为权重均衡分区。

### What changes are included in this PR?

- 新模块 `datafusion/datasource-parquet/src/repartition.rs`:
  - 工作项构建:planned 文件 → 每个被扫描 row group 一项(权重 = 估算扫描 bytes);unplanned 文件 → 以文件 bytes 计权并按占比切成字节块(混合场景保留并行度);
  - LPT 装箱;保序模式下分区名额按有效组权重分配、组内装箱,产出行的工作不跨组(每分区是有序组的子序列,排序声明保持成立);
  - 分区物化:同文件相邻工作项合并——row group 项合成一个子 plan(保持原 row group 总数、`fully_matched` 状态,其余置 `Skip`),相邻字节块合并 range;全 Skip/零行 selection 仅保留一次用于 opener 合法性校验;
  - 回退/不切分条件:无 plan / 任一文件已带 byte range / 保序且有效工作组数 ≥ `target_partitions` / 压缩文件时走默认路径;总输入 bytes 小于 `repartition_file_min_size` 时保留原分组,避免保序 fallback 绕过阈值。
- `ParquetSource` 覆写 `FileSource::repartitioned`(#14754 引入的 hook),优先尝试 plan 感知切分,否则走默认字节切分。
- 21 个单元测试:切分正确性(跳过的 row group 不贡献解码工作、子 plan 长度与 `fully_matched` 保留)、LPT 均衡与分区上限、保序有效组计数、非保序跨组、混合场景字节切分(range 不重叠且覆盖完整)、`min_size` 契约和空计划校验;另有 2 个真实 Parquet 执行回归测试。
- criterion 基准 `datafusion/core/benches/parquet_scan_access_plan.rs`。

### 基准结果

36MB zstd parquet(16 个 row group,不可压缩的 256B payload 列),access plan 选中 128 个散布行,`target_partitions = 8`,每次迭代重建 exec 并断言解码行数:

| 场景 | byte_range(旧) | access_plan(新) | 提升 |
|---|---|---|---|
| clustered(选中行集中在前 1/4 row group,top-N over 时序数据的形态) | 8.38 ms | 4.57 ms | **1.83×** |
| uniform(选中行均匀分布) | 13.58 ms | 10.11 ms | 1.34× |

真实工作负载(OpenObserve,tantivy 索引 + `SELECT * ... ORDER BY _timestamp DESC LIMIT 100`,2 文件 6 个热 row group):端到端查询 215ms → 125ms。

### Are these changes tested?

是。`repartition.rs` 21 个单元测试覆盖切分、混合文件权重、均衡、排序保持、空计划校验与全部回退条件;另有真实 Parquet 集成测试验证非法全 Skip plan 和非法零行 selection 在重分区后仍报错;基准在 `--test` 模式下对每种切分断言解码行数与选中行数一致。

### Are there any user-facing changes?

无 API 变化。行为变化仅影响挂了 `ParquetAccessPlan` 的 scan(此前它们被字节盲切);无 plan 的 scan 路径完全不变。

---

## 附:评审时可能被问到的设计点(预先准备的回答)

1. **`repartition_file_min_size` 如何解释?** 保持 DataFusion 公开配置的现有语义:它是是否执行 file-scan repartition 的总输入 bytes 门槛,不是每个输出 chunk 的最小尺寸。达到门槛后,planned/unplanned 工作项再按估算扫描 bytes 共同均衡。
2. **同一文件多分区 → footer/page index 重复解析?** 与现状字节切分相同(它同样让每个分区各自打开文件);metadata cache(#18470 方向)可进一步缓解。
3. **`Scan` 权重为什么用平均 row-group bytes 估算?** 计划期没有 per-row-group 大小(在 footer 里);估算偏差只影响均衡度,不影响正确性。`Selection` 使用自身 selector 覆盖比例进一步缩放,不依赖可选的文件行数统计。
4. **为什么不加 config 开关?** 行为只在有 plan 时改变,而挂 plan 本身就是显式的用户行为;若评审坚持,可加 `datafusion.execution.parquet.repartition_using_access_plan`(默认 true)。
5. **排序正确性论证**:每个分区中会产出行的部分 = 单一有序组的(文件顺序 × row group 升序)子序列;有序序列的子序列有序,所以每个分区满足声明的 ordering,组间归并由 `SortPreservingMergeExec` 完成。validation-only plan 严格输出零行,因此即使为保留 opener 校验而附加到其他分区也不改变实际行序。
