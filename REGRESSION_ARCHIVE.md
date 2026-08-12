# 异常存档库 / Regression Archive

用于代码修改、问题追踪和生产复跑验收。每项记录一个已定位、修复中或已修复的逻辑冲突场景，
状态必须以条目正文和专项审计为准；修改相关代码时需验证已修复场景不被复现。

---

## 索引

| # | 日期 | 文件 | 标题 |
|---|------|------|------|
| 1 | 2026-07-17 | postprocess_textgrids.py | MFA尾静音被snap回词而非合并到标点 (jie2) |
| 2 | 2026-07-17 | postprocess_textgrids.py | Phase 1 静音合并 vs Phase 3 Rule 3 顺位冲突 (er4) |
| 3 | 2026-07-17 | postprocess_textgrids.py | 跨词界 eps 被 xmax 上限检查漏掉 (le5) |
| 4 | 2026-07-17 | postprocess_textgrids.py | 短NVV强制CTC导致前词被裁短 (ti2/BREATHING) |
| 5 | 2026-07-17 | postprocess_textgrids.py | 词首前拉：能量谷底检测 (na2) |
| 6 | 2026-07-17 | postprocess_textgrids.py | 静音段延伸+end-trimming回截冲突 (ji2) |
| 7 | 2026-07-23 | postprocess_textgrids.py | pinyin_phones首音素与词界间隙 (ru2→r, NVV邻接) |
| 8 | 2026-07-27 | postprocess_textgrids.py | 标点间隙中<sp0>因has_punct跳过合并→mid_sp误报 |
| 9 | 2026-07-27 | postprocess_textgrids.py | 标点后<spN>未被标点吸收→mid_sp误报 |
| 10 | 2026-07-27 | postprocess_textgrids.py | Hanzi tier拼音残留：NW对齐+过度宽松匹配→CJK被跳过 (qie4/切) |
| 11 | 2026-07-27 | postprocess_textgrids.py | cjk_mismatch/hanzi_pinyin 检查在路径决策之后执行→文件未被重定向 |
| 12 | 2026-07-27 | postprocess_textgrids.py | CTC snap重叠修复产生倒置interval + 尾静音未被最后标点吸收 (xia4/ta1) |
| 13 | 2026-07-27 | postprocess_textgrids.py, finalize_textgrids.py | NVV 大小写不一致：MFA 小写 laughter 不被识别/包裹 → pinyin_phones 残留小写 |
| 14 | 2026-07-28 | postprocess_textgrids.py | Hanzi tier NVV 括号污染：<_build_hanzi_tier alpha 分支未 strip <> → label 残留 `<SURPRISE-WA>` |
| 15 | 2026-07-28 | postprocess_textgrids.py | _extract_word_chars 拆分 NVV 尖括号：<LAUGHTER> → [<, LAUGHTER, >] |
| 16 | 2026-07-28 | run_pipeline.py | MFA --fine_tune 默认开启→边界漂移+对齐失败 |
| 17 | 2026-07-28 | postprocess_textgrids.py, ctc_prealign.py | NVV改写+首词标点+音素断层+轨道不同步+连字符替换 |
| 18 | 2026-07-29 | ctc_prealign.py | NVASR 情绪标签后残留开头标点 → hanzi 首词为标点 |
| 19 | 2026-07-29 | postprocess_textgrids.py | _snap_to_ctc 丢弃开头静音 → hanzi 缺失 <sp1> interval |
| 20 | 2026-07-29 | postprocess_textgrids.py | strip_edge_punctuation 尾随标点过度剥离 → 句尾 。！？ 全部丢失 |
| 21 | 2026-07-29 | postprocess_textgrids.py, pipeline_utils.py | is_punct 误判 <spN> 为标点 → strip_edge_punctuation 开头静音被吸收进首词 |
| 22 | 2026-07-29 | postprocess_textgrids.py | MFA 对齐偏差→snap 修正后标点被 <spN> 替换→被吞标点恢复 |
| 23 | 2026-07-29 | run_pipeline.py | step_normalize_punct 缺少 NVV 连字符保护 → QUESTION-EI 被拆成 QUESTION，EI |
| 24 | 2026-07-29 | ctc_prealign.py | ASR 空输出先写 TextGrid 再检测 → 留下孤立的空 TextGrid |
| 25 | 2026-07-29 | postprocess_textgrids.py | _inject_punctuation 重叠判断 + _punct_before 范围 + 恢复匹配策略 |
| 26 | 2026-08-04 | postprocess_textgrids.py | pinyin_phones 声母独占→韵母消失 (ch/zh/sh → 缺 final) |
| 27 | 2026-08-04 | postprocess_textgrids.py | words tier 区间重叠 — 相邻词时间边界交叉 |
| 28 | 2026-08-04 | postprocess_textgrids.py | 倒置 interval — xmin > xmax |
| 29 | 2026-08-04 | postprocess_textgrids.py | 极短内容词 (< 30ms) 物理不可能 |
| 30 | 2026-08-04 | postprocess_textgrids.py, adjust_ctc_boundaries.py, pipeline_utils.py | 参考文本模糊子串匹配 — 纯英文 CTC 锚点标定风险分析 |
| 31 | 2026-08-04 | normalize_english_tokens.py, ctc_prealign.py | 英文 CTC 锚点三重修复 — 碎片化 / 合并错误 / NVV 误判 |
| 32 | 2026-08-05 | postprocess_textgrids.py | 英文词自引用单音素 — pp 仅 1 个 phone 但 dict 期望 2+ |
| 33 | 2026-08-05 | postprocess_textgrids.py | 英文词音素不足 — pp 音素数 < dict 期望 |
| 34 | 2026-08-05 | postprocess_textgrids.py | pinyin_phones 轨道间隙 — 相邻 phone 不连续 |
| 35 | 2026-08-05 | postprocess_textgrids.py | words 轨道间隙 — 非静音词间空洞 |
| 36 | 2026-08-05 | postprocess_textgrids.py | 轨道系统性不连续 — >10% interval 有间隙 |
| 37 | 2026-08-05 | postprocess_textgrids.py | en_mfa_windows 重复词覆盖 → 英文词音素全部丢失 (FULL_WORD_AS_PHONE) |
| 38 | 2026-08-05 | postprocess_textgrids.py | MFA/CTC 边界 2ms 死区 → 文本重叠 11% |
| 39 | 2026-08-05 | postprocess_textgrids.py | MFA 帧精度间隙 5-30ms → words/hanzi/pp 三轨道间隙 |
| 40 | 2026-08-05 | postprocess_textgrids.py | English phone 边界未 snap → phone 与 word 不对齐 |
| 41 | 2026-08-05 | postprocess_textgrids.py | CTC 锚点膨胀 → 异常长词 (le5 5.6s) + 无检测 |
| 42 | 2026-08-05 | postprocess_textgrids.py | pinyin_in_text 误报 — <sp1> 被正则 \b[a-z]+[1-5]\b 匹配为拼音 |
| 43 | 2026-08-05 | dict/cmudict.dict | CMUdict 缺缩写词条目 → English MFA/G2P 无法对齐 → pp 自引用 (数据修复) |
| 44 | 2026-08-05 | postprocess_textgrids.py | 声母占比异常 — MFA phone boundary 过晚 → 声母>60%词长，韵母被压缩 |
| 45 | 2026-08-05 | postprocess_textgrids.py | 韵母被词边界拉长 — Phase 5 尾 phone 无条件扩展 → 韵母 900ms+ |
| 46 | 2026-08-05 | postprocess_textgrids.py | pp tier phone↔punct 结构重叠 — _inject_punctuation 标点与 phone 重叠 |
| 47 | 2026-08-05 | postprocess_textgrids.py | init_only_phone 检测改进 — 区分零声母/自引用/真缺失韵母 |
| 48 | 2026-08-05 | postprocess_textgrids.py | silence_boundary_split 误报 — MFA 真 phone 被误判为 silence 分界 |
| 49 | 2026-08-05 | postprocess_textgrids.py | english_single_phone / english_phone_deficit 用中文拼音词典查英文词 → 误报/死代码 |
| 50 | 2026-08-05 | postprocess_textgrids.py | Interval.mark 属性不存在 → postprocess 全线崩溃 |
| 51 | 2026-08-05 | run_pipeline.py | ctc_pretg_adj 空目录未回退 → postprocess 找不到 txt/lab |
| 52 | 2026-08-05 | postprocess_textgrids.py | CTC 宽跨标点锚点 → _inject_punctuation 碎片喷溅 + _fix_overlapping_boundaries 阈值拒绝 → 词级大重叠 |
| 53 | 2026-08-05 | postprocess_textgrids.py | 拼音-汉字全局错位 — STT 错误→pypinyin 上下文污染→级联位移 (pinyin_displacement) |
| 54 | 2026-08-05 | adjust_ctc_boundaries.py | NVV 边界夹制豁免 → 实词边界自由越过 NVV → 标点区间倒置 (nvv_clamp_skip) |
| 55 | 2026-08-05 | postprocess_textgrids.py | 全静音 phone list → MFA split 用静音边界切分声母/韵母 → 垃圾时间 (all_silence_mfa_split) |
| 56 | 2026-08-05 | postprocess_textgrids.py | Phase 5 首音素回拉不对称 → 前词末 phone 未延伸 → pp 轨道间隙 (first_phone_snap_asymmetry) |
| 57 | 2026-08-05 | postprocess_textgrids.py | MFA/CTC 混合决策独立 → 词间空隙 > 20ms → words 轨道间隙 (mixed_decision_word_gap) |
| 58 | 2026-08-05 | postprocess_textgrids.py | handle_unexpected_silences 标记 <sp1-3> 后不合并 → mid_sp 误报 (sp1_3_flag_not_merge) |
| 59 | 2026-08-05 | postprocess_textgrids.py | CTC 替换部分成功 → 拼音片段被合并守卫排除 → hanzi 拼音残留 (partial_ctc_merge_guard) |
| 60 | 2026-08-05 | postprocess_textgrids.py | fix_short_words 仅覆盖虚词 → 实词 < 50ms 不被修复 → short_word 过滤 (content_short_word_unfixed) |
| 61 | 2026-08-05 | ctc_prealign.py | 参考文本模式英文词被 NVASR tokenizer 拆碎 → 最终 TextGrid 英文变形 (ref_text_english_tokenizer_mangle) |
| 62 | 2026-08-05 | postprocess_textgrids.py | _normalize_word_spellings 三通道修复: 用原始 .txt 覆盖 tokenizer 损坏的英文词 (ref_text_english_correction_in_postprocess) |
| 63 | 2026-08-06 | postprocess_textgrids.py | CTC 锚点错位导致参考文本与 hanzi tier CJK 字符序列不匹配 → text_order_mismatch 过滤 (ctc_anchor_text_order) |
| 64 | 2026-08-06 | postprocess_textgrids.py | english_phone_deficit 读取错误的 Interval.mark → 英文音素不足检测永久失效（已修复） (english_deficit_mark_regression) |
| 65 | 2026-08-06 | postprocess_textgrids.py, run_pipeline.py | 单文件后处理异常只写 report 不返回非零 → 管线误报成功（已修复） (postprocess_error_exit_masking) |
| 66 | 2026-08-06 | postprocess_textgrids.py | 派生 tier 同步异常被静默吞掉 → 过期/不同步 tier 仍可能进入输出（已修复） (silent_derived_tier_sync_failure) |
| 67 | 2026-08-06 | postprocess_textgrids.py, run_pipeline.py | text_order_mismatch 用 CTC 归一化文本做参考 → 检查失效 → 改用原始 txt (text_order_wrong_ref_source) |
| 68 | 2026-08-06 | ctc_prealign.py, normalize_english_tokens.py, postprocess_textgrids.py, run_pipeline.py | 参考文本未贯穿 CTC→MFA→后处理链，ASR 文本覆盖权威文本并造成严重词面/字符错位；CTC batch/token 映射还有错配风险（已修复） (reference_text_ctc_anchor_authority) |
| 69 | 2026-08-06 | normalize_english_tokens.py, run_pipeline.py, adjust_ctc_boundaries.py | 参考文本权威半修复残留：normalize_en Pass 2 自拼英文词、外层退出码和 adjusted CTC 丢失 _ref（已修复） (reference_authority_followthrough) |
| 70 | 2026-08-06 | postprocess_textgrids.py, run_pipeline.py | `filter_suspicious: false` 仍被 `tier_discontinuity` 影响；自然停顿被稀疏 `pinyin_phones` 轨道误判为系统性断层（已修复，待 Linux 全量复跑） (semantic_tier_discontinuity_gate) |
| 71 | 2026-08-06 | postprocess_textgrids.py | MFA HMM 软边界导致 pinyin_phones 层韵母→声母重叠 40-100ms，新增 _fix_pp_phone_overlaps 去重叠 (pp_phone_overlap_deoverlap) |
| 72 | 2026-08-06 | pipeline_utils.py, ctc_prealign.py, run_pipeline.py | cn2an 把拼音声调数字写成汉字，18,000 个 lab 全量 OOV（修复草案已写入，暂停复审/复跑） |
| 73 | 2026-08-06 | pipeline_utils.py, postprocess_textgrids.py | MFA unknown 被判为标点，CTC Rule 0 永远不可达（修复草案已写入，暂停复审/复跑） |
| 74 | 2026-08-06 | postprocess_textgrids.py | 0 pinyin vs N CJK 只写 warning，结构崩溃仍可进入 ok（修复草案已写入，暂停复审/复跑） |
| 75 | 2026-08-06 | postprocess_textgrids.py | 用派生 raw_text 对比派生 hanzi，空==空造成 CJK 假通过（修复草案已写入，暂停复审/复跑） |
| 76 | 2026-08-06 | run_pipeline.py | 分片 MFA 只看退出码且丢弃日志，缺失 stem 静默成功（部分草案，仍有未闭环项） |
| 77 | 2026-08-06 | run_pipeline.py | postprocess 以已有 aligned 为分母，139 条缺失未进入报告（修复草案已写入，暂停复审/复跑） |
| 78 | 2026-08-06 | pipeline_utils.py, run_pipeline.py | staging 非版本化、filtered 复用且 NAS 混入 2,150 条陈旧结果（未闭环，暂停修复） |
| 79 | 2026-08-06 | run_pipeline.py, postprocess_textgrids.py | tone_mapping.json 默认写仓库 output，未随本次结果交付（修复草案已写入，暂停复审/复跑） |
| 80 | 2026-08-06 | pipeline_utils.py, ctc_prealign.py, recover_ctc_labs.py | 旧 CTC bundle 含零时长词仍被视为可恢复（校验草案已写入，待单条 CTC 重跑） |
| 81 | 2026-08-06 | ctc_prealign.py, run_pipeline.py, normalize_english_tokens.py | ria 合并未在 lab、tokens、CTC TextGrid 三份载体中原子同步（未闭环，暂停修复） |
| 82 | 2026-08-06 | run_pipeline.py, recover_ctc_labs.py | normalize marker 生命周期与 MFA 入口校验不完整，可跳过过期 bundle（未闭环，暂停修复） |
| 83 | 2026-08-06 | run_pipeline.py | MFA TextGrid 仅字符串探测、分片启动异常未完整收敛（未闭环，暂停修复） |
| 84 | 2026-08-06 | pipeline_utils.py | 裸词 unk 被一律视为 MFA unknown，可能误伤真实英文词（未修复） |
| 85 | 2026-08-06 | hecheng_ria_0805.yaml, ctc_prealign.py | nvv_enabled=false 与“需从音频发现 NVV”要求存在配置冲突（待确认并修复） |
| 86 | 2026-08-06 | 数据集, run_pipeline.py | 18,000 条音频仅 17,999 条参考文本，1 条静默退回 ASR（待补参考或隔离） |
| 87 | 2026-08-06 | run_pipeline.py, 操作流程 | 只运行 postprocess --overwrite 不会重建已损坏 CTC/MFA，上游污染原样继承（操作风险） |
| 88 | 2026-08-06 | audit_strict_ok.py, run_pipeline.py | strict-ok v3.1 独立发布门禁：通过集必须有磁盘级来源证据（待真实 canary） |
| 89 | 2026-08-06 | align_english_mfa.py, postprocess_textgrids.py | 旧英文空 phones 被 CMU/G2P/均分 fallback 伪装为可用音素（已用严格来源契约封闭，待真实数据验收） |
| 90 | 2026-08-06 | 英文源数据, 旧 CTC 缓存 | 英文批次分母不一致与混合缓存污染风险：54,000 WAV / 53,998 txt / 2 缺参考（待隔离准备） |
| 91 | 2026-08-06 | align_english_mfa.py, audit_strict_ok.py | strict-en-mfa-v1 来源链缺口：局部拒绝、运行异常、segment 身份与证据原子性（修复中） |
| 92 | 2026-08-06 | hecheng_english_mfa.yaml, prepare_hecheng_english_ctc_ready.py | 最新英文配置与 7,204 原样复制 + 46,586 规范化 + 208 重跑的隔离准备风险（修复中） |
| 93 | 2026-08-06 | 旧 CTC TextGrid, ctc_prealign.py | 旧 writer 将 `item [2]` 写进 words tier 头部，标准 parser 得到 words=0（已定位，待隔离规范化方案） |
| 94 | 2026-08-06 | prepare_hecheng_english_ctc_ready.py, normalize_english_tokens.py | v3 首版严格解析器与真实旧 grammar 不一致，且规范化误用 token end（已阻断生产，修复中） |
| 95 | 2026-08-06 | 旧 ctc_pretg, pad_silence_edges.py, prepare_hecheng_english_ctc_ready.py | 历史 padding 原地污染 CTC 时间轴，真实 v3 inspect 仅得 9/70/53,919（已阻断生产，重新定界中） |
| 96 | 2026-08-06 | prepare_hecheng_english_ctc_ready.py, verify_hecheng_english_ctc_ready_v4.py, ctc_prealign.py | v4 首版验证语义与真实 rerun namespace 冲突，独立证据可自证（已阻断生产，修复前不得 prepare/GPU） |
| 97 | 2026-08-06 | ctc_prealign.py, normalize_english_tokens.py, prepare_hecheng_english_ctc_ready.py | CTC manifest 在英文归一化前冻结，最终 bundle 与 provenance 自相矛盾（P0，已阻断声学 canary/生产） |
| 98 | 2026-08-06 | ctc_prealign.py, prepare_hecheng_english_ctc_ready.py, verify_hecheng_english_ctc_ready_v4.py | encoder 60ms 网格时长冒充 WAV 轴，fresh CTC 必被 v4 严格域校验拒绝（P0，已阻断声学 canary/生产） |
| 99 | 2026-08-06 | ctc_prealign.py, prepare_hecheng_english_ctc_ready.py, verify_hecheng_english_ctc_ready_v4.py | 仅固定模型路径而未固定实际模型文件树，CTC 声学 provenance 可被同路径替换（P1，生产 prepare 前必须修复） |
| 100 | 2026-08-06 | ctc_prealign.py | blank-run pause 未移除 4 个 query frame，停顿时间整体偏移约 240ms（P0，已阻断声学 canary/生产） |
| 101 | 2026-08-06 | ctc_prealign.py | all-GPU 父合并静默跳过碰撞/坏 manifest，并可能提前合入 shard marker（P0，已阻断声学 canary/生产） |
| 102 | 2026-08-07 | ctc_prealign.py | reference-only `--no-nvv` 仍允许 ASR 内容污染 required sidecar |
| 103 | 2026-08-07 | run_pipeline.py, ctc_prealign.py | `run_pipeline.py` 未支持 `--all-gpus` 多卡 CTC 推理（待实施） |
| 104 | 2026-08-07 | run_pipeline.py | MFA Popen OSError 引用未初始化变量 |
| 105 | 2026-08-07 | streaming_pipeline.py | 父进程 MFA jobs 计算值未传递到子进程 |
| 106 | 2026-08-07 | streaming_pipeline.py, launch_8gpu.py | 硬编码 `--force --overwrite` 覆盖配置禁止 |
| 107 | 2026-08-07 | ctc_prealign.py | all-GPU shard 合并存在 TOCTOU 竞态窗口 |
| 108 | 2026-08-07 | streaming_pipeline.py | 批量上传共享目录导致跨批次文件覆盖 |
| 109 | 2026-08-07 | streaming_pipeline.py | Checkpoint 在上传完成前标记成功 |
| 110 | 2026-08-07 | streaming_pipeline.py, run_pipeline.py | `strict_ok` 输出路径与流式上传器不匹配 |
| 111 | 2026-08-07 | streaming_pipeline.py | 断点续跑丢失批次级进度 |
| 112 | 2026-08-07 | ctc_prealign.py | all-GPU preflight namespace 拒绝 `.ctc_run_receipt.json` |
| 113 | 2026-08-07 | run_pipeline.py | `step_prealign` 陈旧 `.TextGrid` 存在性短路导致分母缩小 |
| 114 | 2026-08-07 | run_pipeline.py | `step_adjust_ctc` 陈旧 `.TextGrid` 存在性短路 |
| 115 | 2026-08-07 | run_pipeline.py | `step_link_ctc` 可解析旧 manifest 直接短路 |
| 116 | 2026-08-07 | run_pipeline.py | `--skip-to` 静默追加不属于当前模式的步骤 |
| 117 | 2026-08-07 | streaming_pipeline.py | `--scan-only` 在 full 模式下执行破坏性 trim |
| 118 | 2026-08-07 | run_pipeline.py | `--validate` 失败不影响最终退出码 |
| 119 | 2026-08-07 | run_pipeline.py | `_run_direct` 丢弃子进程 return code |
| 120 | 2026-08-07 | streaming_pipeline.py | `_prefetch_worker` 忽略 copy failures 并允许失败 batch 进入 process queue |
| 121 | 2026-08-07 | streaming_pipeline.py | `StreamingPipeline.run()` 失败 batch 仍被推入 upload queue |
| 122 | 2026-08-07 | streaming_pipeline.py | `run_batch` / `run_pipelined_batch` 返回 None 导致主调无法感知失败 |
| 123 | 2026-08-07 | streaming_pipeline.py | staged CPU upload 对 rsync/copy 失败仅告警并返回 True |
| 124 | 2026-08-07 | streaming_pipeline.py | batch_ctc_ready 缺失音频数据集被 skip 而不进入 fail_list |
| 125 | 2026-08-07 | run_pipeline.py | 非 strict 模式输出路径无 run-specific 隔离 |
| 126 | 2026-08-07 | run_pipeline.py | 无统一 config schema 校验 |
| 127 | 2026-08-07 | postprocess_textgrids.py, pipeline_utils.py | 权威标点/连字符英文投影与 phone 越界 |
| 128 | 2026-08-07 | ctc_prealign.py, run_pipeline.py | CTC 全 GPU 合并隔离与输入副本安全 |
| 129 | 2026-08-10 | audit_strict_ok.py, verify_strict_ok.py | strict manifest 未绑定 `pipeline-run-receipt-v2` |
| 130 | 2026-08-12 | streaming_pipeline.py, launch_8gpu.py, launch_multi_gpu.sh | 批量 GPU/CPU 资源未统一规划，旧分片入口可能竞争 strict artifacts |
| 131 | 2026-08-12 | run_pipeline.py, pad_silence_edges.py | 分 speaker 子目录的 pre-CTC WAV 被根目录扫描漏掉，54k 任务在 pad_silence 阶段误报空分母 |
| 132 | 2026-08-12 | run_pipeline.py, ctc_prealign.py, pipeline_utils.py, postprocess_textgrids.py | 先前 MFA 对齐失败问题的统一根因、修复链路与验收索引 |

### 索引完整性与非 Case 章节

截至 2026-08-12，Case 索引已覆盖 Case 1–132，每个编号各出现一次；Case 标题与正文
均可按同一编号定位。除 Case 条目外，文档还包含以下纳入索引范围的专题章节：

| 章节 | 内容 |
|------|------|
| `2026-08-10 当前全量运行总结` | 本次 54k 全量运行的输入隔离、CTC、音频、中文/英文 MFA、后处理、strict audit、过滤和发布证据 |
| `项目管线结构 / Pipeline Architecture` | full、ctc_ready、streaming 等模式的入口、输入输出和步骤关系 |
| `修改点汇总` | 跨 Case 的代码修改、验证方法和当前遗留风险汇总 |

索引中的日期、文件和标题是导航信息；每个 Case 的实际状态、验证结果和未闭环项以对应
正文为准。`filtered`、`rejected`、`missing` 或“待修复/待复跑”条目不代表成功通过。

---

## 2026-08-10 当前全量运行总结

本节只记录 `/mnt/nvme3/mfa_workspace_54k_full_20260810` 这次当前全量运行的证据，
并明确区分“代码已实现”和“本次运行已验证”。运行配置为
`/mnt/local_E/MFA_Pause/repo/configs/hecheng_ria_fresh.yaml`；所有输出使用
run-specific 目录（例如 `strict_ok_runs/20260810T032955Z_2109063/`），不把历史
workspace 或 NAS 旧结果作为本次输入。

### 代码已实现的契约

- source/input isolation：CTC 输入来自 `data_dir`/reference sidecar，CTC 运行在
  run-local staging/merge namespace；`--no-nvv --all-gpus` 的 argv、分片 receipt 和
  `/mnt/nvme3/mfa_workspace_54k_full_20260810/ctc_pretg/manifest.json` 提供来源记录。
- CTC reference mode：reference 文本是 required sidecar、CTC words 与后续 MFA 的内容
  权威；`--no-nvv` 仅禁止自由解码新增 NVV，不删除 reference 中已有 NVV。
- `padded_audio/` 与 `audio_16k/` 用途分离：前者保留 48 kHz padded WAV 作为音频侧/最终
  交付语义，后者是 MFA 使用的 16 kHz WAV；二者不是同一目录的替代品。
- partial MFA、English MFA provenance、strict audit、filtered integrity、run-specific
  output 和 versioned publish 均已有代码路径/验证器；有效结果必须保留 English 对齐、
  NVV 标签、句首 `<sp>`/`sp` 和标点，filtered 结果永远不是成功结果。

### 本次运行已验证的结果

- CTC：53998 输入、53998 输出、0 failed；运行参数含 `--no-nvv --all-gpus`。
- 音频：`padded_audio/` 有 53998 个 48 kHz WAV；`audio_16k/` 有 53998 个 16 kHz MFA WAV。
- 中文 MFA：53998 expected、53392 produced、606 missing；缺失统一归类为
  `missing_mfa_alignment`，见
  `/mnt/nvme3/mfa_workspace_54k_full_20260810/mfa_logs/20260810T024653Z_2045521/mfa_output_manifest.json`。
- English MFA：135003 expected segments、131700 produced、3303 rejected；281617 verified
  words、4004 rejected words。来源链见
  `/mnt/nvme3/mfa_workspace_54k_full_20260810/en_phones/en_alignment_manifest.json`。
- 后处理：53998 rows，17959 ok、36039 filtered、0 error；报告为
  `/mnt/nvme3/mfa_workspace_54k_full_20260810/strict_ok_runs/20260810T032955Z_2109063/output/postprocess_report.jsonl`。
- `strict-ok-v3.2`：17640 ok、36358 rejected，互斥且并集为 53998；另有 313
  out-of-bounds candidates、6 reference semantic sequence mismatch。strict manifest 为
  `/mnt/nvme3/mfa_workspace_54k_full_20260810/strict_ok_runs/20260810T032955Z_2109063/output/strict_ok_manifest.json`。
- 发布清单 `/mnt/nvme3/mfa_workspace_54k_full_20260810/strict_ok_runs/20260810T032955Z_2109063/output/.publish_manifest.json`
  含 17640 个顶层 TextGrid、English provenance 和元数据，但不含 WAV；它不是 NAS
  音频复制完成证明。root 复制监控快照仅记录截至快照的活动标记：16 workers，
  `max_current=042817_弹幕互动_互动游戏.wav`；不能由此换算完成率、目标总数或 ETA，
  本次记录未复核、控制或操作复制进程。

### 门禁状态与残余风险

当前代码要求 `pipeline-run-receipt-v2`，但这次旧运行 artifact 使用
`pipeline-run-receipt-v1`（证据：
`/mnt/nvme3/mfa_workspace_54k_full_20260810/ctc_pretg/.ctc_run_receipt.json`）。因此
不能宣称旧运行通过新的 v2 发布门禁；上述 strict/filtered 数字是运行审计结果，
不是 v2 publish approval。发布前仍需以 v2 receipt 重新满足 source/eligible/exclusion
守恒，并再次确认 English provenance、NVV、句首静音与标点保留。

## 项目管线结构 / Pipeline Architecture

### 入口

```
scripts/run_pipeline.py          — 主控脚本，调度全部步骤
scripts/streaming_pipeline.py    — 流式批处理（NAS→本地SSD→回传）
```

### 三种管线模式

| 模式 | 步骤序列 | 适用场景 |
|------|---------|---------|
| `full` | trim → resample → prealign → normalize_punct → normalize → normalize_ria → normalize_en → adjust → align → align_en → postprocess | 原始 WAV 无任何预处理 |
| `nvrasr_fallback` | prealign → pad_silence → normalize_punct → normalize → normalize_ria → normalize_en → resample → adjust → align → align_en → postprocess | 音频已预裁剪，只需重新跑 NVASR |
| `ctc_ready` | link → pad_silence → normalize_punct → normalize → normalize_ria → normalize_en → resample → adjust → align → align_en → postprocess | 已有 NVASR CTC 输出，跳过 trim + prealign |

---

### 步骤详解（full 模式执行顺序）

#### 1. trim — 音频预处理
- **脚本**: `scripts/trim_silence_batch.py`
- **功能**: 切除句内长静音（>1.0s），规范首尾静音至 0.5s
- **输入**: 原始 WAV
- **输出**: `workspace/audio/`

#### 2. resample — 重采样至 16kHz
- **代码**: `run_pipeline.py::step_resample_for_mfa`
- **功能**: 多线程重采样至 16kHz 单声道（MFA 要求）
- **输入**: `audio/`
- **输出**: `workspace/audio_16k/`

#### 3. prealign — NVASR CTC 预对齐
- **脚本**: `scripts/ctc_prealign.py`
- **功能**: NVASR（SenseVoice-Small）CTC 强制对齐，产出词级锚点
- **输入**: `audio_16k/` + 可选参考文本
- **输出**: `workspace/ctc_pretg/`

  **内部处理流程**:
  ```
  NVASR Encoder → CTC logits
    ├─ blank-frame NVV bias（提升 NVV token 检测）
    ├─ pause→ellipsis 注入（空白帧 ≥ 阈值 → 插入 … token）
    ├─ NVV bracket 转换（[Surprise-oh] → SURPRISE-OH）
    ├─ ria 变体还原（rui4 ya4 → ria）
    ├─ 数字→中文（cn2an）
    ├─ 标点规范化（normalize_punct_inline）
    └─ Token 级 CTC 时间戳解码
         │
         ├─ .lab              MFA 语料（pinyin+NVV）
         ├─ .TextGrid         CTC 锚点（words tier）
         ├─ _tokens.jsonl     逐词时间戳
         ├─ _punct.json       标点时间戳
         ├─ _text_cn.txt      中文文本（raw_text tier 用）
         └─ _text_raw.txt     ASR 原始输出（含 NVV 标签）
  ```

  **输出后处理**（`main()` 末尾自动调用）:
  - `_normalize_punct`: ASCII→CJK 标点映射 + 相邻合并 + 非白名单替换。**含 NVV 连字符保护（Case 17-E）**
  - `_normalize_numerals`: 仅在人类文本中执行阿拉伯数字→中文；不得改写 MFA lab 的拼音声调数字
  - `_normalize_english`: 英文 token 规范化（`scripts/normalize_english_tokens.py`）
  - `_normalize_ria`: ria 音译还原（仅 ctc_ready/nvrasr_fallback 模式）

  **关键函数**:
  | 函数 | 作用 |
  |------|------|
  | `make_patched_inference` | 批量推理，返回 CTC 对齐结果 |
  | `chars_and_pinyin` | 汉字→拼音映射（pypinyin） |
  | `nvv_to_mfa` | `[Surprise-oh]` → `SURPRISE-OH` |
  | `write_textgrid` | 写 CTC 锚点 TextGrid |
  | `clean_unsupported_punct` | 过滤白名单外标点字符 |
  | `_normalize_punct` | 标点规范化（含 NVV 连字符保护） |
  | `_normalize_numerals` | 人类文本数字→中文；CTC transcript 只做三方校验或从可信 tokens 恢复 |

#### 4. normalize_punct — 标点规范化
- **代码**: `run_pipeline.py::step_normalize_punct`
- **功能**: ASCII 标点→CJK，合并相邻标点，同步更新 `_punct.json`
- **注意**: 若 prealign 已运行输出后处理，此步骤操作的是已有文件

#### 5. normalize — 数字规范化
- **代码**: `run_pipeline.py::step_normalize_text`
- **功能**: 只对 `_text_cn.txt` 等人类文本执行 cn2an；`.lab` 不做字符级数字转换，只能在 tokens 与 CTC TextGrid 一致后从 tokens 重建

#### 6. normalize_ria — ria 音译还原
- **代码**: `run_pipeline.py::step_normalize_ria`
- **功能**: `rui4 ya4` → `ria`；目标契约要求 `.lab`、`_tokens.jsonl`、CTC TextGrid words 三方原子同步（当前尚未闭环，见 Case 81）

#### 7. normalize_en — 英文 token 规范化
- **脚本**: `scripts/normalize_english_tokens.py`
- **功能**: NVASR 拼音碎片→英文原词（如 `li ve` → `live`）

#### 8. adjust — CTC 边界能量修正
- **脚本**: `scripts/adjust_ctc_boundaries.py`
- **功能**: 音频能量分析修正 CTC 锚点边界（词首能量上升→前推，词尾能量下降→后延）
- **输入**: `ctc_pretg/` + `audio_16k/`
- **输出**: `workspace/ctc_pretg_adj/`
- **保护**: NVV token 不被修改；标点同步调整

  **关键函数**:
  | 函数 | 作用 |
  |------|------|
  | `adjust_boundaries` | 主逻辑：能量上升/下降检测 + 边界调整 |
  | `rebuild_textgrid` | 从调整后的 tokens 重建 TextGrid |
  | `process_one` | 单文件处理入口 |

#### 9. validate — MFA 语料验证
- **代码**: `run_pipeline.py::step_mfa_validate`
- **默认跳过**（`skip_validate: true`），MFA align 内部已验证

#### 10. align — MFA 强制对齐
- **代码**: `run_pipeline.py::step_mfa_align`
- **功能**: MFA 使用 NVASR `.lab` 作语料 + CTC TextGrid 作锚点 → 音素级对齐
- **输入**: `ctc_pretg_adj/` + `audio_16k/` + `mfa_ipa.dict`
- **输出**: `workspace/aligned/`（含 words + phones 两个 tier）
- **关键参数**:
  - `--beam 20 --retry_beam 80`：Viterbi 波束宽度
  - `--fine_tune false`：默认关闭（Case 16），CTC 锚点已由 adjust 修正
  - `--fine_tune_boundary_tolerance 0.02`：仅 fine_tune=true 时生效

#### 11. align_en — 英文 MFA 对齐
- **脚本**: `scripts/align_english_mfa.py`
- **功能**: 从 CTC 边界提取英文词段 → English MFA（`english_us_arpa`）对齐
- **输入**: `ctc_pretg_adj/` + `audio_16k/` + `cmudict.dict`
- **输出**: `workspace/en_phones/`（`*_en_phones.json`）

  **关键函数**:
  | 函数 | 作用 |
  |------|------|
  | `find_english_segments` | 扫描 CTC 输出，识别英文词段 |
  | `build_en_corpus` | 构建 English MFA 语料目录 |
  | `build_en_dict` | 构建英文发音词典（CMUdict + G2P fallback） |
  | `run_en_mfa` | 调用 English MFA align |
  | `collect_en_phones` | 收集 English MFA 音素→写入 JSON |

#### 12. postprocess — 后处理（最复杂步骤）
- **脚本**: `scripts/postprocess_textgrids.py`
- **功能**: 从 MFA 对齐结果构建最终 5-tier TextGrid + 质检
- **输入**: `aligned/` + `ctc_pretg_adj/` + `en_phones/` + dicts
- **输出**: `workspace/output/`（通过质检）+ `workspace/filtered/`（未通过）

  **Phase 执行顺序**:
  ```
  Phase 1 — 声学前处理
    ├─ merge_short_silences    短静音合并
    ├─ fix_short_words          短词修复（能量检测）
    └─ 重建 pinyin_phones       （若 merge/fix 修改了边界）

  Phase 2 — 文本修正 + tier 定型
    ├─ silence relabel          <eps>/sil → <spN> 按duration
    ├─ _finalise_textgrid       构建 raw_text/pinyin/hanzi/words/pp
    │   ├─ normalize NVV brackets → <BREATHING>
    │   └─ sp1 normalization
    ├─ 标点-静音交叉校验
    └─ 输出路径选择（output vs filtered 预判）

  Phase 3 — 边界调整（顺序不可变）
    ├─ A. _snap_to_ctc           MFA边界→CTC锚点（权威）
    ├─ B. _refine_boundaries_by_energy  能量微调
    │   └─ 同步 pinyin_phones    ← 修改 words 后立即重建 pp
    └─ C. _inject_punctuation    标点注入 words tier

  Phase 3.5 — 英文音素注入
    └─ _apply_en_phones          英文 MFA 音素→phones tier
        └─ 同步 pinyin_phones    ← 修改 phones 后立即重建 pp

  Phase 4 — 后边界处理（顺序不可变）
    ├─ D.  handle_unexpected_silences  意外静音检测
    ├─ D2. absorb_nvv_trailing         NVV 吞并尾部标点+静音链
    ├─ D3. absorb_silence_into_punct   残余静音→标点吸收
    ├─ D4. strip_edge_punctuation      边缘标点剥离（Case 17-B）
    ├─ ── SYNC ──                      _sync_derived_tiers
    │   └─ 从 words+phones 重建 hanzi + pinyin_phones（Case 17-F）
    ├─ E.  NVV+ellipsis 无条件合并
    ├─ F.  _merge_nvv_ellipsis        能量判断 NVV+省略号合并
    ├─ G.  _extend_word_into_ellipsis  能量判断词延伸入省略号
    └─ ── SYNC ──                      _sync_derived_tiers

  Phase 5 — 最终文本同步 + 质检
    ├─ NVV 边界回退到 CTC 锚点
    ├─ 被吞标点检测 + raw_text 同步删除
    ├─ _normalize_word_spellings      英文词碎片→规范拼写
    │   └─ NVV token 文本保护（Case 17-A）
    ├─ _build_hanzi_tier              从 words + raw_text 重建 hanzi
    ├─ build_pinyin_phones_tier       从 phones + words 重建 pinyin_phones
    │   ├─ 泄漏音素过滤（Case 17-C）
    │   ├─ en_mfa_windows 缺失时 fallback（Case 17-D）
    │   └─ NVV token 自引用音素
    └─ QC 检查（10+ 过滤规则）→ output/ 或 filtered/
  ```

  **Tier 同步架构**（Case 17-F 建立）:
  ```
  words tier  ← 唯一权威数据源
     │
     ├─ hanzi tier       ← 从 words + raw_text 派生
     └─ pinyin_phones    ← 从 phones + words + pinyin_dict 派生
  
  规则: 任何修改 words tier 边界/文本的操作，必须立即调用
        _sync_derived_tiers() 重建 hanzi + pinyin_phones。
        不允许延迟到 Phase 5。"修改谁，同步全部"。
  ```

  **关键函数**:
  | 函数 | Phase | 作用 |
  |------|-------|------|
  | `merge_short_silences` | 1 | 合并短静音段 |
  | `fix_short_words` | 1 | 能量检测修复短词边界 |
  | `_finalise_textgrid` | 2 | 构建 5-tier TextGrid |
  | `_snap_to_ctc` | 3A | MFA→CTC 边界权威对齐 |
  | `_refine_boundaries_by_energy` | 3B | 能量微调词边界 |
  | `_inject_punctuation` | 3C | CTC 标点注入 words tier |
  | `_apply_en_phones` | 3.5 | 英文 MFA 音素注入 phones tier |
  | `absorb_nvv_trailing` | 4-D2 | NVV 吞并尾部标点+静音 |
  | `absorb_silence_into_punct` | 4-D3 | 残余静音→标点吸收 |
  | `strip_edge_punctuation` | 4-D4 | 边缘标点剥离 |
  | `_sync_derived_tiers` | 4-sync | **统一同步：words→hanzi+pp** |
  | `_merge_nvv_ellipsis` | 4-F | 能量判断 NVV+省略号合并 |
  | `_extend_word_into_ellipsis` | 4-G | 能量判断词延伸入省略号 |
  | `_normalize_word_spellings` | 5 | 英文碎片→规范拼写（含 NVV 保护） |
  | `_build_hanzi_tier` | 5 | 从 words 重建 hanzi |
  | `build_pinyin_phones_tier` | 5 | 从 phones+words 重建 pinyin_phones |

---

## 修改点汇总

| ID | 位置 | 修改 |
|----|------|------|
| A | `_snap_to_ctc` ~2519 | Rule 3 绕过: ratio_skip (pattern a+b) |
| B | `_snap_to_ctc` ~2569 | 中间点保护: keep_mfa_end |
| C | `_inject_punctuation` ~1877 | gap kind 扩展: "gap" 可被标点吸收 |
| D | `_snap_to_ctc` ~2541 | Pattern (b) 缩进修复 |
| E | `_snap_to_ctc` ~2598 | 重叠防护: prev_was_silence_extended |
| F | 已移除 | gap_was_merged 被能量分析否决 |
| G | `_snap_to_ctc` ~2526,~2572 | has_trailing_sil: xmax→xmin |
| H | `_snap_to_ctc` ~2484 | NVV 短时长例外 (<100ms 用MFA) |
| I | `_refine_boundaries_by_energy` ~2320 | 词尾能量延伸 + NVV前向延伸 |
| J | `_refine_boundaries_by_energy` ~2320 | 词首前拉：能量谷底检测 |
| K | `_refine_boundaries_by_energy` ~2415,~2623 | 静音段延伸(K1-K3)+punct全范围检查(K5)+延伸保护(K4) |
| Q | `build_pinyin_phones_tier` ~533,~539 | 逻辑修复: 首音素start/末音素end 无条件 snap 到词界 |
| R | `process_one` ~3708 | Phase 3.B 后同步：能量调整 words 后立即重建 pinyin_phones |
| S | `process_one` ~3760 | Phase 3.5 后同步：英语音素注入后立即重建 pinyin_phones |
| T | `_snap_to_ctc` ~3118 | words tier 连续性清理：吸收 ≤5ms 词间间隙 |
| U | `handle_unexpected_silences` ~636-641 | `<sp0>` 无条件合并：`has_punct` 不再拦截 `<sp0>` |
| V | `handle_unexpected_silences` ~668-693 | has_punct 时 `<sp0>` 合并到标点而非前词 |
| W | `step_mfa_align` ~1002, DEFAULT_CFG ~205 | **Case 16**: fine_tune 默认 True→False，tolerance 0.1→0.02 |
| X | `_normalize_word_spellings` ~1374 | **Case 17-A**: NVV token 文本永不改写 |
| Y | `strip_edge_punctuation` ~879 (新增) | **Case 17-B**: 边缘标点剥离函数 |
| Z | `build_pinyin_phones_tier` ~469 | **Case 17-C**: 泄漏音素检测（首音素起点>词起点30%→清空） |
| AA | `build_pinyin_phones_tier` ~521 | **Case 17-D**: en_mfa_windows 缺失时清空 word_phones |
| AB | `_normalize_punct` ~651 (ctc_prealign.py) | **Case 17-E**: 字母间 `-` 不当作标点（NVV 连字符保护） |
| AC | `_sync_derived_tiers` ~879 (新增) | **Case 17-F**: 统一 tier 同步函数 + Phase 4 调用点 |
| W1 | `absorb_nvv_trailing` ~758-851 (新增) | Pass 1: NVV 吸收标点+静音链 (D2 步骤) |
| W2 | `absorb_silence_into_punct` ~854-920 (新增) | Pass 2 (兜底): 标点吸收残余 `<spN>` (D3 步骤) |
| X | `_word_matches` ~1069-1076 | 拼音→英文匹配增加元音约束 (≥2 vowels) |
| Y | `_build_hanzi_tier` ~1173-1298 (重写) | 顺序CJK映射替代NW全局对齐 + 贪心alpha匹配 |
| Z | `_alpha_text_matches` ~1136-1170 (新增) | Alpha token ↔ ref unit 匹配（从旧 _word_matches 抽取） |
| AA1 | `process_one` ~4717-4749 | Hanzi tier 完整性检查移至路径决策之前（修复时序bug） |
| AA2 | `process_one` ~4724-4734 (新增) | 直接 pinyin 残留扫描：`is_pinyin_syllable` on hanzi labels → `hanzi_pinyin` |
| AC | `_snap_to_ctc` ~3317 | 重叠修复后防倒置 interval：`word_end < word_start` 时延伸 word_end |
| AD | `_inject_punctuation` ~2277 | 最后标点保留条件：`m[0] < punct_start` → `m[1] <= punct_start + 0.001` |
| AE | `_NVV_PATTERN` ~87-91 | 添加 `re.IGNORECASE` + 字符类 `[A-Za-z-]` 支持小写 NVV 匹配 |
| AF | `_finalize_textgrid` ~141 | NVV 替换统一大写：`r"<\1>"` → `lambda m: f"<{m.group(1).upper()}>"` |
| AG | `build_pinyin_phones_tier` ~485-488 | NVV 文本规范化：`f"<{...strip('<>').upper()}>"` 防大小写+括号残留 |
| AH | `finalize_textgrids.py` ~50-54 | NVV 规范化：strip+upper 防双重包裹和大小写不一致 |
| AI | `_build_hanzi_tier` ~1246-1278 | Alpha 分支 strip `<>`：`clean_token` 参与匹配和 fallback label，防 hanzi 污染 |
| AJ | `_extract_word_chars` ~1004-1032 | `<` `>` 特殊处理：flush + 开新 group / flush + 闭合，防 NVV 被拆分 |
| AK | `strip_edge_punctuation` ~997-1017 | **Case 20**: 删除尾随标点剥离逻辑（镜像设计错误——开头和尾随不是对称场景） |
| AL | `_inject_punctuation` ~2482-2484 | **Case 20-B**: 最后标点 30ms 地板，防末词 xmax==tier.xmax 时坍缩为零时长 |
| AM | `strip_edge_punctuation` ~983-985 | **Case 21**: 开头剥离增加 `not is_silence()` 检查，防 `<spN>` 被 `is_punct` 误判为标点 |
| AN | `process_one` ~4343,~4366 | **Case 22-A**: Phase 4 前后标点快照比对 → `_swallowed_puncts`（被吞标点追踪） |
| AO | `process_one` ~4665 | **Case 22-B**: 仅对 `_swallowed_puncts` 中确认被吞的标点, 若位置现为 `<spN>` → 替换恢复 |
| AP | `step_normalize_punct` ~680 | **Case 23**: Phase 2 增加 `is_hyphen_in_nvv` 检查，防 NVV 连字符被当作标点替换 |
| AQ | `ctc_prealign.py` ~1193,~1217 | **Case 24**: TextGrid 写入移至空检测之后，避免 ASR 空输出留下孤立的空 TextGrid |
| AR | `_inject_punctuation` ~2346 | **Case 25-A**: else 分支改为 split: word 在 punct 内时不删除, 拆分 punct 为左右两段 |
| AS | `postprocess_textgrids.py` ~4361,~4701 | **Case 25-B**: `_punct_before` 监控扩展至 `，。！？…、；：` |
| AT | `postprocess_textgrids.py` ~4717-4748 | **Case 25-C**: 吞标点恢复从时间重叠匹配改为 CTC 序列顺序匹配（前后邻词定位） |
| AU | `run_pipeline.py` ~238, `postprocess_textgrids.py` ~4942 | **Case 25-D**: bgm_threshold 天花板 `bgm_max_threshold` (默认 0.05)，防 60 分位噪声底被污染 |
| AV | `postprocess_textgrids.py` ~4961 | **Case 25-E**: bgm 检测从整段平均值改为 50ms 帧级别，≥20% 帧超阈值才触发 |
| AW | `postprocess_textgrids.py` ~4819-4888 | **Case 25-F**: 末尾标点强制保留：从前词截取 ≥60ms（弹性取 max(60ms, CTC原始时长)），同步 words/hanzi/pinyin_phones |
| AX | `postprocess_textgrids.py` ~3028-3051 | **Case 25-G**: `_refine_boundaries_by_energy` 标点边界保护：标点起点与词尾重叠 <100ms 且主体在词尾之后 → 阻止 dead silence 延伸 |
| AY | `run_pipeline.py` ~1049-1059, `align_english_mfa.py` ~382-393 | **Case 25-H**: 英文 MFA 词典/G2P 路径修复：pretrained_models fallback 到 PROJECT_ROOT.parent + G2P 失败时写 base dict 兜底 |
| AZ | `postprocess_textgrids.py` ~87-91 | **Case 25-I**: `_NVV_PATTERN` lookbehind/ahead 加 `<>` 排斥，防已包裹 NVV token 被 `_finalise_textgrid` 再次包裹 → pinyin_phones 出现 `<<TOKEN>>` |
| BA | `build_pinyin_phones_tier` ~510-527 | **Case 26-A**: dict 查询前移 + `word_phones` 空时按比例拆分 |
| BB | `build_pinyin_phones_tier` ~600-616 | **Case 26-B**: 三分支替代 `len(dict_phones)==1 or len(word_phones)<=1` |
| BC | `_INIT_FRAC` ~434 | **Case 26-C**: 按声母类型细化拆分比例 (塞音0.20/鼻边0.22/擦音0.28/塞擦0.35) |
| BD | `process_one` QC 段 | **Case 26-D**: 质检过滤 `init_only_phone` — pinyin_phones 仍残留纯声母 |
| CA | `_fix_overlapping_boundaries` (新增) | **Case 27-A**: 轻度边界重叠修复 (内容词split_overlap / punct clip_punct) |
| CB | `process_one` QC 段 | **Case 27-B**: 质检过滤 `overlapping_words` — 修复后仍重叠的进入 filtered/ |
| DA | `process_one` QC 段 | **Case 28**: 质检过滤 `inverted_interval` — xmin > xmax 倒置检测 |
| EA | `process_one` QC 段 | **Case 29**: 质检过滤 `short_word` — 内容词 < 30ms 检测 |

---

## Case 1: MFA 尾静音被 snap 回词而非合并到标点 (jie2)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`, `_inject_punctuation`
**触发样本**: 合成ria_15653, 词 `jie2` (8.03-8.51s)

### 现象

MFA 对齐后词尾的 `<eps>` 段在最终输出中重新被合并回前一词，而不是归入后续标点。

```
MFA 对齐:    jie2[8.03-8.27]  <eps>[8.27-8.52]  shi4[8.52-8.70]
修复前输出:  jie2[8.03-8.51]  dur=475ms  ，[8.51-8.52]
修复后输出:  jie2[8.03-8.27]  dur=240ms  ，[8.27-8.52]
```

### 根因链

1. **Rule 3 误判**: `ctc_dur(495ms) > mfa_dur(240ms) * 2 → use_mfa=False`。CTC 给 jie2 标了 495ms (包含尾静音)，MFA 正确切分 jie2=240ms + eps=250ms。但 2x 比例检查把"MFA 切掉了尾静音"误判为"词被压缩了"。
2. **`has_mfa_phone_evidence` 漏检**: 音素 `ie2` [8.10-8.51] 跨越了争议区 [8.27-8.505]，但检测未纳入。
3. **中间点未保护**: 即使 `use_mfa=True`，当 `end_diff > 0.15` 时取 CTC/MFA 中间点 (8.39)，没有检测尾静音+标点场景。
4. **微间隙合并 `kind` 不匹配**: `_snap_to_ctc` 插入的静音用 `kind="gap"`，但 `_inject_punctuation` 的合并规则仅匹配 `kind="word"`。

### 修改点

**A. Rule 3 绕过 (ratio_skip)**
**B. 中间点保护 (keep_mfa_end)**
**C. gap kind 扩展**
**D. Pattern (b) 缩进修复**
**G. has_trailing_sil xmax→xmin**

### 验证方法

```python
# 预期: jie2 dur ≈ 240ms (非 475ms), 逗号吸收了尾静音
words_tier["jie2"].duration < 0.30
words_tier after jie2: is_silence or is_punct
```

### 关联样本

- `合成ria_15653` jie2 → 逗号
- `合成ria_15653` le5 → 逗号

---

## Case 2: Phase 1 静音合并 vs Phase 3 Rule 3 顺位冲突 (er4)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `merge_short_silences`, `_snap_to_ctc`
**触发样本**: 合成ria_15653, 词 `er4` / `zhong3` (9.58s)

### 现象

MFA 正确切分 `zhong3 + <eps> + er4`，但最终输出 zhong3 350ms、er4 仅 50ms：

```
MFA aligned:  zhong3[9.23-9.39]  <eps>[9.39-9.58]  er4[9.58-9.64] dur=60ms
Phase 1:      zhong3[9.23-9.58]                    er4[9.58-9.64]
修复前:       zhong3[9.23-9.58]  dur=350ms         er4[9.58-9.63] dur=50ms
修复后:       zhong3[9.23-9.39]  dur=160ms         er4[9.39-9.63] dur=240ms
```

### 根因链

1. **Phase 1** `merge_short_silences`: 能量条件满足，`<eps>` 被合入 zhong3
2. **Rule 3**: er4 `ctc_dur(180) > mfa_dur(60) * 2` → snap 到 CTC
3. **重叠防护 (旧)** 将 er4 推后到 prev_end (9.58)，压成 50ms

### 修改点

**E. 重叠防护 prev_was_silence_extended** — 前词因静音延伸时缩短前词而非推后当前词
**F. 移除 gap_was_merged** — 能量分析否决

### 能量验证

```
9.23-9.38s: zhong3 RMS 0.016→0.001（音節结束）
9.38-9.46s: 静音 RMS 0.0002-0.001
9.47s:      er4 起振 RMS 0.031→0.18
```

CTC 锚点 er4 [9.45-9.63] 更接近真实。MFA `ong3` 音素 310ms 过度延伸。

### 关联样本

- `合成ria_15653` zhong3 → er4

---

## Case 3: 跨词界 eps 被 xmax 上限检查漏掉 (le5)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`
**触发样本**: 合成ria_15653, 词 `le5` (6.27s)

### 现象

同 Case 1 pattern (a) — `le5 + <eps> + ，` — 但修复未生效：

```
MFA aligned:  le5[6.36-6.45]  <eps>[6.45-6.77]  kuai4[6.77-6.94]
修复前:       le5[6.27-6.70] dur=425ms  ，[6.70-6.77] dur=75ms
修复后:       le5[6.36-6.45] dur=90ms   ，[6.45-6.77] dur=320ms
```

### 根因

`has_trailing_sil` 的 `iv.xmax <= ctc_end + 0.05` 要求整个静音段在 CTC 范围内。le5 的 `<eps>` [6.45-6.77] 的 `xmax=6.77 > ctc_end+0.05=6.745`（eps 跨到了 kuai4 的区域），条件失败。

### 修改点

**G. `iv.xmax <=` → `iv.xmin <`** — 两处 has_trailing_sil 检查

### 关联样本

- `合成ria_15653` le5 → 逗号

---

## Case 4: 短 NVV 强制 CTC 导致前词被裁短 (ti2 / BREATHING)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`
**触发样本**: 合成ria_13714, 词 `ti2` + `BREATHING` (2.16s)

### 现象

BREATHING (NVV) 强制用 CTC [2.31-2.37] (60ms)，其 start 与 ti2 MFA end (2.37) 重叠，NVV 重叠规则缩短前词：

```
MFA:    ti2[2.16-2.37] dur=210ms   BREATHING[2.37-2.45] dur=80ms
CTC:    ti2[2.13-2.31]             BREATHING[2.31-2.37] dur=60ms
修复前: ti2[2.16-2.31] dur=145ms   BREATHING[2.31-2.37] dur=60ms
修复后: ti2[2.16-2.37] dur=210ms   BREATHING[2.37-2.45] dur=80ms
```

### 根因

Rule 1 对所有 NVV 无条件设 `use_mfa=False`。但 NVASR 的短 NVV 检测 (< 100ms) 可能是噪声误检，CTC 锚点不可靠，反挤占相邻词边界。

### 修改点

**H. Rule 1 — NVV 短时长例外** (~line 2484)

```python
# 修改前: 所有 NVV 无条件 use_mfa=False
if is_nvv_token(mfa_iv.text) or is_english_token(mfa_iv.text):
    use_mfa = False

# 修改后: NVV CTC 时长 < 100ms 时保留 MFA 边界
if is_nvv_token(mfa_iv.text):
    use_mfa = (ctc_end - ctc_start) < 0.10
elif is_english_token(mfa_iv.text):
    use_mfa = False
```

### 修改点

**I. `_refine_boundaries_by_energy` — 词尾能量延伸** (~line 2320)

当词的元音衰减能量延续到紧邻的 NVV 区间（如 BREATHING），用能量分析将词尾延伸到真正的能量下跌点。仅限 NVV，不碰 silence/punct。保护 NVV 最小 40ms。

```
ti2:  MFA end=2.37 → 能量延伸 → 2.41
```

### 关联样本

- `合成ria_13714` ti2 → BREATHING (60ms NVV, ti2 从 2.37 延伸到 2.41)

---

## Case 5: MFA 词边界过晚——能量谷底在词首之前 (na2)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_refine_boundaries_by_energy`
**触发样本**: 合成ria_01251, 词 `na2` (1.51s)

### 现象

MFA 将 `shi4` 拆成 `<eps> + shi4`，导致 `na2` 的 start 被推到 1.51s。能量显示两个音节之间的谷底在 1.455s。

```
修复前: shi4[1.22-1.51]  na2[1.51-1.58] dur=70ms
修复后: shi4[1.22-1.455] na2[1.455-1.58] dur=125ms
```

### 根因

MFA 的 `<eps>` [1.22-1.45] 吞掉了 `shi4` 的声母 `sh`，Phase 1 把它合给了 `zhen1`。剩余 `shi4` 只有韵母 1.45-1.51，`na2` 被推到 1.51。CTC 锚点 (na2 start=1.41) 偏早。能量谷底在 1.455。

### 修改点

**J. `_refine_boundaries_by_energy` — 词首前拉** (~line 2320)

处理相邻两词时，在边界前 120ms 搜索能量谷底。约束：深谷（< 50% 峰值）、局部极小、后跟上升能量、前词 ≥ 80ms、拉动 25-80ms。

### 关联样本

- `合成ria_01251` na2 (start 1.51→1.455)

---

## Case 6: 静音段延伸被 end-trimming 回截 (ji2)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_refine_boundaries_by_energy`
**触发样本**: 合成ria_36502, 词 `ji2` (11.89s)

### 现象

ji2 后面有 250ms 死静音，应延伸到 12.44s。但 end-extension 延伸后被 end-trimming 截回。

```
修复前: ji2[11.89-12.19] dur=300ms
修复后: ji2[11.89-12.445] dur=555ms  he2[12.445-12.505]
```

### 根因链

1. **`is_punct("<eps>")` 返回 True**：`<eps>` 被误分类为标点，end-extension 跳过静音段
2. **onset 阈值 0.004 太高**：检测不到 he2 的轻辅音 /h/ 起振
3. **延伸后 intervals 重复**：`intervals[i+2]`（旧 he2）未删除，与新的 `intervals[i+1]`（移位 he2）重叠
4. **end-trimming 回截**：延伸后 ji2 尾部是静音，end-trimming 检测到尾部无能量 → 截回到 12.24

### 修改点

**K1.** `is_punct` 前先检查 `is_silence`：`if is_punct(next_iv.text) and not is_silence(next_iv.text): continue`

**K2.** onset threshold: `max(baseline * 3.0, 0.0015)` (原 `max(baseline * 4.0, 0.002)`)

**K3.** 延伸时吸收 silence 后标记 `intervals[i+2]` 为零时长占位，末尾过滤掉

**K4.** end-trimming 不变，改为 `_extended_indices` 集合保护被延伸过的词：`if i in _extended_indices: continue`

**K5.** punct 检查范围从 `gap_end+0.05` 扩大到全区间（含后续词）：避免标点落在 silent run 结束点之后被漏掉

**K6.** `_refine_boundaries_by_energy` 新增 `punct_entries` 参数，调用处传入 CTC 标点数据

### 关联样本

- `合成ria_36502` ji2 (end 12.19→12.445)

---

## Case 7: pinyin_phones 首音素与词界间隙 (ru2→r, NVV 邻接导致)

**日期**: 2026-07-23
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `build_pinyin_phones_tier`
**触发样本**: 合成ria_00001, 词 `ru2` (6.805s), pinyin_phones tier interval 54 "r"

### 现象

输出 TextGrid 中 pinyin_phones tier 在 NVV token `<UHM>` 与后邻词 `ru2` 的首音素 `r` 之间存在 5ms 间隙：

```
words tier:        <UHM>[6.250-6.805]  ru2[6.805-6.870]
pinyin_phones:     <UHM>[6.250-6.805]  (5ms gap)  r[6.810-6.840]  u2[6.840-6.870]
hanzi tier:        <UHM>[6.250-6.805]  如[6.805-6.870]
```

words/hanzi tier 连续无间隙，但 pinyin_phones tier 在 6.805-6.810 存在 5ms uncovered gap。
`r` 音素起点 (6.810) 比词界 (6.805) 晚 5ms。

**test_en_mfa 中也存在相同模式的间隙** (`合成ria_37435` 3.925s 处 5ms 间隙)。

### 根因链

1. **CTC 锚点**: `ru2` 原始 CTC 边界 [6.750-6.870]，`UHM` (NVV) 原始边界 [6.390-6.750]
2. **MFA fine_tune**: NVV token 无 MFA 声学模型，fine_tune 让 `UHM` 边界浮动到 [6.250-6.805]，将 `ru2` 词首从 6.750 推到 6.805
3. **MFA 音素对齐**: 在 `ru2` 词内，MFA 将首音素 [ɻ] (对应 pinyin `r`) 对齐到 6.810（比词首晚 5ms）。5ms 在 MFA 10ms 帧精度内，属 fine_tune 边界与音素对齐的残余偏差
4. **_snap_to_ctc** (Phase 3.A): `ru2` 非 NVV/English，MFA 边界被信任 (`use_mfa=True`)，音素保持 MFA 原始位置不变。词界 6.805，首音素仍 6.810
5. **`build_pinyin_phones_tier`** (Phase 1, line 533): 构建首音素时使用 `word_phones[0][0]`（MFA 音素的实际起点 6.810），而非词界 `w_iv.xmin` (6.805)。间隙被原样保留

### 修改点

**Q. `build_pinyin_phones_tier` — 逻辑修复：首/末音素无条件 snap 到词界** (~line 533, 539)

```python
# 修改前 — 音素边界来自 MFA phones tier（独立于 words tier）
new_intervals.append(Interval(word_phones[0][0], word_phones[0][1], dict_phones[0]))
final_end = word_phones[-1][1]

# 修改后 — 音素边界以 words tier 为权威
new_intervals.append(Interval(w_iv.xmin, word_phones[0][1], dict_phones[0]))
final_end = w_iv.xmax
```

**R. Phase 3.B 后同步 pinyin_phones** (~line 3708)

`_refine_boundaries_by_energy` 只接受/返回 `words_tier`，不碰 `pp_tier`。
能量调整后 words 边界变了但 pinyin_phones 没变——这是不同步的根源。
修改：Phase 3.B 更新 words 后，立即从 `phones_tier` + 新 `words_tier` 重建 `pinyin_phones`。

**S. Phase 3.5 后同步 pinyin_phones** (~line 3760)

`_apply_en_phones` 修改 `phones_tier`（注入英语 MFA 音素），但 `pinyin_phones` 未更新。
修改：英语音素注入后立即重建 `pinyin_phones`。

**T. words tier 连续性清理** (~line 3118, `_snap_to_ctc` 内)

MFA 对齐后词间可能有 ≤5ms 微小间隙（帧精度残余），`_snap_to_ctc` 的
`actual_gap > 0.005` 阈值漏掉了这些间隙。修改：构建完所有 word intervals
后统一扫描，将 ≤5ms 间隙吸收到前词尾部，确保 words tier 自身连续。
下游 tier（hanzi、pinyin_phones）依赖此不变式。

**架构原则：四个修改实现了同一目标——三个边界轨道 (words/hanzi/pinyin_phones)
始终以 words tier 为唯一权威，任何修改 words 或 phones 的操作必须立即同步
重建 pinyin_phones。**

### 验证方法

```python
# 首音素必须对齐词界
for w in words_tier:
    first_phone = first_non_sil_phone_in(w)
    assert abs(first_phone.xmin - w.xmin) < 0.002
    last_phone = last_non_sil_phone_in(w)
    assert abs(last_phone.xmax - w.xmax) < 0.002
```

### 关联样本

- `合成ria_00001` ru2 (6.805s → 6.726s after adjust, 词首间隙 5ms → 0ms)
- `合成ria_00001` ru2→guo3 (词间间隙 4ms → 0ms)
- `合成ria_37435` ian4→i4 (词间间隙 5ms, 修复 Q 後归零)

### 补充修改 (2026-07-17)

**L. start pull-back 搜索窗 120ms→80ms** (~line 2339)

120ms 窗口让 `max_rms` 被 50ms 外的 li4 元音峰值（RMS 0.059）污染，he2 元音衰减（RMS 0.004）被误判为"深谷"。缩短到 80ms 后 max_rms 仅覆盖局部邻域，消除了误判。

---

## 完整修改审查 (2026-07-17)

### `_snap_to_ctc` (Phase 3.A)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| A. ratio_skip (a) 尾静音+标点 | ~2520 | 稳定 | 低 |
| A. ratio_skip (b) 可见eps | ~2541 | 稳定 | 低 |
| B. keep_mfa_end 中间点保护 | ~2569 | 稳定 | 低 |
| D. Pattern (b) 缩进修复 | ~2541 | 稳定 | 已修复 |
| E. prev_was_silence_extended | ~2598 | 稳定 | 低 |
| G. has_trailing_sil xmax→xmin | ~2526,~2572 | 稳定 | 低 |
| H. NVV 短时长 <100ms 用MFA | ~2484 | 稳定 | 中：仅限短NVV |
| M. SILENCE_GAP_SNAP 有标点时跳过 | ~2914 | 稳定 | 低 |
| N. silence-adjacent 词首前拉 | ~2321 | 稳定 | 低：(silence→word only, onset_peak>0.002) |
| O. end-trimming 移除 English 豁免 | ~2623 | 稳定 | 低 |
| P. prev_was_silence_extended >100ms | ~2598 | 稳定 | 低 |

### `_inject_punctuation` (Phase 3.C)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| C. gap kind 扩展 "word"→("word","gap") | ~1877 | 稳定 | 低 |

### `_refine_boundaries_by_energy` (Phase 3.B)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| I. 词尾能量延伸 (NVV+静音) | ~2396 | 稳定 | 低 |
| I. NVV 前向延伸 | ~2518 | 稳定 | 低 |
| J. 词首前拉 (start pull-back) | ~2320 | 稳定 (L修后) | 低：80ms窗+0.003下限 |
| K1. is_punct 前先查 is_silence | ~2415 | 稳定 | 低 |
| K2. onset threshold 降低 | ~2483 | 稳定 | 低 |
| K3. 延伸后旧interval占位清除 | ~2550 | 稳定 | 低 |
| K4. _extended_indices 保护 | ~2396,~2623 | 稳定 | 低 |
| K5. punct 全范围检查 | ~2440 | 稳定 | 低 |
| K6. punct_entries 参数传递 | ~2233,~3624 | 稳定 | 低 |

### `build_pinyin_phones_tier` (Phase 1)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| Q. 首/末音素无条件snap到词界（逻辑修复） | ~533,~539 | 新增 | 低：以words边界为权威 |

### `_snap_to_ctc` (Phase 3.A)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| T. words tier 连续性清理（≤5ms间隙吸收） | ~3118 | 新增 | 低：MFA帧精度级别 |

### `process_one` Phase 3.B / 3.5

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| R. Phase 3.B 后同步 pinyin_phones | ~3708 | 新增 | 低 |
| S. Phase 3.5 后同步 pinyin_phones | ~3760 | 新增 | 低 |

### `load_en_phones` / Phase 3.5

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| 自动检测 en_phones_dir | ~3213 | 稳定 | 低 |
| 缺失告警 | ~3508 | 稳定 | 低 |

### 已知风险项

1. **NVV 短时长例外 (H)**：BREATHING<100ms 用 MFA 边界，可能不适于其他短 NVV
2. **静音延伸 (K) 与 end-trimming 的交互**：依赖 `_extended_indices` 精确保护
3. **intervals 列表顺序**：多次原位修改 intervals[i]/[i+1]/[i+2] 可能导致乱序，需增加排序保护

## Case 8: 标点间隙中的 <sp0> 因 has_punct 跳过合并 → mid_sp 误报

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `handle_unexpected_silences`, `process_one` (mid_sp 检测)
**触发场景**: MFA 在标点与后邻词之间插入与后词同时开始的 `<sp0>` artifact

### 现象

words tier 中，标点后面出现 15ms 的 `<sp0>`，与下一个词同时开始：

```
words tier:
  [6.540-6.630] le5        (90ms)
  [6.630-6.660] ，          (30ms)
  [6.660-6.675] <sp0>       (15ms)  ← MFA artifact
  [6.660-6.820] wei4        (160ms) ← 与 <sp0> 同时开始
```

`<sp0>` 与 `wei4` 同时开始 (6.660)，说明这是 MFA 的对齐 artifact。15ms 的间隙没有任何语义意义。

### 根因链

1. MFA 在 `,` 和 `wei4` 之间插入 `<sp0>` [6.660-6.675]，`<sp0>.xmin == wei4.xmin`
2. `tg_word_idx` 过滤标点 → 内容词 `le5`, `wei4`
3. `gap_sil` 正确捕获 `<sp0>`, `gap_punct` 检测到逗号
4. **`has_punct` 短路跳过** (旧 L637): `if sil_label is None or has_punct: continue` → `<sp0>` 未被合并
5. `mid_sp` 检测命中 → 文件被误滤

### 修改点

**U. `handle_unexpected_silences` — `<sp0>` 无条件合并** (~line 636-641)

修改前:
```python
if sil_label is None or has_punct:
    continue
```

修改后:
```python
if sil_label is None:
    continue
if sil_label == "<sp0>":
    pass  # Always merge <sp0> regardless of punctuation
elif has_punct:
    continue  # <sp1-3> + punct: skip (handled by absorb pass)
elif sil_label in ("<sp1>", "<sp2>", "<sp3>"):
    ...
```

**V. `handle_unexpected_silences` — has_punct 时 `<sp0>` 合并到标点** (~line 668-693)

当 `has_punct=True` 时，`<sp0>` 不是合并到前词（会覆盖标点），而是查找邻近的标点间隔，扩展其 `xmax` 吸收 `<sp0>`，并从三个 tier 中删除 `<sp0>`。

```
修复前: le5[6.540-6.630] ，[6.630-6.660] <sp0>[6.660-6.675] wei4[6.660-6.820]
修复后: le5[6.540-6.630] ，[6.630-6.675]              wei4[6.660-6.820]
```

### 验证方法

```python
# words tier 中不应残留标点后的孤立 <sp0>
for i, iv in enumerate(words_tier.intervals):
    if i > 0 and iv.text.strip() == "<sp0>":
        prev = words_tier.intervals[i - 1]
        assert not is_punct(prev.text.strip()), \
            f"<sp0> after punct not merged: {prev.text} → <sp0>"
```

### 关联样本

- 外部项目: `le5 → ， → <sp0> → wei4`（6.540-6.820s 区间）

---

## Case 9: NVV 后的标点+静音链未被吸收 → mid_sp 误报

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `absorb_nvv_trailing` (新增), `absorb_silence_into_punct` (新增)
**触发场景**: NVV 后跟标点+静音链，标点仅 5ms，`<sp2>` 695ms 成为孤立句中静音

### 现象

```
words tier:
  [9.270-9.745] zhi4
  [9.745-9.81]  <LAUGHTER>  (65ms, NVV)
  [9.81-9.815]  ！           (5ms)   ← 标点
  [9.815-10.51] <sp2>        (695ms) ← 孤立静音
  [10.51-10.65] bie2
```

实际音频中，9.81-10.51 是笑声尾音——它不是静音，是 NVV 的一部分。MFA 因无法对 NVV 做声学建模而将其标记为标点+`<sp2>`。

### 根本原因

MFA 无法对两类 token 做声学建模——**NVV** 和**标点**。它们在 MFA 的声学模型里没有对应 phone：

| Token | MFA 行为 | 产生的 artifact |
|------|---------|---------------|
| NVV (`<LAUGHTER>`, `<BREATHING>`, …) | 保留占位符，边界不精炼 | 后随的标点+静音残留 |
| 标点 (`，`, `！`, `…`, …) | 时长压缩到接近 0 | 后随的 `<sp>` 孤悬 |

CTC 预对齐给了初始边界但不准，MFA 无法精炼，postprocess 必须兜底清理。

### 根因链

1. **CTC prealign**: NVV token 无 phone 序列 → CTC 无法分配帧数 → NVV 只分到 65ms，剩余给了 `！` (5ms) 和 `<sp2>` (695ms)
2. **MFA align**: NVV 无 phone 模型 → 占位符保留，边界不精炼 → 相邻未建模段落变成 `<sp2>`
3. **handle_unexpected_silences**: `has_punct=True` → `<sp2>` 被跳过（合理——有标点的长静音不应标记为 unexpected）
4. **旧代码无补救**: 直到 `mid_sp` 检测前，无代码吸收此链
5. **mid_sp**: 孤立的 `<sp2>` → 文件误滤

### 修改点

**W1. 新增 `absorb_nvv_trailing` — Pass 1: NVV 吸收标点+静音链** (~line 758-851)

NVV 向右吞掉连续的标点+静音，直到下一个内容词。将 NVV 的 `xmax` 延伸到下一个实词的 `xmin`。

```
修复前: <LAUGHTER>[9.745-9.81] ！[9.81-9.815] <sp2>[9.815-10.51] bie2[10.51-10.65]
修复后: <LAUGHTER>[9.745-10.51]                                    bie2[10.51-10.65]
```

同步从 phones 和 pinyin_phones tier 删除被吸收的 `<spN>`（标点无 phone 条目无需处理）。

**W2. `absorb_silence_into_punct` — Pass 2 (兜底): 标点吸收残余 `<spN>`** (~line 854-920)

处理未被 NVV 吸收的残余场景（如无 NVV 时的 `标点 → <spN>`）。扩展标点的 `xmax` 吸收紧随其后的 `<spN>`。

**调用顺序** (~line 4031-4039): Phase 4 中 D → D2(`absorb_nvv_trailing`) → D3(`absorb_silence_into_punct`) → E:

```python
# D. handle_unexpected_silences (gap-level <sp0> merge)
# D2. absorb_nvv_trailing (NVV absorbs punct+silence chain)
# D3. absorb_silence_into_punct (fallback: punct absorbs residual <spN>)
```

### 与 Case 8 的关系

| | Case 8 | Case 9 |
|------|------|------|
| 静音类型 | `<sp0>` (< 0.2s) | `<sp1-3>` (≥ 0.2s) |
| artifact 来源 | 标点后的 MFA 帧精度残余 | NVV 无法声学建模 |
| 旧行为 | 残留 → mid_sp | 残留 → mid_sp |
| 修复位置 | `handle_unexpected_silences` (gap 级别) | `absorb_nvv_trailing` + `absorb_silence_into_punct` |
| 修复策略 | `<sp0>` 无条件合并 | Pass 1: NVV 吞链; Pass 2: 标点兜底 |

### 验证方法

```python
# NVV 后不应残留标点+<spN> 链
for i, iv in enumerate(words_tier.intervals):
    if is_nvv_token(iv.text) and i + 1 < len(words_tier.intervals):
        nxt = words_tier.intervals[i + 1].text.strip()
        assert not (is_punct(nxt) or (is_silence(nxt) and nxt)), \
            f"NVV trailed by punct/sil: {iv.text} → {nxt}"

# 标点后不应有孤立的 <spN>
for i in range(len(words_tier.intervals) - 1):
    cur = words_tier.intervals[i].text.strip()
    nxt = words_tier.intervals[i + 1].text.strip()
    assert not (is_punct(cur) and is_silence(nxt) and nxt), \
        f"<spN> after punct not absorbed: {cur} → {nxt}"
```

### 关联样本

- 外部项目: `zhi4 → <LAUGHTER> → ！ → <sp2> → bie2` (9.270-10.65s)

---

### 待处理

- **37443 yu2**: 目标 end=0.78s，当前 end=0.81s。yu2 尾部 (0.75-0.78) 有明显能量衰减，但 gang1 辅音起振 (0.795s, RMS 0.022) 落在 yu2 边界内，tail_rms gate 阻止了裁剪。end-trimming 无法区分"本词元音衰减"和"后词辅音起振"。需多词边界检测或能量谷底分割。

### 已解决

- **37435 jiu4**：修改 P 修复。jiu4 [10.39-10.49]，目标 [10.38-10.50]。
- **37434 ru2**：silence-adjacent 词首前拉 (N)。冒号 [6.80-7.355]，ru2 [7.355-7.50]。
- **Case 8 (标点间隙<sp0>残留)**：修改 U+V。`handle_unexpected_silences` 中 `<sp0>` 无条件合并。
- **Case 9 (NVV后标点+静音链)**：修改 W1+W2。新增 `absorb_nvv_trailing` (D2) + `absorb_silence_into_punct` (D3 兜底)。
- **Case 10 (Hanzi tier拼音残留)**：修改 X+Y+Z。`_word_matches` 元音约束 + `_build_hanzi_tier` 重写为顺序映射。
- **Case 11 (cjk_mismatch时序bug + 缺直接拼音扫描)**：修改 AA1+AA2。检查移至路径决策前 + 新增 `hanzi_pinyin` 直接扫描。
- **Case 15 (_extract_word_chars NVV拆分)**：修改 AJ。`<` `>` flush 处理，防 NVV token 被拆分为独立单元。

---

## Case 11: cjk_mismatch / hanzi_pinyin 检查在路径决策之后执行 → 文件未被重定向

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (filtering section)
**触发场景**: hanzi tier 中存在拼音残留或 CJK 字符错位，但文件仍写入 `output_dir` 而非 `filtered_dir`

### 现象

当 hanzi tier 构建出错（如 Case 10 的 qie4 残留、CJK 字符丢失），现有的 `cjk_mismatch` 检查能检测到问题，但文件仍被写入 `output/` 目录而不是 `filtered/` 目录：

```
report: { "status": "ok", "cjk_details": { "raw_count": 2, "hanzi_count": 1 } }
文件路径: output/xxx.TextGrid     ← 应该在 filtered/ 下！
```

### 根因链

1. **路径决策在检查之前**：`filter_reasons` → `out_path` 的判断在 `cjk_mismatch` 检查之前执行
2. **`if not filter_reasons` 守卫过严**：`cjk_mismatch` 检查只在无其他过滤原因时执行，有其他原因时跳过
3. **缺少直接拼音残留扫描**：无 `is_pinyin_syllable()` 对 hanzi tier labels 的直接检查，仅依赖间接的 CJK 字符计数对比

### 时序对比

```
修复前:
  4717: if filter_reasons: out_path = filtered_dir     ← 此时 filter_reasons 为空
  4726: else: out_path = output_dir                     ← 文件去 output/
  ...
  4738: if not filter_reasons:                          ← 进入检查
  4752:     filter_reasons.append("cjk_mismatch")       ← 检测到了，但太晚！
  4760: write_textgrid(new_tg, out_path)               ← 写入 output/，不是 filtered/

修复后:
  4717: hanzi tier integrity checks                     ← 先检查
  4731:     filter_reasons.append("hanzi_pinyin")        ← 检测到拼音残留
  4746:     filter_reasons.append("cjk_mismatch")         ← 检测到 CJK 不匹配
  4751: if filter_reasons: out_path = filtered_dir       ← 正确重定向！
```

### 修改点

**AA1. `process_one` — Hanzi tier 完整性检查移至路径决策之前** (~line 4717-4749)

两处检查（拼音残留 + CJK 覆盖）从路径决策之后（原 `if not filter_reasons:` 守卫内）移至路径决策之前，移除 `if not filter_reasons` 守卫，确保所有文件都经过检查。

**AA2. `process_one` — 新增直接拼音残留扫描 `hanzi_pinyin`** (~line 4724-4734)

对 hanzi tier 每个 interval 做 `is_pinyin_syllable()` 扫描，检测到拼音残留时添加 `hanzi_pinyin` 过滤原因：

```python
pinyin_labels: list[str] = []
for iv in hanzi_tier_final.intervals:
    label = iv.text.strip()
    if label and is_pinyin_syllable(label):
        pinyin_labels.append(label)
if pinyin_labels:
    filter_reasons.append("hanzi_pinyin")
    report.setdefault("hanzi_pinyin", {})["count"] = len(pinyin_labels)
    report["hanzi_pinyin"]["labels"] = pinyin_labels[:20]
```

### 两个检查的关系

| 检查 | 检测方式 | 覆盖场景 |
|------|---------|---------|
| `hanzi_pinyin` (新增) | 直接 `is_pinyin_syllable()` on hanzi labels | qie4 残留、任何拼音 token 未转换 |
| `cjk_mismatch` (修复) | raw_text CJK 序列 vs hanzi CJK 序列字符串比对 | CJK 丢失、错位、多余 CJK |

两者互补：
- `hanzi_pinyin` 能直接检测到拼音残留这一**根因**
- `cjk_mismatch` 能检测到 CJK 字符序列的任何不一致（包括但不限于拼音残留）

### 验证方法

```python
# 拼音残留检测
for iv in hanzi_tier.intervals:
    assert not is_pinyin_syllable(iv.text.strip()), \
        f"hanzi_pinyin: {iv.text} at [{iv.xmin}-{iv.xmax}]"

# CJK 覆盖检测
assert raw_cjk == hanzi_cjk, \
    f"cjk_mismatch: raw={len(raw_cjk)} hanzi={len(hanzi_cjk)}"
```

### 关联样本

- 外部项目: `SURPRISE，OH 切片OP` — qie4 → 切 (Case 10 修复预防，Case 11 作为防御)

---

## Case 10: Hanzi tier 拼音残留 — NW 对齐 + 过度宽松匹配 → CJK 被跳过 (qie4/切)

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_build_hanzi_tier`, `_word_matches`, `_finalise_textgrid`
**触发场景**: 参考文本包含多音字 + 短英文词相邻（如"切片OP"中 qie4 与 OH/OP 竞争）

### 现象

参考文本是混合中英文：

```
SURPRISE，OH 切片OP
```

最终 TextGrid 的 `hanzi` tier 里，唯独"切"变成了拼音 `qie4`，其他 token 都是正确的汉字/英文：

| words tier | hanzi tier (修复前) | hanzi tier (修复后) |
|-----------|-------------------|-------------------|
| SURPRISE-OH | SURPRISE | SURPRISE |
| qie4 | **qie4** ✗ | **切** ✓ |
| pian4 | 片 | 片 |
| OP | OP | OP |

### 完整链路

```
参考文本: … SURPRISE，OH 切片OP …
              │
              ▼
  ① chars_and_pinyin() — 逐字转拼音生成 .lab
     pypinyin.lazy_pinyin("切") → "qie4"
     （pypinyin 默认词典，"切"独立出现时选第四声 qie4）
     .lab 文件: SURPRISE-OH qie4 pian4 OP
              │
              ▼
  ② MFA 对齐 — 根据 .lab 拼音做强制对齐
     words tier: SURPRISE-OH | qie4 | pian4 | OP
              │
              ▼
  ③ _build_hanzi_tier() — 拼音词 → 汉字反向映射
     【旧代码】用 Needleman-Wunsch 全局对齐 + _word_matches()
              │
              ▼
  ④ _word_matches() 第 1074 行过度宽松匹配
     qie4 是拼音音节格式 → 匹配任意英文单词（代价 0）
     qie4 ↔ "OH" 代价 0；qie4 ↔ "切" 代价 0
     NW gap-first 回溯：优先消费 "OH"，把 qie4 配对给 OH
     "切" 变 reference-only gap → 被跳过
              │
              ▼
  ⑤ hanzi tier: SURPRISE | qie4 | 片 | OP
     "切" 丢失，"qie4" 作为 CTC-only 标签残留
```

### 根因链

1. **pypinyin 多音字默认声调错误**：`lazy_pinyin("切")` 在逐字模式下返回 `qie4`（第四声），而 "切片" 中应读 `qie1`（第一声）。声调错误意味着 pinyin "qie4" 与 CJK "切" 的精确匹配失败（`py[0] == c` 检查 `py == "qie4"` 而 `lazy_pinyin("切")` 也返回 `"qie4"` — 但这是关键：两边都是 qie4，所以精确匹配实际上通过了）。

    **更正**：实际根因不是声调不匹配，而是 `_word_matches()` 第 1074 行的无差别匹配：`return True` 让任何拼音音节匹配任何英文单词代价为 0。NW gap-first 回溯在代价相同时优先 gap（跳过 ref），选择了代价相同的 `qie4↔OH` 而非 `qie4↔切`。

2. **`_word_matches()` 过度宽松**：拼音音节格式 `[a-z]+[1-5]` 无条件匹配任何英文单词（代价 0），不检查参考词长度或元音数。

3. **NW gap-first 回溯**：在 `dp[i][j] == dp[i-1][j] + 1`（CTC gap）和 `dp[i][j] == dp[i][j-1] + 1`（reference gap）代价相同时，优先选 CTC gap。当 `qie4↔OH` 和 `qie4↔切` 代价都为 0 时，gap-first 回溯选择消费 OH 而非 切。

### 触发场景分类

**场景 A — 多音字默认声调**：pypinyin 对"切"、"的"、"了"、"为"等字选错默认声调。声调错误本身不直接触发 bug（因为 _word_matches 用同样的 lazy_pinyin 做反向匹配），但声调不同让精确匹配代价变 1，不再严格优于宽松的拼音→英文匹配。

**场景 B — 拼音音节 + 短英文词相邻（直接触发）**：words tier 中拼音音节 token 与短英文 token（OH, OP, in, up）相邻，NW 的 gap-first 回溯在代价相同时优先消费英文词。

**场景 C — MFA 合并英文相邻词**：MFA 将相邻英文词合并（SURPRISE + OH → SURPRISE-OH），增加了 token 序列的不确定性。

**场景 D — MFA 丢弃 token（极端）**：上游丢弃 CJK 拼音 token，导致下游错位。

### 修改点

**X. `_word_matches` — 拼音→英文匹配增加元音约束** (~line 1069-1076)

修改前（旧）：
```python
# 旧：return True（任意拼音音节匹配任意英文词）
if len(c) >= 2 and c[-1].isdigit() and c[:-1].isalpha():
    return True
```

修改后（新）：
```python
# 新：仅当英文参考词 ≥2 个元音字母时才匹配
if len(c) >= 2 and c[-1].isdigit() and c[:-1].isalpha():
    vowel_count = sum(1 for ch in r if ch in 'aeiou')
    return vowel_count >= 2
```

效果：
- `qie4` 不再匹配 `OH`（1 个元音）✓
- `qie4` 不再匹配 `OP`（1 个元音）✓
- `ai4` 仍然匹配 `idol`（2 个元音）✓
- `rui4` 仍然匹配 `ria`（2 个元音）✓

**Y. `_build_hanzi_tier` — 重写：顺序 CJK 映射替代 NW 对齐** (~line 1173-1298)

核心思想：CJK 字符按顺序一一对应 MFA 拼音词位置，不依赖 pypinyin 声调准确性。

```
CJK 字符：拼音音节按顺序消费参考文本中的下一个 CJK 字
  不依赖 pypinyin 声调准确性
  不依赖任何字典
  qie4 → 下一个未使用的 CJK 字 → "切" ✓

英文/NVV：贪心子串匹配
  SURPRISE-OH → 消费 "SURPRISE" + "OH" 两个参考单元
  "li" → 匹配 "live"（子串）
```

关键改动：
- CJK 字符和英文词**分开处理**，使用独立的消费游标
- CJK：`is_pinyin_syllable(token)` → 从 `ref_cjk` 队列消费下一个字符
- 英文/NVV：`_alpha_text_matches(token, ref)` → 贪心消费匹配的 alpha 参考单元
- 标点：`is_punct()` → 原样透传，不消耗任何游标
- silence：保持 silence label 透传

**Z. `_alpha_text_matches` — Alpha token ↔ ref unit 匹配** (~line 1136-1170, 新增)

从旧 `_word_matches` 中抽取 alpha 匹配逻辑（去掉 CJK pinyin 精确匹配部分），用于 `_build_hanzi_tier` 的贪心消费。

### 防御性检测（新增）

当 words tier 中的拼音音节数量 ≠ 参考文本中的 CJK 字符数量时，向 `report["warnings"]` 输出 warning：

```
# 拼音 token 多于 CJK 字符（token 冗余）
"hanzi tier mismatch: 4 pinyin tokens vs 2 reference CJK chars
 — 2 pinyin token(s) fell back (no more CJK chars to consume)"

# CJK 字符多于拼音 token（字符丢失）
"hanzi tier mismatch: 2 pinyin tokens vs 3 reference CJK chars
 — 1 reference CJK char(s) were not assigned to any pinyin token"
```

### warnings 参数 threading

`_finalise_textgrid` 新增 `warnings` 参数 → 透传至 `_build_hanzi_tier`。两处调用点（`process_one` Phase 2 的 `_finalise_textgrid` 和 Phase 5 的直接 `_build_hanzi_tier` 调用）均传入 `report.get("warnings", [])`。

### 核心注意点

1. **`lazy_pinyin` 逐字模式的固有限制**：pypinyin 对每个字独立判断，无法利用上下文消歧。解决方案是让 hanzi tier 构建不依赖 pypinyin 的返回值。

2. **顺序映射的前提假设**：MFA words tier 中拼音音节的顺序和数量 = 参考文本中 CJK 字符的顺序和数量。正常 pipeline 中此前提始终成立。

3. **NW 对齐仍然保留**：`_align_word_sequences()` + `_word_matches()` 仍被 `_normalize_word_spellings()` 使用。修复只改了 `_word_matches` 中的一条规则（元音约束），保留了其他所有匹配逻辑。

4. **标点处理**：标点在 words tier 中作为独立 interval 存在时，原样透传不消耗游标。被 MFA 挤掉时对 CJK 顺序映射无影响。

5. **英文词合并与拆分**：合并（SURPRISE-OH → 两个参考词）通过贪心匹配依次消费。拆分（li ve → live）由第一个 token 消耗参考词，第二个变 CTC-only gap。

### 修改文件清单

| 文件 | 行号 | 改动 |
|------|------|------|
| postprocess_textgrids.py | ~924 | `_finalise_textgrid` 新增 `warnings` 参数 |
| postprocess_textgrids.py | ~961 | 向 `_build_hanzi_tier` 传递 warnings |
| postprocess_textgrids.py | ~1069-1076 | `_word_matches` 拼音→英文匹配增加元音约束 |
| postprocess_textgrids.py | ~1136-1170 | `_alpha_text_matches` 新增 |
| postprocess_textgrids.py | ~1173-1298 | `_build_hanzi_tier` 重写：顺序 CJK 映射 + 贪心 alpha 匹配 + 防御检查 |
| postprocess_textgrids.py | ~3967 | `process_one` 向 `_finalise_textgrid` 传递 `report["warnings"]` |
| postprocess_textgrids.py | ~4300 | `process_one` 向第二次 `_build_hanzi_tier` 传递 `report["warnings"]` |

### 验证方法

```python
# 拼音音节 token 必须映射到 CJK 字符，不能残留为拼音
for iv in hanzi_tier.intervals:
    assert not is_pinyin_syllable(iv.text.strip()), \
        f"Pinyin residue in hanzi tier: {iv.text}"

# CJK 字符计数应与拼音 token 计数一致
# (不一致时只在 warnings 中报告，不阻断)
```

### 关联样本

- 外部项目: `SURPRISE，OH 切片OP` (qie4 → 切 修复)

### 补充修改 (2026-07-28)

**warnings 重复修复** — `_finalise_textgrid` 调用不再传递 warnings。

根因：`process_one` 中两次调用 `_build_hanzi_tier`：
1. Phase 2（line ~3976，通过 `_finalise_textgrid`）— 构建的 hanzi tier 在 Phase 5 被替换
2. Phase 5（line ~4311）— 构建最终 hanzi tier

两处都传入 `report["warnings"]` 导致每条 mismatch 警告出现两次。
修复：Phase 2 的 `_finalise_textgrid` 不再传递 warnings（默认 `None`，跳过防御检查），
仅 Phase 5 的调用负责输出 warnings。

---

## Case 12: CTC snap 重叠修复产生倒置 interval + 尾静音未被最后标点吸收 (xia4/ta1)

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`, `_inject_punctuation`
**触发样本**: 花礼_不可言说之小礼猫跳船日_20250331_2331, segment 00060000 (clip0006_clip0000)

### 现象

参考文本末尾"你看一下他。"，实际音频中"他"字发音极短 (MFA 检测仅 0.06s)。Pipeline 输出中 `ta1`（他）从 words/hanzi tier 消失，尾静音 `<sp2>` 与前后 interval 重叠：

```
MFA 对齐:
  xia4[10.79-10.87] dur=0.08s  ta1[10.87-10.93] dur=0.06s  <sp2>[10.93-11.49]  。[10.95-11.565875]

CTC 预对齐:
  xia4[10.79-10.95] dur=0.16s   ta1[10.87-10.93] dur=0.06s

修复前 (有 bug):
  words:  xia4[10.79-10.95]  <sp2>[10.93-11.49]  。[10.95-11.565875]
  hanzi:  下                                    。     ← "他" 丢失!
  ↑ <sp2>.xmin(10.93) < xia4.xmax(10.95) 重叠!
  ↑ 。.xmin(10.95) < <sp2>.xmax(11.49) 重叠!

修复后 (预期):
  words:  xia4[10.79-10.95]  ta1[10.95-11.01]  。[11.01-11.565875]
  hanzi:  下                他                。
  ↑ 边界连续无重叠
```

### 根因链

**Bug 1 — 倒置 interval**:

1. **xia4（下）**：MFA 时长 0.08s，CTC 时长 0.18s → `ctc_dur > mfa_dur * 2`，触发 Rule 3 (`use_mfa=False`)，xmax 从 MFA 的 10.87 snap 到 CTC 的 10.95
2. **ta1（他）**：MFA 时长 0.06s，MFA 和 CTC 偏差均在 0.3s 阈值内 → `use_mfa=True`，保持 MFA 边界 10.87–10.93
3. **重叠检测**：`ta1.xmin(10.87) < xia4.xmax(10.95) - 0.002` → 触发重叠修复
4. **prev_was_silence_extended 分支不匹配**：xia4 不是 NVV，`prev_end(10.95) > prev_ctc_end(10.95) + 0.10` 为 False → 进入 else 分支：`word_start = prev_end = 10.95`
5. **word_end 未更新**：ta1 的 word_end 仍为 MFA 原始值 10.93
6. **倒置 interval**：ta1 = (10.95, 10.93) → `xmin > xmax`！
7. **静默丢弃**：`_inject_punctuation` 中 `if e > s` 过滤（~line 2098）将倒置的 ta1 丢弃，他 从 hanzi tier 消失

**Bug 2 — 尾静音未被吸收**:

1. ta1 被丢弃后，`_snap_to_ctc` 在 `prev_end(10.93)` 之后插入尾静音 `<sp2>(10.93-11.49)`
2. `_inject_punctuation` 最后标点逻辑：查找最后非静音词 → xia4，`punct_start = xia4.xmax = 10.95`
3. 条件 `m[0] < punct_start`：`<sp2>.xmin(10.93) < 10.95` → 被保留！
4. `<sp2>` 与 xia4 重叠 (10.93 < 10.95)，又与扩展后的 。重叠 (10.95 < 11.49)

### 修改点

**AC. `_snap_to_ctc` — 重叠修复后防倒置 interval 保护** (~line 3317)

修改前：
```python
            else:
                word_start = prev_end

        # ── Gap absorption (ORDER CRITICAL — do not reorder) ──
```

修改后：
```python
            else:
                word_start = prev_end

        # Guard against inverted intervals: when overlap fix pushes word_start
        # past word_end (prev word CTC-snapped longer than current word's MFA
        # end), extend word_end to preserve the word with at least its MFA
        # duration or a 30 ms floor.
        if word_end < word_start:
            word_end = word_start + max(mfa_dur, 0.030)

        # ── Gap absorption (ORDER CRITICAL — do not reorder) ──
```

效果：ta1 不再倒置，`word_end = 10.95 + max(0.06, 0.030) = 11.01`，保留原始时长。

**AD. `_inject_punctuation` — 最后标点保留条件从 xmin 改为 xmax** (~line 2277)

修改前：
```python
            elif m[0] < punct_start:
                new_merged.append(m)
```

修改后：
```python
            elif m[1] <= punct_start + 0.001:
                new_merged.append(m)
```

效果：`<sp2>.xmax(11.49) <= 10.95 + 0.001` → `11.49 <= 10.951` → False → 被最后标点吸收。`xia4.xmax(10.95) <= 10.951` → True → 保留。

### 触发条件（三个条件同时满足）

1. **前词被 CTC 快照拉长**：MFA 时长 << CTC 时长（超过 2× 比例阈值，Rule 3），`use_mfa=False`，xmax 延伸到 CTC 边界
2. **当前词保持 MFA 边界**：MFA 和 CTC 偏差均在阈值内，`use_mfa=True`
3. **当前词完全被前词覆盖**：`cur.xmin < prev.xmax`

### 关联样本

- `花礼_不可言说之小礼猫跳船日_20250331_2331` segment `00060000`：xia4(下) → ta1(他) → 。

---

## Case 13: NVV 大小写不一致 — MFA 小写 NVV 不被识别/包裹 → pinyin_phones 残留小写 (laughter)

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/finalize_textgrids.py`
**涉及函数**: `_finalize_textgrid`, `build_pinyin_phones_tier`
**触发样本**: segment 00150009 (花礼 pipeline), `<LAUGHTER>` 出现在 words/hanzi 但 pinyin_phones 残留小写 `laughter`

### 现象

`<LAUGHTER>` 在 words/hanzi/pinyin tier 中是大写并包裹 `<>`，但 pinyin_phones tier 中却是小写无括号：

```
raw_text (T1):   ...<LAUGHTER>！
pinyin (T2):     ... <LAUGHTER> ！
hanzi (T3):      [54] <LAUGHTER>
words (T4):      [54] <LAUGHTER>
pinyin_phones (T5): [95] laughter    ← 小写，无 <>，与其他 tier 不一致！
```

### 根因链

1. **MFA 输出小写**：MFA 将 OOV（含 NVV）统一输出为小写 `laughter`（非 `LAUGHTER`）
2. **`_NVV_PATTERN` 大小写敏感**：regex 仅匹配 `(?<![A-Z-])` + 大写 `NVV_NAMES`，小写 `laughter` 不匹配 → `_finalize_textgrid` 中 `_NVV_PATTERN.sub` 不生效
3. **CTC token 提供大写**：CTC 预对齐输出 `LAUGHTER`（大写），words tier 在某步被更新为大写，后续 `_finalize_textgrid` 能匹配并包裹
4. **`build_pinyin_phones_tier` 时机早**：在 words tier 被 CTC 修正前就已执行，复制了小写 `laughter`，后续不再重建
5. **`finalize_textgrids.py` 无防护**：直接用 `f"<{iv.text}>"` 包裹，保留原始大小写，且无双重包裹防护（若已包裹 `<>` 会变成 `<<LAUGHTER>>`）

### 修改点

**AE. `_NVV_PATTERN` — 添加 `re.IGNORECASE` + 扩展字符类** (~line 87-91)

修改前：
```python
_NVV_PATTERN = re.compile(
    r"(?<![A-Z-])("
    + "|".join(re.escape(name) for name in sorted(NVV_NAMES, key=len, reverse=True))
    + r")(?![A-Z-])"
)
```

修改后：
```python
_NVV_PATTERN = re.compile(
    r"(?<![A-Za-z-])("
    + "|".join(re.escape(name) for name in sorted(NVV_NAMES, key=len, reverse=True))
    + r")(?![A-Za-z-])",
    re.IGNORECASE
)
```

效果：`laughter`, `Laughter`, `LAUGHTER` 均被匹配，同时确保不被部分匹配（如 `slaughter`）。

**AF. `_finalize_textgrid` — 替换文本统一大写** (~line 141)

修改前：
```python
iv.text = _NVV_PATTERN.sub(r"<\1>", iv.text)
```

修改后：
```python
iv.text = _NVV_PATTERN.sub(lambda m: f"<{m.group(1).upper()}>", iv.text)
```

效果：无论 MFA 输出什么大小写，最终统一为 `<LAUGHTER>`。

**AG. `build_pinyin_phones_tier` — NVV 文本规范化** (~line 485-488)

修改前：
```python
# NVV token: one self-referential phone
if is_nvv_token(w_iv.text):
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
    continue
```

修改后：
```python
# NVV token: one self-referential phone — normalize to <UPPERCASE>
if is_nvv_token(w_iv.text):
    nvv_text = f"<{w_iv.text.strip().strip('<>').upper()}>"
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, nvv_text))
    continue
```

效果：pinyin_phones tier 中 NVV 始终以规范格式 `<UPPERCASE>` 写入，与 words tier 一致。`strip('<>')` 防双重包裹。

**AH. `finalize_textgrids.py` — NVV 规范化** (~line 50-54)

修改前：
```python
for tier in tg.tiers:
    for iv in tier.intervals:
        if iv.text and is_nvv(iv.text):
            iv.text = f"<{iv.text}>"
```

修改后：
```python
for tier in tg.tiers:
    for iv in tier.intervals:
        if iv.text and is_nvv(iv.text):
            # Normalize to <UPPERCASE> — strip existing brackets/case first
            # to avoid double-wrapping and mixed-case inconsistencies.
            cleaned = iv.text.strip().strip('<>').upper()
            iv.text = f"<{cleaned}>"
```

效果：独立脚本也输出一致的 `<UPPERCASE>` 格式，且不会 `<<LAUGHTER>>`。

### 验证方法

```python
# _NVV_PATTERN 大小写不敏感
assert _NVV_PATTERN.search("laughter")
assert _NVV_PATTERN.search("Laughter")
assert _NVV_PATTERN.search("LAUGHTER")
assert not _NVV_PATTERN.search("slaughter")  # 不部分匹配

# 替换统一大写
assert _NVV_PATTERN.sub(lambda m: f"<{m.group(1).upper()}>", "laughter") == "<LAUGHTER>"

# pinyin_phones NVV 规范化
assert normalize_nvv("laughter") == "<LAUGHTER>"
assert normalize_nvv("<LAUGHTER>") == "<LAUGHTER>"  # 无双重包裹

# finalize_textgrids 规范化
assert finalize_nvv("laughter") == "<LAUGHTER>"
assert finalize_nvv("Laughter") == "<LAUGHTER>"
```

### 关联样本

- `00150009` (花礼 pipeline): `<LAUGHTER>` — words/hanzi tier 大写包裹，pinyin_phones 小写无包裹

---

## Case 14: Hanzi tier NVV 括号污染 — `_build_hanzi_tier` alpha 分支未 strip `<>` → label 残留 `<SURPRISE-WA>`

**日期**: 2026-07-28
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_build_hanzi_tier`, `_alpha_text_matches`
**触发样本**: segment 00150015 (20241117 花礼 pipeline)

### 现象

Hanzi tier 中 `<SURPRISE-WA>` 残留了 NVV 括号包裹，且 `ai1` 未映射到汉字 `哎`：

```
words tier:     [2] <SURPRISE-WA>    [3] ai1
hanzi (修复前): [2] <SURPRISE-WA>    [3] ai1      ← 括号污染 + 拼音残留
hanzi (修复后): [2] SURPRISE-WA      [3] 哎        ← 正确
```

逗号时间在三个轨道一致（1.71–2.43），问题仅限于 hanzi tier 逗号之前的映射。

### 根因链

1. **`SURPRISE-WA` 是标准 NVV 名称**：MFA OOV 后 `_finalize_textgrid` 将其包裹为 `<SURPRISE-WA>`
2. **`_alpha_text_matches` 内部能 strip `<>`**：子串匹配 `"surprise-wa" in "<surprise-wa>"` 正常返回 True
3. **但 fallback 路径使用原始 `token`**（含 `<>`）：当 `ref_alpha` 为空或匹配失败时，`matched_refs = []`，label 回退为原始 token 文本 `<SURPRISE-WA>`（`<` 不是 alpha/hyphen，`all(c.isalpha() or c == '-' for c in token)` 失败 → 最终 `label = token`）
4. **`continue_match` 检查也使用原始 `token`**：`token.lower()` 为 `"<surprise-wa>"`，多词消费判断受 `<>` 干扰
5. **CJK 游标受影响**：alpha 未匹配成功 → `alpha_idx` 不推进 → `ai1` 可能错误匹配到 alpha ref 而非 CJK 字符

### 修改点

**AI. `_build_hanzi_tier` — Alpha 分支 strip `<>`** (~line 1246-1278)

修改前：
```python
        # ── English / NVV token → greedy match against alpha refs ──
        matched_refs: list[str] = []
        while alpha_idx < len(ref_alpha):
            ref_unit = ref_alpha[alpha_idx]
            if _alpha_text_matches(token, ref_unit):
                ...
                if next_ref in token.lower() or token.lower() in next_ref:
                    ...
        ...
        elif token.isascii() and all(c.isalpha() or c == '-' for c in token):
            label = token
        else:
            label = token
```

修改后：
```python
        # ── English / NVV token → greedy match against alpha refs ──
        # Strip angle brackets: _finalize_textgrid may have already
        # wrapped NVV tokens with < > before we run.  Matching and
        # fallback labels must use the clean form to avoid bracket
        # pollution in the hanzi tier and misaligned cursors.
        clean_token = token.strip('<>')
        matched_refs: list[str] = []
        while alpha_idx < len(ref_alpha):
            ref_unit = ref_alpha[alpha_idx]
            if _alpha_text_matches(clean_token, ref_unit):
                ...
                if next_ref in clean_token.lower() or clean_token.lower() in next_ref:
                    ...
        ...
        elif clean_token.isascii() and all(c.isalpha() or c == '-' for c in clean_token):
            label = clean_token
        else:
            label = clean_token
```

效果：
- `<SURPRISE-WA>` → `clean_token = "SURPRISE-WA"` → 匹配 + fallback 均使用无括号形式
- `all(c.isalpha() or c == '-' for c in "SURPRISE-WA")` → True → fallback label 为 `SURPRISE-WA`（而非 `<SURPRISE-WA>`）
- `continue_match` 检查使用 `clean_token` → 多词消费不受 `<>` 干扰

### 验证方法

```python
clean = token.strip('<>')
# 匹配
label = _alpha_match_and_fallback(clean, ref_alpha)
assert '<' not in label and '>' not in label  # 无括号污染
# CJK 游标正确
assert cjk_idx == expected  # ai1 → 哎 (CJK 正确消费)
```

### 关联样本

- `00150015` (20241117 花礼 pipeline): `<SURPRISE-WA>` + `ai1` → 逗号之前 hanzi 映射错误

### 本段全部修改汇总

| 段 | 问题 | Case | 修复 |
|---|------|------|------|
| 20250331/00060000 | `_snap_to_ctc` 倒置 interval → 他 丢失 | 12 | overlap fix: push word_end |
| 20250331/00060000 | `<sp2>` 未被末尾标点吸收 → 边界重叠 | 12 | `m[1] <= punct_start` |
| 20250202/00150009 | pinyin_phones `laughter` ≠ words `<LAUGHTER>` | 13 | case-insensitive NVV 包裹 + 统一大写 |
| 20241117/00150015 | hanzi 残留 `<SURPRISE-WA>` 和 `ai1` | 14 | strip `<>` + alpha 匹配用 `clean_token` |

---

## Case 15: `_extract_word_chars` 拆分 NVV 尖括号 — `<LAUGHTER>` → `[<, LAUGHTER, >]`

**日期**: 2026-07-28
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_extract_word_chars`, `_build_hanzi_tier`
**发现方式**: CROSS_CASE_ANALYSIS.md Risk 4.5 验证过程中发现

### 现象

`_extract_word_chars` 将 NVV token 的尖括号当作标点拆分：

```
_extract_word_chars("<LAUGHTER>你好")
修复前: ['<', 'LAUGHTER', '>', '你', '好']   ← NVV 被拆成 3 个单元
修复后: ['<LAUGHTER>', '你', '好']            ← 正确
```

### 根因链

1. **`<` 和 `>` 不在 alpha 字符集中**：`_extract_word_chars` 的 alpha buffer 条件为 `c.isalpha() or c == '-'`，未包含 `<>`
2. **尖括号被当作标点**：落到 `else` 分支 → flush buffer → 每个尖括号作为独立标点 entry
3. **`is_word_like` 过滤差异**：`"<"` 不是 word-like（无 alpha/CJK/digit）→ 被过滤；`"LAUGHTER"` 是 word-like（以 alpha 开头）→ 进入 `ref_alpha`
4. **最终输出被 `_finalize_textgrid` 掩盖**：`_finalize_textgrid` 在所有 tier 上运行 `_NVV_PATTERN.sub` 包裹 `<>`，所以 hanzi tier 中 `LAUGHTER` 被重新包裹为 `<LAUGHTER>`

### 为什么之前未被发现

此 bug 被三层防护掩盖：
1. `is_word_like` 过滤掉裸 `<` `>` → 不在 ref_cjk/ref_alpha 中 → 不参与游标消费
2. `_alpha_text_matches` 内部 strip `<>` → `clean` 匹配仍成功 → cursor 推进
3. `_finalize_textgrid` 事后包裹 `<>` → 最终输出括号正确

但在以下场景可能暴露：
- 多个 NVV 相邻（如 `<LAUGHTER><BREATHING>`）→ `_extract_word_chars` 产生 `['<','LAUGHTER','>','<','BREATHING','>']` → 两个 `>` 被当作独立 punct → `is_word_like` 过滤 → ref_alpha = `['LAUGHTER','BREATHING']` → 游标消费仍正确但中间状态混乱

### 修改点

**AJ. `_extract_word_chars` — `<` `>` 特殊处理** (~line 1004-1032)

修改前：
```python
elif c.isalpha() or c == '-':
    buf += c
elif c.isdigit():
    buf += c
else:
    # punct / whitespace
```

修改后：
```python
elif c == '<':
    if buf:
        result.append(buf)
        buf = ""
    buf += c
elif c == '>':
    buf += c
    result.append(buf)
    buf = ""
elif c.isalpha() or c == '-':
    buf += c
elif c.isdigit():
    buf += c
else:
    # punct / whitespace
```

效果：
- `<LAUGHTER>` → `['<LAUGHTER>']` ✓
- `<LAUGHTER>你好` → `['<LAUGHTER>', '你', '好']` ✓（`>` flush 后下一个字独立）
- `<sp1>test` → `['<sp1>', 'test']` ✓（silence tag 也正确分组）
- `hello-world` → `['hello-world']` ✓（hyphen 不受影响）
- `a—b` → `['a', '—', 'b']` ✓（em-dash 仍正确拆分）

### 与其他 Case 的关系

| Case | 问题 | 与本 Case 的关系 |
|------|------|----------------|
| 13 | NVV 大小写不一致 | 同属 NVV 文本处理链：Case 13 修复识别，本 Case 修复拆分 |
| 14 | hanzi tier NVV 括号污染 | Case 14 修复了 alpha 分支的 `<>` strip，本 Case 从源头防止 `<>` 被拆分 |
| 10 | hanzi tier 拼音残留 | 同属 `_extract_word_chars` → `_build_hanzi_tier` 数据流 |

### 验证方法

```python
assert _extract_word_chars("<LAUGHTER>你好") == ["<LAUGHTER>", "你", "好"]
assert _extract_word_chars("<sp1>test") == ["<sp1>", "test"]
assert _extract_word_chars("<BREATHING>hello") == ["<BREATHING>", "hello"]
assert _extract_word_chars("hello-world") == ["hello-world"]  # 不受影响
assert _extract_word_chars("a—b") == ["a", "—", "b"]  # 不受影响
```

---

---
## Case 16: MFA `--fine_tune` 默认开启导致边界漂移 + 对齐失败

**日期**: 2026-07-28
**涉及文件**: `scripts/run_pipeline.py`
**涉及函数**: `step_mfa_align`, `DEFAULT_CFG`
**触发场景**: 所有含 NVV token、BGM、英文词、短虚词的音频（花礼/鸣海派/Xiaoyuan/嘉然等直播数据集）

### 现象

**修复前**（fine_tune 默认 True）:
- hualishaxuan 全部 4 次 MFA align 尝试均失败（exit_code=1）
- CTC 锚点经 `adjust_ctc_boundaries.py` 能量修正后质量良好，但 MFA fine_tune 将其浮动到错误位置
- `_snap_to_ctc` 无法纠正：fine_tune 偏移 ≤0.1s，远低于 snap 阈值 0.3s

**修复后**（fine_tune 默认 False）:
- MFA 对齐成功（已验证：无 fine_tune 的测试运行 exit_code=0）
- 词边界保持在 adjust_ctc_boundaries 修正后的 CTC 锚点位置
- MFA 仍能在固定词边界内做音素级对齐（首次对齐 pass 不受 fine_tune 影响）

### 根因链

1. **adjust_ctc_boundaries.py**: 用音频能量分析精细化 CTC 词边界 → 产出高质量锚点
2. **MFA `--fine_tune`**（默认 True，tolerance=0.1s）: 用纯语音训练的声学模型浮动锚点边界。模型对 NVV（无模型）、BGM（未训练）、短虚词（能量弱）、英文（错误模型）均表现差 → 锚点被推偏
3. **_snap_to_ctc**（snap_threshold=0.3s）: 检查 MFA 偏离 CTC >0.3s 时拉回。fine_tune 仅移动 ≤0.1s → 永远不会触发 snap → 漂移永久保留

**设计冲突本质**: `adjust_ctc_boundaries`（能量修正）和 MFA fine_tune（声学模型修正）对同一数据做出矛盾决策，而 _snap_to_ctc 的阈值设计使后者覆盖前者。

### 修改点

**A. `step_mfa_align` — fine_tune 默认值 True → False** (~line 1002)

```python
# 修改前
if mc.get("fine_tune", True):        # 默认开启
    mfa_args.append("--fine_tune")
fine_tune_tolerance = mc.get("fine_tune_boundary_tolerance", 0.1)  # 100ms

# 修改后
if mc.get("fine_tune", False):       # 默认关闭
    mfa_args.append("--fine_tune")
fine_tune_tolerance = mc.get("fine_tune_boundary_tolerance", 0.02) # 20ms
```

**B. `DEFAULT_CFG["mfa"]` — 新增默认值** (~line 205)

```python
"fine_tune": False,          # DISABLED: adjust_ctc_boundaries already refines anchors
"fine_tune_boundary_tolerance": 0.02,  # only used when fine_tune: true
```

**C. `config.yaml` — 参考文档同步更新**

### 关联样本

- hualishaxuan (花礼筛选): 87 文件全部 MFA 对齐失败
- FILTER_ANALYSIS_REPORT.md: ~60% word_in_silence 命中归因于 fine_tune
- 已禁用 fine_tune 的配置（均正常）: hechengria_test100, test_shayi_1, aed_3files, test_en_mfa, hechengria_10, test_hechengria_1

### 验证方法

```python
# 检查 command_history.yaml 中的所有 MFA align 调用
# 修复后不应再出现 --fine_tune 标志（除非数据集显式配置）
```

---

## Case 17: NVV token 被 `_normalize_word_spellings` 改写 + 首词为标点 + pinyin_phones 断层

**日期**: 2026-07-28
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/ctc_prealign.py`
**涉及函数**: `_normalize_word_spellings`, `strip_edge_punctuation`, `build_pinyin_phones_tier`, `_normalize_punct`, `_sync_derived_tiers`
**触发样本**: 花礼_emo来听小鼠歌吧_20250504_2058_3700c4b4_clip0003_clip0012

### 现象

修复前：
```
words tier:       <sp1>[0.000-1.170]  …[1.170-1.710]  SURPRISE[1.710-1.950]  qie4[1.950-2.150]
pinyin_phones:    <sp1>[0.000-1.170]  …[1.170-1.710]  en:tɕʰ[1.940-1.950]  q[1.950-2.100]
```
三个症状：
1. **`…` 为首词**: 省略号出现在 `<sp1>` 之后，SURPRISE 之前
2. **SURPRISE pinyin_phones 断层**: 词区间 [1.71-1.95] 内 pinyin_phones 为 `en:tɕʰ`（这是下一词 `qie4` 的声母泄漏），SURPRISE 本体无音素覆盖，形成 230ms 空洞
3. **pinyin_phones 与 hanzi/words 不同步**: 三级轨道边界不重合

### 根因链

**子问题 A: `…` 为首词**

1. **NVASR CTC 预处理** (ctc_prealign.py): 原始 ASR 输出 `<|zh|><|HAPPY|>…[Surprise-oh]切片OP…`，NVASR 在产出 `_text_cn.txt` 时去掉 `<|zh|>` `<|HAPPY|>` 标签，但 `…` 残留在文本开头
2. **`_punct.json` 保留 `…`**: CTC 停顿检测将 `…` 作为标点条目写入 `_punct.json`，时间戳 [1.17-1.71]
3. **`_inject_punctuation`**: 将 `…` 插入 words tier，填充 `<sp1>` 与 SURPRISE 之间的间隙 → `…` 成为首词

**子问题 B: SURPRISE 音素断层**

4. **`_normalize_punct` 连字符→逗号**: `_text_cn.txt` 中 `SURPRISE-OH` 经过 `_normalize_punct` 时，`is_punct('-')` 返回 True 且 `-` 不在白名单中 → 替换为 `，` → `_text_cn.txt` 变成 `SURPRISE，OH`。同批次 18 个含连字符 NVV 的文件 100% 受影响。但 `.lab`/`_tokens.jsonl` 不被 `_normalize_punct` 修改 → 保持 `SURPRISE-OH`
5. **MFA align**: `SURPRISE-OH` 无对应声学模型 → MFA 将其对齐为 `spn` [1.72-1.94]
6. **`_normalize_word_spellings`** (Phase 5): 用 raw_text 中的 `SURPRISE`（`_finalise_textgrid` 吞掉了 `，OH` 标点）替换 words tier 中的 `surprise-oh`。改写后 SURPRISE 不再是 NVV token → `is_nvv_token("SURPRISE")` = False → 被当作英文词处理
7. **English MFA**: SURPRISE 不在 en_mfa_windows 中 → 无声学音素数据
8. **`build_pinyin_phones_tier`**: 收集 SURPRISE 词区间内的 MFA phones → `spn` 被过滤，只有 `tɕʰ` [1.94-1.95] 残留（从 qie4 泄漏）。`en_mfa_windows` 无 SURPRISE 条目时，使用 `(w_iv.xmin, w_iv.xmax)` 作为缺省窗口 → 泄漏音素未被过滤 → pinyin_phones 出现 `en:tɕʰ`（错误标注）
9. SURPRISE 的 pinyin_phones 覆盖范围 [1.94-1.95] 与词区间 [1.71-1.95] 不重合 → 230ms 空洞

**子问题 C: 轨道不同步（系统性问题）**

10. postprocess 处理链中，多个阶段独立修改 words tier（`_snap_to_ctc`、`_refine_boundaries_by_energy`、`_inject_punctuation`、Phase 4 合并操作），但 hanzi 和 pinyin_phones 的同步重建仅在 Phase 5 末尾进行
11. Phase 4 的 D2 (`absorb_nvv_trailing`)、D3 (`absorb_silence_into_punct`)、D4 (`strip_edge_punctuation`) 在 textgrid 上原地修改 words tier 边界，但不触发 hanzi/pinyin_phones 重建 → Phase 4 后续 E/F/G 步骤读取的 pp_tier 是过时数据
12. NVV token 一旦被 `_normalize_word_spellings` 改写为普通英文词，所有下游 NVV 处理逻辑（边界吸收、自引用音素）均被跳过

**架构根因**: 管线没有统一的 tier 同步机制。words tier 是权威数据源，hanzi 和 pinyin_phones 是派生数据，但派生数据的重建散布在各处（Phase 3B、3.5、5），缺乏系统性的"修改 words → 立即重建派生 tier"的不变式。

### 修改点

**A. `_normalize_word_spellings` — NVV token 保护** (~line 1374)

```python
# 修改前
if ref_spelling != w_text and ref_spelling.isascii():
    words_tier.intervals[wi].text = ref_spelling

# 修改后 — 增加 NVV 检查
if is_nvv_token(w_text):
    continue  # 永远不改写 NVV token 的文本
if ref_spelling != w_text and ref_spelling.isascii():
    words_tier.intervals[wi].text = ref_spelling
```

**B. `strip_edge_punctuation` — 新增函数** (~line 852)

新增函数扫描 words tier，将首词前/尾词后的孤立标点吸收到相邻区间。在 Phase 4 中 `absorb_silence_into_punct` 之后调用。

**C. `build_pinyin_phones_tier` — 泄漏音素过滤** (~line 469)

```python
# 当第一个非静音音素起点超过词起点 30% 时长时，
# 判定为相邻词泄漏音素，清空 word_phones 走自引用 fallback
```

**D. `build_pinyin_phones_tier` — en_mfa_windows 缺失时的安全 fallback** (~line 521)

```python
# 修改前: en_mfa_windows.get(wl, (w_iv.xmin, w_iv.xmax))
# → 缺省窗口 = 整个词区间 + 0.3s margin → 泄漏音素全部通过

# 修改后: 若 wl 不在 en_mfa_windows 中，清空 word_phones
# → 走自引用 fallback，整词区间覆盖
```

**E. `_normalize_punct` — NVV 标签内部连字符保护** (~line 651, `ctc_prealign.py`)

```python
# 修改前 — '-' 被 is_punct 判为标点，非白名单 → 替换为 '，'
for ch in text:
    if is_punct(ch):
        char_info.append(("punct", ch in _NORM_ALLOWED_PUNCT, ch))

# 修改后 — 字母间的 '-' 是 NVV 标签连字符，不当作标点
for i, ch in enumerate(text):
    is_hyphen_in_nvv = (
        ch == '-'
        and i > 0 and i + 1 < len(text)
        and text[i - 1].isascii() and text[i - 1].isalpha()
        and text[i + 1].isascii() and text[i + 1].isalpha()
    )
    if is_punct(ch) and not is_hyphen_in_nvv:
        char_info.append(("punct", ch in _NORM_ALLOWED_PUNCT, ch))
```

此修复从源头阻止了 `SURPRISE-OH` → `SURPRISE，OH` 的转换。同批次
18 个含连字符 NVV 的文件全部受益。配合修改 A（NVV 文本保护），
形成双重防护。

**F. `_sync_derived_tiers` — 统一 tier 同步函数** (~line 879, 新增)

新增 `_sync_derived_tiers(textgrid, ...)` 函数，从当前 words + phones tier
重建 hanzi 和 pinyin_phones。在 Phase 4 的两个关键点调用：

1. **Phase 4 D4 之后**: D2/D3/D4 原地修改 words tier → 立即同步，确保后续 E/F/G
   读取的 pp_tier 是新鲜的
2. **Phase 4 末尾**: 确保进入 Phase 5 前所有派生 tier 与 words tier 一致

```python
def _sync_derived_tiers(textgrid, ipa_to_pinyin, pinyin_dict,
                         raw_text, en_mfa_windows, report_warnings):
    # 1. Rebuild hanzi from updated words tier
    # 2. Rebuild pinyin_phones from updated phones + words tiers
```

**架构原则**: words tier 是唯一权威数据源。hanzi 和 pinyin_phones 是派生 tier。
任何修改 words tier 边界/文本的操作，必须立即通过 `_sync_derived_tiers` 重建
派生 tier，不允许延迟到 Phase 5。"修改谁，同步全部"。

### 关联样本

- 花礼_emo来听小鼠歌吧_20250504_2058_3700c4b4_clip0003_clip0012: 触发全部三个子问题
- 此前 Cases Q, R, S, T (2026-07-27) 已部分修复轨道同步，但未覆盖 NVV 文本改写 + 泄漏音素场景

### 验证方法

```python
# 验证 NVV 保护
assert is_nvv_token("SURPRISE-OH")  # True — NVV 名称完整保留
assert not is_nvv_token("SURPRISE") # False — 被截断的形式不会被误判

# 验证边缘标点剥离
# 任何文件的 words tier 首 interval (非 silence/NVV) 不应是标点
# 任何文件的 words tier 尾 interval (非 silence) 不应是标点

# 验证音素无泄漏
# pinyin_phones 中 NVV token 的词区间内不应出现以 en: 前缀的音素
# (除非该词确实在 en_mfa_windows 中有条目)

# 验证 _normalize_punct 不再破坏 NVV 连字符
# 含连字符的 NVV (SURPRISE-OH, QUESTION-YI 等) 在 _text_cn.txt 中应保留 '-'
# 不应出现 SURPRISE，OH 形式的逗号拆分
```

---

## Case 18: NVASR 情绪标签后残留开头标点 → hanzi 首词为 `，`

**日期**: 2026-07-29
**涉及文件**: `scripts/ctc_prealign.py`
**触发样本**: 花礼_八分音符鼠_20241122_2201_d8293668_clip0000_clip0000

### 现象

```
ASR 输出:   <|zh|><|SAD|>， call恩你...
去除标签:                 ， call恩你...   ← 逗号成为第一个字符
hanzi:      ， | call | 恩 | ...           ← 首 interval 是标点
```

### 根因链

1. SenseVoice 在音频开头检测到情绪标签（`<|SAD|>`）后自动附加标点
2. `text_cn` 清洗只去标签不移除紧随其后的标点
3. CTC 为该标点分配 0s 锚点 → postprocess 注入 → hanzi 首词是标点

### 修改点

**A. `ctc_prealign.py` — 输出前检测并删除开头标点** (~L1196, ~L1277)

分两处：part A 从 `punct_entries` 删除首个标点锚点（start<100ms），part B 从 `text_asr` 中删除标点字符。后续词时间戳不平移。


## Case 19: `_snap_to_ctc` 丢弃开头静音 → hanzi 缺失 `<sp1>` interval

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`
**触发样本**: 花礼_八分音符鼠_20241122_2201_d8293668_clip0000_clip0000

### 现象

```
MFA raw:   <eps>[0~1.88]  call[1.88~...]     ← 有开头静音
最终输出:  call[1.95~...]                      ← 开头静音消失，raw_text 有 <sp1> 但 words/hanzi 无对应 interval
```

### 根因链

1. `_snap_to_ctc` 的 `mfa_words` 过滤掉所有 silence/`<eps>`（L3284），只处理内容词
2. 重建 words tier 时只处理了尾静音（L3610-3614），未补回开头静音
3. 第一个词起点前的 gap 没被填充 → 开头静音 interval 丢失

### 修改点

**A. `_snap_to_ctc` — 补回头部静音** (~L3610)

尾静音处理前插入：首个 word start>0.005s 时，在 `new_word_ivs` 头部补入 silence gap。

**B. `_finalize_textgrid` — 兜底插入** (~L148)

多 interval tier 首 interval 不从 0 开始且非 `<spN>` 时，补插 `<sp1>(0, first.xmin)`。确保即便上游丢弃了静音 interval，最终输出仍有 `<sp1>`。


## Case 20: strip_edge_punctuation 尾随标点过度剥离 → 句尾 。！？ 全部丢失

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `strip_edge_punctuation`, `process_one`
**触发样本**: 花礼_不困小鼠_20241117_2158_4ad6dee3_clip0023_clip0001

### 现象

所有 pipeline 输出的句尾标点（`。！？`）全部消失：

```
_text_cn.txt:     ...知道什么意思吧？            ← 有 ？
TextGrid raw_text: ...知道什么意思吧              ← ？ 消失
TextGrid words:    ... ba5[10.22-10.51] <sp2>...  ← 无 ？ interval
TextGrid hanzi:    ... 吧                          ← 无 ？
```

受影响的 87 个文件中，几乎所有以 `。！？` 结尾的句子都丢失了结尾标点。

### 设计缺陷："镜像处理"的逻辑谬误

`strip_edge_punctuation` 是在 Case 17-B 中新增的函数，其设计意图是剥离**孤儿标点**——
NVASR 去除情绪标签（`<|HAPPY|>`、`<|zh|>` 等）后，标签之间的标点（主要是 `…`）
残留在文本开头/结尾，被 CTC 分配时间锚点后注入 words tier，成为孤立的首/尾标点 interval。

函数的处理逻辑是对称的：

```
for i in range(0, first_real):         ← 开头: 第一个实词前的 punct → 剥离
    if is_punct(intervals[i].text): ...
for i in range(last_real + 1, len):    ← 尾随: 最后一个实词后的 punct → 剥离
    if is_punct(intervals[i].text): ...
```

**开头剥离是正确的**：在第一个实词之前，不应该存在任何标点。
如果出现（情绪标签剥离残留），它必然是孤儿，应该被吸收。

**尾随剥离是错误的**：在最后一个实词之后，标点分两种：
1. **孤儿 `…`**：NVV 标签剥离残留 → 但情绪标签几乎只在句首，不在句尾。即使标签在句尾（极其罕见），`你好<|HAPPY|>…` 去标签后变 `你好…`——这与正常句尾省略号**无法区分**，也不需要剥离
2. **合法句尾标点 `。！？…`**：ASR 原始文本的句末标点 → **必须保留**

尾随剥离从 Case 17-B 引入之初就没有需要解决的实际 bug——它是作为开头剥离的"镜像"顺手加上的。开头剥离有明确的孤儿标点场景（情绪标签在句首），尾随剥离没有。两者不是对称场景。

**为什么"镜像"在此不成立**：

句子结构是 `[内容词...] [标点]`，标点天然在句尾。开头的标点一定是异常的，
但尾随的标点大多数是合法的。开头和尾随不是对称场景，镜像处理是设计错误。

### 根因链

1. **Case 17-B 引入 `strip_edge_punctuation`** (~L938)：解决 NVASR 剥离 `<|HAPPY|>` 情绪标签后残留 `…` 成为首词的问题。设计为"镜像处理"——开头的孤儿标点和尾随的标点一起剥离
2. **镜像假设错误**：句子结构是 `[内容] [标点]`，标点天然在末尾。开头标点必然是异常，尾随标点大多数是合法的。两者不是对称场景
3. **尾随剥离用 `is_punct()` 一刀切** (~L997-1011)：从 `last_real`（最后一个实词）之后扫描所有 interval，凡是 `is_punct()` 的一律清空，无法区分孤儿 `…` 和合法句尾标点 `。！？`
4. **absorber 链已覆盖尾随清理**：`absorb_nvv_trailing` (Case 9 W1, Phase 4-D2) 和 `absorb_silence_into_punct` (Case 9 W2, Phase 4-D3) 在 `strip_edge_punctuation` (Phase 4-D4) **之前**执行，已经处理了 NVV 后的标点+静音链。尾随剥离不仅是逻辑错误的，而且是**冗余的**
5. **Phase 5 不可逆传播** (~L4654)：`raw_text` 从 `hanzi` tier 重建覆盖 → 被剥离的标点永久消失，无法恢复

### 完整数据流追踪

以 `花礼_不困小鼠_clip0023` 为例，追踪 `？` 的完整生命周期：

```
Phase 1-2:  _text_cn.txt → raw_text 变量 → raw_text tier
            "...知道什么意思吧？"                       ✅ ？ 存在

Phase 3C:   _inject_punctuation: CTC ？[11.07-11.16] 注入 words tier
            words: ... ba5[10.22-10.51] <sp2>[10.51-11.125] ？[11.125-11.16]
            last_punct → punct_start=ba5.xmax=10.51
            ？ → (10.51, 11.125, "？", "punct")  [615ms]  ✅ ？ 存在

Phase 4-D2: absorb_nvv_trailing — 无 NVV → 无操作

Phase 4-D3: absorb_silence_into_punct — 无残余 <spN> → 无操作

Phase 4-D4: strip_edge_punctuation  ← 问题发生点！
            last_real = ba5 的 index
            for i in range(last_real+1, len):    # 扫描 ba5 之后
              <sp2> → is_punct? NO  → skip
              ？   → is_punct? YES → trailing_punct_indices.append(i)
            → ？ 被清空: text="", xmin=0, xmax=0
            → <sp2>.xmax 扩展到 11.16 (吸收 ？ 的时间)
                                               ❌ ？ 被剥离！

Phase 5:    _build_hanzi_tier(words) → hanzi 无 ？
            raw_text = "".join(hanzi) → 覆盖 raw_text tier
                                               ❌ ？ 永久消失！
```

### 触发条件

任何满足以下条件的句子：
- 最后一个实词后存在标点（`。！？`，经 `_inject_punctuation` 注入 words tier）
- `strip_edge_punctuation` 将该标点识别为 "trailing punct" → 清空

### 修改点

**A. `strip_edge_punctuation` — 删除尾随标点剥离逻辑** (~L997-1017)

尾随剥离从 Case 17-B 引入之初就是错误的——它作为开头剥离的"镜像"被顺手加上，
但没有任何实际 bug 驱动。情绪标签几乎只在句首（`<|HAPPY|>…内容`），不在句尾。
句尾的标点都是合法的句末标点，不应被剥离。

修改前 (~L997-1011, 与开头剥离对称的镜像代码):
```python
# ── Strip trailing punct: absorb into the following interval ──
trailing_punct_indices = []
for i in range(last_real + 1, len(intervals)):
    if is_punct(intervals[i].text):
        trailing_punct_indices.append(i)

for pi in sorted(trailing_punct_indices, reverse=True):
    p_iv = intervals[pi]
    if pi + 1 < len(intervals):
        intervals[pi + 1] = _replace(intervals[pi + 1], xmin=p_iv.xmin)
    elif pi > 0:
        intervals[pi - 1] = _replace(intervals[pi - 1], xmax=p_iv.xmax)
    intervals[pi] = _replace(intervals[pi], xmin=0, xmax=0, text="")
```

修改后（整个尾随剥离代码块删除，替换为注释说明）:
```python
    # NOTE: There is intentionally NO trailing strip.
    # Trailing punctuation (。！？…) after the last real word is ALWAYS
    # legitimate — sentences naturally end with punctuation.  The "mirror"
    # design (stripping both edges) is a logical error because leading
    # and trailing edges are NOT symmetric:
    #   - Leading punct: always orphaned (tag-stripping artifact) → strip
    #   - Trailing punct: always legitimate (end-of-sentence) → keep
    # NVV-trailing punct+silence chains are already handled upstream by
    # absorb_nvv_trailing (Case 9 W1) and absorb_silence_into_punct (Case 9 W2).
```

函数现在只做一件事：剥离开头（first_real 之前的）孤儿标点。这是它唯一的实际职责。

**B. `_inject_punctuation` — 最后标点 30ms 地板保护** (~L2482-2484, 防御性)

修改前：
```python
new_merged.append((punct_start, words_tier.xmax, punct_text, "punct"))
```

修改后：
```python
punct_end = max(punct_start + 0.030, words_tier.xmax)
new_merged.append((punct_start, punct_end, punct_text, "punct"))
```

防止末词 xmax == words_tier.xmax 时标点坍缩为零时长（独立于本 Case 的边缘保护）。

### 与关联 Case 的关系

| Case | 问题 | 与本 Case 的关系 |
|------|------|----------------|
| 9 (W1/W2) | NVV后标点+静音链未被吸收 | `absorb_nvv_trailing` + `absorb_silence_into_punct` 已覆盖，尾随剥离冗余 |
| 17-B | `…` 为首词（NVV 标签剥离残留） | 本 Case 的根因——17-B 引入 `strip_edge_punctuation`，尾随剥离过于激进 |
| 18 | 情绪标签后残留开头标点 | ctc_prealign 层面修复，不依赖 postprocess 剥离 |

### Phase 4 执行顺序与责任分工

```
Phase 4 执行顺序:
  D.  handle_unexpected_silences    ← gap 级 <sp0> 无条件合并 (Case 8)
  D2. absorb_nvv_trailing           ← NVV 吞并尾部标点+静音链 (Case 9 W1)
  D3. absorb_silence_into_punct     ← 兜底: 标点吸收残余 <spN> (Case 9 W2)
  D4. strip_edge_punctuation        ← 边缘标点剥离 (Case 17-B)  ← 本 Case
  ── SYNC ── _sync_derived_tiers   ← words → hanzi + pp 同步 (Case 17-F)
```

D2 已经处理了 "NVV → 标点 → 静音" 链（如 `<LAUGHTER> → ！ → <sp2>`），
D3 兜底处理了 "标点 → 残余 `<spN>`"。到 D4 执行时，所有 NVV 相关的尾随孤儿标点
已经被 D2 吸收。D4 扫到的尾随标点**只可能是合法的句末标点**。

D4 的尾随剥离不仅逻辑错误（镜像假设不成立），而且在执行顺序上也是冗余的。

### 验证方法

```python
# 句尾标点必须在 words tier 中存在
for iv in words_tier.intervals:
    if iv.text.strip() in '。！？':
        assert iv.xmax > iv.xmin + 0.001, f"zero-duration ending punct: {iv.text}"

# raw_text tier 结尾标点应与 _text_cn.txt 一致
assert raw_text_raw.endswith('？') == text_cn_raw.endswith('？')
```

### 关联样本

- `花礼_不困小鼠_20241117_2158_4ad6dee3_clip0023_clip0001`：`知道什么意思吧？` → `？` 丢失
- 花礼筛选全部 87 文件：句尾 `。！？` 系统性丢失


## Case 21: `is_punct` 误判 `<spN>` 为标点 → `strip_edge_punctuation` 开头静音被吸收进首词

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/pipeline_utils.py`
**涉及函数**: `strip_edge_punctuation`, `is_punct`
**触发样本**: 花礼_变身_小鼠天使_20240924_2105_a90ca312_clip0010_clip0001

### 现象

首词 `喵` 从 0s 开始，开头静音 `<sp2>` 完全消失：

```
修复前:
  words:  miao1[0.000-1.830]  …[1.830-2.010]  hao3[2.010-...]
  hanzi:  喵[0.000-1.830]      …[1.830-2.010]  好[2.010-...]
          ↑ 开头静音消失，首词从 0s 开始

修复后:
  words:  <sp2>[0.000-1.410]  miao1[1.410-1.830]  …[1.830-2.010]  hao3[2.010-...]
  hanzi:  <sp2>[0.000-1.410]  喵[1.410-1.830]      …[1.830-2.010]  好[2.010-...]
          ↑ 开头静音正确保留
```

MFA 对齐中 `<eps>` [0.000-1.940] 正确存在于开头，但最终输出中消失。

### 根因链

1. **`is_punct()` 定义过于宽泛** (`pipeline_utils.py` L771-773)：
   ```python
   def is_punct(s: str) -> bool:
       return bool(s.strip()) and not is_word_like(s)
   ```
   `<sp2>` 以 `<` 开头，不是 alpha/CJK/digit，所以 `is_word_like("<sp2>")` → False，进而 `is_punct("<sp2>")` → True。

2. **`strip_edge_punctuation` 开头剥离未排除静音** (L983-985)：开头剥离扫描 `first_real` 之前的所有 interval，发现 `<sp2>` → `is_punct` True → 加入剥离列表

3. **剥离逻辑将静音吸收进首词** (L987-995)：`<sp2>` 是第一个 interval（pi=0），无前序 interval → `elif pi+1 < len`：下一个 interval (`miao1`) 的 xmin 被设为 `<sp2>.xmin = 0.0`

### 与 Case 20 的同源性

Case 20 和 Case 21 的根因都指向同一个设计问题：

| | Case 20（尾随） | Case 21（开头） |
|------|------|------|
| 被剥离对象 | `。！？`（合法句尾标点） | `<sp2>`（开头静音标签） |
| `is_punct` 返回值 | True（正确——确实是标点） | True（**错误**——静音不是标点） |
| 问题本质 | 设计错误：不应剥离尾随 | **bug**：`is_punct` 定义太宽，未排除 `<spN>` |

两者都在 `strip_edge_punctuation` 中触发，但性质不同。Case 20 已经通过删除尾随剥离解决，Case 21 需要修复开头剥离中的 `is_silence` 前置检查。

### 修改点

**A. `strip_edge_punctuation` — 开头剥离增加 `is_silence` 检查** (~L983-985)

修改前：
```python
leading_punct_indices = []
for i in range(first_real):
    if is_punct(intervals[i].text):
        leading_punct_indices.append(i)
```

修改后：
```python
leading_punct_indices = []
for i in range(first_real):
    if not is_silence(intervals[i].text) and is_punct(intervals[i].text):
        leading_punct_indices.append(i)
```

`<spN>` 静音标签在 `is_punct` 检查之前被 `is_silence` 排除，不会被误剥离。

### Case 20 + 21 的综合修复

`strip_edge_punctuation` 函数最终的职责范围：

```
strip_edge_punctuation:
  ├─ 开头剥离: first_real 之前的 punct (排除 <spN>) → 吸收   ← Case 21 修复
  └─ 尾随剥离: 已删除                                       ← Case 20 修复
```

这两个修复一起，使函数只做它最初设计要做的一件事：剥离情绪标签去除后残留在句首的孤儿 `…`。

### 关联样本

- `花礼_变身_小鼠天使_20240924_2105_a90ca312_clip0010_clip0001`：`<sp2>` 被吃 → 喵从 0s 开始
- 花礼筛选全部 87 文件：所有句首静音均受影响

### 潜在其他影响范围

`is_punct` 的宽泛定义可能在其他地方造成类似问题。任何仅用 `is_punct` 而未先检查 `is_silence` 的地方，都可能将 `<spN>` 误判为标点。搜索代码库中所有 `is_punct()` 调用可审计。


## Case 22: MFA 对齐偏差 → snap 修正后标点被 `<spN>` 替换 → 被吞标点恢复

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (Phase 5 末尾)
**触发样本**: VR研学_小鼠粽子喜欢甜的还是咸的_20250531_2222_b330ee29_clip0010_clip0007, 3.15s 处的逗号

### 现象

ASR 识别出标点，CTC 有标点锚点，`_text_cn.txt` 中有标点，但最终 words tier 中标点消失，变成了 `<spN>` 静音间隙：

```
_text_cn.txt:   "...迪士尼，谢谢..."       ← ASR 有逗号
CTC punct:       ，[3.245-4.710]           ← 有锚点
最终 words:      ni2[2.75-3.15] <sp2>[3.15-3.81] xie4[3.81-4.05]
                 ↑ 逗号消失，变成 <sp2>
```

### 根因链

这是一个多步骤连锁反应：

1. **MFA 对齐偏差**：MFA 将 `xie4` 错放到 [3.15-3.26]（仅 110ms，实际应该是 [3.81-4.05]），导致 `ni2` 和 `xie4` 之间出现本不应存在的大段 `<eps>`

2. **`_snap_to_ctc` 修正词边界**：检测到 CTC 时长 > MFA 时长 ×2，把 `xie4` 拉到 CTC 正确位置 [3.81-4.05]。`ni2` [2.75-3.15] 和 `xie4` [3.81-4.05] 之间出现 660ms 间隙 → 被 `<sp2>` 填充。**词边界修正了，但间隙中的标点被埋没了**

3. **`_inject_punctuation` 重叠裁剪跳过静音**：`，` [3.245-4.710] 与 `<sp2>` [3.15-3.81] 重叠，但 `<sp2>` 因 `is_silence=True` 被重叠裁剪逻辑跳过（L2338-2340），逗号只与后续 `xie4` 发生裁剪，被推到 [4.29-4.71]

4. **逗号被孤立**：推到 [4.29-4.71] 的逗号位于 `xie4` 和 `shou1` 之间，但后续 Phase 4 步骤中可能被某个操作误吸收（具体步骤待精确追踪确认）。最终 `shou1` 从 CTC [4.71-4.95] 变成 [4.29-4.95]——逗号的时间被吸收了

5. **间隙变回 `<spN>`**：逗号消失后，`xie4` 和 `shou1` 之间的间隙被重新标记为静音

### 设计认知

这个 bug 的根源不是某个步骤的代码错误，而是**两种修正逻辑的交互冲突**：

| 步骤 | 做什么 | 副作用 |
|------|--------|--------|
| `_snap_to_ctc` | 把词边界拉到 CTC 锚点（修正 MFA 偏差） | 词之间产生新的间隙，埋没原本覆盖在上的标点 |
| `_inject_punctuation` | 把 CTC 标点注入 words tier | 重叠裁剪时 `<spN>` 不参与，标点被推向词边界之外 |

两者各自正确，但交互产生了一个"无主之地"——MFA 对齐偏差导致的间隙中，标点被推向边界后孤立，最终被后续 absorb 步骤吃掉。

### 修复策略

**事后兜底**（不改变现有逻辑，在 Phase 5 末尾做最后检查）：

在所有处理完成后，扫描 words tier 中的 `<spN>` 区间。若 CTC 标点锚点落在该区间内，且该标点未出现在 words tier 的其他位置，则直接将 `<spN>` 替换为该标点字符，时长不变。然后同步重建 hanzi、pinyin_phones、raw_text。

选择"事后兜底"而非"修改前置逻辑"的原因：
- `_snap_to_ctc` 的边界修正逻辑已经过 19 个 Case 的验证，不宜改动
- `_inject_punctuation` 跳过 `<spN>` 是刻意的（静音不参与标点裁剪），改动风险大
- 兜底修复只影响"标点被吞"这一种已确认的失效模式，不改变正常路径

### 修改点

**A. `process_one` Phase 5 — 被吞标点恢复** (~L4665 之后，QC 之前)

```python
# ── 被吞标点恢复 ──
# 若 <spN> 区间包含 CTC 标点锚点, 且该标点未在 words 中,
# 则替换 <spN> 为该标点, 时长不变, 同步所有派生 tier.
if punct_entries:
    _words_t = tier_by_name(new_tg, "words")
    if _words_t:
        _existing_punct = {iv.text.strip() for iv in _words_t.intervals
                           if is_punct(iv.text) and not is_silence(iv.text)}
        for iv in _words_t.intervals:
            if not is_silence(iv.text): continue
            for p in punct_entries:
                if p["word"] not in '，。！？': continue
                if p["word"] in _existing_punct: continue
                overlap = min(iv.xmax, p["end_s"]) - max(iv.xmin, p["start_s"])
                if overlap > 0:
                    iv.text = p["word"]
                    _existing_punct.add(p["word"])
                    # → 同步重建 hanzi, pinyin_phones, raw_text
                    break
```

### 与关联 Case 的关系

| Case | 问题 | 与本 Case 的关系 |
|------|------|----------------|
| 1-3 | MFA 尾静音被 snap 回词 | 同属 snap→间隙→标点问题的前置条件 |
| 7 (Mod T) | ≤5ms 间隙吸收 | snap 后间隙处理的同族逻辑 |
| 12 | overlap fix 产生倒置 interval | snap 副作用导致标点异常 |
| 20 | strip_edge_punctuation 尾随剥离 | 另一个导致标点消失的路径（已修复） |

### 验证方法

```python
# 所有 CTC 标点锚点必须在 words tier 中有对应
for p in punct_entries:
    if p["word"] in '，。！？':
        found = any(iv.text.strip() == p["word"] and
                    abs(iv.xmin - p["start_s"]) < 0.5
                    for iv in words_tier.intervals)
        assert found, f"CTC punct {p['word']} at {p['start_s']}s not in words tier"
```

### 关联样本

- `VR研学_小鼠粽子喜欢甜的还是咸的_20250531_2222_b330ee29_clip0010_clip0007`：3.15s 逗号被吞

### 被吞标点恢复的触发条件（精确版）

```
① CTC 标点锚点存在于 _punct.json（ASR 识别到了）
② _inject_punctuation 后该标点在 words tier 中（快照确认）
③ Phase 4 后该标点从 words tier 消失（比对确认 → _swallowed_puncts）
④ 该位置现在是 <spN>（间隙被重新暴露）
→ 四个条件全部满足 → 替换 <spN> 为原标点
```

只恢复"确认被吞"的标点，不会误恢复从未存在过的标点。


## Case 23: `step_normalize_punct` 缺少 NVV 连字符保护 → `QUESTION-EI` 被拆成 `QUESTION，EI`

**日期**: 2026-07-29
**涉及文件**: `scripts/run_pipeline.py`
**涉及函数**: `step_normalize_punct`
**触发样本**: 花礼_和帕小聊_20241221_0914_7103ddf1_clip0016_clip0008, 6.99s 处的 `QUESTION-EI`

### 现象

含连字符的 NVV token `QUESTION-EI` 在 pipeline 处理后被拆散：

```
修复前:
  words:         <QUESTION-EI>[6.99-7.29]  EI[7.29-7.38]        ← EI 独立成词
  hanzi:         QUESTION[6.99-7.29]       EI[7.29-7.38]        ← 无 <>，被拆开
  pinyin_phones: <<QUESTION-EI>>[6.99-7.29]  en:ei[7.29-7.38]   ← 双重包裹 + 音素泄漏

修复后:
  words:         <QUESTION-EI>[6.99-7.29]                       ← 完整 NVV
  hanzi:         QUESTION-EI[6.99-7.29]                         ← 正确
  pinyin_phones: <QUESTION-EI>[6.99-7.29]                       ← 正确
```

### 根因链

1. **NVASR 输出** `[Question-ei]`（NVV 方括号格式），`ctc_prealign` 转换为 `QUESTION-EI`
2. **`.lab` 文件**中 `QUESTION-EI` 保留连字符（不经过 `step_normalize_punct`）
3. **`_text_cn.txt`** 经过 `step_normalize_punct`（Phase 2 标点分类）：
   ```python
   for ch in text:
       if is_punct(ch):  # '-' 命中! is_punct 定义为 not is_word_like
           char_info.append(("punct", ch in ALLOWED_PUNCT, ch))
           # '-' 不在白名单 → 后续被替换为 ，
   ```
4. **`_text_cn.txt` 变成** `QUESTION，EI`（连字符变逗号）
5. **MFA 对齐**：`.lab` 中有 `QUESTION-EI EI`（lab 不受影响），MFA 将 `EI` 当作独立词
6. **Postprocess**：`EI` 被 `is_english_token` 识别 → 送入 English MFA → 泄漏 `en:ei` 音素
7. **Hanzi tier**：`QUESTION` 不是 NVV 格式（无 `<>`），`_build_hanzi_tier` 将其当作普通英文词

### Case 17-E 修复不完整

Case 17-E 在 `ctc_prealign._normalize_punct` 中添加了 NVV 连字符保护，但 **`run_pipeline.step_normalize_punct` 没有被同步修复**。两个函数做同样的标点规范化，但只有前者有 `is_hyphen_in_nvv` 检查。

```
ctc_prealign._normalize_punct     ← Case 17-E 已修复 ✅
run_pipeline.step_normalize_punct ← 遗漏！ ❌ → Case 23
```

### 修改点

**A. `step_normalize_punct` — Phase 2 增加 `is_hyphen_in_nvv` 检查** (~L680)

修改前：
```python
for ch in text:
    if is_punct(ch):
        char_info.append(("punct", ch in ALLOWED_PUNCT, ch))
    else:
        char_info.append(("other", None, ch))
```

修改后：
```python
for i, ch in enumerate(text):
    is_hyphen_in_nvv = (
        ch == '-'
        and i > 0 and i + 1 < len(text)
        and text[i - 1].isascii() and text[i - 1].isalpha()
        and text[i + 1].isascii() and text[i + 1].isalpha()
    )
    if is_punct(ch) and not is_hyphen_in_nvv:
        char_info.append(("punct", ch in ALLOWED_PUNCT, ch))
    else:
        char_info.append(("other", None, ch))
```

### 全面审计

对 pipeline 中所有处理 `_text_cn.txt` 标点规范化的函数进行了审计：

| 函数 | 文件 | 连字符保护 | 状态 |
|------|------|-----------|------|
| `_normalize_punct` | ctc_prealign.py | ✅ Case 17-E | 已修复 |
| `step_normalize_punct` | run_pipeline.py | ✅ Case 23 | **本次修复** |
| `normalize_punct_inline` | pipeline_utils.py | N/A（`-` 不在其处理范围） | 安全 |

### 关联样本

- `花礼_和帕小聊_20241221_0914_7103ddf1_clip0016_clip0008`：`QUESTION-EI` → `QUESTION，EI` + `EI` 独立成词


## Case 24: ASR 空输出先写 TextGrid 再检测 → 留下孤立的空 TextGrid

**日期**: 2026-07-29
**涉及文件**: `scripts/ctc_prealign.py`
**涉及函数**: `main` (推理循环 ~L1193-1218)
**触发样本**: `花礼_-晚间吃口小鼠_20251120_2059_365e41bc_clip0017_clip0019` (raw_text=`<sp1>`，纯静音)

### 现象

ASR 输出为空的 stem 在 `ctc_pretg/` 中留下孤立的空 TextGrid，但没有 `.lab`、`_tokens.jsonl`、`_punct.json`、`_text_cn.txt`、`_text_raw.txt`：

```
ctc_pretg/
  ├─ 花礼_-晚间吃口...TextGrid   ← 孤立的空 TG（words tier 仅一个空 interval）
  ├─ （没有 .lab）
  ├─ （没有 _tokens.jsonl）
  ├─ （没有 _punct.json）
  ├─ （没有 _text_cn.txt）
  └─ （没有 _text_raw.txt）
```

### 根因链

1. **写 TextGrid 在空检测之前**（L1193-1194）：无论 ASR 是否有输出，`write_textgrid()` 都在 `continue` 之前执行
2. **空检测在 L1213-1218**：`lab_tokens.strip()` 为空时打印 SKIP 并 `continue`，跳过 `.lab` 等后续文件写入
3. **下游步骤通过不同文件类型发现 stem**：`adjust_ctc` glob `*_tokens.jsonl`，`normalize_*` glob `*_text_cn.txt` / `*.lab`，MFA align 用 `.lab` 作 corpus → 空 stem 被自然排除
4. **实际无功能影响**：下游从不触碰这个孤立的空 TextGrid，不影响管线结果
5. **但造成不一致**：87 个 TextGrid vs 86 个其他文件，目录状态不一致

### 修改点

**AQ. `ctc_prealign.py` — TextGrid 写入移至空检测之后** (~L1193, ~L1217)

修改前：
```python
# L1193 — 空检测之前写 TextGrid
out_tg = args.output_dir / f"{stem}.TextGrid"
write_textgrid(words_pinyin, duration_s, out_tg, pauses=pauses)

# L1213 — 空检测，跳过后续文件
lab_tokens = " ".join(w["word"] for w in words_pinyin)
if not lab_tokens.strip():
    print(f"  SKIP {stem}: ASR produced no text — skipping MFA alignment")
    skipped.setdefault("empty_asr", []).append(stem)
    continue

# 写 .lab ...
```

修改后：
```python
# L1210 — 空检测，全部跳过（含 TextGrid）
lab_tokens = " ".join(w["word"] for w in words_pinyin)
if not lab_tokens.strip():
    print(f"  SKIP {stem}: ASR produced no text — skipping MFA alignment")
    skipped.setdefault("empty_asr", []).append(stem)
    continue

# L1217 — TextGrid 和 .lab 一起写入
out_tg = args.output_dir / f"{stem}.TextGrid"
write_textgrid(words_pinyin, duration_s, out_tg, pauses=pauses)

out_lab = args.output_dir / f"{stem}.lab"
out_lab.write_text(lab_tokens + "\n", encoding="utf-8")
```

### 关联样本

- `花礼_-晚间吃口小鼠_20251120_2059_365e41bc_clip0017_clip0019`：16s 纯静音/噪音，ASR 输出 `<sp1>`


## Case 25: `_inject_punctuation` 标点包含词时被误删 + `_punct_before` 漏追踪 `…` + 恢复匹配仅按时间

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_inject_punctuation`, process_one (吞标点恢复逻辑)
**触发样本**: `花礼_呼噜呼噜小鼠毛_20250326_2058_cd0aeff6_clip0019_clip0027` (5.37s, `…` 被 `<sp2>` 替代 → mid_sp 过滤)

### 现象

CTC 预对齐在 `bu4` 和 `zhi3` 之间插入 900ms 的 `…`（省略号/长停顿标记），但最终 words tier 中 `…` 消失，被 `<sp2>` 替代，触发 `mid_sp` 过滤：

```
CTC tokens+punct:  bu4[4.35-5.37]  …[5.37-6.27]  zhi3[6.27-6.39]
MFA aligned:       ...wo3, bu4[5.49-5.60], <eps>[5.60-6.05], zhi3[6.05-6.13]...
最终 words tier:   bu4[4.35-5.37]  <sp2>[5.37-6.065]  zhi3[6.065-6.26]
                                                        ↑ … 变成 <sp2>！
```

### 根因链

**Bug A — `_inject_punctuation` 重叠判断缺分支 (~L2338-2347)**

四分支重叠判断覆盖了三种正常情况，但 `else` 分支同时包含两种相反的场景：

| 场景 | 条件 | 实际含义 | 旧行为 | 正确行为 |
|------|------|---------|--------|---------|
| punct 在 word 内 | ws≤ps ∧ we≥pe | word 包含 punct | 删除 ✓ | 删除 |
| word 左盖 punct | ws≤ps ∧ we<pe | 左侧重叠 | trim ✓ | trim |
| word 右盖 punct | ws>ps ∧ we≥pe | 右侧重叠 | trim ✓ | trim |
| ***word 在 punct 内*** | ws>ps ∧ we<pe | **punct 包含 word** | **删除 ❌** | **split** |

本例中 `…` [5.37-6.27] 包含 `zhi3` [6.065-6.26]（`5.37 < 6.065` 且 `6.27 > 6.26`），走 `else` 分支直接删除，900ms 的 `…` 全部丢失。

**Bug B — `_punct_before` 监控范围过窄 (~L4361)**

吞标点检测只追踪 `，。！？` 四种标点，`…`（省略号）被排除在外：

```python
if p["word"] not in '，。！？':   # ← …、；：等全被漏掉
    continue
```

即使 `…` 在 Phase 3.C 成功注入后被 Phase 4 吸收吞掉，也不会进入 `_swallowed_puncts`，恢复逻辑完全看不到它。

**Bug C — 吞标点恢复按时间重叠匹配而非顺序 (~L4726)**

旧代码用时间区间重叠匹配被吞标点和 `<spN>`：

```python
overlap = min(iv.xmax, p["end_s"]) - max(iv.xmin, p["start_s"])
if overlap > 0:
    iv.text = p["word"]
```

当 MFA 对齐偏差导致 `<spN>` 的边界与 CTC punct 锚点偏差较大时，重叠为 0 或负，匹配失败。

### 修改点

**AR. `_inject_punctuation` — else 分支改为 split 而非 delete** (~L2346)

修改前：
```python
            else:
                resolved[pi] = (0, 0, "", pkind)
```

修改后：
```python
            else:
                # word inside punct (ws > ps and we < pe):
                # split punct into left part (before word) + right part (after word)
                left_part  = (ps, ws, ptext, pkind)
                right_part = (we, pe, ptext, pkind)
                resolved[pi] = left_part
                if right_part[1] > right_part[0] + 0.001:
                    resolved.append(right_part)
```

效果：`…` [5.37-6.27] 被 `zhi3` [6.065-6.26] 切开 → `…` [5.37-6.065] + `zhi3` [6.065-6.26] + `…` [6.26-6.27]

**AS. `_punct_before` — 扩展追踪标点集合** (~L4361, ~L4701)

修改前：`if p["word"] not in '，。！？': continue`

修改后：`if p["word"] not in '，。！？…、；：': continue`

两处相同守卫条件同步修改。

**AT. 吞标点恢复 — 时间匹配改为按 CTC 序列顺序匹配** (~L4717-4748)

修改前按时间重叠匹配：
```python
for iv in _words_t.intervals:
    if not is_silence(iv.text): continue
    overlap = min(iv.xmax, p["end_s"]) - max(iv.xmin, p["start_s"])
    if overlap > 0: ...
```

修改后按 CTC 序列中的前后邻词匹配：
```python
# Build CTC timeline: all tokens + puncts sorted by start
_ctc_timeline = [(kind, word, start_s), ...]

for p in _swallowed_puncts:
    # Find p's neighbors in CTC sequence (prev_word, next_word)
    # Walk words tier: find <spN> between same prev_word/next_word
    for i in range(1, len(_word_ivs) - 1):
        iv = _word_ivs[i]
        if not is_silence(iv.text): continue
        if _word_ivs[i-1].text == prev_word and _word_ivs[i+1].text == next_word:
            iv.text = p['word']  # sequential match → replace
```

### 关联样本

- `花礼_呼噜呼噜小鼠毛_20250326_2058_cd0aeff6_clip0019_clip0027`：`…` [5.37s] 被 zhi3 包含 → 被删 → `<sp2>` → mid_sp 过滤

### 补充修改 (同日): 恢复逻辑中 `tokens` 变量作用域修复

修改 AT 在吞标点恢复中引用 `tokens` 变量，但该变量在 `process_one` 中名为 `ctc_tokens` 且只在 `if tokens_path.exists():` 块内定义，恢复逻辑位置在其作用域外 → 32 条文件报 `NameError: name 'tokens' is not defined`。

修复：恢复逻辑不从外部引用 `tokens`/`ctc_tokens`，改为直接从 `_tokens.jsonl` 重新读取构建 CTC timeline。

### 补充修改 (同日): BGM 检测噪声底污染 + 帧级别检测

**触发样本**: `花礼_emo来听小鼠歌吧_20250504_2058_3700c4b4_clip0003_clip0012`（`<sp1>` [0-1.71s] 内含巨响 RMS 3000-7000，但未被过滤）

**Bug D — 60 分位噪声底被高能量内容污染**:

bgm 检测用全音频 RMS 的 60 分位作为噪声底：
```python
k = max(1, int(len(all_rms) * 0.6))
nf_bgm = float(np.partition(all_rms, k)[k])
```

当 `<sp1>` 区间有语音级能量时，60 分位被拉高 → `bgm_threshold` 被拉到语音级别 → 异常静音反而检不出。**声音越大越检不出**。

修复：添加 `bgm_max_threshold`（默认 0.05）作为阈值天花板：
```python
bgm_threshold = min(bgm_threshold, args.bgm_max_threshold)
```

**Bug E — `_word_rms()` 整段平均掩盖局部高能量**:

旧代码对 silence interval 取整段 mean absolute amplitude。长静音中夹杂短时巨响时，平均值被前后静音稀释（0.111 → 0.026），不触发检测。

修复：改为 50ms 帧级别扫描，统计超阈值帧占比 ≥20% 才标记为可疑：
```python
# 50ms frames with 25ms hop
for each frame in silence interval:
    frame_energy = mean(abs( audio[frame_start:frame_end] ))
    if frame_energy > bgm_threshold:
        high_frames += 1
if high_frames / n_frames >= 0.20:
    suspect  # sustained high energy, not a transient click
```

**配置变更**:
- `run_pipeline.py` DEFAULT_CFG: 新增 `bgm_max_threshold: 0.05`
- `config.yaml`: 新增 `bgm_max_threshold: 0.05` 参考项
- `postprocess_textgrids.py`: 新增 `--bgm-max-threshold` 参数


### 补充修改 (同日): 末尾标点强制保留

**触发样本**: `直播回放_zzZ_2026年06月03日14点场_ae037890_clip0008_clip0012`（`醉…BREATHING。`→ BREATHING 吸收末尾 `。`→ 句末无标点）

**Bug F — 末尾标点被 NVV 吸收后无法恢复**:

CTC 序列 `… BREATHING 。` 中 `。` 是最后一项。Phase 4 D2 中 BREATHING 吸收尾部 `。`，吞标点恢复逻辑因 `next_word = None` 跳过末尾标点。

修复：在吞标点恢复之后、最终 QC 之前，新增**末尾标点强制保留**步骤：
1. 找 CTC 最后一个标点（按 `end_s`）
2. 若 words tier 末尾不是该标点 → 从前词末尾截取 ≥60ms
3. 截取量 = `max(0.060, CTC原始时长)` — 弹性，至少 60ms
4. 同步更新 words、hanzi、pinyin_phones 三个 tier

```python
_carve_s = max(0.060, (_last_punct_end - _last_punct["start_s"]))
# Trim last word xmax to _punct_start, append punct interval
# Sync all three tiers: words, hanzi, pinyin_phones
```

**同步修复**: 初版 hanzi tier 只 append 不 trim → 旧 BREATHING 与 `。` 重叠。改为统一处理：先 trim 末位 interval 的 xmax，再 append 标点。

### 补充修改 (2026-07-30): 标点边界保护 — 防止 dead silence 延伸覆盖标点

**触发样本**: `直播回放_三周年纪念活动进行中_2025年12月04日20点场_4d04746b_clip0010_clip0001`（`zai4` 被延伸至 5.47s 覆盖 `…` [4.95-5.43]→ `…` 被 `_inject_punctuation` 当作"word 包含 punct"删除）

**Bug G — `_refine_boundaries_by_energy` dead silence 延伸未检测与词尾重叠的标点**:

CTC 标点起点（4.95s）略早于 MFA 词尾（5.02s）→ 70ms 重叠。Dead silence 检查 `iv.xmax <= p["start_s"]` 失败，延伸未阻断 → `zai4` 延至 5.47 → Phase 3.C 中 `…` 被删。

修复：新增标点边界邻近检测：
```python
_near = abs(p["start_s"] - iv.xmax) < 0.100    # 标点起点在词尾 ±100ms 内
_body_past = p["end_s"] > iv.xmax + 0.060      # 标点主体(>60ms)在词尾之后
if _near and _body_past:
    has_punct_in_gap = True  # 阻断延伸, 标点留给 _inject_punctuation 处理
```

**参数传递链**: `_refine_boundaries_by_energy` 新增 `_punct_boundary_hits` 参数 → 由 `process_one` 传入 → 触发时记录词、标点、偏移量到 `report["punct_boundary_guard"]`。

### 补充修改 (同日): 英文 MFA 词典/G2P 路径修复

**触发样本**: 纱依100 全部 9 个英文词 `phones: []` → 英文音素全部三等分

**Bug H1 — `pretrained_models/` 在 repo 外**:

DEFAULT_CFG 中 `g2p_model: pretrained_models/g2p/english_us_arpa.zip` 相对 PROJECT_ROOT 解析 → `/mnt/project/MFA_Pause/repo/pretrained_models/...` 不存在。实际路径在 `/mnt/project/MFA_Pause/pretrained_models/...`（PROJECT_ROOT.parent）。

修复：`run_pipeline.py` 路径解析增加 fallback：
```python
if not resolved.exists() and "pretrained_models" in val:
    _parent_resolved = PROJECT_ROOT.parent / val
    if _parent_resolved.exists():
        resolved = _parent_resolved
```

**Bug H2 — G2P 失败后 `en_combined.dict` 未写入**:

`build_en_dict` L336 先定义 `combined = temp_dir / "en_combined.dict"`，但 G2P 失败时 L384 直接 `return combined` — 文件从不存在 → MFA 拿到的 dict 路径指向不存在的文件 → 全部 0 个 TextGrid → `collect_en_phones` 全部 fallback 空 phones → `_apply_en_phones` 三等分。

修复：G2P 失败或无输出时，写入纯 base dict（CMUdict）让 MFA 至少能对齐词典内已有的词：
```python
if not g2p_output.exists():
    with open(combined, 'w', encoding='utf-8') as outf:
        if base_dict.exists():
            outf.write(_read_dict(base_dict))
    return combined
```

**验证结果**: man 音素从三等分变为真实英文 MFA 对齐 `M:17% AE1:72% N:12%`。

### 补充修改 (同日): `_NVV_PATTERN` 重复包裹 `<>`

**触发样本**: `直播回放_三周年纪念活动进行中_2025年12月07日19点场_7bd7ea1c_clip0007_clip0013`（pinyin_phones tier 中 `<CONFIRMATION-EN>` 变成 `<<CONFIRMATION-EN>>`）

**Bug I — `_NVV_PATTERN` 匹配已包裹的 NVV token**:

`_NVV_PATTERN` 的 lookbehind `(?<![A-Za-z-])` 和 lookahead `(?![A-Za-z-])` 不排斥 `<>`。`_finalise_textgrid` L142 用 `.sub()` 包裹未包裹的 NVV 时，已包裹的 `<CONFIRMATION-EN>` 被再次匹配 → `<<CONFIRMATION-EN>>`。

words/hanzi tier 不受影响（只在 `_finalise_textgrid` 中包裹一次），但 pinyin_phones tier 在 `build_pinyin_phones_tier` 和后续 `_sync_derived_tiers` 之间被多次调用 `.sub()`，造成累积。

修复：lookbehind 和 lookahead 均加入 `<>` 排斥：
```python
# 旧: (?<![A-Za-z-]) ... (?![A-Za-z-])
# 新: (?<![A-Za-z<>-]) ... (?![A-Za-z<>-])
```

已包裹的 token 不再被匹配，未包裹的仍正常包裹。`build_pinyin_phones_tier` 的 `strip('<>')` 负责修复已有双重包裹数据。


## Case 26: pinyin_phones 声母独占总时长 → 韵母消失 (ch/zh/sh → 缺 final)

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `build_pinyin_phones_tier`
**触发样本**: shayi_huali/纱依/output/ ~15.5% 的 zh/ch/sh 声母词受影响

### 现象

words tier 中拼音词为 `chang4`，但 pinyin_phones tier 只有声母 `ch` 占据整词时长，
韵母 `ang4` 完全消失：

```
words tier:       chang4  [8.850s - 9.210s]  dur=360ms
pinyin_phones:    ch      [8.850s - 9.210s]  dur=360ms  ← ang4 丢失!
```

同样问题影响所有 zh/ch/sh 声母词：`zhi3`→只有 `zh`、`shi4`→只有 `sh`、`zhong4`→只有 `zh`。

还存在另一种表现 FULL_WORD_AS_PHONE：当 MFA 完全没产出任何音素时，
整词作为单个 phone 写入（如 `chang4 [全区间]`），也失去了声母/韵母拆分。

### 范围

对 shayi_huali/纱依/output/ 2894 个文件的采样扫描（每 3 取 1，共 965 文件）：

| 类型 | 数量 | 占比 |
|------|------|------|
| MISSING_FINAL（声母独占，韵母消失） | 601 | 8.7% |
| FULL_WORD_AS_PHONE（整词作为单音素） | 454 | 6.5% |
| 正常拆分 | 5861 | 84.5% |

按声母: `sh` 538 > `zh` 380 > `ch` 156。

### 根因链

两种子类型共享同一个根因 — MFA 对该音节的音素产出不足，而代码在音素不足时
只取字典第一个条目（声母）覆盖全区间：

**子类型 A — MISSING_FINAL** (原 line 562-564):

1. MFA 对齐 `chang4` 时只产出 **1 个** IPA 音素（声母 IPA），未将韵母区分为独立音素
2. 代码查字典 `fullpinyin_enword.dict` → `dict_phones = ['ch', 'ang4']`（2 个）
3. `word_phones` 只有 1 个 MFA 音素 → 条件 `len(word_phones) <= 1` 为 True
4. 旧代码: `dict_phones[0]` (`'ch'`) 分配给整词区间 → `'ang4'` 彻底丢失

**子类型 B — FULL_WORD_AS_PHONE** (原 line 494-496):

1. MFA 对该词完全没产出音素（`word_phones` 为空），或被泄漏音素过滤清空
2. 旧代码: 直接用词文本 `'chang4'` 作为单音素兜底，未查字典拆分
3. 损失了声母/韵母的时间分辨率（对于 TTS 训练，单音素 `chang4` 不如 `ch` + `ang4`）

### 修改点

**BA. `build_pinyin_phones_tier` — 字典查前移 + 空音素时按比例拆分 (~line 494-527)**

字典查询从原 line 498 移至 `word_phones` 空检查之前，使空音素场景也能利用字典信息。
当 `dict_phones >= 2` 且 `word_phones` 为空时，按 35:65 比例拆分词区间为声母+韵母，
每段地板 30ms。NVV/English/punct 不进入此分支（保留旧兜底行为）。

**BB. `build_pinyin_phones_tier` — 单音素场景按比例拆分 (~line 584-616)**

原代码 `len(dict_phones) == 1 or len(word_phones) <= 1` 将"零声母"和"音素不足"
混为一谈。修复后拆分为三个分支：
- `len(dict_phones) == 1`: 零声母 → 单音素覆盖全区间（保留旧行为）
- `len(word_phones) >= 2`: MFA 足量音素 → 用 MFA 边界精确拆分（保留旧行为）
- 否则（`dict_phones >= 2` 但 `word_phones <= 1`）: MFA 音素不足 → 按比例拆分

### 修改前

```python
# (原 line 494) 空音素 → 整词兜底，不查字典
if not word_phones:
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, word))
    continue

# ... 字典查询在下方的 line 498 ...

# (原 line 562-564) 零声母和音素不足混为一谈
if len(dict_phones) == 1 or len(word_phones) <= 1:
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dict_phones[0]))
```

### 修改后

```python
# 字典查询已移至 word_phones 空检查之前

# 空音素 → 字典有 2+ 条目时按比例拆分
if not word_phones:
    if dict_phones and len(dict_phones) >= 2 and not punct/NVV/English:
        word_dur = w_iv.xmax - w_iv.xmin
        _init_frac = 0.35; _min_seg = 0.030
        split = w_iv.xmin + max(_min_seg, word_dur * _init_frac)
        split = min(split, w_iv.xmax - _min_seg)
        new_intervals.append(Interval(w_iv.xmin, split, dict_phones[0]))
        new_intervals.append(Interval(split, w_iv.xmax, dict_phones[1]))
    else:
        new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, word))
    continue

# 三分支: 零声母 | MFA足量 | MFA不足→比例拆分
if len(dict_phones) == 1:
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dict_phones[0]))
elif len(word_phones) >= 2:
    # MFA 边界精确拆分（不变）
    ... 
else:
    # dict_phones >= 2 但 word_phones <= 1 → 比例拆分（同上）
    ...
```

### 比例选择依据

- 35:65 是汉语声母:韵母的典型时长比（塞擦音 ch/zh ~30-40%，擦音 sh ~25-35%）
- 30ms 地板防止极短音节出现零时长段
- 音节 < 60ms 时退化为 50:50 均分

### 关联样本

- `直播回放_3D初披露--_2024年2月29日20点场_97187255_clip0012_clip0000.TextGrid`:
  `chang4` [8.850-9.210] pinyin_phones 只有 `ch`，缺 `ang4`
- `直播回放_zzZ_2026年06月07日20点场_1341f0ea_clip0012_clip0013.TextGrid`:
  4 个 `chang4` 中 2 个异常（1 个 MISSING_FINAL + 1 个 FULL_WORD_AS_PHONE）

### 补充修改 (2026-08-04): 按声母类型细化拆分比例

原修复使用统一 35% 拆分声母:韵母。对塞音 (b/p/d/t/g/k, ~15-25%) 和鼻音/边音
(m/n/l, ~15-25%) 偏高。新增模块级 `_INIT_FRAC` 字典 (~line 434)，按声母类型返回
对应比例：塞音 0.20、鼻音/边音 0.22、擦音 0.28、塞擦音 0.35 (默认)。

### 关联修改点

| 修改 | 位置 | 作用 |
|------|------|------|
| BA  | `build_pinyin_phones_tier` ~line 510-527 | FULL_WORD_AS_PHONE: 字典前移 + 比例拆分 |
| BB  | `build_pinyin_phones_tier` ~line 600-616 | MISSING_FINAL: 三分支替代 OR 条件 |
| BC  | 模块级 `_INIT_FRAC` ~line 434 | 按声母类型细化拆分比例 |
| BD  | `process_one` ~line 5350 前 (新增) | 质检过滤: `init_only_phone` 残留检测 |

---

## Case 27: words tier 区间重叠 — 相邻词时间边界交叉

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_fix_overlapping_boundaries` (新增), `process_one` (过滤)
**触发样本**: shayi_huali/纱依/output/ ~9.7% 文件存在不同程度的重叠

### 现象

words tier 中相邻 interval 的 xmax > 下一 interval 的 xmin，即时间区间重叠：

```
轻度 (< 20ms):
  ye3 [7.349-7.554] ↔ yi3 [7.537-7.657]  overlap=17ms

中度 (标点侵入词):
  ，[3.990-4.740] ↔ zuo2 [4.600-4.740]    overlap=140ms

重度 (双重标点):
  ，[4.850-9.360] ↔ ，[5.100-9.360]       overlap=4260ms
  ，[5.100-9.360] ↔ y   [5.455-5.610]     overlap=3905ms
```

### 根因链

**子类型 A1 — 轻度边界重叠 (< 30ms)**:
1. MFA 帧精度 (10ms) + `_snap_to_ctc` 将不同词 snap 到不同 CTC 锚点
2. 相邻词的 MFA 边界和 CTC 锚点来自独立的决策，未做连续性校验
3. 结果：prev.xmax (来自 MFA/CTC 决策 A) > next.xmin (来自决策 B)

**子类型 A2 — 标点重叠**:
1. `_inject_punctuation` 注入标点时使用了过宽的区间范围
2. 连续标点未做去重或裁剪，导致多个标点 interval 使用相同或高度重叠的范围
3. 标点 xmax 延伸到后续内容词内部

**子类型 A3 — 极短词挤压**（与 Case 29 同源）:
1. 极短词 (< 30ms) 挤在两个正常词之间
2. 因为太短，其 xmin < prev.xmax（物理上必然重叠）

### 修改点

**CA. `_fix_overlapping_boundaries` — 新增函数，修复轻度边界重叠** (~line 新增)

对 words tier 扫描相邻 interval 重叠：
- 两内容词重叠 < 30ms → 取中点分界 (`split_overlap`)
- 内容词与标点重叠 → 标点裁短 (`clip_punct`)
- 重叠 ≥ 30ms 或涉及静音/多重重叠 → 不修复，交给过滤

修复后调用 `_sync_derived_tiers` 同步 hanzi + pinyin_phones。

**CB. `process_one` — 过滤: `overlapping_words`** (~line 5350 前)

修复后仍存在重叠的 → 添加 `overlapping_words` 过滤原因，文件进 `filtered/`。

### 关联样本

- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0002_clip0005.TextGrid`:
  3 处重叠 (17ms/103ms/7ms)
- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0002_clip0012.TextGrid`:
  3 处重度重叠 (140ms/424ms/596ms)

---

## Case 28: 倒置 interval — xmin > xmax

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (过滤，不修复根因)
**触发样本**: shayi_huali/纱依/output/ 个别文件

### 现象

三个 tier 同时出现 xmin > xmax 的倒置 interval：

```
hanzi:          '这'  [1.0283-1.0200]  dur=-8.3ms
words:          'zhe4' [1.0283-1.0200]  dur=-8.3ms
pinyin_phones:  'zhe4' [1.0283-1.0200]  dur=-8.3ms
```

相邻词正常：`ge4 [1.0200-1.1100] dur=90ms`。

### 根因链

1. `_refine_boundaries_by_energy` 或 `_inject_punctuation` 将 `zhe4.xmin` 推到 1.0283
2. `zhe4.xmax` 被锚定为 `ge4.xmin = 1.0200`（下一词的起点）
3. 词首前推 + 词尾被后邻词锚定 → xmin > xmax 倒置
4. Case 12 的修复 AC 只覆盖 `_snap_to_ctc` 重叠路径的倒置，此处的倒置走另一条路径
5. 倒置 interval 通过 `_sync_derived_tiers` 同步到三个 tier

### 修改点

**DA. `process_one` — 过滤: `inverted_interval`** (~line 5350 前)

扫描所有 tier 的所有 interval，检测 `xmin > xmax + 0.001`。
存在倒置 → 添加 `inverted_interval` 过滤原因。

不修复根因：涉及能量调整和标点注入的深层交互，风险较高。
修复线索：在 `_refine_boundaries_by_energy` 词首前拉逻辑中添加 xmin ≤ xmax - 40ms 约束。

### 关联样本

- `直播回放_三周年纪念直播_2025年11月25日21点场_c2ad2c62_clip0010_clip0018.TextGrid`:
  zhe4/这 xmin=1.0283 > xmax=1.0200

---

## Case 29: 极短内容词 — (< 30ms) 物理不可能

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (过滤，不修复根因)
**触发样本**: shayi_huali/纱依/output/ ~91/414 文件存在极短词

### 现象

words tier 中存在 < 30ms 的内容词（非标点、非静音）：

```
ke3  [7.554-7.564]  dur=10ms   ← 物理上不可能
shi2 [10.174-10.184] dur=10ms
ting2 [4.979-5.017]  dur=38ms  ← 边界值
doing [4.480-4.510]  dur=30ms  ← 英文词
```

30ms 以下的音节在生理上不可能 — 汉语单音节最短约 60-80ms，英语最短约 80-100ms。

### 根因链

1. `_snap_to_ctc` 将相邻词边界向 CTC 锚点靠拢
2. 中间词被挤压到剩余空间 → 可能被压缩到 < 30ms
3. 极短词引发 Case 27 的子类型 A3（重叠），产生连锁问题
4. 英文词 `doing` 30ms 可能是 ASR 误检 + MFA 无法对齐

### 修改点

**EA. `process_one` — 过滤: `short_word`** (~line 5350 前)

扫描 words tier，检测内容词（非 punct/NVV/silence）时长 < 30ms。
存在极短词 → 添加 `short_word` 过滤原因。

不修复根因：涉及 `_snap_to_ctc` 边界压缩逻辑和 CTC 锚点质量，
简单合并到相邻词可能引入新问题。建议作为后续优化通过改进边界调整逻辑来根治。

### 关联样本

- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0002_clip0005.TextGrid`:
  ke3 10ms, shi2 10ms
- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0005_clip0021.TextGrid`:
  doing 30ms

## Case 30: 参考文本模糊子串匹配 — 纯英文/混杂英文 CTC 锚点标定风险分析

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/adjust_ctc_boundaries.py`, `scripts/pipeline_utils.py`
**涉及函数**: `_word_matches`, `_alpha_text_matches`, `_align_word_sequences`, `_snap_to_ctc`, `_normalize_word_spellings`, `_build_hanzi_tier`, `align_sequences`
**触发条件**: 参考文本含英文单词（纯英文或中英混杂），且英文词长度 ≤3 字符或为其他词的子串

### 现象

Post-MFA 参考文本匹配链路中，存在两套不同的对齐策略用于英文 token：

| 阶段 | 函数 | 匹配策略 | 使用位置 |
|------|------|---------|---------|
| CTC 锚点边界对齐 | `align_sequences` (pipeline_utils.py:963) | **精确匹配** (`a==b`) | `_snap_to_ctc` (postprocess_textgrids.py:3460) |
| 拼写规范化 + hanzi 构建 | `_word_matches` / `_alpha_text_matches` (postprocess_textgrids.py:1347/1450) | **模糊子串** (`c in r or r in c`) | `_normalize_word_spellings` (postprocess_textgrids.py:1657), `_build_hanzi_tier` (postprocess_textgrids.py:1571) |

这两套策略在英文 token 上的行为不一致：

```
# align_sequences (精确匹配):
CTC "Pop" vs MFA "Pop" → match ✓
CTC "Pop" vs MFA "K-Pop" → NO match ✗  (子串但不等)
CTC "Up"  vs MFA "Up"  → match ✓

# _word_matches (模糊子串):
CTC "Pop" vs ref "K-Pop" → match ✓  (子串包含)
CTC "Up"  vs ref "V-Up"  → match ✓  (子串包含)
CTC "a"   vs ref "cat"   → match ✓  (单字母=任意包含)
CTC "he"  vs ref "the"   → match ✓  (子串包含)
```

### 根因链

**A. 精确匹配层 (`_snap_to_ctc`) 的问题:**

1. `align_sequences`(pipeline_utils.py:981) 仅接受 `cost=0 if a[i-1]==b[j-1] else 1`
2. 英文词在 CTC 输出中常以原词出现（如 `Pop`, `Up`），但若 MFA 将其视为 OOV 输出 `<unk>`，精确匹配失败
3. 若 CTC token 计数 ≠ MFA token 计数 → 进入 NW 对齐（postprocess_textgrids.py:3456-3460）→ 精确匹配找不到对应 → 英文 token 无 CTC 锚点 → 回退到 MFA 边界（postprocess_textgrids.py:3500）
4. **英文 token 无条件信任 CTC**（postprocess_textgrids.py:3534-3535: `use_mfa=False`），但若 CTC token 已因精确匹配失败被标记为 `None`，则 MFA 边界被保留——产生矛盾

**B. 模糊子串层 (`_word_matches` / `_alpha_text_matches`) 的问题:**

1. **核心**: 子串包含 `c in r or r in c`（postprocess_textgrids.py:1370）过于宽松
   - `"Pop"` in `"K-Pop"`? `"Pop"` 不在 `"k-pop"` 中！——不匹配 ✗
   - `"Pop"` in `"Pop"`? 自身包含 ✓
   - 实际上 `"Pop" in "K-Pop"` → False (因为 `-Pop` 不包含不含连字符的 Pop)... 

   等等，`"pop" in "k-pop"` → True! 因为 lowercase 后 `"k-pop"` 包含 `"pop"`。所以 `-` 不干扰子串匹配。
   
   真正的风险在于短词:
   - `"a"` 匹配任何含 `a` 的英文词（`"cat"`, `"day"`, `"and"` 等）
   - `"he"` 匹配 `"the"`, `"there"`, `"where"` 等
   - `"or"` 匹配 `"for"`, `"more"`, `"word"` 等
   - `"in"` 匹配 `"win"`, `"within"`, `"inside"` 等

2. **单字母 CTC token 显式允许**（postprocess_textgrids.py:1374-1375）:
   ```python
   if len(c) == 1 and c.isalpha():
       return c in r     # 单字母匹配任意含该字母的参考词
   ```

3. **NW gap-first 回溯**（postprocess_textgrids.py:1426-1431）: 当多个模糊匹配代价相同时（均为 0），优先丢弃 CTC token（gap），保留参考词。这在英文碎片化场景中是正确的（CTC token 更多），但在短词歧义场景中可能丢弃正确的 CTC token。

4. **贪婪 alpha 消费**（postprocess_textgrids.py:1576-1582）: `continue_match` 使用相同子串逻辑，可能过度消费参考 alpha 单元。

**C. CTC 锚点标定 (`adjust_ctc_boundaries`) 层面：无参考文本参与**

`adjust_ctc_boundaries.py` 的 `adjust_boundaries()` (adjust_ctc_boundaries.py:121) 纯能量驱动:
- 英文词通过 `_is_nvv` 过滤（adjust_ctc_boundaries.py:127）— 仅跳过 NVV token
- 普通英文词仍会被 `_search_energy_rise` / `_search_energy_fall` 修正边界
- **不涉及参考文本，不存在文本混乱风险** ✅

### 针对当前数据集 (hecheng_english_mfa) 的实际风险评估

数据集特征（`/mnt/local_E/Voxcpm/output/generated_scripts.jsonl`）:
- 54000 条中英混杂文本
- 英文词: 仅 1021 条含英文，绝大多数 1 个英文词/条
- 英文词长: 仅 2-3 字符（Pop, Up, PV, MV, AI, SOS, BGM, GPT, ...）
- 无 ≥5 字母的英文词，无 3+ 英文词的条目
- 英文词以品牌名/缩写为主，非自然语言句子

**针对此数据集的结论**:
1. 英文词均为短缩写/品牌名 → 子串匹配风险较低（词形独特，不太可能嵌套）
2. 但 `"Up"` 匹配 `"V-Up"` 中的 `"up"` 子串 ✓ 正确，但也匹配 `"Upload"`, `"Popup"` 等 —— 当前数据集不含这些词
3. 单字母 token 不会出现（CTC 模型不会把 "K-Pop" 拆成单字母）
4. **主要风险**: CTC 中文模型对英文词的边界标定本身不准（duration 偏长/偏短），而非文本匹配问题

**普遍性纯英文文本的风险**（若后续扩展数据集）:
1. 短词子串歧义: high risk
2. NW gap-first 可能丢弃正确 CTC token: medium risk
3. `_snap_to_ctc` 精确匹配 + 无条件 CTC 的矛盾: needs attention
4. `adjust_ctc_boundaries` 能量修正无风险: safe ✅

### 修改点

暂无代码修改。此 Case 为风险分析记录，供后续纯英文文本扩展时参考。

**潜在改进方向**:
- `_word_matches`: 对纯英文参考文本，可将 `c in r or r in c` 改为优先精确匹配，仅在不匹配时 fallback 到子串
- `_snap_to_ctc`: 当 `ctc_aligned[idx] is None` 且 `is_english_token(mfa_iv.text)` 时，应警告而非静默使用 MFA 边界
- 可增加 QC 规则: 检测英文词的 CTC 时长是否在合理范围 (80ms-2000ms)

### 验证方法

```python
# 测试 1: 检查英文词的 CTC-MFA 边界偏差
for tok in ctc_tokens:
    if is_english_token(tok["word"]):
        dur = tok["end_s"] - tok["start_s"]
        assert 0.08 <= dur <= 2.0, \
            f"English word {tok['word']} CTC duration {dur:.3f}s out of range"

# 测试 2: 检查 _word_matches 子串假阳性
# 构造 (ctc_token, ref_unit) 对照表，检查匹配结果
test_cases = [
    ("he", "the", True, "substring — 可接受的假阳性"),
    ("a", "cat", True, "单字母 — 当 CTC 碎片化时正确，否则假阳性"),
    ("or", "for", True, "substring — 假阳性"),
    ("in", "win", True, "substring — 假阳性"),
]
for ctc, ref, expected, note in test_cases:
    actual = _word_matches(ctc, ref)
    if actual != expected:
        print(f"MISMATCH: _word_matches({ctc!r}, {ref!r}) = {actual} (expected {expected}) — {note}")
```

### 关联样本

- 数据集: `/mnt/Raw/新版合成英文数据` (54000 条, 1021 条含英文)
- JSONL: `/mnt/local_E/Voxcpm/output/generated_scripts.jsonl`

### 实测结果 (2026-08-04, 10 条样本 CTC prealign)

**测试命令**:
```bash
python3 scripts/prepare_english_tts.py \
  --jsonl /mnt/local_E/Voxcpm/output/generated_scripts.jsonl \
  --audio-root /tmp/test_en_ctc/audio_data \
  --output-dir /tmp/test_en_ctc/ctc_pretg --limit 10

python ctc_prealign.py \
  --data-dir /tmp/test_en_ctc/audio_data \
  --pinyin-dir /tmp/test_en_ctc/ctc_pretg \
  --output-dir /tmp/test_en_ctc/ctc_pretg \
  --model-path .../Multilingual-NVASR --device cuda:0 --limit 10
```

**10 条样本: 全部通过** — 无词序错位、无 token 重叠、无时间戳倒序。

**发现的真实问题（非参考文本混淆，而是 CTC 分词器行为）**:

| 问题 | 样本 | 详情 |
|------|------|------|
| 英文词碎片化 | 000021 | `"SOS"` → CTC 输出 `"SOS"` (180ms) + `"OS"` (420ms)，normalize_en 未合并残留 `"OS"` |
| 英文词碎片化 | 000135 | `"fan"` → CTC 输出 `"f"` (60ms) + `"an"` (60ms)，极短碎片 |
| 英文词碎片化 | 000248 | `"show"` → CTC 输出 `"s"` (60ms) + `"how"` (360ms) |
| 英文词碎片化 | 000352 | `"fan"` → CTC 输出 `"f"` (60ms) + `"an"` (300ms) |
| normalize_en 过度合并 | 000044 | `"In"+"s"+"ta"+"gram"` → `"Instagramsta"` (多出 `"sta"`) |
| normalize_en 误合并 | 000021 | `"li"+"ve"` → `"leave"` (应为 `"live"`) |
| 拼音误判为 NVV | 000021 | `"zan2"` → `"BREATHING"` (240ms) |
| 拼音误判为 NVV | 000248 | `"jin1"` → `"BREATHING"` (180ms) |

**结论**: 对于此数据集（中英混杂，英文为 2-3 字母品牌名），CTC 锚点标定的参考文本匹配**没有导致错位**。实际风险在于：
1. **CTC 分词器碎片化** — 中文 CTC 模型将英文词拆成碎片，normalize_en 无法全部修复
2. **拼音→NVV 误判** — 短拼音音节被 NVASR 误识别为 NVV token
3. **纯英文场景的子串匹配风险**在本数据集中未触发（英文词太少太短），但仍需关注

---

## Case 31: 英文 CTC 锚点三重修复 — 碎片化 / 合并错误 / NVV 误判

**日期**: 2026-08-04
**涉及文件**: `scripts/normalize_english_tokens.py`, `scripts/ctc_prealign.py`
**涉及函数**: `_token_matches_ref`, `normalize_stem`, `make_patched_inference`
**触发样本**: 实测 10 条样本中发现 3 类共性问题

### 问题全景

```
                    ┌──────────────────────────┐
                    │ NVASR ASR (SenseVoice)   │
                    │ tokenizer: 中文词表为主    │
                    │ OOV English → 碎片化       │
                    └──────────┬───────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
      ┌──────────┐     ┌────────────┐     ┌──────────────┐
      │ 碎片化    │     │ ASR 幻觉    │     │ NVV 误判     │
      │ fan→f+an │     │ Instagram  │     │ zan2→BREATH  │
      │ SOS→SOS+ │     │ →Instagra │     │ jin1→BREATH  │
      │    OS    │     │   msta     │     │              │
      └────┬─────┘     └─────┬──────┘     └──────┬───────┘
           │                 │                   │
           ▼                 ▼                   ▼
    normalize_en       normalize_en         CTC logits
    合并碎片          用错误参考合并        blank bias=4.0
    f+an→fan          In+s+ta+gram        → NVV token
    SOS+OS→???        →Instagramsta
```

### 根因分析 (逐问题)

---

### 问题 1: 英文词碎片化 — CTC tokenizer OOV 拆字

**现象**:

| 样本 | 参考英文词 | CTC 碎片 | normalize_en 结果 | 状态 |
|------|-----------|---------|-------------------|------|
| 000021 | SOS | `SOS`(180ms) + `OS`(420ms) | 残留 `OS` | ❌ 未修复 |
| 000135 | fan | `f`(60ms) + `an`(60ms) | 残留 `f`+`an` | ❌ 未修复 |
| 000248 | show | `s`(60ms) + `how`(360ms) | 残留 `s`+`how` | ❌ 未修复 |
| 000352 | fan | `f`(60ms) + `an`(300ms) | 残留 `f`+`an` | ❌ 未修复 |
| 000021 | live | `li`(180ms) + `ve`(→leave) | `leave` | ❌ 错误合并 |
| 000125 | live | `li` + `ve` | `live` | ✅ 正确 |
| 000223 | deadline | `de` + `ad` + `line` | `deadline` | ✅ 正确 |
| 000223 | live | `li` + `ve` | `live` | ✅ 正确 |

**根因链**:

1. SenseVoice tokenizer 词表以中文为主，OOV 英文词无对应 token
2. CTC 解码时将英文词拆成 BPE 级子词碎片（`fan` → `f` + `an`、`show` → `s` + `how`）
3. `normalize_english_tokens.py` 通过 NW 对齐 + 参考文本匹配合并碎片
4. 但合并依赖两个条件同时满足：
   - **条件 A**: `_token_matches_ref` 返回 True（子串包含或拼音→英文）
   - **条件 B**: `all_fragments` 检查通过（每个碎片至少与目标词共享一个字母）
5. 条件 A 的拼音→英文匹配（normalize_english_tokens.py:126）无条件 `return True`，过于宽松
6. 当碎片是英文子串（如 `f` 在 `fan` 中）时条件 A 通过，但 `an` 也是独立英文词（`is_english_token("an")=True`），`an in "fan"` → True
7. **关键缺陷**: `all_fragments` 检查（normalize_english_tokens.py:264-265）将已经是完整英文词的 fragment 当作"substring of target"放行:
   ```python
   elif is_english_token(t) and t.lower() in en_lower:
       pass  # substring of target (e.g. "play" in "cosplay")
   ```
   但 `"OS"` 是英文词且 `"os" in "sos"` → True → 被当作 target 的子串放行
   然而 `"SOS"` 才是参考词，`"OS"` 是残留碎片——这个逻辑仅检查"fragment 是否是 target 的子串"，不检查"reference word 是否完整"

8. 对于 `fan` → `f` + `an`：`"f" in "fan"` ✓, `"an" in "fan"` ✓ → 理论上能合并为 `fan`，但 normalize_en 没有触发。排查：`extract_word_chars("今天天气超好，心情也超好，作为一个fan老用户...")` → `["今","天","天","气","超","好"，"心","情","也","超","好"，"作","为","一","个","fan","老","用","户",...]` → `fan` 被识别为英文参考词 ✓。但 `_token_matches_ref("an", "fan")` 返回 True（`"an" in "fan"`），`all_fragments` 检查 `"f" in "fan"` ✓，`is_english_token("an") and "an" in "fan"` ✓ → 应该触发合并。

   **实际未触发的原因**: NW 对齐时，`fan` 参考词与 `.lab` tokens 的 DP 匹配可能将 `f` 和 `an` 分别对齐到了不同的参考词位置。`.lab` 中 `fan` 位置前后是 `ge4 f an lao3`，而参考词序列是 `[..., "fan", ...]`。`f` 可能和前面的 pinyin 对齐了（pinyin→English 无条件 `return True`），导致 `f` 没有被分配到 `fan` 的 ref 索引下。

**设计修复 — Fix 1: `normalize_english_tokens.py` 增加碎片回收 Pass**:

在 `normalize_stem` 末尾增加 **Pass 2: 碎片回收** — 对 normalize_en 后仍然残留的英文碎片进行二次回收:

```python
# Pass 2: Fragment reclamation (NEW)
# After Pass 1 merge, check for orphan fragments: single-letter ASCII
# tokens or short English tokens (< 3 chars) adjacent to merged English
# words.  Absorb them into the nearest English token.
def _reclaim_fragments(lab_tokens, ctc_tokens):
    """Merge orphan English fragments into adjacent English tokens."""
    # Identify English tokens and their fragments
    # Rule: if a fragment is a substring of an adjacent English word,
    # merge it into that word's time range.
    ...
```

**实现要点**:
1. 扫描 `.lab` tokens，找到所有 `is_english_token(t)` 的 token
2. 对于每个长度 ≤2 的英文 token，检查其前后邻接是否有长度 ≥3 的英文 token
3. 如果短 token 是长 token 的子串 → 合并
4. 如果两个相邻的短 token 可以拼成完整英文词 → 合并
5. 同步更新 `_tokens.jsonl` 的时间戳（start=min, end=max）

**涉及文件**: `scripts/normalize_english_tokens.py`
**涉及函数**: `normalize_stem` (新增 `_reclaim_fragments` 调用)

---

### 问题 2: normalize_en 合并错误 — 参考文本污染 + 过度宽松匹配

**现象**:

| 样本 | ASR 原始输出 (_text_raw.txt) | 参考词 | normalize_en 结果 |
|------|---------------------------|--------|-------------------|
| 000044 | `搭配 Instagramsta效果满分` | `Instagramsta` (错误!) | `In`+`s`+`ta`+`gram` → `Instagramsta` |
| 000021 | `leave走起` | `leave` (应为 `live`) | `li`+`ve` → `leave` |

**000044 "Instagramsta" 根因链**:

1. NVASR ASR 模型听 "Instagram" → 输出 "Instagramsta"（幻觉，多出 "sta"）
2. `_text_cn.txt` 被写为 ASR 输出文本（含 "Instagramsta"）
3. CTC forced alignment 将 "Instagramsta" 拆成 BPE 碎片: `In` + `s` + `ta` + `gram`
4. `normalize_stem` 读 `_text_cn.txt` → `extract_word_chars` → 参考英文词 = `Instagramsta`
5. NW 对齐: 碎片 ↔ `Instagramsta` → 合并 → `.lab` 写入 `Instagramsta`
6. **最终结果**: 参考文本被 ASR 幻觉污染，经 normalize_en 固化到 `.lab` 和 tokens

**根本原因**: `_text_cn.txt` 的内容来自 ASR 输出而非原始 JSONL 参考文本。`prepare_english_tts.py` 写入的原始 `_text_cn.txt` 是正确的（来自 JSONL），但 `ctc_prealign.py` 的 `make_patched_inference` 将 ASR 文本写入了 `_text_cn.txt`，覆盖了原始文本。

**000021 "leave" vs "live" 根因链**:

1. ASR 输出 "leave" (误听)
2. CTC 碎片: `li` + `ve`
3. 参考词: `leave` → normalize_en 合并 = `leave`
4. 实际应该是 `live`

这两个案例的共同根因: **参考文本来自 ASR 输出而非原始标注**。

**设计修复 — Fix 2: 参考文本源头保护 + 匹配约束**:

**2a. `ctc_prealign.py` 保护原始 `_text_cn.txt`**:

`make_patched_inference` (ctc_prealign.py:279-281) 在有关联文本时使用 `ref_texts[stem]`:
```python
if stem in ref_texts:
    align_text = ref_texts[stem].strip()
```

但 `ref_texts` 来自 `_text_cn.txt` 文件，而这些文件在 pipeline 启动时可能已被之前的运行污染。

**修改**: 在 `make_patched_inference` 中，当 ref_texts 可用时，ASR 解码结果应**仅用于 NVV/标点检测**，英文词应**保留 ref_texts 中的原始英文词**:

```python
# NEW: Cross-check ASR English tokens against reference text
# If ref_text has an English word at the same position, use ref version
ref_eng_words = re.findall(r'[a-zA-Z]{2,}', ref_text)
asr_eng_words = re.findall(r'[a-zA-Z]{2,}', asr_text)
if set(asr_eng_words) != set(ref_eng_words):
    # ASR hallucinated English → revert to reference
    ...
```

**2b. `normalize_english_tokens.py` 拼音→英文匹配增加元音约束**:

`_token_matches_ref` (normalize_english_tokens.py:126-127):
```python
# Before (unconditional):
if len(t) >= 2 and t[-1].isdigit() and t[:-1].isalpha():
    return True

# After (with vowel guard, same as Case 10 fix):
if len(t) >= 2 and t[-1].isdigit() and t[:-1].isalpha():
    vowel_count = sum(1 for ch in r if ch in 'aeiou')
    return vowel_count >= 2
```

这防止短英文词（`OH`, `OP`, `Up`, `in` 等 ≤1 元音）被 pinyin 音节错误匹配。

**2c. `normalize_english_tokens.py` 增加合并结果验证**:

在 `normalize_stem` 末尾增加验证:
```python
# NEW: Verify merged result matches reference
for en_word, indices in changes:
    current = [lab_tokens[i] for i in indices]
    merged = "".join(current)
    # If merged letters don't form the reference word, warn
    if sorted(merged.lower()) != sorted(en_word.lower()):
        print(f"  [WARN] {stem}: merged '{merged}' vs ref '{en_word}' — possible hallucination")
```

**涉及文件**: `scripts/normalize_english_tokens.py` (2b, 2c), `scripts/ctc_prealign.py` (2a)

---

### 问题 3: 拼音→NVV 误判 — blank-frame bias 过度激进

**现象**:

| 样本 | CTC 片段 | NVV 分类 | 实际应为 | 时长 |
|------|---------|---------|---------|------|
| 000021 | `zan2` | `BREATHING` | 拼音 `zan2` (咱) | 240ms |
| 000248 | `jin1` | `BREATHING` | 拼音 `jin1` (今) | 180ms |

**根因链**:

1. SenseVoice 模型有 30 类 NVV token (Breathing, Laughter, Crying, ...) 位于 token ID 25025-25054
2. `make_patched_inference` (ctc_prealign.py:218-221) 对 CTC blank 帧施加 NVV bias:
   ```python
   top_pred = x.argmax(dim=-1)
   is_blank = (top_pred == BLANK_ID)
   x[is_blank, NVV_START:NVV_END + 1] += bias_value  # default 4.0
   ```
3. NVV_BIAS_DEFAULT = 4.0 → logit 加 4.0 → softmax 后概率 ≈ 0.98
4. 当短拼音音节（如 `zan2` 240ms）前后有 silence/blank 帧时：
   - 拼音音节本身被正确解码
   - **但**周围的 blank 帧被 biased → NVV token 概率飙升
   - CTC greedy decoding 选择了 NVV token 而非 blank → 在音节边界插入 NVV
5. 更关键的是: NVV bias 作用于 **所有 blank 帧**，不区分"真正的 silence"（数百ms静音）和"音节间的短暂 blank"（60-120ms 间隔）
6. `zan2` [9.27-9.51] 后的空白帧在 `pause_threshold` (8帧≈480ms) 内被检测为 NVV

**为什么 `zan2` → BREATHING 而不是其他 NVV**:
- CTC blank 帧被 bias 后，NVVS_START..NVV_END 范围内 BREATHING token 的原始 logit 最高（预训练权重偏差）
- 即使原始 logit 差异很小，+4.0 后 softmax 将其推到接近 1.0

**设计修复 — Fix 3: NVV 后处理还原 + bias 策略调整**:

**3a. `ctc_prealign.py` 新增 NVV→拼音还原 Pass**:

在 `_normalize_english` 之后增加 `_reclaim_nvv_pinyin`:

```python
def _reclaim_nvv_pinyin(ctc_dir: Path, pinyin_dir: Path) -> int:
    """Revert NVV tokens that are actually misclassified pinyin syllables.
    
    A NVV token is likely pinyin if ALL of:
      - Word matches pinyin syllable pattern: [a-z]+[1-5]  (has tone digit)
      - Duration < 400ms (true NVV like BREATHING > 500ms)
      - Adjacent tokens are pinyin syllables (not English / other NVV)
      - Position in sentence is mid-sentence (not at a natural pause)
      - The .lab file has the original pinyin at this position
    """
    reverted = 0
    for tokens_path in sorted(ctc_dir.glob("*_tokens.jsonl")):
        stem = tokens_path.stem.replace("_tokens", "")
        tokens = [...]  # load
        lab_path = pinyin_dir / f"{stem}.lab"
        lab_tokens = lab_path.read_text().strip().split() if lab_path.exists() else []
        
        for i, tok in enumerate(tokens):
            if not is_nvv_token(tok["word"]):
                continue
            dur = tok["end_s"] - tok["start_s"]
            
            # Check 1: Duration fits pinyin syllable range
            if dur > 0.400:
                continue
            
            # Check 2: Look up original pinyin from .lab
            if i < len(lab_tokens):
                lab_tok = lab_tokens[i]
                if is_pinyin_syllable(lab_tok):
                    # Revert NVV → original pinyin
                    tok["word"] = lab_tok
                    reverted += 1
                    print(f"  [nvv_reclaim] {stem}: {tok['word']} ← NVV")
        
        # Write back...
    return reverted
```

**3b. (可选) `make_patched_inference` 中 NVV bias 条件化**:

当前 bias 对所有 blank 帧生效。改为仅对**连续 blank 帧**中段施加 bias:

```python
# Before: all blank frames get bias
x[is_blank, NVV_START:NVV_END + 1] += bias_value

# After: only sustained blank runs (≥3 consecutive blank frames ≈ 180ms)
blank_runs = []
j = 0
while j < len(is_blank):
    if is_blank[j]:
        s = j
        while j < len(is_blank) and is_blank[j]:
            j += 1
        if j - s >= 3:  # sustained silence
            blank_runs.append((s, j))
    else:
        j += 1

biased_mask = torch.zeros_like(is_blank)
for s, e in blank_runs:
    # Bias only middle frames (avoid edge frames near speech)
    mid_start = s + 1
    mid_end = e - 1
    if mid_end > mid_start:
        biased_mask[mid_start:mid_end] = True

x[biased_mask, NVV_START:NVV_END + 1] += bias_value
```

这确保 NVV bias 仅在持续静音段（≥180ms）生效，不在音节边界触发。

**涉及文件**: `scripts/ctc_prealign.py` (3a, 3b), `scripts/pipeline_utils.py` (无修改)

---

### 修改点汇总

| ID | 文件 | 函数 | 修改 | 优先级 |
|----|------|------|------|--------|
| **Fix-1** | `normalize_english_tokens.py` | `normalize_stem` (新增) | **Pass 2 碎片回收**: 合并残留单字母/短英文碎片到相邻英文词 | **P0** |
| **Fix-2a** | `ctc_prealign.py` | `make_patched_inference` | **参考文本保护**: ASR 英文词与 ref_text 不一致时回退到 ref | **P1** |
| **Fix-2b** | `normalize_english_tokens.py` | `_token_matches_ref` | **元音约束**: pinyin→English 仅当 ref ≥2 元音 (对齐 Case 10) | **P0** |
| **Fix-2c** | `normalize_english_tokens.py` | `normalize_stem` (新增) | **合并验证**: 合并后字母组成与参考词不一致时告警 | **P1** |
| **Fix-3a** | `ctc_prealign.py` | `_reclaim_nvv_pinyin` (新增) | **NVV 还原**: pinyin 模式 + <400ms → 还原为拼音 | **P0** |
| **Fix-3b** | `ctc_prealign.py` | `make_patched_inference` | **NVV bias 条件化**: 仅 ≥3 帧连续 blank 中段 bias | **P2** (可选优化) |

### 预期修复效果

| 问题样本 | 修复前 | Fix-1 | Fix-2a/b | Fix-3a | 修复后 |
|---------|--------|-------|----------|--------|--------|
| 000021 SOS+OS | `SOS`(180ms) + `OS`(420ms) | ✅→SOS(600ms) | — | — | `SOS` 合并 |
| 000135 f+an | `f`(60ms) + `an`(60ms) | ✅→fan(120ms) | — | — | `fan` 合并 |
| 000248 s+how | `s`(60ms) + `how`(360ms) | ✅→show(420ms) | — | — | `show` 合并 |
| 000352 f+an | `f`(60ms) + `an`(300ms) | ✅→fan(360ms) | — | — | `fan` 合并 |
| 000044 Instagram→sta | `Instagramsta` | — | ✅→Instagram | — | `Instagram` 修正 |
| 000021 li+ve→leave | `leave` | — | ✅→live | — | `live` 修正 |
| 000021 zan2→BREATH | `BREATHING`(240ms) | — | — | ✅→zan2 | `zan2` 还原 |
| 000248 jin1→BREATH | `BREATHING`(180ms) | — | — | ✅→jin1 | `jin1` 还原 |

### 验证方法

```python
# Fix-1 验证: 无残留短英文碎片
for tokens_path in ctc_dir.glob("*_tokens.jsonl"):
    tokens = json.loads(...)
    for i, t in enumerate(tokens):
        if is_english_token(t["word"]) and len(t["word"]) <= 2:
            dur = t["end_s"] - t["start_s"]
            if dur < 0.080:
                # Check if adjacent to longer English word
                if not ((i>0 and is_english_token(tokens[i-1]["word"]) and len(tokens[i-1]["word"])>=3)
                     or (i<len(tokens)-1 and is_english_token(tokens[i+1]["word"]) and len(tokens[i+1]["word"])>=3)):
                    print(f"  ORPHAN: {t['word']} {int(dur*1000)}ms in {tokens_path.stem}")

# Fix-2 验证: 英文词与原始 JSONL 一致
for stem, orig_text in original_jsonl_texts.items():
    tokens = load_tokens(stem)
    orig_eng = set(re.findall(r'[a-zA-Z]{2,}', orig_text))
    ctc_eng = set(t["word"] for t in tokens if is_english_token(t["word"]))
    if orig_eng != ctc_eng:
        print(f"  DRIFT: {stem}: {orig_eng} -> {ctc_eng}")

# Fix-3 验证: 无 pinyin 被误判为 NVV
for tokens_path in ctc_dir.glob("*_tokens.jsonl"):
    tokens = json.loads(...)
    for t in tokens:
        if is_nvv_token(t["word"]):
            dur = t["end_s"] - t["start_s"]
            if dur < 0.400:
                print(f"  NVV_SUSPECT: {t['word']} {int(dur*1000)}ms in {tokens_path.stem}")
```

---

### ria 名字完整性专项保护 (Fix-4)

**触发条件**: "ria" 是核心 VTuber 名字，必须保证在任何管线阶段都是完整小写 `ria`，不被碎片化或大小写不一致。

**当前 ria 处理链路及缺口**:

```
文本级: replace_ria_variants() → "瑞娅/瑞亚/瑞雅/瑞啊" → "ria"  ✅
                                                ↓
CTC 分词: NVASR tokenizer 对 OOV "ria" 的 3 种拆分:
  ├─ Pattern A: "rui4" + "ya4"  (中文音译)
  ├─ Pattern B: "R" + "I" + "A"  (单字母拆分)  
  └─ Pattern C: "R" + "ia"      (混合拆分)
                                                ↓
Token 合并: 单字母合并 (ctc_prealign.py:1325-1356)
  ├─ Pattern A: 不适用 (拼音音节非单字母)
  ├─ Pattern B: "R"+"I"+"A" → "RIA" ✅ (但大写!)
  └─ Pattern C: "R"+"ia" → "R"+"ia" ❌ 未合并 (ia 是2字符)
                                                ↓
_normalize_ria: regex 替换 (ctc_prealign.py:838-839)
  ├─ Pattern A: "rui4 ya4" → "ria" ✅
  ├─ Pattern B: "RIA" → 不匹配 ❌ (全大写不匹配 regex)
  └─ Pattern C: "R ia" → 不匹配 ❌
                                                ↓
_merge_ria_tokens: tokens.jsonl 合并 (ctc_prealign.py:795-796)
  ├─ Pattern A: "ruiN+yaN" → "ria" ✅
  ├─ Pattern A': "ruiN+aN" → 不合并 ❌ (仅检查 yaN!)
  ├─ Pattern B: "R"+"I"+"A" → 已合并不适用
  └─ Pattern C: 不适用
                                                ↓
normalize_en: 参考文本匹配
  ├─ Pattern B: "RIA"→"ria" ✅ (参考文本有 "ria")
  └─ Pattern C: "R"+"ia"→"ria" ✅ (碎片匹配合并)
```

**识别出的 3 个缺口**:

| Gap | 位置 | 现象 | 严重度 |
|-----|------|------|--------|
| **Gap-A** | `_merge_ria_tokens` line 795-796 | `ruiN + aN` → .lab 修复了但 tokens.jsonl **未合并** | **P0** |
| **Gap-B** | 单字母合并 line 1347-1348 | `"R"+"I"+"A"` → `"RIA"` 大写，依赖 normalize_en 修正大小写 | **P1** |
| **Gap-C** | normalize_en `_token_matches_ref` line 126 | `"ria"` 参考词可能被 pinyin→English 无条件匹配污染，导致碎片合入错误的目标词 | **P1** |

**设计修复 — Fix 4a: `_merge_ria_tokens` 增加 `ruiN + aN` 模式**:

`ctc_prealign.py` `_merge_ria_tokens`, line 795-796:
```python
# Before:
if (re.match(r'^rui[0-5]$', w) and i + 1 < len(entries)
        and re.match(r'^ya[0-5]$', entries[i + 1]["word"])):

# After:
if (re.match(r'^rui[0-5]$', w) and i + 1 < len(entries)
        and re.match(r'^(ya|a)[0-5]$', entries[i + 1]["word"])):
```

**设计修复 — Fix 4b: 单字母合并后强制 ria 小写**:

在单字母合并逻辑 (ctc_prealign.py line 1347-1348) 之后增加规范化:

```python
# After merged_pinyin.append({"word": "".join(letters), ...})
# NEW: Force lowercase for known proper names
_KNOWN_NAMES = frozenset({"ria", "noa", "mila"})
merged_word = "".join(letters)
if merged_word.lower() in _KNOWN_NAMES:
    merged_pinyin[-1]["word"] = merged_word.lower()
```

**设计修复 — Fix 4c: normalize_en 增加 ria 硬保护**:

在 `normalize_stem` (normalize_english_tokens.py) 中增加 ria 专项检查:

```python
# normalize_english_tokens.py normalize_stem, before changes loop (~line 271)
# NEW: Hard protection for "ria" — never merge into another word,
# never leave as fragments. Ria is always standalone lowercase.

_PROTECTED_NAMES = frozenset({"ria"})

for ri, indices in sorted(ref_to_lab.items()):
    en_word = en_ref_positions[ri]
    
    # [NEW] Protected name: ensure standalone and lowercase
    if en_word.lower() in _PROTECTED_NAMES:
        # 1. Gather all fragment indices that form "ria"
        # 2. Force lowercase in .lab and tokens
        # 3. Merge timestamps: start=earliest, end=latest
        # 4. Mark as handled, skip normal merge logic
        _merge_protected_name(ri, en_word.lower(), indices, ...)
        continue
```

**设计修复 — Fix 4d (推荐): 统一 ria 后处理校验函数**:

新增 `_protect_ria` 函数, 在 ctc_prealign.py 和 normalize_english_tokens.py 两处调用:

```python
# ctc_prealign.py / normalize_english_tokens.py 通用
def _protect_ria(tokens: list[dict], lab_tokens: list[str]) -> tuple[list, list]:
    """Ensure "ria" is always a complete, lowercase, standalone token.
    
    Handles all known fragmentation patterns:
    - "R"+"I"+"A" → "ria"  (single-letter split)
    - "R"+"ia"    → "ria"  (mixed split)
    - "RIA"       → "ria"  (case normalization)
    - "ruiN"+"yaN"→ "ria"  (pinyin phonetic, also covered by _normalize_ria)
    - "ruiN"+"aN" → "ria"  (pinyin variant, Gap-A)
    
    Returns (updated_tokens, updated_lab_tokens).
    """
    RIA_FRAGMENT_SETS = [
        frozenset({"r", "i", "a"}),      # single letters
        frozenset({"r", "ia"}),           # mixed
        frozenset({"ria"}),               # case fix only
        frozenset({"rui4", "ya4"}),       # pinyin phonetic
        frozenset({"rui2", "a1"}),        # pinyin variant
    ]
    
    i = 0
    new_tokens, new_lab = [], []
    while i < len(tokens):
        # Check if tokens[i:i+N] form "ria" fragments
        matched = None
        for n in range(1, 4):  # try 1-3 consecutive tokens
            if i + n > len(tokens):
                break
            fragment_set = frozenset(
                t["word"].lower() if isinstance(t, dict) else t.lower()
                for t in (tokens[i:i+n] if isinstance(tokens[0], dict) else lab_tokens[i:i+n])
            )
            if fragment_set in RIA_FRAGMENT_SETS:
                matched = n
                break
        
        if matched:
            # Merge fragments → single "ria" token
            if isinstance(tokens[0], dict):
                s = tokens[i]["start_s"]
                e = tokens[i + matched - 1]["end_s"]
                new_tokens.append({"word": "ria", "start_s": s, "end_s": e,
                                   "start_ms": round(s*1000), "end_ms": round(e*1000),
                                   "type": "word"})
            new_lab.append("ria")
            i += matched
        else:
            if isinstance(tokens[0], dict):
                new_tokens.append(tokens[i])
            new_lab.append(lab_tokens[i] if isinstance(lab_tokens, list) else tokens[i])
            i += 1
    
    return new_tokens, new_lab
```

**调用点**:
1. `ctc_prealign.py` `make_patched_inference` → 单字母合并之后调用 `_protect_ria(words_pinyin, lab_tokens)`
2. `normalize_english_tokens.py` `normalize_stem` → 合并循环之后调用 `_protect_ria(new_ctc, new_lab)`

**修改点汇总 — ria 专项**:

| ID | 文件 | 函数 | 修改 | 优先级 |
|----|------|------|------|--------|
| **Fix-4a** | `ctc_prealign.py` | `_merge_ria_tokens` | `ya[0-5]` → `(ya\|a)[0-5]` 扩展 tokens.jsonl 合并 | **P0** |
| **Fix-4b** | `ctc_prealign.py` | 单字母合并后 (line ~1356) | 已知名称 (ria/noa/mila) 强制小写 | **P0** |
| **Fix-4c** | `normalize_english_tokens.py` | `normalize_stem` | ria 硬保护: 独立 token + 小写 | **P1** |
| **Fix-4d** | `ctc_prealign.py` | `_protect_ria` (新增) | **推荐**: 统一 ria 后处理校验，所有 2 个调用点 | **P0** |

### 关联样本

- 测试集: `/tmp/test_en_ctc/ctc_pretg/` (10 条)
- 数据集: `/mnt/Raw/新版合成英文数据` (54000 条)

### 实施记录 (2026-08-04)

**已实施的修改**:

| ID | 文件 | 行号 | 修改 | 状态 |
|----|------|------|------|------|
| **Fix-4a** | `ctc_prealign.py` | ~796 | `ya[0-5]` → `(ya\|a)[0-5]` 扩展 `_merge_ria_tokens` | ✅ |
| **Fix-4b** | `ctc_prealign.py` | ~42-45, ~1357-1364 | `_SINGLE_LETTER_LOWERCASE_NAMES` 常量 + 单字母合并后强制小写 | ✅ |
| **Fix-4d** | `ctc_prealign.py` | ~824-880, ~1480 | `_protect_ria()` 函数 + 单字母合并/NVV去重后调用 | ✅ |
| **Fix-4d** | `normalize_english_tokens.py` | — | (ria 保护由 ctc_prealign 层覆盖, normalize_en 不再额外处理) | ✅ |
| **Fix-2b** | `normalize_english_tokens.py` | ~124-129 | `_token_matches_ref` 拼音→英文增加 ≥2 元音约束 | ✅ |
| **Fix-3a** | `ctc_prealign.py` | ~989-1059 | ~~`_reclaim_nvv_pinyin()`~~ **已删除**: 时序缺陷, `--no-nvv` 从源头阻断后不需要 | 🗑 |
| **Fix-3a** | `normalize_english_tokens.py` | ~303 | NVV token 排除出 `en_ref_positions` (防 normalize_en 把拼音合并到 NVV) | ✅ |
| **Fix-3a** | `normalize_english_tokens.py` | ~323-329 | ~~NVV pre-reclaim~~ **已删除**: 同时序缺陷, 读取已被污染的 output .lab | 🗑 |
| **Fix-1** | `normalize_english_tokens.py` | ~63-130, ~389-434 | `_reclaim_fragments()` Pass 2 (始终运行 + 双向碎片合并) + 重构流程 | ✅ |
| **Fix-NEW** | `ctc_prealign.py` | ~154-168, ~228-232 | `make_patched_inference` 增加 `enable_nvv` 参数, False 时跳过 blank-frame NVV bias | ✅ |
| **Fix-NEW** | `ctc_prealign.py` | ~1089-1090 | `--no-nvv` CLI 开关 | ✅ |
| **Fix-NEW** | `ctc_prealign.py` | ~1145 | `--all-gpus` 子进程转发 `--no-nvv` | ✅ |
| **Fix-NEW** | `run_pipeline.py` | ~754-755 | `nvv_enabled: false` 配置 → `--no-nvv` 转发 | ✅ |
| **Fix-NEW** | `hecheng_english_mfa.yaml` | ~80 | `nvv_enabled: false` 配置项 | ✅ |

**实测验证 (10 条样本, `--no-nvv`)**:

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| NVV token 误判 | 2 个 (`zan2`/`jin1` → BREATHING) | **0** ✅ |
| 英文碎片残留 | "SOS"+"OS" 两个独立 token | "SOS" 合并为 600ms ✅ |
| ria 完整性 | 依赖 normalize_en 链式修正 | 单字母合并 + 小写 + `_protect_ria` ✅ |
| fragment reclaim | 无 | 2 文件各吸收 1 碎片 ✅ |
| `f`+`an` → `fan` | 未合并 | **✅ 已修复** (Pass 2 始终运行 + 双向碎片合并) |

**已知剩余问题**:

1. ~~`f`(60ms) + `an`(60ms) → 应合并为 `fan`~~ **已修复 (2026-08-04)**

2. ~~`_reclaim_nvv_pinyin` 时序敏感~~ **已清理 (2026-08-04)**: 详细分析见下方。

**NVv 防御架构审查与清理 (2026-08-04)**:

审查结论: `--no-nvv` (enable_nvv=False) 从源头阻断 NVV bias → 模型不输出 NVV token。后处理还原函数存在固有缺陷（需要 pinyin_dir ≠ output_dir 才能访问原始 .lab），且在有 `--no-nvv` 的场景下完全不会触发。

删除的代码:

| 删除位置 | 函数/代码块 | 原因 |
|---------|------------|------|
| `ctc_prealign.py` ~989-1059 | `_reclaim_nvv_pinyin()` 完整函数 | 时序缺陷: 依赖 pinyin_dir/.lab，而 pipeline 中 pinyin_dir 可能等于 output_dir |
| `ctc_prealign.py` ~1835 | `_reclaim_nvv_pinyin(...)` 调用 | 同上 |
| `normalize_english_tokens.py` ~323-329 | NVV pre-reclaim 代码块 | 同时序缺陷: 读取 output_dir/.lab 已被 CTC 污染，`is_pinyin_syllable("BREATHING")=False` |

保留的防御:

| 层 | 位置 | 机制 | 可靠性 |
|----|------|------|--------|
| **Primary** | `ctc_prealign.py:230` | `enable_nvv=False` → 跳过 blank-frame NVV bias | ✅ 源头阻断 |
| **Secondary** | `normalize_english_tokens.py:303` | `not is_nvv_token(u)` 排除 NVV 参考词 | ✅ 防止 normalize_en 把拼音合并进 NVV |

最终 NVV 防御策略: **单层源头阻断**。`nvv_enabled: false` 配置 → `--no-nvv` → `enable_nvv=False` → blank-frame NVV bias 不执行 → 模型不输出 NVV token。无需任何后处理还原。

**`--no-nvv` 方案说明**:

`--no-nvv` 是解决 NVV 误判的**首选方案**。设置后 `make_patched_inference` 跳过 blank-frame NVV bias, 模型完全不会检测 NVV token, 仅用 CTC 锚点给参考文本做时间戳。相比后处理还原 (`_reclaim_nvv_pinyin`), 这是从源头消除问题。

```bash
# 命令行
python ctc_prealign.py ... --no-nvv

# 配置文件
ctc_prealign:
  nvv_enabled: false
```

---

## Case 32: 英文词自引用单音素 — pp 仅 1 个 phone (english_single_phone)

**日期**: 2026-08-05 (updated 2026-08-05)
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)
**触发场景**: 英文 token（非 NVV、非标点）的 pinyin_phones 只有 1 个自引用 phone（整词标签），而非 en:-prefixed ARPABET 音素。

### 现象

```
words:           RIA  [dict: R, IY1, AH0]
pinyin_phones:   RIA  [全区间]  ← 自引用整词, 未拆分为 ARPABET

words:           vup  [不在任何 dict]
pinyin_phones:   vup  [全区间]  ← 也不在 pinyin_dict, 同样自引用
```

英文版 Case 26 FULL_WORD_AS_PHONE。受影响词：RIA (59/200)、AI (32/200)、BGM (8/200)、live (6/200) 等。

### 根因

1. English MFA 对短英文词/缩写产出 phone 不足或未产出
2. `_apply_en_phones` 在 CMUdict/English MFA 均缺失时回退到自引用
3. `build_pinyin_phones_tier` 中 `is_english_token` 守卫使英文词走到自引用路径
4. 旧检查仅覆盖 pinyin_dict 中有 2+ 条目的英文词，漏掉了不在 dict 中的英文词

### 修改点

**CE. `process_one` QC section — `english_single_phone` 过滤（v2 增强）**

v1: 只检查 pinyin_dict 中有 2+ 音素的英文词。
v2: 移除 pinyin_dict 依赖，显式排除 silence/NVV/punct（`is_silence(_wt) or is_punct(_wt) or is_nvv_token(_wt)`），覆盖所有 `is_english_token()` 为 True 但 pinyin_phones 只有 1 个自引用 phone 的词。不在 dict 中的词标注 `(not in dict)`。

```python
# v2: 跳过 silence/punct/NVV, 只检查英文 token
if not _wt or is_silence(_wt) or is_punct(_wt) or is_nvv_token(_wt):
    ... skip ...
if not is_english_token(_wt):
    ... skip ...
# 自引用: 只有 1 个 phone 且等于词本身
if len(_w_phones) == 1 and _w_phones[0] in (_wt, _wt.lower(), _wt.upper()):
    _en_single_count += 1
```

### 关联样本

- `000004_直播流程_开场介绍.TextGrid`: `AI` → pinyin_phones `AI`
- `000023_直播流程_开场介绍.TextGrid`: `RIA` → pinyin_phones `RIA`
- 不在 dict 的词（如 `vup`, `VUp` 等）同样会被检测

---

## Case 33: 英文词音素不足 — pp 音素数 < dict 期望 (english_phone_deficit)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)
**触发场景**: 英文 token 的 pinyin_phones 有 ≥2 个 phone 但仍少于 pinyin_dict 期望。

### 现象

```
words:           BGM  [dict 期望: B, IY1, JH, IY1, EH1, M 共 6 个]
pinyin_phones:   en:B  +  en:M  (仅 2 个, 缺 4 个)
```

英文 MFA 产出 2 个 IPA phone 但 CMUdict 需要 6 个。虽未完全丢失，但音素序列不完整，TTS 训练时缺少关键音素过渡。

### 根因

1. `_apply_en_phones` 中 `n_cmu > n_ipa` 分支拆分 IPA 切片来塞入 CMUdict 音素
2. 当 `n_ipa ≪ n_cmu` 时，大量音素来自纯比例拆分，无真实声学边界参考
3. 部分 phone 可能完全缺失（如 BGM 只有 B 和 M，中间全部丢失）

### 修改点

**CF. `process_one` QC section — 新增 `english_phone_deficit` 过滤**

对英文 token，比较 pinyin_phones 实际非静音 phone 数量与 dict 期望数量。若 2 ≤ 实际 < 期望 → `filter_reasons.append("english_phone_deficit")`。

### 关联样本

- `000069_直播流程_开场介绍.TextGrid`: `BGM` got 1 phone, dict expects 6

---

## Case 34: pinyin_phones 轨道间隙 — 相邻 phone 不连续 (pp_tier_gaps)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)
**触发场景**: pinyin_phones tier 中相邻 interval 之间存在 >10ms 的时间间隙，破坏连续性。

### 现象

新版合成英文数据对齐中 26.7% 文件存在 pp_tier_gaps，间隙分布为：
- `zh→zh` (中文词间): 41 处
- `punct→zh`: 13 处  
- `zh→en`: 13 处

### 根因

pp_tier_gaps 通常继承自 words tier 的间隙。当 words tier 中相邻词之间存在未吸收的 MFA 帧精度残余间隙 (<30ms)，pinyin_phones 也随之产生间隙。

### 修改点

**CG. `process_one` QC section — 新增 `pp_tier_gaps` 过滤**

扫描 pinyin_phones tier，统计相邻 interval 之间 >10ms 的间隙数。>0 处 → `filter_reasons.append("pp_tier_gaps")`。

---

## Case 35: words 轨道间隙 — 非静音词间空洞 (words_tier_gaps)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)
**触发场景**: words tier 中相邻非静音词之间存在 >20ms 的时间间隙。

### 现象

纱依: 21/2894 (0.7%)，花礼: 306/22285 (1.4%)，新版英文: 160/200 (batch2 48%)

```
dao4 → zhen1   25ms 空洞
BREATHING → ke3 85ms 空洞
na4 → ，       873ms 空洞 (极端)
```

### 根因

1. MFA 对齐后词间存在帧精度残余间隙（5-30ms）
2. `_snap_to_ctc` 的 ≤5ms 间隙吸收 (Case 7, T 修改点) 阈值不够覆盖
3. 部分大间隙来自 NVV token 或标点处理后的边界残余

### 修改点

**CH. `process_one` QC section — 新增 `words_tier_gaps` 过滤**

扫描 words tier，统计相邻非静音词之间 >20ms 的间隙。>0 处 → `filter_reasons.append("words_tier_gaps")`。

---

## Case 36: 轨道系统性不连续 (tier_discontinuity)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)
**触发场景**: 任意轨道中 >10% 的 interval 存在 >10ms 间隙，表明系统性问题而非孤立 MFA 残余。

### 现象

```
hanzi(7/51), words(7/51) — 13.7% interval 有间隙
```

与 Case 35 的区别：Case 35 检测孤立间隙（一处也报），Case 36 检测系统性不连续（阈值 10%）。

### 修改点

**CI. `process_one` QC section — 新增 `tier_discontinuity` 过滤**

对每个轨道，统计 >10ms 的间隙，若超过 interval 总数的 10% → `filter_reasons.append("tier_discontinuity")`。

### 修改点汇总

| ID | 位置 | Case | 过滤条件 |
|------|------|:--:|------|
| CE | `process_one` QC | 32 | `english_single_phone` — 英文词自引用单音素 (v2: 不限 dict, 排除 NVV/punct) |
| CF | `process_one` QC | 33 | `english_phone_deficit` — 英文词音素不足 |
| CG | `process_one` QC | 34 | `pp_tier_gaps` — pinyin_phones 轨道间隙 |
| CH | `process_one` QC | 35 | `words_tier_gaps` — words 轨道间隙 |
| CI | `process_one` QC | 36 | `tier_discontinuity` — 轨道系统性不连续 |

---

---
## Case 37: en_mfa_windows 重复词覆盖 → English 词音素全部丢失 (english_phone_loss)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `build_pinyin_phones_tier`, `process_one`, `_sync_derived_tiers`, `_apply_en_phones`
**触发样本**: 新版合成英文数据 — RIA/BGM/AI 等重复英文词

### 现象

pinyin_phones tier 中英文词（RIA/BGM/AI 等）Phone 数严重不足：
- RIA (dict 3 phones) → pp 仅 1 个 (RIA 整词自引用): 131 例中 31 例不足
- BGM (dict 6 phones) → pp 1-2 个
- AI (dict 2 phones) → pp 仅 1 个: 17/18 例
共 35% 英文词受影响。

### 根因链

1. `en_mfa_windows` 按 `word_text` 作 key，同一文件中重复出现的英文词后者覆盖前者
2. Phase 5 `build_pinyin_phones_tier` 使用被覆盖后的窗口匹配，找不到匹配 → `word_phones = []`
3. 空 `word_phones` 导致 fallthrough 到自引用 label（`Interval(xmin, xmax, word_text)`）
4. Phase 3.5 的 `build_pinyin_phones_tier` 调用未传 `en_mfa_windows`，但 Phase 5 传了——造成 Phase 5 二次过滤时丢失已在 Phase 3.5 正确注入的 phones

### 修改点

**A. `en_mfa_windows` key 改为 `(word_text, start_time)`** (~line 4594)
- 从 `dict[str, tuple]` 改为 `dict[tuple[str, float], tuple]`
- 同一词不同出现有各自的时间窗口

**B. `build_pinyin_phones_tier` English phone 处理重写** (~line 558)
- `en:` 前缀 phones（_apply_en_phones 注入的）无条件保留
- 非 `en:` phones 用时间限定 key 查找 MFA 窗口过滤
- 向后兼容裸 string key

**C. Phase 3.5 传入 `en_mfa_windows`** (~line 4637)
- 确保 English phones 在任何阶段都能被正确识别

### 关联样本

- 新版合成英文数据对齐 → ria/花礼/雪狐桑 — RIA token 131 例中 31 例不足

---

## Case 38: MFA/CTC 边界 2ms 死区 → 文本重叠 (boundary_overlap_deadzone)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`, `_fix_overlapping_boundaries`
**触发样本**: 新版合成英文数据 — 中文词与英文 token 相邻重叠 2ms (44 files, 11%)

### 现象

words tier 中文词与英文 token 相邻时出现 2ms 边界重叠：
```
Instagrams[5.862-6.762] ↔ 上[6.760-7.070]  重叠 2ms
R[6.522-6.882]         ↔ 的[6.880-6.980]  重叠 2ms
```

### 根因链

1. English token 强制使用 CTC 边界 (`use_mfa=False`)，中文词使用 MFA 边界
2. CTC 帧移 40ms vs MFA 帧移 10ms → 系统精度差异 → 微重叠
3. `_snap_to_ctc` 重叠预防阈值 0.002s（2ms）：恰好 2ms 重叠时条件 `word_start < prev_end - 0.002` 为 False
4. `_fix_overlapping_boundaries` 的 5ms floor：2ms < 5ms → 被跳过
5. `_fix_overlapping_boundaries` 对 English/NVV token 重叠无专门处理

### 修改点

**A. `_snap_to_ctc` 重叠零容忍** (~line 3787)
- `prev_end - 0.002` → `prev_end`（任何重叠都修复）

**B. `_fix_overlapping_boundaries` 全面重写** (~line 976)
- 5ms floor → 0.5ms floor
- 新增 English/NVV + content word 重叠处理：clip English/NVV 侧
- `cur_is_content` / `nxt_is_content` 移除 NVV 排除（让 English/NVV 也能参与重叠修复）

**C. `_snap_to_ctc` 新增微重叠吸收** (~line 3937)
- 在 tiny gap 吸收循环中同步吸收 ≤ 3ms 重叠

### 关联样本

- 新版合成英文数据对齐 → batch1 (直播流程) + batch2 (礼物互动) — 44/400 文件

---

## Case 39: MFA 帧精度间隙 5-30ms → words/hanzi/pp 三层间隙 (frame_precision_gaps)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`, `_absorb_tiny_gaps` (new), `_sync_derived_tiers`, `process_one`
**触发样本**: 新版合成英文数据 — Batch1 18%, Batch2 48% 文件有间隙

### 现象

words 和 hanzi tier 存在 5-30ms 间隙（完全镜像，证明同步正确但源头有洞）：
- Batch1 (直播流程): 33/188 文件 (18%) — 41 处间隙
- Batch2 (礼物互动): 86/180 文件 (48%) — 160 处间隙
- 模式: lao2→lao2 15ms, tui1→le5 30ms, idol→... 5ms

连锁导致 pinyin_phones tier 27% 文件不连续。

### 根因链

1. MFA 帧移 10ms → 词边界精度 ±10ms
2. `_snap_to_ctc` gap 吸收阈值仅 5ms → 5-30ms gap 被保留为 `<spN>` 标签
3. `_inject_punctuation` 的词间 gap 处理仅针对标点邻接，通用词间间隙无吸收逻辑
4. pp tier 重建时继承 words tier gap → 三层间隙

### 修改点

**A. `_snap_to_ctc` gap 吸收阈值 5ms → 30ms** (~line 3929)
- 吸收 ≤ 30ms (3 MFA 帧) 的间隙

**B. 新增 `_absorb_tiny_gaps()` 函数** (~line 1052)
- 通用微间隙吸收：遍历 words tier，≤ 30ms 的 silence gap 吸收到邻近词
- 集成进 `_sync_derived_tiers`，每次同步自动吸收

**C. pp tier 微间隙吸收** (~line 5010)
- Phase 5 重建 pp tier 后，吸收 ≤ 10ms 间隙和 ≤ 3ms 微重叠

### 关联样本

- 新版合成英文数据对齐 → 全量 6881 文件中 1839 个存在 pp 间隙

---

## Case 40: English phone 边界未 snap → phone 与 word 不对齐 (en_phone_boundary_offset)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_apply_en_phones`, `process_one` (Phase 5)
**触发样本**: 新版合成英文数据 — 80 处英文 phone 边界偏移

### 现象

英文 token 的 phone (pinyin_phones tier) 起点/终点与 word (words tier) 边界不对齐。

### 根因链

1. `_apply_en_phones` 将 English MFA 音素线性映射到 CTC-snapped word 边界，但不做首尾 snap
2. English MFA 在 padded segment 上运行，其 word_start/word_end 与 CTC-snapped words tier 边界存在系统性偏移
3. 线性映射保留了这个偏移
4. Phase 5 pp 重建时显式跳过 English 词的 first phone snap（`if not is_en`）

### 修改点

**A. `_apply_en_phones` 注入后 snap** (~line 4165)
- 每个 English word 注入 phones 后，snap 首 phone start 到 word start，尾 phone end 到 word end

**B. Phase 5 移除 `is_en` 限制** (~line 4965)
- first phone snap 对所有词生效（English MFA phones 可能在 Phase 4 边界变更后偏移）

### 关联样本

- 新版合成英文数据对齐 → 80 处偏移

---

## Case 41: CTC 锚点膨胀 → 异常长词 + 无检测 (ctc_anchor_inflation)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`, `detect_issues`
**触发样本**: 新版合成英文数据 — le5 = 5.6s (hao3[9.720-9.914] → le5[9.720-15.554] → 。[15.554-15.987])

### 现象

le5 从正常 0.2s 被拉到 5.6s（在 hao3 和 。 之间吞掉大段静音/未识别内容）。

### 根因链

1. NVASR CTC 将 hao3 和 。 之间的大段内容归入 le5 的 token span
2. `_snap_to_ctc` 的 duration ratio 规则检测到 ctc_dur(5.6s) >> mfa_dur(0.2s)，触发 `use_mfa=False`
3. `ratio_skip` 保护未触发：trailing silence 不是 `<eps>` 形态，gap_sil 检测也漏过
4. 后处理管线有 `fix_short_words` / `word_too_short` 检测，但无对应 `word_too_long` 检测

### 修改点

**A. CTC_MAX_DUR 绝对保护** (~line 3713)
- CTC duration > 3s 且 MFA duration < 1s 且非 English/NVV → 强制 `ratio_skip = True`
- CTC end 超过 MFA end 500ms+ → 同样强制 `ratio_skip`

**B. `detect_issues` 新增 `word_too_long`** (~line 2408)
- 中文词 > 3s、English/NVV > 8s → 标记异常

### 关联样本

- 新版合成英文数据对齐 → hao3→le5→。
- 新版合成英文数据中可能有更多同类 case

---

## Case 42: pinyin_in_text 误报 — `<sp1>` 被正则匹配为拼音 (sp1_false_positive)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)
**触发样本**: `shayi_huali_new` — 纱依/花礼 300 文件 100% 命中

### 现象

`shayi_huali_new` 数据集的 filtered 目录中，100% 的文件都有 `pinyin_in_text` 过滤原因。
实际情况是 raw_text tier 以 `<sp1>` 开头，正则 `\b[a-z]+[1-5]\b` 将 `sp1` 误判为拼音音节。

```
raw_text:  "<sp1>好久不见你个riaRIA..."
regex hit: "sp1"        ← 误报：这是 silence marker，不是 pinyin
真实拼音:  (无)           ← 没有任何真正的拼音泄漏
```

### 根因链

1. `_finalise_textgrid` (Phase 2) 在 `final_text` 前追加 `<sp1>`（raw_text 需要保留 `<sp1>`——这是设计要求）
2. QC 正则 `\b[a-z]+[1-5]\b` 在未经 strip 的 raw_text 上运行，将 `sp1` 匹配为拼音音节：
   - `\b` 在 `<`（非 word char）和 `s`（word char）之间匹配
   - `[a-z]+` 匹配 `sp`，`[1-5]` 匹配 `1`
   - `\b` 在 `1`（word char）和 `>`（非 word char）之间匹配
3. `<sp2>`, `<sp3>` 同理会被匹配为 `sp2`, `sp3`

### 修改点

**QC 正则双保险** (~line 5289-5293) — 仅修改检查侧，不修改 raw_text 内容：

- Layer 1: `_re.sub(r'<sp\d+>', '', _iv.text)` — 检查前 strip 所有 `<spN>` 标签
- Layer 2: `(?!sp\d\b)` 负向预查 — 即使标签残留也不匹配

### 影响范围

修复前：`shayi_huali_new` 100% 文件被标记 `pinyin_in_text`，纱依 8730 + 花礼 11034 = 19764 文件被过滤
修复后：`pinyin_in_text` 回归真实语义（真正的拼音泄漏才触发），过滤数大幅下降

### 关联样本

- `shayi_huali_new` — 纱依 8730 filtered, 花礼 11034 filtered
- 新版合成英文数据对齐 — 同样受影响

---

## Case 43: build_pinyin_phones_tier 英文词无 pinyin_dict fallback → 缩写词自引用

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `build_pinyin_phones_tier`
**触发样本**: 新版合成英文数据对齐 — RIA(59), AI(32), ria(16), BGM(8), live(6)

### 现象

英文缩写词在 pinyin_phones tier 中只有自引用（整词作为单音素），而 pinyin_dict 已有按字母拆好的 ARPABET 音素：

```
words:         AI   [dict: EY1, AY1]
pinyin_phones: AI   [全区间]  ← 自引用，未拆分为 EY1 + AY1

words:         BGM  [dict: B, IY1, JH, IY1, EH1, M]
pinyin_phones: BGM  [全区间]  ← 自引用，丢失全部 6 个 ARPABET
```

### 根因链

1. **en_phones JSON 目录不存在** → `en_data = None`（未跑 English MFA 或输出路径不匹配）
2. `_apply_en_phones` (Phase 3.5) 收到 `en_data=None` → 直接 return，不注入任何英文 phone
3. `build_pinyin_phones_tier` (Phase 5) 处理英文词时：
   - `is_english_token` → True
   - `en_mfa_windows` → None
   - `word_phones` → 空（中文 MFA 无法对齐英文词）
   - **直接自引用** — 没有检查 pinyin_dict
4. pinyin_dict 早已包含这些词的 ARPABET 分解（RIA/BGM/AI 等是 pipeline 上一版本留下的固定词条），但代码未使用

### 典型缩写词的 pinyin_dict 映射

| 词 | pinyin_dict ARPABET | 字母拆分 | CMUdict |
|------|------|------|:--:|
| AI | `EY1, AY1` | A(EY1)+I(AY1) | ✅ |
| RIA | `R, IY1, AH0` | R(AA1,R)+I(AY1)+A(EY1) | ✅ |
| BGM | `B, IY1, JH, IY1, EH1, M` | B+G+M | ❌ |
| live | `L, AY1, V` | — | ✅ |

AI 的字母拼读恰好等于整词发音（A+I = EY1+AY1），RIA/BGM 按字母拆分。

### 修改点

**CJ. `build_pinyin_phones_tier` — 英文词自引用前增加 pinyin_dict 均匀分布** (~line 633)

修改前：
```python
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
            continue
```

修改后：
```python
            # pinyin_dict fallback for English words (Case 43)
            if dict_phones and len(dict_phones) >= 2:
                n = len(dict_phones)
                for i, arpa in enumerate(dict_phones):
                    s = round(w_iv.xmin + (i / n) * word_dur, 4)
                    e = round(w_iv.xmin + ((i + 1) / n) * word_dur, 4)
                    new_intervals.append(Interval(s, e, arpa))
            else:
                new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
            continue
```

### 关联样本

- `000004_直播流程_开场介绍.TextGrid`: `AI` → self-reference, dict=`['EY1','AY1']`
- `000023_直播流程_开场介绍.TextGrid`: `RIA` → self-reference, dict=`['R','IY1','AH0']`
- `000069_直播流程_开场介绍.TextGrid`: `BGM` → self-reference, dict=`['B','IY1','JH','IY1','EH1','M']`

---

## Case 44: 声母占比异常 — MFA phone boundary 过晚 (init_too_long)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `build_pinyin_phones_tier`
**触发样本**: `shayi_huali_new` — 10.1% 音节 (142/1404)

### 现象

```
hao3   总长270ms → h=190ms(70%)   ao3=80ms(30%)     正常: h~80ms(30%)
shang4 总长180ms → sh=130ms(72%)  ang4=50ms(28%)    正常: sh~60ms(33%)
de5    总长160ms → d=110ms(69%)   e5=50ms(31%)      正常: d~40ms(25%)
```

声母占词长的 60-75%，韵母被严重压缩，触发 `long_consonant_phone`。

### 根因链

1. `build_pinyin_phones_tier` 正常分支用 MFA 第一个 phone 的 xmax 作声母→韵母分界点
2. MFA 在擦音/送气音+元音的模糊过渡处（如 h→ao, sh→ang）把 phone boundary 放得过晚
3. 代码无条件信任该 boundary，不做比例校验
4. Case 26 的 proportional split 只在 `word_phones <= 1` 时触发，`>= 2` 时不检查

### 修改点

**A. 新增 `_INIT_MAX_FRAC` 字典** (~line 445)
- 按语音学规律设定每类声母的最大词长占比上限
- 不送气塞音 (b/d/g): 35%, 送气塞音 (p/t/k): 40%, 擦音 (h/sh/x/f/s/r): 50%, 塞擦音 (zh/ch/z/c/j/q): 45%

**B. `build_pinyin_phones_tier` 正常分支加比例保护** (~line 668)
- 当 MFA boundary 给出的声母占比 > `_INIT_MAX_FRAC` 上限时，回退到 proportional split
- 仅对 > 60ms 的词生效（超短词不受限）

### 关联样本

- `shayi_huali_new` — 纱依/花礼 filtered 文件

---

## Case 45: 韵母被词边界拉长 — Phase 5 尾 phone 无条件扩展 (final_too_long)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (Phase 5 pp rebuild)
**触发样本**: `shayi_huali_new` — ~2% 音节，但极端 case 韵母 900ms+

### 现象

```
piao4 总长1066ms → p=80ms  iao4=986ms(92%)    MFA原始: p~80ms iao4~290ms
ming2 总长700ms  → m=70ms  ing2=630ms(90%)    MFA原始: m~70ms ing2~200ms
```

声母保持 MFA 原始时长（正常），韵母被扩展填满词尾，触发 `long_vowel_phone`。

### 根因链

1. Postprocessing 把词边界拉长（CTC snap / silence absorption / NVV extension）
2. Phase 5 尾 phone 扩展逻辑无条件把韵母 end 推到 `w_iv.xmax`
3. 没有时长上限检查 — 词被拉多长，韵母就填多长

### 修改点

**Phase 5 尾 phone 扩展加时长上限** (~line 5057)
- Vowel/final: max(400ms, orig_dur × 1.5)
- Consonant/initial: max(200ms, orig_dur × 1.5)
- Single-phone word: max(500ms, orig_dur × 1.5)
- 超出部分不填充，保留为自然 silence gap

### 关联样本

- `shayi_huali_new` — `piao4` 986ms, `ming2` 630ms, `huang3` 415ms

---

## Case 46: pp tier phone↔punct 结构重叠 (pp_punct_overlap)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_inject_punctuation`
**触发样本**: `shayi_huali_new` — 32.5% 文件 (13/40)

### 现象

pp tier 中不同 label 的 Interval 重叠 10-155ms：
```
，[4.000-4.750]  ↔  z[4.610-4.659]    overlap=140ms  (punct↔content)
。[2.370-2.825]  ↔  …[2.670-2.825]    overlap=155ms  (punct↔punct)
```

导致 `long_consonant_phone` 和结构异常的误报。

### 根因链

1. `_inject_punctuation` 把 CTC 标点注入 pp tier 时使用 CTC 时间戳
2. CTC 标点可能落在 content word 内部（帧移 40ms 精度限制）
3. pp tier 重建时 phone 被 max/min clip 到 word 范围，但未检查 phone↔punct 重叠

### 修改点

**pp tier 重建后增加重叠解决步骤** (~line 2790)
- Punct→content: punct 保留 ≥60ms，phone 从 punct end 开始
- Content→punct: content 在 punct start 处截断
- Punct→punct: 各保留 ≥60ms，后一个从 prev end 开始
- Content→content: midpoint split

### 关联样本

- `shayi_huali_new` — 纱依 13/40 文件

---

## Case 47: init_only_phone 检测改进 — 单 phone 音节分类 (single_phone_audit)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)
**触发样本**: `shayi_huali_new` — 2182 个单 phone 音节，0 个真缺失韵母

### 现象

此前分析中 `init_only_phone` 被误报为 99.1% 命中率，实际数据中 0 个真 case。
原因是 2182 个单 phone 拼音音节全部是零声母音节（`yao2→iao2`、`wan3→uan3`），
pp tier 正确处理为 1 个 phone（韵母），不是 bug。

### 修改点

**检测逻辑改进** (~line 5748)
- Type A (init_only): 单 phone = dict initial → **真 bug**，触发过滤
- Type B (zero_initial): 单 phone ≠ dict initial → 零声母，正确行为，不触发
- Type C (self_ref): 单 phone = word text → 自引用 fallback，不触发
- Report 新增 `single_phone_breakdown` 字段：`{init_only_error, zero_initial_ok, self_ref_ok}`

### 关联样本

- `shayi_huali_new` — 纱依 896 + 花礼 1286 = 2182 个零声母（全部正确）

---

## Case 48: silence_boundary_split 误报 — MFA 真 phone 被误判 (sb_false_positive)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section — `silence_boundary_split` 检测)
**触发样本**: `shayi_huali_new` — 重处理后 53.4% 仍过滤文件命中

### 现象

重跑 postprocess 后，`silence_boundary_split` 成为最大剩余过滤原因（62/116 文件）。
典型 report 示例：

```
xie4→ɕ[10ms] +j e˥˩
hou4→x[10ms] +o˥˩ w
jia1→tɕ[10ms] +j a˥
nv3→n[10ms] +y˨˩˦
```

所有 case 的第一个 phone 都恰好 10ms，看起来像 silence 分界。

### 根因链

1. **检测逻辑**（line 5798-5840）：多 phone dict 词的首 phone < 10ms → 判定为 silence 分界
2. 检测注释假设：首 phone 过短 → 必定来自 MFA silence/spn fragment 的边界
3. **实际数据**：MFA 源文件的首 phone 是**真实的 IPA phone**（`ɕ`, `x`, `tɕ`, `n`），不是 silence

真实场景分两类：

**A. 连续相同词（~30%）**：
```
xie4[8.280-8.410] → xie4[8.410-8.575] → a5[8.575-8.815]
MFA phones for 2nd xie4: ɕ[8.410-8.420 10ms] j[8.420-8.430] e˥˩[8.430-8.460]
```
前一个 `xie4` 的 `e˥˩` 刚结束，后一个 `xie4` 的 `ɕ` 立即开始。MFA 在相同 phone 序列的边界处给出 10ms 的 `ɕ`——这是 MFA 声学模型的自然行为（phone 边界在相同过渡处更精确），不是错误。

**B. 普通短 phone（~70%）**：
```
shi2 → hou4[70ms] → gen1
MFA phones: x[10ms] o˥˩[30ms] w[30ms]
```
MFA 给 `x`（擦音声母）仅 10ms。虽然偏短，但仍是真实的 MFA phone，不来自 silence。

**关键证据**：所有 28 个受检 case 的 MFA 源 phone 全部是 `all_real=True`——没有 silence/spn。检测的假设（"来自 silence fragment"）不成立。

### 修改点

**检测逻辑修正** (~line 5829)：在判定为 `silence_boundary_split` 之前，增加检查——首个 pp phone 的**文本 label** 是否匹配 silence/spn 标签。如果是真实 IPA phone label（如 `ɕ`, `x`, `n`, `tɕ`），则不触发。

```python
# 修改前:
if len(_w_phones) >= 2:
    _first_dur = _w_phones[0].xmax - _w_phones[0].xmin
    if _first_dur < _SILENCE_SPLIT_FLOOR_S:
        _silence_split_count += 1  # ← 所有短首 phone 都触发

# 修改后:
if len(_w_phones) >= 2:
    _first_dur = _w_phones[0].xmax - _w_phones[0].xmin
    _first_label = _w_phones[0].text.strip()
    # Only flag when the first "phone" is actually a silence/spn label
    if _first_dur < _SILENCE_SPLIT_FLOOR_S and is_silence(_first_label):
        _silence_split_count += 1
```

### 关联样本

- `shayi_huali_new` — 62/116 重处理后仍过滤文件（53.4%）

---

## 模板 (新 Case 用)

```markdown
## Case N: [标题]

**日期**: YYYY-MM-DD
**涉及文件**: `scripts/xxx.py`
**涉及函数**: `xxx`, `yyy`
**触发样本**: xxx

### 现象

[修复前 vs 修复后的数据对比]

### 根因链

1. [步骤1]: ...
2. [步骤2]: ...

### 修改点

**X. `xxx` — 修改描述** (~line N)

[代码 diff]

### 关联样本

- [样本]
```

---

## Case 49: english_single_phone / english_phone_deficit 用中文拼音词典查英文词 → 误报 (wrong_dict_for_english_qc)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC section)

### 现象

`english_single_phone`（Case 32）和 `english_phone_deficit`（Case 33）两个 QC 检查在诊断消息和期望音素数判断中使用了中文拼音词典 `pinyin_dict`，而非英文 CMUdict：

1. **`english_single_phone`** (Case 32): 自引用检测逻辑本身正确（检查 phone 文本是否等于 word 文本），不依赖 `pinyin_dict`。但示例消息用 `pinyin_dict` 查找期望音素数来丰富诊断信息——对 English token 用中文拼音词典在语义上是错误的。例如 English "di" 在 `pinyin_dict` 为 `['d', 'i4']`（恰好是拼音），但正确的英文期望来自 CMUdict: `['D', 'IY1']`（2 个 ARPABET phones）。

2. **`english_phone_deficit`** (Case 33): **检测逻辑完全依赖 `pinyin_dict`**——对所有 English token 用 `pinyin_dict.get()` 查期望音素数。`pinyin_dict` 条目最多只有 2 个 phone（声母+韵母），条件 `_n_got >= 2 and _n_got < _n_exp` 要求 `_n_exp > 2`，**此条件在 pinyin_dict 上永远不成立**——该检查是死代码。

### 根因链

1. **字典语义错配**: `pinyin_dict` 是中文音节分解字典（如 `mao1 → ['m', 'ao1']`），English token 的正确字典是 CMUdict（如 `hello → ['HH', 'AH0', 'L', 'OW1']`）。
2. **Case 33 死代码**: `_n_exp = len(_dp) = 2`（最多），`_n_got >= 2 and _n_got < 2` → `_n_got < 2` 永远 False → 整个检查从不触发。
3. **Case 32 误报风险**: 当 English token 恰好也是合法拼音音节（如 "di" → `pinyin_dict["di"] = ['d', 'i4']`），且 G2P 失败产生自引用 phone，示例消息会显示误导性的 pinyin 期望 `['d', 'i4']` 而非 CMUdict 的真实期望 `['D', 'IY1']`。

### 影响

- **Case 32**: 检测逻辑不受影响（自引用检测独立于字典）。仅诊断消息使用错误字典。
- **Case 33**: 检查从未触发过——所有 English phone 不足的情况（如 CMUdict 期望 5 个 ARPABET phones 但 MFA 只产出 2 个）被**静默放过**，未进入 `filtered/`。

### 修改点

**A. QC 段加载 CMUdict** (~line 5386)

```python
# 修改前: 无 CMUdict 加载，QC 检查依赖 pinyin_dict
filter_reasons = []

# 修改后: 在 QC 开始处 lazy-load CMUdict
from pipeline_utils import _load_cmudict as _load_cmu
_cmu = _load_cmu()
```

**B. `english_single_phone` 诊断消息改用 CMUdict** (~line 5949-5954)

```python
# 修改前: 用 pinyin_dict 查 English token（语义错误）
_dp = (pinyin_dict.get(_wt) or pinyin_dict.get(_wt.upper())
       or pinyin_dict.get(_wt.lower())) if pinyin_dict else None
_en_single_examples.append(
    f"{_wt}→{_w_phones[0]!r}" +
    (f" (dict:{_dp})" if _dp else " (not in dict)"))

# 修改后: 用 _cmu (CMUdict) 查 English token
_cmu_entry = _cmu.get(_wt.lower())
_en_single_examples.append(
    f"{_wt}→{_w_phones[0]!r}" +
    (f" (cmu:{_cmu_entry})" if _cmu_entry else " (not in CMUdict)"))
```

**C. `english_phone_deficit` 检测逻辑完全改用 CMUdict** (~line 5965-6003)

```python
# 修改前: 用 pinyin_dict 查所有 English token（死代码——_n_exp 最大为 2）
if words_tier is not None and pp_tier is not None and pinyin_dict is not None:
    ...
    _dp = (pinyin_dict.get(_wt) or pinyin_dict.get(_wt.upper())
           or pinyin_dict.get(_wt.lower()))
    if not _dp or len(_dp) < 2:
        ...

# 修改后: 用 _cmu (CMUdict) 查 English token（_n_exp 可为 3-15）
if words_tier is not None and pp_tier is not None:
    ...
    _dp = _cmu.get(_wt.lower())  # CMUdict 条目的 phone 数通常 2-15
    if not _dp or len(_dp) < 2:
        ...
```

### 验证方法

```python
# english_phone_deficit 应对 CMUdict 中的多 phone English 词触发
# 例如: "hello" → CMUdict expects 4 phones, MFA produces only 2 → flagged

# english_single_phone 示例消息应显示 CMUdict 条目而非 pinyin_dict
# 修复前: "di→'di' (dict:['d', 'i4'])"    ← 误导：用的是中文拼音 dict
# 修复后: "di→'di' (cmu:['D', 'IY1'])"    ← 正确：用的是 CMUdict
```

### 关联样本

- 外部项目: 包含 English "di"/"can"/"tan" 等恰好是拼音音节的 English token 的段

---

## Case 50: Interval.mark 属性不存在 → postprocess 全线崩溃 (interval_mark_attr_mismatch)

### 现象

- 修复前: postprocess 17510/17541 条报 `'Interval' object has no attribute 'mark'`，0 条成功
- 修复后: 正常产出 TextGrid

### 根因链

1. `Interval` 类定义 (line 58-61) 使用 `.text` 属性名
2. `_worker_fn` → english QC 检测 (line 5941-5943) 访问 `_p.mark`
3. `AttributeError` 被 `as_completed` 捕获 (line 6294)，状态记为 `"error"`
4. 所有含 English token 的段均触发此路径，最终 0 成功

### 修改点

`scripts/postprocess_textgrids.py:5941-5943` — `_p.mark` → `_p.text` (3 处)

```python
# 修改前
if (_p.xmax > _ws + 0.001 and _p.mark
        and not is_silence(_p.mark.strip())):
    _w_phones.append(_p.mark.strip())

# 修改后
if (_p.xmax > _ws + 0.001 and _p.text
        and not is_silence(_p.text.strip())):
    _w_phones.append(_p.text.strip())
```

### 验证方法

```bash
python3 scripts/run_pipeline.py --config configs/hecheng_ria_0805.yaml \
    --python .../mfa-dev/bin/python --step postprocess --overwrite
# 预期: 产出 17000+ TextGrid，不再有 AttributeError
```

---

## Case 51: ctc_pretg_adj 空目录未回退 → postprocess 找不到 txt/lab (empty_ctc_adj_no_fallback)

### 现象

- 修复前: `--step postprocess` 报 17541 条 `Missing txt/lab: .../ctc_pretg_adj/...`
- 修复后: 正常从 ctc_pretg 读取参考文本

### 根因链

1. `ctx["ctc_pretg_adj"]` 目录始终被创建 (line 2549)，ctc_adjust 禁用时为空
2. `step_postprocess` (line 1502) 和 `step_align_en` (line 1433) 用 `.get("ctc_pretg_adj", ...)` 取值
3. 空目录 `.exists()` 返回 True → 被选为 `ctc_dir`
4. `--txt-dir` / `--raw-text-dir` 指向空目录 → postprocess 找不到任何 `.txt` / `.lab`

### 修改点

`scripts/run_pipeline.py:1434, 1503` — `.exists()` 检查后追加 `not any(ctc_dir.iterdir())`

```python
# 修改前
if not ctc_dir.exists():
    ctc_dir = ctx["ctc_pretg"]

# 修改后
if not ctc_dir.exists() or not any(ctc_dir.iterdir()):
    ctc_dir = ctx["ctc_pretg"]
```

注意: line 1290 (`step_align`) 已有 `.glob("*.lab")` 检查，无需修改。

### 验证方法

```bash
python3 scripts/run_pipeline.py --config configs/hecheng_ria_0805.yaml \
    --python .../mfa-dev/bin/python --step postprocess --overwrite
# 预期: 不再报 Missing txt/lab，正常产出 TextGrid
```

---

## Case 52: CTC 宽跨标点锚点 → _inject_punctuation 碎片喷溅 + _fix_overlapping_boundaries 阈值拒绝 → 词级大重叠 (wide_ctc_punct_fragmentation)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_inject_punctuation` (line 2537), `_fix_overlapping_boundaries` (line 1002)
**触发样本**: 036014（,↔ria 859ms / ,↔lai2 1370ms）, 036015（。↔lve4 2135ms）

### 现象

遍历 `filtered/` 输出的 words tier，大量相邻 interval 出现 xmax > 下一个 xmin 超过 5ms 的重叠：

**036014 — 单 CTC 逗号碎片化为 5 个重叠逗号：**
```
[2]  1.041-2.620  ，       ← CTC 锚点 start, 覆盖 ria+lai2+bao4+dao4
[3]  1.761-1.921  ria      ← 重叠 859ms (，xmax 2.620 > ria xmin 1.761)
[4]  1.921-3.510  ，       ← 从 ria end 延伸到 zuo2 start, 覆盖 lai2+bao4+dao4+la5
[5]  2.140-2.270  lai2     ← 重叠 1370ms
[7]  2.270-3.510  ，       ← 从 bao4 start 延伸到 zuo2 start
[9]  2.400-3.510  ，       ← 从 dao4 start 延伸到 zuo2 start
[11] 2.620-3.510  ，       ← 与 la5 完全重叠
```

**036015 — 宽跨句号：**
```
[25] 6.730-9.650  。       ← CTC 句号锚点宽 2.92s
[26] 7.515-8.930  lve4     ← 重叠 2135ms (。xmax 9.650 > lve4 xmin 7.515)
```

**根本规律**: CTC 标点锚点 `start_s↔end_s` 跨多个词（如 `，` [1.041, 3.321] 跨 2.28s, `。` [6.71, 11.24] 跨 4.53s），经过 `_inject_punctuation` 后不是被裁到下一个词边界，而是**碎裂成多个相互重叠的碎片**，每个碎片的 xmax 都停留在原 CTC end time 并被 step 6 延伸到最近词界。

### 根因链

1. **CTC 模型输出宽跨锚点**: CTC 标点检测在静音段上放置单个标点锚点，但其 `end_s` 往往延伸到下一个大句段开始（而非下一个词的 xmin）。例如 `，` [1.041, **3.321**] 覆盖了 ria、lai2、bao4、dao4、la5 五个词。CTC 给出的是**连续的 token→token 边界**（标点 end == 下一个 token start 在 CTC 层面），但 MFA 处理后词界与 CTC 锚点出现结构性偏移。

2. **`_inject_punctuation` 重叠消解循环使用固定 range** (line 2587):
   ```python
   for pi in range(len(resolved)):  # ← len(resolved) 只计算一次！
   ```
   当 wide punct 触发 "word inside punct" 分支 (line 2607-2616) 时，`resolved.append(right_part)` 将右半截追加到列表末尾，但 `range` 不会包含新增索引 → **右半截永远不会被重叠消解**。

3. **split 分支不更新 ps/pe** (line 2612-2616):
   ```python
   left_part  = (ps, ws, ptext, pkind)
   right_part = (we, pe, ptext, pkind)
   resolved[pi] = left_part        # 原地替换
   if right_part[1] > right_part[0] + 0.001:
       resolved.append(right_part)
   # BUG: ps, pe 未更新！下一轮 wi 仍用原始 ps 继续匹配
   ```
   导致内层循环对每个后续词都重复 split，每次 `resolved[pi]` 被覆写，**前面的 left_part 全部丢失**。最终只有最后一次 split 的 left_part 存活。

4. **碎片链累积**:
   - vs ria: split → append right [1.921, 3.321]
   - vs lai2: split → append right [2.270, 3.321]
   - vs bao4: split → append right [2.400, 3.321]
   - vs dao4: split → append right [2.620, 3.321]
   - vs la5: we≥pe → trim end to 2.620（幸存 left_part: [1.041, 2.620]）
   
   结果: 1 个幸存 left_part + 4 个未处理的 right_part = **5 个标点 interval 全部重叠**。

5. **Step 6 右边界延伸** (line 2673-2684): 每个未处理的 right_part 被延伸到 `<500ms` 外的下一个词 `zuo2` start (3.510s) → 所有碎片都伸长到 3.510s。

6. **`_fix_overlapping_boundaries` 阈值拒绝** (line 1068):
   ```python
   elif is_punct(cur_text) and nxt_is_content and overlap < 0.100:  # ← 100ms 硬阈值
   ```
   碎片产生的重叠均在 700-2000ms → 远超 100ms → 被跳过，留给下游 QC filter 标记（但标记后文件已被归档，不会自动修复）。

7. **`absorb_silence_into_punct` 无法补救** (line 1280): 此函数依赖 `is_punct(cur) and is_silence(next)` 模式——但当标点已经**重叠进下一个词内部**时，标点和词之间没有静音 interval，吸收条件永远不满足。标点直接侵入词内，不是"标点+静音+词"的正常序列。

### 修改点

**A. `_inject_punctuation` — 将 for-range 改为 while 循环** (~line 2587)

```python
# 修改前 (line 2587)
for pi in range(len(resolved)):

# 修改后
pi = 0
while pi < len(resolved):
```

**B. `_inject_punctuation` — split 分支更新 ps + 保留 left_part 历史** (~line 2607-2616)

```python
# 修改前
else:
    # word inside punct (ws > ps and we < pe):
    # split punct into left part (before word) + right part (after word)
    left_part  = (ps, ws, ptext, pkind)
    right_part = (we, pe, ptext, pkind)
    resolved[pi] = left_part
    if right_part[1] > right_part[0] + 0.001:
        resolved.append(right_part)

# 修改后
else:
    # word inside punct (ws > ps and we < pe):
    # split punct into left part (before word) + right part (after word)
    # Regr. Case 52: ps must be advanced past the word so subsequent
    # word checks use the right part's position.  Left parts are
    # accumulated in a list and appended after the inner loop to
    # avoid overwriting resolved[pi].
    _left_parts = [(ps, ws, ptext, pkind)]
    ps = we  # advance past this word
    # Continue inner loop; further overlaps will split against updated ps.
    # After inner loop, replace resolved[pi] with first fragment and
    # append the rest (including the final trailing portion).
```

**C. `_inject_punctuation` — 内层循环结束后提交碎片** (~after line 2616)

在内层 `for wi` 结束后、`pi += 1` 前增加：

```python
    # After inner word loop: commit accumulated fragments
    if _left_parts:
        resolved[pi] = _left_parts[0]
        for frag in _left_parts[1:]:
            if frag[1] > frag[0] + 0.001:
                resolved.append(frag)
        # Trailing portion after last word
        trailing = (ps, pe, ptext, pkind)
        if trailing[1] > trailing[0] + 0.001:
            resolved.append(trailing)
```

**D. `_fix_overlapping_boundaries` — 移除/放宽 punct-content 重叠阈值** (~line 1068)

```python
# 修改前
elif is_punct(cur_text) and nxt_is_content and overlap < 0.100:

# 修改后: 无条件裁切标点 (标点流进内容词永远不合理)
elif is_punct(cur_text) and nxt_is_content:
```

同理 ~line 1063:
```python
# 修改前
elif cur_is_content and is_punct(nxt_text) and overlap < 0.100:

# 修改后
elif cur_is_content and is_punct(nxt_text):
```

### 验证方法

```python
# 用 036014 的 CTC 数据模拟 _inject_punctuation：
# 修复前: 1 个 CTC "，" → 输出 5 个重叠逗号，最大重叠 1370ms
# 修复后: 1 个 CTC "，" → 输出 ≤1 个逗号（或按词间间隙分段的多个非重叠逗号），所有 overlap ≤ 5ms

# 批量验证
python3 scripts/audit_textgrids.py --dir /mnt/nvme3/mfa_workspace_ria_0805/filtered/ \
    --check overlaps --threshold 0.005
# 预期修复后: overlap 报告从 ~7000+/7698 文件降至 <100
```

### 关联样本

- `/mnt/nvme3/mfa_workspace_ria_0805/filtered/036014_*.TextGrid` — 碎片化逗号（5→1）
- `/mnt/nvme3/mfa_workspace_ria_0805/filtered/036015_*.TextGrid` — 句号侵入 lve4 2135ms
- 整体统计: 7698 个 filtered 文件中大量出现同模式标点→词重叠

---

## Case 53: 拼音-汉字全局错位 — STT 错误→pypinyin 上下文污染→级联位移 (pinyin_displacement)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (QC 段)
**测试目录**: `\\RS3621\Research_TTS\Data\Raw\0805test` (9,843 文件)
**影响范围**: 8,098/9,843 (82.3%) 文件存在拼音-汉字不匹配
**严重程度**: 6,480/9,843 (65.8%) 文件错位率 > 30%

### 现象

MFA 对齐产出的最终 TextGrid 中，words tier 的拼音与 hanzi tier 的汉字之间存在大规模系统性错位。一个汉字被赋予相邻汉字的拼音，导致从错位起点到文件末尾的所有字符全部偏移。

**典型样本** (`036000_弹幕互动_回应吐槽弹幕.TextGrid`):

```
位置  hanzi  words(实际)  期望pinyin   状态
 13   播      he2          bo1          ← 错位起点
 14   和      fan4         he2          ← 被传染
 15   饭      na3          fan4         ← 继续偏移
 ...后续 22/31 个字符全部偏移...
```

**错位类型分布** (7,921 个位移段):
- SCRAMBLED（STT 错词导致非简单平移）: 4,323
- LEFT -1 位置偏移: 3,355
- RIGHT +1 位置偏移: 243

### 根因链

1. **NVASR STT 产生错误文本** (`ctc_prealign.py`): 语音识别在混合中英文、噪声、NVV 环境下产生大量错误:
   - 字符插入: "AP**播**和饭" 应为 "AP和饭"
   - 字符替换: "**兽**大家" 应为 "**受**大家"
   - 英文词碎片化: `Macbook` → `MAacbook`（驼峰融合）、`Instagram` → `Instagramstagram`（自重复拼接）
   - 词序错乱: "**好**评论区告诉" 应为 "评论区告诉**我**"

2. **pypinyin 逐字转换被污染**: pypinyin 对每个汉字独立做拼音转换。当 STT 多出了 "播" 这个字时，pypinyin 为它生成了正确的拼音 "bo1"——但后续所有字的拼音序列整体错位了一格。

3. **MFA 使用错误拼音做强制对齐**: `.lab` 中的错误拼音序列 → MFA 在音频上做强制对齐 → 相邻 token 边界互相挤占 → 音素边界漂移。

4. **计数一致但映射全错**: words tier 的拼音 token 数量与 hanzi tier 的汉字数量始终相同，所以现有的 `cjk_mismatch` 检查（仅比对整个 CJK 字符串）无法发现错位。

5. **位移无自愈机制**: 一旦偏移开始，后面所有字符都会继续错位，直到文件末尾。

### MAacbook / Instagramstagram 类畸形文本的起源

通过全文检索证实: `MAacbook`、`Instagramstagram`、`Instagramsta`、`Vcocal` 等畸形英文词**来自 NVASR STT 输出**，而非输入源文本:

1. **NVASR SenseVoice-Small 的固有限制** (~200M 参数):
   - 英文大小写边界区分能力弱 → `Macbook` → `MA` + `acbook` → `MAacbook`（解码器 token 融合错误）
   - 英文长词自重复拼接 → `Instagram` → `Instagram` + `stagram` → `Instagramstagram`
   - 英文词被拼音化 → `vocal` → `V` + `cocal` → `Vcocal`

2. **`normalize_english_tokens.py` 未覆盖**: 该步骤仅处理 NVASR 的拼音碎片→英文原词（如 `li ve` → `live`），不处理 ASR 直接输出的畸形英文。

3. **完整路径**:
```
原始语音: "用 Macbook 学 PV 真的太方便啦"
    ▼ NVASR STT
raw_text: "用MAacbook学PV真的太方便啦"
    ▼ pypinyin chars_and_pinyin
.lab: ... MAacbook xue2 PV ...
    ▼ MFA align (MAacbook 不在中英文词典中 → self-referential phone)
words tier: MAacbook
pinyin_phones: MAacbook (单个自引用 phone, 时长异常)
```

### 修改点

**`scripts/postprocess_textgrids.py` — `process_one` QC 段** (~line 5755, `cjk_mismatch` 检查之后)

新增 `pinyin_displacement` 检查 (~80 行):

对 hanzi tier 中每个 CJK 字符:
1. 用 `pypinyin.lazy_pinyin(style=TONE3, neutral_tone_with_five=True)` 获取期望拼音
2. 去除声调数字后与 words tier 的实际拼音对比
3. 检测连续 ≥3 字符的错位段（displacement runs）
4. 触发条件: `mismatch_rate ≥ 25%` OR `任一错位段 ≥ 6 字符`
5. 触发后: `filter_reasons.append("pinyin_displacement")` → 文件写入 `filtered/`

与现有检查的关系:
| 检查 | 检测什么 | pinyin_displacement 覆盖吗? |
|------|---------|---------------------------|
| `hanzi_pinyin` | hanzi tier 中残留的拼音 token（如 "qie4"） | 否 — 互补 |
| `cjk_mismatch` | raw_text CJK 序列 vs hanzi CJK 序列字面不一致 | 否 — 互补 |
| **`pinyin_displacement`** | **words tier 拼音 vs hanzi 期望拼音不一致** | **是 — 新增覆盖** |

### 修复方案分析 (四种方向对比)

| 方案 | 阶段 | 策略 | 优点 | 缺点 | 推荐度 |
|------|------|------|------|------|-------|
| **A** | ctc_prealign | 拼音一致性验证 + 拦截 | 源头阻止，不浪费 MFA 计算 | 需修改上游，需确定拦截后策略 | ★★★ |
| **B** (已实现) | postprocess | 检测 + 过滤到 filtered/ | 改动最小，纯防御 | 已浪费 MFA，不修复数据 | ★★☆ |
| **C** | 全管线 | STT→纠错→重对齐 | 根本修复 | 复杂度高，需纠错模型 | ★☆☆ |
| **D** | postprocess | pypinyin 原地纠正 words tier | 修复而非丢弃 | 多音字声调不准，可能引入新错误 | ★★☆ |

### 验证方法

```python
# 已知错位文件 → 应被过滤
#   036000: 71% mismatch, 3 runs → FILTERED ✓
# 已知正常文件 → 应通过
#   036040: 0% mismatch, 0 runs → PASS ✓

# 批量验证:
python3 scripts/postprocess_textgrids.py --txt-dir ... --textgrid-dir ... \
    --output-dir ... --filtered-dir ... --pinyin-dict ...
# 预期: filtered/ 中包含 pinyin_displacement 文件,
#       report 中包含 "pinyin_displacement" 字段及 mismatch_rate / runs

# 统计分析:
python3 scripts/audit_textgrids_deep.py /mnt/Raw/0805test
# 获取全量错位率分布
```

### 关联样本

- `036000_弹幕互动_回应吐槽弹幕.TextGrid`: 拼音-汉字全局错位 (71% mismatch, 22/31 chars)
- `036046_弹幕互动_回应吐槽弹幕.TextGrid`: `MAacbook` (STT 驼峰融合)
- `036038_弹幕互动_回应吐槽弹幕.TextGrid`: `Instagramstagram` (STT 自重复拼接)
- `036075_弹幕互动_回应吐槽弹幕.TextGrid`: `Instagramsta` + 全局错位

---

## Case 54: NVV 边界夹制豁免 → 实词边界自由越过 NVV → 标点区间倒置 (nvv_clamp_skip)

### 现象

修复前: CTC 边界调整 (`adjust_ctc_boundaries.py`) 对 NVV token 的边界夹制在 4 处被显式跳过：
- Part 1 推结束边界时，若下一词是 NVV 则不夹制 → 实词 end 可越过 NVV start
- Part 2 扩展/缩短结束边界时，若下一词/上一词是 NVV 则不夹制 → 同上
- Part 1 和 Part 2 独立修改 `punct.start_s` / `punct.end_s` 无交叉验证 → 标点 start_s > end_s（倒置区间）
- 倒置区间被行 320-322 的创可贴修补：`end_s = start_s + 0.060`（盲补 60ms 宽度）

结果：`overlapping_words` 过滤高发。

### 根因链

1. **NVV 声学透明 ≠ 时间透明**: NVV 的 CTC 边界确实不可靠（跳过能量搜索合理），但 NVV 仍占用时间区间——实词不应越过它。
2. **Part 1/Part 2 独立操作同一标点**: Part 1 回推 `punct.end_s`（跟实词 start），Part 2 前推 `punct.start_s`（跟实词 end），两个循环无交叉验证，可产生 `start_s > end_s`。
3. **创可贴掩盖根因**: 行 322 的 `p["end_s"] = p["start_s"] + 0.060` 防止崩溃但不修复边界交叉。

### 修改点

**A. `adjust_ctc_boundaries.py` 行 166-167 — Part 1 夹制移除 NVV 豁免**

```python
# 修改前: NVV 豁免夹制
if not _is_nvv(next_tok["word"]):
    pushed_end = min(pushed_end, next_tok["start_s"] - 0.02)

# 修改后: 无条件夹制到下一词起始
pushed_end = min(pushed_end, next_tok["start_s"] - 0.02)
```

**B. `adjust_ctc_boundaries.py` 行 205-207 — Part 2 夹制移除 NVV 豁免**

```python
# 修改前: NVV 豁免夹制
if next_tok and not _is_nvv(next_tok["word"]):
    if new_end >= next_tok["start_s"] - 0.02:
        new_end = next_tok["start_s"] - 0.02

# 修改后: 无条件夹制
if next_tok:
    if new_end >= next_tok["start_s"] - 0.02:
        new_end = next_tok["start_s"] - 0.02
```

**C. `adjust_ctc_boundaries.py` 行 221-224 — Part 2 缩短分支加 NVV 前向夹制**

```python
# 修改前: 无 NVV 前向夹制
elif new_end < old_end - 0.04:
    if new_end <= tok["start_s"] + 0.04:
        continue
    tok["end_s"] = new_end

# 修改后: 前一词是 NVV 时夹制 new_end 不小于 NVV end + 0.02
elif new_end < old_end - 0.04:
    if new_end <= tok["start_s"] + 0.04:
        continue
    if idx > 0 and _is_nvv(tokens[idx - 1]["word"]):
        prev_nvv_end = tokens[idx - 1]["end_s"]
        if new_end < prev_nvv_end + 0.02:
            new_end = prev_nvv_end + 0.02
    tok["end_s"] = new_end
```

**D. `adjust_ctc_boundaries.py` 行 320-322 — 替换创可贴为合理修复**

```python
# 修改前: 盲补 60ms 右侧宽度
for p in adj_punct:
    if p["end_s"] <= p["start_s"]:
        p["end_s"] = p["start_s"] + 0.060

# 修改后: 信任 Part 2 声学证据 (start_s)，回拉 end_s
for p in adj_punct:
    if p["end_s"] <= p["start_s"]:
        p["start_s"] = round(p["end_s"] - 0.030, 3)
        if p["start_s"] < 0:
            p["start_s"] = 0.0
            p["end_s"] = 0.030
```

### 验证方法

```python
# 修复后，实词边界扩展应停在 NVV 边界 - 0.02s
# adj_punct 中不应再有 end_s <= start_s
# 完整管线运行后 overlapping_words 计数应下降

# 检查代码:
all(p["end_s"] > p["start_s"] for p in adj_punct)  # 应全为 True
```

---

## Case 55: 全静音 phone list → MFA split 用静音边界切分声母/韵母 → 垃圾时间 (all_silence_mfa_split)

### 现象

修复前: 泄漏过滤器（`build_pinyin_phones_tier` 行 516-528）清空词的所有实音素后，`word_phones` 只剩静音/spn 条目。当 `len(word_phones) >= 2` 时，MFA-precise 分支（行 665）用静音时间边界做声母/韵母切分，产出垃圾时间。

典型: `chang4` → `ch[0-5ms] + ang4[5-360ms]`（5ms 的 "ch" 是静音片段残余，非真实声母）。

修复后: 检测到 `word_phones` 全是静音/spn 时，跳过 MFA split，回退到比例切分（行 680 的 `if not use_mfa_split:` 分支），产出合理的声母/韵母分割。

### 根因链

1. **泄漏过滤器** (~line 516): MFA 将 NVV/英语词对齐全为静音/spn → 第一个实音素起始 > 词起始 30% → 裁剪全部实音素 → `word_phones` 只剩静音。
2. **MFA-precise 分支无守卫** (~line 665): `len(word_phones) >= 2` 未检查 phone 是否为实音素 → 用静音边界做切分。
3. **检测 (Case 26-E) 只能标不能防**: 行 5845 的 `silence_boundary_split` 检测在 QC 段运行——只能标记不能阻止垃圾时间产生。

### 修改点

**`postprocess_textgrids.py` 行 665 — `build_pinyin_phones_tier`**

```python
# 修改前: 未检查 phone 是否为实音素
if len(word_phones) >= 2:

# 修改后: 添加实音素守卫
_real_phones = [(s, e, t) for s, e, t in word_phones
                if not is_silence(t) and t != "spn"]
if len(word_phones) >= 2 and _real_phones:
```

### 验证方法

```python
# 找触发 silence_boundary_split 的文件，debug-print word_phones:
#   修复前: [(0, 0.005, '<sp0>'), (0.005, 0.360, '<sp0>')]  ← 全静音
#   修复后: word_phones 全静音 → 走比例切分 → silence_boundary_split 不触发

# 确认:
[len([p for p in word_phones if not is_silence(p[2])]) for w in flagged_words]
# 修复前: [0, 0, ...] (全 0)
# 修复后: 比例切分替代 MFA split → split 用 _INIT_FRAC 而非静音边界
```

---

## Case 56: Phase 5 首音素回拉不对称 → 前词末 phone 未延伸 → pp 轨道间隙 (first_phone_snap_asymmetry)

### 现象

修复前: Phase 5 首音素回拉 (`new_pp_ivs[first].xmin = w_iv.xmin`) 是单向操作——将当前词首音素左拉到词起始，但前一词的末音素不被右推来填补空隙。对比末音素扩展逻辑（行 5068-5101）主动搜索下一词首音素并夹制，首音素回拉完全不对称。

结果: `pp_tier_gaps`（pinyin_phones 轨道间隙 > 10ms）在词边界处频发。

修复后: 首音素回拉执行后，同步搜索前一词的末音素并向前延伸以闭合空隙（与末音素扩展对称，含相同时长上限）。

### 根因链

1. **首音素回拉只拉一侧** (~line 5072): 只有 `new_pp_ivs[first].xmin = w_iv.xmin`，无对应 `new_pp_ivs[prev_last].xmax = w_iv.xmin` 操作。
2. **末音素扩展有完整对称逻辑** (~line 5068-5101): 搜索下一词、找首音素、夹制 extend_to、含时长上限——为对称修复提供了精确模板。
3. **微隙吸收阈值盲区** (~line 5121-5154): Phase 5 末端的微隙吸收仅处理 ≤ 10ms 的 gap，首音素回拉产生的 gap 可超过 10ms。

### 修改点

**`postprocess_textgrids.py` 行 5073 之后 — Phase 5 pp rebuild loop**

首音素回拉后添加对称延伸逻辑（~45 行），与前一词末音素对接，相同时长上限（元音 400ms / 辅音 200ms，1.5x 原始时长）。

### 验证方法

```python
# 找 pp_tier_gaps > 0 的文件
# 检查空隙是否发生在词边界 (即前一词末 ≠ 当前词首)
# 修复后: pp_tier_gaps 计数应下降

# 确认修复未引入重叠:
all(new_pp_ivs[i].xmax <= new_pp_ivs[i+1].xmin + 0.002 for i in range(len(new_pp_ivs)-1))
```

---

## Case 57: MFA/CTC 混合决策独立 → 词间空隙 > 20ms → words 轨道间隙 (mixed_decision_word_gap)

### 现象

修复前: `_snap_to_ctc` 中每个词独立决定信任 MFA 边界还是快照到 CTC 边界。当词 N 信任 MFA（`xmax = mfa_end`）而词 N+1 快照到 CTC（`xmin = ctc_start`），且 `ctc_start > mfa_end + 0.020`，空隙 > 20ms 打开。

现有的缓解措施（逐词间隙吸收、静音插入、微隙吸收）都有阈值盲区——当空隙超过各阈值时，空隙保留到最终输出。

修复后: 在主循环结束后、Leading silence 处理前，添加后循环连续性遍历——检测相邻实词间 > 20ms 的空隙，吸收进较长的词。

### 根因链

1. **逐词独立决策** (~line 3637-3784): 每个词单独在 MFA/CTC 之间选择，无跨词一致性约束。
2. **逐词间隙吸收覆盖面不足** (~line 3932-3943): 仅处理 `use_mfa=False` + `gap <= 0.2s` + `extended_dur <= prev_ctc_dur * 2.0` 的特定场景。
3. **无声插入充填空隙** (~line 3949-3957): 用 `<spN>` 填充而非消除空隙 → 空隙仍在，只是被标记了。

### 修改点

**`postprocess_textgrids.py` 行 4008 之后 — `_snap_to_ctc` 主循环结束**

```python
# 后循环连续性遍历:
_WT_GAP_LIMIT = 0.020  # 与 QC 检测阈值一致
for _gi in range(len(new_word_ivs) - 1):
    cur, nxt = new_word_ivs[_gi], new_word_ivs[_gi + 1]
    if cur[3] != "word" or nxt[3] != "word":
        continue
    _gap = nxt[0] - cur[1]
    if _gap > _WT_GAP_LIMIT:
        # 吸收进较长词
        if cur[1] - cur[0] >= nxt[1] - nxt[0]:
            new_word_ivs[_gi] = (cur[0], nxt[0], cur[2], cur[3])
        else:
            new_word_ivs[_gi + 1] = (cur[1], nxt[1], nxt[2], nxt[3])
```

### 验证方法

```python
# 找 words_tier_gaps > 0 的文件
# 检查哪对邻接词产生空隙: gap = next.xmin - cur.xmax
# 修复后: 后循环遍历闭合空隙 → words_tier_gaps 计数下降

# 确认: 空隙 > 20ms 的邻接词对在修复后应连续
```

---

## Case 58: handle_unexpected_silences 标记 <sp1-3> 后不合并 → mid_sp 误报 (sp1_3_flag_not_merge)

### 现象

修复前: `handle_unexpected_silences` (行 805-813) 对两个普通实词之间的 `<sp1-3>`（无标点匹配）执行的操作是：
1. `filter_reasons.append("unexpected_silence")` — 标记为异常
2. `continue` — 跳过合并块

后续吸收遍（`absorb_nvv_trailing`, `absorb_silence_into_punct`）只覆盖 NVV 邻接和标点邻接模式。两个普通实词之间、无标点的 `<sp1-3>` 静音全部漏过，最终触发 `mid_sp` 过滤。

修复后: 标记仍保留（提供诊断信息），但代码不再 `continue`——静音被合并进前一词。`unexpected_silence` 过滤标记保留，`mid_sp` 不再误报。

### 根因链

1. **标记但不修复** (行 810-813): `filter_reasons.append` + `continue` — 异常被记录但问题不被解决。
2. **四道吸收工序覆盖盲区**: `<sp1-3>` 在两个普通实词之间、无标点、无 NVV → 四道全都跳过。
3. **mid_sp 是下游级联后果**: 静音仍在 tier 中 → `mid_sp` 检测触发 → 文件被双重标记。

### 修改点

**`postprocess_textgrids.py` 行 805-813 — `handle_unexpected_silences`**

```python
# 修改前: 标记后 continue → 跳过合并
elif sil_label in ("<sp1>", "<sp2>", "<sp3>"):
    if not (is_english_token(prev_text) or ...):
        filter_reasons.append("unexpected_silence")
    continue

# 修改后: 标记后落入合并块; English/NVV 邻接仍跳过
elif sil_label in ("<sp1>", "<sp2>", "<sp3>"):
    if not (is_english_token(prev_text) or ...):
        filter_reasons.append("unexpected_silence")
        # Fall through to merge block below
    else:
        continue
```

### 验证方法

```python
# 找同时有 mid_sp 和 unexpected_silence 的文件
# 检查词层中是否有: [实词] <sp1/2/3> [实词] (无标点)
# 修复后: mid_sp 不再触发，unexpected_silence 仍记录
```

---

## Case 59: CTC 替换部分成功 → 拼音片段被合并守卫排除 → hanzi 拼音残留 (partial_ctc_merge_guard)

### 现象

修复前: 当 MFA 把英语词拆成拼音近音片段（如 "roughly" → "ru4" + "fei1"），CTC 替换（行 4464-4468）只替换了 "ru4"（有重叠）而未替换 "fei1"（无重叠）。合并逻辑（行 4528-4530）要求 `is_english_token(iv.text)` 同时为真，"fei1" 是拼音音节→`is_english_token` 返回 False → 合并被阻止。

未合并的 "fei1" 进入 `_build_hanzi_tier` 消费一个 CJK 字符 → 游标漂移 → 后续拼音 token 回退到原始拼音 → `hanzi_pinyin` + `cjk_mismatch` 双触发。

修复后: 合并条件扩展为同时接受 `is_english_token` 和 `is_pinyin_syllable`，允许未替换的拼音片段在与前一个已替换英语 token 重叠同一 CTC token 时被合并。

### 根因链

1. **CTC 替换只按时间重叠匹配** (~line 4473-4478): `best_overlap > 0` 阈值——MFA 片段时间与 CTC 锚点无重叠时不被替换。
2. **合并守卫过于严格** (~line 4530): `is_english_token(iv.text)` 排除所有拼音音节——合理防止合并独立中文词，但也排除了未替换的英语碎片。
3. **合并内部有 CTC 重叠验证** (~line 4487-4491): 已确认两个区间重叠同一 CTC token——此检查天然排除独立中文词，合并守卫的严格限制是多余的。

### 修改点

**`postprocess_textgrids.py` 行 4530 — 合并守卫条件**

```python
# 修改前: 仅接受英语 token
and is_english_token(iv.text.strip())):

# 修改后: 同时接受英语 token 和拼音音节（未替换的英语碎片）
and (is_english_token(iv.text.strip())
     or is_pinyin_syllable(iv.text.strip()))):
```

合并结果文本来自 `prev.text`（已替换的英语 token），不受影响。CTC 重叠检查验证同一 token，防止误合并独立中文词。

### 验证方法

```python
# 找同时有 hanzi_pinyin 且有英语词的文件的
# 检查英语词附近词层: 应有 pinyin 碎片 (如 "fei1") 紧接在英语词后
# 修复后: 碎片被合并进英语词 → pinyin_count 下降 → hanzi_pinyin 不触发
```

---

## Case 60: fix_short_words 仅覆盖虚词 → 实词 < 50ms 不被修复 → short_word 过滤 (content_short_word_unfixed)

### 现象

修复前: `fix_short_words` (行 2203-2254) 只在以下全部条件满足时激活：
- 词是 `CHINESE_SHORT_WORDS` 集合中的虚词（的、了、着、呢…）
- 后跟 ≥ 0.4s 静音
- 词时长 < `fix_short_word_sec`（默认 0.25s）

被挤压在两个非短词之间的实词（如一个二字词的孤立碎片 < 50ms）不被修复，最终被 `short_word` 过滤器捕获。

修复后: 添加实词候选收集（< 50ms、在两个非短词之间），通过双向能量搜索尝试扩展实词边界（有能量证据时延伸，无证据时仍被 filter 捕获）。

### 修改点

**A. `postprocess_textgrids.py` 行 2223 之后 — 实词候选收集**

```python
content_candidates = []
for idx, iv in enumerate(words.intervals[1:-1], start=1):
    if (not is_silence(iv.text) and not is_punct(iv.text)
            and not is_nvv_token(iv.text)
            and iv.duration < 0.050
            and iv.text.strip().lower().rstrip('12345')
            not in {w.rstrip('12345') for w in CHINESE_SHORT_WORDS}):
        prev_iv = words.intervals[idx - 1]
        next_iv = words.intervals[idx + 1]
        if (not is_silence(prev_iv.text) and not is_silence(next_iv.text)
                and prev_iv.duration >= 0.050 and next_iv.duration >= 0.050):
            content_candidates.append(idx)
```

**B. `postprocess_textgrids.py` 行 2269 之前 — 实词双向能量搜索**

```python
for word_idx in content_candidates:
    # 向右搜索: 短词+后邻接词起始区域是否有连续语音能量
    region = find_speech_in_silence(
        audio, sr, word_iv.xmin, min(next_iv.xmin + 0.10, next_iv.xmax),
        search_sec=0.15, frame_ms=args.fix_frame_ms,
        hop_ms=args.fix_hop_ms, thresh_ratio=args.fix_threshold_ratio,
        min_region_sec=0.015)
    if region and sp_end > word_iv.xmax:
        word_iv.xmax = sp_end    # 扩展短词
        next_iv.xmin = sp_end    # 缩小邻接词
```

### 验证方法

```python
# 找 short_word 过滤中有非 CHINESE_SHORT_WORDS 的实词
# 检查该词是否在两个较长的非静音词之间
#   是 → 修复后若能量支持则被扩展 (content_short_word_fix)
#   否 → 是真正的 MFA artifact，仍被 short_word catch
```

---

## Case 61: 参考文本模式英文词被 NVASR tokenizer 拆碎 → 最终 TextGrid 英文变形 (ref_text_english_tokenizer_mangle)

### 现象

参考文本模式 (nvv_enabled=false)，原始 txt 含英文专有名词（Claude, kimi, MacBook, PV, K-Pop），最终 TextGrid raw_text 中英文严重变形：

| 原始 txt | lab (拼音) | text_cn | 最终 raw_text |
|----------|-----------|---------|-------------|
| Claude | Cla ude | Cud | Cudude / RIA |
| PV | PV | PV | 保留但错位 |
| ria | ria | RIA | RIA 了爱 |

### 根因链

1. `ctc_prealign.py:284`: 参考文本正确赋值 `align_text = ref_texts[stem].strip()`
2. `replace_ria_variants()` / `normalize_punct_inline()` 处理（对中文有益，英文基本无损）
3. **关键**: NVASR `tokenizer.text2tokens(align_text)` — 中文 tokenizer 将英文词拆为字母碎片（Claude → Cla + ude），每个碎片作为独立 token
4. CTC forced alignment 将碎片映射到音频帧 → 碎片之间的时间边界被 CTC blank 帧或低置信帧填充
5. normalize_en 步骤试图合并碎片但无完整上下文 → 输出 Cudude / RIA 等
6. postprocess `enable_text_correction: true` 只修正标点/静音，不修复被 tokenizer 损坏的词

### 影响

- 受影响的 token 类型: 英文专有名词、品牌名、缩写（Claude, kimi, MacBook, PV, BGM, K-Pop 等）
- 中文部分基本不受影响
- **已修复**: Case 62 — `_normalize_word_spellings` 三通道修复，使用原始 .txt 参考文本覆盖 tokenizer 损坏的英文词

### 修复

见 Case 62: `_normalize_word_spellings` 英文词参考文本覆盖修复。

### 验证方法

```python
# 在 ria 数据集中搜索含英文词的段，对比原始 txt 和最终 raw_text
stem = "036001_弹幕互动_回应吐槽弹幕"
# 原始: "Claude的推送"
# 预期 raw_text 含 "Claude"
# 实际: "Cudude的推送" / "RIA"
```

---

## Case 62: `_normalize_word_spellings` 用原始 .txt 参考文本覆盖 tokenizer 损坏的英文词 (english_reference_overwrite)

**日期**: 2026-08-05
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_normalize_word_spellings`
**触发场景**: 参考文本模式 (nvv_enabled=false)，原始 txt 含英文专有名词（Claude, kimi, MacBook 等），
NVASR tokenizer 拆碎后 `normalize_english_tokens.py` 未能合并/错误合并

### 现象

Case 61 记录了根因：NVASR 中文 tokenizer 将英文词拆为字母碎片，`normalize_english_tokens.py`
使用 `_text_cn.txt`（ASR 输出）作参考而非原始 `.txt` 参考文本，导致碎片未合并或错误合并。
最终 words/hanzi tier 中英文词严重变形。

```
原始 txt:       "Claude的推送"
tokenizer 输出: "Cla" + "ude" + "的" + "推" + "送"
normalize_en:  _text_cn.txt 可能含错误 ASR 文本 → 合并失败或输出 "Cudude"
最终 words:    "Cudude" / "RIA" / "Cla" + "ude" (碎片残留)
```

### 根因链

1. **`ctc_prealign.py:312`**: NVASR `tokenizer.text2tokens(align_text)` — 中文 tokenizer 将英文词拆为字母碎片（Claude → Cla + ude），每个碎片作为独立 token 参与 CTC 强制对齐
2. **`normalize_english_tokens.py:285`**: 使用 `_text_cn.txt`（ASR decode 输出）作为参考文本，而非原始 `.txt` 参考文本。ASR 对英文专有名词可能转录错误 → 碎片合并失败或产生错误合并
3. **`_normalize_word_spellings` (旧)**: 设计为仅修复 "ria" 一种场景，注释写明 "Only replace for 'ria'". 其他英文词（live, BGM, Claude 等）保留 tokenizer 碎片
4. **`raw_text` 已正确加载但未充分利用**: postprocess 中 `raw_text = find_original_text(stem, args.raw_text_dir)` 返回正确的原始 `.txt` 参考文本，但 `_normalize_word_spellings` 仅用它做 NW 对齐，替换范围被限制在 "ria"

**两阶段失效**:
- **Stage 1 (normalize_en)**: 参考文本来源错误（ASR 而非 original txt）→ 碎片合并可能失败
- **Stage 2 (postprocess)**: 正确的参考文本已在手但未充分利用 → 英文碎片残留

### 修改点

**A. `_normalize_word_spellings` — 三通道英文词修复** (~line 1766)

完全重写函数，从"仅 ria"扩展为三个通道：

**Pass 1 — 替换已匹配的英文词**:
- 对所有 ASCII-alpha 参考词（len >= 2，排除 NVV）
- 当 word-tier token 文本与参考文本不一致时，用参考文本覆盖
- 跟踪 `fixed_ctc_indices` 记录被修正的 token 位置
- NVV token 永不修改（Case 17 保护保留）

**Pass 2 — 合并孤儿 ASCII-alpha 碎片**:
- 扫描 NW 对齐中未匹配（ref_i=None）的 CTC token
- 仅处理 ASCII-alpha 碎片（无数字、无 CJK、无 NVV）——pinyin 音节如 `rui4` 被保护
- 向左/右搜索最近的已修正英文词，扩展其时域覆盖此碎片
- 碎片 interval 置为零长占位（末尾清理）
- 跳过中间的 CJK/pinyin/NVV token（安全边界）

**Pass 3 — 未匹配参考英文词 → 替换孤儿 CTC token**:
- 处理参考英文词完全未被 NW 匹配的情况（如错误合并 "Cudude" 不匹配任何 ref）
- 用相邻已匹配的 CJK 锚点限定搜索区域
- 在区域内找到第一个 ASCII-alpha 孤儿 token → 替换为参考英文词
- 同一区域内的相邻孤儿碎片合并入此词

**清理**: 删除所有零时长占位 interval，保证 words tier 连续。

修改前（旧逻辑）:
```python
# Pass 1: Only replace for "ria" — other English words keep fragments
for ctc_i, ref_i in alignment:
    if ctc_i is None or ref_i is None:
        continue
    ref_spelling = ref_units[ref_i][1]
    if not (ref_spelling.isascii() and ref_spelling.isalpha() and len(ref_spelling) >= 2):
        continue
    wi, w_text = word_entries[ctc_i]
    if is_nvv_token(w_text):
        continue
    if ref_spelling != w_text and ref_spelling.isascii():
        words_tier.intervals[wi].text = ref_spelling
# No gap merging (Pass 2 removed)
```

修改后（新逻辑）:
```python
# Pre-scan: detect ALL English reference words (not just "ria")
en_ref_positions: dict[int, str] = {}
for ri, (ci, u) in enumerate(ref_units):
    if u.isascii() and u.isalpha() and len(u) >= 2 and not is_nvv_token(u):
        en_ref_positions[ri] = u

# Pass 1: Replace ALL mismatched English words
for ctc_i, ref_i in alignment:
    if ctc_i is None or ref_i is None: continue
    if ref_i not in en_ref_positions: continue
    ref_spelling = en_ref_positions[ref_i]
    wi, w_text = word_entries[ctc_i]
    if is_nvv_token(w_text): continue
    if ref_spelling != w_text:
        words_tier.intervals[wi].text = ref_spelling
        fixed_ctc_indices.add(ctc_i)

# Pass 2: Merge orphan ASCII-alpha fragments into fixed English words
for ctc_i, ref_i in alignment:
    if ref_i is not None or ctc_i is None: continue
    wi, w_text = word_entries[ctc_i]
    if not (w_text.isascii() and w_text.isalpha()): continue
    if is_nvv_token(w_text): continue
    # Search left/right for nearest fixed English word, merge into it

# Pass 3: Unmatched reference English words → replace orphan CTC tokens
for ref_i, en_word in en_ref_positions.items():
    if ref_i in ref_to_ctc: continue  # already matched
    # Use neighbouring CJK anchors to bound search region
    # Replace first orphan ASCII-alpha token in region with en_word
    # Merge additional orphans into it

# Cleanup: remove zero-duration placeholders
words_tier.intervals = [iv for iv in words_tier.intervals
                       if iv.xmax - iv.xmin > 0.001]
```

**B. 调用点注释更新** (~line 5226)

```python
# 修改前:
# 1. Normalise English token fragments ("R"->"ria") so words &
#    pinyin_phones tiers use the canonical reference spelling.

# 修改后:
# 1. Normalise English words against original reference text (.txt).
#    NVASR tokenizer (Chinese-centric) breaks English words into letter
#    fragments (e.g. "Claude"→"Cla"+"ude") which may survive
#    normalize_english_tokens.py when _text_cn.txt (ASR) differs from
#    the reference.  raw_text from the original .txt is ground truth.
#    Regression Case 62.
```

### 三个通道覆盖的场景

| 场景 | 示例 | 处理通道 |
|------|------|---------|
| 英文词被拆碎，首碎片匹配参考，其余残留 | "Cla"+"ude" → NW: "Cla"↔"Claude", "ude"↔None | Pass 1 替换 "Cla"→"Claude", Pass 2 合并 "ude" |
| 英文词被错误合并为一个整体 | "Cudude" → NW: 不匹配 "Claude" | Pass 3 在限定区域内替换 "Cudude"→"Claude" |
| 多个英文词相邻且都被拆碎 | "MacBook Pro" → 三个碎片组 | 各通道独立处理每个参考英文词 |
| 英文词已被正确修复 | "Claude" = "Claude" | 无操作（`ref_spelling == w_text` 跳过） |
| 无英文词（纯中文） | "你好世界" | `en_ref_positions` 为空 → 立即返回 |
| NVV token 被误认为英文 | `<LAUGHTER>` | `is_nvv_token` 保护 → 跳过 |

### 安全保护

- **pinyin 音节保护**: Pass 2/3 仅处理 `w_text.isascii() and w_text.isalpha()` 的 token。`rui4`（含数字）不被合并
- **CJK 锚点边界**: Pass 3 用相邻已匹配的 CJK ref unit 限定搜索范围，不会越界替换
- **NVV 保护**: 三层检查 — Pass 1 的 `is_nvv_token(w_text)`、Pass 2 的相同检查、Pass 3 的 `is_nvv_token()` 排除
- **空操作 fallback**: 无英文参考词时 `en_ref_positions` 为空 → 立即返回，不影响纯中文数据

### 与 normalize_english_tokens.py 的关系

| 特性 | normalize_english_tokens.py | _normalize_word_spellings (修复后) |
|------|---------------------------|----------------------------------|
| 运行时机 | pre-MFA (lab 级别) | post-MFA (words tier 级别) |
| 参考文本来源 | `_text_cn.txt` (ASR) | `raw_text` (原始 .txt) |
| 修改对象 | `.lab` + `_tokens.jsonl` + `.TextGrid` | words tier intervals (in-place) |
| 覆盖范围 | CJK pinyin→英文还原 | 所有 ASCII-alpha 英文词 |
| 失败模式 | ASR 错误 → 碎片未合并/错误合并 | 作为安全网捕获 normalize_en 的漏网之鱼 |

两者互补: normalize_en 在 MFA 之前修复 `.lab`（最佳时机），postprocess 用原始参考文本兜底修复 MFA 之后的残留错误。

### 验证方法

```python
# 在 ria 数据集中搜索含英文词的段，对比原始 txt 和最终 words tier
stem = "036001_弹幕互动_回应吐槽弹幕"
raw_text = open(f"{stem}.txt").read()  # "Claude的推送"

# 提取 raw_text 中的英文词
import re
en_words = re.findall(r'[A-Za-z]{2,}', raw_text)  # ["Claude"]

# 在最终 words tier 中验证每个英文词都存在
for en in en_words:
    found = any(iv.text.strip() == en for iv in words_tier.intervals)
    assert found, f"English word '{en}' from reference not found in words tier"

# 反之：words tier 中不应有tokenizer碎片（单字母/短碎片在已修复英文词旁）
for i, iv in enumerate(words_tier.intervals):
    if iv.text.strip().isascii() and iv.text.strip().isalpha():
        # 英文 token 应至少 2 个字符且匹配参考文本中的某个词
        assert len(iv.text.strip()) >= 2, f"Single-letter fragment: {iv.text}"
```

### 关联样本

- Case 61: 根因分析 — 参考文本模式英文词被 tokenizer 拆碎
- Case 17: NVV 文本保护（`is_nvv_token` guard 保留）
- Case 31: normalize_english_tokens.py 英文 CTC 锚点三重修复（互补关系）

---

## Case 63: CTC 锚点错位导致参考文本与 hanzi tier CJK 字符序列不匹配 → text_order_mismatch 过滤 (ctc_anchor_text_order)

### 现象

CTC 推理的非确定性导致锚点漂移，hanzi tier 中 CJK 字符序列与原始参考文本不同。
例如：参考 "来了来了，ria上线！用kimi搜了一下" → hanzi "来了来了，RIA瑞！亚kimi上线用，main搜..."

### 根因链

1. NVASR CTC 批量推理每次产生不同的 logits → CTC 强制对齐的锚点位置不同
2. 锚点漂移导致 token 时间戳错位 → pinyin/word 与 hanzi 错配
3. 现有 pinyin_displacement 检测拼音级错配，但不直接检测**字符序列顺序**

### 修改点

`scripts/postprocess_textgrids.py` — 在 pinyin_displacement 检查之后新增 text_order_mismatch

```python
# 提取参考文本 CJK 字符序列
_ref_cjk = [c for c in re.sub(r'<sp\d+>', '', raw_text) if CJK(c)]
# 提取 hanzi tier CJK 字符序列
_hanzi_cjk = [c for h_iv in hanzi_tier_final if CJK(c)]
# LCS 比率 < 0.6 → text_order_mismatch
```

### 验证方法

```python
# 在 ria 数据集上重跑 postprocess
python3 scripts/run_pipeline.py --config configs/hecheng_ria_0805.yaml \
    --python .../mfa-dev/bin/python --step postprocess --overwrite
# 预期: 新增 text_order_mismatch 过滤类别，捕获 CTC 锚点错位文件
```
```

---

## Case 64: `english_phone_deficit` 读取错误的 `Interval.mark` → 检测永久失效（已修复） (english_deficit_mark_regression)

**日期**: 2026-08-06
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one`（English phone QC）
**关联历史问题**: Case 33、Case 49、Case 50

### 现象

Case 50 修复了 English QC 访问 `_p.mark` 导致的 `AttributeError`，但当前 Case 33 的音素不足检查仍然使用 `mark`：

```python
if (_p.xmax > _ws + 0.001 and getattr(_p, 'mark', None)
        and not is_silence(getattr(_p, 'mark', '').strip())):
    _w_phones.append(getattr(_p, 'mark', _p.text if hasattr(_p, 'text') else '').strip())
```

项目自定义 `Interval` 只有 `xmin`、`xmax`、`text`，没有 `mark` 属性。因此 `getattr(_p, 'mark', None)` 永远是 `None`，`_w_phones` 永远收集不到音素，`english_phone_deficit` 永远不会触发。

### 根因与影响

1. Case 50 只修复了直接崩溃路径；本段后来改成 `getattr` 后避免了异常，却保留了错误的属性名。
2. `english_phone_deficit` 的期望音素数已正确改为 CMUdict（Case 49），但实际音素计数始终为 0，导致“实际 2 个音素、期望 4 个音素”等真正缺失场景不被过滤。
3. Case 32 的 `english_single_phone` 使用 `.text`，所以该问题只影响 Case 33，容易被“没有崩溃”误判为已修复。

### 证据

- `Interval` 定义：`scripts/postprocess_textgrids.py:48-57`，字段为 `xmin/xmax/text`。
- 错误读取：`scripts/postprocess_textgrids.py:6401-6403`。
- 过滤条件：`scripts/postprocess_textgrids.py:6405-6417`。
- 编译检查通过，但编译无法发现 `getattr` 导致的逻辑死路。

### 修复要求

已将 `mark` 读取统一改为 `text`。English 词的 CMUdict 期望音素数与实际 `text` 音素数现在可以正常比较；实际少于期望时会加入 `english_phone_deficit`。

---

## Case 65: 单文件后处理异常只写 report 不返回非零 → 管线误报成功（已修复） (postprocess_error_exit_masking)

**日期**: 2026-08-06
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/run_pipeline.py`
**涉及函数**: `main`, `step_postprocess`

### 现象

`postprocess_textgrids.py::main()` 在串行和并行路径都捕获每个文件的异常，只追加：

```python
{"stem": tgp.stem, "status": "error", "error": str(exc)}
```

随后无论 `reports` 中是否包含 error，都写入 `postprocess_report.jsonl` 并自然返回；脚本没有 `sys.exit(1)`。`run_pipeline.py::step_postprocess()` 只返回 `run_python(...)` 的进程码，因此会把“全部文件失败”或“部分文件失败”视为步骤成功。

### 根因与影响

1. 单文件异常被设计为可继续批处理，但没有把 error 数量传递到进程退出状态。
2. 主管线的步骤失败列表不会包含 `postprocess`，后续缓存/完成状态可能照常保存。
3. `--validate` 只检查 glob 是否存在；旧输出文件或少量输出也可能满足模式，不能替代失败码。
4. 结果是监控端看到 `DONE/Success`，实际数据可能只有 filtered 文件、错误记录或混合不完整结果。

### 证据

- 异常捕获：`scripts/postprocess_textgrids.py:6649-6657`、`6682-6690`、`6701-6707`。
- 无错误退出：`scripts/postprocess_textgrids.py:6709-6717` 只统计并打印 counts。
- 管线调用：`scripts/run_pipeline.py:1495-1573`，没有解析 `postprocess_report.jsonl` 的 error 状态。
- 主管线成功判定：`scripts/run_pipeline.py:2555-2560` 仅依赖子进程返回码。

### 修复要求

postprocess 写完 report 后会根据 `status == "error"` 返回非零；无 TextGrid 输入也返回失败。主管线因此会停止并报告 `postprocess` 失败。

---

## Case 66: 派生 tier 同步异常被静默吞掉 → 过期/不同步 tier 仍可能进入输出（已修复） (silent_derived_tier_sync_failure)

**日期**: 2026-08-06
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_sync_derived_tiers`

### 现象

`_sync_derived_tiers()` 在重建 `hanzi` 和 `pinyin_phones` 时分别使用宽泛的 `except Exception: pass`：

```python
try:
    ... _build_hanzi_tier(...)
except Exception:
    pass

try:
    ... build_pinyin_phones_tier(...)
except Exception:
    pass
```

如果输入字典、英文窗口、tier 结构或数据类型触发异常，函数会保留旧的派生 tier，并继续进入后续 Phase 5 和输出；调用方也不会得到同步失败标记。

### 根因与影响

1. 项目将 `words` 作为唯一权威 tier，但同步失败后旧 `hanzi/pinyin_phones` 仍可能与新 `words` 边界或文本不一致，违反 Case 17-F 的架构不变量。
2. 由于异常被吞掉，QC 只可能检查到“旧 tier 的合法性”，无法可靠发现本次同步失败；若旧 tier 恰好存在，甚至不会触发缺失检查。
3. 多处 Phase 4 调用同步函数，异常不会使文件进入 `error` 或 `filtered`，问题被静默放大。

### 证据

- `scripts/postprocess_textgrids.py:1169-1187`：hanzi 重建异常静默忽略。
- `scripts/postprocess_textgrids.py:1189-1201`：pinyin_phones 重建异常静默忽略。
- 调用点：`scripts/postprocess_textgrids.py:5027-5031`、`5051-5053`、`5118-5120`。
- 现有 QC 在输出前没有 `sync_failed` 或派生 tier 版本/边界一致性断言。

### 修复要求

同步失败现在抛出带上下文的 `RuntimeError`，由 `process_one` 的文件级异常处理捕获并记录为 `error`；不会再保留旧 tier 冒充同步成功。

---

## Case 67: text_order_mismatch 用 CTC 归一化文本做参考 → 检查失效 (text_order_wrong_ref_source)

### 现象

`text_order_mismatch` 报告所有文件 `in_order: true`，包括已知锚点错位的文件。例如 `036002`：
- 原始 txt: `来啦来啦，ria上线！用kimi搜了一下...`
- hanzi tier: `来,了,来,了,瑞,亚,上,线,用,搜,了,一,下,为,你,推...`
- 应检测到字符 `瑞/亚/为` 不在原文中 → 顺序错误
- 实际报告: `in_order: true`

### 根因链

1. `process_one:4641`: `raw_text = find_original_text(stem, args.raw_text_dir)`
2. `args.raw_text_dir` = `ctc_pretg`（管线传的 `--raw-text-dir`）
3. `ctc_pretg` 目录无 `{stem}.txt` → `find_original_text` 返回空
4. 回退: `raw_text = {stem}_text_cn.txt` — **CTC 归一化后的文本**
5. 归一化文本已被 CTC 修改（`来啦→来了`、`ria/main/kimi` 被删除）
6. 归一化文本与 hanzi tier 自然匹配 → `in_order: true` → 检查完全失效

### 修改点

**A. `run_pipeline.py:1514`** — 新增 `--original-txt-dir` 指向原始 `data_dir`

```python
# 新增行
"--original-txt-dir", str(ctx["data_dir"]),
```

**B. `postprocess_textgrids.py:6124-6128`** — text_order 检查改用原始 txt

```python
# 修改前: 用 raw_text (可能是 CTC 归一化文本)
_ref_cjk = [c for c in re.sub(r'<sp\d+>', '', raw_text) if CJK(c)]

# 修改后: 先尝试从 original_txt_dir 加载原始 txt
_orig_txt = raw_text
if getattr(args, 'original_txt_dir', None):
    _orig_path = Path(args.original_txt_dir) / f"{stem}.txt"
    if _orig_path.exists():
        _orig_txt = _orig_path.read_text(encoding="utf-8").strip()
_ref_cjk = [c for c in re.sub(r'<sp\d+>', '', _orig_txt) if CJK(c)]
```

### 验证方法

```python
# 重跑 postprocess 后检查 036002
# 预期: status 含 "text_order_mismatch", text_order.in_order = false
```



## Case 68: 参考文本未贯穿 CTC→MFA→后处理链，ASR 文本覆盖权威文本并造成严重错位（已修复） (reference_text_ctc_anchor_authority)

**日期**: 2026-08-06
**涉及文件**: scripts/ctc_prealign.py, scripts/normalize_english_tokens.py, scripts/postprocess_textgrids.py, scripts/run_pipeline.py

### 场景与现象

参考文本模式的设计是“音频和对应文本已知，ASR 只提供 CTC 帧级时间锚点、停顿和中英文/NVV 分类”。实际代码却在后续阶段混入 ASR 文本，导致：

- CTC 强制对齐使用了正确的参考文本，但 postprocess 因找不到原始 .txt 而回退到 _text_cn.txt；
- ctc_ready 复制的参考文件名是 {stem}_ref.txt，原查找逻辑只识别 {stem}.txt；
- normalize_english_tokens.py 原本无条件以 _text_cn.txt 作为英文参考，可能把正确的参考英文词改成 ASR 词；
- 后处理会依据 CTC token 的时间重叠改写 MFA words tier 的 pinyin/英文词面。在参考文本模式下，这相当于允许不可靠的 ASR/CTC 解码结果篡改权威文本；
- ctc_prealign.py 原来按 all_results 的位置绑定 stems，batch 返回顺序改变或部分结果缺失时，会把一个音频的锚点写到另一个音频；
- tokenizer 一个逻辑 token 展开为多个 CTC id 时，原代码仍按 speech_tokens[tid] 取标签，展开点之后的 token 时间标签会发生偏移。

### 根因链

参考文本
  ├─ CTC forced align → .lab / TextGrid（正确）
  ├─ ASR decode → _text_cn.txt（仅应为诊断信息）
  └─ 管线后处理 raw_text / normalize_en / CTC rewrite 错误读取或覆盖为 ASR
                                      ↓
                       MFA words / hanzi / English 词面错位

问题本质不是 CTC 不能做锚点，而是“文本内容”和“时间锚点”的职责边界没有在文件接口及后处理逻辑中固定下来。

### 修复内容

1. ctc_prealign.py 在存在参考文本时写出 {stem}_ref.txt；ASR 输出继续写入 _text_raw.txt / _text_cn.txt，明确作为诊断或无参考文本时的 fallback。
2. normalize_english_tokens.py 优先读取 {stem}_ref.txt，只有不存在时才回退到 _text_cn.txt，使 .lab、TextGrid 和英文合并使用同一文本源。
3. postprocess_textgrids.py::find_original_text() 支持 {stem}_ref.txt；发现参考文本后设置权威模式，禁止 CTC token 改写 MFA words tier 的词面或按 CTC 英文 token 合并区间。CTC 仍可参与时间边界、停顿和语言类别处理。
4. run_pipeline.py 为 full、ctc_ready、batch_ctc_ready 传递正确的 raw_text_dir；外部 text_dir、工作区中已链接的 {stem}_ref.txt 和原始数据目录均可恢复参考文本。text_order QC 也使用同一目录。
5. step_link_ctc() 额外保留 CTC 源目录中已有的可选 {stem}_ref.txt，兼容“预先生成 CTC 目录、未配置 text_dir”的用法。
6. CTC 结果写出改为按结果 key 解析音频 stem，拒绝无法匹配或重复的输入音频结果；token id 展平时同步建立 flat_token_labels，消除逻辑 token 与 CTC id 数量不一致造成的标签漂移。
7. CTC 强制对齐按返回的 target token id 在目标序列中定位，而不是按非 blank group 计数；检测到任意目标 token 零帧时拒绝生成该文件，并以非零退出码阻止管线继续。

### 设计不变量

- 参考文本存在时：参考文本是 .lab、MFA words、hanzi 和英文词面的唯一权威来源。
- ASR 文本：只用于诊断、无参考文本时的后备，以及从音频提取 CTC 分类信息；不得覆盖参考文本。
- CTC：只提供时间锚点、停顿和语言/NVV 类别；不得凭时间重叠替换参考词面。
- 每个 CTC 结果必须通过 result[key] 映射回同名输入音频；不能依赖 batch 位置。

### 验证

- 临时回归：目录同时存在 demo_ref.txt = “参考 life” 和错误的 _text_cn.txt = “参考 live” 时，英文归一化输出为 life。
- 临时回归：find_original_text("demo", ...) 正确返回 demo_ref.txt。
- python3 -W error -m compileall -q scripts check_ipa_mapping.py verify_risks.py 通过。
- git -c core.whitespace=cr-at-eol diff --check 通过。

### 关联问题

- Case 61/62：参考文本英文词被 tokenizer 拆碎及 postprocess 兜底修复。
- Case 63/67：CTC 锚点字符顺序检查及检查参考源错误。

---

## Case 69: 参考文本权威半修复残留：normalize_en Pass 2 自拼英文词、外层退出码和 adjusted CTC 丢失 _ref（已修复） (reference_authority_followthrough)

状态：已修复；参考文本权威链完成闭环。

**日期**: 2026-08-06
**涉及文件**: scripts/ctc_prealign.py, scripts/normalize_english_tokens.py, scripts/run_pipeline.py, scripts/adjust_ctc_boundaries.py, scripts/verify_reference_authority.py

### 场景与现象

Case 68 已经让 `normalize_english_tokens.py` 优先读取 `{stem}_ref.txt`，但仍存在半修复残留：

- `{stem}_ref.txt` 为 `life`，`{stem}_text_cn.txt` 为 ASR 诊断文本 `live`；
- `.lab` / `_tokens.jsonl` 中的 CTC 碎片为 `li ve`；
- Pass 1 因 `ve` 不是 `life` 的连续子串而不合并到参考词；
- Pass 2 `_reclaim_fragments()` 脱离参考文本，把 `li + ve` 自拼为 `live`；
- MFA 前的 `.lab` 和 `_tokens.jsonl` 因此仍被 ASR/CTC 碎片词面污染。

同时还有两个关联的完成度问题：

- `run_pipeline.py` 收集失败步骤后没有将 `failed` 转为非零退出码，上层 streaming/multi-GPU 调度可能误判成功；
- `adjust_ctc_boundaries.py` 输出 adjusted CTC 目录时只复制 `.lab` 和 `_text_cn.txt`，不复制 `_ref.txt`；直接用 adjusted 目录跑 postprocess 时仍可能退回 ASR 文本。
- 数字归一化以 `_text_cn.txt` 是否变化决定是否更新 `.lab`；当 ASR 与参考文本的数字形式不同，MFA transcript 可能不会按自身内容完成归一化。

### 根因链

```
_ref.txt = life
      ↓
normalize_en Pass 1 读取了 reference，但过度保守：li ve ↛ life
      ↓
Pass 2 _reclaim_fragments 不再看 reference：li + ve → live
      ↓
.lab / _tokens.jsonl 在 MFA 前被错误词面固化
```

### 修复内容

1. `normalize_english_tokens.py` 增加参考文本权威模式：当 `{stem}_ref.txt` 存在时，英文 spelling 必须来自参考文本。
2. Pass 1 使用 `_tokens_plausibly_realise_reference()`，允许 `live → life`、`li+ve → life` 这类近似碎片被参考词覆盖；无 `_ref.txt` 时保持旧的保守/legacy 行为。
3. 有 `_ref.txt` 时禁用 reference-agnostic 的 Pass 2 自拼逻辑，防止再次合成不在参考文本中的英文词。
4. `normalize_english_tokens.py` 的 worker 异常会累计并以非零退出码返回。
5. `normalize_english_tokens.py` 读取 `_ref/_text_cn/.lab/_tokens.jsonl` 时兼容 UTF-8 BOM。
6. `normalize_english_tokens.py` 并行分支只在系统支持时使用 `fork`，Windows 自动回退到平台默认启动方式。
7. `run_pipeline.py` 改为 `main() -> int`，失败步骤、`--force` 后累计失败和 output staging 同步失败都会返回非零退出码；`adjust_ctc_boundaries.py` 和 ctc_ready link 会保留可选的 `*_ref.txt`，不改变旧数据的必需文件校验。
8. `adjust_ctc_boundaries.py` 复制 `_text_raw.txt` 和 `_ref.txt`，让 adjusted CTC 目录保留参考文本权威文件。
9. `ctc_prealign.py` 和 `run_pipeline.py` 的数字归一化独立处理现有 `.lab`，不再由 ASR `_text_cn.txt` 的变化决定权威 transcript 是否更新。
10. 新增 `scripts/verify_reference_authority.py`，不依赖 MFA/NVASR，专门验证 `_ref.txt` 优先、pre-MFA normalize 不被 ASR 覆盖、legacy 无参考行为兼容、ctc_ready 复制可选参考副本、postprocess 兜底仍有效。

### 设计不变量

- 只要 `{stem}_ref.txt` 存在，pre-MFA 和 post-MFA 的英文词面都不得由 `_text_cn.txt` 或碎片拼接结果决定。
- `_text_cn.txt` 是诊断/fallback，不是参考文本模式下的权威。
- adjusted CTC 是 raw CTC 的时间修正版，不能丢失 raw CTC 中的参考文本权威文件。
- 任何子步骤失败最终都必须能通过进程退出码传给调用方。

### 验证方法

```bash
python scripts/verify_reference_authority.py
python -m compileall -q scripts/normalize_english_tokens.py scripts/run_pipeline.py scripts/adjust_ctc_boundaries.py scripts/verify_reference_authority.py
```

### 旧工作区缓存兼容

旧版本 `ctc_prealign.py` 写入的空 `.ctc_normalized` 不代表当前版本的归一化契约已经执行。现已将 marker 内容版本化为 `reference-authority-v2`：空 marker、旧版本 marker 或无法读取的 marker 都会被主管线视为过期并重新执行 normalize 链；只有当前版本 marker 才允许跳过。该修复不改变 Linux 的 GPU 映射、MFA `num_jobs`、NVMe/cache 路径或其他资源分配。

历史任务首次重跑建议使用 `--overwrite`，以重建旧的 CTC、adjusted CTC、MFA 和后处理产物；`--force` 只控制失败后的继续策略，不负责覆盖已有输出。

---

## Case 70: `tier_discontinuity` 把合法停顿当成轨道断层（已修复，待 Linux 全量复跑） (semantic_tier_discontinuity_gate)

**日期**: 2026-08-06
**涉及文件**: scripts/postprocess_textgrids.py, scripts/run_pipeline.py
**涉及函数**: _count_internal_pp_gaps, _collect_tier_discontinuities, _record_filterable_qc

### 现象

一次管线运行只有约 0.3% 通过，并产生 12,462 条 `tier_discontinuity`。

### 根因链

1. 旧逻辑对所有 tier 的相邻 interval 统一统计空隙。
2. `pinyin_phones` 是稀疏声学轨道，词间自然停顿可能没有 phone interval，不能按普通 words/hanzi 轨道处理。
3. 该类结构 QC 没有统一尊重 `filter_suspicious: false`，关闭质量过滤时仍可能把诊断升级为过滤原因。

### 修改点

- `words`、`hanzi` 继续检查系统性结构断层。
- `pinyin_phones` 只检查落在同一实词区间内部的空隙；跨词停顿不计入断层。
- QC 始终写入报告；只有 `filter_suspicious: true` 时才追加过滤原因。
- 新增 `scripts/verify_tier_discontinuity.py`，覆盖自然停顿、词内断层和关闭过滤三种场景。

### 验证方法

```bash
python scripts/verify_tier_discontinuity.py
# 预期: 三种场景均通过
```

- 本地单元回归通过。
- Python 编译检查通过。
- 尚未在 Linux/NVASR/MFA 全量数据上复跑，12,462 的真实下降幅度待确认。

状态：已修复，待 Linux 全量复跑。

---

## 未解决问题统一追踪（历史通用项）

本表保留跨批次的历史通用问题；hecheng_ria_0805 的当前状态以 Case 72–87 和后文”专项审计总记录”为准。

以下问题目前仍保留在档案中，尚未宣称解决。后续新增验证、修复和结论只更新本文件。

| ID | 当前状态 | 问题 | 尝试方案 |
|---|---|---|---|
| U1 | 待验证 | CTC/MFA 边界冲突仍依赖启发式逐词仲裁 | 建立跨词连续性约束，以音频能量谷/起振点作为第三仲裁信号 |
| U2 | 待验证 | `_refine_boundaries_by_energy` 与 CTC overlap 保护的交互缺少实测 | 增加阶段快照和边界不变量测试，验证能量修正不得重新制造 overlap/inversion |
| U3 | 待修复 | `_inject_punctuation` 对倒置/近零 interval 仍有静默丢弃路径 | 改为记录 warning，统一由边界修复器决定保留、裁剪或过滤 |
| U4 | 待设计 | Phase 间 tier 变更缺少统一 dirty/version 追踪 | words 变更后强制重建 hanzi/pinyin_phones，并记录派生 tier 版本 |
| U5 | 待验证 | `mid_sp` 中长停顿无标点、CTC 锚点错误两类根因缺少系统回归 | 分别建立长停顿样本和错误锚点样本，避免用单一阈值混合处理 |
| U6 | 待验证 | 纯英文/混合英文 CTC 模糊匹配的泛化边界 | 用参考词序列、CTC token 序列和时间重叠构造离线基准集 |

---

## Case 71: MFA HMM 软边界导致 pinyin_phones 层韵母→声母重叠 40-100ms (pp_phone_overlap_deoverlap)

状态：已修复。

---

### 现象

2,150 个内容文件中 88 个 (4.1%) 的 `pinyin_phones` 层存在相邻 phone 区间重叠：

```
uo3[6.030-6.240] ↔ x[6.170-6.270]    overlap=70ms
i1[5.130-5.340]  ↔ x[5.270-5.370]    overlap=70ms
an1[7.124-7.403] ↔ l[7.310-7.340]    overlap=93ms
```

全部是前音节末音素（韵母/final）侵入后音节首音素（声母/initial）。

### 根因链

1. MFA 的 HMM 对齐使用 soft boundary → 音素过渡区产生重叠
2. `_fix_overlapping_boundaries` 只处理 **words** 层（line 5109），不处理 **pinyin_phones** 层
3. 韵母→声母重叠未被检测/修复，进入最终输出

### 修改点

**A. `postprocess_textgrids.py`** — 新增 `_fix_pp_phone_overlaps(pp_tier)` 函数

```python
# 策略:
#  - 标点被内容 phone 覆盖 → 裁剪标点 side
#  - en: phone 被内容 phone 覆盖 → 裁剪 en: side
#  - 两个内容 phone 重叠 → 对半 split
mid = round((cur.xmax + nxt.xmin) / 2.0, 4)
intervals[i] = Interval(cur.xmin, mid, cur.text)
intervals[i+1] = Interval(mid, nxt.xmax, nxt.text)
```

**B. 调用点**: Phase 4 sync 之后、Phase 5 QC 之前（line 5263+）

```python
_pp = tier_by_name(new_tg, "pinyin_phones")
if _pp is not None:
    _pp_fixed = _fix_pp_phone_overlaps(_pp)
```

### 验证方法

```python
# 重跑 postprocess 后检查 pinyin_phones 层
# 预期: 0 个重叠 interval，report 含 pp_deoverlap_fixed
```

---

## Case 72: cn2an 污染拼音声调数字，18,000 个 lab 全量 OOV（修复草案已写入，暂停复审/复跑） (tone_digit_cn2an_corruption)

状态：修复草案已写入，暂停复审/复跑。

**日期**: 2026-08-06
**涉及文件**: scripts/pipeline_utils.py, scripts/ctc_prealign.py, scripts/run_pipeline.py, scripts/recover_ctc_labs.py
**涉及函数**: normalize_reference_numerals, step_normalize_text, _normalize_numerals, validate_ctc_transcript_bundle
**触发批次**: hecheng_ria_0805，18,000/18,000 个 lab

### 现象

- CTC tokens 和 CTC words tier 正确保存 rui4、shi4、juan3。
- normalize 阶段对完整 lab 调用 cn2an.transform(an2cn)，把词尾声调 1–5 当作普通数字：

      rui4 shi4 juan3 men5
      ↓
      rui四 shi四 juan三 men五

- 全部 18,000 个 lab 均出现污染；MFA 词典没有 rui四 一类词条，导致大规模 OOV。
- 17,861 个已生成 aligned TextGrid 中共有 630,473 个 MFA unknown；另有 139 条没有 aligned 输出。

### 根因链

1. 数字归一化没有区分“人类参考文本”和“已经 token 化的 MFA transcript”。
2. ctc_prealign.py::_normalize_numerals 与 run_pipeline.py::step_normalize_text 都可能处理完整 lab。
3. 原回归只模拟 123→一百二十三，反而把“lab 应变成中文数字”写成预期，没有覆盖 ma1..ma5。
4. 空或旧 v2 marker 也无法表达新的 transcript bundle 契约。

### 修复

1. v3 marker 集中到 pipeline_utils.py，旧 marker 自动过期。
2. cn2an 只用于 _text_cn.txt 等人类文本；保护拼音、NVV、括号标签和大写标识符。
3. lab 永不再做字符级数字转换。
4. 旧 lab 仅允许在 tokens 与 CTC words tier 完全一致后，从 tokens 的 word 序列原子重建。
5. 新增 recover_ctc_labs.py：默认 dry-run；只有显式 --apply 才修改隔离副本。
6. marker 写入前强制验证 lab == tokens == CTC words，失败返回非零。

### 保护不变量与验证

- ma1..ma5 多次归一化后逐字不变；NVV、标点、句首 sp1 不参加数字恢复判断。
- 真正的参考文本数字可在 tokenizer 前转换，但不能在 lab 中把 tone digit 变成汉字。
- 新增 tone 1–5、损坏 lab 从 tokens 恢复、缺 tokens 返回非零、旧 marker 失效回归。
- 生产目录 dry-run：18,000 bundles；17,999 可恢复；1 条因零时长 CTC 被严格拒绝（Case 80）。

---

## Case 73: MFA unknown 被误判为标点，CTC 回填 Rule 0 成为死代码（修复草案已写入，暂停复审/复跑） (unknown_token_punctuation_dead_branch)

状态：修复草案已写入，暂停复审/复跑。

**日期**: 2026-08-06
**涉及文件**: scripts/pipeline_utils.py, scripts/postprocess_textgrids.py
**涉及函数**: is_unknown_token, is_word_like, is_punct, _snap_to_ctc, process_one
**触发样本**: 036022_弹幕互动_回应吐槽弹幕

### 现象与根因

- MFA words tier 全是 unknown，phones tier 全是 spn。
- is_word_like 对尖括号 unknown 返回 False，随后 is_punct 返回 True。
- _snap_to_ctc 构造 mfa_words 时先排除“标点”，unknown 根本不会进入循环。
- 内部 Rule 0 虽写着恢复 unknown，却永远不可达；日志为 MFA=0、CTC=35。

### 修复

1. 新增 is_unknown_token，统一识别 MFA unknown 和 bracketed placeholder。
2. unknown 定义为 lexical unknown：word-like=True、punct=False、silence=False、pinyin=False、English=False、NVV=False。
3. _snap_to_ctc 能把 unknown 与对应 CTC token 对齐并恢复词面/边界。
4. process_one 在回填前保存 unknown_source_count。即使词面恢复，文件仍加入 mfa_unknown_source，不能伪装成声学对齐成功。

### 修复前后对比

- 修复前：只剩 sp1 和标点，报告为 0 pinyin vs 29 CJK。
- 修复后隔离样本：29 pinyin 对 29 CJK，汉字序列完全一致，raw_text、标点和 sp1 全保留；但原 MFA phone 仍是 spn，状态为 filtered_mfa_unknown_source，进程非零。
- 正确生产修复必须重新运行 MFA；回填词面不能替代音素对齐。

---

## hecheng_ria_0805 专项审计总记录（2026-08-06，修复暂停）

### 当前状态与范围

- 用户执行的是 run_pipeline.py 的单独 postprocess 步骤并带 overwrite。该操作只重建后处理产物，不会重建已经损坏的 CTC lab，也不会重跑 MFA。
- 本节记录的是对现有 0805 工作区、后处理报告及 NAS 结果的只读审计结论。
- Case 72 至 Case 80 已有部分代码草案，但尚未完成最终复审、完整回归、隔离 CTC 修复、MFA 重跑或全量验收，不能标记为生产已修复。
- Case 81 至 Case 87 是后续复审新增的未闭环项。
- 在用户要求暂停后，只允许继续编辑本档案；代码修复、测试、样本重跑、全量任务和 NAS 发布全部暂停。

### 术语澄清：0 pinyin tokens vs N reference CJK chars

该提示比较的是两个不同来源的语义计数：

- pinyin tokens：最终 words 或 pinyin 层中带 1 至 5 声调数字的中文拼音词，例如 ni3、hao3。
- reference CJK chars：权威参考文本中的汉字数量。

以下内容按设计不参加拼音计数，因此它们本身不是错误：

- NVV 标签，例如 <LAUGHTER>、<BREATHING>；
- 中文或英文标点；
- 句首 <sp1> 及其他静音标签；
- 英文词。

本批次的 0 vs N 不是由上述合法标签造成。它表示参考文本明明含 N 个汉字，但最终结果中一个有效声调拼音都没有。现有数据的直接原因是 lab 的声调数字被改成中文数字，MFA 将内容词大规模输出为 unknown 和 spn；后处理又把 unknown 误当标点排除，最终只剩句首 <sp1> 和标点。

用户指定的输出契约如下：

1. 句首必须有且只能有一个 <sp1>。
2. 参考文本中合法的 NVV 标签必须保留，大小写和尖括号格式规范化，但不得被删掉或计入拼音数量。
3. 标点必须保留原有顺序，不得为满足拼音计数而删除。
4. 仅汉字与带调拼音建立一一对应；NVV、标点、静音和英文分别走各自的结构校验。
5. 任何 reference CJK 大于 0 且 pinyin 为 0 的文件都属于硬完整性失败，不能进入 ok。

### 全量只读审计证据

| 检查对象 | 审计结果 | 结论 |
|---|---:|---|
| 16 kHz WAV | 18,000 | 音频全集 |
| 原始参考 txt | 17,999 | 缺少 1 条权威参考文本 |
| CTC bundle | 18,000 | 文件数量齐，但不代表内容有效 |
| 被 cn2an 污染的 lab | 18,000 / 18,000 | 全量 transcript 声调损坏 |
| 已生成 MFA aligned | 17,861 | 缺失 139 条 |
| aligned 中 MFA unknown | 630,473 | 语义对齐大面积失败 |
| 不含正常 tone 1–5 拼音的 aligned | 17,849 / 17,861 | 几乎全量不可用 |
| 仍含中文声调数字拼音的 aligned | 12 | 少量污染以另一形态残留 |
| postprocess report 行数 | 17,861 | 错误地以 aligned 子集为分母 |
| report 状态 | ok 14,200；filtered 3,661；error 0 | ok 数量不可信 |
| 0 pinyin vs 正数 CJK warning | 17,861 / 17,861 | 全部报告行均命中 |
| warning 累计缺失 CJK | 599,619 | 语义内容整体坍缩 |
| NAS TextGrid | 16,350 | 超过本次可信 ok 集合 |
| NAS 陈旧或非本次结果 | 2,150 | 旧文件混入当前目录 |
| workspace filtered | 3,667 | 比本次 filtered 多 6 条陈旧文件 |

补充结论：

- WAV 均为 16 kHz 单声道，WAV 与 TextGrid 时长最大差约 0.375 ms。物理时长正常只能证明容器和总时长可读，不能证明文本、拼音或音素语义正确。
- 样本 036022_弹幕互动_回应吐槽弹幕 的参考文本含中文、Claude、dancer 和 ria；旧 aligned words 几乎全为 unknown，phones 为 spn；旧最终 raw 只剩 <sp1> 与标点。
- 对旧 aligned 做隔离后处理验证时，CTC 可以回填出 29 个拼音并恢复汉字、标点、NVV 及句首 <sp1>，但原 MFA phone 仍是 spn，因此必须继续判为 mfa_unknown_source，不能将词面回填等同于声学对齐修复。
- 旧 CTC 的只读恢复预检结果为 18,000 个 bundle 中 17,999 个可从 tokens 恢复 lab，1 个严格校验失败，见 Case 80。

---

## Case 74: 0 pinyin vs N CJK 仅为 warning，结构坍缩仍进入 ok（已修复） (cjk_hard_integrity_exit_code)

状态：已修复：postprocess main() 在 hard_integrity 失败时返回非零退出码；`assess_reference_coverage()` 检测到 cjk_alignment_collapse 等结构性失败后，管线可正确识别。

### 现象

本批次 17,861 条报告全部出现 0 pinyin vs 正数 reference CJK，但其中 14,200 条状态仍为 ok。warning 因此没有阻止坏文件交付。

### 根因

1. 拼音计数差异只被追加到 warnings。
2. filter_suspicious=false 被错误理解为可关闭所有过滤；实际上该开关只能关闭启发式质量过滤，不能关闭结构完整性校验。
3. 后处理进程没有把此类严重 warning 汇总为非零退出码。

### 影响

- 语义内容已经完全消失的 TextGrid 仍被计为成功。
- ok 数量、通过率和 NAS 文件数失去可信度。
- 仅检查进程退出码或输出文件存在性无法发现问题。

### 修复方案

1. 将以下情况定义为 hard integrity failure：有 CJK 参考但无拼音、CJK 与拼音数量不等、CJK 字符序列不一致、参考文本为空、存在 MFA 源 unknown。
2. hard integrity 与 filter_suspicious 解耦，任何配置下都必须进入 filtered 或 error。
3. report 写入 hard_integrity_reasons；只要存在一条 hard failure，postprocess 返回非零。
4. 保留诊断 warning，但 warning 不再是唯一信号。

### 验收

- reference CJK 大于 0 且 pinyin 为 0 的样本绝不允许 status=ok。
- NVV、标点、英文和 <sp1> 不影响拼音计数。
- 汇总中 hard failure 大于 0 时，主管线和调用方都得到非零退出码。

---

## Case 75: 派生 raw_text 与派生 hanzi 空值相等，CJK 检查假通过（修复草案，未验证）

状态：修复草案已写入，暂停复审/复跑。

### 现象

当 MFA unknown 被排除后，派生 words、hanzi 和 raw_text 都可能只剩标点或为空。旧检查比较两个同源派生结果，空序列等于空序列，因此报告 CJK 一致。

### 根因

1. 参考端不是不可变的原始 txt 或 _ref.txt，而是后处理过程中重建的 raw_text。
2. 被测端和参考端都来自同一个已经损坏的 words tier，形成自证循环。
3. 空等于空被当作正常，而没有先检查原始参考是否包含汉字。

### 修复方案

1. process_one 开始时保存 immutable reference_text_original 和 reference_source。
2. 所有字符数、字符顺序和英文拼写校验均以该不可变参考为基准。
3. raw_text tier 仅作为交付展示，不得反向成为权威参考。
4. 添加 empty_reference、no_lexical_reference 和 cjk_alignment_collapse 独立原因。

### 验收

- 原始参考含“你好”而派生 hanzi 为空时，必须同时报告 collapse、数量不符和字符不符。
- 原始参考为空时必须单独隔离，不能以空等于空通过。

---

## Case 76: 分片 MFA 只看退出码、日志和 stem 完整性不足（部分草案，已实施 strict validator）

状态：代码已实施（2026-08-07）。`pipeline_utils.py` 新增 `validate_strict_mfa_textgrid`；`run_pipeline.py:_run_mfa_sharded` 用 strict validator 替换字符串匹配；单进程 MFA 路径增加 strict 验证；输出 manifest 增加 `invalid_detail` 结构化错误记录。

### 现象

CTC 有 18,000 条，aligned 只有 17,861 条，缺 139 条。旧分片执行可在部分 shard 缺文件时继续合并，且子进程输出被丢弃，无法从主日志追溯每个失败 stem。

### 根因

1. 成功条件主要依赖 shard 进程退出码，没有比较 expected stems 与 produced stems。
2. shard 工作目录可复用，旧结果可能掩盖本次缺失。
3. stdout 和 stderr 未形成每 shard 的持久日志。
4. TextGrid 合法性仅做浅层检查，不能证明 words 与 phones tier 可解析。
5. 子进程启动阶段的 OSError 等异常尚未完整转换为统一失败清单。

### 当前草案剩余风险

- 已有 run-specific shard 目录、日志和 stem manifest 草案。
- 仍需把字符串包含 tier 名称的检查替换为真正 TextGrid 解析。
- 单进程与分片路径必须使用同一验证器。
- Popen 启动异常、超时、非零退出和缺 stem 必须统一收敛并保留现场。

### 修复方案

1. MFA 启动前冻结 expected stem manifest。
2. 每 shard 使用全新目录和独立日志；失败目录不清理。
3. 逐个解析输出 TextGrid，要求 words 和 phones tier 存在且 interval 合法。
4. 合并后要求 produced stems 与 expected stems 精确相等，无缺失、无多余、无重复。
5. 任一 shard 失败、超时、启动异常或输出不完整都返回非零，禁止 postprocess。

### 验收

- 预期 18,000 条时，17,999 条输出必须失败。
- 人为删除一个 shard TextGrid 或破坏 phones tier，主管线必须在 align 阶段停止。
- 日志可由 run id、shard id 和 stem manifest 完整回溯。

---

## Case 77: postprocess 以 aligned 子集为分母，139 条缺失完全不进报告（已修复） (postprocess_denominator_union)

状态：已修复：回退分母改为 lab+audio 并集（非仅 lab stems）；postprocess 后验证 passed/filtered 集合契约；进程启动前检测缺失 aligned stem 并拒绝执行。

### 现象

postprocess 只枚举 aligned 目录中的 17,861 个 TextGrid，因此未对齐的 139 条既没有 report 行，也没有 error 或 filtered 状态。

### 根因

后处理把“已有输入”误当成“应该交付的全集”，没有从 CTC lab、音频或运行 manifest 得到 expected stems。

### 修复方案

1. 后处理启动前比较 CTC lab、音频、aligned 三个 stem 集合。
2. 缺 aligned、缺音频或多余 aligned 都是前置硬失败。
3. report 必须对 expected stem 一条且仅一条。
4. output 与 filtered 的 stem 并集必须等于 expected，交集必须为空；error 则使本次运行整体失败。

### 验收

- 139 条缺失必须在 postprocess 启动前被明确列出，而不是生成 17,861 行的“完整报告”。
- report 重复 stem、少 stem、多 stem均返回非零。

---

## Case 78: staging、filtered 和 NAS 发布非版本化，陈旧文件污染结果（未闭环）

状态：修复方案已记录，未闭环，暂停修复。

### 现象

- 本次 report 可信 ok 标称 14,200，但 NAS 有 16,350 个 TextGrid。
- 其中 2,149 条属于本次 filtered 却仍留在 NAS，另有 1 条是本次缺 aligned 的陈旧文件，共 2,150 条非本次交付。
- workspace filtered 有 3,667 条，本次报告只有 3,661 条，多出 6 条陈旧文件。

### 根因

1. 旧发布方式向固定 NAS 目录增量复制，不删除旧文件，也没有 manifest 对账。
2. 当前草案虽使用 run-specific 本地 staging，但最终目标仍是配置中的固定非空目录；安全发布函数会拒绝它，所以“版本化目标”尚未真正生成。
3. filtered_dir 仍是共享目录，新旧运行会混合。
4. 历史逻辑可能在上游失败后仍尝试发布已有 staging。

### 修复方案

1. 每次运行使用唯一 run id，同时生成独立 output、filtered、report、logs 和 manifest。
2. NAS 只发布到从未存在或为空的新版本目录，例如 0805test.runs/运行号；不得覆盖或清理旧目录。
3. 发布前要求全管线成功并通过 exact stem、文件大小和 manifest 校验。
4. 发布后从目标重新读取 manifest 做逐文件核对。
5. 旧 0805test 保留为不可变问题现场，不执行删除或镜像覆盖。

### 验收

- 新版本目录文件集合与本次 output manifest 完全相等。
- filtered 文件不进入发布目录。
- 重跑不会复用上一次 output 或 filtered。
- 任何步骤失败时 NAS 新版本目录不得出现。

---

## Case 79: tone_mapping.json 写入仓库默认 output，未随本次交付（修复草案，未验证）

状态：修复草案已写入，暂停复审/复跑。

### 现象

后处理 TextGrid 位于工作区或 NAS staging，但 tone_mapping.json 落在仓库 output 目录。结果消费者无法确定该映射属于哪次运行。

### 根因

tone-ref 没有由主管线显式绑定到本次 output_dir，默认相对路径与运行目录耦合。

### 修复方案

1. 主管线显式传入本次 staging 下的 tone_mapping.json。
2. report 和 publish manifest 记录其路径、大小和内容摘要。
3. 映射生成失败或不属于本 run id 时，postprocess 和发布均失败。

### 验收

- 每个版本化交付目录都有唯一、可解析、与本次报告匹配的 tone_mapping.json。
- 仓库工作目录不再被生产运行隐式写入。

---

## Case 80: 051809 CTC 前八个词为零时长，不能从 tokens 直接恢复（待单条 CTC 重跑）

状态：词序校验已实施（`validate_ctc_transcript_bundle` 验证 lab/tokens/TextGrid 三方言序一致）；零时长 token 时序校验已在 `_clamp_words_to_wav_axis` 中实施；已通过 `_case_ctc_bundle_rejects_zero_duration_tokens` 单元测试；待单条 CTC 重跑（051809）。

### 触发样本

051809_礼物互动_特殊礼物反馈

### 现象

严格 dry-run 检查发现该样本前 8 个 token（“欢迎每一位瑞士卷”对应拼音）均为 start=end=0；CTC TextGrid 中对应 interval 同样为零时长。音频总长约 7.336 秒，参考文本存在。

### 根因与风险

- tokens 和 TextGrid 虽然彼此“相等”，但共同包含不可能的时间戳。
- 只校验 lab、tokens、TextGrid 的词序相等会把共同损坏误判为可信。
- 不能通过平均分时长或从邻词猜测边界修复，否则会伪造声学证据。

### 修复方案

1. 从原音频和权威参考文本只重跑该 stem 的 CTC。
2. 要求每个 lexical/NVV token end 大于 start，整体单调、不重叠并在音频时长范围内。
3. 用新 bundle 替换隔离恢复副本中的同 stem 文件，不修改旧工作区。
4. 再对 18,000 个 bundle 做严格全量验证，只有全部通过才允许恢复 lab。

### 验收

- 该样本不再含零时长 lexical interval。
- 词序与权威参考一致，tokens、lab、CTC words tier 三者一致。
- 单条重跑仍失败时必须隔离并停止全量 MFA，不得猜时长。

---

## Case 81: ria 合并未在三份 CTC 载体中原子同步（未闭环）

状态：修复方案已记录，未闭环，暂停修复。

### 现象

旧 ctc_prealign.py 的 RIA safety net 将 rui4 ya4 合并为 ria 时只修改 lab 与 tokens，CTC TextGrid words tier 保持旧的两个词。当前 run_pipeline.py 已有三份同步草案，但 TextGrid 原地写入先于临时 lab/tokens 替换，异常中途仍可能留下半更新 bundle。

### 根因

RIA 归一化被当作普通文本替换，而 lab、tokens、CTC TextGrid 实际构成一个不可拆分的 transcript bundle。

### 修复方案

1. ctc_prealign 和主管线统一调用同一个三方同步函数。
2. 先在临时文件中生成 lab、tokens 和 TextGrid，完整校验后再提交。
3. 任一文件缺失、解析失败或词序不一致时三份都不提交。
4. 更新后清除旧 normalize marker，重新验证并写新 marker。

### 验收

- rui4 ya4 或 rui4 a4 合并后，三份载体都只含一个 ria，起止时间覆盖原两个 token。
- 在提交中途注入异常时，原 bundle 三份文件保持一致，不出现半更新。

---

## Case 82: normalize marker 生命周期与 MFA 入口验证不完整（已修复） (marker_content_identity_v4)

状态：已修复：marker v4 嵌入 stem 数量和 manifest SHA-256 digest；`_skip_if_ctc_normalized` 解析并验证内容身份，拒绝过期/篡改 marker；`recover_ctc_labs.py` 重建 lab 后自动清除旧 marker。

### 现象

1. marker 是固定版本字符串，无法证明 marker 对应当前文件内容。
2. 恢复脚本 apply 修改 lab 后没有清除旧 marker。
3. 分步执行 normalize 链时，成功后的 marker 写入路径不统一。
4. align 入口尚未无条件重新验证全部 CTC bundle；旧 marker 或直接指定 step=align 可能绕过校验。

### 根因

1. 旧 marker 被设计为简单的完成标志（常量字符串），未绑定任何内容身份证明，无法区分"本次数据已验证"和"标记残留"。
2. 恢复脚本（`recover_ctc_labs.py`）修改 lab 后未联动清除 marker，导致下游误判数据已就绪。
3. 分步执行模式下 normalize 链的 marker 写入分散在多个调用点，缺少统一的事务边界。

### 风险

损坏的 lab、tokens 或 TextGrid 可能因 marker 存在而跳过 normalize，随后进入 MFA。用户单步运行命令更容易触发该绕过。

### 修复方案

1. 任何 transcript 文件变更前先使 marker 失效。
2. 只有完成全目录 strict validation 后才写 marker。
3. marker 至少记录契约版本、stem 数量和 manifest 摘要；不能只依赖常量字符串。
4. MFA align 启动前无条件执行快速 bundle 校验，marker 只允许优化，不允许替代验证。
5. 缺 marker 不应自动修改数据；只表示必须验证或重新执行明确步骤。

### 验收

- 修改任一 lab 后旧 marker 不能让 align 继续。
- 直接运行 step=align 遇到 Case 80 或任意三方不一致时，在清理 aligned 之前失败。
- 恢复 apply 后 marker 被清除，重新校验成功后才产生新 marker。

---

## Case 83: MFA TextGrid 验证和子进程异常处理仍不充分（已实施 strict validator + 进程异常捕获）

状态：代码已实施（2026-08-07）。`pipeline_utils.py` 新增 `validate_strict_mfa_textgrid` 严格 long TextGrid parser/validator，显式检查 grammar、tier 唯一性、interval 合法性、WAV domain；`run_pipeline.py:_run_mfa_sharded` 用 strict validator 替换原 `"words" in content` 字符串匹配，Popen 增加 OSError 异常捕获，超时/信号/非零退出统一写入结构化失败记录；单进程 MFA 路径同样调用 strict validator。`verify_strict_ctc_ready_import.py` 新增 9 个 TextGrid validator 负例测试，`verify_reference_authority.py` 新增 1 个回归 case。

### 现象

当前分片草案通过搜索 name = words 和 name = phones 判断输出存在 tier。损坏、截断或 interval 非法的 TextGrid 仍可能包含这两个字符串。单进程和分片路径的检查粒度也不完全一致。

### 根因

1. MFA 分片验证采用字符串包含匹配（`"words" in content`），而非结构化 TextGrid 解析，无法检测语法损坏。
2. 单进程路径和分片路径各自实现了不同的验证逻辑，未收敛到统一的 validator。
3. 子进程异常（OSError、超时、信号退出）未被系统捕获并转换为结构化失败报告，部分失败模式静默。

### 修复方案

1. 使用项目 TextGrid parser 真正解析每个输出。
2. 校验 tier 唯一性、interval 数量、非负时长、单调性、音频边界和必要词面。
3. 统一单进程与分片的验证和 manifest 生成。
4. 捕获进程启动 OSError、超时、信号退出及返回码异常，全部写入结构化失败报告。

### 验收

- 只有 tier 名字符串但语法损坏的文件必须失败。
- words 存在而 phones 缺失、interval 倒置或超出音频时长都必须失败。

---

## Case 84: 裸词 unk 被一律当作 MFA unknown，可能误伤真实英文词（未修复）

状态：问题已定位，未修复。

### 现象

当前 unknown helper 同时识别 <unk>、[bracketed] 和裸词 unk。裸词 unk 可能是参考文本中的真实英文拼写，不能仅凭字符值判为 MFA 占位符。

### 根因

1. `is_unknown_token()` 在 `pipeline_utils.py` 中的实现过于宽泛，将裸词 `unk` 与 MFA 明确占位符 `<unk>` 等同处理。
2. 缺少参考文本对照和 MFA phone 上下文（是否为 `spn`）的二次确认，纯字符匹配无法区分"MFA 对齐失败的占位符"和"参考文本中的真实英文词"。

### 修复方案

1. 默认只识别 MFA 明确占位格式，例如 <unk>。
2. 对裸词 unk 必须结合参考文本、MFA phone=spn 或解析器元数据判断。
3. NVV 标签在 unknown 判断前优先分类，避免合法标签误入 unknown。

### 验收

- 参考英文句中真实的 unk 保持英文词。
- MFA 输出的 <unk> 仍可触发 CTC 词面回填和 mfa_unknown_source 硬失败。

---

## Case 85: nvv_enabled=false 与 NVV 发现需求存在配置冲突（待确认）

状态：待确认并修复。

### 现象

hecheng_ria_0805.yaml 当前设置 ctc_prealign.nvv_enabled=false。该设置会关闭从音频中发现额外 NVV 的能力。

### 根因

1. 配置层面未区分"参考文本已标注的 NVV（应始终保留）"和"需从音频自动发现的 NVV（受 nvv_enabled 控制）"两种语义。
2. `nvv_enabled` 作为一个统一开关同时影响 blank-frame NVV bias 和 NVV token 的发现/保留逻辑，导致关闭自动发现时可能连带影响参考标注 NVV 的贯穿。

### 边界说明

- 如果权威参考文本已经显式包含 NVV，关闭自动发现不应删除这些标签；它们必须沿参考文本贯穿 CTC、MFA 占位和最终 tiers。
- 如果业务要求从音频自动发现参考文本中未写出的 NVV，则当前配置不满足要求，必须启用并单独评估误检和漏检。
- 无论是否自动发现，NVV 都不计入 CJK 与拼音一一对应数量。

### 修复方案

1. 明确本批次 NVV 的权威来源：参考标注、音频自动检测，或两者合并。
2. 参考标注始终保留；自动检测结果必须记录来源和置信策略，不能覆盖参考词序。
3. 为 NVV 建立独立的标签顺序、数量、边界和格式 QC。

### 验收

- 参考文本中的每个 NVV 在最终 words、hanzi/raw 展示层按契约保留。
- 若启用自动发现，新增标签有独立 provenance，且不会造成汉字拼音计数误报。

---

## Case 86: 18,000 条音频只有 17,999 条权威参考文本（待补参考或隔离）

状态：已修复。`--no-nvv` 参考文本模式下，无参考文本的 stem 直接跳过不处理（不退回 ASR），管线正常继续。`ctc_prealign.py` line 1466–1472。

### 缺失 stem

036000_弹幕互动_回应吐槽弹幕

### 现象与风险

现有流程对该文件退回 ASR 文本，因此一个批次中混入了两种文本权威级别。若仍以与其余 17,999 条相同的标准发布，会破坏”参考文本是唯一权威”的契约。

### 根因

1. 管线在 `{stem}.txt` 缺失时静默退回 ASR 解码文本（`_text_cn.txt`），未区分”有权威参考”和”ASR fallback”两种质量等级。
2. 缺少批次级的 reference_source 枚举和 manifest 记录，17,999 条 reference 和 1 条 asr_fallback 在同一个 output 目录中不可区分。

### 修复方案

优先级如下：

1. 从源数据补齐对应 txt，并按完整参考模式重跑。
2. 若无法补齐，将该 stem 显式隔离为 missing_reference，不进入权威参考批次。
3. 只有业务明确接受 ASR fallback 时，才能另建不同质量等级的交付，并在 manifest 标明 reference_source=asr_fallback。

### 验收

- 全量 manifest 对每个 stem 明确记录 reference_source。
- 权威参考批次中不存在静默 fallback。
- 若要求最终必须交付 18,000 条，则补齐该 txt 是前置条件。

---

## Case 87: 只运行 postprocess --overwrite 无法修复上游污染（操作风险）

状态：操作风险已记录，非代码缺陷。

### 现象

用户命令指定 step=postprocess。overwrite 只覆盖该步骤将要写出的后处理文件，不会重新生成 ctc_pretg、lab、aligned 或英文 MFA 结果。因此：

- 18,000 个污染 lab 不会改变；
- 17,861 个含 unknown/spn 的 aligned 不会改变；
- 缺失的 139 个 aligned 不会补齐；
- 后处理最多能回填展示词面，无法恢复真实 phone 对齐。

### 根因

1. `--step postprocess --overwrite` 只覆盖后处理阶段产物，上游步骤（CTC prealign、normalize、MFA align）的缓存/输出被复用。
2. 管线设计允许任意步骤独立运行，但缺少跨步骤的产物版本校验——后处理不检查上游 CTC lab 是否已被 cn2an 污染或 MFA aligned 是否包含 unknown/spn。
3. 这是操作流程风险而非代码逻辑 bug：正确的修复需要从污染源（CTC lab）开始全链路重跑，而非单独重跑后处理。

### 修复后的正确运行边界

恢复工作必须使用新工作区和版本化输出，从严格验证或恢复后的 CTC 开始重新执行 normalize、MFA align、English align 和 postprocess；不得在旧 0805test 上原地覆盖。

### 命令行注意

Shell 多行命令中的反斜杠必须是该行最后一个字符。反斜杠后若存在空格，续行可能失效。正式运行前应保存并回显完整解析后的参数。

### 验证方法

此案例为操作风险文档，不涉及代码修改验证。预防措施：
- 全量重跑应使用新工作区和新 `--output-dir`，不在旧目录上 `--overwrite`。
- 运行前用 `recover_ctc_labs.py --dry-run` 预检上游 CTC bundle 完整性。
- 在 `run_pipeline.py` 中通过 marker 版本化阻止过期产物被误用（见 Case 82）。

---

## 暂停后的统一修复方案（仅记录，尚未执行）

### 阶段 0：冻结问题现场

1. 保留原工作区 /mnt/nvme3/mfa_workspace_ria_0805 和 NAS /mnt/Raw/0805test 不变。
2. 保存旧 report、CTC/MFA stem manifest、NAS 文件清单和问题样本。
3. 所有恢复只在新工作区与新版本输出目录进行。

### 阶段 1：完成代码契约与回归

1. 完成 Case 72 至 87 的代码复审，尤其是 RIA 三方原子同步、marker 生命周期、MFA 入口校验和真正版本化发布。
2. 建立 tone 1–5、unknown、NVV、标点、句首 <sp1>、纯英文、混合英文、缺参考和零时长 token 回归。
3. 单进程和分片 MFA 使用同一输出验证器。

### 阶段 2：隔离恢复 CTC

1. 只重跑 Case 80 的 051809 CTC，不猜测时间。
2. 处理 Case 86：补齐 036000 参考文本或明确隔离。
3. 将旧 CTC 复制到新工作区，在副本上从已验证 tokens 重建 17,999 个损坏 lab。
4. 对全部目标 stem 验证 lab、tokens、CTC words 三方词序以及时间合法性。

### 阶段 3：小规模 canary

选择至少包含以下类型的样本：

- 036022：中文、Claude、dancer、ria 混合；
- 051809：历史零时长 CTC；
- 含 NVV、标点和句首 <sp1>；
- 纯中文、纯英文、混合英文；
- 036000 或其 missing_reference 隔离路径。

canary 必须从新 CTC 工作区运行到 MFA 和 postprocess，不能只跑 postprocess。

### 阶段 4：全量新工作区重跑

1. 生成冻结的 expected stem manifest。
2. 在全新 aligned、en_phones、output 和 filtered 目录运行。
3. 持续监控每 shard 退出码、日志、GPU/CPU 进程、产出计数和超时。
4. 任一阶段失败立即停止后续发布，不使用 force 掩盖失败。

### 阶段 5：版本化发布

1. 只将通过 hard integrity 的 output 发布到全新 NAS 版本目录。
2. filtered、error、logs、report 和 manifest 保存在同 run id 的审计目录，但 filtered 不混入交付 output。
3. 发布后重新核对文件集合、大小和 manifest。

---

## 全量验收矩阵

| 层级 | 必须满足 |
|---|---|
| 数据全集 | 每个 WAV 都有明确 reference_source；缺参考不得静默 fallback |
| CTC transcript | lab、tokens、CTC words 词序完全一致 |
| CTC 时间 | lexical/NVV interval 均为正时长、单调、在音频范围内 |
| 声调 | 拼音只使用 tone 1–5，不得出现 ma一、rui四 等污染 |
| RIA | lab、tokens、CTC TextGrid 同步为一个 ria |
| MFA 数量 | aligned stems 与 expected stems 精确相等 |
| MFA 结构 | words、phones tier 可解析，lexical 内容不得来自 unknown/spn |
| 参考覆盖 | 每个汉字恰有一个带调拼音，字符序列与权威参考一致 |
| NVV | 标签按权威来源保留并独立校验，不计入拼音数 |
| 标点 | 顺序与参考一致，不因计数或 snap 被删除 |
| 句首静音 | 恰有一个 <sp1>，位置在句首 |
| 英文 | 拼写来自权威参考，音素来自 English MFA 或明确失败 |
| 报告 | expected stem 每条恰有一行；无重复、无遗漏 |
| 输出集合 | output 与 filtered 互斥，并集等于 expected；error 使整次运行失败 |
| tone mapping | 与本次 run id 同目录并纳入 manifest |
| NAS | 全新版本目录，文件集合与发布 manifest 完全一致，无陈旧文件 |
| 进程状态 | 任一 hard failure、缺 stem、超时或解析失败均向上传递非零退出码 |

### 当前结论

现有 /mnt/Raw/0805test 不能作为合格 MFA 结果交付。0 pinyin vs N reference CJK 是本批次上游 transcript 和 MFA 对齐整体损坏的直接信号，不是 NVV、标点或句首 <sp1> 导致。所有修复、测试和运行工作现已按用户要求暂停；恢复执行前应先以本节为唯一问题清单逐项确认。

## Case 88: strict-ok v3.1 独立发布门禁（2026-08-06）

**日期**: 2026-08-06
**涉及文件**: scripts/audit_strict_ok.py, scripts/verify_strict_ok.py, scripts/run_pipeline.py

### 现象

旧发布流程中，后处理 QC 的 `status=ok` 直接决定文件进入 output 目录，NAS 发布依赖进程内判断
而无独立磁盘级交叉验证。发布后的文件集合缺少可审计的 manifest，陈旧文件可能混入交付目录
（见 Case 78）。

### 根因链

1. 后处理 report 的 `ok/filtered/error` 分类完全依赖进程内 QC 逻辑，无外部验证。
2. NAS 发布为增量复制到固定目录，不校验目标文件集合与本次运行结果的一致性。
3. output/filtered 共享目录可被多次运行复用，缺少 run-id 级别的隔离。
4. 无发布 manifest 记录交付文件的 identity（路径、大小、SHA-256），交付后无法审计。

### 修复方案

- 新增 `scripts/audit_strict_ok.py`：独立磁盘 auditor，从 output 目录重读 TextGrid、WAV、
  权威参考、CTC bundle、MFA aligned 来源及 postprocess report，逐文件交叉验证；报告阳性
  只可否决，不能证明通过。
- 新增 `scripts/verify_strict_ok.py`：strict manifest 验证器，拒绝文件或集合漂移。
- output/filtered 使用同一 run-id 的私有目录且集合守恒（并集等于 expected，交集为空）；
  二次失败会原子隔离到该 run 的 filtered，safe_empty 明确不可发布。
- manifest 记录 strict-ok-v3.1 契约版本、最终 TextGrid/参考 SHA-256、英文 MFA 自包含证据、
  已执行检查与未评估的主观声学自然度。
- 版本发布仅接受有效 strict manifest，且目标必须是新的 `<configured>.runs/<run_id>` 目录。

### 验证方法

```bash
python scripts/verify_strict_ok.py <strict_ok_manifest.json> <output_dir>
# 预期: 0 errors，manifest 文件集合与实际目录完全一致
python scripts/audit_strict_ok.py --require-expected-counts
# 预期: 独立重读验证通过，文件身份与 manifest 一致
```

状态：实现草案经 root 复审，待真实 canary；未对旧 NVMe/NAS 结果运行或发布。

---

## Case 89: 旧英文空 phones 被 fallback 伪装为可用音素

### 现象

旧英文产物中有 10,742 个 JSON 文件出现 `phones: []`。旧后处理路径可在英文 MFA 没有提供可验证 phone 时，改用 CMU/G2P 词典序列或在词区间内等分时间；最终 TextGrid 外观完整，却不能证明 phone 标签和边界来自该条音频的 English MFA 对齐。

### 根因

旧契约只验证“最终是否有英文 phone”，没有验证“每个 phone 是否逐项来自本次成功的 English MFA 源 TextGrid”。同一个词典序列既被当作发音先验又被错误地当作声学对齐证据，空 phone 因而能够被补齐后进入通过集。

### 修复方案与状态

1. 严格模式使用 `strict-en-mfa-v1` 全局 manifest 和逐 stem ledger，冻结 CTC full-tier ordinal、segment/word ID、源 TextGrid 路径与 SHA-256。
2. 最终英文 phone 必须与源 MFA ARPABET 序列完全一致，时间只能由源 word/phone 区间做仿射映射；CMU、G2P、自引用词面和等分时间只允许存在于非严格兼容路径。
3. 任一英文 segment 缺源、空 phone、未知 phone、错序、越界、哈希不符或局部拒绝，只过滤受影响 stem；不得生成 fallback phone。
4. 独立 auditor 再次重读 CTC、源 MFA TextGrid 和最终 TextGrid，不信任后处理自报。

状态：合成回归已覆盖空 phone、缺源、哈希篡改、重复英文词和中英文分隔；仍须以真实 canary 和全量输出确认。旧英文结果不能因外观完整而追认通过。

---

## Case 90: 英文批次分母不一致与混合缓存污染风险

### 已确认盘点

- 权威源目录 `/mnt/Raw/新版合成英文数据`：54,000 个 WAV，53,998 个 txt。
- 精确缺参考 stem：`024198_杂谈互动_数据里程牌庆祝`、`036000_弹幕互动_回应吐槽弹幕`。
- 旧 `/mnt/nvme3/mfa_workspace/ctc_pretg` 有 54,000 个六件套 CTC bundle。真实只读分类为：7,204 条标准 TextGrid 可原样复制，46,586 条通过共享 transcript 契约但属于 Case 93 的精确旧畸形格式、须在隔离目录规范化，208 条共享契约硬错误必须从原音频重跑。
- `/mnt/nvme3/mfa_audio_cache` 有 84,788 个 WAV，manifest 指向另一来源 `/mnt/Raw/shayi_huali_wav`，不是本批次权威音频集合。
- 旧 aligned 仅 53,364 条，不能作为 53,998 条目标分母的完整输入。

### 风险

若用 54,000 WAV 当权威参考分母，会让 2 条缺 txt 静默退回 ASR；若使用 84,788 条混合 cache，会把其他数据集音频混入本次运行；若直接复用全部旧 CTC，会让 208 条严格无效 bundle 进入后续 MFA。

### 修复方案

目标分母固定为 53,998 条有权威 txt 的 stem，2 条缺参考明确排除并写入 manifest。只在全新 run root 中处理旧 bundle：7,204 条标准 bundle 普通复制，46,586 条按 Case 93 的严格、可证明转换方案只规范化 TextGrid，精确 208 条从原音频单 GPU 重跑；三类结果都必须得到标准 parser 可读、集合精确相同的六件套，再合并为 53,998 条 ready 集。禁止使用混合 audio cache，也不得修改旧 CTC、旧 aligned 或旧 NAS 输出。

状态：2026-08-06 已完成三轮真实只读盘点，确认 54,000 WAV、53,998 txt、2 个指定缺参考、0 个 txt-only，以及 7,204/46,586/208 的精确三类分布。run root 仍不存在。准备器尚未实现严格旧格式规范化，当前不得据此写入生产数据。

---

## Case 91: strict-en-mfa-v1 来源链的局部/全局失败语义不完整

### 复审发现的问题

1. 全局 manifest 若把一个可局部过滤的 segment rejection 当作整批失败，会不必要地阻断其他可证明正确的 stem；反之，MFA 非零、超时、启动异常、非空 exception、模型/词典哈希失败若只做局部拒绝，会让未知全局状态进入通过集。
2. 仅比较 ledger 中的 phone 列表属于自证；auditor 必须重读源 English MFA TextGrid。
3. segment ID/ordinal 若未绑定唯一、同名的源 TextGrid，相同英文片段可能复用错误来源。
4. 最终 words 内的英文 phone 即使正确，其他位置额外出现 `en:` phone 仍是污染。
5. 逐 stem 直接复制证据时，后续全局失败可能留下孤立 `_provenance`，不能把未完成证据误认为可发布证明。
6. 所有英文段均因过短等本地原因拒绝时，虽然 MFA 不启动，仍必须能哈希实际模型和配置词典；纯中文 `no_english` 才允许完全不依赖英文模型。

### 契约

- 全局异常：整次审计非零、不可发布。
- 局部英文来源异常：只过滤含异常 segment 的 stem；其他 stem 仍需逐条独立证明。
- 通过 stem：CTC full ordinal、ledger、源 MFA words/phones、最终 words/`en:` phones 和证据副本形成一一对应闭环。
- filtered 结果不纳入准确性结论，但必须与 output 互斥且并集守恒。

状态：producer、postprocess 与 auditor 的合成 P0 回归及 root 独立复跑均已通过；仍待 Case 92/93 runner 与真实 canary 验收。

---

## Case 92: 最新英文配置和隔离 CTC 准备器存在执行级风险

### 根审查发现的问题

初版准备器虽然能粗分类旧 bundle，但生成了 `ctc_prealign.py` 不支持的 `--stems-file`，遗漏必需的 `--pinyin-dir`，并使用了与“208 条隔离单 GPU 重跑”不一致的 `--all-gpus`。它还可能把 repo 词典直接交给会追加英文词条的 CTC 脚本，缺少 `use_cache: false`，只核对部分计数，未完整证明 source/destination copy equality、精确 rerun stem set 和 ready 产物哈希。后续真实盘点又发现 Case 93 的 46,586 条旧格式规范化需求，以及主管线 link 使用 symlink/hardlink 会被 pad/normalize 穿透改写 ready 源集。

### 修复方案

1. 配置固定 `ctc_ready`、全新 workspace、`use_cache: false`、`output_staging: false`、strict English provenance、MFA timeout 7200 秒和 G2P timeout 600 秒。
2. prepare 创建 run-local 普通词典副本并记录源/目标 size 与 SHA；CTC 重跑只写 run-local 字典。
3. 生产门禁精确检查 54,000 WAV、53,998 txt、53,998 authoritative、2 个指定 missing-reference stem、0 个 txt-only stem。
4. 重跑命令只作用于隔离的 rerun audio 目录，提供 `--pinyin-dir`，默认 `--device cuda:0`，保留 NVV。
5. finalize 必须在复制前验证重跑输出 stem set 精确等于待重跑 set；任何 extra、missing、损坏、symlink、路径逃逸或冲突都非零停止。
6. ready manifest 记录全部音频和 CTC 六件套哈希，并由 read-only `verify-ready` 重读磁盘复核。

状态：准备器第一轮 P0 已通过合成回归，但 Case 93 规范化、主管线普通复制/精确 evidence 分母、MFA 工作词典隔离仍在 Sol/Terra 完整调度中；完成前不创建真实 run root。

---

## Case 93: 旧 CTC writer 生成非标准双 tier TextGrid，标准 parser 得到空 words

### 真实只读盘点结果

2026-08-06 对 `/mnt/Raw/新版合成英文数据` 与 `/mnt/nvme3/mfa_workspace/ctc_pretg` 运行新版准备器的只读 `inspect`：

- WAV 54,000；txt 53,998；权威交集 53,998；缺参考为 Case 90 的精确 2 条；txt-only 为 0。
- 在不创建 run root、不修改任何源文件的前提下，准备器首轮标准 named-tier gate 将 53,998 条全部判为不可直接使用，因此得到 `legacy_valid=0 / needs_rerun=53,998`，触发生产停止条件。
- 三条抽样均为 `words/tokens count mismatch`。例如 `000000_直播流程_开场介绍` 的共享旧校验器返回无错误，tokens 有 43 条，而标准 parser 得到 words tier 0 条、pauses tier 81 条。

### 根因

旧 writer 的原始顺序为：

```text
name = "words"
intervals: size = 43
item [2]:
intervals [0]: ... "rui4"
...
name = "pauses"
intervals: size = 38
```

也就是在写完 words 的数量声明后、真正写 words intervals 之前提前插入 `item [2]`。标准 parser 因而把后续 43 个实词 interval 归入第二 tier，并在读到 `name = "pauses"` 后继续把 38 个 pause interval 追加到同一 tier。旧共享 reader 与 English `parse_textgrid_simple` 只是按 `name="words"` 到 `name="pauses"` 的文本顺序宽松扫描，所以仍能恢复 43 个词；这解释了旧版为何曾估计 53,790 条“有效”。当前 `ctc_prealign.py::write_textgrid` 的写入顺序已经正确，但旧 54,000 个文件不会自动修复。

### 风险

1. MFA/English 分段若使用宽松 reader，可能继续工作；独立 auditor 使用标准 tier parser 时会得到空英文集合或 ordinal 错位，两条路径对同一文件产生矛盾解释。
2. 直接放宽标准 parser 会把未知结构损坏也当作已知 writer bug，扩大伪通过面。
3. 直接把 53,998 条全部送 GPU 重跑成本高且混淆“格式恢复”和“声学重算”；但未经证明的文本重写也不能替代真正无效的 208 条 CTC。

### 精确分类补充

随后用共享六件套契约与独立标准 parser 重新只读分类全部 53,998 条：

- 7,204 条：标准 words/pauses tier，可由通用 parser 正确读取；
- 46,586 条：共享 transcript 契约通过，但属于上述唯一旧畸形 grammar；
- 208 条：共享契约已失败，主要含零时长 token，必须声学重跑。

TextGrid 与 `_tokens.jsonl` 必须有相同 lexical 词序和对应 start，但不能错误要求每个 end 完全相等：当前 writer 的 TextGrid word end 使用下一 lexical word start/CTC end，而 tokens 的中间 end 可受标点 start 影响，最后 end 会由 VAD 重估。两套 end 各自都必须正时长、单调且在音频内，规范化旧 TextGrid 时须保留旧 TextGrid 自身的 lexical start/end，不得用 tokens end 猜写。

### 待实施的严格方案

1. 只识别并接受这一种精确旧畸形模式；任何 tier 顺序、数量或字段不同均拒绝。
2. 对候选逐项比较旧 words interval 与 `_tokens.jsonl` 的词面和 start；两种容器分别要求正时长、有限、单调、不重叠并在对应 WAV 时长范围内。只有已知旧畸形 writer（其 words 确由 tokens 重写）才额外要求 end 一致；标准 writer 不作跨容器 end 等值要求。同时继续执行 lab/tokens/reference 三方契约。
3. 对满足上述证明的 46,586 条候选，只在 fresh run root 中确定性写出标准 named `words`/`pauses` tiers；源 TextGrid、tokens、音频和目标 TextGrid 的路径、size、SHA-256 与转换版本全部进入 evidence。不得原样覆盖旧文件。7,204 条标准文件普通复制并证明 source/destination 相等。
4. 规范化后必须由标准 parser 重读；唯一 words tier 的非空 interval 必须保留已证明的旧 TextGrid 词面/start/end，并与 tokens 的词面/start 对应。full-tier ordinal 以后只以规范化文件为准。
5. 不能满足精确旧模式或任一内容/时间检查的 stem 才进入原音频 CTC 重跑集合；重新计算后才能确认是否仍为 53,790/208。
6. pauses tier 必须由精确旧 grammar 状态机单独恢复并验证数量、正时长、单调和音频边界；不得把标准 parser 当前看到的 81 条合并 tier 直接复制为合法 pauses。无法无歧义恢复的 stem 转入重跑。

状态：问题已定位并进入 Sol high 架构复审；尚未实施转换、未创建 run root、未启动 GPU。

---

## Case 94: v3 首版严格解析器没有忠实实现真实旧 grammar

### 主审发现与真实证据

在接受 Terra 的第一版 `hecheng-english-ctc-ready-v3` 实现前，root 直接只读检查了真实文件
`/mnt/nvme3/mfa_workspace/ctc_pretg/000000_直播流程_开场介绍.TextGrid`。真实旧格式为：

1. `name = "words"` 后直接出现 `intervals: size = 43`，words tier 没有自己的 `xmin/xmax`；
2. 随后提前出现 `item [2]` 和 `class = "IntervalTier"`；
3. words interval 使用零基编号 `intervals [0]` 至 `intervals [42]`；
4. words 结束后才出现 `name = "pauses"`，pauses tier 带有 `xmin/xmax`，其 interval 使用一基编号。

首版状态机却要求 words tier 先有 `xmin/xmax`，并要求 words interval 从 1 开始。因此它虽然测试通过，仍会把真实 46,586 条可证明恢复的旧文件全部拒绝。测试 fixture 复制了实现的错误假设，而不是生产文件的精确语法。

同一次主审还发现三项相关缺陷：

- CTC 六件套缺文件时，`classify_ctc_bundle` 返回单个 `errors` 列表，而调用方按三元组解包，缺文件会从预期的严格分类失败变成异常崩溃；
- `_canonical_words` 从 `_tokens.jsonl` 取词的 start/end，违反 Case 93 已冻结的转换契约。即使 token end 与旧 TextGrid end 只相差容差内数毫秒，目标文件也必须逐值保留旧 TextGrid lexical start/end，不能用 token end 代写；
- 标准 TextGrid 检查只要求能找到唯一 `words`/`pauses`，未拒绝第三个未知 tier、错误 tier 顺序或 tier domain 与全局 domain 不一致，证据边界不够封闭。

### 影响

若直接执行生产 `inspect`，分类计数将偏离已独立确认的 7,204 / 46,586 / 208，导致不必要的 46,586 条 GPU 重跑或生产门禁停止。若仅放宽计数继续，规范化目标又会丢失旧 lexical end 的逐值来源证明，不能成为 strict-ok 的可信 CTC authority。缺文件分支还可能绕过结构化错误报告。

### 修复与验收方案

1. 状态机只接受上述“words 无 domain + 零基 words + 一基 pauses”的唯一生产 grammar；注入 words domain、把 words 改为一基、缺/多 interval 或追加尾字段均须拒绝。
2. 缺文件路径固定返回 `(None, errors, None)`，并用缺任一六件套的 fixture 验证不会崩溃且归入 rerun。
3. canonical words 从解析出的旧 `words` interval 构建；tokens 只用于词面/start/end 容差证明和 token→规范 full-tier ordinal 映射。测试必须构造旧 end 与 token end 不同但在容差内的样本，并断言输出精确保留旧 end。
4. 转换 evidence 增加固定 parser signature、transform version、word count、pause count，并由 `verify_transforms` 逐项重算/拒绝篡改。
5. 标准输入和规范化输出都只允许顺序严格为 `words`、`pauses` 的两个 tier，两个 tier domain 均须等于全局 domain。
6. 先通过合成回归，再重新执行真实只读 `inspect`；只有计数精确回到 7,204 / 46,586 / 208 才允许创建 fresh run root。

状态：已在任何生产写入前阻断，并退回 Terra 修复；真实 run root 仍未创建，GPU 尚未启动。

---

## Case 95: 历史 padding 原地污染旧 CTC 时间轴，真实 v3 分类再次失败

### 真实门禁结果

Case 94 合成回归通过后，root 于 2026-08-06 执行只读生产门禁：

```bash
python scripts/prepare_hecheng_english_ctc_ready.py inspect --require-expected-counts
```

命令逐条读取 53,998 个权威 stem，约 24 分钟后以非零退出：

```text
actual=(54000, 53998, 53998, 2, 0, 9, 70, 53919)
expected=(54000, 53998, 53998, 2, 0, 7204, 46586, 208)
```

生产 run root 在此之前和之后都未创建，GPU 未启动。门禁正确阻止了错误的准备结果。

### 样本证据与根因链

`000000_直播流程_开场介绍` 的权威 WAV 位于
`/mnt/Raw/新版合成英文数据/雪狐桑/`，时长为 12.480 秒；其 SHA-256 与
`/mnt/nvme3/mfa_audio_cache_ria/雪狐桑/` 中的缓存副本完全相同。旧 CTC 却有：

- TextGrid 全局域：`0.208–12.688`，跨度恰好仍为 12.480 秒；
- tokens 首词从 0.718 秒开始，末尾标点到 12.688 秒；
- 历史 `padded_audio` 时长仅为 12.392 秒。

历史 `pad_silence_edges.py` 会直接修改传入的 `ctc_pretg`：根据头部补/裁得到一个 `time_offset`，把 TextGrid、tokens 和 punct 的所有时间统一平移并将负数截到 0；但尾部静音裁剪只缩短音频，不同步修正/限制 CTC 末端。本样本原始 CTC 显然被整体加了 0.208 秒，随后 padded WAV 又在尾部缩短约 0.296 秒。因此旧 `ctc_pretg` 已不是与权威源 WAV 同轴的原始声学结果，也不与历史 padded WAV 自洽。

首版 v3 准备器把“区间必须落在权威 WAV 的 0–duration”作为正确门禁，因而真实分类仅剩 9 条标准、70 条可规范化，其余 53,919 条被拒绝。这不是可以通过放宽越界判断解决的问题；必须先证明每个 stem 的历史平移及信息是否可逆，再决定确定性重基或声学重跑。

### Sol high 最终实现审计同时发现的准备层 P0

1. 旧畸形 parser 仍错误要求 TextGrid lexical end 与 token end 相等；冻结契约只允许跨容器比较词面、顺序和 start，同时要求各容器自身时间合法。
2. 标准路径没有独立证明 token 时间有限、正时长、单调并在对应音频轴内。
3. `finalize` 用双模式校验 rerun 输出，可能接受旧畸形 grammar 后原样复制；rerun 和最终 ready 都必须严格为标准 grammar。
4. transform identity 未把 `source_audio` 绑定到该 stem 的权威 WAV，存在跨 stem 音频替换空间。
5. standalone `verify-ready` 只做 stems 的 set equality，重复或乱序列表可能由准备器自身漏检；类别列表也应精确、排序、唯一。

### 重新定界方案

1. 对全部 stem 只读统计 `(TextGrid xmin, xmax, WAV duration)`、tokens/punct 极值及历史 padded WAV；按可证明的共同平移、发生负值截断、尾裁越界和其他结构错误分类，不沿用 7,204/46,586/208 作为未经复核的生产事实。
2. 只有当源音频 SHA 与权威 WAV 相符、CTC 全部时间字段共享唯一可逆偏移、重基后所有容器落在 0–duration 且词面/start 契约成立时，才允许在 fresh run root 中同时转换 TextGrid、tokens 和 punct，并记录每个字段的旧/新哈希、偏移和变换版本。
3. 发生 `max(0, t + offset)` 信息丢失、无法唯一推导 offset、重基后仍越界/重叠/零时长，或任何 transcript/grammar 错误的 stem 必须从权威 WAV 声学重跑，不得猜测修补。
4. 修复 Sol 审计的五项准备证据缺口，增加真实时间轴 fixture、畸形 rerun、跨 stem source audio、重复/乱序 evidence 列表回归。
5. 重新执行真实只读 inspect 并冻结新的精确分类计数；只有新分类可由独立统计复现且全套回归通过，才允许创建 run root。

状态：生产继续阻断；runner 隔离层通过 Sol 审查，但准备层和旧 CTC 时间轴未获接受。

### Sol high 最终重定结论

旧 manifest 只有旧 audio 路径、duration、words/pauses，没有输入音频 SHA、head/tail、offset 或 new-duration 记录。它能佐证 `000000` 的 `+0.208s` 原地平移，却不能把旧 CTC 声学来源绑定到与权威 WAV 字节相同的输入；负 offset 又经过 `max(0, t + offset)`，存在不可逆截断。因此 v3 被正式废弃，不采用 `9/70/53,919` 作为新生产分类，也不再猜测恢复旧 CTC。

本批次采用最小安全 v4 路径：

1. schema 升级为 `hecheng-english-ctc-ready-v4`，最终音频轴固定为未改动的权威 WAV；
2. 53,998 个有权威参考的 stem 全部标记为 `acoustic_rerun`，原因统一为 `legacy_audio_provenance_unbound`；
3. fresh CTC 直接在 run-local `audio_view` / `reference_view` 上多 GPU 生成，旧 CTC 不进入任何生产 ready artifact；
4. rerun 和最终 ready 只接受标准双 tier grammar，并独立验证 TextGrid、tokens、punct 的有限、正时长、时序、毫秒/秒一致性与音频边界；跨容器只比较词面/顺序/start，不比较 end；
5. v4 strict runner 禁止并省略 downstream `pad_silence`。若未来必须统一 0.5 秒边缘静音，应先生成 append-only padded audio、冻结其样本/hash，再在该音频上生成 CTC；不得在 CTC 之后改音频；
6. v4 evidence 与独立 verifier 绑定全量 stem、action taxonomy、权威源/运行副本音频身份、WAV frames/rate/channels、参考、六件套、运行词典及精确 namespace；
7. 只有全量 fresh CTC、finalize、独立 verify-ready、SHA/taxonomy pin 全部通过，才进入 canary/full MFA。

这一路径牺牲旧 CTC 复用成本，换取可机器证明的单一时间轴和声学来源，是“通过集不得含已知机器错误”目标下的最终选择。

### 多次累积 mutation 的只读溯源证据

Terra 对 Git 历史、旧 manifest、权威 WAV、现存 padded WAV 与 CTC 做了逐样本只读比对，确认 Case 95 不是一次可统一反向平移的污染，而是同一个 `ctc_pretg` 被重复运行 padding 后累计修改；音频目录却只保留最后一次变换：

- `000000` 权威 WAV 头静音约 0.448 秒，单次脚本应平移 `+0.052s`；现存 CTC 净移为 `+0.208s`，恰好累计 4 次。最后一次 padded 音频与权威 WAV 在映射保留区逐样本一致，但尾部删去 6,720 samples；
- `000001` 为 `+0.244×4=+0.976s`，`001000` 为约 `+0.201333×4=+0.805332s`；
- `005000`、`010000` 累计约 3 次；`020000`、`030000`、`040000`、`050000`、`053999` 累计约 2 次；
- 负偏移样本 `036059` 的 manifest 首词为 0.93 秒，现存值已被 `max(0, t+offset)` 压成 0；累计净移 `-1.133334s`，信息不可逆。`036018` 同样发生两次负移并删头/删尾；
- `000000` 现存 pause 终点可到 12.868 秒，而最后一次 padded WAV 只有 12.392 秒，证明 CTC 累积次数和音频变换次数不同。

旧 `/mnt/nvme3/mfa_workspace/ctc_pretg/manifest.json` 大小为 251,014,502 bytes，SHA-256 为
`127bb36d2645c2072c2822342ed3e34efdb8dea8e3e298e5aead5baaf5d2cabc`。它没有输入/CTC hash、head/tail、offset 或 new-duration；其中指向的 flat cache 音频已不存在，后来同路径 cache 又被替换为其他数据源。因而即使部分正偏移样本能从几何关系求出累计净移，也无法恢复每次顺序、被截断字段、被裁音频或最初声学输入身份。

该证据使“全量 53,998 fresh acoustic rerun、禁止复用/转换旧 lineage”从保守选择升级为确定的生产要求。

---

## Case 96: v4 首版校验器与真实 CTC 产物契约不一致

### 主代理接收前复核发现

Sol high 已冻结“token 的 start/end 分别单调、允许相邻 token 合法重叠；punct 保留原顺序并允许重叠”的安全契约。但 v4 首版准备器与独立验证器都用“当前 start 不得早于上一个 end”判断 token/punct，会把正常 CTC overlap 误拒为时间错误，与 `pipeline_utils.load_ctc_token_entries` 的已冻结语义相矛盾。

真实 `ctc_prealign.py --all-gpus` 还会对每个成功 stem 生成 `<stem>_ref.txt`，分片合并时将其移入主 rerun 目录。首版 finalize 的 exact namespace 却只允许六件套、`manifest.json` 、`summary.txt` 和 `.ctc_normalized`，因此一次完全成功的真实 rerun 也必然被 finalize 拒绝。反向地，新 writer 只在检测到标点时写 `_punct.json`，空标点样本可能缺失六件套成员；必须在正式 GPU 前用源文本盘点和合成 fixture 确认，不能等 53,998 条跑完才发现。

独立 verifier 另有以下“自证”缺口：

1. 未用磁盘上的 `prepare_manifest.json` 重算并校验 `prepare_manifest_sha256`；
2. standalone 路径不重新扫描权威源目录，未把 `authoritative_audio/reference.path` 精确绑定到该 stem 在 `/mnt/Raw/新版合成英文数据` 中的实际路径；只要 evidence 指向另一份与运行副本自洽的文件就可能通过；
3. 未重算权威 inventory/taxonomy，也未在 standalone 路径固定 54,000/53,998/2/0 和两个精确 missing-reference stem；
4. 未拒绝 audio/reference/ready/rerun 目录的多余文件或非普通项，未证明 source/run-local dictionary 哈希始终相同；
5. 只比较 `.lab`/tokens/TextGrid 词面，没有利用 rerun `_ref.txt` 与运行副本参考文本证明此 stem 的声学任务使用了对应权威参考；
6. 未解析 rerun `manifest.json` 的精确 stem 集合、`audio`/`textgrid`/`lab` 路径、duration 和 `_words`，因而没有用生成器自身记录证明每个 CTC bundle 确实指向 run-local 权威音频副本；
7. 顶层 rerun 命令虽未带 `--overwrite`，但 `ctc_prealign.py --all-gpus` 会在内部给每个子进程强制追加 `--overwrite`，且 shard 目录以 `exist_ok=True` 打开。这与本任务“不覆盖、不复用失败根”的操作契约冲突，还可在意外重启时把新旧 shard 混合；
8. `finalize` 写出 ready evidence 后未自动调用独立 verifier，不能把“evidence 文件已存在”当成“独立验证已通过”。

### 修复与验收要求

1. token 要求 start/end 各自有限、正时长、分别单调且允许 overlap；punct 要求每个 interval 有效、在音频边界内并保留原数组顺序，不以区间重叠作为拒绝条件。准备器与独立 verifier 必须有一致但不共享实现的 fixture。
2. 对真实 writer 冻结一个唯一 namespace：要么让 rerun 明确不生成 `_ref.txt`，要么将其作为每 stem 必需的 rerun 证据输入，精确校验其内容/哈希等于 `reference_view/<stem>.txt`，但最终 `ctc_ready` 仍只包含标准六件套。
3. writer 对无标点样本也必须写入合法的空数组 `_punct.json`；增加无标点、token overlap、punct overlap 回归。
4. v4 rerun 必须有 fresh-output 前置门禁：目标和 shard 一旦存在就在加载模型/GPU 前失败；移除 all-gpus 内部强制 `--overwrite`，失败后只能使用新 run root，不在原根上续跑。
5. 独立 verifier 从固定/显式 `source-dir` 独立扫描 WAV/TXT，重算 exact inventory/taxonomy，按 stem 比较权威路径与哈希，重算 prepare manifest 哈希，解析 rerun manifest 并把其 audio/textgrid/lab/duration/words 与运行副本和六件套交叉核对，再重跑所有 exact namespace/dictionary/copy 校验。
6. `finalize` 只在全量预检查和复制完成后写 evidence，然后必须调用独立 verifier；任一缺口非零停止，不得配置 pin，不得启动 MFA。

### 第二轮主审仍未闭环的问题

Terra 第二轮已修复 `_ref.txt`、rerun manifest、fresh-output 和权威源重扫，但 root 逐行复核后仍阻断生产：

1. punct 仍被强制要求 start/end 分别单调。冻结契约只要求 punct 逐项有效、保留数组顺序并允许重叠；嵌套 overlap 可出现“后一项 end 早于前一项 end”，当前 fixture 只覆盖了 end 仍上升的重叠，没有真正证明契约。
2. verifier 虽重扫了源 stem，却没有重算/比较 `inventory_sha256`；`prepare_manifest.json` 中的 inventory、missing/txt-only、axis/padding、action counts、`prepared_files_sha256`、exact prepared copy mapping 和 `rerun_command` 也没有全量绑定。
3. standalone verifier 仍未执行 audio_view、reference_view、dict、ctc_ready、ctc_rerun_output 的 exact regular-file namespace 检查，多余文件或目录可通过；也未重验 audio/reference/dictionary/rerun→ready 为普通 copy 而非 hardlink inode alias。
4. source/run-local dictionary 只各自自洽，没有要求 path 指向冻结的项目词典且 size/hash 完全相等；`.ctc_normalized` marker 内容和 summary 计数也未核对。
5. 所有边界都共用 3 ms 容差，违反 Sol “音频轴/序列化容差与 lexical-start 3 ms 分离”的明确要求。TextGrid 全局/tier domain 应使用 WAV sample/六位小数精度，seconds↔milliseconds 使用约 0.51 ms，只有词起点交叉比较可使用 3 ms。
6. 第二轮测试未覆盖已要求的 extra namespace、manifest tamper、authority substitution、dictionary drift/hardlink 和嵌套 punct overlap，所以“测试通过”尚不等于 Case 96 验收完成。

真实源文本的附加只读盘点已确认：53,998 个 txt 全部非空、均含管线可识别标点、无日语假名，且原文字节全部恰好等于生成器的 `strip()+"\\n"` `_ref.txt` 规范。因此本数据可安全做严格 `_ref.txt` 字节等值比较，但 writer 仍应为一般无标点输入写 `[]`。

第三轮实现虽已加入 inventory digest、marker/summary、词典等值和大部分 namespace 检查，但主审发现代码与“已完成”报告仍不一致：

- standalone verifier 没有对 `ctc_rerun_output` 调用 exact-directory gate，因而 rerun extra 仍可通过；
- 它没有重算 `prepared_files_sha256`、没有核对 107,997 条 prepared copy 的精确顺序/路径/evidence/inode，也没有核对 `rerun_command`；
- 准备器的二次 `_v4_verify_copies` 未再拒绝准备后被替换成 hardlink 的目标；
- 实际执行的 `v4_main` 仍只有重排 stem、nonfinite token 和 malformed TextGrid 等少数负例，并未实作 extra namespace、prepared-manifest tamper、authority substitution、dictionary drift/hardlink、marker/summary 错误等已要求 fixture。

此外，rerun manifest 的 `duration_s` 仍使用 3 ms 比较，而该字段来自同一 WAV 的未截断 Python float，应使用 axis/serialization 容差；3 ms 仍只属于 lexical start 交叉比较。

### 修复后的合成与真实只读验收

第四轮实现及 root 补充回归已完成以下闭环：

- 准备器专项 16 项全部通过，实际覆盖 taxonomy、fresh/no-overwrite、token/punct overlap、空 punct、authority、重排/nonfinite、prepared manifest、audio/rerun extra、跨 stem authority、marker、summary、dictionary drift、ASR command identity、hardlink 和 malformed rerun；
- runner/import 专项 12/12 通过，覆盖 v4 evidence/tamper、full/canary 精确分母、普通 copy/inode、direct verifier、无 padding 步骤、无 padded denominator、active audio shrink 和真实配置 placeholder 阻断；
- ASR Python 和 model path 已从 manifest 自证改为独立 verifier 的显式期望值；rerun manifest duration 改用 axis 容差；`_ref.txt` 也拒绝 inode alias；
- scoped Python compile 通过；配置在 evidence/taxonomy SHA 仍为 placeholder 时正确非零退出，且退出前未创建 workspace。

2026-08-06 执行了修复后的真实 v4 只读 `inspect --require-expected-counts`，结果成功：

- WAV 54,000；TXT 53,998；权威 stem 53,998；两个精确 missing-reference；txt-only 0；
- `action_counts={"acoustic_rerun": 53998}`，53,998 行 taxonomy 全部为 `legacy_audio_provenance_unbound -> acoustic_rerun`，stem 精确排序唯一；
- `taxonomy_sha256=163587d39963f2f3441a4cb99315b4e4efe5d28832a6566f034c36ab8373193c`；
- `inventory_sha256=888db5297cdc1f70dd0a86c9d3cd2678dcaf332a247df481a04bb52fe44e814f`，root 独立重算相等；
- 完整报告 `/tmp/hecheng_english_v4_inspect_20260806.json` 为 23,943,369 bytes，SHA-256 `7f3306e27991d362d31ea54778c3ec2a49158ab2ec0c2421997f4a1478de0daa`。

此检查之后 `/mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0` 仍不存在，未启动 GPU。

状态：Case 96 代码与真实只读 inventory 门禁已通过；生产仍冻结到 Sol high 终审和真实小样本声学 canary 通过。

---

## Case 97: CTC manifest 在英文归一化前冻结，最终 bundle 与 provenance 自相矛盾

### Sol high 终审发现的 P0

`ctc_prealign.py` 当前在主推理循环中先从归一化前的词结果构造 manifest entry，并在约第
1774–1776 行写出 `manifest.json`；真正会修改产物的 `_normalize_punct`、
`_normalize_numerals`、`_normalize_ria` 和 `_normalize_english` 却到约第 1856–1860 行才执行。
其中 `normalize_english_tokens.py` 的英语合并逻辑会重写 `.lab`、`_tokens.jsonl` 和
`.TextGrid`，可能改变词面、词数和词边界。

因此，一个声学推理本身成功且最终六件套合法的 English stem，仍可能留下归一化前的
`manifest.n_words` / `manifest._words`。v4 finalize 与独立 verifier 会把 manifest 与最终
tokens/TextGrid 逐 stem 交叉校验，遇到实际英语片段合并时必然拒绝整批。当前
`.ctc_normalized` marker 的验证只覆盖最终 bundle，没有同步证明 manifest 已刷新，不能封闭该缺口。

多 GPU 路径也受影响：父进程合并各 shard 的 manifest；如果子进程落盘的是归一化前 manifest，
父进程只会忠实合并过期证据，无法在顶层恢复正确 provenance。

### 风险与停止条件

1. 真实 6-stem 声学 canary 可能在 finalize 阶段失败；即使恰好未触发合并，也不能证明全量
   53,998 stem 不会触发。
2. 若放宽 v4 manifest 校验，会把生成器记录与最终交付物不一致的问题隐藏进 strict-ok 证据链，
   违反“通过结果无机器可检测错误”的验收目标。
3. 在修复并由回归强制触发一次英语合并前，不得启动声学 canary、生产 prepare 或全量 GPU。

### 冻结的修复与验收方案

1. 所有 normalizer 成功后，从最终 `_tokens.jsonl`（并与最终 TextGrid/WAV 校验）重建每个
   manifest entry 的 `_words`、`n_words`、`n_punct` 和 `duration`；不得保留归一化前词表。
2. 只有最终六件套全量验证成功后才原子写入最终 `manifest.json`，marker 的生成/验证顺序必须
   保证 manifest 与 bundle 同属一次成功事务；失败不得留下可被误认成完成态的最终 manifest。
3. all-GPU 子进程各自写出归一化后的最终 manifest，父进程仅合并这些最终 shard manifest，
   并继续执行顶层全量 bundle/namespace 校验。
4. 增加强制触发英语或 `ria` 合并的无 GPU 回归，断言归一化后 manifest 的词面、词数、start
   与最终 lab/tokens/TextGrid 一致；另验证 manifest 只在后处理成功后出现或被替换。
5. 修复后重新执行 Case 96 的 16 项准备器专项、12 项 runner/import 专项、既有 provenance/
   reference/strict/tier 回归、compile 与 scoped diff check，并交回 Sol high 做最终 GO/NO-GO。

状态：代码已实施（`_rebuild_final_manifest` 在所有 normalizer 之后从最终 `_tokens.jsonl` 重建 manifest）；已通过 `_case_manifest_rebuilt_from_final_tokens` 单元测试；待 GPU canary 集成验证。

---
## Case 98: encoder 60ms 网格时长冒充权威 WAV 时间轴

### Sol high 终审发现与真实数据证据

`ctc_prealign.py` 当前用 encoder 输出长度 `elens` 计算
`duration_s = (total_frames - 4) * 0.06`，并把这个 60ms 网格值用于 TextGrid 全局域、
punct 边界和 `manifest.duration`。它描述的是模型帧轴近似值，不是该 stem 的实际 WAV
`frames / sample_rate`。

v4 准备器与独立 verifier 则有意把最终音频轴冻结为未修改的 authoritative WAV，并要求
TextGrid domain、tier domain 与 rerun manifest duration 在 1µs 轴容差内等于 WAV header 时长。
两边契约不能同时成立。

Sol 对真实源目录前 200 个 WAV 做只读 header 抽查，其中 122/200 的时长不落在 60ms 网格；
常见值包括 8.80、9.44、11.84 秒，与最近 60ms 网格相差 20ms。多数样本有权威参考文本，
包括 `036001`、`036003`、`036006` 等。因此这不是极端理论分支：fresh CTC 即使模型推理、
词面与时间点都成功，也会因全局域/manifest 与 WAV 不同而被 v4 finalize 确定性拒绝。

### 风险与停止条件

1. 将 v4 的 1µs 轴容差放宽到模型帧误差会破坏“最终 bundle 精确绑定 authoritative WAV”
   的证据契约，并允许区间真实越界，不能作为修复。
2. 仅改 manifest duration 不够；TextGrid 全局/tier domain、punct 及所有 lexical/pause endpoint
   都必须在同一 WAV 轴上有效。
3. 在 writer 明确使用 run-local WAV header 且非 60ms 时长 fixture 通过前，不得启动声学
   canary、生产 prepare 或全量 GPU。

### 冻结的修复与验收方案

1. 每 stem 从实际 run-local WAV header 读取 `frames / sample_rate`，将其作为唯一
   authoritative duration，传入 TextGrid、punct 和 manifest writer；禁止用 encoder grid
   填写容器全局域或 provenance duration。
2. 对模型生成的 token/pause interval 定义一致的边界策略：浮点非有限、负时长或 start 超出
   WAV domain 必须失败；仅由 encoder 最后一帧量化造成的 endpoint 超出可在写盘前明确裁到
   WAV duration，裁后仍须保持正时长、各字段单调和既有 overlap 语义。不得靠 verifier 放宽容差。
3. 所有最终 bundle 在写 marker/manifest 前按 WAV duration 再验证；多 GPU shard 同样逐 stem
   使用自己的 WAV header，不得使用 batch/global 近似时长。
4. 增加至少 1.00s 与 9.44s 的非 60ms 网格 WAV fixture，走 writer/最终 manifest 刷新到 v4
   finalize/独立验证的关键路径；断言 TG/tier/manifest domain 精确等于 WAV，所有 endpoint 在域内。
5. 与 Case 97 的后归一化 manifest 修复共同回归，并由 Sol high 复核后才解除 canary 冻结。

状态：代码已实施（`_wav_duration_s` 使用 wave 模块读取 WAV header 时长，`_clamp_words_to_wav_axis` 强制边界策略）；已通过 `_case_wav_duration_from_header_not_encoder_grid` 和 `_case_clamp_words_to_wav_axis_enforces_boundaries` 单元测试；待 GPU canary 集成验证。

## Case 99: 模型路径已固定但实际模型文件树未绑定

### Sol high provenance 审计发现

v4 准备器和独立 verifier 当前会检查预期 rerun command 中的 ASR Python 与 model path，
但 `ctc_prealign.py` 的最终 manifest 没有记录模型目录的逐文件身份、模型树 digest、实际 argv、
运行词典 digest 或输入 stem 集合 digest。现有证据证明的是“命令计划指向该路径”，并不能证明
推理进程加载时该路径包含哪一版模型。

因此，只要在相同 model path 下替换例如大型 `model.pt`，planned command、准备 manifest、
CTC 六件套及当前独立验证仍可能全部通过。词典和权威源数据的复制/哈希链已经闭环，但模型
这一声学决定因素尚未达到同等级 provenance。

### 风险与停止条件

1. 不能仅把 model path 字符串或目录 mtime 当作模型身份；同路径可变内容会让两次运行不可区分。
2. 只在 prepare 时计算 digest 仍不够；必须由实际 CTC 进程写运行 receipt，再由 finalize/独立
   verifier 与 prepare 冻结值交叉核对，才能证明计划与执行相同。
3. 该项为 provenance P1，不是 Case 97/98 的即时功能崩溃原因；但若在真实 prepare 后才修复，
   已准备或已生成的 root 缺少不可追补的执行时证据，应废弃并使用新 root。因此生产 prepare
   与声学 canary 的最终验收必须先纳入此契约。

### 冻结的修复与验收方案

1. prepare 对 model directory 做 exact regular-file tree 清单，至少记录每个文件的相对路径、
   size、SHA-256，并计算确定性 tree digest；拒绝 symlink、目录逃逸、重复/乱序或其他非普通项。
2. `ctc_prealign.py` 由实际进程在成功后写原子 run receipt，绑定规范化实际 argv、ASR Python、
   model path/tree digest、dictionary path/digest、输入 stem 精确排序唯一集合及其 digest；多 GPU
   须证明所有 shard 使用同一模型/词典身份，并由父进程生成精确合并 receipt。
3. finalize 与独立 verifier 重算当前 model tree，核对 prepare 冻结值、CTC run receipt 和实际
   文件树三者一致；同时核对 rerun manifest stem 集与 receipt input/success stem 集无缺失或额外项。
4. 增加模型文件内容替换但路径不变、symlink/extra/non-regular 文件、argv 漂移、词典漂移、
   shard receipt 不一致和 stem digest 篡改的负例。任何一项必须在进入 MFA 前非零失败。

状态：代码已实施（2026-08-07）。`pipeline_utils.py` 新增 `compute_model_tree_digest`、`write_ctc_run_receipt`、`write_ctc_shard_receipt`；`ctc_prealign.py` 启动时计算 model tree digest，成功后写 run receipt，all-GPU 预检验证 shard receipt；`prepare_hecheng_english_ctc_ready.py` prepare 时冻结 model tree 到 prepare_manifest.json；`verify_hecheng_english_ctc_ready_v4.py` 交叉核对 prepare 冻结值、receipt 值和实际文件树；`verify_strict_ctc_ready_import.py` 新增 7 个负例测试。待 GPU canary 集成验证。

---

## Case 100: blank-run pause 坐标包含 query frames，整体偏移约 240ms

### Sol high 逐帧审计发现

`ctc_prealign.py` 的 blank-run 检测目前在包含 4 个 query frames 的 `raw_y` 上记录 `(s, e)`；
后续构造 pauses 时却直接使用 `s * 60ms` / `e * 60ms`，没有像 lexical forced-alignment
路径那样切到 speech slice 或减去 `QUERY_FRAMES`。因此 pauses tier 的声学空白坐标会整体向后
偏移 `4 * 60ms = 240ms`，尾部还可能越过实际 WAV duration。

这类 pause 即使随后被简单裁到 WAV 末端，也只隐藏尾部越界，无法纠正前面所有停顿的 240ms
系统偏移；它会污染 TextGrid pauses tier、manifest pauses 及下游停顿判断。

### 修复与验收方案

1. blank-run 必须在去除 query frames 的 speech slice 上检测；等价实现只能先与
   `[QUERY_FRAMES, total_frames)` 相交，再将坐标减去 `QUERY_FRAMES`，不得只在 writer 末端减
   一个常数而保留跨 query/speech 边界的伪 run。
2. 转到秒轴后使用 Case 98 的 authoritative WAV duration gate：非有限、负坐标、start 超域、
   非正时长均失败；仅不超过一个 encoder frame 加轴 epsilon 的末端量化越界可裁到 WAV 末端，
   更大越界必须失败。pause 裁剪后重算 `duration_ms` 和 label。
3. 增加精确 fixture：在 query frame 与 speech frame 交界处构造 blank run，断言输出 pause 从
   speech 轴 0 开始而非 0.240 秒；另覆盖尾部小幅可裁、>60ms 拒绝和裁后零长处理。
4. 与 Case 97/98 的 final manifest、非 60ms WAV 与 marker 事务回归共同通过后，再交 Sol high
   终审。

状态：代码已实施（`blank_runs_speech` 减去 `QUERY_FRAMES`，过滤跨 query/speech 边界的 run）；已通过 `_case_blank_run_subtracts_query_frames` 单元测试；待 GPU canary 集成验证。

---

## Case 101: all-GPU 父合并不是严格事务，可拼出混合或不完整完成态

### 独立 Terra high 与 Sol high 审计发现

`ctc_prealign.py --all-gpus` 当前收集 shard 文件时，如果主输出中同名目标已存在会静默跳过；
读取 shard `manifest.json` 失败只打印 warning 后继续；没有在移动前证明 shard stem 集互斥、
文件 namespace 精确、manifest stem 与文件 stem 一一对应。shard 自己生成的 `.ctc_normalized`
也可能作为普通文件在第一个 shard 阶段提前进入主根。

父进程随后还会先写 merged manifest/summary，再做有限 bundle 校验和 manifest 重建。这样一旦
发生重复 stem、旧目标碰撞、坏 manifest、部分 shard 缺文件或父进程中途失败，主根可能同时
包含不同来源的 artifacts 和看似完成的 marker/manifest，fresh-output 的入口检查不能替代
合并过程内部的事务完整性。

### 修复与验收方案

1. 父进程在修改最终 namespace 前，解析并严格验证每个 shard：精确允许文件集、最终 manifest、
   marker 内容、请求 stem 集/成功 stem 集、普通文件类型；任何 parse warning 升级为非零失败。
2. 汇总阶段先在内存中证明所有 shard stem 集互斥、并集等于全局预期 stem，文件名无重复、
   每个文件只属于其 manifest stem。主目标存在任何同名项必须失败，禁止静默 skip。
3. shard marker 不得复制到主根；父级只在全部 artifact 合并、路径重写、全量 bundle/domain/
   namespace 校验、最终 manifest 与 summary 原子写成功后，最后原子发布唯一父 marker。
4. 失败根不可续跑；任何阶段失败均不得留下父级 final marker。生产操作必须换新 run root，
   不得以 `--overwrite` 或重复启动修补。
5. 增加重复 stem、同名文件碰撞、坏 shard manifest、缺 artifact、多余 artifact、shard marker
   提前合入及父验证失败的无 GPU fixture；均须证明非零失败且无父 marker/final manifest。

状态：代码已实施（all-GPU preflight 验证 shard stem 互斥、namespace 精确、manifest/summary/marker 校验；父 marker 在所有 artifact 合并和全量验证之后最后写入）；待 GPU canary 集成验证。

---

## Case 102: reference-only `--no-nvv` 仍允许 ASR 内容污染 required sidecar

### 复现与根因

在 reference-only forced-alignment 模式中，`--no-nvv` 旧实现只关闭 blank-frame NVV bias；
它没有在自由解码使用的 logits 副本上屏蔽 NVV ID 区间，因此原始 CTC logits 的自然 argmax
仍可能产生 ASR-added NVV。更严重的是，CTC writer 曾从自由 `text_asr` 写入
`_text_raw.txt`，再由该文本生成 `_text_cn.txt`，使 ASR-only NVV、CJK、英文或标点进入
required sidecar，违背 reference 是唯一内容权威的契约。

### 风险与停止条件

1. reference 不含 NVV 时，required artifacts 可能凭 ASR argmax 增加 NVV；reference 已有
   NVV 时也可能重复或重排。
2. 自由 ASR 的 CJK、英文和标点可能改变 `_text_raw/_text_cn`，而 `.lab`、tokens、TextGrid
   和 punct 又分别依赖不同来源，最终 bundle 发生不可见内容漂移。
3. 仅检查 argv 中存在 `--no-nvv` 或生成器自报 reference-only 不足以证明内容隔离；在修复、
   fault tests 和独立 semantic verifier 全部通过前，禁止 English canary、MFA 或生产 rerun。

### 冻结的修复与验收方案

1. `--no-nvv` 时仅在自由解码 logits clone 上将 NVV ID 区间设为 `-inf`；forced alignment
   继续使用干净原 logits 和 reference target，保证 reference NVV 仍可对齐。
2. required `_ref.txt`、`.lab`、`_tokens.jsonl`、TextGrid words、`_punct.json`、
   `_text_raw.txt` 和 `_text_cn.txt` 全部从 canonical reference 或其确定性 transform 生成；
   自由 ASR 结果只能作为 diagnostic，不得成为 required artifact 的词面来源。
3. 独立 verifier 从 reference 重算 lexical/NVV/标点序列，并拒绝 extra/missing/reordered
   content、`0 pinyin`、非法时间轴和集合不守恒；非 reference-only 模式仍保留 ASR-added NVV
   能力。
4. 增加 mock logits、monkeypatch `text_asr`、reference NVV、sidecar extra NVV、argv
   缺失/重复和 shard 未继承等正负测试；所有违规必须非零，且 filtered 不得掩盖全局集合错误。

状态：代码已实施（`_free_decode_logits` 在 `reference_only=True` 时将 NVV ID 区间设为 `-inf`）；已通过 `_case_reference_only_masks_nvv_ids_in_free_decode` 单元测试；待 GPU canary 集成验证。

---

## 验证执行记录（2026-08-07）

### Phase A: 代码修复

| 修复 | 案例 | 文件 | 状态 |
|------|------|------|------|
| hard_integrity 非零退出码 | Case 74 | postprocess_textgrids.py | ✅ |
| stem 集合回退分母 | Case 77 | run_pipeline.py | ✅ |
| marker v4 内容身份 | Case 82 | pipeline_utils.py, ctc_prealign.py, run_pipeline.py, recover_ctc_labs.py | ✅ |

### Phase B+D: 验证套件结果（全部通过）

| 验证器 | 测试数 | 结果 |
|--------|--------|------|
| verify_reference_authority.py | 24 | 24/24 OK |
| verify_tier_discontinuity.py | 3 | 3/3 OK |
| verify_strict_ok.py (self-test) | 30 | 30/30 OK |
| verify_strict_ctc_ready_import.py | 12 | 12/12 OK |
| verify_prepare_hecheng_english_ctc_ready.py | 17 | 17/17 OK |
| **总计** | **86** | **86/86 OK** |

### Phase D: 新增测试覆盖

| 新增测试 | 覆盖案例 |
|---------|---------|
| `_case_wav_duration_from_header_not_encoder_grid` | Case 98 |
| `_case_clamp_words_to_wav_axis_enforces_boundaries` | Case 98 |
| `_case_blank_run_subtracts_query_frames` | Case 100 |
| `_case_manifest_rebuilt_from_final_tokens` | Case 97 |
| `_case_ctc_bundle_rejects_zero_duration_tokens` | Case 80 |
| `_case_merge_ria_tokens_produces_single_token` | Case 81 |
| `_case_protect_ria_fragment_merge` | Case 81 |
| `_case_reference_only_masks_nvv_ids_in_free_decode` | Case 102 |
| `_case_v4_marker_encodes_stem_count_and_manifest_digest` | Case 82 |

### Phase E: P0 代码审计结论

| Case | 代码实施 | 单元测试 | 状态 |
|------|---------|---------|------|
| 97 | ✅ `_rebuild_final_manifest` 在所有 normalizer 之后重建 | ✅ | 待 GPU canary |
| 98 | ✅ `_wav_duration_s` + `_clamp_words_to_wav_axis` | ✅ | 待 GPU canary |
| 99 | ❌ 代码不存在 | N/A | 仅文档方案 |
| 100 | ✅ `blank_runs_speech` 减去 `QUERY_FRAMES` | ✅ | 待 GPU canary |
| 101 | ✅ all-GPU preflight + parent marker 最后写 | 代码审计 | 待 GPU canary |
| 102 | ✅ `_free_decode_logits` reference_only 屏蔽 NVV | ✅ | 待 GPU canary |

### 仍需 GPU 验证的案例

Cases 80（单条 CTC 重跑 051809）、81（ria 三方原子同步集成）、85–86（配置+生产数据验证）、99（代码已实施，待 GPU canary 集成验证）。

### 已实施的非 GPU 案例（2026-08-07）

- Case 76 (R7): strict MFA TextGrid validator 替换字符串匹配
- Case 83 (R7): strict MFA TextGrid validator + 进程异常捕获
- Case 99 (R5): model tree digest + CTC run receipt + shard receipt

### 前置条件

生产 canary 执行前必须完成：
1. ~~Case 99 模型树绑定代码实施~~ ✓ 已完成
2. 补齐 Case 86 缺失参考文本（036000）或显式隔离
3. 生成冻结的 canary stems 文件（基于 `test_data/canary_cases_69_102.txt` 设计，使用生产数据实际 stem ID）
4. Sol high 对 Case 96 v4 prepare 的最终 GO

---

## Case 103: `run_pipeline.py` 未支持 `--all-gpus` 多卡 CTC 推理（待实施） (all_gpus_pipeline_support)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py, scripts/ctc_prealign.py, configs/hecheng_ria_fresh.yaml

### 现象

`ctc_prealign.py` 已实现 `--all-gpus` 多卡并行模式，但 `run_pipeline.py` 的 `step_prealign` 没有传递该参数。
当前 54,000 条 CTC 推理只能用单卡（`--device cuda:0`），56 分钟跑完；其余 7 张 GPU 空闲。

### 修复方案

1. `configs/hecheng_ria_fresh.yaml` 中 `ctc_prealign` 增加 `all_gpus: true` 配置项
2. `run_pipeline.py::step_prealign` 读取 `pc.get("all_gpus", False)`，为 `True` 时追加 `--all-gpus` 到 `prealign_args`

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `configs/hecheng_ria_fresh.yaml` | `ctc_prealign` 下增加 `all_gpus: true` |
| `scripts/run_pipeline.py` ~line 838 | `if pc.get("all_gpus", False): prealign_args.append("--all-gpus")` |

状态：已实施。`configs/hecheng_ria_fresh.yaml` 增加 `all_gpus: true`；`run_pipeline.py::step_prealign` line ~845 传递 `--all-gpus`。

---

## Case 104: MFA Popen OSError 引用未初始化变量 (mfa_popen_uninit)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`_run_mfa_sharded` 中 Popen OSError handler（~line 1474-1483）引用 `_return_codes` 和 `_failed`，
但这两个变量在 Popen 循环结束后才初始化（原 line 1491-1492）。若 `subprocess.Popen()` 抛出
OSError（如可执行文件缺失、资源耗尽），会先触发 `UnboundLocalError` 掩盖真实错误。

修复前：OSError → NameError: name '_return_codes' is not defined
修复后：OSError → 正确记录 `os_error:<errno>`，保留 shard 现场

### 根因链

1. `_run_mfa_sharded` Popen 启动循环（~line 1434）
2. OSError handler 引用 `_return_codes[_si]` 和 `_failed.append(_si)`（~line 1476-1477）
3. 变量初始化位于 wait 循环之前（~line 1491-1492）— 在引用之后
4. Python 在函数内赋值即视为局部变量，引用未赋值局部变量 → `UnboundLocalError`

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` ~line 1433 | `_failed` 和 `_return_codes` 初始化移至 Popen 循环之前 |
| `scripts/run_pipeline.py` ~line 1491 | 移除重复初始化（避免覆盖 OSError 收集的数据）|

### 验证方法

```python
# 模拟：mock subprocess.Popen 抛出 OSError
# 预期：shard 记录为 os_error，循环继续，不抛 NameError
```

状态：已修复。

---

## Case 105: 父进程 MFA jobs 计算值未传递到子进程 (mfa_jobs_contract_mismatch)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`run_batch`（~line 1775-1803）根据 CPU 核数、batch 大小和并行度计算 `_effective_mfa_jobs`，
但只存储在内存 `args._effective_mfa_jobs` 和 in-memory config dict 中。子进程通过
`subprocess.run` 启动 `run_pipeline.py` 时，命令中不包含 `--mfa-jobs` 参数，
子进程从磁盘 YAML 读取原始配置值，父进程的自动缩放计算被丢弃。

修复前：子进程使用 config YAML 中的 `mfa.num_jobs: 64`（可能超过安全限制）
修复后：子进程命令包含 `--mfa-jobs {calculated_value}`，尊重父进程的资源预算

### 根因链

1. `run_batch` 计算 `_effective_mfa_jobs` → 存到 `args._effective_mfa_jobs`
2. `_execute_staged` → `process_worker` → `_process_one_batch` 构建子进程命令
3. 子进程命令未包含 `--mfa-jobs` → `run_pipeline.py` 重新从 YAML 读取 → 使用原始配置值
4. 父进程的 CPU 预算、batch 上限约束全部失效

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` `_process_one_batch` | 接受 `mfa_num_jobs`/`mfa_en_num_jobs` 参数，追加到子进程命令 |
| `scripts/streaming_pipeline.py` `_execute_staged` | 签名增加 `mfa_num_jobs`/`mfa_en_num_jobs`，传递到 `_process_one_batch` |
| `scripts/streaming_pipeline.py` `run_batch` | 计算 `_effective_mfa_en_jobs`，传递到 `_execute_staged` |
| `scripts/streaming_pipeline.py` `_run_cpu_phase` | 接受并传递 MFA jobs 参数 |
| `scripts/streaming_pipeline.py` `StreamingPipeline` | 类增加 `mfa_num_jobs`/`mfa_en_num_jobs` 属性 |

### 验证方法

```bash
# 运行 dry-run / --plan-json，检查子进程命令包含 --mfa-jobs N
python scripts/streaming_pipeline.py --config configs/batch_all.yaml --plan-json /tmp/plan.json
python -c "import json; plan=json.load(open('/tmp/plan.json')); assert any('--mfa-jobs' in c for c in plan['commands'])"
```

状态：已修复。

---

## Case 106: 硬编码 --force --overwrite 覆盖配置禁止 (force_overwrite_hardcoded)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py, scripts/launch_8gpu.py

### 现象

`streaming_pipeline.py` 和 `launch_8gpu.py` 中所有子进程命令无条件包含 `--overwrite --force`。
`configs/hecheng_english_mfa.yaml` 明确禁止这些参数，但硬编码导致配置禁止无效。
子进程会覆盖已有输出和 workspace，破坏 strict 不变量。

修复前：所有子进程强制 `--overwrite --force`，配置 `allow_overwrite: false` 被忽略
修复后：子进程命令根据 config `pipeline.allow_overwrite` / `pipeline.allow_force` 决定是否追加

### 根因链

1. `_process_one_batch`、`_run_gpu_phase`、`_run_cpu_phase`、`StreamingPipeline._process_batch` 硬编码 flags
2. `launch_8gpu.py` 同样硬编码
3. 无 CLI 或 config 读取逻辑来覆盖
4. strict 配置的禁止声明被绕过

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` | 添加 `--no-overwrite`/`--no-force` CLI flags；在 `run_batch` 中读取 config 解析策略；所有子进程命令条件化 |
| `scripts/launch_8gpu.py` ~line 91 | 读取 config `pipeline.allow_overwrite`/`allow_force`，条件化 flags |

### 验证方法

```bash
python scripts/streaming_pipeline.py --config configs/hecheng_english_mfa.yaml --no-overwrite --no-force --plan-json /tmp/plan.json
python -c "import json; plan=json.load(open('/tmp/plan.json')); assert not any('--overwrite' in c for c in plan['commands'])"
```

状态：已修复。

---

## Case 107: all-GPU shard 合并存在 TOCTOU 竞态窗口 (ctc_shard_merge_toctou)

**日期**: 2026-08-07
**涉及文件**: scripts/ctc_prealign.py

### 现象

`ctc_prealign.py` all-GPU merge 阶段有 preflight 碰撞检查（~line 1356-1361）和
每文件移动前二次检查（~line 1382），但两次检查和 `shutil.move` 之间没有原子保护。
外部进程可在窗口期写入目标目录导致覆盖。

修复前：preflight 检查通过 → 窗口期 → move（可能覆盖外部写入）
修复后：增加 `.merge_lock` sentinel 文件，合并前获取，完成后释放

### 根因链

1. preflight 碰撞检查遍历所有文件
2. 无文件级或目录级锁
3. `shutil.move` 在检查之后执行
4. 外部并发写入可导致覆盖

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/ctc_prealign.py` ~line 1362 | merge 前创建 `.merge_lock`；已存在则拒绝重复合并 |
| `scripts/ctc_prealign.py` ~line 1478 | `try/finally` 确保 lock 释放 |

### 验证方法

代码审计：确认 merge 代码块被 `try/finally` 包裹，lock 在异常路径也会释放。

状态：已修复。

---

## Case 108: 批量上传共享目录导致跨批次文件覆盖 (batch_shared_upload_dirs)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`_upload_one_batch` 将多个 batch 的输出直接合并到同一个 `output/`、`filtered/` 目录。
同一数据集的 batch_0000 和 batch_0001 上传到相同的 NAS 路径，`rsync -a` 合并时
同名文件被后上传的 batch 覆盖。`_upload_worker` 更严重：所有数据集的所有 batch
合并到单一 `output` 目录。

修复前：多个 batch 写入同一 NAS 目录，文件合并结果取决于上传顺序
修复后：每个 batch 上传到隔离的 `.staging/batch_XXXX/` 目录，合并由显式 reducer 步骤完成

### 根因链

1. `_upload_one_batch` 目的地 = `nas_output_root/ds_name/output`（所有 batch 共享）
2. `_upload_worker` 目的地 = `nas_output_root/output`（所有数据集共享）
3. 不同 batch 产生的同名 TextGrid/lab 文件相互覆盖
4. 最终结果取决于最后完成的 batch

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` `_upload_one_batch` | 目的地改为 `ds_name/.staging/batch_XXXX/`，增加 `batch_idx` 参数 |
| `scripts/streaming_pipeline.py` 调用点 | 传递 `batch_idx` 到 `_upload_one_batch` |

### 验证方法

```python
# 处理 2 个 batch，验证 .staging/ 下有两个独立目录
# 验证 batch_0000/output/ 和 batch_0001/output/ 互不包含对方文件
```

状态：已修复。

---

## Case 109: Checkpoint 在上传完成前标记成功 (checkpoint_before_upload)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`_execute_staged` Phase 2 完成后立即将数据集写入 `completed_set` 并保存 checkpoint
（原 ~line 1677），然后才执行 Phase 3 上传。若上传阶段进程崩溃（网络故障、NAS 满），
已写入磁盘的 checkpoint 显示数据集"已完成"，但实际结果未到达 NAS。
重启时 `_load_checkpoint` 跳过该数据集，导致数据丢失。

修复前：Phase 2 完成 → checkpoint 标记 completed → Phase 3 上传（可能崩溃）
修复后：Phase 2 完成 → 临时记录 _phase2_ok → Phase 3 上传成功 → checkpoint 标记 completed

### 根因链

1. Phase 2 处理成功后立即 `completed_set.add(ds_name)`
2. `_save_checkpoint` 写入磁盘
3. Phase 3 上传期间崩溃 → checkpoint 已永久标记 completed
4. 重启后跳过该数据集 → 结果从未到达 NAS

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` `_execute_staged` | Phase 2 使用临时 `_phase2_ok` 集合；Phase 3 成功后才 `completed_set.add` |
| `scripts/streaming_pipeline.py` `_execute_staged` | ok_count 基于 `_phase2_ok` 减 upload_failures 计算 |

### 验证方法

```python
# 模拟：Phase 3 中 kill 进程；重启后确认该数据集未在 completed_set 中
# 预期：checkpoint 不包含该数据集，重新处理 Phase 2（跳过已完成的 batch）+ Phase 3
```

状态：已修复。

---

## Case 110: strict_ok 输出路径与流式上传器不匹配 (strict_output_path_mismatch)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py, scripts/run_pipeline.py

### 现象

`run_pipeline.py` strict_ok 模式将输出重定向到 `workspace/strict_ok_runs/{run_id}/output`，
但 `_upload_one_batch` 从 `local_dir/output` 读取。若配置的 output-dir 为空（strict 模式常见），
上传器上传空目录，strict 实际结果留在 NVMe 后被 cleanup 删除。

修复前：strict 结果写入 `workspace/strict_ok_runs/{run_id}/output`，上传器读 `batch_dir/output`（空）
修复后：上传器检测 strict_ok 输出，自动从 `strict_ok_runs/{run_id}/output` 读取

### 根因链

1. `run_pipeline.py` `strict_run_paths()` 返回重定向的 output_dir
2. `_process_one_batch` 传递 `--output-dir batch_dir/output`，但 `run_pipeline.py` 内部重定向
3. `_upload_one_batch` 从 `batch_dir/output` 读取 → 空目录
4. cleanup 删除 batch_dir → strict 结果永久丢失

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` | 新增 `_detect_strict_output(workspace)` 函数 |
| `scripts/streaming_pipeline.py` `_upload_one_batch` | 若 configured output 为空，自动检测并使用 strict_ok output |

### 验证方法

```python
# strict_ok=True 的配置运行 batch；验证上传器从 strict_ok_runs/ 读取
# 预期：strict output 被正确上传到 staging 目录
```

状态：已修复。

---

## Case 111: 断点续跑丢失批次级进度 (batch_level_resume_lost)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`_save_batch_progress` 将每数据集批次数写入 `.batch_progress.json`，但 `_load_checkpoint`
只恢复数据集名集合，`.batch_progress.json` 从未被加载。若数据集有 50 个 batch、已处理 27 个后
崩溃，重启时整个数据集的 50 个 batch 全部重新处理。

修复前：已处理 27/50 batch → 崩溃 → 重启后全部 50 个重新处理
修复后：已处理 27/50 batch → 崩溃 → 重启后跳过前 27 个，仅处理剩余 23 个

### 根因链

1. `_save_batch_progress` 写入 `.batch_progress.json`（best-effort）
2. 无对应的 `_load_batch_progress` 函数
3. `run_batch` 构建 `all_batches` 时不检查已完成的 batch 数
4. 每个数据集从 batch 0 重新开始

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` | 新增 `_load_batch_progress(ckpt_path)` 函数 |
| `scripts/streaming_pipeline.py` `run_batch` | 加载 batch progress；构建 `all_batches` 时跳过已完成的 batch |

### 验证方法

```python
# 1. 创建包含 {'ds1': {'done': 3, 'fail': 0, 'total': 10}} 的 .batch_progress.json
# 2. 重启 run_batch
# 3. 验证 ds1 的前 3 个 ctc_ready batch 被跳过
```

---

## Case 112: all-GPU preflight namespace 拒绝 `.ctc_run_receipt.json` (allgpu_namespace_receipt_conflict)

**日期**: 2026-08-07
**涉及文件**: scripts/ctc_prealign.py

### 现象

all-GPU preflight 校验在 `scripts/ctc_prealign.py:1285-1288` 构建的 `_allowed` namespace 集合
不包含 `.ctc_run_receipt.json`，但同一个 preflight 循环在 `scripts/ctc_prealign.py:1330-1332`
强制要求该文件存在且可解析。结果是任何成功写出 shard receipt 的正常 shard 都会因
namespace mismatch 被拒绝。

修复前：所有 all-GPU shard 在 preflight 阶段被误判为失败
修复后：`.ctc_run_receipt.json` 在 namespace 校验中作为合法 artifact 被接受

### 根因链

1. Case 99 (R5) 实现为每个 shard 写入 `.ctc_run_receipt.json` 以记录模型/字典 identity
2. all-GPU preflight（line 1285-1288）定义 `_allowed` namespace = artifact suffixes ∪ {manifest.json, summary.txt, .ctc_normalized}
3. 同一 preflight（line 1330-1332）要求 `_receipt_path = _shard_dir / ".ctc_run_receipt.json"` 存在
4. `{p.name for p in _files} != _allowed` 在 line 1292 因 `_allowed` 缺少 `.ctc_run_receipt.json` 而抛出 RuntimeError
5. 两个校验要求矛盾：namespace 排除该文件，但文件必须存在

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/ctc_prealign.py` line 1286 | `_allowed` 集合增加 `".ctc_run_receipt.json"` |

### 验证方法

```python
# 模拟 all-GPU shard 目录包含 manifest.json、summary.txt、.ctc_normalized、.ctc_run_receipt.json
# 和所有 stem 的 artifacts
# import json, ast; ast.parse(Path("scripts/ctc_prealign.py").read_text())  # 确认语法正确
# 预期：namespace 校验通过（不再抛出 RuntimeError("shard namespace mismatch")）
```

---

## Case 113: `step_prealign` 陈旧 .TextGrid 存在性短路导致分母缩小 (prealign_stale_shortcut)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`step_prealign` 在 `run_pipeline.py:810-813` 仅检查 `ctc_out.exists() and any(ctc_out.glob("*.TextGrid"))`
即返回成功。这会接受：
- 部分完成（仅 50% stems 有 .TextGrid）的陈旧输出
- 缺少 `.ctc_normalized` marker 的不完整 bundle
- v3/v4 marker manifest digest 不匹配的篡改/陈旧数据

修复后：必须通过 v4 marker content identity 校验（stem count + manifest SHA-256），或接受 v3 legacy marker
但标注警告。无 marker 的目录视为不完整，触发重跑。

### 根因链

1. `step_prealign` 的 fast path 用 `any()` 判断存在性 — 任意一个 .TextGrid 即视为完成
2. CTC 预对齐可能被中断（如 OOM、超时），留下部分产物
3. 后续 `step_mfa_align` 以 `.lab` 子集建立 alignment 分母 — 静默缩小 stem 集
4. 最终 output 计数小于实际输入，但管线报告成功

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 810-839 | 替换 `any(ctc_out.glob("*.TextGrid"))` 为 v4 marker + manifest digest 校验 |

### 验证方法

```python
# 1. 创建仅含 1 个 .TextGrid 的 ctc_out 目录（无 .ctc_normalized marker）
# 2. 运行 step_prealign（无 --overwrite）
# 3. 预期：报告 "incomplete bundle"，触发重跑（不返回 0）
```

---

## Case 114: `step_adjust_ctc` 陈旧 .TextGrid 存在性短路 (adjust_ctc_stale_shortcut)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`step_adjust_ctc` 在 `run_pipeline.py:1275-1278` 仅检查 `ctc_out.exists() and any(ctc_out.glob("*.TextGrid"))`
即返回成功并设置 `ctx["ctc_pretg_adj"]`。部分完成或与输入 stem 集不一致的输出被当作完整结果复用。

修复后：验证输出 stem 集合与输入 stem 集合完全一致（通过 .lab 文件名匹配），否则触发重跑。

### 根因链

1. `step_adjust_ctc` 检查输出目录存在且包含任意 .TextGrid 就跳过
2. 未校验输出 stem 是否覆盖全部输入 stem
3. 缺失/额外的 .TextGrid 导致后续 MFA 使用不对齐的锚点集合

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 1301-1321 | 替换 `any()` 为 input/output stem 集合交集校验 |

### 验证方法

```python
# 1. 在 ctc_pretg_adj 中仅保留部分 stem 的 .TextGrid
# 2. 运行 step_adjust_ctc（无 --overwrite）
# 3. 预期：报告 "incomplete ... re-running"，输入/输出 stem 集合不匹配触发重跑
```

---

## Case 115: `step_link_ctc` 可解析旧 manifest 直接短路 (link_ctc_manifest_shortcut)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`step_link_ctc` 在 `run_pipeline.py:2948-2959` 发现旧 `ctc_ready_manifest.json` 可解析为 JSON
且包含非空 stems 列表即返回成功。manifest 中的 stems 可能对应已删除的 .lab 文件、或被意外清空的
CTC workspace。

修复后：逐条验证 manifest 中每个 stem 的 .lab 文件确实存在于 workspace 中，任何缺失都触发重新扫描。

### 根因链

1. ctc_ready_manifest.json 被解析 → `prev.get("stems", [])` 非空 → 直接返回 0
2. 实际链接阶段（line 3180-3202）中缺失文件仅 warning，不阻断
3. 后续步骤在残缺链接的 workspace 上运行，分母远小于预期

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 2990-3007 | 增加逐 stem .lab 文件存在性校验 |

### 验证方法

```python
# 1. 创建合法的 ctc_ready_manifest.json（包含 10 个 stems）
# 2. 删除其中 3 个 stem 的 .lab 文件
# 3. 运行 step_link_ctc（无 --overwrite）
# 4. 预期：报告 "3/10 missing .lab files — re-scanning"
```

---

## Case 116: `--skip-to` 静默追加不属于当前模式的步骤 (skipto_cross_mode_append)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`run_pipeline.py:3849-3855` 的 `--skip-to` 处理在目标步骤不属于当前模式的 step_order 时，
执行 `step_order.append(args.skip_to)` 静默追加。这可能导致：
- `ctc_ready` 模式下 `--skip-to trim` 追加 trim 到 pipeline
- `full` 模式下 `--skip-to link` 追加不存在的 link 步骤
- 跨模式 route 组合产生未定义行为

修复后：`--skip-to` 指向非当前模式步骤时直接返回非零错误，列出允许的步骤。

### 根因链

1. `args.skip_to` 传入步骤名
2. `args.skip_to not in step_order` 为 True（步骤属于其他模式）
3. `step_order.append(args.skip_to)` 静默修改 route
4. 执行混合 route，无任何警告

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 3853-3857 | `step_order.append` 替换为 error + return 1 |

### 验证方法

```python
# python scripts/run_pipeline.py --config configs/hecheng_ria_test1.yaml --mode ctc_ready --skip-to trim
# 预期：非零退出，输出 "ERROR: --skip-to 'trim' is not in the 'ctc_ready' route"
```

---

## Case 117: `--scan-only` 在 full 模式下执行破坏性 trim (scan_only_full_trim)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`run_pipeline.py:3859-3865` 的 `--scan-only` 在 `full` 模式下因 route 中无 `link` 步骤，
执行 `run_list[:1]` = `["trim"]`。trim 是破坏性步骤（裁剪静音、归一化音频），
违背 `--scan-only` 的"只读扫描"语义。

修复后：`full` + `--scan-only` 直接返回错误，提示使用 `ctc_ready` 模式做只读扫描。

### 根因链

1. `--scan-only` 设计用于 `ctc_ready` 模式的 link 步骤（只读扫描）
2. `full` 模式的 route 以 `trim` 开头（破坏性）
3. `run_list[:1]` 无条件取第一个步骤
4. trim 修改原始音频文件

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 3861-3868 | full + no-link → error 而非 run_list[:1] |

### 验证方法

```python
# python scripts/run_pipeline.py --config configs/hecheng_ria_test1.yaml --mode full --scan-only
# 预期：非零退出，输出 "ERROR: --scan-only in full mode without link would run trim"
```

---

## Case 118: `--validate` 失败不影响最终退出码 (validate_no_exit_code)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`run_pipeline.py:3968-3976` 的 `--validate` 检查在步骤成功后调用 `validate_step_output()`，
但检查失败仅打印 `[VALIDATE] ... failed` 而不将错误加入 `failed` 列表。
管线报告 "DONE: Success" 但 validate 实际检测到输出规范违反。

修复后：validate 失败加入 `failed` 列表（以 `validate:{step_name}` 形式），
非零退出；`--force` 下继续收集所有 validate 错误。

### 根因链

1. 步骤执行返回 `rc == 0` → 进入 `elif args.validate` 分支
2. `validate_step_output()` 返回非空 issues 列表
3. issues 仅被打印，`failed` 列表不变
4. 管线最终 `return 0` — 假阳性

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 3974-3981 | validate 失败追加到 `failed` 列表 + 非 force 时 break |

### 验证方法

```python
# python scripts/run_pipeline.py --config configs/hecheng_ria_test1.yaml --mode ctc_ready --validate
# 若 postprocess 输出不符合 output_spec：
# 预期：非零退出，"FAILED: validate:postprocess"
```

---

## Case 119: `_run_direct` 丢弃子进程 return code (direct_subprocess_rc_lost)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`streaming_pipeline.py:966-983` 的 `_run_direct` 调用 `subprocess.run(cmd)` 但不检查 return code。
子进程（run_pipeline.py）失败时 streaming_pipeline 仍退出 0。

修复后：`_run_direct` 检查 `subprocess.run(cmd).returncode`，非零时 `sys.exit(rc)` 传播错误。

### 根因链

1. `_run_direct` 调用 `subprocess.run(cmd)` — 正确的 API 调用但丢弃结果
2. 返回的 `CompletedProcess` 包含 `returncode` 但未使用
3. `main()` 在 `_run_direct` 之后隐式返回 None → 进程退出码 0
4. 即使 run_pipeline.py 内部失败，上游调用者也看不到

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` line 983-986 | `subprocess.run(cmd)` → `rc = subprocess.run(cmd).returncode; if rc: sys.exit(rc)` |

### 验证方法

```python
# 模拟：传入不存在的数据目录触发 run_pipeline.py 失败
# python scripts/streaming_pipeline.py --data-dir /nonexistent --direct
# 预期：非零退出（传播 run_pipeline.py 的错误码）
```

---

## Case 120: `_prefetch_worker` 忽略 copy failures 并允许失败 batch 进入 process queue (prefetch_fail_open)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`streaming_pipeline.py:780-783` 的 `_prefetch_worker` 做两个决策错误：
1. `ok = (missing_audio == 0)` — 忽略 `failed`（文件拷贝失败数）
2. prefetch 失败仍递增 `prefetched` 计数（不递增 `prefetch_fail`）

修复后：`ok = (missing_audio == 0 and failed == 0)`；prefetch 失败正确记录到 `prefetch_fail`。

### 根因链

1. 文件拷贝失败加入 `failed` 计数器
2. `ok` 仅检查 `missing_audio`，不检查 `failed`
3. 拷贝失败的 batch 进入 `prefetch_queue` → process queue
4. process 在残缺数据上运行 → 子进程失败 → 但根因被隐藏

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` line 782-783 | `ok = (missing_audio == 0 and failed == 0)` |
| `scripts/streaming_pipeline.py` line 790-798 | prefetch 失败记录到 `prefetch_fail` 而非 `prefetched` |

### 验证方法

```python
# 模拟：对只读源文件系统触发 PermissionError，使部分 copy 失败
# 预期：batch 不进入 prefetch_queue；stats.prefetch_fail += 1
```

---

## Case 121: `StreamingPipeline.run()` 失败 batch 仍被推入 upload queue (process_fail_upload_leak)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`streaming_pipeline.py:936-937` 无论 `_process_batch` 成功或失败，都将 batch_idx 推入
`upload_queue`。失败 batch 的残缺/空输出被上传到 NAS，覆盖之前的有效结果。
同时 `all_ok`（line 944）仅检查 `process_fail`，忽略 `prefetch_fail` 和 `upload_fail`。

修复后：失败 batch 不进入 upload queue；`all_ok` 综合检查 three-phase failures。

### 根因链

1. `_process_batch(batch_idx)` 返回 False
2. `self.upload_queue.put(batch_idx)` 无条件执行
3. `_upload_worker` 上传可能为空或残缺的本地输出
4. `_merge_to_nas` 可能覆盖之前正确上传的 NAS 数据

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` line 937 | `upload_queue.put(batch_idx)` 移至 `if ok:` 块内 |
| `scripts/streaming_pipeline.py` line 944-946 | `all_ok` 增加 `prefetch_fail == 0 and upload_fail == 0` |

### 验证方法

```python
# 模拟：使 _process_batch 返回 False
# 预期：upload_queue 不包含该 batch_idx；all_ok 为 False
```

---

## Case 122: `run_batch` / `run_pipelined_batch` 返回 None 导致主调无法感知失败 (batch_return_none)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`streaming_pipeline.py` 的 `run_batch`（line 1824）和 `run_pipelined_batch`（line 2639）
签名均为 `-> None`。内部打印 `"BATCH COMPLETE: X/Y OK"` 但未将失败状态传递给调用者。
`main()` 在 `run_batch(args)` 后执行 `return`（隐式返回 None）→ 进程退出码 0。

修复后：两个函数返回 `bool`（all_ok）；`main()` 在 batch 失败时 `sys.exit(1)`。

### 根因链

1. `run_batch` 内部收集 `ok_count` 和 `fail_list`，打印汇总
2. 函数无 return 语句 → 调用者收到 None
3. `main()` 的 `return` 导致进程退出 0
4. 批处理失败在退出码层面不可见

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` line 1824 | `-> None` → `-> bool` |
| `scripts/streaming_pipeline.py` line 2320-2322 | 汇总后 `return all_ok` |
| `scripts/streaming_pipeline.py` line 1979-1980 | 传播 `run_pipelined_batch` 返回值 |
| `scripts/streaming_pipeline.py` line 2639 | `-> None` → `-> bool` |
| `scripts/streaming_pipeline.py` line 2901-2908 | 汇总后 `return all_ok` |
| `scripts/streaming_pipeline.py` line 1188-1191 | `if not run_batch(args): sys.exit(1)` |

### 验证方法

```python
# 模拟：batch_cache 中包含一个不存在的数据集
# python scripts/streaming_pipeline.py --batch-cache cache/test_fail.cache.json
# 预期：非零退出；输出 "BATCH COMPLETE ... WITH FAILURES"
```

---

## Case 123: staged CPU upload 对 rsync/copy 失败仅告警并返回 True (staged_upload_fail_open)

**日期**: 2026-08-07
**涉及文件**: scripts/streaming_pipeline.py

### 现象

`streaming_pipeline.py:2616-2631` 的 `_run_cpu_phase` 在 rsync 非零退出、超时、异常时仅打印
`WARNING`，函数继续执行 cleanup 并返回 True。完整的 output/filtered/ctc_pretg_adj 未被上传
到 NAS，但调用者（CPU worker）将其视为成功，进而将数据集标记为 DONE。

修复后：所有上传错误改为 `ERROR`；`_upload_ok` flag 收集失败；任何上传失败返回 False 并保留本地目录。

### 根因链

1. rsync 返回非零 / 超时 / 抛异常
2. `print("WARNING: ...")` — 不改变控制流
3. `shutil.rmtree(local_dir)` 删除本地证据
4. `return True` → caller 标记数据集 DONE → 永久丢失数据

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/streaming_pipeline.py` line 2601-2641 | WARNING → ERROR + `_upload_ok = False`；失败返回 False 并保留 local_dir |

### 验证方法

```python
# 模拟：mock rsync 返回非零或设为不存在的路径
# 预期：函数返回 False；local_dir 被保留（.FAILED 重命名或原位保留）
```

---

## Case 124: batch_ctc_ready 缺失音频数据集被 skip 而不进入 fail_list (batch_missing_audio_skip)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`run_pipeline.py:3589-3592` 在 `batch_ctc_ready` 模式下，当数据集的 `audiodir` 不存在时，
仅打印 `SKIP (no audio)` 并 `continue`，不将数据集名加入 `fail_list`。
结果是缺失音频的静默遗漏 — 分母包含该数据集但处理结果中无其踪迹。

修复后：缺失音频的数据集加入 `fail_list`，计入最终失败汇总。

### 根因链

1. 扫描得到 datasets 列表（来自 cache 或 discovery）
2. `audiodir = audio_root / ds_name / "wavs"` 不存在
3. `continue` — 跳过但不记录失败
4. `ok_count / len(datasets)` 显示部分成功，但缺失项不可见

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 3592 | SKIP 前增加 `fail_list.append(ds_name)` |

### 验证方法

```python
# 在 batch cache 中引用一个不存在音频目录的数据集
# python scripts/run_pipeline.py --config configs/batch_test.yaml --mode batch_ctc_ready
# 预期：输出 "Failed: missing_dataset_name"；非零退出
```

---

## Case 125: 非 strict 模式输出路径无 run-specific 隔离 (flat_output_no_isolation)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py, scripts/pipeline_utils.py

### 现象

当 `strict_ok` 和 `output_staging` 均未启用时，`output_dir` 和 `filtered_dir` 直接指向
`workspace / "output"` 和 `workspace / "filtered"`。两次运行之间新旧产物混合，无隔离边界。

修复后：默认使用 `workspace/runs/<run_id>/output` 和 `workspace/runs/<run_id>/filtered`，
每次运行写入独立目录；运行结束时写入 `.pipeline_run_receipt.json` 记录实际路径、stem 集合
和失败步骤。

### 根因链

1. 非 strict 非 staging 路径进入 else 分支（line 3810-3811）
2. 仅 `filtered_dir` 被初始化，`output_dir` 使用配置的扁平路径
3. 无 run ID 注入 → 多次运行共享同一输出目录
4. 无收据证明某次运行的输入→输出映射

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 3810-3817 | else 分支使用 `workspace/runs/<run_id>/output|filtered` |
| `scripts/run_pipeline.py` line 4088-4105 | 写入 `.pipeline_run_receipt.json` |
| `scripts/pipeline_utils.py` line 1872-1946 | 新增 `make_pipeline_run_id()` 和 `write_pipeline_run_receipt()` |
| `scripts/run_pipeline.py` line 44-53 | 导入 `make_pipeline_run_id`, `write_pipeline_run_receipt` |

### 验证方法

```python
# 运行两次同一管线（无 --overwrite）：
# python scripts/run_pipeline.py --config configs/hecheng_ria_test1.yaml --mode ctc_ready
# python scripts/run_pipeline.py --config configs/hecheng_ria_test1.yaml --mode ctc_ready
# 预期：两次输出位于不同的 workspace/runs/<run_id>/ 子目录
# 预期：每个 output/ 目录包含 .pipeline_run_receipt.json
```

---

## Case 126: 无统一 config schema 校验 (config_schema_missing)

**日期**: 2026-08-07
**涉及文件**: scripts/run_pipeline.py

### 现象

`load_config`（`run_pipeline.py:324-334`）仅做递归字典合并，不对以下做任何校验：
- 未知顶层键（typo 被静默忽略）
- 类型不匹配（如 `mfa: "string"` 而非 dict）
- 跨字段矛盾（all-GPU + wrong mode、NVV bias + disabled、strict + pad_silence）
- 必填路径或资源缺失

修复后：新增 `validate_config()` 在模式解析完成后立即运行，检查 9 类规则；
发现错误非零退出（`--force` 可覆盖）。

### 根因链

1. YAML 中打错配置键名 → `cfg.get("ctc_prelaign")` 返回 None，无提示
2. 将 dict 字段误写为字符串 → 后续 `.get()` 在 str 上调取 → AttributeError（离 config 加载点很远）
3. 矛盾配置（all_gpus + ctc_ready mode）→ 运行时才暴露

### 涉及修改

| 文件 | 修改内容 |
|------|---------|
| `scripts/run_pipeline.py` line 3393-3503 | 新增 `validate_config(cfg, mode) -> list[str]` |
| `scripts/run_pipeline.py` line 3588 | 占位注释 |
| `scripts/run_pipeline.py` line 3628-3641 | 模式解析后立即调用 `validate_config` |

### 验证方法

```python
# 测试 config 包含未知键 + 类型错误 + 逻辑矛盾：
# python scripts/run_pipeline.py --config configs/bad_config.yaml --mode ctc_ready
# 预期：非零退出，输出 "ERROR: Config validation failed (N issues):"
# python scripts/run_pipeline.py --config configs/bad_config.yaml --mode ctc_ready --force
# 预期：打印错误但继续执行

from run_pipeline import validate_config
errors = validate_config({'unknown_key': 1, 'mfa': 'not_dict'}, 'full')
assert len(errors) >= 2  # unknown key + type mismatch
errors = validate_config({'ctc_prealign': {'all_gpus': True}}, 'ctc_ready')
assert any('all_gpus' in e and 'full' in e for e in errors)
```

状态：已修复。

---

## Case 127: 权威标点/连字符英文投影与 phone 越界 (reference_projection_contract)

**日期**: 2026-08-07
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/pipeline_utils.py`, `scripts/audit_strict_ok.py`

### 现象

独立 canary 中发现三类相互独立的问题：

1. CTC 将 `K-Pop` 切成 `kp`/`op`，导致 hanzi 与权威英文拼写不一致；
2. CTC 长停顿或后处理边界扩展使 `pinyin_phones` 跨越相邻 words，产生
   `phone_outside_word`，并在严格英文注入后留下中文 phone 重叠；
3. CTC 标点缺失或额外终止标点会反向修改 `_ref.txt` 的 raw 文本，破坏权威标点。

### 修复约束

- reference text 是 raw/标点/英文表面形式的唯一权威；CTC 标点只提供局部时间锚点；
- 连字符英文的 words/strict English phone 实例保持 MFA ledger 的原始分段，hanzi/pinyin
  投影恢复完整参考拼写，不伪造英文 phone；
- derived phone 只裁回其最大重叠 word，严格英文 `en:` phone 的 affine 时间不被非英文
  de-overlap 改写；
- 每轮最终发布前重新检查五层 TextGrid、reference sequence、phone ownership 和
  strict English provenance。

### 验证

```bash
python3 scripts/verify_reference_authority.py
python3 -B -m compileall -q scripts check_ipa_mapping.py verify_risks.py
```

独立 128 条 canary 的最终结果：`70/128 = 54.69%` 进入 strict-ok 发布集；其余 58 条
均进入过滤集，严格审计没有发现新的 phone 越界、语义序列或 phone 重叠违规。

---

## Case 128: CTC 全 GPU 合并隔离与输入副本安全 (canary_transaction_isolation)

**日期**: 2026-08-07
**涉及文件**: `scripts/ctc_prealign.py`, `scripts/run_pipeline.py`

### 现象

共享临时目录和 symlink/hardlink 输入会使并发 canary 或 CTC-ready 后处理修改权威
CTC 文件；全 GPU 合并若直接写入 live output，则失败时会留下不完整 namespace。

### 修复

- 每次 pipeline 使用 run-local `temp`；CTC-ready 输入统一独立 `copy2`；
- all-GPU merge 先在带 PID 的 staging 目录完成 manifest/receipt/完整性校验，再原子
  发布；失败时保留隔离的 shard evidence，不污染 live 输出；
- 原 54k workspace 只读保留，canary/recovery 使用独立 workspace 和 versioned publish。

### 验证

```bash
python3 scripts/verify_strict_ctc_ready_import.py
python3 scripts/verify_reference_only_ctc.py
python3 scripts/verify_reference_authority.py
```

状态：已修复并通过可运行回归测试；54k 全量仍受 canary 放行线约束，未启动。

---

## Case 129: strict manifest 未绑定 pipeline-run-receipt-v2 (strict_receipt_publish_gate)

**日期**: 2026-08-10
**涉及文件**: scripts/audit_strict_ok.py, scripts/verify_strict_ok.py,
scripts/prepare_hecheng_english_ctc_ready.py, scripts/verify_hecheng_english_ctc_ready_v4.py

### 现象

严格清单原先仅由 `*.lab` 推导 expected 集合，54000 个 WAV 中缺少 TXT 的
stem 可能被误当作可处理输入；发布前也没有可验证的 source/eligible/exclusion
守恒证据。现在 strict audit 要求有效的 `pipeline-run-receipt-v2`，并令
`expected_stems == receipt.eligible.stems`。缺失、legacy v1、字段/摘要篡改或
集合不匹配均在目标目录创建前失败；`missing_reference` 只能出现在 exclusions，
处理后的质量拒绝只能出现在 filtered。

### 回归覆盖

```text
R129-a expected == eligible
R129-b 54000 == 53998 eligible + 2 missing_reference exclusions
R129-c missing/legacy/tampered receipt => verify failure and no target
R129-d exclusion vs processed rejection bucket separation
```

### 验证命令

```bash
python scripts/verify_receipt_accounting.py
python scripts/verify_strict_ok.py
python scripts/verify_prepare_hecheng_english_ctc_ready.py
python scripts/verify_hecheng_english_ctc_ready_v4.py --help
```

### 遗留集成要求

`pipeline_utils.publish_output_versioned()` 仍可被其他调用方直接调用；其所有者
必须在复制/创建 target 之前调用 strict manifest verifier，并验证 v2 receipt 的
source conservation、eligible membership 与 exclusion/filtered 分区。此文件未修改
该公共发布函数，避免越权改变其接口。

---

## Case 130: 批量 GPU/CPU 资源统一规划与旧分片启动器 (bounded_batch_resources)

**日期**: 2026-08-12
**涉及文件**: `scripts/streaming_pipeline.py`, `scripts/launch_8gpu.py`,
`scripts/launch_multi_gpu.sh`

### 现象

旧的 8-GPU 启动器会为同一 strict run 创建多个独立分片；这些分片没有共享的资源
计划，也可能竞争同一批中间产物。批量流式路径此前也允许 dataset worker 与每个 MFA
进程池的请求并发相乘，导致 CPU 过量订阅。无界的 phase 间队列还会在 NAS 较慢时持续
占用本地 NVMe。

### 修复

- `plan_streaming_resources()` 在 ordinary 和 pipelined 路径计算同一 CPU/GPU 预算，
  将 CPU worker、每个 MFA/MFA EN pool 的 jobs，以及实际 batch 数限制在安全上限内；
- pipelined 模式使用有界 GPU→CPU 队列和 CPU/upload 队列，以反压限制本地积压；
- `launch_8gpu.py` 改为只启动一个 `streaming_pipeline.py --pipelined` 命令，并拒绝
  strict `ctc_ready` 配置；
- `launch_multi_gpu.sh --streaming` 默认使用 `--pipelined`，并提供
  `--no-pipelined` 选择普通批量路径。

### 回归覆盖

```bash
python -m pytest tests/test_streaming_resources.py tests/test_multi_gpu_launchers.py
```

文档与批量示例配置公开 `streaming.num_gpus`、队列缓冲、`pipelined.cpu_workers` 和
MFA jobs 请求值。它们仍是请求值；运行时资源规划器是最终的上限执行者。

状态：已修复；严格生产运行仍必须使用其专用 strict workflow，而非通用批量启动器。

---

## Case 131: 分 speaker 音频目录导致 pre-CTC 分母误判为空 (recursive_pre_ctc_wav_denominator)

**日期**: 2026-08-12
**涉及文件**: `scripts/run_pipeline.py`, `scripts/pad_silence_edges.py`,
`configs/hecheng_ria_fresh.yaml`

### 现象

在 GPU 已可见、NVMe 音频缓存可读的实际 54k 启动中，主管线在 `pad_silence` 阶段停止：

```text
ERROR: pre-CTC physical WAV denominator is empty or duplicated
Stopping. Use --force to continue on errors.
```

输入目录实际布局为：

```text
/mnt/nvme3/mfa_audio_cache_ria/
├── ria/     18000 WAV, 17999 TXT
├── 花礼/    18000 WAV, 17999 TXT
└── 雪狐桑/  18000 WAV, 18000 TXT
```

旧代码只对 `audio_dir.iterdir()` 做根目录平面扫描。由于根目录只有三个 speaker
子目录，没有直接 WAV，因此冻结出的分母为空；即使绕过该错误，后续 transform receipt
仍会用 `audio_dir / f"{stem}.wav"`，无法解析子目录内的源 WAV。

### 根因

pre-CTC 的 nvrasr/full 路径没有 CTC manifest 可复用，必须从物理 WAV 冻结 source
denominator。实现错误地假设所有输入都是 flat layout，而当前 RIA 数据集采用
speaker-partitioned layout。该错误发生在 GPU 推理之前，与 NVIDIA/CUDA 无关。

### 修复

- `step_pad_silence()` 改用递归 `rglob("*.wav")` 扫描所有 speaker 子目录；
- 按 `Path.stem` 建立索引并拒绝重复 stem，避免后续 flat CTC/MFA namespace 发生碰撞；
- 继续写入排序且唯一的 `pre_ctc_stems.txt`，保持分母冻结契约；
- 生成 audio transform receipt 时使用 `find_wav()` 递归解析真实源路径，并对缺失源文件
  fail-closed；
- 没有使用 `--force` 绕过分母校验，也没有修改原始音频。

### 实际验证

只读检查结果：

```text
physical WAVs = 54000
unique stems  = 54000
duplicate     = 0
ria/花礼/雪狐桑 = 18000/18000/18000 WAV
```

修复后的冻结逻辑在临时 workspace 中生成 54,000 条唯一 manifest；相关回归测试：

```bash
python -m compileall -q scripts/run_pipeline.py
python -m pytest -q \
  tests/test_streaming_resources.py \
  tests/test_multi_gpu_launchers.py \
  tests/test_gpu1000_postprocess_audit.py \
  tests/test_ctc_all_gpu_merge_receipts.py \
  tests/test_axis_contracts.py
# 25 passed
```

### 运行要求

修复只解决输入分母和路径解析问题；生产运行仍要求 NVIDIA 设备透传、
`nvidia-smi -L` 可见 8 卡，以及 `/mnt/nvme3` 可写。该案例修复后应重新启动完整
`nvrasr_fallback` 管线，不应从失败 run 目录强行续跑。

状态：已修复；已通过 54k 输入布局验证，待 GPU 环境下重新执行主管线验收。

---

## Case 132: 先前 MFA 对齐失败问题的统一根因、修复链路与验收索引 (mfa_alignment_repair_index)

**日期**: 2026-08-12
**涉及文件**: `scripts/run_pipeline.py`, `scripts/ctc_prealign.py`,
`scripts/pipeline_utils.py`, `scripts/pad_silence_edges.py`,
`scripts/postprocess_textgrids.py`, `scripts/align_english_mfa.py`,
`configs/hecheng_ria_fresh.yaml`

本条不是新的单点 bug，而是把先前已经定位和修复的 MFA 失败原因集中登记，作为重新执行
主管线前的验收清单。具体历史证据仍以所列 Case 正文为准。

### 1. 音频轴与 CTC/MFA 时间轴冲突

- Case 95：旧 `pad_silence_edges.py` 原地修改 CTC TextGrid/tokens/punct，重复运行会
  累积平移，且尾部裁剪没有同步限制 CTC 末端，造成 MFA 输入轴与 CTC 锚点严重冲突；
- Case 98/100：encoder 网格时长冒充 WAV 时长、blank-run pause frame 未正确移除，导致
  锚点整体漂移；
- Case 131：分 speaker 子目录被错误当作 flat 目录，pre-CTC 分母为空，任务在 MFA 前
  就被错误阻断。

修复链路：CTC 生成前冻结物理 WAV 分母；CTC 与 MFA 共用权威 WAV 轴；strict workflow
禁止在 CTC 之后再 pad；每个音频变换写入 transform receipt；所有 TextGrid/tokens/
punct 必须落在 WAV domain 内。当前 RIA 配置的 `audio_axis.post_ctc_pad_silence` 为
`forbidden`。

### 2. CTC transcript/参考文本不一致导致 MFA 词面错位

- Case 61/62/68/69：ASR tokenizer 拆碎英文词或覆盖权威参考文本，导致 `.lab`、tokens、
  TextGrid 与 MFA 词典不一致；
- Case 81/82：RIA 归一化只修改部分 bundle，或 marker 使过期 bundle 绕过 normalize，
  造成 MFA 接收到不同步的 lab/tokens/TextGrid；
- Case 102：`--no-nvv` reference-only 路径仍可能让 ASR 内容污染 required sidecar。

修复链路：原始 TXT/reference 是唯一词面权威；CTC 只提供时间锚点；`.lab`、
`_tokens.jsonl`、TextGrid、`_ref.txt` 作为不可拆分 bundle 原子生成/校验；RIA 合并必须
三方同步；normalize marker 只能优化，不能替代 bundle 内容校验；reference-only 模式
禁止 ASR 文本覆盖 required sidecar。

### 3. MFA 输出缺失、损坏或误报成功

- Case 76/83/104：分片 MFA 只看退出码或字符串，Popen 异常/超时/损坏 TextGrid 可能
  被当作成功；
- Case 77/113/114/115：部分旧产物或已有 `.TextGrid` 触发短路，分母被静默缩小，未对齐
  stem 未进入报告；
- Case 73/84/89：`unknown`/`spn` 或空 English phones 被当成可用对齐结果，后处理可能
  伪造通过。

修复链路：启动前冻结 expected stem manifest；单进程和分片 MFA 统一使用严格 TextGrid
 parser，检查 tier、interval、正时长和 WAV domain；捕获非零、超时、信号和 Popen
 异常并写结构化失败记录；aligned 分母必须与 frozen manifest 对账；MFA `unknown/spn`
 和无来源 English phone 进入 hard failure/filtered，不得伪造音素。

### 4. MFA 边界、词典和后处理冲突

- Case 16：`fine_tune` 默认开启造成边界漂移和对齐失败；
- Case 32/33/43/49：英文词典缺项、英文词被中文词典检查或 phone 数不足，造成 MFA/
  postprocess 误过滤；
- Case 34–41、44–48、52–60、63–75：phone 重叠/间隙、CTC 跨标点、声母韵母压缩、
  CJK/拼音错位和错误过滤条件把合法 MFA 结果判为失败，或把结构坏结果放行。

修复链路：当前 RIA 配置显式设置 `fine_tune: false`、`beam: 20`、`retry_beam: 80`、
`single_speaker: true`、`allow_partial: true`、`min_output_ratio: 0.54` 和
`timeout: 7200`；后处理按 reference sequence、phone ownership、tier 连续性、
MFA source provenance 分层检查；合法 canonical silence labels 不参与 lexical semantic
比较；`filter_suspicious: false` 与 `enable_word_in_silence_filter: false` 均显式记录，
避免旧默认值重新启用误过滤。

### 5. 资源、缓存和并发造成的 MFA 不稳定

- Case 90/125/128：混合缓存、旧 workspace、非版本化 output 或共享 staging 可能把其他
  批次的 CTC/MFA 产物混入当前运行；
- Case 105/130：父进程计算的 MFA jobs 未传递给子进程，或 GPU/CPU 并发相乘造成过量订阅；
- Case 120–123：复制、处理或上传失败仍可能把失败 batch 送入下一阶段。

修复链路：每次运行使用 run-specific workspace/output；CTC all-GPU 使用隔离 shard 和
原子 merge；父进程把 effective `--mfa-jobs`/`--mfa-en-jobs` 传递给子进程；流式队列有界
并对 copy/process/upload 失败 fail-closed；通用 `launch_8gpu.py` 不再拆分 strict run，
strict `nvrasr_fallback` 由 `run_pipeline.py --ctc_prealign.all_gpus=true` 统一调度。

### 当前验收状态

已完成的代码/回归验证：

```text
targeted pytest suite: 25 passed
continuation regressions: 24 PASS
filtered-recovery regressions: 8 PASS
54k input inventory: 54000 WAV / 54000 unique stems / 0 duplicate
```

尚未宣称完成的部分：GPU 环境下的 54k 全量 CTC→MFA→postprocess→strict-ok 仍需重新
执行；当前只证明修复链路和过滤/分母契约，不把旧 run 的 aligned 或 filtered 结果当作
新运行证据。

状态：历史 MFA 问题已统一登记；对应代码修复和专项回归已通过，等待最新 GPU 主管线
重新运行验收。
