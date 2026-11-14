"""Compatibility recognizers for persisted and current bot reply formats."""

import re
import unicodedata


WORD_CARD_HEADER_RE = re.compile(
    r"(?m)^\s*(?:词库暂无收录)?「(?P<word>[^「」\r\n]{1,128})」"
    r"(?:未收录[:：][^\n]*|词库里还没有，要加入吗？[^\n]*|(?:候选编码)?[:：]\s*$|已在词库[（(][^\n]*)"
)
NUMBERED_WORD_HEADER_RE = re.compile(
    r"(?m)^\s*\d{1,3}[.)、]\s*(?:"
    r"[「“『](?P<quoted_word>[^「」“”『』\r\n]{1,128})[」”』][^\n]*|"
    r"(?P<word>[\u3400-\u9fffA-Za-z0-9_-]{1,128})\s*)$"
)
CANDIDATE_ROW_RE = re.compile(
    r"(?mi)^\s*(?P<index>[1-9]\d{0,2})[.)、]\s*"
    r"(?P<code>[a-z]{1,12})\s*(?:—|–|-)\s*(?P<label>[^\n]+)$"
)
CANDIDATE_CODE_LINE_RE = re.compile(
    r"(?m)^\s*(?:\d+\.\s*)?(?P<code>[a-z]+)\s*[-—–]\s*(?P<label>.+?)\s*$"
)
NUMBERED_CANDIDATE_ROW_RE = re.compile(
    r"(?mi)^\s*\d+\s*[.．、)）]\s*(?:`|\*\*)?([a-z]{1,12})(?:`|\*\*)?"
    r"\s*(?:[—–\-:：]|\s)\s*[^\n]*"
    r"(?:已有|占用|占位|空位|推荐)"
)
COMPACT_CANDIDATE_HEADER_RE = re.compile(
    r"(?m)^\s*(?:「[^」\n]+」(?:未收录[:：]|候选编码[:：]|词库里还没有，要加入吗？)|"
    r"\d+[.)、]\s*「[^」\n]+」[^\n]+，(?:可自动通过|需管理员审核))"
)
CANDIDATE_RECOMMENDATION_LABEL_RE = re.compile(
    r"✅\s*推荐|[（(]\s*(?:✅\s*)?推荐\s*[）)]|←\s*常用度推荐"
)
UNKNOWN_OCCUPANT_LABEL = "已有占用"
OCCUPIED_NO_WORD_LABELS = (UNKNOWN_OCCUPANT_LABEL, "已占用")
PENDING_ADD_BINDING_RES = (
    re.compile(r"以编码\s*(?P<code>[a-z]+)\s*将「(?P<word>.+?)」加入草稿"),
    re.compile(
        r"(?m)^\s*[-•]\s*[“『]「(?P<word>[^」\n]+)」\s*占\s*"
        r"(?P<code>[a-z]+)\s*、\s*「[^」\n]+」顺延[”』]"
        r"(?:\s*[（(][^）)\n]*[）)])?\s*$"
    ),
    re.compile(
        r"(?m)^\s*推荐[：:]\s*「(?P<word>[^」\n]+)」\s*占\s*"
        r"(?P<code>[a-z]+)\s*、\s*「[^」\n]+」顺延(?:[；;].*)?$"
    ),
    re.compile(
        r"(?m)^\s*[•-]\s*「(?P<word>[^」\n]+)」\s*→\s*"
        r"(?P<code>[a-z]+)\s*[（(]推荐[）)]\s*$"
    ),
)
BATCH_RECOMMENDED_CODE_RE = re.compile(r"(?mi)^推荐编码:(?P<code>[a-z]{1,12})\s*$")
RECOMMENDED_CODE_RE = re.compile(
    r"(?mi)^推荐编码\s*:\s*(?P<code>[a-z]{1,12})(?:\(本次仅查询\))?\s*$"
)
RECOMMENDED_CANDIDATE_RE = re.compile(
    r"(?:^|[：:｜|])\s*\d{1,3}[.)、]\s*"
    r"(?P<code>[a-z]{1,12})\s*(?:—|–|-)\s*"
    r"[^\n｜|]{0,64}[（(]\s*(?:✅\s*)?推荐\s*[）)]",
    re.MULTILINE | re.IGNORECASE,
)
BLOCKED_RECOMMENDATION_RE = re.compile(
    r"推荐编码\s*[:：]\s*暂无[（(]现有候选均需点名顶替[）)]|"
    r"占用，本次跳过；要顶替请单独发"
)
LEGACY_REVIEW_RE = re.compile(r"(?m)^\s*(?:喵喵)?审词[:：](?P<review>[^\n]*)$")
LEGACY_SINGLE_WORD_FOOTERS = (
    "可回复编号、编码，或「都加」；可多选，如「添加2、4」。",
    "可回复编号、编码，或「都加」；",
    "可回复编号、编码，或「都加」。",
    "也可回复编号选其他编码。",
    "可回复编号或编码选择其他编码；可多选，如「添加2、4」。",
    "可回复编号或编码选择其他编码；可多选，如「添加1、2」。",
)
COMPACT_SINGLE_WORD_FOOTER_RE = re.compile(
    r"(?m)^回复「加入」或「加入并提交」[^\n]*(?:\n|$)"
)
FREQUENCY_COMPARISON_RE = re.compile(
    r"(?:词频[:：]|[；;])\s*(?P<first>[^\s；;>=]+)\s+(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|无)"
    r"\s*[>=]\s*(?P<second>[^\s；;>=]+)\s+(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|无)"
)
DRAFT_DISPLAY_ROW_RE = re.compile(r"^\s*(?:[•-]\s*)?\d+[.、]\s*(?:新增|修改|删除)\s+")
COMMONNESS_COMPARISON_COPY_RE = re.compile(
    r"(?:常用度(?:对比|比较)|较[^\n，。；]{0,16}常用|"
    r"更[^\n，。；]{0,16}常用|频率[^\n，。；]{0,16}(?:高|低)|"
    r"日常语感|直觉比较|词频[:：])"
)
COMMONNESS_PREFIX_RE = re.compile(r"^(?:常用度(?:对比|比较|评估)|依据|词频)[:：]")
COMMONNESS_SPECULATION_RE = re.compile(r"(?:日常语感|直觉|凭感觉|我觉得|可能略?[高低])")
ORIGINAL_COMMAND_LINE_RE = re.compile(
    r"^(?:原始操作指令|原始指令|原指令)\s*[：:]\s*[「“\"]?(.+?)[」”\"]?$"
)
LEGACY_REFERENCE_COMPARISON_RE = re.compile(
    r"([\u4e00-\u9fff]{1,12})\s*(?:≫|>|＞|更常用|优于|高于|大于)\s*([\u4e00-\u9fff]{1,12})"
)
REFERENCE_HEADING_CLEAN_PREFIX_RE = re.compile(r"^[\s#>*\-•·\d.、:：|]+")
REFERENCE_HEADING_NONWORD_RE = re.compile(r"^[^\u4e00-\u9fff「」]+")
MEMORY_DIFF_RE = re.compile(r"diff\s+\S+[\s\S]*?(?=\n\n|当前草稿|草稿地址|$)")
MEMORY_HIGH_MARKERS = (
    "偏好", "习惯", "记住", "以后", "称呼", "不要", "别再",
    "词库操作", "已处理加词草稿", "已提交当前用户草稿审核", "已确认添加到草稿",
)
STALE_OPTION_EMPTY_COPY = "现在没有待确认的操作。"
STALE_OPTION_MISSING_COPY = "当前没有可执行的确认记录；本次未写入。"
STALE_OPTION_EXPIRED_COPY = "当前没有可执行的确认记录；之前的确认已过期或记录已不存在；本次未写入。"
STALE_OPTION_PREFIXES = ("上一条提议是：", "上一条提议的选项是：")
OPTION_PROPOSAL_RE = re.compile(r"计划|方案|将执行|将把|是否|建议|要不要")
COLLISION_REPLAN_PREFIX = "已调整："
UNDERSTOOD_COMMAND_PREFIX = "我理解为：「"
COMMONNESS_HINT_PREFIX = "常用度提示："
WRITE_DENIAL_RE = re.compile(
    r"(?:未|没有|并未|尚未)(?:(?:成功)?写入|执行(?:任何)?(?:新)?写入)|无法执行"
)
SUCCESS_CLAIM_RES = tuple((kind, re.compile(r"(?<![未没不无])(?:" + pattern + r")"))
                         for kind, pattern in (
    ("submit", r"已(?:经)?(?:成功)?(?:为你)?(?:重新)?提交|提交成功|审核中"),
    ("dictionary", r"已(?:经)?(?:加入|写入)词库|已入库|入库成功|(?:已)?审核通过|已通过审核|已批准|批准成功"),
    ("write", r"已(?:经)?(?:成功)?写入(?!词库)|写入成功|已加入草稿|(?:已添加|添加成功|已更新|更新成功)(?=[^。！？\n]{0,20}(?:草稿|词库|批次|条目|编码))"),
    ("delete", r"已(?:经)?(?:成功)?(?:删除|移除)|(?:删除|移除)成功"),
))
ACTION_CLAIM_LEAD_RE = re.compile(r"^(?:好的|好|收到)[，,：:]\s*")
CURRENT_ACTION_PREFIX_RE = re.compile(r"(?:✅|本次|本轮|现在|刚刚|现已|我(?:$|已)|已确认)")
BARE_ACTION_CLAIM_PREFIXES = frozenset({"", "批次", "整批", "草稿"})
PENDING_REVIEW_CLAIM = "审核中"
NEGATIVE_WRITE_CLAIM_RE = re.compile(
    r"(?:^|[，,。\n])(?:本次|本轮)(?:没有|未)(?:执行(?:任何)?(?:新的?)?写入|写入|添加)"
)
SHAPE_CODE_HINT_RE = re.compile(r"加入编码\s+([a-z]+)\+形码")
AUTHORITATIVE_LINK_LABEL_RE = re.compile(
    r"(?:草稿地址|批次地址|草稿/批次地址|PR|"
    r"旧\s*PR(?:地址|可见于)?|查看旧\s*PR)[：:]?"
)
AUTHORITATIVE_LINK_END_RE = re.compile(
    r"\s*(?:草稿地址|批次地址|草稿/批次地址|PR|"
    r"旧\s*PR(?:地址|可见于)?|查看旧\s*PR)[：:]?\s*$"
)
TRAILING_MANUAL_REVIEW_RE = re.compile(
    r"(?:^|[，,；;。])\s*(?:(?:本轮|仍|该词|预计)\s*)*"
    r"(?:需要|需)管理员(?:审核|确认)[。.]?$"
)


def normalized_copy(text):
    return unicodedata.normalize("NFKC", str(text or ""))


def referenced_card_words(text):
    return tuple(dict.fromkeys(match["word"].strip() for match in WORD_CARD_HEADER_RE.finditer(text)))


def missing_word_card_word(text):
    match = WORD_CARD_HEADER_RE.search(str(text or ""))
    return match["word"].strip() if match is not None else ""


def pending_add_card_word(text):
    for pattern in PENDING_ADD_BINDING_RES:
        match = pattern.search(str(text or ""))
        if match is not None:
            return match["word"].strip()
    return missing_word_card_word(text)


def pending_add_recommended_code(text):
    """Keep the legacy confirmation, reorder, then compact-binding priority."""
    for pattern in PENDING_ADD_BINDING_RES:
        match = pattern.search(str(text or ""))
        if match is not None:
            return match["code"]
    if COMPACT_CANDIDATE_HEADER_RE.search(str(text or "")):
        return marked_candidate_code(text)
    return ""


def has_pending_add_card_contract(text):
    source = str(text or "")
    return bool(
        re.search(r"以编码\s*[a-z]+\s*将「[^」\n]+」加入草稿|"
                  r"(?m:^\s*[•-]\s*「[^」\n]+」\s*→\s*[a-z]+\s*[（(]推荐[）)]\s*$)|"
                  r"「[^」\n]+」\s*占\s*[a-z]+\s*、\s*「[^」\n]+」顺延", source)
        or no_recommended_candidate(source)
        or ("未收录：" in source or "候选编码：" in source or "词库里还没有，要加入吗？" in source)
        and COMPACT_SINGLE_WORD_FOOTER_RE.search(source)
    )


def candidate_rows(text):
    return tuple(CANDIDATE_ROW_RE.finditer(normalized_copy(text)))


def numbered_candidate_codes(text):
    """Accept bare occupant labels only inside an approved compact card."""
    source = str(text or "")
    if COMPACT_CANDIDATE_HEADER_RE.search(source):
        return tuple(row["code"].lower() for row in candidate_rows(source))
    return tuple(match[1].lower() for match in NUMBERED_CANDIDATE_ROW_RE.finditer(source))


def marked_candidate_code(text):
    for row in candidate_rows(text):
        if CANDIDATE_RECOMMENDATION_LABEL_RE.search(row["label"]):
            return row["code"].lower()
    return ""


def batch_block_recommended_code(text):
    """Keep the legacy parenthesized candidate marker before the code line."""
    source = normalized_copy(text)
    for pattern in (RECOMMENDED_CANDIDATE_RE, BATCH_RECOMMENDED_CODE_RE):
        match = pattern.search(source)
        if match is not None:
            return match["code"].lower()
    return marked_candidate_code(source)


def recommended_candidate_code(text):
    source = normalized_copy(text)
    match = RECOMMENDED_CODE_RE.search(source)
    if match is not None:
        return match["code"].lower()
    match = RECOMMENDED_CANDIDATE_RE.search(source)
    if match is not None:
        return match["code"].lower()
    for pattern in (
        r"以编码\s*(?P<code>[a-z]+)\s*将「[^」\n]+」加入草稿",
        r"(?m)^\s*[•-]\s*「[^」\n]+」\s*→\s*(?P<code>[a-z]+)\s*\(推荐\)\s*$",
        r"「[^」\n]+」\s*占\s*(?P<code>[a-z]+)\s*、\s*「[^」\n]+」顺延",
    ):
        match = re.search(pattern, source)
        if match is not None:
            return match["code"].lower()
    return marked_candidate_code(source)


def no_recommended_candidate(text):
    return BLOCKED_RECOMMENDATION_RE.search(str(text or "")) is not None


def candidate_label_is_available(label):
    return label.strip().startswith("✅") or "空位" in label


def candidate_label_occupants(label):
    match = re.search(r"已有「(.+?)」", label)
    if match is not None:
        return tuple(part.strip() for part in match[1].split("、") if part.strip())
    if candidate_label_is_available(label):
        return ()
    label = re.split(r"✅|←", label, maxsplit=1)[0]
    label = re.sub(r"[（(].*?[）)]", "", label).strip()
    if label in OCCUPIED_NO_WORD_LABELS:
        return ()
    label = re.sub(r"已有\s*", "", label).replace("✔️", "")
    return tuple(part.strip() for part in re.split(r"[、,，]\s*", label) if part.strip())


def legacy_review_text(text):
    match = LEGACY_REVIEW_RE.search(str(text or ""))
    return match["review"].strip() if match is not None else ""


def draft_display_lines(text):
    lines = []
    for line in str(text or "").splitlines():
        if DRAFT_DISPLAY_ROW_RE.match(line):
            line = re.sub(r"^\s*[•-]\s*", "", line)
            line = re.sub(r"\s*[（(]权重[:：]\s*100[）)]\s*$", "", line)
            lines.append(re.sub(r"\s+", " ", line).strip())
    return tuple(lines)


def has_completed_draft_receipt(text):
    return any(marker in str(text or "") for marker in (
        "✅ 已加入草稿", "已提交审核", "✅ 已入库：", "✅ 已确认添加到草稿：",
    ))


def has_skipped_candidate_notice(text):
    return re.search(r"(?m)^未(?:选择|加)[:：]", str(text or "")) is not None


def has_candidate_card(text):
    source = str(text or "")
    return "候选" in source or bool(candidate_rows(source) and (
        missing_word_card_word(source) or NUMBERED_WORD_HEADER_RE.search(source)
    ))


def has_submitted_draft_receipt(text):
    return any(marker in str(text or "") for marker in (
        "已提交审核", "已加入词库", "已入库", "尚未提交审核",
    ))


def is_additional_code_hint(line):
    return str(line or "").startswith("可再追加一个编码")


def has_exact_submission_status(text, auto_approved):
    source = str(text or "")
    if auto_approved:
        return "✅ 批次已加入词库。" in source or "✅ 已入库：" in source
    return any(marker in source for marker in ("✅ 批次已提交审核。", "✅ 已提交审核，投票 ", "✅ 已提交审核："))


def candidate_cards_match(first, second):
    """Compare approved compact cards while keeping legacy exact-copy checks."""
    first_lines = tuple(line.strip() for line in str(first).splitlines() if line.strip())
    second_lines = tuple(line.strip() for line in str(second).splitlines() if line.strip())
    if first_lines == second_lines:
        return True
    compact_sources = [str(text) for text in (first, second) if COMPACT_CANDIDATE_HEADER_RE.search(str(text))]
    if not compact_sources:
        return False
    for source in compact_sources:
        for line in source.splitlines():
            if not line.strip():
                continue
            if (WORD_CARD_HEADER_RE.fullmatch(line) or NUMBERED_WORD_HEADER_RE.fullmatch(line)
                    or CANDIDATE_ROW_RE.fullmatch(line) or FREQUENCY_COMPARISON_RE.match(line)
                    or re.fullmatch(r"\d+ 个词的候选[:：]", line)
                    or re.fullmatch(r"回复「加入」或「加入并提交」(?:用推荐码)?[，,](?:或「取消」。|或「换 \d+」改选(?:、?「添加\d+(?:、\d+)+」多选)?(?:，或「取消」)?。)", line)
                    or re.fullmatch(r"或改选如「[^」\n]+」[（(]末尾加「，加入并提交」直接提交[）)]", line)):
                continue
            return False

    def fingerprint(text):
        source = normalized_copy(text)
        headings = list(NUMBERED_WORD_HEADER_RE.finditer(source))
        if not headings:
            headings = list(WORD_CARD_HEADER_RE.finditer(source))
        sections = []
        for index, heading in enumerate(headings):
            groups = heading.groupdict()
            word = groups.get("quoted_word") or groups.get("word")
            end = headings[index + 1].start() if index + 1 < len(headings) else len(source)
            block = source[heading.end():end]
            rows = candidate_rows(block)
            if not word or not rows:
                return ()
            review = (True if re.search(r"(?<!无)需(?:要)?管理员审核", heading[0] + block)
                      else False if "可自动通过" in heading[0] + block else None)
            reading_bindings = legacy_card_reading_bindings(block, tuple(row["code"] for row in rows))
            reading = (compact_card_reading_pinyin(heading[0], word)
                       or review_reading_pinyin(legacy_review_text(block)))
            readings = tuple((row["code"], reading_bindings.get(row["code"])
                              or review_reading_pinyin(row["label"]) or reading) for row in rows)
            sections.append((word, tuple((int(row["index"]), row["code"].lower(),
                                          candidate_label_is_available(row["label"]),
                                          candidate_label_occupants(row["label"])) for row in rows),
                             recommended_candidate_code(block), review, readings))
        return tuple(sections)

    expected = fingerprint(second)
    return bool(expected and fingerprint(first) == expected)


def displayed_candidate_statuses(text, pairs):
    source = normalized_copy(text)
    statuses = {}
    for word, code in pairs:
        match = re.search(
            re.escape(word) + r"[」”』]?\s*(?:→|->)\s*" + re.escape(code)
            + r"\s*(?:[（(](?P<annotation>[^）)\n]{0,128})[）)])?",
            source, re.IGNORECASE,
        )
        annotation = str((match["annotation"] or "") if match is not None else "")
        occupied = (False if re.search(r"空位|未占用", annotation) else True
                    if re.search(r"已有|已占用|占用中", annotation) else None)
        manual = (True if re.search(r"(?<!无)需(?:要)?管理员审核|待管理员审核", annotation) else False
                  if re.search(r"可自动通过|无需管理员审核|自动通过", annotation) else None)
        if match is None:
            headers = list(NUMBERED_WORD_HEADER_RE.finditer(source)) or list(WORD_CARD_HEADER_RE.finditer(source))
            for index, header in enumerate(headers):
                groups = header.groupdict()
                if (groups.get("quoted_word") or groups.get("word")) != word:
                    continue
                end = headers[index + 1].start() if index + 1 < len(headers) else len(source)
                block = source[header.start():end]
                row = next((row for row in candidate_rows(block) if row["code"] == code), None)
                if row is not None:
                    occupied = not candidate_label_is_available(row["label"])
                manual = (True if re.search(r"(?<!无)需(?:要)?管理员审核", block) else False
                          if "可自动通过" in block else None)
                break
        statuses[word] = (occupied, manual)
    return statuses


def community_vote_receipt_matches(text, batch_url):
    """Bind a vote advertisement to its own draft link, never a nearby batch."""
    lines = str(text or "").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != f"草稿地址：{batch_url}":
            continue
        header = lines[index - 1].strip() if index else ""
        if re.fullmatch(r"✅ 已提交审核，投票 \d+/3（引用本条回复「同意」）", header):
            return True
        if header != "批次已加入提交，审核中。":
            continue
        tail = []
        for following in lines[index + 1:]:
            if following.startswith(("草稿地址：", "批次已", "✅ ")):
                break
            tail.append(following)
        if "当前投票：0/3\n可以引用该条消息，回复同意来投票通过" in "\n".join(tail):
            return True
    return False


REVIEW_PINYIN_RE = re.compile(
    r"读音\s*([A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ\s]+)"
)
LEGACY_READING_LINE_RE = re.compile(
    r"(?m)^\s*(?P<index>\d+)\.\s*"
    r"(?P<pinyin>[A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ\s]+?)"
    r"\s*[；;]\s*来源(?:\s|$)"
)
LEGACY_READING_HEADER_RE = re.compile(
    r"(?m)^\s*候选编码[（(]读音\s*(?P<index>\d+)[）)]\s*[:：]\s*$"
)
CANDIDATE_ROW_NUMBER_RE = re.compile(
    r"(?m)^\s*\d+[.)、]\s*(?=[a-z]+\s*[—–-])"
)
LEGACY_OCCUPIED_GUIDANCE_RES = (
    re.compile(
        r"(?m)^若所选编号显示[“\"]已有…[”\"]，直接回复该编号表示添加重码；"
        r"回复[“\"]编号 重新编码[”\"]或[“\"]原词 重新编码[”\"]则挪开原词。\s*$"
    ),
    re.compile(r"(?m)^若选的是已有词编码，回复[“\"]编号 重新编码[”\"]可挪开原词。\s*$"),
    re.compile(
        r"(?m)^若要挪开(?:已有词「[^」\n]+」|该已有词)，"
        r"回复[“\"][1-9]\d{0,2} 重新编码[”\"]。\s*$"
    ),
)


def review_reading_pinyin(text):
    match = REVIEW_PINYIN_RE.search(str(text or ""))
    return re.sub(r"\s+", " ", match[1]).strip() if match is not None else ""


def compact_card_reading_pinyin(text, word):
    match = re.search(
        rf"(?m)^\s*(?:\d+[.)、]\s*)?「{re.escape(word)}」(?:未收录[:：]|词库里还没有，要加入吗？读音\s*)?\s*"
        r"(?P<pinyin>[A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ\s]+?)"
        r"\s*[，,](?:可自动通过|需(?:要)?管理员审核|预审未完成)",
        str(text or ""),
    )
    return re.sub(r"\s+", " ", match["pinyin"]).strip() if match is not None else ""


def protected_occupant_comparison_word(text, occupant):
    match = re.search(
        rf"「{re.escape(occupant)}」(?:不弱于|较)「([^」\n]+)」"
        rf"|「([^」\n]+)」与「{re.escape(occupant)}」(?:的)?常用度",
        str(text or ""),
    )
    if match is not None:
        return match[1] or match[2]
    for comparison in FREQUENCY_COMPARISON_RE.finditer(str(text or "")):
        if comparison["first"] == occupant:
            return comparison["second"]
        if comparison["second"] == occupant:
            return comparison["first"]
    return ""


def legacy_card_reading_bindings(text, codes):
    readings = {
        int(match["index"]): re.sub(r"\s+", " ", match["pinyin"]).strip()
        for match in LEGACY_READING_LINE_RE.finditer(text)
    }
    bindings = {}
    headings = tuple(LEGACY_READING_HEADER_RE.finditer(text))
    for index, heading in enumerate(headings):
        pinyin = readings.get(int(heading["index"]), "")
        if not pinyin:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        for row in CANDIDATE_CODE_LINE_RE.finditer(text[heading.end():end]):
            if row["code"] in codes:
                bindings[row["code"]] = pinyin
    return bindings


def word_card_section(text, word):
    header = re.search(rf"(?m)^(?:\d+[.)、]\s*)?「{re.escape(word)}」[^\n]*$", text)
    if header is None:
        return ""
    start = header.end()
    following = re.search(r"(?m)^(?:\d+[.)、]\s*)?「[^」]+」[^\n]*$", text[start:])
    end = start + following.start() if following is not None else len(text)
    return text[start:end]


def strip_legacy_occupied_guidance(text):
    for pattern in LEGACY_OCCUPIED_GUIDANCE_RES:
        text = pattern.sub("", text)
    return text


def has_reorder_choice_copy(text):
    return bool(
        re.search(r"(?m)^\s*推荐[：:](?:.+顺延(?:[；;].*)?|\s*\n[-•].+顺延.+)$", text)
        and re.search(r"(?m)^\s*(?:不重排选|换)\s+\d+", text)
    )


def first_occupied_candidate(text):
    for row in candidate_rows(text):
        occupants = candidate_label_occupants(row["label"])
        if occupants:
            return int(row["index"]), occupants[0]
    return None


def strip_candidate_row_numbers(text):
    return CANDIDATE_ROW_NUMBER_RE.sub("", text)


def is_explicit_code_template_hint(line):
    return "格式为：加入编码" in line or "指定完整码「加入编码" in line


def normalize_generated_review_copy(text):
    text = str(text or "")
    for old, new in (
        ("自动审核：预计需管理员审核", "自动审核：需管理员审核"),
        ("自动审核:预计需管理员审核", "自动审核：需管理员审核"),
        ("自动审核：预计需要管理员审核", "自动审核：需管理员审核"),
        ("自动审核:预计需要管理员审核", "自动审核：需管理员审核"),
        ("自动审核：该词需管理员审核", "自动审核：需管理员审核"),
        ("自动审核：预计可通过", "自动审核：可自动通过"),
        ("自动审核:预计可通过", "自动审核：可自动通过"),
        ("自动审核：该词可自动通过", "自动审核：可自动通过"),
    ):
        text = text.replace(old, new)
    text = re.sub(
        r"(?P<term>[\u3400-\u9fff]{1,12})[？?]\s*(?:不|不是)[，,]\s*"
        r"(?=(?P=term))",
        "",
        text,
    )
    return re.sub(r"[；;]\s*提交整批时会(?:重新审核|重审|复审)", "", text)


LEGACY_BATCH_PROMPT_MARKERS = (
    "一起加入草稿", "一起加这两个词", "两个词是否一起加", "两词是否一起加",
)
COMPACT_SELECTION_SUFFIX_RE = re.compile(
    r"或改选如[「“『](?P<example>[^」”』\n]+)[」”』][（(]末尾加\s*$"
)
UNRESOLVED_REVIEW_WORD_RE = re.compile(
    r"「(?P<word>[^「」\r\n]{1,20})」(?:存在[^\r\n]{0,40}多个读音|"
    r"的[^\r\n]{0,20}读音[^\r\n]{0,20}(?:冲突|未确定))"
)
OPERATION_COMPLETE_PREFIX = "✅ 操作已完成"
DRAFT_RECEIPT_CHANGE_PREFIX = "已变更："
DRAFT_RECEIPT_REPLACED_PREFIXES = ("已变更：", "未新增变更：", "发布状态：")
BASIC_AUTHORITATIVE_LINK_END_RE = re.compile(
    r"\s*(?:草稿地址|批次地址|草稿/批次地址|PR)[：:]?\s*$"
)
OPERATION_MEMORY_PREFIX_RE = re.compile(
    r"^词库操作：(?P<actor>.+?)(?:[（(][^)）]+[）)])?\s+"
    r"(?P<rest>(?:已提交审核|已加入草稿).*)$"
)
PROVISIONAL_DRAFT_LINK_COPY = "待确认后生成"


def legacy_batch_add_pairs(text):
    confirm_line = next((line.strip() for line in text.splitlines()
                         if "一起加入草稿" in line and "将「" in line), "")
    inline = re.findall(r'(?:以编码\s*)?([a-z]{2,12})\s*将「(.+?)」', confirm_line,
                        re.IGNORECASE) if confirm_line else []
    arrows = [(match[2], match[1]) for match in re.finditer(
        r'(?m)^\s*[-•]\s*「([^」]+)」\s*(?:→|->)\s*([a-z]{2,12})\s*$', text, re.IGNORECASE,
    )]
    inline_arrows = [(match[3], match[1] or match[2]) for match in re.finditer(
        r'(?:「([^」]+)」|([\u4e00-\u9fff]{1,12}))\s*(?:→|->)\s*([a-z]{2,12})', text, re.IGNORECASE,
    )]
    return (*inline, *arrows, *inline_arrows)


def ensure_cancel_control(text):
    return text if "「取消」" in text else text.rstrip() + "\n或「取消」。"


def append_pending_challenge(text, challenge, batch_challenge):
    if batch_challenge in text:
        compact = re.sub(r"\n+\s*" + re.escape(challenge) + r"\s*$", " " + challenge, text.rstrip())
        if challenge not in compact:
            compact += " " + challenge
        return compact
    return text if challenge in text else text.rstrip() + "\n\n" + challenge


def replace_additional_code_template(text, code):
    return "\n".join(
        line.replace("<code>（将 <code> 换成实际编码）", code).replace("<code>", code)
        if is_additional_code_hint(line) else line
        for line in text.splitlines()
    )


def candidate_reference_copy(text):
    return strip_candidate_row_numbers(text).replace("候选编码", "编码参考").replace("候选", "编码参考")


def candidate_diagnostic_header_copy(line, word):
    return line.replace(f"「{word}」候选编码：", f"「{word}」占用情况：")


def draft_receipt_contents(text):
    return [
        set(re.sub(r" 等 \d+ 条$", "", line.removeprefix("草稿内容：")).split("、"))
        for line in text.splitlines() if line.startswith("草稿内容：")
    ]
