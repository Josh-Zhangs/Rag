import logging
from typing import List, Dict, Any, Set, Optional
import json
import re

from services.llm_service import LLMService
from config.settings import settings


logger = logging.getLogger(__name__)

# 形态素解析：解决日语代词检测的词语边界问题
try:
    from janome.tokenizer import Tokenizer
    _tokenizer = Tokenizer()
    _HAS_JANOME = True
except ImportError:
    _tokenizer = None
    _HAS_JANOME = False
    logger.warning("janome 未安装，代词检测回退到子串匹配模式。建议: pip install janome")


# ==========================================================================
# 代词检测列表（语言学的封闭类）
# ==========================================================================
DEMONSTRATIVES: List[str] = [
    # ================================================================
    # こそあど体系 — 口语指示词（完整封闭词类，45 词）
    # ================================================================
    # コ系（近称）— 指说话人身边/自身相关
    "これ", "この", "こう", "こちら", "こっち", "こんな", "ここ",
    # ソ系（中称）— 指听话人身边/上文提及
    "それ", "その", "そう", "そちら", "そっち", "そんな", "そこ",
    # ア系（远称）— 指双方之外的远处事物
    "あれ", "あの", "ああ", "あちら", "あっち", "あんな", "あそこ",
    # ド系（不定称）— 疑问/不确定指代
    "どれ", "どの", "どう", "どちら", "どっち", "どんな", "どこ",

    # ================================================================
    # 书面指示词 — 正式文书中常见的上文引用（封闭词类）
    # 日本企业文书（契約書、稟議書、仕様書等）高频出现
    # ================================================================
    # 上下引用
    "上記", "下記", "上述", "後述", "先述", "既述",
    # 左右引用（竖排文书传统）
    "右記", "左記",
    # 前后引用
    "前述", "前記", "前出", "後記", "後出",
    # 事项引用
    "当該", "該当", "本件", "当件",
]

# ==========================================================================
# 追问词列表（语言学的封闭类）
# ==========================================================================
# 追问词是日语中封闭的追问表达体系，包含：
#   1. 程度追加：もっと、さらに、引き続き
#   2. 详细请求：詳細、詳しく、具体的に
#   3. 原因追问：なぜ、どうして、理由
#   4. 替代请求：他に、他には、別の
# 这些词天然指向上一轮回答，不含此类词的问题才是独立问题。
# ==========================================================================
FOLLOW_UP_INDICATORS: List[str] = [
    # 程度追加
    "もっと", "さらに", "引き続き",
    # 详细请求
    "詳細", "詳しく", "具体的に", "具体的な",
    # 原因追问
    "なぜ", "どうして", "理由",
    # 替代/追加请求
    "他に", "他には", "別の", "ほかに",
]


class QueryPreprocessor:

    def __init__(self):
        self._llm_service = LLMService(
            model=settings.ollama_preprocess_model
        )

    # ==================================================================
    # 主流程
    # ==================================================================

    def process(
        self, question: str, history: List[Dict[str, str]]
    ) -> Dict[str, Any]:

        # 无历史时跳过预处理
        if not history:
            logger.info("无对话历史，跳过预处理")
            return {"resolved_question": question, "topic_changed": False}

        # ----------------------------------------------------------
        # Stage 1：上下文依赖性分类
        # ----------------------------------------------------------
        classification = self._classify_question(question, history)

        if classification == "self-contained":
            logger.info(f"分类结果: 自包含问题 → 透传 '{question[:50]}'")
            return {"resolved_question": question, "topic_changed": False}

        # ----------------------------------------------------------
        # Stage 2：context-dependent → 选择处理策略
        # ----------------------------------------------------------
        logger.info(f"分类结果: 上下文依赖 → 选择策略 '{question[:50]}'")

        if self._has_pronouns(question):
            # 含代词 → LLM 改写
            return self._llm_rewrite(question, history)
        else:
            # 追问 → 上下文拼接
            contextual_query = self._build_contextual_query(question, history)
            logger.info(f"  策略: 上下文拼接")
            return {"resolved_question": contextual_query, "topic_changed": False}

    # ==================================================================
    # Stage 1：LLM 上下文依赖性分类
    # ==================================================================

    def _classify_question(
        self, question: str, history: List[Dict[str, str]]
    ) -> str:
        
        q = question.strip()

        # ---- Layer 1: 确定性规则 ----
        # 规则1：短问题通常依赖上下文
        # 阈值 15 字：捕获企业场景中主宾语省略的短句
        # 如「通知は書面ですか？」（11字）「条件は？」（4字）「理由は？」（4字）
        if len(q) <= 15:
            logger.debug(f"分类[规则]: 短问题 len={len(q)} → context-dependent")
            return "context-dependent"

        # 规则2：含指示词的问题必然依赖上下文
        # 指示词（こそあど体系 + 书面指示词）是语言学封闭词类，
        # 包含「上記」「それ」「この」等，含此类词的问题无法独立理解
        if self._has_pronouns(q):
            logger.debug(f"分类[规则]: 含指示词 → context-dependent")
            return "context-dependent"

        # 规则3：含追问词的问题必然依赖上下文
        # 追问词（もっと、詳細、なぜ等）天然指向上一轮回答
        if self._has_follow_up(q):
            logger.debug(f"分类[规则]: 含追问词 → context-dependent")
            return "context-dependent"

        # 规则4：长问题几乎一定自包含
        if len(q) > 60:
            logger.debug(f"分类[规则]: 长问题 len={len(q)} → self-contained")
            return "self-contained"

        # ---- Layer 2: LLM 语义判断（仅模糊边界） ----
        try:
            messages = self._build_classification_prompt(q, history)
            raw_output = self._llm_service.chat(
                messages,
                temperature=0.0,
                max_tokens=8,
                num_ctx=4096,
            )

            if not raw_output:
                logger.warning("LLM 分类返回空，回退到 context-dependent")
                return "context-dependent"

            result = raw_output.strip().lower()
            logger.debug(f"LLM 分类原始输出: '{result}'")

            if result == "context-dependent":
                return "context-dependent"
            elif result == "self-contained":
                return "self-contained"
            elif "context" in result:
                logger.debug(f"LLM 分类模糊匹配: '{result}' → context-dependent")
                return "context-dependent"
            else:
                logger.debug(f"LLM 分类模糊匹配: '{result}' → self-contained")
                return "self-contained"

        except Exception as e:
            logger.error(f"LLM 分类失败: {e}，回退到 context-dependent（安全侧）")
            return "context-dependent"

    def _build_classification_prompt(
        self, question: str, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        # 取最近 5 轮，倒序排列（与代词改写窗口一致，防止远距离引用漏判）
        recent = history[-5:]
        recent_reversed = list(reversed(recent))
        history_lines = []
        for i, turn in enumerate(recent_reversed, 1):
            answer_preview = turn.get("answer", "")[:150]
            history_lines.append(
                f"[ターン{i}] ユーザー: {turn.get('question', '')}\n"
                f"[ターン{i}] アシスタント: {answer_preview}"
            )
        history_text = "\n".join(history_lines)

        system_message = (
            "# タスク\n"
            "あなたはクエリ分類エンジンです。\n"
            "対話履歴と最新質問を読み、その質問が「履歴なしで単独理解できるか」を判定してください。\n"
            "この判定結果により、後続の検索パイプラインに履歴を注入するかが決まります。\n\n"

            "# 出力形式\n"
            "以下の 1 単語のみを出力し、説明・補足・改行は一切追加しないでください：\n"
            "self-contained\n"
            "または\n"
            "context-dependent\n\n"

            "# 判定基準（優先度順）\n"
            "## 優先度1: 話題の一貫性\n"
            "- 最新質問の主題が履歴の主題と無関係 → self-contained\n"
            "- 無関係の判断根拠: キーワードの重複ゼロ、かつ明らかに異なる分野\n"
            "- 例: 履歴が「契約書」→ 最新質問が「天気」→ self-contained\n\n"
            "## 優先度2: 省略の有無\n"
            "- 日本語の主語や目的語が省略され、履歴から補完しないと意味が曖昧 → context-dependent\n"
            "- 省略検出の手がかり:\n"
            "  - 文末が「〜ですか」「〜か」で、主題が直前の履歴に依存している\n"
            "  - 例: 履歴で「通知」の話をした後 →「それは書面ですか？」→ context-dependent\n"
            "  - 例: 履歴で「通知」の話をした後 →「通知は書面ですか？」→ 語としては完結しているが\n"
            "    「通知」が履歴のどの通知を指すか不明 → context-dependent\n\n"
            "## 優先度3: 自己完結性\n"
            "- 質問が固有名詞・具体的な主題を含み、履歴がなくても検索意図が明確 → self-contained\n"
            "- 例: 履歴が何であれ「RAGシステムのアーキテクチャを教えて」→ self-contained\n\n"

            "# 判断の tie-break ルール\n"
            "- 迷った場合は context-dependent を選ぶ（履歴ありで検索しても害は少ないが、\n"
            "  履歴なしで検索すると必要な文脈を失うリスクがあるため）\n"
            "- 話題が変わった場合は、たとえ省略表現があっても self-contained とする\n\n"

            "# Few-shot 例\n"
            "## 例1: 省略があるので context-dependent\n"
            "履歴:\n"
            "[ターン1] ユーザー: 契約書の解除条件は？\n"
            "[ターン1] アシスタント: 30日前の通知が必要です。\n"
            "最新質問: 通知は書面で必要ですか？\n"
            "出力: context-dependent\n\n"
            "## 例2: 具体的主題があるので self-contained\n"
            "履歴:\n"
            "[ターン1] ユーザー: 浅田稔さんは誰ですか\n"
            "[ターン1] アシスタント: 大阪大学大学院教授で、ロボカップの提唱者です。\n"
            "最新質問: RAGシステムとは何ですか\n"
            "出力: self-contained\n\n"
            "## 例3: 話題が変わったので self-contained\n"
            "履歴:\n"
            "[ターン2] ユーザー: 契約書の保管期限は？\n"
            "[ターン2] アシスタント: 10年間の保管が義務付けられています。\n"
            "[ターン1] ユーザー: ところで、社内の勤怠管理システムについて教えて\n"
            "[ターン1] アシスタント: 現在は○○システムを利用しています。\n"
            "最新質問: 有給休暇の申請方法は？\n"
            "出力: self-contained\n\n"
            "## 例4: 履歴の延長なので context-dependent\n"
            "履歴:\n"
            "[ターン1] ユーザー: プロジェクトAの進捗は？\n"
            "[ターン1] アシスタント: 現在80%完了、来週リリース予定です。\n"
            "最新質問: 残りの20%は何が残っていますか？\n"
            "出力: context-dependent\n"
        )

        user_message = (
            f"=== 対話履歴（最新が上） ===\n"
            f"{history_text}\n"
            f"=== 履歴ここまで ===\n\n"
            f"最新質問: {question}\n\n"
            f"出力:"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    # ==================================================================
    # Stage 2a：代词检测
    # ==================================================================

    # 代词词表 → 集合（O(1) 查找）
    _DEMONSTRATIVE_SET: Set[str] = set(DEMONSTRATIVES)

    def _has_pronouns(self, question: str) -> bool:

        q = question.strip()
        if _HAS_JANOME:
            for token in _tokenizer.tokenize(q):
                surface = token.surface
                if surface in self._DEMONSTRATIVE_SET:
                    logger.debug(
                        f"代词检测命中(形態素): '{surface}' "
                        f"(品詞={token.part_of_speech}) in '{q[:30]}'"
                    )
                    return True
            return False
        else:
            for demo in DEMONSTRATIVES:
                if demo in q:
                    logger.debug(f"代词检测命中(子串): '{demo}' in '{q[:30]}'")
                    return True
            return False

    # 追问词词表 → 集合（O(1) 查找）
    # 注意：本集合与 FOLLOW_UP_INDICATORS 语义一致，仅形态不同。
    # janome 会把「具体的に」拆成「具体的」+「に」，所以此处存 token 级表层形。
    # 维护规则：在 FOLLOW_UP_INDICATORS 中增删词条时，必须同步更新本集合。
    _FOLLOW_UP_TOKEN_SET: Set[str] = {
        "もっと", "さらに", "引き続き",
        "詳細", "詳しく", "具体的",
        "なぜ", "どうして", "理由",
        "他", "別", "ほか",
    }

    def _has_follow_up(self, question: str) -> bool:
        q = question.strip()

        if _HAS_JANOME:
            for token in _tokenizer.tokenize(q):
                surface = token.surface
                if surface in self._FOLLOW_UP_TOKEN_SET:
                    logger.debug(
                        f"追问词检测命中(形態素): '{surface}' "
                        f"(品詞={token.part_of_speech}) in '{q[:30]}'"
                    )
                    return True
            return False
        else:
            for indicator in FOLLOW_UP_INDICATORS:
                if indicator in q:
                    logger.debug(f"追问词检测命中(子串): '{indicator}' in '{q[:30]}'")
                    return True
            return False

    # ==================================================================
    # Stage 2b：上下文拼接
    # ==================================================================

    # 错误回答的特征模式
    _ERROR_PATTERNS: List[str] = [
        "回答を生成できませんでした",
        "エラーが発生しました",
        "内部エラー",
        "タイムアウト",
    ]
    # 从对话历史中找到最近 N 轮「有效」的对话（跳过空回答和错误回答）。
    def _find_last_valid_turns(
        self, history: List[Dict[str, str]], n: int = 2
    ) -> List[Dict[str, str]]:

        if not history:
            return []

        valid_turns: List[Dict[str, str]] = []
        for turn in reversed(history):
            answer = turn.get("answer", "").strip()
            if not answer:
                logger.debug(f"上下文拼接: 跳过空回答 turn")
                continue
            is_error = any(pattern in answer for pattern in self._ERROR_PATTERNS)
            if is_error:
                logger.debug(f"上下文拼接: 跳过错误回答 '{answer[:60]}...'")
                continue
            valid_turns.append({
                "question": turn.get("question", ""),
                "answer": answer,
            })
            if len(valid_turns) >= n:
                break

        if not valid_turns:
            last = history[-1]
            logger.debug(f"上下文拼接: 所有轮次被跳过，fallback 到最后一轮")
            valid_turns.append({
                "question": last.get("question", ""),
                "answer": last.get("answer", ""),
            })

        return valid_turns

    # 将当前问题与最近 2 轮有效对话拼接，生成带上下文的检索 query。
    def _build_contextual_query(
        self, question: str, history: List[Dict[str, str]]
    ) -> str:

        valid_turns = self._find_last_valid_turns(history, n=2)

        context_parts: List[str] = []
        for i, turn in enumerate(valid_turns):
            q = turn["question"]
            a = turn["answer"]
            # 每轮回答截断 150 字，2 轮共 300 字，与原来 1 轮 300 字总量持平
            context_parts.append(f"前回の質問: {q} 回答: {a[:150]}")

        context = " | ".join(context_parts)

        # 最终检索 query: 上下文 + 分隔符 + 当前问题
        contextual = f"{context} | 現在の質問: {question}"

        logger.debug(
            f"上下文拼接: {len(valid_turns)} 轮有效对话, "
            f"final_len={len(contextual)}"
        )
        return contextual

    # ==================================================================
    # Stage 2c：LLM 代词改写
    # ==================================================================

    def _llm_rewrite(
        self, question: str, history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
       
        try:
            messages = self._build_pronoun_rewrite_prompt(question, history)
            raw_output = self._llm_service.chat(
                messages,
                temperature=0.0,
                max_tokens=512,
                num_ctx=4096,
                stop=["\n\n\n", "USER:", "新しい質問:"],
            )

            if not raw_output:
                logger.warning("代词改写 LLM 返回空，回退到原始问题")
                return {"resolved_question": question, "topic_changed": False}

            logger.info(f"代词改写 LLM 原始输出: {raw_output!r}")

            result = self._parse_output(raw_output, question)
            result = self._validate_result(result, question)

            logger.info(
                f"代词改写完成: {result['resolved_question'][:80]}..., "
                f"topic_changed={result['topic_changed']}"
            )
            return result

        except Exception as e:
            logger.error(f"代词改写失败: {e}，回退到原始问题")
            return {"resolved_question": question, "topic_changed": False}

    def _build_pronoun_rewrite_prompt(
        self, question: str, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        # 取最近 5 轮，倒序排列（最新的在前，LLM 注意力更集中在近期对话）
        recent = history[-5:]
        recent_reversed = list(reversed(recent))
        history_lines = []
        for i, turn in enumerate(recent_reversed, 1):
            answer_preview = turn.get("answer", "")[:200]
            history_lines.append(
                f"[ターン{i}] ユーザー: {turn.get('question', '')}\n"
                f"[ターン{i}] アシスタント: {answer_preview}"
            )
        history_text = "\n".join(history_lines)

        system_message = (
            "# タスク\n"
            "あなたは検索クエリ最適化エンジンです。\n"
            "ユーザーの最新質問を、対話履歴を参照して「単独で完全に理解できる質問」に書き換えてください。\n"
            "この質問は後続のベクトル検索に直接渡されるため、具体性と正確性が検索結果の品質を決定します。\n\n"

            "# 出力形式\n"
            "以下の JSON のみを出力し、説明や補足は一切追加しないでください：\n"
            '{"resolved_question": "<書き換えた質問>", "topic_changed": <true/false>}\n\n'

            "# 書き換えルール\n"
            "## 1. 指示詞の解決\n"
            "- 「これ/それ/あれ/この/その/あの」→ 履歴の回答文から最も近い名詞句を抽出して置換\n"
            "- 「こんな/そんな/あんな」→ 履歴から該当する属性・状態を補完\n"
            "- 「ここ/そこ/あそこ/どこ」→ 履歴から場所・位置情報を補完\n"
            "- 指示詞が指す対象が履歴から特定できない場合は、無理に推測せず指示詞を残す\n\n"
            "## 2. 文書参照語の解決\n"
            "- 「上記/前述/前記/当該/本件/右記/左記」→ 履歴の回答文の主題を抽出して置換\n"
            "- 「上述/後述/先述/既述/前出」→ 同様に履歴から該当内容を補完\n"
            "- 文書参照語が複数の候補を指す可能性がある場合、最も直近の回答から補完する\n\n"
            "## 3. 省略の補完\n"
            "- 日本語の主語省略（「〜は」がない文）→ 履歴の話題から主語を補完\n"
            "- 目的語の省略（「確認したい」「教えて」のみ）→ 履歴から目的語を補完\n"
            "- 補完後も自然な日本語文になること（機械的な文字列連結は禁止）\n\n"
            "## 4. 禁止事項\n"
            "- 履歴に存在しない情報を捏造しない\n"
            "- 元の質問に含まれる固有名詞・数値・日付を改変しない\n"
            "- 質問の意図（依頼/質問/確認）を変えない\n"
            "- 自己完結済みの質問を不必要に改変しない\n\n"

            "# topic_changed 判定基準\n"
            "- true: 最新質問のキーワードが履歴の話題と無関係である場合\n"
            "  - 例: 履歴が「契約書」の話なのに、最新質問が「天気」や「昼食」について\n"
            "  - 例: ユーザーが「ところで」「話は変わりますが」で話題を明示的に切り替えた\n"
            "- false: 最新質問が履歴の延長線上にある場合（代名詞/省略/詳細質問を含む）\n"
            "- topic_changed=true の場合、resolved_question は元の質問をそのまま出力する\n\n"

            "# Few-shot 例\n"
            "## 例1: 指示詞の解決\n"
            "履歴:\n"
            "[ターン1] ユーザー: RAGシステムの導入費用はいくらですか？\n"
            "[ターン1] アシスタント: 初期費用50万円、月額保守5万円です。\n"
            "新しい質問: それは高いですか？\n"
            '出力: {"resolved_question": "RAGシステムの導入費用（初期費用50万円、月額保守5万円）は高いですか？", "topic_changed": false}\n\n'
            "## 例2: 文書参照語の解決\n"
            "履歴:\n"
            "[ターン1] ユーザー: 浅田稔さんは誰ですか\n"
            "[ターン1] アシスタント: 浅田稔さんは大阪大学大学院教授で、ロボカップの提唱者です。\n"
            "新しい質問: 上記情報はどのファイルに記述されていますか\n"
            '出力: {"resolved_question": "浅田稔さん（大阪大学大学院教授、ロボカップ提唱者）の情報はどのファイルに記述されていますか", "topic_changed": false}\n\n'
            "## 例3: 省略の補完\n"
            "履歴:\n"
            "[ターン1] ユーザー: 社内AI検索システムのセキュリティ対策は？\n"
            "[ターン1] アシスタント: SSL暗号化、アクセス制御、監査ログの3つを実装しています。\n"
            "新しい質問: その詳細を教えて\n"
            '出力: {"resolved_question": "社内AI検索システムのセキュリティ対策（SSL暗号化、アクセス制御、監査ログ）の詳細を教えてください。", "topic_changed": false}\n\n'
            "## 例4: 話題が変わった場合\n"
            "履歴:\n"
            "[ターン2] ユーザー: 契約書の解除条件は？\n"
            "[ターン2] アシスタント: 30日前の通知が必要で、違約金は発生しません。\n"
            "[ターン1] ユーザー: ところで、今日の天気は？\n"
            "[ターン1] アシスタント: 申し訳ありません、天気情報にはアクセスできません。\n"
            "新しい質問: 明日の天気は？\n"
            '出力: {"resolved_question": "明日の天気は？", "topic_changed": true}\n\n'
            "## 例5: 自己完結済み（改変禁止）\n"
            "履歴:\n"
            "[ターン1] ユーザー: Pythonのバージョンは？\n"
            "[ターン1] アシスタント: 3.11 です。\n"
            "新しい質問: FastAPIの認証ミドルウェアの実装方法を教えてください\n"
            '出力: {"resolved_question": "FastAPIの認証ミドルウェアの実装方法を教えてください", "topic_changed": false}\n'
        )

        user_message = (
            f"=== 対話履歴（最新が上） ===\n"
            f"{history_text}\n"
            f"=== 履歴ここまで ===\n\n"
            f"最新質問: {question}\n\n"
            f"出力:"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    # ==================================================================
    # Stage 3：后处理校验
    # ==================================================================

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        从文本中提取第一个完整的 JSON 对象（支持花括号嵌套）。

        通过括号计数配平，比正则更健壮。当 resolved_question 值中包含
        「{」「}」字符时（如代码片段），正则的 [^{}]* 会提前截断。
        """
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _parse_output(
        self, raw_output: str, fallback_question: str
    ) -> Dict[str, Any]:
        """
        解析 LLM 输出的 JSON，提取 resolved_question 和 topic_changed。

        三层防御：
        - 直接 JSON 解析（理想路径）
        - 正则提取 JSON 片段（模型多输出了废话）
        - 纯文本兜底（仅在输出像合理问题时使用）
        """
        resolved_question = fallback_question
        topic_changed = False

        # 第一层：直接 JSON 解析
        try:
            data = json.loads(raw_output.strip())
            if isinstance(data, dict):
                rq = data.get("resolved_question", "")
                if rq and isinstance(rq, str):
                    resolved_question = rq.strip()
                tc = data.get("topic_changed", False)
                if isinstance(tc, bool):
                    topic_changed = tc
                elif isinstance(tc, str):
                    topic_changed = tc.lower().strip() in ("true", "yes", "変更あり")
                logger.info("JSON 解析成功（直接解析）")
                return {
                    "resolved_question": resolved_question,
                    "topic_changed": topic_changed,
                }
        except (json.JSONDecodeError, TypeError):
            pass

        # 第二层：括号配平提取 JSON 片段（支持 resolved_question 值中的花括号）
        json_str = self._extract_json(raw_output)
        if json_str:
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    rq = data.get("resolved_question", "")
                    if rq and isinstance(rq, str):
                        resolved_question = rq.strip()
                    tc = data.get("topic_changed", False)
                    if isinstance(tc, bool):
                        topic_changed = tc
                    elif isinstance(tc, str):
                        topic_changed = tc.lower().strip() in ("true", "yes", "変更あり")
                    logger.info("JSON 解析成功（括号配平提取）")
                    return {
                        "resolved_question": resolved_question,
                        "topic_changed": topic_changed,
                    }
            except (json.JSONDecodeError, TypeError):
                pass

        # 第三层：纯文本兜底（仅当输出看起来像合理的日语问题）
        cleaned = raw_output.strip()
        # 去掉 markdown 代码块包裹
        if cleaned.startswith("```"):
            code_lines = cleaned.split("\n")
            if len(code_lines) >= 3:
                cleaned = "\n".join(code_lines[1:-1]).strip()
            elif len(code_lines) == 2:
                cleaned = code_lines[1].rstrip("`").strip()
        # 去掉常见前缀
        for prefix in ["書き直した質問:", "解決質問:", "質問:", "resolved_question:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        # 去掉引号/括号
        cleaned = cleaned.strip("\"'「」『』")

        # 安全检查：输出必须像合理的日语问题，否则回退
        if self._is_plausible_question(cleaned):
            logger.info(f"纯文本兜底解析成功: {cleaned[:80]}...")
            return {
                "resolved_question": cleaned,
                "topic_changed": False,
            }

        logger.warning(
            f"所有解析层失败，回退到原始问题，"
            f"raw_output={raw_output[:80]!r}..."
        )
        return {
            "resolved_question": fallback_question,
            "topic_changed": False,
        }

    @staticmethod
    def _is_plausible_question(text: str) -> bool:
        """
        检查文本是否像一个合理的日语问题（防止 LLM 输出垃圾进入检索管线）。
        """
        if not text or len(text) < 3:
            return False
        if len(text) > 500:
            return False
        # 必须包含日语字符
        has_japanese = any(
            "\u3041" <= c <= "\u309f"   # ひらがな
            or "\u30a1" <= c <= "\u30ff"  # カタカナ
            or "\u4e00" <= c <= "\u9fff"  # 漢字
            for c in text
        )
        if not has_japanese:
            return False
        # 排除明显的错误/道歉文本
        garbage_patterns = [
            "申し訳ありません",
            "わかりません",
            "エラー",
            "error",
            "I don't know",
            "I'm sorry",
        ]
        for pattern in garbage_patterns:
            if pattern.lower() in text.lower():
                return False
        return True

    def _validate_result(
        self, result: Dict[str, Any], original_question: str
    ) -> Dict[str, Any]:
        
        resolved = result.get("resolved_question", "")

        if not resolved or not resolved.strip():
            logger.warning("校验失败：resolved_question 为空，回退到原始问题")
            return {"resolved_question": original_question, "topic_changed": False}

        max_length = max(len(original_question) * 5, 200)
        if len(resolved) > max_length:
            logger.warning(
                f"校验失败：resolved_question 过长 "
                f"({len(resolved)} > {max_length})，回退到原始问题"
            )
            return {"resolved_question": original_question, "topic_changed": False}

        if resolved.strip() == original_question.strip() and result.get("topic_changed", False):
            logger.warning("校验：重写结果与原问题相同但判定话题切换，修正 topic_changed=False")
            result["topic_changed"] = False

        return result


# 全局单例实例
query_preprocessor = QueryPreprocessor()