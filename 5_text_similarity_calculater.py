import os
import re
from collections import namedtuple
from datetime import datetime
from typing import Optional, List

import jieba
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# —— 项目根目录（用于定位资源路径） ——
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# —— NLTK 资源目录（resources/nltk_data） ——
NLTK_DATA_DIR = os.path.join(BASE_DIR, "resources", "nltk_data")

# —— 注册到 NLTK 的搜索路径 ——
nltk.data.path.append(NLTK_DATA_DIR)

# 命名元组：用于存放相似评论与相似度
SimilarComment = namedtuple('SimilarComment', ['content', 'similarity'])

# 中文停用词（可按需扩展）
STOPWORDS_ZH = {
    "的", "了", "是", "在", "和", "就", "都", "我", "你", "他", "这", "那", "着", "与", "或", "及", "要",
    "也", "又", "很", "没有", "可以", "因为", "所以"
}

# 英文停用词（可按需扩展）
STOPWORDS_EN = {
    "the", "a", "an", "in", "on", "at", "for", "with", "by", "to", "of", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "can"
}


def preprocess_zh(text: str) -> str:
    """中文文本预处理：清洗符号 → 精确分词 → 停用词与单字过滤 → 拼接"""
    try:
        # 仅保留：汉字/字母/数字/空格
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

        # 精确模式分词
        tokens = jieba.cut(cleaned_text, cut_all=False)

        # 过滤停用词与单字
        filtered = [tok for tok in tokens if tok not in STOPWORDS_ZH and len(tok) > 1]
        return " ".join(filtered)
    except Exception as err:
        print(f"中文预处理失败: {err}")
        return ""


def preprocess_en(text: str) -> str:
    """英文文本预处理：小写化 → 分词 → 词形还原 → 停用词/非字母过滤 → 拼接"""
    try:
        # 小写 + 分词
        raw_tokens = nltk.word_tokenize(text.lower())

        # 词形还原（动词优先）
        wnl = WordNetLemmatizer()
        lemmas_v = [wnl.lemmatize(tok, pos='v') for tok in raw_tokens]

        # 去停用词 + 仅保留字母词
        filtered = [tok for tok in lemmas_v if tok not in STOPWORDS_EN and tok.isalpha()]
        return " ".join(filtered)
    except Exception as err:
        print(f"英文预处理失败: {err}")
        return ""


def detect_language(text: str) -> str:
    """简易语言检测：按中文字符占比判断（<5字符则走快速路径）"""
    # 短文本直接判断是否包含中文
    if len(text) < 5:
        return 'zh' if any('\u4e00' <= ch <= '\u9fff' for ch in text) else 'en'

    # 粗略统计中文字符比例
    cn_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    if cn_count / len(text) > 0.3:  # 中文占比 > 30% 视为中文
        return 'zh'
    return 'en'


def preprocess_mixed(text: str) -> str:
    """多语言统一预处理：先判别语言，再走对应预处理管线"""
    try:
        lang_code = detect_language(text)
    except Exception:
        lang_code = "en"

    if lang_code == "zh":
        return preprocess_zh(text)
    else:
        return preprocess_en(text)


def find_similar_comments(current_text, history_comments, limit=1) -> Optional[List[SimilarComment]]:  # Optional[X] 等价于 Union[X, None]
    """
    在历史评论中查找与当前文本最相近的若干条。

    参数:
        current_text (str): 待匹配的当前文本
        history_comments (list): 历史评论集合
        limit (int): 返回相似结果的数量上限

    返回:
        List[SimilarComment]: 含评论内容与相似度的命名元组列表
    """
    try:
        # —— 基础校验 ——
        now_tick = datetime.now()
        if not current_text or not isinstance(current_text, str):
            raise ValueError("current_text 必须为非空字符串")

        if not history_comments or not isinstance(history_comments, list):
            raise ValueError("history_comments 必须为非空列表")

        if limit <= 0:
            raise ValueError("limit 必须大于 0")

        # —— 文本预处理（中英文混合清洗/分词等） ——
        norm_query = preprocess_mixed(current_text)
        norm_corpus = [preprocess_mixed(cmt) for cmt in history_comments]

        # —— 向量空间构建（TF-IDF） ——
        corpus_all = [norm_query] + norm_corpus
        tfidf = TfidfVectorizer(token_pattern=r'\b\w+\b')
        X = tfidf.fit_transform(corpus_all)

        # —— 拆分查询向量与语料向量 ——
        q_vec = X[0]
        doc_mat = X[1:]

        # —— 计算余弦相似度 ——
        scores = cosine_similarity(q_vec, doc_mat)  # 形状 (1, N)

        # —— 相似度排序（降序） ——
        order = scores.argsort()[0][::-1]

        # —— 组装结果，按 limit 截断 ——
        top_matches = []
        for i in range(min(limit, len(history_comments))):
            j = order[i]
            top_matches.append(SimilarComment(
                content=history_comments[j],
                similarity=float(scores[0][j])
            ))

        return top_matches

    except Exception as e:
        # —— 兜底：异常返回空列表 ——
        return []



# 示例用法
if __name__ == "__main__":
    # 历史评论库
    # 新的历史评论：外卖评价
    history_comments = [
        "麻辣香锅太辣了，吃完胃疼",
        "包装干净，送餐速度快，五星好评",
        "菜品新鲜，分量足，味道正宗",
        "外卖送错餐，客服处理态度很好",
        "炸鸡外酥里嫩，配酱超好吃",
        "米饭有点硬，其他都很满意",
        "酸辣粉酸爽开胃，汤底浓郁",
        "配送员很有礼貌，提前十分钟到达",
        "价格小贵，但味道对得起价格",
        "深夜下单，居然还能准时送达，惊喜",
        "菜量太少，性价比一般",
        "番茄炒蛋太咸，下次备注少盐",
        "酸菜鱼无腥味，鱼片嫩滑",
        "奶茶甜度刚好，珍珠软糯",
        "烧烤火候到位，孜然味重",
        "凉皮酱汁调得地道，辣椒香",
        "泡菜汤偏淡，但料多",
        "麻辣拌蔬菜种类多，口感丰富",
        "咖喱饭味道浓郁，土豆软糯",
        "韩式炸鸡酱料偏甜，适合小孩",
        "外卖盒漏汤，袋子全湿了",
        "骑手提前到，电话通知及时",
        "冬天送来还是热的，保温效果好",
        "筷子忘记给，差评",
        "包装太豪华，环保减分",
        "雨天准时送达，给骑手点赞",
        "餐盒结实，汤没洒",
        "外卖袋破损，食物差点掉出来",
        "配送时间比预计快五分钟",
        "外卖员找不到楼号，耽误十分钟",
        "少送一份米饭，立马退款处理",
        "客服回复慢，但补偿了优惠券",
        "商家主动打电话确认辣度，贴心",
        "申请发票秒开，效率高",
        "退款到账快，体验不错",
        "整体满意，会再回购",
        "味道一般，不会点第二次",
        "性价比之王，学生党福音",
        "分量感人，两人吃到撑",
        "颜值在线，拍照发圈点赞多"
    ]

    # 新的待检测评论
    new_comment = "今晚点的麻辣香锅又辣又香，分量也够，吃完很满足"

    # 查找相似评论
    similar_comments = find_similar_comments(new_comment, history_comments)

    # 打印结果
    print(f"当前评论: {new_comment}")
    print("\n最相似的历史评论:")
    for i, comment in enumerate(similar_comments):
        print(f"{i + 1}. {comment.content} | 相似度: {comment.similarity:.4f}")
