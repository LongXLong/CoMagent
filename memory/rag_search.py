# -*- coding: utf-8 -*-
"""RAG 检索：基于树形 LTM 的逐层搜索（从根节点一层一层向下），每层由 LLM 挑选合适节点。"""

from typing import Any

from utils.llm import llm_call, parse_json_from_llm
from utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeNode:
    """知识树节点。"""

    def __init__(
        self,
        topic: str,
        content: str,
        question_type: str | None = None,
        children: list["KnowledgeNode"] | None = None,
    ):
        self.topic = topic
        self.content = content
        self.question_type = question_type
        self.children = children or []


def build_knowledge_tree(ltm: dict[str, Any]) -> KnowledgeNode:
    """
    将 LTM 的树形结构转为 KnowledgeNode 树。
    从 ltm["tree"] 开始构建，每个节点包含 topic、content、question_type、children。
    """
    tree_data = ltm.get("tree", {})
    if not tree_data:
        return KnowledgeNode(topic="root", content="")
    
    def build_node(key: str, node_data: dict[str, Any]) -> KnowledgeNode:
        children = []
        children_data = node_data.get("children", {})
        for child_key, child_data in children_data.items():
            children.append(build_node(child_key, child_data))
        return KnowledgeNode(
            topic=key,
            content=node_data.get("content", ""),
            question_type=node_data.get("question_type"),
            children=children,
        )
    
    # 创建根节点，将所有一级分类作为子节点
    root = KnowledgeNode(topic="root", content="")
    for key, node_data in tree_data.items():
        root.children.append(build_node(key, node_data))
    
    return root


class RAGSearch:
    """
    基于树形 LTM 的逐层检索：从根节点开始，一层一层向下搜索。
    每一层由 LLM 根据用户问题从当前层节点中挑选最合适的节点，再进入下一层。
    """

    def __init__(self, ltm_root: KnowledgeNode | None = None):
        self.root = ltm_root or KnowledgeNode(topic="root", content="")

    def _llm_select_nodes(self, query: str, nodes: list[KnowledgeNode], max_per_layer: int = 3) -> list[KnowledgeNode]:
        """
        让 LLM 从当前层节点中挑选与用户问题最相关的节点。
        返回选中的节点列表（最多 max_per_layer 个）。
        """
        if not nodes:
            logger.debug("[RAG] _llm_select_nodes: 候选节点为空，跳过选择")
            return []
        
        logger.info("[RAG] LLM节点选择开始 - 候选节点数=%d, 最大选择数=%d", len(nodes), max_per_layer)
        logger.debug("[RAG] 候选节点列表: %s", [n.topic for n in nodes])
        
        if len(nodes) <= max_per_layer:
            # 节点数不多时，让 LLM 决定是否全部相关或只选部分
            pass
        def node_line(i: int, n: KnowledgeNode) -> str:
            c = n.content or ""
            if len(c) > 200:
                c = c[:200] + "..."
            else:
                c = c or "(无正文)"
            return f"{i+1}. 【{n.topic}】 {c}"

        node_list_text = "\n".join(node_line(i, n) for i, n in enumerate(nodes))
        prompt = '''用户问题：{query}

当前层候选节点（编号、话题、内容摘要）：
{node_list}

请从上述节点中选出与用户问题最相关的节点，最多选 {max_per_layer} 个；若都不相关可填 0。
请输出的格式严格按照下面（只输出该 JSON，不要其他文字）：
{{
    "selected": [1, 3]
}}
selected 为节点编号列表（对应上述 1、2、3...），不相关则填 [0]。
'''.format(query=query, node_list=node_list_text, max_per_layer=max_per_layer)
        logger.debug("[RAG] LLM选择节点提示词长度=%d字符", len(prompt))
        try:
            raw = llm_call(prompt)
            logger.debug("[RAG] LLM原始返回: %s", raw[:200] if raw and len(raw) > 200 else raw)
            obj = parse_json_from_llm(raw)
            chosen: list[int] = []
            if obj is not None:
                sel = obj.get("selected")
                if isinstance(sel, list):
                    for x in sel:
                        if x == 0:
                            continue
                        idx = int(x) if isinstance(x, (int, float)) else (int(x) if str(x).isdigit() else None)
                        if idx is not None and 1 <= idx <= len(nodes):
                            chosen.append(idx - 1)
                elif isinstance(sel, str):
                    for s in sel.replace("，", ",").replace(" ", ",").split(","):
                        s = s.strip()
                        if s == "0" or not s:
                            continue
                        if s.isdigit():
                            idx = int(s)
                            if 1 <= idx <= len(nodes):
                                chosen.append(idx - 1)
            if not chosen and raw:
                # 兜底：按原始文本解析
                for s in (raw or "").strip().replace("，", ",").replace(" ", ",").split(","):
                    s = s.strip()
                    if s == "0" or not s:
                        continue
                    if s.isdigit():
                        idx = int(s)
                        if 1 <= idx <= len(nodes):
                            chosen.append(idx - 1)
            seen = set()
            selected = []
            for i in chosen:
                if i not in seen and len(selected) < max_per_layer:
                    seen.add(i)
                    selected.append(nodes[i])
            if selected:
                logger.info("[RAG] LLM选中节点: %s", [n.topic for n in selected])
                return selected
            logger.warning("[RAG] LLM未选中任何节点，使用默认前%d个", max_per_layer)
            return nodes[:max_per_layer]
        except Exception as e:
            logger.error("[RAG] LLM节点选择异常: %s，使用默认前%d个节点", str(e), max_per_layer)
            return nodes[:max_per_layer]

    def search(self, query: str, ltm: dict[str, Any]) -> str:
        """
        逐层搜索：从根节点开始，每一层由 LLM 挑选与问题最相关的节点，再进入下一层。
        仅收集第三层（叶子层）节点的 content 作为检索结果。
        """
        logger.info("RAG search 开始 query=%s", query)
        results: list[str] = []
        max_per_layer = 3
        leaf_depth = 3  # 不含根节点共三层，条目在第三层

        def layer_search(nodes: list[KnowledgeNode], current_path: list[str]) -> None:
            if not nodes:
                logger.debug("[RAG] 当前层无节点，路径=%s", current_path)
                return
            current_depth = len(current_path) + 1
            logger.info("[RAG] 进入第%d层搜索 - 当前路径=%s, 候选节点数=%d", current_depth, current_path, len(nodes))
            selected = self._llm_select_nodes(query, nodes, max_per_layer=max_per_layer)
            logger.info("[RAG] 第%d层选中%d个节点: %s", current_depth, len(selected), [n.topic for n in selected])
            for node in selected:
                path = current_path + [node.topic]
                # 只收集第三层（叶子层）的条目
                if len(path) == leaf_depth and node.content and node.content.strip():
                    logger.debug("[RAG] 收集叶子节点内容 - 路径=%s, 内容长度=%d字符", path, len(node.content))
                    results.append(node.content)
                if node.children:
                    layer_search(node.children, path)

        if self.root.children:
            layer_search(self.root.children, [])

        if not results:
            # 回退：遍历整棵树，仅收集第三层中包含 query 关键词的节点内容
            logger.warning("[RAG] LLM逐层搜索无结果，启动关键词回退搜索")
            def fallback_collect(node: KnowledgeNode, depth: int) -> None:
                if depth == leaf_depth and node.content and (
                    query in (node.content or "") or query in (node.topic or "")
                ):
                    logger.debug("[RAG] 回退搜索命中 - 节点=%s, 深度=%d", node.topic, depth)
                    results.append(node.content)
                for child in node.children:
                    fallback_collect(child, depth + 1)
            fallback_collect(self.root, 0)
            logger.info("[RAG] 回退搜索完成，命中%d条结果", len(results))

        if results:
            out = "\n\n".join(results)
            logger.info("RAG search 完成 共 %s 段 总长度 %s 字", len(results), len(out))
            return out
        logger.info("RAG search 无匹配，返回默认提示")
        return "（知识库暂无匹配）"
