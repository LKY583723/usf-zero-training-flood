"""GE 图运行时 —— 正式满足论文 G1-G4 的最小图工程框架。

对照 Macedo 2026（arXiv:2607.27578）的 prompt graph engineering 定义：
  G1 显式结构：节点 = 确定性变换（或 prompt 参数化调用），边 = 显式数据/控制依赖。
  G2 结构与内容分离：图结构（节点+边）与节点内容（参数，如阈值）相互独立。
  G3 可执行语义：调度器按拓扑序执行节点，共享状态在节点间流动；分支/并行/循环均可在
     图上表达（本实现支持分支输出，循环由上层 while 驱动）。
  G4 一等图制品：Graph 是可检查（describe）、可序列化（to_dict）、可执行（execute）的
     独立对象，与任何具体执行解耦。

用法：
    g = Graph("flood", "zero-training flood graph")
    g.add_node("int", fn_int, deps=("sar",), content={"pol": "or"})
    g.add_node("conn", fn_conn, deps=("dark",), content={"min_size": 500})
    state = {"_sar": sar_array}
    out = g.execute(state, output="conn")   # 只执行 output 的闭包
    print(g.describe())
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List


class GraphNode:
    """图中的一个节点：名字 + 确定性变换 + 上游依赖 + 可分离的内容参数。"""

    def __init__(self, name: str, fn: Callable, deps: Iterable[str] = (),
                 content: Dict[str, Any] | None = None):
        self.name = name
        self.fn = fn              # fn(state, content) -> 写入 state[name] 或返回输出
        self.deps = tuple(deps)   # 上游节点名（边）
        self.content = content or {}

    def __repr__(self):
        return f"Node({self.name}, deps={list(self.deps)}, content={self.content})"


class Graph:
    """一等图制品：节点集合 + 边（deps）+ 执行能力。"""

    def __init__(self, name: str = "graph", description: str = ""):
        self.name = name
        self.description = description
        self.nodes: Dict[str, GraphNode] = {}

    # ---- 结构构造（G1/G4）----
    def add_node(self, name: str, fn: Callable, deps: Iterable[str] = (),
                 content: Dict[str, Any] | None = None) -> "Graph":
        if name in self.nodes:
            raise ValueError(f"duplicate node: {name}")
        self.nodes[name] = GraphNode(name, fn, deps, content)
        return self

    def add_edge(self, src: str, dst: str) -> "Graph":
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"edge {src}->{dst} references unknown node")
        if src not in self.nodes[dst].deps:
            self.nodes[dst].deps += (src,)
        return self

    # ---- 可执行语义（G3）----
    def _needed(self, output: str) -> set:
        """output 的传递闭包：只执行到达 output 所必需的节点。"""
        needed: set = set()
        stack = [output]
        while stack:
            n = stack.pop()
            if n in needed:
                continue
            needed.add(n)
            stack.extend(self.nodes[n].deps)
        return needed

    def _topo_order(self, outputs: Iterable[str]) -> List[str]:
        """Kahn 拓扑排序，仅限给定输出的闭包子图。"""
        needed = set()
        for o in outputs:
            needed |= self._needed(o)
        indeg = {n: len([d for d in self.nodes[n].deps if d in needed])
                 for n in needed}
        ready = [n for n in needed if indeg[n] == 0]
        order: List[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in needed:
                if n in self.nodes[m].deps:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        ready.append(m)
        if len(order) != len(needed):
            raise RuntimeError("graph has a cycle")
        return order

    def execute(self, state: Dict[str, Any], output: str | Iterable[str] | None = None):
        """调度执行。默认执行全部节点并返回 state；指定 output 时只执行其闭包。

        循环语义：本框架不内置 while；上层可在 execute 之间改写 state 并再次调用，
        等价于在图上重放（GE 的 loop 由外层驱动，LangGraph 亦如此）。
        """
        outputs = [output] if isinstance(output, str) else (output or list(self.nodes))
        order = self._topo_order(outputs)
        for name in order:
            node = self.nodes[name]
            state[name] = node.fn(state, node.content)
        if isinstance(output, str):
            return state[output]
        return state

    # ---- 一等制品（G4）：检查 / 序列化 ----
    def describe(self) -> str:
        lines = [f"Graph: {self.name} — {self.description}",
                 f"nodes ({len(self.nodes)}):"]
        for n, nd in self.nodes.items():
            lines.append(f"  - {n}: deps={list(nd.deps)} content={nd.content}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "nodes": {n: {"deps": list(nd.deps), "content": nd.content}
                      for n, nd in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Graph":
        g = cls(d["name"], d.get("description", ""))
        for n, nd in d["nodes"].items():
            g.add_node(n, lambda s, c: None, deps=nd["deps"], content=nd["content"])
        return g
